"""Batch-LBFGS adapter backed by ``CalcABC.calculate_many``.

This adapter lets the existing batched optimizer consume MAPLE's unified
calculator protocol without depending on a model-specific optimizer wrapper.
It owns only coordinate packing/unpacking; all scientific model evaluation
continues to live behind the calculator's ``calculate_many`` implementation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from ase import Atoms

from ....calculator._batch_eval import (
    ALL_BATCH_SIZE,
    AUTO_BATCH_SIZE,
    _AutoBatchSizer,
    _calculator_batch_size,
    _is_cuda_oom,
)
from ....calculator._batch_types import BatchResult


class CalculateManyBatchCalc:
    """Minimal ``BatchLBFGS`` calculator interface using ``calculate_many``."""

    def __init__(
        self,
        calc,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not hasattr(calc, "calculate_many"):
            raise TypeError(
                f"{type(calc).__name__} does not implement calculate_many()."
            )
        if not bool(getattr(calc, "supports_batch_energy_forces", False)):
            raise NotImplementedError(
                "CalculateManyBatchCalc requires a validated native "
                f"calculate_many() path; got {type(calc).__name__}."
            )
        self.calc = calc
        self.device = torch.device(
            device if device is not None else getattr(calc, "device", "cpu")
        )
        self.dtype = dtype

        self._atoms_list: List[Atoms] = []
        self._atoms_B = 0
        self._ptr = None
        self.N_atoms = 0
        self.Nmax_atoms = 0
        self.nmax_dof = 0
        self.coord = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self._coord_backup = None

    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None) -> None:
        """Pack current atom coordinates into the optimizer coordinate buffer."""
        self._atoms_list = list(atoms_list)
        if any(len(at) == 0 for at in self._atoms_list):
            raise ValueError(
                "CalculateManyBatchCalc does not support empty structures."
            )
        if any(bool(getattr(at, "constraints", None)) for at in self._atoms_list):
            raise NotImplementedError(
                "CalculateManyBatchCalc does not support ASE constraints; "
                "BatchLBFGS cannot project constrained Cartesian steps/forces."
            )
        self._atoms_B = len(self._atoms_list)

        counts = [len(at) for at in self._atoms_list]
        ptr = [0]
        for count in counts:
            ptr.append(ptr[-1] + int(count))
        self._ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.N_atoms = ptr[-1]
        self.Nmax_atoms = max(counts, default=0)

        required_dof = 3 * self.Nmax_atoms
        if fixed_nmax is None:
            self.nmax_dof = required_dof
        else:
            self.nmax_dof = int(fixed_nmax)
            if self.nmax_dof < required_dof:
                raise ValueError(
                    f"fixed_nmax={self.nmax_dof} is too small for the current "
                    f"batch; need at least {required_dof} Cartesian DOFs."
                )
            if self.nmax_dof % 3 != 0:
                raise ValueError(
                    f"fixed_nmax={self.nmax_dof} is not a multiple of 3 "
                    "Cartesian DOFs."
                )

        if self.N_atoms:
            coords = np.concatenate(
                [at.get_positions() for at in self._atoms_list],
                axis=0,
            )
            if not np.all(np.isfinite(coords)):
                raise ValueError(
                    "CalculateManyBatchCalc input coordinates must be finite."
                )
            self.coord = torch.tensor(
                coords,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.coord = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self._coord_backup = None

    @torch.no_grad()
    def backup_coords(self) -> None:
        self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self) -> None:
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)
            self._coord_backup = None

    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor) -> None:
        """Apply a padded Cartesian step to the active coordinate buffer."""
        expected = (self._atoms_B, self.nmax_dof)
        if tuple(s_cart.shape) != expected:
            raise ValueError(
                f"step_cart_ expects {expected}, got {tuple(s_cart.shape)}"
            )
        s_cart = s_cart.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(s_cart).all().item()):
            raise FloatingPointError(
                "CalculateManyBatchCalc received a non-finite Cartesian step."
            )
        starts = self._ptr[:-1]
        stops = self._ptr[1:]
        for i in range(self._atoms_B):
            n_atoms = int((stops[i] - starts[i]).item())
            if n_atoms:
                self.coord[starts[i]:stops[i], :].add_(
                    s_cart[i, : 3 * n_atoms].reshape(n_atoms, 3)
                )

    def get_ef_gpu(self):
        """Return energies and padded forces in Hartree / Hartree Å⁻¹."""
        if self._atoms_B == 0:
            return (
                torch.zeros((0,), dtype=self.dtype, device=self.device),
                torch.zeros((0, 0), dtype=self.dtype, device=self.device),
            )

        atoms_batch = self._atoms_with_current_positions()
        result = self._calculate_many_chunked(atoms_batch)
        if not isinstance(result, BatchResult):
            raise TypeError(
                f"{type(self.calc).__name__}.calculate_many() must return "
                f"BatchResult, got {type(result).__name__}"
            )
        result.validate_against(atoms_batch, ("energy", "forces"))
        if result.energies is None or len(result.energies) != self._atoms_B:
            got = None if result.energies is None else len(result.energies)
            raise RuntimeError(
                "calculate_many returned the wrong number of energies for "
                f"BatchLBFGS: expected {self._atoms_B}, got {got}"
            )
        if result.forces is None or len(result.forces) != self._atoms_B:
            got = None if result.forces is None else len(result.forces)
            raise RuntimeError(
                "calculate_many returned the wrong number of force arrays for "
                f"BatchLBFGS: expected {self._atoms_B}, got {got}"
            )

        energies = torch.tensor(
            result.energies,
            dtype=self.dtype,
            device=self.device,
        )
        forces = torch.zeros(
            (self._atoms_B, self.nmax_dof),
            dtype=self.dtype,
            device=self.device,
        )
        for i, (at, force) in enumerate(zip(atoms_batch, result.forces)):
            arr = np.asarray(force, dtype=np.float64)
            expected = (len(at), 3)
            if arr.shape != expected:
                raise RuntimeError(
                    "calculate_many returned an invalid force shape for "
                    f"BatchLBFGS: forces[{i}].shape={arr.shape}, "
                    f"expected {expected}"
                )
            if len(at):
                forces[i, : 3 * len(at)] = torch.tensor(
                    arr.reshape(-1),
                    dtype=self.dtype,
                    device=self.device,
                )
        if not bool(torch.isfinite(energies).all().item()):
            raise FloatingPointError(
                "CalculateManyBatchCalc received non-finite energies."
            )
        if not bool(torch.isfinite(forces).all().item()):
            raise FloatingPointError(
                "CalculateManyBatchCalc received non-finite forces."
            )
        return energies, forces

    def _calculate_many_chunked(self, atoms_batch: List[Atoms]) -> BatchResult:
        """Evaluate the active optimizer batch with ordered CUDA-OOM backoff."""
        n_total = len(atoms_batch)
        setting = _calculator_batch_size(
            self.calc,
            "optimization_batch_size",
        )
        sizer = _AutoBatchSizer(
            self.calc,
            atoms_batch,
            ("energy", "forces"),
            kind="optimization",
        )
        if setting == AUTO_BATCH_SIZE:
            chunk = sizer.chunk
        elif setting == ALL_BATCH_SIZE:
            chunk = n_total
        else:
            chunk = int(setting)
        chunk = max(1, min(chunk, n_total))
        sizer.chunk = chunk
        setattr(self.calc, "_auto_batch_size_last", chunk)

        energies = []
        forces = []
        start = 0
        while start < n_total:
            sub = atoms_batch[start : start + sizer.chunk]
            try:
                result = self.calc.calculate_many(
                    sub,
                    properties=("energy", "forces"),
                )
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or not sizer.backoff_after_oom():
                    raise
                continue
            if not isinstance(result, BatchResult):
                raise TypeError(
                    f"{type(self.calc).__name__}.calculate_many() must return "
                    f"BatchResult, got {type(result).__name__}"
                )
            result.validate_against(sub, ("energy", "forces"))
            energies.extend(float(value) for value in result.energies)
            forces.extend(np.asarray(value, dtype=np.float64) for value in result.forces)
            start += len(sub)

        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64),
            forces=forces,
        ).validate_against(atoms_batch, ("energy", "forces"))

    def _atoms_with_current_positions(self) -> List[Atoms]:
        coords = self.coord.detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        out: List[Atoms] = []
        for i, atoms in enumerate(self._atoms_list):
            current = atoms.copy()
            current.calc = self.calc
            current.set_positions(coords[ptr[i] : ptr[i + 1]], apply_constraint=False)
            out.append(current)
        return out
