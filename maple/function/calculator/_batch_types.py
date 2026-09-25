"""Return types for the unified batched calculator interface.

`BatchResult` is the single contract that `CalcABC.calculate_many` returns and
that every evaluator in `_batch_eval` consumes. Keeping it return-value driven
(rather than mutating `calc.results`) prevents the batched paths from
clobbering ASE's single-structure result cache that other modules
(frequency, irc, scan) still depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BatchResult:
    """Result of `CalcABC.calculate_many(atoms_list, properties)`.

    Attributes
    ----------
    energies
        (B,) float64 array of energies (Hartree if the calculator's
        single-structure path returns Hartree). `None` when 'energy' was not
        requested.
    forces
        Length-B list of (N_i, 3) float64 arrays in Hartree/Å. Per-structure
        atom counts may differ across the list; the caller is responsible for
        matching each `forces[i]` to its `atoms_list[i]`. `None` when 'forces'
        was not requested.
    hessians
        Length-B list of (3*N_i, 3*N_i) float64 arrays. `None` when 'hessian'
        was not requested.
    padding_counts
        Optional (B,) int array. Some batch back-ends (notably AIMNet2's
        existing GPU path) work in a padded DOF space and report the padding
        per entry so downstream code can slice. Empty (`None`) when no
        padding is in use.
    """

    energies: Optional[np.ndarray] = None
    forces: Optional[tuple[np.ndarray, ...]] = None
    hessians: Optional[tuple[np.ndarray, ...]] = None
    padding_counts: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        lengths = []
        if self.energies is not None:
            energies = np.array(self.energies, dtype=np.float64, copy=True)
            if energies.ndim != 1:
                raise ValueError(
                    f"BatchResult.energies must be a 1D array, got shape "
                    f"{energies.shape}"
                )
            if not np.all(np.isfinite(energies)):
                raise ValueError("BatchResult.energies must contain only finite values")
            energies.setflags(write=False)
            object.__setattr__(self, "energies", energies)
            lengths.append(("energies", len(energies)))
        if self.forces is not None:
            force_arrays = tuple(
                np.array(forces, dtype=np.float64, copy=True)
                for forces in self.forces
            )
            object.__setattr__(self, "forces", force_arrays)
            lengths.append(("forces", len(force_arrays)))
            for i, arr in enumerate(force_arrays):
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "BatchResult.forces entries must have shape (N, 3), "
                        f"got forces[{i}].shape={arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"BatchResult.forces[{i}] must contain only finite values"
                    )
                arr.setflags(write=False)
        if self.hessians is not None:
            hessian_arrays = tuple(
                np.array(hessian, dtype=np.float64, copy=True)
                for hessian in self.hessians
            )
            object.__setattr__(self, "hessians", hessian_arrays)
            lengths.append(("hessians", len(hessian_arrays)))
            for i, arr in enumerate(hessian_arrays):
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                    raise ValueError(
                        "BatchResult.hessians entries must be square 2D "
                        f"arrays, got hessians[{i}].shape={arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"BatchResult.hessians[{i}] must contain only finite values"
                    )
                arr.setflags(write=False)
        if self.padding_counts is not None:
            raw_padding = np.asarray(self.padding_counts)
            if raw_padding.ndim != 1:
                raise ValueError(
                    "BatchResult.padding_counts must be a 1D array, got "
                    f"shape {raw_padding.shape}"
                )
            try:
                numeric_padding = np.asarray(raw_padding, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "BatchResult.padding_counts must contain non-negative integers"
                ) from exc
            if (
                not np.all(np.isfinite(numeric_padding))
                or not np.all(numeric_padding == np.floor(numeric_padding))
            ):
                raise ValueError(
                    "BatchResult.padding_counts must contain exact integer values"
                )
            if np.any(numeric_padding < 0):
                raise ValueError(
                    "BatchResult.padding_counts must contain non-negative values"
                )
            if np.any(numeric_padding > np.iinfo(np.int64).max):
                raise ValueError(
                    "BatchResult.padding_counts values exceed the int64 range"
                )
            padding_counts = numeric_padding.astype(np.int64)
            padding_counts.setflags(write=False)
            object.__setattr__(self, "padding_counts", padding_counts)
            lengths.append(("padding_counts", len(padding_counts)))

        if lengths:
            expected_name, expected_len = lengths[0]
            for name, got_len in lengths[1:]:
                if got_len != expected_len:
                    raise ValueError(
                        "BatchResult field lengths must match: "
                        f"{expected_name} has length {expected_len}, "
                        f"{name} has length {got_len}"
                    )

        if self.forces is not None and self.hessians is not None:
            for i, (forces, hessian) in enumerate(zip(self.forces, self.hessians)):
                n_atoms = np.asarray(forces).shape[0]
                expected = (3 * n_atoms, 3 * n_atoms)
                arr = np.asarray(hessian)
                if arr.shape != expected:
                    raise ValueError(
                        "BatchResult.hessians entries must match the "
                        "corresponding force atom count: "
                        f"hessians[{i}].shape={arr.shape}, expected {expected}"
                    )

    def validate_against(
        self,
        atoms_list: Sequence,
        requested: Sequence[str] | str | None,
    ) -> "BatchResult":
        """Validate cardinality and per-structure shapes against a request.

        ``__post_init__`` validates each numeric field in isolation. This method
        completes the contract at the backend boundary, where the requested
        properties and the corresponding structures are both available.
        """
        self._revalidate_numeric_fields()
        atoms_list = list(atoms_list)
        if requested is None:
            requested_props = ("energy",)
        elif isinstance(requested, str):
            requested_props = (requested,)
        else:
            requested_props = tuple(requested)

        allowed = {"energy", "forces", "hessian"}
        unknown = sorted({str(prop) for prop in requested_props} - allowed)
        if unknown:
            raise ValueError(
                "BatchResult cannot validate unsupported requested properties: "
                + ", ".join(unknown)
            )

        required_fields = {
            "energy": ("energies", self.energies),
            "forces": ("forces", self.forces),
            "hessian": ("hessians", self.hessians),
        }
        for prop in requested_props:
            field_name, value = required_fields[prop]
            if value is None:
                raise ValueError(
                    f"BatchResult requested property {prop!r} but {field_name} is None"
                )

        batch_size = len(atoms_list)
        for field_name in ("energies", "forces", "hessians", "padding_counts"):
            value = getattr(self, field_name)
            if value is not None and len(value) != batch_size:
                raise ValueError(
                    f"BatchResult.{field_name} has length {len(value)}, "
                    f"expected batch size {batch_size}"
                )

        if self.forces is not None:
            for i, (atoms, forces) in enumerate(zip(atoms_list, self.forces)):
                expected = (len(atoms), 3)
                if forces.shape != expected:
                    raise ValueError(
                        f"BatchResult.forces[{i}].shape={forces.shape}, "
                        f"expected {expected}"
                    )

        if self.hessians is not None:
            for i, (atoms, hessian) in enumerate(
                zip(atoms_list, self.hessians)
            ):
                size = 3 * len(atoms)
                expected = (size, size)
                if hessian.shape != expected:
                    raise ValueError(
                        f"BatchResult.hessians[{i}].shape={hessian.shape}, "
                        f"expected {expected}"
                    )

        return self

    def _revalidate_numeric_fields(self) -> None:
        """Recheck stored arrays at every backend boundary.

        Arrays are copied and marked read-only during construction. Rechecking
        here also fails closed if a caller deliberately re-enables NumPy writes
        or otherwise mutates a nested value after construction.
        """
        if self.energies is not None:
            energies = np.asarray(self.energies)
            if energies.ndim != 1 or not np.all(np.isfinite(energies)):
                raise ValueError(
                    "BatchResult.energies must remain a finite 1D array"
                )

        if self.forces is not None:
            for i, forces in enumerate(self.forces):
                arr = np.asarray(forces)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "BatchResult.forces entries must retain shape (N, 3), "
                        f"got forces[{i}].shape={arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"BatchResult.forces[{i}] must contain only finite values"
                    )

        if self.hessians is not None:
            for i, hessian in enumerate(self.hessians):
                arr = np.asarray(hessian)
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                    raise ValueError(
                        "BatchResult.hessians entries must remain square 2D "
                        f"arrays, got hessians[{i}].shape={arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"BatchResult.hessians[{i}] must contain only finite values"
                    )

        if self.padding_counts is not None:
            padding = np.asarray(self.padding_counts)
            if (
                padding.ndim != 1
                or not np.issubdtype(padding.dtype, np.integer)
                or np.any(padding < 0)
            ):
                raise ValueError(
                    "BatchResult.padding_counts must remain a 1D array of "
                    "non-negative integers"
                )
