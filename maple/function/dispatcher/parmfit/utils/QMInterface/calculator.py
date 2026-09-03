"""Lightweight ASE-compatible external QM force calculator."""

from __future__ import annotations

from pathlib import Path
import os
from dataclasses import dataclass

import numpy as np
from ase import Atoms

@dataclass(frozen=True)
class QMReferenceConfig:
    iqm: bool = False
    qm_engine: str = "g16"
    opt_level: str = "B3LYP/def2-SVP"
    sp_level: str = "B3LYP/def2-SVP"
    opt_route: str = ""
    sp_route: str = ""
    qm_nproc: int = 8
    qm_mem: int = 24
    qm_mode: int = 1
    qm_compare: bool = False


@dataclass(frozen=True)
class QMReferenceResult:
    atoms: Atoms
    energy_hartree: float
    input_path: str
    log_path: str
    hessian: np.ndarray | None = None
    wfn_path: str | None = None
    

class QMExternalCalculator:
    def __init__(
        self,
        runner,
        *,
        work_dir: str | os.PathLike[str],
        charge: int | None = None,
        multiplicity: int | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.runner = runner
        self.work_dir = Path(work_dir)
        self.charge = charge
        self.multiplicity = multiplicity
        self.last_wfn_path = os.fspath(wfn_path) if wfn_path is not None else None
        self._eval_count = 0
        self._cache_symbols: tuple[str, ...] | None = None
        self._cache_positions: np.ndarray | None = None
        self._cache_energy: float | None = None
        self._cache_forces: np.ndarray | None = None

    def _cache_matches(self, atoms: Atoms) -> bool:
        if self._cache_symbols is None or self._cache_positions is None:
            return False
        if tuple(atoms.get_chemical_symbols()) != self._cache_symbols:
            return False
        positions = np.asarray(atoms.get_positions(), dtype=float)
        return positions.shape == self._cache_positions.shape and np.allclose(
            positions,
            self._cache_positions,
            rtol=0.0,
            atol=1.0e-10,
        )

    def _evaluate(self, atoms: Atoms) -> None:
        if self._cache_matches(atoms):
            return
        self._eval_count += 1
        job_dir = self.work_dir / f"eval_{self._eval_count:06d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"wfn_path": self.last_wfn_path}
        if self.charge is not None:
            kwargs["charge"] = self.charge
        if self.multiplicity is not None:
            kwargs["multiplicity"] = self.multiplicity
        energy, forces, wfn_path = self.runner.gradient(
            atoms,
            str(job_dir / "eval"),
            **kwargs,
        )
        self.last_wfn_path = os.fspath(wfn_path)
        self._cache_symbols = tuple(atoms.get_chemical_symbols())
        self._cache_positions = np.asarray(atoms.get_positions(), dtype=float).copy()
        self._cache_energy = float(energy)
        self._cache_forces = np.asarray(forces, dtype=float).copy()

    def get_potential_energy(self, atoms: Atoms | None = None, force_consistent: bool = False) -> float:
        del force_consistent
        if atoms is None:
            raise ValueError("QMExternalCalculator requires atoms for energy evaluation.")
        self._evaluate(atoms)
        if self._cache_energy is None:
            raise RuntimeError("QM energy cache was not populated.")
        return float(self._cache_energy)

    def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
        if atoms is None:
            raise ValueError("QMExternalCalculator requires atoms for force evaluation.")
        self._evaluate(atoms)
        if self._cache_forces is None:
            raise RuntimeError("QM force cache was not populated.")
        return np.asarray(self._cache_forces, dtype=float).copy()
