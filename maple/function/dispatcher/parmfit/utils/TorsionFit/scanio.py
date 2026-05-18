"""Usage: read torsion scan xyz files into scan data objects."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .records import TorsionScanData


HARTREE_TO_KCAL_MOL = 627.509474


def _parse_scan_comment(comment: str) -> tuple[float, float]:
    coords = comment.split("[", 1)[1].split("]", 1)[0]
    angle_deg = float(coords.split(",", 1)[0].strip())
    energy_hartree = float(comment.split("Energy =", 1)[1].split()[0])
    return angle_deg, energy_hartree


def _relative_to_reference(values: np.ndarray, ref_idx: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - values[ref_idx]


def read_scan_xyz(path: str) -> TorsionScanData:
    angles_deg: list[float] = []
    qm_hartree: list[float] = []
    qm_kcal: list[float] = []
    frames: list[Atoms] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            natoms_line = handle.readline()
            if natoms_line == "":
                break
            if not natoms_line.strip():
                continue

            natoms = int(natoms_line.strip())
            angle_deg, energy_hartree = _parse_scan_comment(handle.readline())

            symbols: list[str] = []
            positions: list[tuple[float, float, float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                symbols.append(parts[0])
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))

            angles_deg.append(angle_deg)
            qm_hartree.append(energy_hartree)
            qm_kcal.append(energy_hartree * HARTREE_TO_KCAL_MOL)
            frames.append(Atoms(symbols=symbols, positions=positions))

    if not frames:
        raise ValueError(f"No scan points were read from {path}")

    angles_array = np.asarray(angles_deg, dtype=float)
    qm_hartree_array = np.asarray(qm_hartree, dtype=float)
    qm_kcal_array = np.asarray(qm_kcal, dtype=float)
    ref_idx = int(np.argmin(qm_kcal_array))
    return TorsionScanData(
        angles_deg=angles_array,
        qm_hartree=qm_hartree_array,
        qm_kcal=qm_kcal_array,
        frames=frames,
        source_path=str(path),
        ref_idx=ref_idx,
        qm_rel=_relative_to_reference(qm_kcal_array, ref_idx),
    )
