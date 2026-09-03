"""Usage: provide public helpers for running and reading silent scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from .engine import SilentScanEngine
from .models import ScanConstraint, SilentScanOptions, SilentScanResult


def _constraint_rows(constraints: list) -> list:
    rows = []
    for constraint in constraints:
        if isinstance(constraint, ScanConstraint):
            rows.append(constraint.as_legacy_row())
        else:
            rows.append(list(constraint))
    return rows


def run_silent_scan(
    *,
    output: str,
    atoms: Atoms,
    constraints: list,
    options: SilentScanOptions | None = None,
    params: dict | None = None,
    method: str = "lbfgs",
    constraint_mode: str = "fixinternals",
) -> SilentScanResult:
    resolved_params = options.to_params() if options is not None else dict(params or {})
    if options is not None:
        method = options.backend
        constraint_mode = options.constraint_mode
    xyz_path = SilentScanEngine(
        output=output,
        atoms=atoms,
        constraints=_constraint_rows(constraints),
        params=resolved_params,
        method=method,
        constraint_mode=constraint_mode,
    ).run()
    return SilentScanResult(output_path=output, xyz_path=xyz_path)


def read_scan_final_atoms(xyz_path: str) -> Atoms:
    lines = Path(xyz_path).read_text(encoding="utf-8").splitlines()
    index = 0
    last_symbols: list[str] = []
    last_positions: list[list[float]] = []
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        natoms = int(lines[index].strip())
        frame_start = index + 2
        frame_end = frame_start + natoms
        if frame_end > len(lines):
            raise ValueError(f"Incomplete XYZ frame in scan file: {xyz_path}")
        symbols: list[str] = []
        positions: list[list[float]] = []
        for raw in lines[frame_start:frame_end]:
            parts = raw.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom row in scan file {xyz_path}: {raw!r}")
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        last_symbols = symbols
        last_positions = positions
        index = frame_end
    if not last_symbols:
        raise ValueError(f"Scan XYZ file contains no frames: {xyz_path}")
    return Atoms(symbols=last_symbols, positions=np.asarray(last_positions, dtype=float))
