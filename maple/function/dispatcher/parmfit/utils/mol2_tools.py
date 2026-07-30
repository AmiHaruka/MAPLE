"""Token-preserving MOL2 coordinate, charge, and atom-type updates."""

from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np


_TOKEN_RE = re.compile(r"\S+")


def _replacement_map(
    atom_index: int,
    *,
    positions: np.ndarray | None,
    charges: np.ndarray | None,
    atom_types: tuple[str, ...] | None,
) -> dict[int, str]:
    replacements: dict[int, str] = {}
    if positions is not None:
        replacements.update(
            {
                2: f"{positions[atom_index, 0]:.6f}",
                3: f"{positions[atom_index, 1]:.6f}",
                4: f"{positions[atom_index, 2]:.6f}",
            }
        )
    if atom_types is not None:
        replacements[5] = atom_types[atom_index]
    if charges is not None:
        replacements[8] = f"{charges[atom_index]:.8f}"
    return replacements


def write_updated_mol2(
    source_path: str | os.PathLike[str],
    target_path: str | os.PathLike[str],
    *,
    positions=None,
    charges=None,
    atom_types=None,
) -> str:
    """Update selected atom fields while preserving all other MOL2 content."""
    source_path = os.fspath(source_path)
    target_path = os.fspath(target_path)
    position_array = None if positions is None else np.asarray(positions, dtype=float)
    charge_array = None if charges is None else np.asarray(charges, dtype=float).reshape(-1)
    type_values = None if atom_types is None else tuple(str(value) for value in atom_types)

    expected = None
    for values in (position_array, charge_array, type_values):
        if values is None:
            continue
        count = int(values.shape[0]) if hasattr(values, "shape") else len(values)
        if expected is None:
            expected = count
        elif count != expected:
            raise ValueError("MOL2 update arrays must have identical atom counts.")
    if position_array is not None and (
        position_array.ndim != 2 or position_array.shape[1] != 3
    ):
        raise ValueError("MOL2 positions must have shape (N, 3).")
    if charge_array is not None and not np.all(np.isfinite(charge_array)):
        raise ValueError("MOL2 charges must be finite.")

    with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    output_lines: list[str] = []
    in_atoms = False
    atom_index = 0
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("@<TRIPOS>"):
            in_atoms = stripped.upper() == "@<TRIPOS>ATOM"
            output_lines.append(raw)
            continue
        if not in_atoms or not stripped:
            output_lines.append(raw)
            continue

        spans = [match.span() for match in _TOKEN_RE.finditer(raw)]
        if len(spans) < 6:
            raise ValueError(f"Invalid MOL2 atom line in {source_path}: {raw.rstrip()}")
        replacements = _replacement_map(
            atom_index,
            positions=position_array,
            charges=charge_array,
            atom_types=type_values,
        )
        appended_fields: list[str] = []
        if charge_array is not None and len(spans) < 9:
            charge_value = replacements.pop(8)
            if len(spans) == 6:
                appended_fields.extend(("1", "MOL"))
            elif len(spans) == 7:
                appended_fields.append("MOL")
            appended_fields.append(charge_value)
        updated = raw
        for token_index in sorted(replacements, reverse=True):
            start, end = spans[token_index]
            updated = updated[:start] + replacements[token_index] + updated[end:]
        if appended_fields:
            newline = "\n" if updated.endswith("\n") else ""
            updated = updated.rstrip("\r\n") + " " + " ".join(appended_fields) + newline
        output_lines.append(updated)
        atom_index += 1

    if expected is not None and atom_index != expected:
        raise ValueError(
            f"MOL2 atom count ({atom_index}) does not match update size ({expected})."
        )
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as handle:
        handle.writelines(output_lines)
    return os.path.abspath(target_path)
