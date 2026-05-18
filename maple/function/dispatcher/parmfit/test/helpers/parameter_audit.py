"""Usage: inspect generated parmfit parameter files in tests."""

from __future__ import annotations

from pathlib import Path


def gromacs_section_rows(path: str | Path, section_name: str) -> list[list[str]]:
    """Return tokenized, non-comment rows from a GROMACS topology section."""
    target = section_name.strip().lower()
    active: str | None = None
    rows: list[list[str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            active = stripped.strip("[]").strip().lower()
            continue
        if active != target or stripped.startswith(";"):
            continue
        row = stripped.split(";", 1)[0].split()
        if row:
            rows.append(row)
    return rows


def gromacs_row_by_atoms(path: str | Path, section_name: str, atoms: tuple[int, ...]) -> list[str]:
    """Find the first topology row matching an explicit atom-index tuple."""
    atom_tokens = tuple(str(atom) for atom in atoms)
    for row in gromacs_section_rows(path, section_name):
        if tuple(row[: len(atom_tokens)]) == atom_tokens:
            return row
    raise AssertionError(f"Could not find GROMACS [{section_name}] row for atoms {atoms}.")
