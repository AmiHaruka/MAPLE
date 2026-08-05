"""Locate the AmberTools data tree supplied by the user's environment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def _candidate_data_dirs() -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        candidates.append(("AMBERHOME", Path(amberhome).expanduser() / "dat" / "leap"))

    tleap = shutil.which("tleap")
    if tleap:
        candidates.append(("tleap executable", Path(tleap).absolute().parent.parent / "dat" / "leap"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(("CONDA_PREFIX", Path(conda_prefix).expanduser() / "dat" / "leap"))

    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for source, path in candidates:
        normalized = path.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append((source, normalized))
    return unique


def amber_data_dir() -> Path:
    """Return an AmberTools ``dat/leap`` directory containing lib and parm."""

    candidates = _candidate_data_dirs()
    for _source, path in candidates:
        if (path / "lib").is_dir() and (path / "parm").is_dir():
            return path

    checked = ", ".join(f"{source}={path}" for source, path in candidates) or "no candidates"
    raise FileNotFoundError(
        "AmberTools force-field data were not found. Source amber.sh so AMBERHOME is set, "
        "or activate an AmberTools conda environment containing dat/leap. "
        f"Checked: {checked}."
    )


def amber_lib_dir() -> Path:
    return amber_data_dir() / "lib"


def amber_parm_dir() -> Path:
    return amber_data_dir() / "parm"
