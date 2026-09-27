"""Small shared helpers for preserving prior TS-run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union


PathLike = Union[str, Path]


def archive_artifacts(
    paths: Iterable[PathLike], history_dir: PathLike
) -> Optional[Path]:
    """Move existing artifacts into a new numbered history directory.

    No history directory is created when none of ``paths`` exists. Each call
    containing existing files gets a distinct ``run-NNNN`` directory.
    """
    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        return None

    history = Path(history_dir)
    history.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = history / f"run-{index:04d}"
        try:
            destination.mkdir()
        except FileExistsError:
            index += 1
            continue
        break

    for source in existing:
        source.replace(destination / source.name)
    return destination
