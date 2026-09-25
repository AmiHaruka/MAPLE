"""Compatibility gate for the shared PDB writer introduced by upstream PR57."""

from __future__ import annotations

from importlib import import_module


def require_shared_pdb_writer(structures, owner: str) -> None:
    """Fail closed until upstream's format-aware writer is importable."""
    if not any(atoms.info.get("pdb_template") for atoms in structures):
        return
    try:
        writer = import_module(
            "maple.function.read.filereader.pdb_reader"
        )
    except ImportError as exc:
        raise NotImplementedError(
            f"{owner} PDB output requires the shared format-aware writer; "
            "the isolated batch branch fails closed."
        ) from exc
    required = ("write_pdb", "write_pdb_model", "write_pdb_trajectory")
    if any(not callable(getattr(writer, name, None)) for name in required):
        raise NotImplementedError(
            f"{owner} PDB output requires the complete shared format-aware "
            "writer contract."
        )
