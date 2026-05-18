from __future__ import annotations

from maple.function.dispatcher.parmfit.utils.context import locate_context
from maple.function.dispatcher.parmfit.utils.structure import get_resid_info, read_pdb


__test__ = False


def recognize(
    pdb_path: str,
    target: str,
    keep: str = "",
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    structure = read_pdb(pdb_path, keep_altloc=keep_altloc)
    context = locate_context(structure, target=target, keep=keep.split(), bond_policy=bond_policy)
    return {
        "target": get_resid_info(context["target"]),
        "prev_residue": get_resid_info(context["prev_residue"]),
        "next_residue": get_resid_info(context["next_residue"]),
        "keep_residues": [get_resid_info(residue) for residue in context["keep_residues"]],
    }


def parse_pdb(path: str, keep_altloc: str = "A") -> list[dict]:
    structure = read_pdb(path, keep_altloc=keep_altloc)
    return [get_resid_info(residue) for residue in structure["residues"]]


__all__ = ["parse_pdb", "recognize"]
