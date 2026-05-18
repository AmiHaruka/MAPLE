from __future__ import annotations

from maple.function.dispatcher.parmfit.utils.capping import build_ace_cap, build_nme_cap
from maple.function.dispatcher.parmfit.utils.context import locate_context
from maple.function.dispatcher.parmfit.utils.structure import (
    ION_ELEMENTS,
    _norm,
    boundary_h_length,
    copy_residue,
    find_external_partners,
    get_atom_xyz,
    get_resid_info,
    is_peptide_like,
    make_atom,
    make_h_name,
    max_serial,
    read_pdb,
    refresh_resid,
    search_atom,
)


__test__ = False

def _skip_boundary_partner(target: dict, partner: dict, prev_residue: dict, next_residue: dict) -> bool:
    if is_peptide_like(target):
        n_atom = search_atom(target, "N")
        c_atom = search_atom(target, "C")
        prev_c = search_atom(prev_residue, "C") if prev_residue is not None else None
        next_n = search_atom(next_residue, "N") if next_residue is not None else None
        if n_atom is not None and prev_c is not None and partner["serial"] == prev_c["serial"]:
            return True
        if c_atom is not None and next_n is not None and partner["serial"] == next_n["serial"]:
            return True
    return False


def _add_boundary_hydrogens(
    structure: dict,
    selected_residues: list[dict],
    target_original: dict,
    prev_residue: dict,
    next_residue: dict,
    bond_policy: str,
    next_serial: int,
) -> int:
    selected_serials = {atom["serial"] for residue in selected_residues for atom in residue["atoms"]}

    for residue in selected_residues:
        if residue["kind"] == "ion":
            continue
        original_residue = structure["serial_to_residue"].get(residue["atoms"][0]["serial"]) if residue["atoms"] else None
        for atom in list(residue["atoms"]):
            if atom["element"] in ION_ELEMENTS or atom["element"] == "H":
                continue
            original_atom = structure["serial_to_atom"].get(atom["serial"])
            if original_atom is None:
                continue
            partners = find_external_partners(structure, original_atom, selected_serials, bond_policy=bond_policy)
            for partner in partners:
                if original_residue is target_original and _skip_boundary_partner(target_original, partner, prev_residue, next_residue):
                    continue
                direction = get_atom_xyz(partner) - get_atom_xyz(original_atom)
                try:
                    h_direction = _norm(direction)
                except ValueError:
                    continue
                h_name = make_h_name(residue, atom)
                residue["atoms"].append(
                    make_atom(next_serial, h_name, "H", get_atom_xyz(atom) + h_direction * boundary_h_length(atom))
                )
                next_serial += 1
        refresh_resid(residue)
    return next_serial


def addcap(
    pdb_path: str,
    target: str,
    rn: str,
    keep: str = "",
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    structure = read_pdb(pdb_path, keep_altloc=keep_altloc)
    context = locate_context(structure, target=target, keep=keep.split(), bond_policy=bond_policy)

    target_original = context["target"]
    prev_residue = context["prev_residue"]
    next_residue = context["next_residue"]
    keep_residues = context["keep_residues"]

    target_residue = copy_residue(target_original, resname=rn)
    kept = [copy_residue(residue) for residue in keep_residues]
    selected = [target_residue] + kept
    caps: list[dict] = []

    next_serial = max_serial(structure) + 1
    if is_peptide_like(target_original):
        ace_residue, next_serial = build_ace_cap(target_original, next_serial, prev_residue=prev_residue)
        nme_residue, next_serial = build_nme_cap(target_original, next_serial, next_residue=next_residue)
        caps.extend([ace_residue, nme_residue])

    _add_boundary_hydrogens(
        structure,
        selected_residues=selected,
        target_original=target_original,
        prev_residue=prev_residue,
        next_residue=next_residue,
        bond_policy=bond_policy,
        next_serial=next_serial,
    )

    assembled = []
    if caps:
        assembled.append(caps[0])
    assembled.append(target_residue)
    assembled.extend(kept)
    if len(caps) == 2:
        assembled.append(caps[1])

    return {
        "target": get_resid_info(target_residue),
        "caps": [get_resid_info(cap) for cap in caps],
        "assembled_residues": [get_resid_info(residue) for residue in assembled],
        "rn": rn,
    }


def add_caps(
    pdb_path: str,
    target: str,
    rn: str,
    keep: str = "",
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    return addcap(pdb_path, target=target, rn=rn, keep=keep, keep_altloc=keep_altloc, bond_policy=bond_policy)


__all__ = ["add_caps", "addcap"]
