"""Usage: infer and project charges for MetalAA models."""

from __future__ import annotations

from ..structure import CHARGED_STANDARD_RESIDUES, get_resid_key


def infer_large_model_charge(metal_formal_charge: int, large_model: dict) -> int:
    charge = int(metal_formal_charge)
    target_key = large_model.get("target_key")
    for residue in large_model["residues"]:
        residue_key = get_resid_key(residue)
        if residue_key == target_key:
            continue
        charge += CHARGED_STANDARD_RESIDUES.get(residue["resname"].upper(), 0)
    return charge


def project_resp_charges_onto_site_model(site_model: dict, charged_large_model: dict) -> tuple[dict, list[str]]:
    charge_map: dict[int, float] = {}
    for residue in charged_large_model["residues"]:
        for atom in residue["atoms"]:
            if "charge" not in atom:
                continue
            charge_map[int(atom["serial"])] = float(atom["charge"])

    for residue in site_model["residues"]:
        for atom in residue["atoms"]:
            serial = int(atom["serial"])
            if serial not in charge_map:
                residue_key = get_resid_key(residue)
                raise ValueError(
                    f"RESP charge for original atom serial {serial} ({residue_key}:{atom['name']}) "
                    "was not found in large_model."
                )
            atom["charge"] = charge_map[serial]
            if "atom_type" in atom and str(atom["atom_type"]).strip():
                atom["atom_type"] = str(atom["atom_type"]).strip()
            else:
                atom.pop("atom_type", None)
    return site_model, []
