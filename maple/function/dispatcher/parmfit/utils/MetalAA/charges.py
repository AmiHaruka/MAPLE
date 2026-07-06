"""Usage: infer and project charges for MetalAA models."""

from __future__ import annotations

from ..structure import CHARGED_STANDARD_RESIDUES, get_resid_key


def infer_model_charge(base_charge: int, model: dict) -> int:
    charge = int(base_charge)
    target_key = model.get("target_key")
    for residue in model["residues"]:
        residue_key = get_resid_key(residue)
        if residue_key == target_key:
            continue
        charge += CHARGED_STANDARD_RESIDUES.get(residue["resname"].upper(), 0)
    return charge


def project_resp_charges_onto_site_model(site_model: dict, charged_large_model: dict) -> tuple[dict, list[str]]:
    large_residues = {
        get_resid_key(residue): residue
        for residue in charged_large_model["residues"]
    }

    for residue in site_model["residues"]:
        residue_key = get_resid_key(residue)
        large_residue = large_residues.get(residue_key)
        if large_residue is None:
            raise ValueError(f"RESP charges for site residue {residue_key} were not found in large_model.")

        site_atoms = list(residue["atoms"])
        large_atoms = list(large_residue["atoms"])
        if len(site_atoms) != len(large_atoms):
            raise ValueError(
                f"RESP charge projection atom count mismatch for residue {residue_key}: "
                f"site has {len(site_atoms)} atoms, large_model has {len(large_atoms)}."
            )

        site_names = [atom["name"] for atom in site_atoms]
        large_names = [atom["name"] for atom in large_atoms]
        if site_names != large_names:
            raise ValueError(
                f"RESP charge projection atom order mismatch for residue {residue_key}: "
                f"site atoms {site_names}, large_model atoms {large_names}."
            )

        for atom, charged_atom in zip(site_atoms, large_atoms, strict=True):
            if "charge" not in charged_atom:
                raise ValueError(
                    f"RESP charge for site atom {residue_key}:{charged_atom['name']} "
                    "was not found in large_model."
                )
            atom["charge"] = float(charged_atom["charge"])
            if "atom_type" in atom and str(atom["atom_type"]).strip():
                atom["atom_type"] = str(atom["atom_type"]).strip()
            else:
                atom.pop("atom_type", None)
    return site_model, []
