"""Usage: build large and site models for the MetalAA workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..context import collect_environment_residues
from ..model import build_capped_selected_model, copy_structure_subset
from ..structure import get_resid_key, residue_sort_key
from .recognize import MetalSiteSelection, find_metal_site_core


@dataclass
class MetalModelBundle:
    selection: MetalSiteSelection
    large_model: dict
    site_model: dict | None = None
    large_charge: int | None = None
    large_mult: int | None = None


def build_metal_site_model(
    structure: dict,
    core_residues: list[dict],
    donor_atoms: Optional[dict[tuple[str, int, str], list[str]]] = None,
    *,
    bond_policy: str = "auto",
) -> dict:
    del bond_policy
    core_keys = [get_resid_key(residue) for residue in sorted(core_residues, key=residue_sort_key)]
    model = copy_structure_subset(structure, core_residues)
    model["name"] = "site_model"
    ion_residue = next((residue for residue in model["residues"] if residue["kind"] == "ion"), None)
    model["target_key"] = get_resid_key(ion_residue) if ion_residue is not None else None
    model["core_keys"] = core_keys
    model["donor_atoms"] = dict(donor_atoms or {})
    return model


def build_metal_large_model(
    structure: dict,
    target: str,
    add_resid: Optional[list[str]] = None,
    cluster_cutoff: float = 3.0,
    donor_cutoff: float = 2.7,
    bond_policy: str = "auto",
    core: Optional[MetalSiteSelection] = None,
) -> dict:
    selection = core or find_metal_site_core(
        structure,
        target=target,
        add_resid=add_resid,
        donor_cutoff=donor_cutoff,
        bond_policy=bond_policy,
    )
    core_keys = {get_resid_key(residue) for residue in selection.core_residues}
    environment = collect_environment_residues(
        structure,
        selection.target,
        cutoff=cluster_cutoff,
        excluded_keys=core_keys,
        include_water=False,
    )
    model = build_capped_selected_model(
        structure,
        selection.core_residues + environment,
        bond_policy=bond_policy,
    )
    model["name"] = "large_model"
    model["target_key"] = get_resid_key(selection.target)
    model["core_keys"] = [get_resid_key(residue) for residue in sorted(selection.core_residues, key=residue_sort_key)]
    model["environment_keys"] = [get_resid_key(residue) for residue in sorted(environment, key=residue_sort_key)]
    model["donor_atoms"] = dict(selection.donor_atoms)
    model["warnings"] = list(selection.warnings)
    model["cluster_cutoff"] = float(cluster_cutoff)
    return model


def build_metal_model_bundle(
    structure: dict,
    *,
    target: str,
    add_resid: Optional[list[str]] = None,
    cluster_cutoff: float = 3.0,
    donor_cutoff: float = 2.7,
    bond_policy: str = "auto",
    selection: Optional[MetalSiteSelection] = None,
) -> MetalModelBundle:
    resolved_selection = selection or find_metal_site_core(
        structure,
        target=target,
        add_resid=add_resid,
        donor_cutoff=donor_cutoff,
        bond_policy=bond_policy,
    )
    return MetalModelBundle(
        selection=resolved_selection,
        large_model=build_metal_large_model(
            structure,
            target=target,
            add_resid=add_resid,
            cluster_cutoff=cluster_cutoff,
            donor_cutoff=donor_cutoff,
            bond_policy=bond_policy,
            core=resolved_selection,
        ),
    )
