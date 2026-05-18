"""Usage: recognize metal sites and cofactor templates for MetalAA."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import interface as amber_interface
from ..context import collect_environment_residues, find_prev_next_peptide_residues, find_unique_residue
from ..readparm import Mol2Topology, parse_mol2
from ..runtime import parmfit_output_dir
from ..structure import (
    CHARGED_STANDARD_RESIDUES,
    METAL_SITE_DONOR_ELEMENTS,
    get_atom_xyz,
    get_resid_info,
    get_resid_key,
    get_resid_label,
    get_resid_mindist,
    read_pdb,
    residue_sort_key,
)


@dataclass(frozen=True)
class MetalSiteSelection:
    target: dict
    core_residues: list[dict]
    auto_core_residues: list[dict]
    manual_core_residues: list[dict]
    donor_atoms: dict[tuple[str, int, str], list[str]]
    warnings: list[str]
    donor_cutoff: float

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "core_residues": self.core_residues,
            "auto_core_residues": self.auto_core_residues,
            "manual_core_residues": self.manual_core_residues,
            "donor_atoms": self.donor_atoms,
            "warnings": self.warnings,
            "donor_cutoff": self.donor_cutoff,
        }


MetalSiteCore = MetalSiteSelection


@dataclass(frozen=True)
class CofactorMol2Template:
    path: str
    residue_key: tuple[str, int, str]
    resname: str
    mol2: Mol2Topology


def _safe_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    return token.strip("_") or "cofactor"


def _mol2_atoms_by_name(mol2: Mol2Topology, path: str) -> dict[str, object]:
    atoms_by_name = {}
    duplicates = []
    for atom in mol2.atoms:
        if atom.name in atoms_by_name:
            duplicates.append(atom.name)
        atoms_by_name[atom.name] = atom
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"cfmol2 {path} has duplicate atom names: {joined}")
    return atoms_by_name


def _match_residue_by_atom_names(
    structure: dict,
    mol2_names: set[str],
    path: str,
    claimed_keys: set[tuple[str, int, str]],
) -> dict:
    candidates = []
    for residue in structure["residues"]:
        if residue.get("kind") not in {"ligand", "cofactor"}:
            continue
        if get_resid_key(residue) in claimed_keys:
            continue
        residue_names = {atom["name"] for atom in residue["atoms"]}
        if residue_names == mol2_names:
            candidates.append(residue)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        joined = ", ".join(sorted(mol2_names))
        raise ValueError(f"Could not match cfmol2 {path} to a ligand/cofactor residue by atom names: {joined}")
    labels = ", ".join(get_resid_label(residue) for residue in candidates)
    raise ValueError(f"cfmol2 {path} matches multiple ligand/cofactor residues: {labels}")


def _inject_atom_types_and_bonds(
    structure: dict,
    residue: dict,
    mol2: Mol2Topology,
    atoms_by_name: dict[str, object],
    path: str,
) -> None:
    residue_atoms_by_name = {atom["name"]: atom for atom in residue["atoms"]}
    for name, mol2_atom in atoms_by_name.items():
        residue_atoms_by_name[name]["atom_type"] = mol2_atom.atom_type
        residue_atoms_by_name[name]["cfmol2_charge"] = mol2_atom.charge
        residue_atoms_by_name[name]["_cfmol2_path"] = path
    residue["_cfmol2_path"] = path

    atom_id_to_name = {atom.atom_id: atom.name for atom in mol2.atoms}
    explicit_pairs = structure.setdefault("explicit_pairs", set())
    for bond in mol2.bonds:
        left_name = atom_id_to_name[bond.atom1]
        right_name = atom_id_to_name[bond.atom2]
        left = residue_atoms_by_name[left_name]["serial"]
        right = residue_atoms_by_name[right_name]["serial"]
        explicit_pairs.add(tuple(sorted((int(left), int(right)))))
    structure.setdefault("_pair_cache", {}).clear()


def apply_cfmol2_templates(structure: dict, cfmol2_paths: list[str]) -> list[CofactorMol2Template]:
    templates: list[CofactorMol2Template] = []
    claimed_keys: set[tuple[str, int, str]] = set()
    for path in cfmol2_paths:
        mol2 = parse_mol2(path)
        atoms_by_name = _mol2_atoms_by_name(mol2, path)
        residue = _match_residue_by_atom_names(structure, set(atoms_by_name), path, claimed_keys)
        _inject_atom_types_and_bonds(structure, residue, mol2, atoms_by_name, path)
        residue_key = get_resid_key(residue)
        claimed_keys.add(residue_key)
        templates.append(
            CofactorMol2Template(
                path=path,
                residue_key=residue_key,
                resname=residue["resname"].upper(),
                mol2=mol2,
            )
        )
    return templates


def build_cofactor_orig_frcmods(output: str, templates: list[CofactorMol2Template]) -> list[str]:
    if not templates:
        return []

    final_dir = parmfit_output_dir(output)
    base = _safe_token(os.path.splitext(os.path.basename(output))[0])
    frcmods: list[str] = []
    for index, template in enumerate(templates, start=1):
        input_path = os.path.abspath(template.path)
        workdir = os.path.dirname(input_path)
        residue_name = f"{base}_{_safe_token(template.resname)}_{index}_orig"
        result = amber_interface.run_parmchk2(
            os.path.basename(input_path),
            {"residue_name": residue_name},
            True,
            workdir,
        )
        final_path = os.path.join(final_dir, f"{residue_name}.frcmod")
        if os.path.abspath(result.frcmod_path) != os.path.abspath(final_path):
            shutil.move(result.frcmod_path, final_path)
        frcmods.append(final_path)
    return frcmods


def _residue_has_formal_charge_hint(residue: dict) -> bool:
    return residue["resname"].upper() in CHARGED_STANDARD_RESIDUES


def find_metal_site_core(
    structure: dict,
    *,
    target: str | None = None,
    target_residue: dict | None = None,
    add_resid: Optional[list[str]] = None,
    donor_cutoff: float = 3.0,
    bond_policy: str = "auto",
) -> MetalSiteSelection:
    if target_residue is None:
        if target is None:
            raise ValueError("Metal site selection requires either 'target' or 'target_residue'.")
        target_residue = find_unique_residue(structure, target)
    if target_residue["kind"] != "ion":
        raise ValueError(f"Target {get_resid_label(target_residue)} is not recognized as an ion residue.")

    metal_atom = target_residue["atoms"][0]
    metal_xyz = get_atom_xyz(metal_atom)
    manual_residues = [find_unique_residue(structure, selector) for selector in (add_resid or [])]

    auto_residues: list[dict] = []
    donor_atoms: dict[tuple[str, int, str], list[str]] = {}
    for residue in structure["residues"]:
        if get_resid_key(residue) == get_resid_key(target_residue):
            continue
        if residue["kind"] == "ion":
            distance = get_resid_mindist(target_residue, residue)
            if distance <= donor_cutoff:
                raise ValueError(
                    f"Metal route currently supports a single metal center; found neighboring ion {get_resid_label(residue)}."
                )
            continue

        ligand_names: list[str] = []
        for atom in residue["atoms"]:
            if atom["element"] not in METAL_SITE_DONOR_ELEMENTS:
                continue
            delta = get_atom_xyz(atom) - metal_xyz
            distance = float(np.sqrt(np.dot(delta, delta)))
            if distance <= donor_cutoff:
                ligand_names.append(atom["name"])
        if ligand_names:
            donor_atoms[get_resid_key(residue)] = sorted(set(ligand_names))
            if residue["kind"] == "protein":
                auto_residues.append(residue)
            elif residue["kind"] in {"ligand", "cofactor"}:
                if residue.get("_cfmol2_path"):
                    auto_residues.append(residue)
                else:
                    raise ValueError(
                        f"Non-protein metal donor {get_resid_label(residue)} was found within donor_cutoff, "
                        "but no matching cfmol2 atom types were provided."
                    )

    residues: list[dict] = [target_residue]
    seen = {get_resid_key(target_residue)}
    for residue in sorted(auto_residues, key=residue_sort_key):
        key = get_resid_key(residue)
        if key not in seen:
            residues.append(residue)
            seen.add(key)
    for residue in sorted(manual_residues, key=residue_sort_key):
        key = get_resid_key(residue)
        if key not in seen:
            residues.append(residue)
            seen.add(key)

    warnings: list[str] = []
    for residue in manual_residues:
        key = get_resid_key(residue)
        if key == get_resid_key(target_residue):
            continue
        if key not in donor_atoms and residue["kind"] == "protein":
            warnings.append(
                f"Manual core residue {get_resid_label(residue)} was added without an automatically detected donor atom."
            )

    for residue in residues:
        if residue["kind"] != "protein":
            continue
        prev_residue, next_residue = find_prev_next_peptide_residues(structure, residue, bond_policy=bond_policy)
        residue["_prev_peptide_key"] = get_resid_key(prev_residue) if prev_residue is not None else None
        residue["_next_peptide_key"] = get_resid_key(next_residue) if next_residue is not None else None

    environment_charge_hints = [
        get_resid_label(residue)
        for residue in structure["residues"]
        if get_resid_key(residue) not in seen and _residue_has_formal_charge_hint(residue)
    ]
    if environment_charge_hints:
        warnings.append(
            "Environment contains potentially charged standard residues outside the site core: "
            + ", ".join(environment_charge_hints[:8])
        )

    return MetalSiteSelection(
        target=target_residue,
        core_residues=residues,
        auto_core_residues=sorted(auto_residues, key=residue_sort_key),
        manual_core_residues=sorted(
            [residue for residue in manual_residues if get_resid_key(residue) != get_resid_key(target_residue)],
            key=residue_sort_key,
        ),
        donor_atoms=donor_atoms,
        warnings=warnings,
        donor_cutoff=float(donor_cutoff),
    )


def identify_metal_site_core(
    structure: dict | str,
    target: str,
    add_resid: str = "",
    donor_cutoff: float = 3.0,
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    structure = read_pdb(structure, keep_altloc=keep_altloc) if isinstance(structure, str) else structure
    return find_metal_site_core(
        structure,
        target=target,
        add_resid=add_resid.split(),
        donor_cutoff=donor_cutoff,
        bond_policy=bond_policy,
    ).to_dict()


def extract_metal_cluster(
    structure: dict | str,
    target: str,
    add_resid: str = "",
    cluster_cutoff: float = 3.0,
    donor_cutoff: float = 3.0,
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    if cluster_cutoff < 0.0:
        raise ValueError(f"cluster_cutoff must be >= 0.0, got {cluster_cutoff}.")

    structure = read_pdb(structure, keep_altloc=keep_altloc) if isinstance(structure, str) else structure
    core = find_metal_site_core(
        structure,
        target=target,
        add_resid=add_resid.split(),
        donor_cutoff=donor_cutoff,
        bond_policy=bond_policy,
    )
    excluded = {get_resid_key(residue) for residue in core.core_residues}
    environment = collect_environment_residues(
        structure,
        core.target,
        cutoff=cluster_cutoff,
        excluded_keys=excluded,
        include_water=False,
    )

    warnings = list(core.warnings)
    charged_env = [get_resid_label(residue) for residue in environment if _residue_has_formal_charge_hint(residue)]
    if charged_env:
        warnings.append(
            "Cluster environment includes potentially charged standard residues outside the RESP/Hessian core: "
            + ", ".join(charged_env[:8])
        )

    return {
        "target": get_resid_info(core.target),
        "core_residues": [get_resid_info(residue) for residue in core.core_residues],
        "auto_core_residues": [get_resid_info(residue) for residue in core.auto_core_residues],
        "keep_residues": [get_resid_info(residue) for residue in core.manual_core_residues],
        "environment_residues": [get_resid_info(residue) for residue in environment],
        "donor_atoms": {f"{key[0]}{key[1]}{key[2]}": list(names) for key, names in core.donor_atoms.items()},
        "cutoff": float(cluster_cutoff),
        "warnings": warnings,
    }
