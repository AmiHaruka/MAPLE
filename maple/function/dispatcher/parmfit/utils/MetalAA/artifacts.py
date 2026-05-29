"""Usage: write MetalAA atom types, frcmod, mol2, PDB, and tleap files."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from ..ionparams import infer_ion_frcmod_name
from ..model import copy_structure_subset, flatten_model_atoms, infer_bond_pairs, rebuild_model_index, write_model_pdb
from ..resp import lookup_standard_atom_entry, lookup_standard_residue_entry, write_resp_mol2
from ..runtime import parmfit_output_dir
from ..structure import ATOMIC_MASSES, copy_residue, get_resid_key, get_resid_label, residue_sort_key
from .parameters import (
    AmberParameterDB,
    TorsionParameter,
    canonical_angle,
    canonical_pair,
    load_parameters,
    lookup_ion_lj_from_frcmod,
    match_dihedral_template,
    match_improper,
    parse_amber_frcmod,
)

_LOCAL_TYPE_LETTERS = ("Y", "Z", "U", "V", "I", "J", "K", "L", "N", "P", "Q", "R", "S", "T", "W", "X", "B", "E")
_TYPE_DIGITS = "123456789ABCDEF0"
_WATER_REFERENCE_TYPES = {"O": "OW", "H": "HW"}


# ---------------------------------------------------------------------------
# Records and Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetalArtifacts:
    files: dict[str, str]
    resp_files: dict[str, str]
    mol2_files: dict[str, str] = field(default_factory=dict)
    cofactor_frcmods: list[str] = field(default_factory=list)
    cofactor_frcmod_by_residue: dict[tuple[str, int, str], str] = field(default_factory=dict)
    tleap_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetalAtomTypeRow:
    residue: str
    atom_name: str
    element: str
    old_type: str
    new_type: str
    charge: float


@dataclass(frozen=True)
class MetalSiteTyping:
    atom_type_rows: list[MetalAtomTypeRow]
    mol2_atom_types: dict[int, str]
    atom_type_overrides: dict[int, str]
    old_type_by_index: dict[int, str]
    renamed_atom_indices: set[int]
    ion_frcmods: list[str]
    residue_names: dict[tuple[str, int, str], str] = field(default_factory=dict)
    mol2_files: dict[str, str] = field(default_factory=dict)
    cofactor_frcmods: list[str] = field(default_factory=list)
    cofactor_frcmod_by_residue: dict[tuple[str, int, str], str] = field(default_factory=dict)
    tleap_lines: list[str] = field(default_factory=list)
    metal_formal_charge: int | None = None
    metal_fitted_charge: float | None = None
    watm: str = "opc"
    ionm: str = "12_6"
    prom: str = "ff14SB"


def _output_paths(output: str) -> dict[str, str]:
    output_dir = parmfit_output_dir(output)
    base_name = os.path.splitext(os.path.basename(output))[0]
    base = os.path.join(output_dir, base_name)
    return {
        "large_raw_pdb": f"{base}_metal_large_raw.pdb",
        "large_opt_pdb": f"{base}_metal_large_opt.pdb",
        "site_pdb": f"{base}_metal_site_opt.pdb",
        "mol2": f"{base}_metal_site.mol2",
        "site_mol2": f"{base}_metal_site.mol2",
        "frcmod": f"{base}_metal.frcmod",
        "tleap_pdb": f"{base}_metal_tleap.pdb",
        "tleap_input": f"{base}_metal_tleap.in",
    }


def plan_metal_artifacts(
    output: str,
    *,
    gaussian_input: str = "",
    resp_files: dict[str, str] | None = None,
) -> MetalArtifacts:
    paths = _output_paths(output)
    return MetalArtifacts(
        files={
            "large_raw_pdb": paths["large_raw_pdb"],
            "large_opt_pdb": paths["large_opt_pdb"],
            "site_pdb": paths["site_pdb"],
            "gaussian_input": gaussian_input,
            "mol2": paths["mol2"],
            "site_mol2": paths["site_mol2"],
            "frcmod": paths["frcmod"],
            "tleap_pdb": paths["tleap_pdb"],
            "tleap_input": paths["tleap_input"],
        },
        resp_files=dict(resp_files or {}),
    )


# ---------------------------------------------------------------------------
# Shared Atom Views
# ---------------------------------------------------------------------------


def _atom_context(
    model: dict,
) -> tuple[
    list[tuple[dict, dict]],
    dict[tuple[str, int, str], list[tuple[int, dict]]],
    dict[tuple[tuple[str, int, str], str], int],
    dict[int, dict],
]:
    flattened = flatten_model_atoms(model)
    entries_by_residue: dict[tuple[str, int, str], list[tuple[int, dict]]] = defaultdict(list)
    index_by_residue_atom: dict[tuple[tuple[str, int, str], str], int] = {}
    residue_by_index: dict[int, dict] = {}
    for atom_index, (residue, atom) in enumerate(flattened, start=1):
        residue_key = get_resid_key(residue)
        entries_by_residue[residue_key].append((atom_index, atom))
        index_by_residue_atom[(residue_key, atom["name"])] = atom_index
        residue_by_index[atom_index] = residue
    return flattened, dict(entries_by_residue), index_by_residue_atom, residue_by_index


# ---------------------------------------------------------------------------
# Typing and Rename
# ---------------------------------------------------------------------------


def _allocate_local_type(
    used_types: set[str],
    prefixes: tuple[str, ...],
    start_index: int,
) -> tuple[str, int]:
    index = start_index
    total = len(prefixes) * len(_TYPE_DIGITS)
    while index < total:
        prefix = prefixes[index // len(_TYPE_DIGITS)]
        digit = _TYPE_DIGITS[index % len(_TYPE_DIGITS)]
        candidate = f"{prefix}{digit}"
        index += 1
        if candidate not in used_types:
            used_types.add(candidate)
            return candidate, index
    raise ValueError("MetalAA local atom-type export ran out of unique two-character atom types.")


def _build_residue_adjacency(residue: dict) -> tuple[dict[str, dict], dict[str, set[str]]]:
    atoms = sorted(residue["atoms"], key=lambda atom: atom["serial"])
    name_to_atom = {atom["name"]: atom for atom in atoms}
    adjacency = {atom["name"]: set() for atom in atoms}
    for left, right in infer_bond_pairs({"residues": [residue]}):
        left_name = atoms[left - 1]["name"]
        right_name = atoms[right - 1]["name"]
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)
    return name_to_atom, adjacency


def _protein_reference_category(residue: dict) -> str | None:
    if residue.get("_prev_peptide_key") is None and residue.get("_next_peptide_key") is not None:
        return "nterm"
    if residue.get("_next_peptide_key") is None and residue.get("_prev_peptide_key") is not None:
        return "cterm"
    if residue.get("_prev_peptide_key") is not None and residue.get("_next_peptide_key") is not None:
        return "internal"
    return "internal"


def _histidine_reference_names(residue: dict, adjacency: dict[str, set[str]], name_to_atom: dict[str, dict]) -> list[str]:
    protonated_nd1 = any(name_to_atom[neighbor]["element"] == "H" for neighbor in adjacency.get("ND1", ()))
    protonated_ne2 = any(name_to_atom[neighbor]["element"] == "H" for neighbor in adjacency.get("NE2", ()))
    ordered: list[str] = []
    if protonated_nd1 and protonated_ne2:
        ordered.append("HIP")
    elif protonated_nd1:
        ordered.append("HID")
    elif protonated_ne2:
        ordered.append("HIE")
    for candidate in ("HID", "HIE", "HIP"):
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _reference_residue_names(residue: dict, adjacency: dict[str, set[str]], name_to_atom: dict[str, dict]) -> list[str]:
    resname = residue["resname"].upper()
    if resname == "HIS":
        return _histidine_reference_names(residue, adjacency, name_to_atom)
    return [resname]


def _hydrogen_reference_candidates(heavy_name: str) -> list[str]:
    if heavy_name == "N":
        return ["H", "HN", "H1", "H2", "H3"]
    if heavy_name == "CA":
        return ["HA", "HA1", "HA2", "HA3"]
    if heavy_name in {"C", "O", "OXT"}:
        return []

    suffix = heavy_name[1:]
    stripped_suffix = suffix.rstrip("123456789")
    candidates = [f"H{suffix}", f"H{suffix}1", f"H{suffix}2", f"H{suffix}3"]
    if stripped_suffix and stripped_suffix != suffix:
        candidates.extend([f"H{stripped_suffix}", f"H{stripped_suffix}1", f"H{stripped_suffix}2", f"H{stripped_suffix}3"])
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _lookup_reference_entry(
    residue: dict,
    atom: dict,
    *,
    name_to_atom: dict[str, dict],
    adjacency: dict[str, set[str]],
    prom: str,
) -> tuple[str, float] | None:
    resname = residue["resname"].upper()
    if resname == "ACE":
        return lookup_standard_residue_entry(resname="ACE", atom_name=atom["name"], category="nterm", prom=prom)
    if resname == "NME":
        return lookup_standard_residue_entry(resname="NME", atom_name=atom["name"], category="cterm", prom=prom)
    if resname == "GLY" and residue.get("kind") != "protein":
        return lookup_standard_residue_entry(resname="GLY", atom_name=atom["name"], category="internal", prom=prom)

    if residue.get("kind") != "protein":
        return lookup_standard_atom_entry(residue, atom, prom=prom)

    category = _protein_reference_category(residue)
    if category is None:
        return None

    residue_names = _reference_residue_names(residue, adjacency, name_to_atom)
    for residue_name in residue_names:
        entry = lookup_standard_residue_entry(
            resname=residue_name,
            atom_name=atom["name"],
            category=category,
            prom=prom,
        )
        if entry is not None:
            return entry
        if atom["element"] != "H":
            continue
        heavy_neighbors = [
            neighbor
            for neighbor in adjacency.get(atom["name"], ())
            if name_to_atom[neighbor]["element"] != "H"
        ]
        if not heavy_neighbors:
            continue
        ordered_neighbors = sorted(
            heavy_neighbors,
            key=lambda neighbor: float(np.linalg.norm(atom["xyz"] - name_to_atom[neighbor]["xyz"])),
        )
        for neighbor in ordered_neighbors:
            for candidate_name in _hydrogen_reference_candidates(neighbor):
                entry = lookup_standard_residue_entry(
                    resname=residue_name,
                    atom_name=candidate_name,
                    category=category,
                    prom=prom,
                )
                if entry is not None:
                    return entry
    return None


def _resolve_old_type(residue: dict, atom: dict, *, watm: str, prom: str) -> str:
    explicit = str(atom.get("atom_type", "")).strip()
    if explicit:
        return explicit

    name_to_atom, adjacency = _build_residue_adjacency(residue)
    entry = _lookup_reference_entry(residue, atom, name_to_atom=name_to_atom, adjacency=adjacency, prom=prom)
    if entry is not None and entry[0].strip():
        return entry[0].strip()

    if residue.get("kind") == "water":
        water_type = _WATER_REFERENCE_TYPES.get(atom["element"].upper())
        if water_type is not None:
            return water_type
    if residue.get("kind") == "ion":
        return atom["element"].upper()

    raise ValueError(
        f"Could not determine a valid Amber/GAFF atom type for {get_resid_label(residue)}:{atom['name']}."
    )


def _canonical_export_resname(residue: dict) -> str:
    if residue.get("kind") == "ion":
        return residue["atoms"][0]["element"].upper()
    if residue.get("kind") == "protein" and residue["resname"].upper() == "HIS":
        name_to_atom, adjacency = _build_residue_adjacency(residue)
        return _histidine_reference_names(residue, adjacency, name_to_atom)[0]
    return residue["resname"].upper()


def _deployment_residue_code(residue: dict, counter: int) -> str:
    base = _canonical_export_resname(residue)
    if residue.get("kind") == "ion":
        return f"{base[:2]}{counter}"
    if len(base) >= 3:
        return f"{base[0]}{base[2]}{counter}"
    return f"{base}{counter}"


def _deployment_residue_names(
    site_model: dict,
    selected_names_by_key: dict[tuple[str, int, str], set[str]],
) -> dict[tuple[str, int, str], str]:
    counters: dict[str, int] = {}
    names: dict[tuple[str, int, str], str] = {}
    for residue in sorted(site_model["residues"], key=residue_sort_key):
        residue_key = get_resid_key(residue)
        if residue_key not in selected_names_by_key:
            continue
        base = _canonical_export_resname(residue)
        counters[base] = counters.get(base, 0) + 1
        names[residue_key] = _deployment_residue_code(residue, counters[base])
    return names


def _build_site_typing(
    site_model: dict,
    *,
    watm: str,
    ionm: str,
    prom: str = "ff14SB",
    cofactor_frcmods: list[str] | None = None,
    cofactor_frcmod_by_residue: dict[tuple[str, int, str], str] | None = None,
) -> MetalSiteTyping:
    flattened = flatten_model_atoms(site_model)
    donor_atoms = {key: set(names) for key, names in site_model.get("donor_atoms", {}).items()}

    atom_type_rows: list[MetalAtomTypeRow] = []
    mol2_atom_types: dict[int, str] = {}
    atom_type_overrides: dict[int, str] = {}
    old_type_by_index: dict[int, str] = {}
    renamed_atom_indices: set[int] = set()
    ion_frcmods: list[str] = []
    seen_ion_frcmods: set[str] = set()
    used_types: set[str] = set()

    selected_names_by_key: dict[tuple[str, int, str], set[str]] = {}
    for residue in site_model["residues"]:
        residue_key = get_resid_key(residue)
        if residue.get("kind") == "ion":
            selected_names_by_key[residue_key] = {atom["name"] for atom in residue["atoms"]}
            continue
        donor_names = donor_atoms.get(residue_key)
        if donor_names:
            selected_names_by_key[residue_key] = {name for name in donor_names}
    residue_names = _deployment_residue_names(site_model, selected_names_by_key)

    metal_index = 0
    site_index = 0
    metal_formal_charge: int | None = None
    metal_fitted_charge = 0.0
    for atom_index, (residue, atom) in enumerate(flattened, start=1):
        old_type = _resolve_old_type(residue, atom, watm=watm, prom=prom)
        old_type_by_index[atom_index] = old_type
        mol2_atom_types[atom_index] = old_type
        used_types.add(old_type)
        residue_key = get_resid_key(residue)
        selected_names = selected_names_by_key.get(residue_key)
        if not selected_names or atom["name"] not in selected_names:
            continue

        renamed_atom_indices.add(atom_index)
        if residue.get("kind") == "ion":
            new_type, metal_index = _allocate_local_type(used_types, ("M",), metal_index)
            frcmod_name = infer_ion_frcmod_name(watm=watm, ionm=ionm, residue=residue)
            if "formal_charge" in residue:
                metal_formal_charge = int(residue["formal_charge"])
            metal_fitted_charge += float(atom.get("charge", 0.0))
            if frcmod_name not in seen_ion_frcmods:
                seen_ion_frcmods.add(frcmod_name)
                ion_frcmods.append(frcmod_name)
        else:
            new_type, site_index = _allocate_local_type(used_types, _LOCAL_TYPE_LETTERS, site_index)

        mol2_atom_types[atom_index] = new_type
        atom_type_overrides[atom_index] = new_type
        atom_type_rows.append(
            MetalAtomTypeRow(
                residue=get_resid_label(residue),
                atom_name=atom["name"],
                element=atom["element"],
                old_type=old_type,
                new_type=new_type,
                charge=float(atom.get("charge", 0.0)),
            )
        )

    return MetalSiteTyping(
        atom_type_rows=atom_type_rows,
        mol2_atom_types=mol2_atom_types,
        atom_type_overrides=atom_type_overrides,
        old_type_by_index=old_type_by_index,
        renamed_atom_indices=renamed_atom_indices,
        ion_frcmods=ion_frcmods,
        residue_names=residue_names,
        metal_formal_charge=metal_formal_charge,
        metal_fitted_charge=metal_fitted_charge if metal_index else None,
        watm=watm,
        ionm=ionm,
        prom=prom,
        cofactor_frcmods=list(cofactor_frcmods or []),
        cofactor_frcmod_by_residue=dict(cofactor_frcmod_by_residue or {}),
    )


# ---------------------------------------------------------------------------
# Bond Graph
# ---------------------------------------------------------------------------


def _metal_atom_indices(flattened: list[tuple[dict, dict]]) -> set[int]:
    return {
        atom_index
        for atom_index, (residue, _atom) in enumerate(flattened, start=1)
        if residue.get("kind") == "ion"
    }


def _single_atom_residue_atom(residue: dict, *, label: str) -> dict:
    if len(residue.get("atoms", [])) != 1:
        raise ValueError(f"{label} must be a single-atom residue.")
    return residue["atoms"][0]


def _donor_metal_pairs(
    site_model: dict,
    index_by_residue_atom: dict[tuple[tuple[str, int, str], str], int],
) -> set[tuple[int, int]]:
    residues = _residue_by_key(site_model)
    target_key = site_model.get("target_key")
    metal_residue = residues.get(target_key)
    if metal_residue is None:
        return set()
    metal_atom = _single_atom_residue_atom(metal_residue, label="MetalAA target metal residue")
    metal_index = index_by_residue_atom[(target_key, metal_atom["name"])]
    pairs: set[tuple[int, int]] = set()
    for donor_key, atom_names in site_model.get("donor_atoms", {}).items():
        for atom_name in atom_names:
            donor_index = index_by_residue_atom.get((donor_key, atom_name))
            if donor_index is None:
                continue
            pairs.add(tuple(sorted((donor_index, metal_index))))
    return pairs


def _bond_pairs_from_names(
    name_pairs,
    index_by_name: dict[str, int],
    *,
    label: str,
) -> list[tuple[int, int]]:
    if not name_pairs:
        return []
    pairs: set[tuple[int, int]] = set()
    for left_name, right_name in name_pairs:
        try:
            left_index = index_by_name[left_name]
            right_index = index_by_name[right_name]
        except KeyError as exc:
            raise ValueError(f"{label} cfmol2 bond references unknown atom {exc.args[0]!r}.") from exc
        pairs.add(tuple(sorted((left_index, right_index))))
    return sorted(pairs)


def _export_bond_pairs(site_model: dict) -> list[tuple[int, int]]:
    flattened, entries_by_residue, index_by_residue_atom, _ = _atom_context(site_model)
    metal_indices = _metal_atom_indices(flattened)
    donor_pairs = _donor_metal_pairs(site_model, index_by_residue_atom)
    pairs: set[tuple[int, int]] = set()
    typed_cofactor_atom_indices: set[int] = set()
    for residue in site_model["residues"]:
        if residue.get("_cfmol2_bond_name_pairs"):
            residue_key = get_resid_key(residue)
            index_by_name = {atom["name"]: atom_index for atom_index, atom in entries_by_residue[residue_key]}
            typed_cofactor_atom_indices.update(atom_index for atom_index, _atom in entries_by_residue[residue_key])
            pairs.update(
                _bond_pairs_from_names(
                    residue["_cfmol2_bond_name_pairs"],
                    index_by_name,
                    label=get_resid_label(residue),
                )
            )

    for left, right in infer_bond_pairs(site_model, source_structure=site_model):
        pair = tuple(sorted((left, right)))
        if typed_cofactor_atom_indices.intersection(pair):
            continue
        if metal_indices.intersection(pair):
            if pair in donor_pairs:
                pairs.add(pair)
            continue
        pairs.add(pair)
    pairs.update(donor_pairs)
    return sorted(pairs)


def _adjacency_from_pairs(pairs: list[tuple[int, int]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _enumerate_angles_from_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    angles: list[tuple[int, int, int]] = []
    adjacency = _adjacency_from_pairs(pairs)
    for center, neighbors in sorted(adjacency.items()):
        for left, right in combinations(sorted(neighbors), 2):
            angles.append((left, center, right))
    return angles


def _enumerate_dihedrals_from_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    adjacency = _adjacency_from_pairs(pairs)
    seen: set[tuple[int, int, int, int]] = set()
    dihedrals: list[tuple[int, int, int, int]] = []
    for center_left in sorted(adjacency):
        for center_right in sorted(adjacency[center_left]):
            if center_left >= center_right:
                continue
            for outer_left in sorted(adjacency[center_left] - {center_right}):
                for outer_right in sorted(adjacency[center_right] - {center_left}):
                    if len({outer_left, center_left, center_right, outer_right}) != 4:
                        continue
                    path = (outer_left, center_left, center_right, outer_right)
                    reverse = tuple(reversed(path))
                    canonical = path if path <= reverse else reverse
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    dihedrals.append(path)
    return dihedrals


def _enumerate_impropers_from_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    adjacency = _adjacency_from_pairs(pairs)
    impropers: list[tuple[int, int, int, int]] = []
    for center, neighbors in sorted(adjacency.items()):
        if len(neighbors) < 3:
            continue
        for trio in combinations(sorted(neighbors), 3):
            impropers.append((trio[0], trio[1], center, trio[2]))
    return impropers


# ---------------------------------------------------------------------------
# FRCMOD Merge and Remap
# ---------------------------------------------------------------------------


def _has_renamed(atom_indices: tuple[int, ...], typing: MetalSiteTyping) -> bool:
    return any(atom_index in typing.renamed_atom_indices for atom_index in atom_indices)


def _resolved_types(atom_indices: tuple[int, ...], typing: MetalSiteTyping) -> tuple[str, ...]:
    return tuple(typing.atom_type_overrides.get(atom_index, typing.old_type_by_index[atom_index]) for atom_index in atom_indices)


def _old_types(atom_indices: tuple[int, ...], typing: MetalSiteTyping) -> tuple[str, ...]:
    return tuple(typing.old_type_by_index[atom_index] for atom_index in atom_indices)


def _is_metal_related(atom_indices: tuple[int, ...], metal_indices: set[int]) -> bool:
    return bool(metal_indices.intersection(atom_indices))


def _has_renamed_nonprotein_atom(
    atom_indices: tuple[int, ...],
    flattened: list[tuple[dict, dict]],
    typing: MetalSiteTyping,
) -> bool:
    return any(
        atom_index in typing.renamed_atom_indices
        and flattened[atom_index - 1][0].get("kind") in {"ligand", "cofactor"}
        for atom_index in atom_indices
    )


def _improper_reference_source(
    atom_indices: tuple[int, int, int, int],
    flattened: list[tuple[dict, dict]],
) -> str | None:
    residues = [flattened[atom_index - 1][0] for atom_index in atom_indices]
    if len({get_resid_key(residue) for residue in residues}) != 1:
        return None
    kind = residues[0].get("kind")
    if kind == "protein":
        return "protein"
    if kind in {"cofactor", "ligand"}:
        return "cofactor"
    return None


def _bond_term_lookup(bond_terms) -> dict[tuple[int, int], tuple[float, float]]:
    lookup: dict[tuple[int, int], tuple[float, float]] = {}
    for bond in bond_terms:
        if bond.kBond is None or bond.rEq is None:
            continue
        lookup[tuple(sorted(bond.atoms))] = (float(bond.kBond), float(bond.rEq))
    return lookup


def _angle_term_lookup(angle_terms) -> dict[tuple[int, int, int], tuple[float, float]]:
    lookup: dict[tuple[int, int, int], tuple[float, float]] = {}
    for angle in angle_terms:
        if angle.kTheta is None or angle.thetaEq is None:
            continue
        theta_deg = float(np.degrees(angle.thetaEq))
        lookup[angle.atoms] = (float(angle.kTheta), theta_deg)
        lookup[(angle.atoms[2], angle.atoms[1], angle.atoms[0])] = (float(angle.kTheta), theta_deg)
    return lookup


def _format_bond_line(atom_types: tuple[str, str], params: tuple[float, float]) -> str:
    return f"{'-'.join(atom_types):<11s} {params[0]:10.4f} {params[1]:10.4f}\n"


def _format_mass_line(atom_type: str, mass: float) -> str:
    return f"{atom_type:<2s} {mass:10.4f}\n"


def _format_angle_line(atom_types: tuple[str, str, str], params: tuple[float, float]) -> str:
    return f"{'-'.join(atom_types):<11s} {params[0]:10.4f} {params[1]:10.4f}\n"


def _format_dihedral_line(atom_types: tuple[str, str, str, str], term: TorsionParameter) -> str:
    return (
        f"{'-'.join(atom_types):<11s} "
        f"{1:4d} {term.amplitude:10.4f} {term.phase_deg:10.4f} {term.periodicity:10.4f}\n"
    )


def _format_improper_line(atom_types: tuple[str, str, str, str], term: TorsionParameter) -> str:
    return (
        f"{'-'.join(atom_types):<11s} "
        f"{term.amplitude:10.4f} {term.phase_deg:10.4f} {term.periodicity:10.4f}\n"
    )


def _format_nonbond_line(atom_type: str, params: tuple[float, float]) -> str:
    return f"{atom_type:<2s} {params[0]:10.4f} {params[1]:12.8f}\n"


def _canonical_torsion_key(atom_types: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    reverse = tuple(reversed(atom_types))
    return atom_types if atom_types <= reverse else reverse


def _has_wildcard(atom_types: tuple[str, ...]) -> bool:
    return any(atom_type == "X" for atom_type in atom_types)


def _remap_dihedral_template(
    template: tuple[str, str, str, str],
    *,
    old_types: tuple[str, str, str, str],
    resolved_types: tuple[str, str, str, str],
    reversed_match: bool,
) -> tuple[str, str, str, str]:
    aligned_old = tuple(reversed(old_types)) if reversed_match else old_types
    aligned_resolved = tuple(reversed(resolved_types)) if reversed_match else resolved_types
    remapped: list[str] = []
    for template_type, old_type, resolved_type in zip(template, aligned_old, aligned_resolved):
        if template_type == "X":
            remapped.append("X")
        elif template_type == old_type:
            remapped.append(resolved_type)
        else:
            remapped.append(template_type)
    return remapped[0], remapped[1], remapped[2], remapped[3]


def _zero_torsion() -> TorsionParameter:
    return TorsionParameter(amplitude=0.0, phase_deg=0.0, periodicity=3.0)


def _parameter_missing(term_name: str, atom_types: tuple[str, ...]) -> ValueError:
    joined = "-".join(atom_types)
    return ValueError(f"Could not find inherited Amber {term_name} parameters for renamed MetalAA term {joined}.")


def _ion_lookup_by_index(
    flattened: list[tuple[dict, dict]],
    typing: MetalSiteTyping,
) -> dict[int, tuple[str, float, tuple[float, float]]]:
    lookup: dict[int, tuple[str, float, tuple[float, float]]] = {}
    for atom_index, (residue, _atom) in enumerate(flattened, start=1):
        if residue.get("kind") != "ion" or atom_index not in typing.renamed_atom_indices:
            continue
        frcmod_name, _amber_type, mass, nonbond = lookup_ion_lj_from_frcmod(
            watm=typing.watm,
            ionm=typing.ionm,
            residue=residue,
        )
        lookup[atom_index] = (frcmod_name, mass, nonbond)
    return lookup


def _copy_parameter_db(db: AmberParameterDB) -> AmberParameterDB:
    return AmberParameterDB(
        mass=dict(db.mass),
        bond=dict(db.bond),
        angle=dict(db.angle),
        dihedral={key: list(terms) for key, terms in db.dihedral.items()},
        improper={key: list(terms) for key, terms in db.improper.items()},
        nonbond=dict(db.nonbond),
    )


def _merge_parameter_db(into: AmberParameterDB, other: AmberParameterDB) -> AmberParameterDB:
    into.mass.update(other.mass)
    into.bond.update(other.bond)
    into.angle.update(other.angle)
    for key, terms in other.dihedral.items():
        into.dihedral[key] = list(terms)
    for key, terms in other.improper.items():
        into.improper[key] = list(terms)
    into.nonbond.update(other.nonbond)
    return into


def _load_cofactor_parameters(cofactor_frcmods: list[str]) -> AmberParameterDB:
    params = AmberParameterDB()
    for frcmod in cofactor_frcmods:
        _merge_parameter_db(params, parse_amber_frcmod(Path(frcmod)))
    return params


def _load_cofactor_parameters_by_residue(
    cofactor_frcmod_by_residue: dict[tuple[str, int, str], str],
) -> dict[tuple[str, int, str], AmberParameterDB]:
    return {
        residue_key: parse_amber_frcmod(Path(frcmod))
        for residue_key, frcmod in cofactor_frcmod_by_residue.items()
    }


def _default_cofactor_frcmod_by_residue(site_model: dict, cofactor_frcmods: list[str]) -> dict[tuple[str, int, str], str]:
    if len(cofactor_frcmods) != 1:
        return {}
    frcmod = cofactor_frcmods[0]
    return {
        get_resid_key(residue): frcmod
        for residue in site_model["residues"]
        if residue.get("kind") in {"cofactor", "ligand"}
    }


def _same_residue_reference_params(
    atom_indices: tuple[int, ...],
    flattened: list[tuple[dict, dict]],
    *,
    protein_params: AmberParameterDB,
    cofactor_params_by_residue: dict[tuple[str, int, str], AmberParameterDB],
) -> AmberParameterDB | None:
    residues = [flattened[atom_index - 1][0] for atom_index in atom_indices]
    residue_keys = {get_resid_key(residue) for residue in residues}
    if len(residue_keys) != 1:
        return None
    residue = residues[0]
    kind = residue.get("kind")
    if kind == "protein":
        return protein_params
    if kind in {"cofactor", "ligand"}:
        return cofactor_params_by_residue.get(get_resid_key(residue))
    return None


def _match_dihedral_with_wildcard_preference(
    atom_types: tuple[str, str, str, str],
    reference_params: AmberParameterDB,
) -> tuple[tuple[str, str, str, str], list[TorsionParameter], bool] | None:
    wildcard_templates = {
        template: terms
        for template, terms in reference_params.dihedral.items()
        if _has_wildcard(template)
    }
    wildcard_match = match_dihedral_template(atom_types, wildcard_templates)
    return wildcard_match or match_dihedral_template(atom_types, reference_params.dihedral)


def _write_frcmod(path: str, site_model: dict, bond_terms, angle_terms, typing: MetalSiteTyping) -> None:
    flattened = flatten_model_atoms(site_model)
    element_by_index = {
        atom_index: atom["element"]
        for atom_index, (_, atom) in enumerate(flattened, start=1)
    }
    cofactor_params = _load_cofactor_parameters(typing.cofactor_frcmods)
    cofactor_params_by_residue = _load_cofactor_parameters_by_residue(
        typing.cofactor_frcmod_by_residue
        or _default_cofactor_frcmod_by_residue(site_model, typing.cofactor_frcmods)
    )
    system_reference_params = _copy_parameter_db(load_parameters(typing.prom, typing.watm))
    protein_params = load_parameters(typing.prom)
    _merge_parameter_db(system_reference_params, cofactor_params)
    metal_indices = _metal_atom_indices(flattened)
    ion_lookup = _ion_lookup_by_index(flattened, typing)
    export_pairs = _export_bond_pairs(site_model)
    bond_lookup = _bond_term_lookup(bond_terms)
    angle_lookup = _angle_term_lookup(angle_terms)

    # MASS / NONBON for renamed and ion types.
    mass_lines: list[str] = []
    seen_mass_types: set[str] = set()
    for atom_index in sorted(typing.atom_type_overrides):
        atom_type = typing.atom_type_overrides[atom_index]
        if atom_type in seen_mass_types:
            continue
        seen_mass_types.add(atom_type)
        if atom_index in ion_lookup:
            mass = ion_lookup[atom_index][1]
        else:
            old_type = typing.old_type_by_index[atom_index]
            reference_params = _same_residue_reference_params(
                (atom_index,),
                flattened,
                protein_params=protein_params,
                cofactor_params_by_residue=cofactor_params_by_residue,
            )
            mass = (reference_params.mass.get(old_type) if reference_params is not None else None) or system_reference_params.mass.get(old_type)
            if mass is None:
                raise _parameter_missing("MASS", (old_type,))
        if mass is None:
            mass = ATOMIC_MASSES.get(element_by_index[atom_index].upper(), 0.0)
        mass_lines.append(_format_mass_line(atom_type, mass))

    # BOND / ANGLE: fitted metal terms first, inherited same-residue terms otherwise.
    bond_lines: list[str] = []
    seen_bonds: set[tuple[str, str]] = set()
    for pair in export_pairs:
        atoms = tuple(pair)
        if not _has_renamed(atoms, typing):
            continue
        atom_types = _resolved_types(atoms, typing)
        key = canonical_pair(atom_types)
        if key in seen_bonds:
            continue
        if _is_metal_related(atoms, metal_indices):
            fitted = bond_lookup.get(tuple(sorted(atoms)))
            if fitted is None:
                raise ValueError(
                    "Missing fitted MetalAA BOND parameters for selected metal term "
                    f"{'-'.join(atom_types)}."
                )
            bond_lines.append(_format_bond_line(atom_types, fitted))
        else:
            old_key = canonical_pair(_old_types(atoms, typing))
            reference_params = _same_residue_reference_params(
                atoms,
                flattened,
                protein_params=protein_params,
                cofactor_params_by_residue=cofactor_params_by_residue,
            )
            inherited = reference_params.bond.get(old_key) if reference_params is not None else None
            if inherited is None:
                continue
            bond_lines.append(_format_bond_line(atom_types, inherited))
        seen_bonds.add(key)

    angle_lines: list[str] = []
    seen_angles: set[tuple[str, str, str]] = set()
    for angle in _enumerate_angles_from_pairs(export_pairs):
        if not _has_renamed(angle, typing):
            continue
        atom_types = _resolved_types(angle, typing)
        key = canonical_angle(atom_types)
        if key in seen_angles:
            continue
        if _is_metal_related(angle, metal_indices):
            fitted = angle_lookup.get(angle)
            if fitted is None:
                raise ValueError(
                    "Missing fitted MetalAA ANGLE parameters for selected metal term "
                    f"{'-'.join(atom_types)}."
                )
            angle_lines.append(_format_angle_line(atom_types, fitted))
        else:
            old_key = canonical_angle(_old_types(angle, typing))
            reference_params = _same_residue_reference_params(
                angle,
                flattened,
                protein_params=protein_params,
                cofactor_params_by_residue=cofactor_params_by_residue,
            )
            inherited = reference_params.angle.get(old_key) if reference_params is not None else None
            if inherited is None:
                continue
            angle_lines.append(_format_angle_line(atom_types, inherited))
        seen_angles.add(key)

    # DIHE: metal zero torsion or wildcard-aware inherited torsion.
    dihedral_lines: list[str] = []
    seen_dihedrals: set[tuple[str, str, str, str]] = set()
    for dihedral in _enumerate_dihedrals_from_pairs(export_pairs):
        if not _has_renamed(dihedral, typing):
            continue
        old_types = _old_types(dihedral, typing)
        resolved_types = _resolved_types(dihedral, typing)
        if _is_metal_related(dihedral, metal_indices):
            atom_types = resolved_types
            terms = [_zero_torsion()]
        else:
            reference_params = _same_residue_reference_params(
                dihedral,
                flattened,
                protein_params=protein_params,
                cofactor_params_by_residue=cofactor_params_by_residue,
            )
            if reference_params is None:
                continue
            matched = _match_dihedral_with_wildcard_preference(old_types, reference_params)
            if matched is not None:
                template, terms, reversed_match = matched
            else:
                if _has_renamed_nonprotein_atom(dihedral, flattened, typing):
                    raise _parameter_missing("DIHE", resolved_types)
                atom_types = resolved_types
                terms = [_zero_torsion()]
                template = None
                reversed_match = False
            if template is not None:
                atom_types = _remap_dihedral_template(
                    template,
                    old_types=old_types,
                    resolved_types=resolved_types,
                    reversed_match=reversed_match,
                )
            if not terms and _has_renamed_nonprotein_atom(dihedral, flattened, typing):
                raise _parameter_missing("DIHE", atom_types)
            if not terms:
                terms = [_zero_torsion()]
        key = _canonical_torsion_key(atom_types)
        if key in seen_dihedrals:
            continue
        for term in terms:
            dihedral_lines.append(_format_dihedral_line(atom_types, term))
        seen_dihedrals.add(key)

    # IMPROPER: inherited reference only, no fallback creation.
    improper_lines: list[str] = []
    seen_impropers: set[tuple[str, str, str, str]] = set()
    for improper in _enumerate_impropers_from_pairs(export_pairs):
        if not _has_renamed(improper, typing) or _is_metal_related(improper, metal_indices):
            continue
        atom_types = _resolved_types(improper, typing)
        if atom_types in seen_impropers:
            continue
        source = _improper_reference_source(improper, flattened)
        if source is None:
            continue
        reference_params = _same_residue_reference_params(
            improper,
            flattened,
            protein_params=protein_params,
            cofactor_params_by_residue=cofactor_params_by_residue,
        )
        if reference_params is None:
            continue
        reference_improper = reference_params.improper
        terms = match_improper(_old_types(improper, typing), reference_improper)
        if not terms:
            continue
        for term in terms:
            improper_lines.append(_format_improper_line(atom_types, term))
        seen_impropers.add(atom_types)

    nonbond_lines: list[str] = []
    seen_nonbond: set[str] = set()
    for atom_index in sorted(typing.atom_type_overrides):
        atom_type = typing.atom_type_overrides[atom_index]
        if atom_type in seen_nonbond:
            continue
        if atom_index in ion_lookup:
            nonbond = ion_lookup[atom_index][2]
        else:
            old_type = typing.old_type_by_index[atom_index]
            reference_params = _same_residue_reference_params(
                (atom_index,),
                flattened,
                protein_params=protein_params,
                cofactor_params_by_residue=cofactor_params_by_residue,
            )
            nonbond = (
                reference_params.nonbond.get(old_type) if reference_params is not None else None
            ) or system_reference_params.nonbond.get(old_type)
            if nonbond is None:
                raise _parameter_missing("NONBON", (old_type,))
        nonbond_lines.append(_format_nonbond_line(atom_type, nonbond))
        seen_nonbond.add(atom_type)

    # Base cofactor frcmod terms are included after renamed local terms are reserved.
    base_mass_lines = [
        _format_mass_line(atom_type, mass)
        for atom_type, mass in sorted(cofactor_params.mass.items())
        if atom_type not in seen_mass_types
    ]
    base_bond_lines = [
        _format_bond_line(atom_types, params)
        for atom_types, params in sorted(cofactor_params.bond.items())
        if atom_types not in seen_bonds
    ]
    base_angle_lines = [
        _format_angle_line(atom_types, params)
        for atom_types, params in sorted(cofactor_params.angle.items())
        if atom_types not in seen_angles
    ]
    base_dihedral_lines: list[str] = []
    for atom_types, terms in sorted(cofactor_params.dihedral.items()):
        if _canonical_torsion_key(atom_types) in seen_dihedrals:
            continue
        for term in terms:
            base_dihedral_lines.append(_format_dihedral_line(atom_types, term))
    base_improper_lines: list[str] = []
    for atom_types, terms in sorted(cofactor_params.improper.items()):
        if atom_types in seen_impropers:
            continue
        for term in terms:
            base_improper_lines.append(_format_improper_line(atom_types, term))
    base_nonbond_lines = [
        _format_nonbond_line(atom_type, params)
        for atom_type, params in sorted(cofactor_params.nonbond.items())
        if atom_type not in seen_nonbond
    ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("REMARK MAPLE MetalAA generated metal-site frcmod\n\n")
        handle.write("MASS\n")
        handle.writelines(base_mass_lines)
        handle.writelines(mass_lines)
        handle.write("\nBOND\n")
        handle.writelines(base_bond_lines)
        handle.writelines(bond_lines)
        handle.write("\nANGLE\n")
        handle.writelines(base_angle_lines)
        handle.writelines(angle_lines)
        handle.write("\nDIHE\n")
        handle.writelines(base_dihedral_lines)
        handle.writelines(dihedral_lines)
        handle.write("\nIMPROPER\n")
        handle.writelines(base_improper_lines)
        handle.writelines(improper_lines)
        handle.write("\nNONBON\n")
        handle.writelines(base_nonbond_lines)
        handle.writelines(nonbond_lines)


# ---------------------------------------------------------------------------
# Deployment Files
# ---------------------------------------------------------------------------


def _export_atom_name(residue: dict, atom: dict) -> str:
    return atom["name"]


def _copy_residue_for_deployment(residue: dict, resname: str) -> dict:
    copied = copy_residue(residue, resname=resname)
    for atom in copied["atoms"]:
        atom["name"] = _export_atom_name(copied, atom)
    return copied


def _write_deployment_pdb(path: str, structure: dict, typing: MetalSiteTyping) -> dict[tuple[str, int, str], int]:
    model = copy_structure_subset(structure, structure["residues"])
    resid_by_key: dict[tuple[str, int, str], int] = {}
    for new_resseq, residue in enumerate(sorted(model["residues"], key=residue_sort_key), start=1):
        residue_key = get_resid_key(residue)
        resid_by_key[residue_key] = new_resseq
        residue["resseq"] = new_resseq
        new_name = typing.residue_names.get(residue_key)
        if new_name is None:
            continue
        residue["resname"] = new_name
        for atom in residue["atoms"]:
            atom["name"] = _export_atom_name(residue, atom)
    rebuild_model_index(model)
    write_model_pdb(path, model)
    return resid_by_key


def _write_residue_mol2_files(
    artifacts: MetalArtifacts,
    *,
    site_model: dict,
    typing: MetalSiteTyping,
) -> dict[str, str]:
    output_dir = os.path.dirname(artifacts.files["frcmod"])
    _, site_entries_by_residue, _, _ = _atom_context(site_model)
    mol2_files: dict[str, str] = {}
    for residue in sorted(site_model["residues"], key=residue_sort_key):
        residue_key = get_resid_key(residue)
        new_name = typing.residue_names.get(residue_key)
        if new_name is None:
            continue
        export_residue = _copy_residue_for_deployment(residue, new_name)
        local_model = {
            "path": site_model.get("path"),
            "residues": [export_residue],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
        rebuild_model_index(local_model)
        atom_type_overrides: dict[int, str] = {}
        site_entries = site_entries_by_residue[residue_key]
        export_atoms = local_model["residues"][0]["atoms"]
        if len(site_entries) != len(export_atoms):
            raise ValueError(
                f"MetalAA mol2 export atom count mismatch for residue {residue_key}: "
                f"site_model has {len(site_entries)} atoms, export residue has {len(export_atoms)}."
            )
        site_names = [atom["name"] for _global_index, atom in site_entries]
        export_names = [atom["name"] for atom in export_atoms]
        if site_names != export_names:
            raise ValueError(
                f"MetalAA mol2 export atom order mismatch for residue {residue_key}: "
                f"site_model atoms {site_names}, export atoms {export_names}."
            )
        for local_index, ((global_index, _site_atom), _export_atom) in enumerate(
            zip(site_entries, export_atoms, strict=True),
            start=1,
        ):
            atom_type_overrides[local_index] = typing.mol2_atom_types[global_index]
        if export_residue.get("_cfmol2_bond_name_pairs"):
            index_by_name = {atom["name"]: atom_index for atom_index, atom in enumerate(export_atoms, start=1)}
            bond_pairs = _bond_pairs_from_names(
                export_residue["_cfmol2_bond_name_pairs"],
                index_by_name,
                label=get_resid_label(residue),
            )
        else:
            bond_pairs = infer_bond_pairs(local_model, source_structure=site_model)
        path = os.path.join(output_dir, f"{new_name}.mol2")
        write_resp_mol2(path, local_model, bond_pairs, atom_type_overrides=atom_type_overrides, prom=typing.prom)
        mol2_files[new_name] = path
    artifacts.mol2_files.clear()
    artifacts.mol2_files.update(mol2_files)
    typing.mol2_files.clear()
    typing.mol2_files.update(mol2_files)
    return mol2_files


def _residue_by_key(model: dict) -> dict[tuple[str, int, str], dict]:
    return {get_resid_key(residue): residue for residue in model["residues"]}


def _tleap_ref(
    residue_key: tuple[str, int, str],
    atom_name: str,
    resid_by_key: dict[tuple[str, int, str], int],
) -> str:
    try:
        resseq = resid_by_key[residue_key]
    except KeyError as exc:
        raise ValueError(f"Cannot build tleap atom reference for residue key {residue_key!r}.") from exc
    return f"mol.{resseq}.{atom_name}"


def _append_unique(lines: list[str], seen: set[str], line: str) -> None:
    if line in seen:
        return
    seen.add(line)
    lines.append(line)


def _metal_donor_bond_commands(site_model: dict, resid_by_key: dict[tuple[str, int, str], int]) -> list[str]:
    residues = _residue_by_key(site_model)
    target_key = site_model.get("target_key")
    metal_residue = residues.get(target_key)
    if metal_residue is None:
        return []
    metal_atom = _single_atom_residue_atom(metal_residue, label="MetalAA target metal residue")
    lines: list[str] = []
    seen: set[str] = set()
    for donor_key, atom_names in sorted(site_model.get("donor_atoms", {}).items()):
        donor_residue = residues.get(donor_key)
        if donor_residue is None:
            continue
        available = {atom["name"] for atom in donor_residue["atoms"]}
        for atom_name in sorted(atom_names):
            if atom_name not in available:
                continue
            line = (
                f"bond {_tleap_ref(donor_key, atom_name, resid_by_key)} "
                f"{_tleap_ref(target_key, metal_atom['name'], resid_by_key)}\n"
            )
            _append_unique(lines, seen, line)
    return lines


def _peptide_reconnect_commands(
    site_model: dict,
    typing: MetalSiteTyping,
    resid_by_key: dict[tuple[str, int, str], int],
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for residue in sorted(site_model["residues"], key=residue_sort_key):
        residue_key = get_resid_key(residue)
        if residue_key not in typing.residue_names or residue.get("kind") != "protein":
            continue
        prev_key = residue.get("_prev_peptide_key")
        next_key = residue.get("_next_peptide_key")
        if prev_key is not None:
            line = f"bond {_tleap_ref(prev_key, 'C', resid_by_key)} {_tleap_ref(residue_key, 'N', resid_by_key)}\n"
            _append_unique(lines, seen, line)
        if next_key is not None:
            line = f"bond {_tleap_ref(residue_key, 'C', resid_by_key)} {_tleap_ref(next_key, 'N', resid_by_key)}\n"
            _append_unique(lines, seen, line)
    return lines


def _disulfide_bond_commands(structure: dict, resid_by_key: dict[tuple[str, int, str], int]) -> list[str]:
    sulfur_atoms: list[tuple[dict, dict]] = []
    for residue in structure["residues"]:
        if residue["resname"].upper() not in {"CYS", "CYX", "CYM"}:
            continue
        sulfur = next((atom for atom in residue["atoms"] if atom["name"] == "SG"), None)
        if sulfur is not None:
            sulfur_atoms.append((residue, sulfur))

    lines: list[str] = []
    for left_index, (left_residue, left_atom) in enumerate(sulfur_atoms):
        for right_residue, right_atom in sulfur_atoms[left_index + 1 :]:
            distance = float(np.linalg.norm(left_atom["xyz"] - right_atom["xyz"]))
            if distance > 2.35:
                continue
            lines.append(
                f"bond {_tleap_ref(get_resid_key(left_residue), 'SG', resid_by_key)} "
                f"{_tleap_ref(get_resid_key(right_residue), 'SG', resid_by_key)}\n"
            )
    return lines


def _build_tleap_lines(
    artifacts: MetalArtifacts,
    *,
    site_model: dict,
    structure: dict,
    typing: MetalSiteTyping,
    watm: str,
    resid_by_key: dict[tuple[str, int, str], int],
    prom: str = "ff14SB",
) -> list[str]:
    lines: list[str] = [
        f"source leaprc.protein.{prom}\n",
        "source leaprc.gaff2\n",
        f"source leaprc.water.{watm}\n",
    ]
    lines.append("addAtomTypes {\n")
    for row in typing.atom_type_rows:
        element = row.element[:1].upper() + row.element[1:].lower()
        lines.append(f'    {{ "{row.new_type}" "{element}" "sp3" }}\n')
    lines.append("}\n")
    for name in typing.mol2_files:
        lines.append(f"{name} = loadmol2 {os.path.basename(typing.mol2_files[name])}\n")
    for frcmod in typing.ion_frcmods:
        lines.append(f"loadamberparams {frcmod}\n")
    lines.append(f"loadamberparams {os.path.basename(artifacts.files['frcmod'])}\n")
    external_residues = [
        get_resid_label(residue)
        for residue in sorted(structure["residues"], key=residue_sort_key)
        if residue.get("kind") in {"ligand", "cofactor"} and get_resid_key(residue) not in typing.residue_names
    ]
    if external_residues:
        lines.append(f"# External residues not parameterized by MetalAA: {', '.join(external_residues)}\n")
        lines.append("# Load matching ligand/NCAA templates before loadpdb for these residues.\n")
    lines.append(f"mol = loadpdb {os.path.basename(artifacts.files['tleap_pdb'])}\n")
    lines.extend(_metal_donor_bond_commands(site_model, resid_by_key))
    lines.extend(_peptide_reconnect_commands(site_model, typing, resid_by_key))
    lines.extend(_disulfide_bond_commands(structure, resid_by_key))
    base = os.path.splitext(os.path.basename(artifacts.files["tleap_input"]))[0]
    lines.append(f"savepdb mol {base}_dry.pdb\n")
    lines.append(f"saveamberparm mol {base}_dry.prmtop {base}_dry.inpcrd\n")
    lines.append(f"solvatebox mol {'SPCBOX' if watm == 'spce' else watm.upper() + 'BOX'} 10.0\n")
    lines.append("addions mol Na+ 0\n")
    lines.append("addions mol Cl- 0\n")
    lines.append(f"savepdb mol {base}_solvated.pdb\n")
    lines.append(f"saveamberparm mol {base}_solvated.prmtop {base}_solvated.inpcrd\n")
    lines.append("quit\n")
    artifacts.tleap_lines.clear()
    artifacts.tleap_lines.extend(lines)
    typing.tleap_lines.clear()
    typing.tleap_lines.extend(lines)
    return lines


# ---------------------------------------------------------------------------
# Public Writers
# ---------------------------------------------------------------------------


def write_large_pdb(artifacts: MetalArtifacts, model: dict, *, optimized: bool) -> str:
    path = artifacts.files["large_opt_pdb" if optimized else "large_raw_pdb"]
    write_model_pdb(path, model)
    return path


def write_site_model_files(
    artifacts: MetalArtifacts,
    *,
    structure: dict,
    site_model: dict,
    watm: str,
    ionm: str,
    prom: str = "ff14SB",
    cofactor_frcmods: list[str] | None = None,
    cofactor_frcmod_by_residue: dict[tuple[str, int, str], str] | None = None,
) -> tuple[str, str, MetalSiteTyping]:
    write_model_pdb(artifacts.files["site_pdb"], site_model)
    if cofactor_frcmods is not None:
        incoming_cofactor_frcmods = list(cofactor_frcmods)
        artifacts.cofactor_frcmods.clear()
        artifacts.cofactor_frcmods.extend(incoming_cofactor_frcmods)
    if cofactor_frcmod_by_residue is not None:
        artifacts.cofactor_frcmod_by_residue.clear()
        artifacts.cofactor_frcmod_by_residue.update(cofactor_frcmod_by_residue)
    typing = _build_site_typing(
        site_model,
        watm=watm,
        ionm=ionm,
        prom=prom,
        cofactor_frcmods=artifacts.cofactor_frcmods,
        cofactor_frcmod_by_residue=artifacts.cofactor_frcmod_by_residue,
    )
    deployment_bond_pairs = _export_bond_pairs(site_model)
    write_resp_mol2(
        artifacts.files["mol2"],
        site_model,
        deployment_bond_pairs,
        atom_type_overrides=typing.mol2_atom_types,
        prom=prom,
    )
    _write_residue_mol2_files(artifacts, site_model=site_model, typing=typing)
    resid_by_key = _write_deployment_pdb(artifacts.files["tleap_pdb"], structure, typing)
    tleap_lines = _build_tleap_lines(
        artifacts,
        site_model=site_model,
        structure=structure,
        typing=typing,
        watm=watm,
        resid_by_key=resid_by_key,
        prom=prom,
    )
    with open(artifacts.files["tleap_input"], "w", encoding="utf-8") as handle:
        handle.writelines(tleap_lines)
    return artifacts.files["site_pdb"], artifacts.files["mol2"], typing


def write_site_frcmod(
    artifacts: MetalArtifacts,
    *,
    site_model: dict,
    bond_terms,
    angle_terms,
    typing: MetalSiteTyping,
) -> str:
    _write_frcmod(artifacts.files["frcmod"], site_model, bond_terms, angle_terms, typing)
    return artifacts.files["frcmod"]
