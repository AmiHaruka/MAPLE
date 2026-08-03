"""Usage: build capped residue models and convert them to ASE/PDB forms."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np

from .capping import build_ace_cap, build_gly_bridge, build_nme_cap
from .context import find_prev_next_peptide_residues
from .structure import (
    _pair_is_bonded,
    copy_residue,
    covalent_cutoff,
    get_atom_xyz,
    get_resid_key,
    is_peptide_like,
    max_serial,
    residue_sort_key,
)


def flatten_model_atoms(model: dict) -> list[tuple[dict, dict]]:
    flattened: list[tuple[dict, dict]] = []
    for residue in model["residues"]:
        for atom in sorted(residue["atoms"], key=lambda item: item["serial"]):
            flattened.append((residue, atom))
    return flattened


def rebuild_model_index(model: dict) -> dict:
    model["residues"] = sorted(model["residues"], key=residue_sort_key)
    serial_to_atom: dict[int, dict] = {}
    serial_to_residue: dict[int, dict] = {}
    for residue in model["residues"]:
        residue["atoms"] = sorted(residue["atoms"], key=lambda atom: atom["serial"])
        residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)
        for atom in residue["atoms"]:
            serial_to_atom[atom["serial"]] = atom
            serial_to_residue[atom["serial"]] = residue
    model["serial_to_atom"] = serial_to_atom
    model["serial_to_residue"] = serial_to_residue
    for key in ("explicit_pairs", "bond_pairs", "coordination_pairs"):
        if key in model:
            model[key] = {
                pair for pair in model[key] if pair[0] in serial_to_atom and pair[1] in serial_to_atom
            }
    model.pop("_pair_cache", None)
    return model


def copy_structure_subset(structure: dict, residues: list[dict]) -> dict:
    copied_residues = [copy_residue(residue) for residue in sorted(residues, key=residue_sort_key)]
    model = {
        "path": structure.get("path"),
        "residues": copied_residues,
        "explicit_pairs": {
            pair
            for pair in structure["explicit_pairs"]
            if pair[0] in structure["serial_to_atom"] and pair[1] in structure["serial_to_atom"]
        },
        "bond_pairs": {
            pair
            for pair in structure.get("bond_pairs", set())
            if pair[0] in structure["serial_to_atom"] and pair[1] in structure["serial_to_atom"]
        },
        "coordination_pairs": {
            pair
            for pair in structure.get("coordination_pairs", set())
            if pair[0] in structure["serial_to_atom"] and pair[1] in structure["serial_to_atom"]
        },
    }
    return rebuild_model_index(model)


def _structure_residue_map(structure: dict) -> dict[tuple[str, int, str], dict]:
    return {get_resid_key(residue): residue for residue in structure["residues"]}


def _collect_gly_bridge_keys(
    structure: dict,
    selected_residues: list[dict],
    *,
    max_gap: int = 5,
) -> set[tuple[str, int, str]]:
    selected_keys = {get_resid_key(residue) for residue in selected_residues}
    peptide_residues = [
        residue
        for residue in sorted(selected_residues, key=residue_sort_key)
        if residue["kind"] == "protein" and is_peptide_like(residue)
    ]

    bridge_keys: set[tuple[str, int, str]] = set()
    for left, right in zip(peptide_residues, peptide_residues[1:], strict=False):
        if left["chain"] != right["chain"]:
            continue
        gap = int(right["resseq"]) - int(left["resseq"])
        if gap <= 1 or gap > max_gap:
            continue
        for candidate in structure["residues"]:
            if candidate["chain"] != left["chain"]:
                continue
            if candidate["kind"] != "protein" or not is_peptide_like(candidate):
                continue
            if not (int(left.get("_index", 0)) < int(candidate.get("_index", 0)) < int(right.get("_index", 0))):
                continue
            candidate_key = get_resid_key(candidate)
            if candidate_key in selected_keys:
                continue
            bridge_keys.add(candidate_key)
    return bridge_keys


def _replace_bridge_residues(
    model: dict,
    structure: dict,
    bridge_keys: set[tuple[str, int, str]],
    *,
    bond_policy: str = "auto",
) -> None:
    if not bridge_keys:
        return
    source_by_key = _structure_residue_map(structure)
    next_serial = max_serial(model) + 1
    rebuilt_residues: list[dict] = []
    for residue in model["residues"]:
        residue_key = get_resid_key(residue)
        if residue_key not in bridge_keys:
            rebuilt_residues.append(residue)
            continue
        source_residue = source_by_key[residue_key]
        prev_residue, _ = find_prev_next_peptide_residues(structure, source_residue, bond_policy=bond_policy)
        bridge_residue, next_serial = build_gly_bridge(source_residue, next_serial, prev_residue=prev_residue)
        bridge_residue["_index"] = source_residue.get("_index", residue.get("_index", 0))
        rebuilt_residues.append(bridge_residue)
    model["residues"] = rebuilt_residues
    model.pop("bond_pairs", None)
    model.pop("coordination_pairs", None)
    rebuild_model_index(model)


def _cap_outer_peptide_boundaries(model: dict, structure: dict, *, bond_policy: str = "auto") -> None:
    source_by_key = _structure_residue_map(structure)
    included_keys = {get_resid_key(residue) for residue in model["residues"]}
    next_serial = max_serial(model) + 1
    cap_residues: list[dict] = []

    for residue in model["residues"]:
        residue_key = get_resid_key(residue)
        source_residue = source_by_key.get(residue_key)
        if source_residue is None or not is_peptide_like(source_residue):
            continue

        prev_residue, next_residue = find_prev_next_peptide_residues(structure, source_residue, bond_policy=bond_policy)
        residue_index = int(residue.get("_index", source_residue.get("_index", 0)))

        prev_key = get_resid_key(prev_residue) if prev_residue is not None else None
        if prev_key not in included_keys:
            ace_residue, next_serial = build_ace_cap(source_residue, next_serial, prev_residue=prev_residue)
            ace_residue["_index"] = residue_index - 1
            cap_residues.append(ace_residue)

        next_key = get_resid_key(next_residue) if next_residue is not None else None
        if next_key not in included_keys:
            nme_residue, next_serial = build_nme_cap(source_residue, next_serial, next_residue=next_residue)
            nme_residue["_index"] = residue_index + 1
            cap_residues.append(nme_residue)

    if not cap_residues:
        return
    model["residues"].extend(cap_residues)
    model.pop("bond_pairs", None)
    model.pop("coordination_pairs", None)
    rebuild_model_index(model)


def build_capped_selected_model(
    structure: dict,
    selected_residues: list[dict],
    *,
    bond_policy: str = "auto",
) -> dict:
    bridge_keys = _collect_gly_bridge_keys(structure, selected_residues)
    structure_by_key = _structure_residue_map(structure)
    bridge_residues = [structure_by_key[key] for key in sorted(bridge_keys)]
    model = copy_structure_subset(structure, selected_residues + bridge_residues)
    _replace_bridge_residues(model, structure, bridge_keys, bond_policy=bond_policy)
    _cap_outer_peptide_boundaries(model, structure, bond_policy=bond_policy)
    return model


def model_to_atoms(model: dict, charge: Optional[int] = None, mult: Optional[int] = None):
    from ase import Atoms

    symbols: list[str] = []
    coords: list[np.ndarray] = []
    serials: list[int] = []
    residue_offsets: list[tuple[tuple[str, int, str], int, int]] = []
    start = 0
    for residue in model["residues"]:
        atoms = sorted(residue["atoms"], key=lambda atom: atom["serial"])
        for atom in atoms:
            symbols.append(atom["element"].title())
            coords.append(get_atom_xyz(atom))
            serials.append(int(atom["serial"]))
        stop = start + len(atoms)
        residue_offsets.append((get_resid_key(residue), start, stop))
        start = stop

    ase_atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=float))
    resolved_charge = model.get("charge") if charge is None else charge
    resolved_mult = model.get("mult") if mult is None else mult
    if resolved_charge is not None:
        ase_atoms.info["charge"] = int(resolved_charge)
    if resolved_mult is not None:
        ase_atoms.info["mult"] = int(resolved_mult)
        ase_atoms.info["spin"] = (int(resolved_mult) - 1) / 2
    ase_atoms.info["parmfit_serials"] = serials
    ase_atoms.info["parmfit_residue_offsets"] = residue_offsets
    return ase_atoms


def update_model_from_atoms(model: dict, atoms) -> dict:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    updated_residues: list[dict] = []
    serial_to_atom: dict[int, dict] = {}
    serial_to_residue: dict[int, dict] = {}
    offset = 0
    for residue in model["residues"]:
        copied = copy_residue(residue)
        ordered_atoms = sorted(copied["atoms"], key=lambda atom: atom["serial"])
        for local_index, atom in enumerate(ordered_atoms):
            atom["xyz"] = np.array(positions[offset + local_index], dtype=float)
            serial_to_atom[atom["serial"]] = atom
            serial_to_residue[atom["serial"]] = copied
        copied["atoms"] = ordered_atoms
        copied["coords"] = np.asarray([atom["xyz"] for atom in ordered_atoms], dtype=float)
        updated_residues.append(copied)
        offset += len(ordered_atoms)

    updated = dict(model)
    updated["residues"] = updated_residues
    for key in ("explicit_pairs", "bond_pairs", "coordination_pairs"):
        if key in model:
            updated[key] = set(model[key])
    updated["serial_to_atom"] = serial_to_atom
    updated["serial_to_residue"] = serial_to_residue
    updated.pop("_pair_cache", None)
    if "charge" not in updated and "charge" in atoms.info:
        updated["charge"] = int(atoms.info["charge"])
    if "mult" not in updated and "mult" in atoms.info:
        updated["mult"] = int(atoms.info["mult"])
    return updated


def write_model_pdb(path: str, model_or_residues: dict | list[dict]) -> None:
    residues = model_or_residues["residues"] if isinstance(model_or_residues, dict) else model_or_residues
    sorted_residues = sorted(residues, key=residue_sort_key)
    with open(path, "w", encoding="utf-8") as handle:
        for index, residue in enumerate(sorted_residues):
            record = "ATOM" if residue["kind"] in {"protein", "cap", "small_model"} else "HETATM"
            chain = residue["chain"] if residue["chain"] != "_" else "A"
            for atom in sorted(residue["atoms"], key=lambda item: item["serial"]):
                x, y, z = get_atom_xyz(atom)
                element = atom["element"][-2:].rjust(2)
                handle.write(
                    f"{record:<6}{int(atom['serial']):5d} {atom['name'][:4]:>4s} {residue['resname'][:3]:>3s} "
                    f"{chain[:1]:1s}{int(residue['resseq']):4d}{residue['icode'][:1]:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
                )
            next_residue = sorted_residues[index + 1] if index + 1 < len(sorted_residues) else None
            next_is_atom = next_residue is not None and next_residue["kind"] in {"protein", "cap", "small_model"}
            next_chain = None if next_residue is None else ("A" if next_residue["chain"] == "_" else next_residue["chain"])
            if record == "ATOM" and (not next_is_atom or next_chain != chain):
                handle.write("TER\n")
        handle.write("END\n")


def _pair_bonded_without_structure(atom1: dict, atom2: dict) -> bool:
    cutoff = covalent_cutoff(atom1, atom2)
    delta = get_atom_xyz(atom1) - get_atom_xyz(atom2)
    return float(np.sqrt(np.dot(delta, delta))) <= cutoff


def infer_bond_pairs(model: dict, source_structure: Optional[dict] = None, bond_policy: str = "auto") -> list[tuple[int, int]]:
    atoms = [atom for _, atom in flatten_model_atoms(model)]
    serial_to_index = {atom["serial"]: index for index, atom in enumerate(atoms, start=1)}
    if bond_policy == "auto" and len(serial_to_index) == len(atoms):
        graph_source = source_structure if source_structure is not None and "bond_pairs" in source_structure else model
        if "bond_pairs" in graph_source:
            bonds = [
                tuple(sorted((serial_to_index[left], serial_to_index[right])))
                for left, right in graph_source["bond_pairs"]
                if left in serial_to_index and right in serial_to_index
            ]
            known_serials = set(graph_source.get("serial_to_atom", serial_to_index))
            new_indices = [index for index, atom in enumerate(atoms) if atom["serial"] not in known_serials]
            if not new_indices:
                return sorted(set(bonds))
            for left, right in combinations(range(len(atoms)), 2):
                if left not in new_indices and right not in new_indices:
                    continue
                if _pair_bonded_without_structure(atoms[left], atoms[right]):
                    bonds.append((left + 1, right + 1))
            return sorted(set(bonds))
    bonds: list[tuple[int, int]] = []
    for left, right in combinations(range(len(atoms)), 2):
        atom1 = atoms[left]
        atom2 = atoms[right]
        if (
            source_structure is not None
            and atom1["serial"] in source_structure["serial_to_atom"]
            and atom2["serial"] in source_structure["serial_to_atom"]
        ):
            bonded = _pair_is_bonded(
                source_structure,
                source_structure["serial_to_atom"][atom1["serial"]],
                source_structure["serial_to_atom"][atom2["serial"]],
                bond_policy=bond_policy,
            )
        else:
            bonded = _pair_bonded_without_structure(atom1, atom2)
        if bonded:
            bonds.append((left + 1, right + 1))
    return bonds


def build_bond_angle_terms(
    model: dict,
    source_structure: Optional[dict] = None,
    bond_policy: str = "auto",
    bond_pairs: Optional[list[tuple[int, int]]] = None,
):
    try:
        from .readparm import Angle, Bond
    except ImportError:  # pragma: no cover
        from readparm import Angle, Bond  # type: ignore

    atoms = [atom for _, atom in flatten_model_atoms(model)]
    bonds = bond_pairs if bond_pairs is not None else infer_bond_pairs(
        model,
        source_structure=source_structure,
        bond_policy=bond_policy,
    )
    adjacency: dict[int, set[int]] = defaultdict(set)
    bond_terms: list[Bond] = []
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
        atom_types = (
            atoms[left - 1].get("atom_type") or atoms[left - 1].get("amber_type") or atoms[left - 1]["element"],
            atoms[right - 1].get("atom_type") or atoms[right - 1].get("amber_type") or atoms[right - 1]["element"],
        )
        bond_terms.append(Bond(atoms=(left, right), atom_types=atom_types))

    angle_terms: list[Angle] = []
    for center, neighbors in adjacency.items():
        for left, right in combinations(sorted(neighbors), 2):
            atom_types = (
                atoms[left - 1].get("atom_type") or atoms[left - 1].get("amber_type") or atoms[left - 1]["element"],
                atoms[center - 1].get("atom_type") or atoms[center - 1].get("amber_type") or atoms[center - 1]["element"],
                atoms[right - 1].get("atom_type") or atoms[right - 1].get("amber_type") or atoms[right - 1]["element"],
            )
            angle_terms.append(Angle(atoms=(left, center, right), atom_types=atom_types))

    return bond_terms, angle_terms
