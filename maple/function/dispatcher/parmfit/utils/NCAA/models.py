"""Usage: build NCAA residue models, conformers, graph helpers, and torsion filters."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil
import os

import numpy as np

from ..capping import build_ace_cap, build_nme_cap
from ..context import find_residue_by_key
from ..model import flatten_model_atoms, infer_bond_pairs, model_to_atoms, update_model_from_atoms
from ..runtime import (
    copy_thresholds,
    get_potential_energy,
    optimize_atoms_geometry,
    optimize_model_geometry,
)
from ..Scan import run_silent_scan
from ..structure import copy_residue, covalent_cutoff, get_atom_xyz, get_resid_key, get_resid_label, max_serial, search_atom, _measure_dihedral


@dataclass(frozen=True)
class NCAAIdentity:
    residue_key: tuple[str, int, str]
    resname: str
    chirality: str
    sidechain_anchor: str


@dataclass(frozen=True)
class NCAAConformer:
    label: str
    phi_deg: float
    psi_deg: float
    energy: float
    model: dict


def identity_ncaa(target_residue: dict) -> NCAAIdentity:
    sidechain_atom = _find_sidechain_anchor(target_residue)
    chirality = detect_ncaa_chirality(target_residue, sidechain_atom)
    return NCAAIdentity(
        residue_key=get_resid_key(target_residue),
        resname=target_residue["resname"],
        chirality=chirality,
        sidechain_anchor=sidechain_atom["name"],
    )


def _find_sidechain_anchor(residue: dict) -> dict:
    ca_atom = search_atom(residue, "CA")
    n_atom = search_atom(residue, "N")
    c_atom = search_atom(residue, "C")
    if ca_atom is None or n_atom is None or c_atom is None:
        raise ValueError("NCAA chirality detection requires residue backbone atoms N, CA, and C.")

    sidechain_candidates: list[tuple[float, dict]] = []
    for atom in residue["atoms"]:
        if atom["name"] in {"N", "CA", "C", "O"} or atom["element"] == "H":
            continue
        delta = get_atom_xyz(atom) - get_atom_xyz(ca_atom)
        distance = float(np.linalg.norm(delta))
        if distance <= covalent_cutoff(ca_atom, atom):
            sidechain_candidates.append((distance, atom))
    sidechain_candidates.sort(key=lambda item: (item[0], item[1]["serial"]))
    return sidechain_candidates[0][1]


def detect_ncaa_chirality(residue: dict, sidechain_atom: dict | None = None) -> str:
    ca_atom = search_atom(residue, "CA")
    n_atom = search_atom(residue, "N")
    c_atom = search_atom(residue, "C")
    sidechain_atom = _find_sidechain_anchor(residue) if sidechain_atom is None else sidechain_atom

    ca_xyz = get_atom_xyz(ca_atom)
    n_vec = get_atom_xyz(n_atom) - ca_xyz
    c_vec = get_atom_xyz(c_atom) - ca_xyz
    sidechain_vec = get_atom_xyz(sidechain_atom) - ca_xyz
    signed_volume = float(np.linalg.det(np.stack([n_vec, c_vec, sidechain_vec], axis=1)))
    return "L" if signed_volume > 0.0 else "D"


def conformer_targets(chirality: str) -> list[tuple[str, float, float]]:
    if chirality == "L":
        return [("alpha", -60.0, -40.0), ("beta", -120.0, -140.0)]
    else:  #chirality == "D"
        return [("alpha", 60.0, 40.0), ("beta", 120.0, 140.0)]


def build_capped_ncaa_model(target_residue: dict, rn: str) -> dict:
    target_copy = copy_residue(target_residue, resname=rn)
    next_serial = max_serial([target_residue]) + 1
    ace_residue, next_serial = build_ace_cap(target_residue, next_serial)
    nme_residue, next_serial = build_nme_cap(target_residue, next_serial)
    residues = [ace_residue, target_copy, nme_residue]
    return {
        "name": "ncaa_capped_model",
        "target_key": get_resid_key(target_copy),
        "residues": residues,
        "segment_sizes": {
            "ace": len(ace_residue["atoms"]),
            "residue": len(target_copy["atoms"]),
            "nme": len(nme_residue["atoms"]),
        },
    }


def infer_terminal_omit_names(resid: dict) -> list[str]:
    bonds = infer_bond_pairs({"residues": [resid]})
    atom_names = [atom["name"] for atom in sorted(resid["atoms"], key=lambda atom: atom["serial"])]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in bonds:
        left_name = atom_names[left - 1]
        right_name = atom_names[right - 1]
        adjacency[left_name].add(right_name)
        adjacency[right_name].add(left_name)

    omit_names: set[str] = set()
    n_hydrogens = sorted(name for name in adjacency.get("N", set()) if search_atom(resid, name)["element"] == "H")
    if len(n_hydrogens) > 1:
        keep_name = next((name for name in ("H", "HN", "H1", "HN1") if name in n_hydrogens), n_hydrogens[0])
        for atom_name in n_hydrogens:
            if atom_name != keep_name:
                omit_names.add(atom_name)

    c_oxygen_names = sorted(name for name in adjacency.get("C", set()) if search_atom(resid, name)["element"] == "O")
    if len(c_oxygen_names) > 1:
        keep_name = "O" if "O" in c_oxygen_names else c_oxygen_names[0]
        for atom_name in c_oxygen_names:
            if atom_name == keep_name:
                continue
            omit_names.add(atom_name)
            for hydrogen_name in sorted(
                neighbor for neighbor in adjacency.get(atom_name, set()) if search_atom(resid, neighbor)["element"] == "H"
            ):
                omit_names.add(hydrogen_name)

    return sorted(omit_names)


def optimize_capped_reference(
    model: dict,
    *,
    source_atoms,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
    frozen_indices: tuple[int, ...] | None = None,
) -> NCAAConformer:
    if frozen_indices:
        atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
        copy_thresholds(source_atoms, atoms)
        atoms.calc = source_atoms.calc
        from ase.constraints import FixAtoms

        atoms.set_constraint(FixAtoms(indices=list(frozen_indices)))
        try:
            optimize_atoms_geometry(
                atoms,
                output=output,
                max_iter=max_iter,
                max_step=max_step,
                failure_message="NCAA representative minimization.",
            )
        finally:
            atoms.set_constraint(None)
        minimized_model = update_model_from_atoms(model, atoms)
        energy = float(get_potential_energy(atoms))
    else:
        minimized_model = optimize_model_geometry(
            model,
            output=output,
            source_atoms=source_atoms,
            max_iter=max_iter,
            max_step=max_step,
            failure_message="NCAA representative minimization.",
        )
        optimized_atoms = model_to_atoms(
            minimized_model,
            charge=minimized_model.get("charge"),
            mult=minimized_model.get("mult"),
        )
        optimized_atoms.calc = source_atoms.calc
        energy = float(get_potential_energy(optimized_atoms))
    return NCAAConformer(
        label="ref",
        phi_deg=0.0,
        psi_deg=0.0,
        energy=energy,
        model=minimized_model,
    )


def minimize_conformer(
    model: dict,
    *,
    label: str,
    phi_deg: float,
    psi_deg: float,
    source_atoms,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
) -> NCAAConformer:
    try:
        from ase.constraints import FixInternals
    except ModuleNotFoundError:
        class FixInternals:  # pragma: no cover - test fallback only
            def __init__(self, *, dihedrals_deg=None, **kwargs):
                del kwargs
                self.dihedrals = dihedrals_deg or []

    atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
    copy_thresholds(source_atoms, atoms)
    atoms.calc = source_atoms.calc

    index_map = _build_backbone_rotation_map(model)
    current_phi = _measure_dihedral(atoms, index_map["phi"])
    current_psi = _measure_dihedral(atoms, index_map["psi"])
    phi_delta = ((phi_deg - current_phi + 180.0) % 360.0) - 180.0
    psi_delta = ((psi_deg - current_psi + 180.0) % 360.0) - 180.0
    resolved_phi = current_phi + phi_delta
    resolved_psi = current_psi + psi_delta
    phi_steps = max(1, ceil(abs(phi_delta) / 15.0))
    psi_steps = max(1, ceil(abs(psi_delta) / 15.0))

    scan_result = run_silent_scan(
        output=output,
        atoms=atoms,
        constraints=[
            [*(index + 1 for index in index_map["phi"]), phi_delta / phi_steps, phi_steps],
            [*(index + 1 for index in index_map["psi"]), psi_delta / psi_steps, psi_steps],
        ],
        params={
            "mode": "rigid",  #relaxed or rigid
            "opt": {
                "max_iter": max_iter,
                "max_step": max_step,
            },
        },
        method="lbfgs",
    )
    scan_xyz = scan_result.xyz_path

    with open(scan_xyz, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    frame = lines[-(len(atoms) + 2) :]
    minimized_atoms = atoms.copy()
    minimized_atoms.set_positions(np.asarray([line.split()[1:4] for line in frame[2:]], dtype=float))
    copy_thresholds(source_atoms, minimized_atoms)
    minimized_atoms.calc = source_atoms.calc
    constraint = FixInternals(
        dihedrals_deg=[
            [resolved_phi, list(index_map["phi"])],
            [resolved_psi, list(index_map["psi"])],
        ]
    )
    if hasattr(minimized_atoms, "set_constraint"):
        minimized_atoms.set_constraint(constraint)
    else:  # pragma: no cover - test stub fallback
        minimized_atoms.constraints = [constraint]
    try:
        optimize_atoms_geometry(
            minimized_atoms,
            output=f"{os.path.splitext(output)[0]}_opt.out",
            max_iter=max_iter,
            max_step=max_step,
            failure_message=f"NCAA {label} conformer optimization did not converge.",
        )
    finally:
        if hasattr(minimized_atoms, "set_constraint"):
            minimized_atoms.set_constraint(None)
        else:  # pragma: no cover - test stub fallback
            minimized_atoms.constraints = []
    minimized = update_model_from_atoms(model, minimized_atoms)
    return NCAAConformer(
        label=label,
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        energy=float(get_potential_energy(minimized_atoms)),
        model=minimized,
    )


def build_resp_conformers_from_reference(
    reference_model: dict,
    *,
    chirality: str,
    source_atoms,
    output_base: str,
    max_iter: int = 256,
    max_step: float = 0.2,
) -> list[NCAAConformer]:
    conformers: list[NCAAConformer] = []
    for label, phi_deg, psi_deg in conformer_targets(chirality):
        conformers.append(
            minimize_conformer(
                model=reference_model,
                label=label,
                phi_deg=phi_deg,
                psi_deg=psi_deg,
                source_atoms=source_atoms,
                output=f"{output_base}_{label}.out",
                max_iter=max_iter,
                max_step=max_step,
            )
        )
    return conformers


def _build_backbone_rotation_map(model: dict) -> dict[str, tuple[int, int, int, int]]:
    index_by_serial = {atom["serial"]: index for index, (_residue, atom) in enumerate(flatten_model_atoms(model))}

    ace_resid, target_resid, nme_resid = model["residues"]
    return {
        "omega_pre": (
            index_by_serial[search_atom(ace_resid, "OAC")["serial"]],
            index_by_serial[search_atom(ace_resid, "CAC")["serial"]],
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
        ),
        "phi": (
            index_by_serial[search_atom(ace_resid, "CAC")["serial"]],
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
        ),
        "psi": (
            index_by_serial[search_atom(target_resid, "N")["serial"]],
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
            index_by_serial[search_atom(nme_resid, "NNM")["serial"]],
        ),
        "omega_post": (
            index_by_serial[search_atom(target_resid, "CA")["serial"]],
            index_by_serial[search_atom(target_resid, "C")["serial"]],
            index_by_serial[search_atom(nme_resid, "NNM")["serial"]],
            index_by_serial[search_atom(nme_resid, "CNM")["serial"]],
        ),
    }


def build_residue_local_adjacency(residue: dict) -> tuple[list[dict], dict[str, int], dict[int, list[int]]]:
    atoms = sorted(residue["atoms"], key=lambda atom: atom["serial"])
    local_index_by_name = {atom["name"]: index + 1 for index, atom in enumerate(atoms)}
    adjacency = {index: [] for index in range(1, len(atoms) + 1)}
    for left, right in infer_bond_pairs({"residues": [residue]}):
        adjacency[left].append(right)
        adjacency[right].append(left)
    for index in adjacency:
        adjacency[index].sort(key=lambda neighbor: atoms[neighbor - 1]["serial"])
    return atoms, local_index_by_name, adjacency


def infer_mainchain_names(residue: dict) -> list[str]:
    atoms, local_index_by_name, adjacency = build_residue_local_adjacency(residue)
    start = local_index_by_name["N"]
    goal = local_index_by_name["C"]

    parent = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in adjacency[current]:
            if atoms[neighbor - 1]["element"] == "H":
                continue
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if goal not in parent:
        raise ValueError("Could not infer NCAA mainchain path from target residue N to C.")

    path: list[int] = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return [atoms[index - 1]["name"] for index in path[1:-1]]


def build_ncaa_center_bond_filter(representative_model: dict):
    target_residue = find_residue_by_key(
        representative_model,
        representative_model["target_key"],
        label="Representative NCAA target residue",
    )
    residue_atoms, _local_index_by_name, local_adjacency = build_residue_local_adjacency(target_residue)
    residue_start = int(representative_model["segment_sizes"]["ace"]) + 1
    residue_indices = {residue_start + offset for offset in range(len(residue_atoms))}
    backbone_indices = {
        residue_start + offset
        for offset, atom in enumerate(residue_atoms)
        if atom["name"] in {"N", "CA", "C", "O"}
    }

    ca_local_index = next(index for index, atom in enumerate(residue_atoms, start=1) if atom["name"] == "CA")
    sidechain_anchors = [
        neighbor
        for neighbor in sorted(local_adjacency[ca_local_index])
        if residue_atoms[neighbor - 1]["element"] != "H" and residue_atoms[neighbor - 1]["name"] not in {"N", "C", "O"}
    ]
    if not sidechain_anchors:
        raise ValueError(
            f"NCAA target residue {get_resid_label(target_residue)} has no sidechain heavy atoms attached to CA."
        )

    r_group_indices: set[int] = set()
    seen = set(sidechain_anchors)
    stack = sidechain_anchors[:]
    while stack:
        local_index = stack.pop()
        atom = residue_atoms[local_index - 1]
        if atom["element"] == "H" or atom["name"] in {"N", "CA", "C", "O"}:
            continue
        r_group_indices.add(residue_start + local_index - 1)
        for neighbor in sorted(local_adjacency[local_index]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)

    def keep(center_bond: tuple[int, int]) -> bool:
        return (
            center_bond[0] in residue_indices
            and center_bond[1] in residue_indices
            and not (center_bond[0] in backbone_indices and center_bond[1] in backbone_indices)
            and (center_bond[0] in r_group_indices or center_bond[1] in r_group_indices)
        )

    return keep
