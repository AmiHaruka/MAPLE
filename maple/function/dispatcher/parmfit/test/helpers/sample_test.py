from __future__ import annotations

from collections import deque
from math import cos

import numpy as np
from ase import Atoms

from maple.function.dispatcher.parmfit.utils.mechanics import build_mm_topology_cache, dihedral_radians, evaluate_mm_energy
from maple.function.dispatcher.parmfit.utils.readparm import CorrectionParameterSet


__test__ = False

HARTREE_TO_KCAL_MOL = 627.509474

__all__ = ["HARTREE_TO_KCAL_MOL", "run_conformer_benchmark"]


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _copy_atoms_with_calc(atoms: Atoms) -> Atoms:
    copied = atoms.copy()
    copied.calc = atoms.calc
    return copied


def _normalize_deg(angle_deg: float) -> float:
    normalized = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if normalized <= -180.0:
        normalized += 360.0
    return normalized


def _bond_type_map(parameter_set: CorrectionParameterSet) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    id_to_index = parameter_set.mol2.id_to_index
    for bond in parameter_set.mol2.bonds:
        i = id_to_index[bond.atom1]
        j = id_to_index[bond.atom2]
        mapping[_pair(i, j)] = bond.bond_type
    return mapping


def _is_single_bond(bond_type: str) -> bool:
    token = bond_type.strip().lower()
    return token in {"1", "1.0", "s", "single"}


def _is_ring_bond(adjacency: dict[int, set[int]], center_bond: tuple[int, int]) -> bool:
    start, end = _pair(*center_bond)
    if end not in adjacency.get(start, set()):
        return False

    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if (node == start and neighbor == end) or (node == end and neighbor == start):
                continue
            if neighbor in seen:
                continue
            if neighbor == end:
                return True
            seen.add(neighbor)
            queue.append(neighbor)
    return False


def _enumerate_rotatable_bonds(
    parameter_set: CorrectionParameterSet,
    exclude_bonds: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    cache = build_mm_topology_cache(parameter_set)
    adjacency = parameter_set.mol2.adjacency
    bond_types = _bond_type_map(parameter_set)
    excluded = {_pair(*bond) for bond in (exclude_bonds or [])}

    rotatable: list[tuple[int, int]] = []
    for center_bond in sorted(cache.proper_by_center_bond):
        if center_bond in excluded:
            continue
        if center_bond not in bond_types:
            continue
        if not _is_single_bond(bond_types[center_bond]):
            continue
        if _is_ring_bond(adjacency, center_bond):
            continue
        left, right = center_bond
        if len(adjacency.get(left, set())) <= 1 or len(adjacency.get(right, set())) <= 1:
            continue
        rotatable.append(center_bond)
    return rotatable


def _choose_neighbor(
    atom: int,
    blocked: int,
    adjacency: dict[int, set[int]],
    symbols: list[str],
) -> int | None:
    candidates = sorted(adjacency.get(atom, set()) - {blocked}, key=lambda idx: (symbols[idx - 1] == "H", idx))
    return candidates[0] if candidates else None


def _representative_quartet(
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    left, right = _pair(*center_bond)
    symbols = atoms.get_chemical_symbols()
    adjacency = parameter_set.mol2.adjacency
    first = _choose_neighbor(left, right, adjacency, symbols)
    fourth = _choose_neighbor(right, left, adjacency, symbols)
    if first is None or fourth is None:
        return None
    return (first, left, right, fourth)


def _fragment(adjacency: dict[int, set[int]], start: int, blocked_edge: tuple[int, int]) -> set[int]:
    blocked = {_pair(*blocked_edge)}
    fragment: set[int] = set()
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if node in fragment:
            continue
        fragment.add(node)
        for neighbor in adjacency.get(node, set()):
            if _pair(node, neighbor) in blocked or neighbor in fragment:
                continue
            queue.append(neighbor)
    return fragment


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1.0e-12:
        raise ValueError("Cannot rotate around a zero-length axis.")
    ux, uy, uz = axis / axis_norm
    ct = cos(angle_rad)
    st = np.sin(angle_rad)
    one_minus_ct = 1.0 - ct
    return np.asarray(
        [
            [ct + ux * ux * one_minus_ct, ux * uy * one_minus_ct - uz * st, ux * uz * one_minus_ct + uy * st],
            [uy * ux * one_minus_ct + uz * st, ct + uy * uy * one_minus_ct, uy * uz * one_minus_ct - ux * st],
            [uz * ux * one_minus_ct - uy * st, uz * uy * one_minus_ct + ux * st, ct + uz * uz * one_minus_ct],
        ],
        dtype=float,
    )


def _apply_torsion_rotation(
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    target_deg: float,
) -> None:
    quartet = _representative_quartet(atoms, parameter_set, center_bond)
    if quartet is None:
        return

    current_deg = np.degrees(dihedral_radians(atoms.get_positions(), *quartet))
    delta_deg = _normalize_deg(_normalize_deg(target_deg) - _normalize_deg(current_deg))
    if abs(delta_deg) <= 1.0e-10:
        return

    left, right = _pair(*center_bond)
    fragment = _fragment(parameter_set.mol2.adjacency, quartet[3], (left, right))
    positions = np.asarray(atoms.get_positions(), dtype=float).copy()
    origin = positions[left - 1]
    axis = positions[right - 1] - origin
    rotation = _rotation_matrix(axis, np.deg2rad(delta_deg))

    for atom_index in sorted(fragment):
        point = positions[atom_index - 1] - origin
        positions[atom_index - 1] = origin + point @ rotation.T
    atoms.set_positions(positions)


def _heavy_atom_indices(atoms: Atoms) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    heavy = [index for index, symbol in enumerate(symbols) if symbol.upper() != "H"]
    if heavy:
        return np.asarray(heavy, dtype=int)
    return np.arange(len(symbols), dtype=int)


def _kabsch_rmsd(reference: Atoms, candidate: Atoms, atom_indices: np.ndarray) -> float:
    ref = np.asarray(reference.get_positions(), dtype=float)[atom_indices]
    cand = np.asarray(candidate.get_positions(), dtype=float)[atom_indices]

    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cand_centered = cand - cand.mean(axis=0, keepdims=True)

    covariance = cand_centered.T @ ref_centered
    left, _, right_t = np.linalg.svd(covariance)
    if np.linalg.det(left @ right_t) < 0.0:
        left[:, -1] *= -1.0
    rotation = left @ right_t
    aligned = cand_centered @ rotation
    delta = aligned - ref_centered
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _mlp_single_point_hartree(atoms: Atoms) -> float:
    if atoms.calc is None:
        raise ValueError("Atoms has no calculator attached for MLP single-point evaluation.")
    try:
        return float(atoms.calc.get_potential_energy(atoms, force_consistent=True))
    except TypeError:
        return float(atoms.calc.get_potential_energy(atoms))


def run_conformer_benchmark(
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
    n_samples: int,
    k: int,
    keep_top_n: int = 20,
    exclude_bonds: list[tuple[int, int]] | None = None,
    random_seed: int | None = None,
    rmsd_threshold: float = 0.5,
    rotamers_deg: tuple[float, ...] = (60.0, 180.0, 300.0),
) -> tuple[list[Atoms], np.ndarray, np.ndarray, float]:
    if len(atoms) != len(parameter_set.mol2.atoms):
        raise ValueError(
            f"Atoms length ({len(atoms)}) does not match parameter set size ({len(parameter_set.mol2.atoms)})."
        )
    if atoms.calc is None:
        raise ValueError("Atoms has no calculator attached for MLP single-point evaluation.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if keep_top_n < 1:
        raise ValueError("keep_top_n must be at least 1.")
    if not rotamers_deg:
        raise ValueError("rotamers_deg must contain at least one torsion angle.")

    rng = np.random.default_rng(random_seed)
    rotatable_bonds = _enumerate_rotatable_bonds(parameter_set, exclude_bonds)

    raw_conformers: list[Atoms] = []
    if rotatable_bonds and k > 0:
        sample_size = min(int(k), len(rotatable_bonds))
        for _ in range(int(n_samples)):
            conformer = _copy_atoms_with_calc(atoms)
            chosen_indices = rng.choice(len(rotatable_bonds), size=sample_size, replace=False)
            for bond_index in np.atleast_1d(chosen_indices):
                target = float(rng.choice(np.asarray(rotamers_deg, dtype=float)))
                _apply_torsion_rotation(conformer, parameter_set, rotatable_bonds[int(bond_index)], target)
            raw_conformers.append(conformer)
    else:
        raw_conformers.append(_copy_atoms_with_calc(atoms))

    heavy_indices = _heavy_atom_indices(atoms)
    unique_conformers: list[Atoms] = []
    for conformer in raw_conformers:
        is_duplicate = any(
            _kabsch_rmsd(reference, conformer, heavy_indices) < rmsd_threshold for reference in unique_conformers
        )
        if not is_duplicate:
            unique_conformers.append(conformer)

    mlp_hartree = np.asarray([_mlp_single_point_hartree(conformer) for conformer in unique_conformers], dtype=float)
    ranking = np.argsort(mlp_hartree)
    keep_count = min(int(keep_top_n), len(unique_conformers))
    keep_indices = ranking[:keep_count]

    kept_conformers = [unique_conformers[index] for index in keep_indices]
    mlp_hartree_kept = mlp_hartree[keep_indices]
    mlp_kcal_kept = mlp_hartree_kept * HARTREE_TO_KCAL_MOL

    topology_cache = build_mm_topology_cache(parameter_set)
    mm_kcal_kept = np.asarray(
        [evaluate_mm_energy(conformer, parameter_set, topology_cache=topology_cache).total for conformer in kept_conformers],
        dtype=float,
    )

    ref_idx = int(np.argmin(mlp_kcal_kept))
    mlp_rel_kcal = mlp_kcal_kept - mlp_kcal_kept[ref_idx]
    mm_rel_kcal = mm_kcal_kept - mm_kcal_kept[ref_idx]
    energy_rmse_kcal = float(np.sqrt(np.mean((mm_rel_kcal - mlp_rel_kcal) ** 2)))

    return kept_conformers, mlp_rel_kcal, mm_rel_kcal, energy_rmse_kcal
