from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import degrees
from typing import Optional

import numpy as np
from ase import Atoms

from .mmcalc import build_mm_topology_cache, evaluate_mm_energy
from .mmfunc import dihedral_radians
from .parm import Dihedral, FourierTerm
from .readparm import CorrectionParameterSet


HARTREE_TO_KCAL_MOL = 627.509474
_PRIOR_L2_WEIGHT = 6
_ARMIJO_C1 = 1.0e-4
_BACKTRACK_SCALE = 0.5
_MIN_LINESEARCH_STEP = 1.0e-6
_GRADIENT_TOL = 1.0e-10


# -----------------------------------------------------------------------------
# Constants and params
# -----------------------------------------------------------------------------

@dataclass
class TorsionFitParams:
    enabled: bool = True
    center_bonds: Optional[tuple[tuple[int, int], ...]] = None
    scan_step_deg: float = 5.0
    scan_steps: int = 72
    refine_rounds: int = 200
    refine_max_iter: int = 100
    refine_tol: float = 1.0e-4

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid torsionfit boolean value: {value!r}")


def _coerce_int(value, default: int) -> int:
    return default if value is None else int(value)


def _coerce_float(value, default: float) -> float:
    return default if value is None else float(value)


# -----------------------------------------------------------------------------
# Center-bond utilities
# -----------------------------------------------------------------------------

def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def normalize_center_bond(center_bond: tuple[int, int]) -> tuple[int, int]:
    return _pair(*center_bond)


def _parse_center_bonds(value) -> Optional[tuple[tuple[int, int], ...]]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    entries = value if isinstance(value, (list, tuple)) else str(value).split(",")
    bonds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        if isinstance(entry, str):
            token = entry.strip()
            if not token:
                continue
            left, right = token.split("-", 1)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            left, right = entry
        else:
            raise ValueError(f"Invalid torsion center bond entry: {entry!r}")
        bond = normalize_center_bond((int(left), int(right)))
        if bond[0] == bond[1]:
            raise ValueError(f"torsion center bond cannot be self-referential: {entry!r}")
        if bond not in seen:
            seen.add(bond)
            bonds.append(bond)
    return tuple(bonds) if bonds else None


def build_torsion_fit_params(paras: Optional[dict]) -> TorsionFitParams:
    params = paras if isinstance(paras, dict) else {}
    root = params
    for alias in ("correction", "parmfit"):
        if isinstance(params.get(alias), dict):
            root = params[alias]
            break
    nested = root.get("torsion")
    source = nested if isinstance(nested, dict) else root
    return TorsionFitParams(
        enabled=_coerce_bool(source.get("torsionfit", source.get("enabled")), True),
        center_bonds=_parse_center_bonds(source.get("torsion_bonds", source.get("center_bonds"))),
        scan_step_deg=_coerce_float(source.get("torsion_scan_step", source.get("scan_step_deg")), 5.0),
        scan_steps=_coerce_int(source.get("torsion_scan_steps", source.get("scan_steps")), 72),
        refine_rounds=_coerce_int(source.get("torsion_refine_rounds", source.get("refine_rounds")), 200),
        refine_max_iter=_coerce_int(source.get("torsion_refine_max_iter", source.get("refine_max_iter")), 100),
        refine_tol=_coerce_float(source.get("torsion_refine_tol", source.get("refine_tol")), 1.0e-4),
    )


def _center_bond_dihedral_indices(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
) -> list[int]:
    center = normalize_center_bond(center_bond)
    return sorted(
        [
            index
            for index, dihedral in enumerate(parameter_set.dihedrals)
            if _pair(dihedral.atoms[1], dihedral.atoms[2]) == center
        ],
        key=lambda index: parameter_set.dihedrals[index].atoms,
    )


def center_bond_dihedrals(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> list[Dihedral]:
    return [parameter_set.dihedrals[index] for index in _center_bond_dihedral_indices(parameter_set, center_bond)]


def representative_dihedral_for_center_bond(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> Dihedral:
    dihedrals = center_bond_dihedrals(parameter_set, center_bond, topology_cache=topology_cache)
    if not dihedrals:
        raise ValueError(f"No proper dihedrals found for center bond {normalize_center_bond(center_bond)}.")
    return dihedrals[0]


def center_bond_group_atoms(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> tuple[int, ...]:
    atoms = {
        atom
        for dihedral in center_bond_dihedrals(parameter_set, center_bond, topology_cache=topology_cache)
        for atom in dihedral.atoms
    }
    return tuple(sorted(atoms))


def is_ring_center_bond(parameter_set: CorrectionParameterSet, center_bond: tuple[int, int]) -> bool:
    start, end = normalize_center_bond(center_bond)
    adjacency = parameter_set.mol2.adjacency
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


def enumerate_fittable_center_bonds(
    parameter_set: CorrectionParameterSet,
    topology_cache=None,
    include_ring: bool = False,
) -> list[tuple[int, int]]:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    center_bonds: list[tuple[int, int]] = []
    for center_bond in sorted(cache.proper_by_center_bond):
        if not cache.proper_by_center_bond[center_bond]:
            continue
        if not include_ring and is_ring_center_bond(parameter_set, center_bond):
            continue
        center_bonds.append(center_bond)
    return center_bonds


def resolve_torsion_center_bonds(
    parameter_set: CorrectionParameterSet,
    params: TorsionFitParams,
    topology_cache=None,
) -> tuple[list[tuple[int, int]], list[str]]:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    warnings: list[str] = []
    if params.center_bonds is None:
        return enumerate_fittable_center_bonds(parameter_set, topology_cache=cache, include_ring=False), warnings

    center_bonds = [normalize_center_bond(bond) for bond in params.center_bonds]
    for center_bond in center_bonds:
        if is_ring_center_bond(parameter_set, center_bond):
            warnings.append(
                f"Explicit torsion center bond {center_bond} is ring-internal; running anyway."
            )
    return center_bonds, warnings


# -----------------------------------------------------------------------------
# Scan I/O and relative-energy alignment
# -----------------------------------------------------------------------------

def _parse_scan_comment(comment: str) -> tuple[float, float]:
    coords = comment.split("[", 1)[1].split("]", 1)[0]
    angle_deg = float(coords.split(",", 1)[0].strip())
    energy_hartree = float(comment.split("Energy =", 1)[1].split()[0])
    return angle_deg, energy_hartree


def _relative_to_reference(values: np.ndarray, ref_idx: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - values[ref_idx]


def read_scan_xyz(path: str) -> dict:
    angles_deg: list[float] = []
    qm_hartree: list[float] = []
    qm_kcal: list[float] = []
    frames: list[Atoms] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            natoms_line = handle.readline()
            if natoms_line == "":
                break
            if not natoms_line.strip():
                continue

            natoms = int(natoms_line.strip())
            angle_deg, energy_hartree = _parse_scan_comment(handle.readline())

            symbols: list[str] = []
            positions: list[tuple[float, float, float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                symbols.append(parts[0])
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))

            angles_deg.append(angle_deg)
            qm_hartree.append(energy_hartree)
            qm_kcal.append(energy_hartree * HARTREE_TO_KCAL_MOL)
            frames.append(Atoms(symbols=symbols, positions=positions))

    if not frames:
        raise ValueError(f"No scan points were read from {path}")

    angles_array = np.asarray(angles_deg, dtype=float)
    qm_hartree_array = np.asarray(qm_hartree, dtype=float)
    qm_kcal_array = np.asarray(qm_kcal, dtype=float)
    ref_idx = int(np.argmin(qm_kcal_array))
    return {
        "angles_deg": angles_array,
        "qm_hartree": qm_hartree_array,
        "qm_kcal": qm_kcal_array,
        "frames": frames,
        "source_path": str(path),
        "ref_idx": ref_idx,
        "qm_rel": _relative_to_reference(qm_kcal_array, ref_idx),
    }


# -----------------------------------------------------------------------------
# Basis construction and term helpers
# -----------------------------------------------------------------------------

def _validate_fit_targets(dihedrals: list[Dihedral]) -> None:
    if not dihedrals:
        raise ValueError("No target dihedrals were selected for torsion fitting.")
    for dihedral in dihedrals:
        if not dihedral.terms:
            raise ValueError(f"Target dihedral {dihedral.atoms} has no torsion terms to fit.")


def _basis_matrix(scan_data: dict, dihedrals: list[Dihedral]) -> np.ndarray:
    frames = scan_data["frames"]
    n_points = len(frames)
    n_terms = sum(len(dihedral.terms) for dihedral in dihedrals)
    basis = np.zeros((n_points, n_terms), dtype=float)

    for point_index, atoms in enumerate(frames):
        positions = atoms.get_positions()
        column = 0
        for dihedral in dihedrals:
            phi = dihedral_radians(positions, *dihedral.atoms)
            for term in dihedral.terms:
                basis[point_index, column] = 1.0 + np.cos(term.period * phi - term.phase)
                column += 1

    return basis


def _global_basis_matrix(
    scan_data: dict,
    parameter_set: CorrectionParameterSet,
    term_paths: tuple[tuple[int, int], ...],
) -> np.ndarray:
    frames = scan_data["frames"]
    n_points = len(frames)
    n_terms = len(term_paths)
    basis = np.zeros((n_points, n_terms), dtype=float)
    dihedrals = parameter_set.dihedrals

    for point_index, atoms in enumerate(frames):
        positions = atoms.get_positions()
        phi_cache: dict[int, float] = {}
        for global_index, (dihedral_index, term_index) in enumerate(term_paths):
            dihedral = dihedrals[dihedral_index]
            if dihedral_index not in phi_cache:
                phi_cache[dihedral_index] = dihedral_radians(positions, *dihedral.atoms)
            term = dihedral.terms[term_index]
            basis[point_index, global_index] = 1.0 + np.cos(term.period * phi_cache[dihedral_index] - term.phase)

    return basis


def _clone_terms(dihedrals: list[Dihedral]) -> list[list[FourierTerm]]:
    return [[FourierTerm(term.kPhi, term.period, term.phase) for term in dihedral.terms] for dihedral in dihedrals]


def _flatten_kphi(terms: list[list[FourierTerm]]) -> np.ndarray:
    return np.asarray([term.kPhi for dihedral_terms in terms for term in dihedral_terms], dtype=float)


def _terms_from_vector(dihedrals: list[Dihedral], vector: np.ndarray) -> list[list[FourierTerm]]:
    expected = sum(len(dihedral.terms) for dihedral in dihedrals)
    if vector.size != expected:
        raise ValueError(f"Expected {expected} torsion kPhi values, got {vector.size}.")

    fitted_terms: list[list[FourierTerm]] = []
    offset = 0
    for dihedral in dihedrals:
        new_terms: list[FourierTerm] = []
        for term in dihedral.terms:
            new_terms.append(FourierTerm(kPhi=float(vector[offset]), period=term.period, phase=term.phase))
            offset += 1
        fitted_terms.append(new_terms)
    return fitted_terms

# -----------------------------------------------------------------------------
# Parameter-set rewriting
# -----------------------------------------------------------------------------

def _apply_center_bond_terms(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
) -> CorrectionParameterSet:
    center = normalize_center_bond(center_bond)
    target_indices = _center_bond_dihedral_indices(parameter_set, center)
    if len(target_indices) != len(fitted_terms):
        raise ValueError(
            "The supplied parameter set does not match the fitted torsion terms for the requested center bond."
        )

    dihedrals = list(parameter_set.dihedrals)
    for fit_index, dihedral_index in enumerate(target_indices):
        dihedral = parameter_set.dihedrals[dihedral_index]
        dihedrals[dihedral_index] = Dihedral(
            atoms=dihedral.atoms,
            atom_types=dihedral.atom_types,
            terms=[FourierTerm(term.kPhi, term.period, term.phase) for term in fitted_terms[fit_index]],
        )

    return CorrectionParameterSet(
        mol2=parameter_set.mol2,
        frcmod=parameter_set.frcmod,
        bonds=parameter_set.bonds,
        angles=parameter_set.angles,
        dihedrals=dihedrals,
        impropers=parameter_set.impropers,
        nonbonds=parameter_set.nonbonds,
        unmatched_bonds=parameter_set.unmatched_bonds,
        unmatched_angles=parameter_set.unmatched_angles,
        unmatched_dihedrals=parameter_set.unmatched_dihedrals,
        unmatched_impropers=parameter_set.unmatched_impropers,
        unmatched_nonbonds=parameter_set.unmatched_nonbonds,
    )


def apply_center_bond_terms(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
) -> CorrectionParameterSet:
    return _apply_center_bond_terms(parameter_set, center_bond, fitted_terms)


def _parameter_set_from_delta(problem: dict, delta_vector: np.ndarray) -> CorrectionParameterSet:
    term_paths = problem["term_paths"]
    stage0_parameter_set = problem["stage0_parameter_set"]
    if delta_vector.size != len(term_paths):
        raise ValueError(f"Expected {len(term_paths)} global torsion deltas, got {delta_vector.size}.")

    replacements: dict[int, list[FourierTerm]] = {}
    for global_index, (dihedral_index, term_index) in enumerate(term_paths):
        if dihedral_index not in replacements:
            dihedral = stage0_parameter_set.dihedrals[dihedral_index]
            replacements[dihedral_index] = [FourierTerm(term.kPhi, term.period, term.phase) for term in dihedral.terms]
        original_term = stage0_parameter_set.dihedrals[dihedral_index].terms[term_index]
        replacements[dihedral_index][term_index] = FourierTerm(
            kPhi=float(original_term.kPhi + delta_vector[global_index]),
            period=original_term.period,
            phase=original_term.phase,
        )

    dihedrals = list(stage0_parameter_set.dihedrals)
    for dihedral_index, terms in replacements.items():
        dihedral = stage0_parameter_set.dihedrals[dihedral_index]
        dihedrals[dihedral_index] = Dihedral(
            atoms=dihedral.atoms,
            atom_types=dihedral.atom_types,
            terms=terms,
        )

    return CorrectionParameterSet(
        mol2=stage0_parameter_set.mol2,
        frcmod=stage0_parameter_set.frcmod,
        bonds=stage0_parameter_set.bonds,
        angles=stage0_parameter_set.angles,
        dihedrals=dihedrals,
        impropers=stage0_parameter_set.impropers,
        nonbonds=stage0_parameter_set.nonbonds,
        unmatched_bonds=stage0_parameter_set.unmatched_bonds,
        unmatched_angles=stage0_parameter_set.unmatched_angles,
        unmatched_dihedrals=stage0_parameter_set.unmatched_dihedrals,
        unmatched_impropers=stage0_parameter_set.unmatched_impropers,
        unmatched_nonbonds=stage0_parameter_set.unmatched_nonbonds,
    )


def apply_global_delta(problem: dict, delta_vector: np.ndarray) -> CorrectionParameterSet:
    return _parameter_set_from_delta(problem, np.asarray(delta_vector, dtype=float))


# -----------------------------------------------------------------------------
# Stage 1 local initializer
# -----------------------------------------------------------------------------

def _solve_local_delta(
    basis: np.ndarray,
    residual: np.ndarray,
    k_orig: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    target = residual - basis @ k_orig
    reg_diag = (_PRIOR_L2_WEIGHT / max(scales.size, 1)) / (scales ** 2)
    lhs = basis.T @ basis + np.diag(reg_diag)
    rhs = basis.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def evaluate_refit_objective(
    scan_data: dict,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
    topology_cache=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    updated_parameter_set = _apply_center_bond_terms(parameter_set, center_bond, fitted_terms)
    qm_rel = np.asarray(scan_data["qm_rel"], dtype=float)
    mm_refit_total = np.asarray(
        [evaluate_mm_energy(atoms, updated_parameter_set, topology_cache=cache).total for atoms in scan_data["frames"]],
        dtype=float,
    )
    mm_refit_rel = _relative_to_reference(mm_refit_total, int(scan_data["ref_idx"]))
    residual_after = qm_rel - mm_refit_rel
    rmse = float(np.sqrt(np.mean(residual_after ** 2)))
    return rmse, mm_refit_rel, residual_after


def fit_torsion_scan(
    scan_data: dict,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> dict:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    center = normalize_center_bond(center_bond)
    target_dihedrals = center_bond_dihedrals(parameter_set, center, topology_cache=cache)
    if not target_dihedrals:
        raise ValueError(f"No proper dihedrals found for center bond {center}.")

    _validate_fit_targets(target_dihedrals)
    basis = _basis_matrix(scan_data, target_dihedrals)
    representative_dihedral = target_dihedrals[0].atoms
    qm_rel = np.asarray(scan_data["qm_rel"], dtype=float)

    orig_mm_total = np.asarray(
        [evaluate_mm_energy(atoms, parameter_set, topology_cache=cache).total for atoms in scan_data["frames"]],
        dtype=float,
    )
    orig_mm_rel = _relative_to_reference(orig_mm_total, int(scan_data["ref_idx"]))

    mm_zeroed_total = np.asarray(
        [
            evaluate_mm_energy(
                atoms,
                parameter_set,
                topology_cache=cache,
                zero_proper_center_bond=center,
            ).total
            for atoms in scan_data["frames"]
        ],
        dtype=float,
    )
    mm_zeroed_rel = _relative_to_reference(mm_zeroed_total, int(scan_data["ref_idx"]))
    residual = qm_rel - mm_zeroed_rel

    original_terms = _clone_terms(target_dihedrals)
    k_orig = _flatten_kphi(original_terms)
    delta_kphi = _solve_local_delta(basis, residual, k_orig, np.maximum(1.0, np.abs(k_orig)))
    fitted_vector = k_orig + delta_kphi
    fitted_terms = _terms_from_vector(target_dihedrals, fitted_vector)
    residual_before = (residual - basis @ k_orig).copy()
    rmse, mm_refit_rel, residual_after = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center,
        fitted_terms,
        topology_cache=cache,
    )
    torsion_fit = mm_refit_rel - orig_mm_rel
    result = {
        "center_bond": center,
        "target_dihedrals": [deepcopy(dihedral) for dihedral in target_dihedrals],
        "original_terms": original_terms,
        "fitted_terms": fitted_terms,
        "delta_kphi": delta_kphi.copy(),
        "residual_before": residual_before,
        "angles_deg": np.asarray(scan_data["angles_deg"], dtype=float).copy(),
        "qm_rel": qm_rel.copy(),
        "orig_mm_rel": orig_mm_rel.copy(),
        "mm_zeroed_rel": mm_zeroed_rel.copy(),
        "torsion_fit": torsion_fit.copy(),
        "mm_refit_rel": mm_refit_rel.copy(),
        "residual_after": residual_after.copy(),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(residual_after))),
        "max_abs_error": float(np.max(np.abs(residual_after))),
        "representative_dihedral": representative_dihedral,
        "scan_source_path": scan_data["source_path"],
    }
    result["report_lines"] = format_torsion_fit_report(result)
    return result


def fit_scan_xyz_to_center_bond(
    scan_xyz_path: str,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> dict:
    scan_data = read_scan_xyz(scan_xyz_path)
    return fit_torsion_scan(scan_data, parameter_set, center_bond, topology_cache=topology_cache)


def apply_fitted_torsion(result: dict, parameter_set: CorrectionParameterSet) -> CorrectionParameterSet:
    return _apply_center_bond_terms(parameter_set, result["center_bond"], result["fitted_terms"])


# -----------------------------------------------------------------------------
# Stage 2 aggregate objective and analytic gradient
# -----------------------------------------------------------------------------

def build_global_torsion_problem(
    stage0_parameter_set: CorrectionParameterSet,
    center_bonds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    scan_map: dict[tuple[int, int], dict],
    topology_cache=None,
) -> dict:
    normalized_center_bonds = tuple(normalize_center_bond(center_bond) for center_bond in center_bonds)
    term_paths: list[tuple[int, int]] = []
    block_slices: dict[tuple[int, int], tuple[int, int]] = {}
    offset = 0

    for center_bond in normalized_center_bonds:
        if center_bond not in scan_map:
            raise ValueError(f"No fixed scan data was supplied for center bond {center_bond}.")
        dihedral_indices = _center_bond_dihedral_indices(stage0_parameter_set, center_bond)
        if not dihedral_indices:
            raise ValueError(f"No proper torsions were found for center bond {center_bond} in the stage-0 parameter set.")

        start = offset
        for dihedral_index in dihedral_indices:
            dihedral = stage0_parameter_set.dihedrals[dihedral_index]
            if not dihedral.terms:
                raise ValueError(f"Target dihedral {dihedral.atoms} has no torsion terms to fit.")
            for term_index in range(len(dihedral.terms)):
                term_paths.append((dihedral_index, term_index))
                offset += 1
        block_slices[center_bond] = (start, offset)

    k_orig = np.asarray(
        [stage0_parameter_set.dihedrals[dihedral_index].terms[term_index].kPhi for dihedral_index, term_index in term_paths],
        dtype=float,
    )
    scales = np.maximum(1.0, np.abs(k_orig))
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(stage0_parameter_set)
    qm_rel_map = {center_bond: np.asarray(scan_map[center_bond]["qm_rel"], dtype=float) for center_bond in normalized_center_bonds}
    base_mm_rel_map: dict[tuple[int, int], np.ndarray] = {}
    centered_basis_map: dict[tuple[int, int], np.ndarray] = {}

    for center_bond in normalized_center_bonds:
        scan_data = scan_map[center_bond]
        base_mm_total = np.asarray(
            [evaluate_mm_energy(atoms, stage0_parameter_set, topology_cache=cache).total for atoms in scan_data["frames"]],
            dtype=float,
        )
        basis = _global_basis_matrix(scan_data, stage0_parameter_set, tuple(term_paths))
        ref_idx = int(scan_data["ref_idx"])
        base_mm_rel_map[center_bond] = _relative_to_reference(base_mm_total, ref_idx)
        centered_basis_map[center_bond] = basis - basis[ref_idx]

    return {
        "stage0_parameter_set": stage0_parameter_set,
        "center_bonds": normalized_center_bonds,
        "scan_map": {center_bond: scan_map[center_bond] for center_bond in normalized_center_bonds},
        "term_paths": tuple(term_paths),
        "block_slices": block_slices,
        "k_orig": k_orig,
        "scales": scales,
        "qm_rel_map": qm_rel_map,
        "base_mm_rel_map": base_mm_rel_map,
        "centered_basis_map": centered_basis_map,
    }


def extract_global_delta(problem: dict, parameter_set: CorrectionParameterSet) -> np.ndarray:
    delta = np.zeros(len(problem["term_paths"]), dtype=float)
    stage0_parameter_set = problem["stage0_parameter_set"]
    for global_index, (dihedral_index, term_index) in enumerate(problem["term_paths"]):
        delta[global_index] = (
            parameter_set.dihedrals[dihedral_index].terms[term_index].kPhi
            - stage0_parameter_set.dihedrals[dihedral_index].terms[term_index].kPhi
        )
    return delta


def _terms_for_center_bond_from_delta(
    problem: dict,
    delta_vector: np.ndarray,
    center_bond: tuple[int, int],
) -> list[list[FourierTerm]]:
    start, end = problem["block_slices"][normalize_center_bond(center_bond)]
    center_terms: list[list[FourierTerm]] = []
    current_dihedral_index: int | None = None
    current_terms: list[FourierTerm] = []
    stage0_parameter_set = problem["stage0_parameter_set"]
    for global_index in range(start, end):
        dihedral_index, term_index = problem["term_paths"][global_index]
        dihedral = stage0_parameter_set.dihedrals[dihedral_index]
        original_term = dihedral.terms[term_index]
        new_term = FourierTerm(
            kPhi=float(original_term.kPhi + delta_vector[global_index]),
            period=original_term.period,
            phase=original_term.phase,
        )
        if current_dihedral_index is None:
            current_dihedral_index = dihedral_index
        if dihedral_index != current_dihedral_index:
            center_terms.append(current_terms)
            current_terms = []
            current_dihedral_index = dihedral_index
        current_terms.append(new_term)
    if current_terms:
        center_terms.append(current_terms)
    return center_terms


def evaluate_global_refit_objective(problem: dict, delta_vector: np.ndarray) -> dict:
    delta = np.asarray(delta_vector, dtype=float)
    per_scan_rmse: dict[tuple[int, int], float] = {}
    per_scan_mse: list[float] = []
    for center_bond in problem["center_bonds"]:
        qm_rel = problem["qm_rel_map"][center_bond]
        mm_rel = problem["base_mm_rel_map"][center_bond] + problem["centered_basis_map"][center_bond] @ delta
        residual = qm_rel - mm_rel
        mse = float(np.mean(residual ** 2))
        per_scan_mse.append(mse)
        per_scan_rmse[center_bond] = float(np.sqrt(mse))

    data_loss = float(np.mean(per_scan_mse)) if per_scan_mse else 0.0
    prior_loss = float(_PRIOR_L2_WEIGHT * np.mean((delta / problem["scales"]) ** 2)) if delta.size else 0.0
    return {
        "total_loss": data_loss + prior_loss,
        "data_loss": data_loss,
        "prior_loss": prior_loss,
        "global_rmse": float(np.sqrt(data_loss)) if data_loss > 0.0 else 0.0,
        "per_scan_rmse": per_scan_rmse,
    }


def _global_objective_gradient(problem: dict, delta_vector: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_vector, dtype=float)
    gradient = np.zeros_like(delta)

    n_scans = max(len(problem["center_bonds"]), 1)
    for center_bond in problem["center_bonds"]:
        basis = problem["centered_basis_map"][center_bond]
        mm_rel = problem["base_mm_rel_map"][center_bond] + basis @ delta
        residual = mm_rel - problem["qm_rel_map"][center_bond]
        gradient += (2.0 / (n_scans * max(basis.shape[0], 1))) * (basis.T @ residual)

    if delta.size:
        gradient += (2.0 * _PRIOR_L2_WEIGHT / delta.size) * (delta / (problem["scales"] ** 2))

    return gradient


def _block_gradient(problem: dict, delta_vector: np.ndarray, start: int, end: int) -> np.ndarray:
    gradient = _global_objective_gradient(problem, delta_vector)
    if gradient.size == 0:
        return np.zeros(end - start, dtype=float)
    return gradient[start:end]


def _block_scales(problem: dict, start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.ones(0, dtype=float)
    return np.maximum(1.0, np.abs(problem["k_orig"][start:end]))


def _initial_block_step(problem: dict, start: int, end: int) -> float:
    scales = _block_scales(problem, start, end)
    if scales.size == 0:
        return 1.0
    return min(1.0, 1.0 / float(np.max(scales)))


def _block_accepts(before_eval: dict, after_eval: dict, tol: float) -> bool:
    return after_eval["total_loss"] + tol ** 2 < before_eval["total_loss"]


def _linesearch_step(
    problem: dict,
    trial_delta: np.ndarray,
    direction: np.ndarray,
    start: int,
    end: int,
    current_eval: dict,
    directional_derivative: float,
) -> tuple[np.ndarray, dict, bool, float]:
    step_length = _initial_block_step(problem, start, end)
    while step_length >= _MIN_LINESEARCH_STEP:
        candidate_delta = trial_delta.copy()
        candidate_delta[start:end] += step_length * direction
        candidate_eval = evaluate_global_refit_objective(problem, candidate_delta)
        if candidate_eval["total_loss"] <= current_eval["total_loss"] + _ARMIJO_C1 * step_length * directional_derivative:
            return candidate_delta, candidate_eval, True, step_length
        step_length *= _BACKTRACK_SCALE
    return trial_delta, current_eval, False, 0.0


def _optimize_block(
    problem: dict,
    block_before_delta: np.ndarray,
    start: int,
    end: int,
    max_block_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict, int]:
    trial_delta = block_before_delta.copy()
    current_eval = evaluate_global_refit_objective(problem, trial_delta)
    iterations = 0

    for _ in range(max_block_iter):
        gradient = _block_gradient(problem, trial_delta, start, end)
        if not np.all(np.isfinite(gradient)):
            break
        if float(np.linalg.norm(gradient)) <= _GRADIENT_TOL:
            break

        direction = -gradient
        directional_derivative = float(np.dot(gradient, direction))
        previous_eval = current_eval
        trial_delta, current_eval, accepted_step, _ = _linesearch_step(
            problem,
            trial_delta,
            direction,
            start,
            end,
            current_eval,
            directional_derivative,
        )
        if not accepted_step:
            break
        iterations += 1
        if (previous_eval["global_rmse"] - current_eval["global_rmse"]) <= tol:
            break

    return trial_delta, current_eval, iterations


def refine_torsion_scans_global(
    problem: dict,
    delta_init: np.ndarray,
    max_sweeps: int,
    max_block_iter: int,
    tol: float,
) -> tuple[np.ndarray, list[dict]]:
    current_delta = np.asarray(delta_init, dtype=float).copy()
    cycles: list[dict] = []

    for cycle in range(1, max_sweeps + 1):
        cycle_before = evaluate_global_refit_objective(problem, current_delta)
        block_reports: list[dict] = []
        accepted_any = False

        for center_bond in problem["center_bonds"]:
            start, end = problem["block_slices"][center_bond]
            block_before_delta = current_delta.copy()
            block_before_eval = evaluate_global_refit_objective(problem, block_before_delta)
            terms_before = _terms_for_center_bond_from_delta(problem, block_before_delta, center_bond)
            trial_delta, current_eval, iterations = _optimize_block(
                problem,
                block_before_delta,
                start,
                end,
                max_block_iter,
                tol,
            )

            accepted = _block_accepts(block_before_eval, current_eval, tol)
            if accepted:
                current_delta = trial_delta
                accepted_any = True
                block_after_eval = current_eval
                terms_after = _terms_for_center_bond_from_delta(problem, current_delta, center_bond)
            else:
                block_after_eval = block_before_eval
                terms_after = terms_before

            block_reports.append(
                {
                    "center_bond": center_bond,
                    "accepted": accepted,
                    "iterations": iterations,
                    "rmse_before": block_before_eval["global_rmse"],
                    "rmse_after": block_after_eval["global_rmse"],
                    "total_loss_before": block_before_eval["total_loss"],
                    "total_loss_after": block_after_eval["total_loss"],
                    "terms_before": terms_before,
                    "terms_after": terms_after,
                    "per_scan_rmse_before": block_before_eval["per_scan_rmse"],
                    "per_scan_rmse_after": block_after_eval["per_scan_rmse"],
                }
            )

        cycle_after = evaluate_global_refit_objective(problem, current_delta)
        accepted_blocks = sum(1 for result in block_reports if result["accepted"])
        cycle_report_lines = [
            f"\ncycle {cycle}: accepted={accepted_blocks} rejected={len(block_reports) - accepted_blocks}  "
            f"objective {cycle_before['total_loss']:.6f} -> {cycle_after['total_loss']:.6f}  "
            f"global RMSE {cycle_before['global_rmse']:.6f} -> {cycle_after['global_rmse']:.6f}\n"
        ]
        for result in block_reports:
            state = "accepted" if result["accepted"] else "rejected"
            cycle_report_lines.append(
                f"  center bond {result['center_bond']}: objective {result['total_loss_before']:.6f} -> {result['total_loss_after']:.6f}  "
                f"global RMSE {result['rmse_before']:.6f} -> {result['rmse_after']:.6f}  "
                f"({state}, iterations={result['iterations']})\n"
            )
            for proper_index, (old_terms, new_terms) in enumerate(zip(result["terms_before"], result["terms_after"]), start=1):
                for term_index, (old_term, new_term) in enumerate(zip(old_terms, new_terms), start=1):
                    cycle_report_lines.append(
                        "    "
                        f"{proper_index:>2d}.{term_index}: "
                        f"kPhi {old_term.kPhi:.6f} -> {new_term.kPhi:.6f}  "
                        f"n={new_term.period:.3f}  phase={degrees(new_term.phase):.3f}\n"
                    )
        cycle_report_lines.append("  per-scan RMSE:\n")
        for center_bond in cycle_before["per_scan_rmse"]:
            cycle_report_lines.append(
                f"    {center_bond}: {cycle_before['per_scan_rmse'][center_bond]:.6f} -> "
                f"{cycle_after['per_scan_rmse'][center_bond]:.6f}\n"
            )

        cycles.append(
            {
                "cycle": cycle,
                "total_loss_before": cycle_before["total_loss"],
                "total_loss_after": cycle_after["total_loss"],
                "global_rmse_before": cycle_before["global_rmse"],
                "global_rmse_after": cycle_after["global_rmse"],
                "accepted_blocks": accepted_blocks,
                "rejected_blocks": len(block_reports) - accepted_blocks,
                "per_scan_rmse_before": cycle_before["per_scan_rmse"],
                "per_scan_rmse_after": cycle_after["per_scan_rmse"],
                "block_reports": block_reports,
                "report_lines": cycle_report_lines,
            }
        )
        if not accepted_any:
            break

    return current_delta, cycles


def refine_torsion_scan(
    scan_data: dict,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    cycle: int,
    topology_cache=None,
    max_iter: int = 30,
    tol: float = 1.0e-4,
) -> dict:
    center = normalize_center_bond(center_bond)
    problem = build_global_torsion_problem(
        parameter_set,
        [center],
        {center: scan_data},
        topology_cache=topology_cache,
    )
    _, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(len(problem["term_paths"]), dtype=float),
        max_sweeps=max(cycle, 1),
        max_block_iter=max_iter,
        tol=tol,
    )
    if not cycles:
        zero_delta = np.zeros(len(problem["term_paths"]), dtype=float)
        terms = _terms_for_center_bond_from_delta(problem, zero_delta, center)
        base_eval = evaluate_global_refit_objective(problem, zero_delta)
        return {
            "cycle": cycle,
            "center_bond": center,
            "accepted": False,
            "iterations": 0,
            "rmse_before": base_eval["global_rmse"],
            "rmse_after": base_eval["global_rmse"],
            "total_loss_before": base_eval["total_loss"],
            "total_loss_after": base_eval["total_loss"],
            "terms_before": terms,
            "terms_after": terms,
            "per_scan_rmse_before": base_eval["per_scan_rmse"],
            "per_scan_rmse_after": base_eval["per_scan_rmse"],
            "report_lines": [],
        }
    return cycles[-1]["block_reports"][0]


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def format_torsion_fit_report(result: dict) -> list[str]:
    lines = [
        "\n",
        "=" * 92 + "\n",
        "Parmfit Torsion Scan Fit".center(92) + "\n",
        "=" * 92 + "\n",
        f"Center bond: {result['center_bond']}\n",
        f"Representative dihedral: {result['representative_dihedral']}\n",
        f"Scan xyz: {result['scan_source_path']}\n",
        f"Target proper count: {len(result['target_dihedrals'])}\n",
        "\n",
        "Target proper order:\n",
    ]

    for index, dihedral in enumerate(result["target_dihedrals"], start=1):
        lines.append(f"  {index:>2d}. atoms={dihedral.atoms} types={dihedral.atom_types}\n")

    lines.append("\nStage-1 delta initializer:\n")
    delta_str = ", ".join(f"{value:.6f}" for value in result["delta_kphi"])
    lines.append(f"  group={tuple(range(1, len(result['target_dihedrals']) + 1))} delta_kPhi=[{delta_str}]\n")

    lines.append("\nStage-1 fitted kPhi per proper:\n")
    for index, (old_terms, new_terms) in enumerate(zip(result["original_terms"], result["fitted_terms"]), start=1):
        for term_index, (old_term, new_term) in enumerate(zip(old_terms, new_terms), start=1):
            lines.append(
                "  "
                f"{index:>2d}.{term_index}: "
                f"kPhi {old_term.kPhi:.6f} -> {new_term.kPhi:.6f}  "
                f"n={new_term.period:.3f}  phase={degrees(new_term.phase):.3f}\n"
            )

    lines.extend(
        [
            "\n",
            "Point table:\n",
            " angle_deg      QM_rel      orig_MM    MM_zeroed_rel    torsion_fit    MM_refit_rel    residual_after\n",
        ]
    )
    for angle_deg, qm_rel, orig_mm_rel, mm_zeroed_rel, torsion_fit, mm_refit_rel, residual_after in zip(
        result["angles_deg"],
        result["qm_rel"],
        result["orig_mm_rel"],
        result["mm_zeroed_rel"],
        result["torsion_fit"],
        result["mm_refit_rel"],
        result["residual_after"],
    ):
        lines.append(
            f"{float(angle_deg):10.4f}  "
            f"{float(qm_rel):10.6f}  "
            f"{float(orig_mm_rel):10.6f}  "
            f"{float(mm_zeroed_rel):14.6f}  "
            f"{float(torsion_fit):12.6f}  "
            f"{float(mm_refit_rel):12.6f}  "
            f"{float(residual_after):14.6f}\n"
        )

    lines.extend(
        [
            "\n",
            f"Local initializer RMSE: {result['rmse']:.6f}\n",
            f"MAE:                   {result['mae']:.6f}\n",
            f"MaxAbsError:           {result['max_abs_error']:.6f}\n",
        ]
    )
    return lines


def format_torsion_stage1_lines(
    params: TorsionFitParams,
    fit_reports: list[dict],
    warnings: list[str],
) -> list[str]:
    lines = ["\n"]
    if not params.enabled:
        lines.append("torsion state:   disabled by parmfit(torsionfit=false)\n")
        return lines
    if warnings:
        lines.append("warnings:\n")
        for warning in warnings:
            lines.append(f"  - {warning}\n")
    if not fit_reports:
        lines.append("torsion state:   no eligible center bonds were selected\n")
        return lines
    for report in fit_reports:
        lines.extend(report["report_lines"])
    return lines


def format_torsion_stage2_lines(
    params: TorsionFitParams,
    fit_reports: list[dict],
    refine_reports: list[dict],
) -> list[str]:
    lines = ["\n"]
    if not params.enabled:
        lines.append("refine state:     disabled because torsion fitting is disabled\n")
        return lines
    if params.refine_rounds <= 0:
        lines.append("refine state:     not requested\n")
        return lines
    if not fit_reports:
        lines.append("refine state:     no eligible center bonds were selected in stage 1\n")
        return lines
    if not refine_reports:
        lines.append("refine state:     no refinement cycles were executed\n")
        return lines
    for cycle in refine_reports:
        lines.extend(cycle["report_lines"])
    return lines
