"""Usage: refine shared torsion scans with global Stage2 optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..readparm import CorrectionParameterSet, FourierTerm
from .topology import apply_center_bond_terms
from .records import (
    TorsionGlobalProblem,
    TorsionLocalProblem,
    TorsionObjectiveEvaluation,
    TorsionRefineBlockReport,
    TorsionRefineCycle,
    TorsionScanData,
)
from .config import normalize_center_bond
from .basis import _build_group_spec, _group_slot_coefficient_basis, _normalize_phase_signed, build_global_torsion_problem
from .quality import (
    _PROFILE_SCALE_FLOOR,
    _profile_fit_scale,
    _profile_loss_metrics,
    _profile_scale_from_arrays,
    _rmse_from_residual,
    _robust_profile_range,
    _scan_energy_weights,
    _scan_profile_quality,
    _stable_rows_from_scan_quality,
    ScanProfileQuality,
)
from .stage1 import (
    _FITTED_TERM_MAX_K,
    _cap_k_value,
    _format_slot_label,
    _group_active_slot_labels,
    _group_slots_by_period,
    _merge_template_and_frozen_terms,
    _relative_profile,
)

_STAGE2_K_PRIOR_WEIGHT = 0.05
_STAGE2_CANCELLATION_RATIO_CAP = 10.0
_STAGE2_K_EFFICIENCY_SMALL_GAIN = 0.50
_STAGE2_K_EFFICIENCY_MEDIUM_GAIN = 1.00
_STAGE2_K_EFFICIENCY_MAX_DELTA_K = 2.00
_STAGE2_K_EFFICIENCY_MAX_K = 3.0


@dataclass(frozen=True)
class _Stage2ScanCache:
    center_bond: tuple[int, int]
    row_slice: slice
    qm_rel: np.ndarray
    constant_rel: np.ndarray
    cos_basis: np.ndarray
    sin_basis: np.ndarray
    weights: np.ndarray
    weight_sum: float
    profile_scale: float
    scale_class: str


@dataclass(frozen=True)
class _Stage2ObjectiveCache:
    center_bonds: tuple[tuple[int, int], ...]
    n_terms: int
    qm_rel: np.ndarray
    constant_rel: np.ndarray
    cos_basis: np.ndarray
    sin_basis: np.ndarray
    scan_caches: dict[tuple[int, int], _Stage2ScanCache]
    prior_weights: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True)
class _Stage2Checkpoint:
    source: str
    vector: np.ndarray
    evaluation: TorsionObjectiveEvaluation
    hard_safe: bool
    hard_reasons: tuple[str, ...]
    diagnostic_flags: tuple[str, ...]
    summary: dict[str, object]


def _global_term_count(problem: TorsionGlobalProblem) -> int:
    return len(problem.term_paths)

def _global_vector_size(problem: TorsionGlobalProblem) -> int:
    return 2 * _global_term_count(problem)

def _original_coefficients(problem: TorsionGlobalProblem) -> tuple[np.ndarray, np.ndarray]:
    k_orig = np.asarray(problem.k_orig, dtype=float)
    phase_orig = np.asarray(problem.phase_orig, dtype=float)
    return k_orig * np.cos(phase_orig), k_orig * np.sin(phase_orig)

def _project_coefficients_to_k_caps(
    cos_coeff: np.ndarray,
    sin_coeff: np.ndarray,
    k_caps: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, int]:
    projected_cos = np.asarray(cos_coeff, dtype=float).reshape(-1).copy()
    projected_sin = np.asarray(sin_coeff, dtype=float).reshape(-1).copy()
    caps = np.broadcast_to(np.asarray(k_caps, dtype=float), projected_cos.shape)
    k_values = np.hypot(projected_cos, projected_sin)
    over_cap = np.isfinite(caps) & (k_values > caps) & (k_values > 1.0e-12)
    if not np.any(over_cap):
        return projected_cos, projected_sin, 0
    scale = np.ones_like(k_values, dtype=float)
    scale[over_cap] = caps[over_cap] / k_values[over_cap]
    projected_cos *= scale
    projected_sin *= scale
    return projected_cos, projected_sin, int(np.count_nonzero(over_cap))

def _split_global_coeff_delta(problem: TorsionGlobalProblem, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_terms = _global_term_count(problem)
    values = np.asarray(vector, dtype=float).reshape(-1)
    expected = _global_vector_size(problem)
    if values.size != expected:
        raise ValueError(f"Expected {expected} coefficient deltas, got {values.size}.")
    return values[:n_terms], values[n_terms:]

def _stage2_active_indices(active_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(active_mask, dtype=bool))

def _pack_stage2_active_vector(vector: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(-1)
    active_indices = _stage2_active_indices(active_mask)
    n_terms = np.asarray(active_mask, dtype=bool).size
    expected = 2 * n_terms
    if values.size != expected:
        raise ValueError(f"Expected {expected} coefficient deltas, got {values.size}.")
    return np.concatenate((values[active_indices], values[n_terms + active_indices]))

def _scatter_stage2_active_vector(vector_base: np.ndarray, active_mask: np.ndarray, active_vector: np.ndarray) -> np.ndarray:
    full_vector = np.asarray(vector_base, dtype=float).reshape(-1).copy()
    active_indices = _stage2_active_indices(active_mask)
    n_terms = np.asarray(active_mask, dtype=bool).size
    expected_full = 2 * n_terms
    expected_active = 2 * active_indices.size
    values = np.asarray(active_vector, dtype=float).reshape(-1)
    if full_vector.size != expected_full:
        raise ValueError(f"Expected {expected_full} base coefficient deltas, got {full_vector.size}.")
    if values.size != expected_active:
        raise ValueError(f"Expected {expected_active} active coefficient deltas, got {values.size}.")
    full_vector[active_indices] = values[: active_indices.size]
    full_vector[n_terms + active_indices] = values[active_indices.size :]
    return full_vector

def _global_coefficients(problem: TorsionGlobalProblem, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_cos, delta_sin = _split_global_coeff_delta(problem, vector)
    orig_cos, orig_sin = _original_coefficients(problem)
    return orig_cos + delta_cos, orig_sin + delta_sin

def _split_global_vector(problem: TorsionGlobalProblem, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos_coeff, sin_coeff = _global_coefficients(problem, vector)
    k_values = np.hypot(cos_coeff, sin_coeff)
    phase_values = np.asarray([_normalize_phase_signed(value) for value in np.arctan2(sin_coeff, cos_coeff)], dtype=float)
    return k_values, phase_values

def _build_stage2_objective_cache(problem: TorsionGlobalProblem) -> _Stage2ObjectiveCache:
    n_terms = _global_term_count(problem)
    qm_blocks: list[np.ndarray] = []
    constant_blocks: list[np.ndarray] = []
    cos_blocks: list[np.ndarray] = []
    sin_blocks: list[np.ndarray] = []
    scan_caches: dict[tuple[int, int], _Stage2ScanCache] = {}
    offset = 0

    for center_bond in problem.center_bonds:
        qm_rel = np.asarray(problem.qm_rel_map[center_bond], dtype=float)
        constant_rel = np.asarray(problem.constant_rel_map.get(center_bond, np.zeros_like(qm_rel)), dtype=float)
        cos_basis = np.asarray(problem.centered_cos_basis_map[center_bond], dtype=float)
        sin_basis = np.asarray(problem.centered_sin_basis_map[center_bond], dtype=float)
        if cos_basis.shape != (qm_rel.size, n_terms):
            raise ValueError(f"centered cos basis for center bond {center_bond} has shape {cos_basis.shape}, expected {(qm_rel.size, n_terms)}.")
        if sin_basis.shape != (qm_rel.size, n_terms):
            raise ValueError(f"centered sin basis for center bond {center_bond} has shape {sin_basis.shape}, expected {(qm_rel.size, n_terms)}.")
        if constant_rel.shape != qm_rel.shape:
            raise ValueError(f"constant relative profile for center bond {center_bond} must match qm_rel shape.")

        target_like = qm_rel - constant_rel
        profile_scale = _profile_fit_scale(qm_rel, target_like)
        _unused_scale, scale_class = _profile_scale_from_arrays(qm_rel, target_like)
        weights = _scan_energy_weights(qm_rel)
        weight_sum = float(np.sum(weights))
        row_slice = slice(offset, offset + qm_rel.size)
        scan_caches[center_bond] = _Stage2ScanCache(
            center_bond=center_bond,
            row_slice=row_slice,
            qm_rel=qm_rel,
            constant_rel=constant_rel,
            cos_basis=cos_basis,
            sin_basis=sin_basis,
            weights=weights,
            weight_sum=weight_sum,
            profile_scale=float(profile_scale),
            scale_class=scale_class,
        )
        qm_blocks.append(qm_rel)
        constant_blocks.append(constant_rel)
        cos_blocks.append(cos_basis)
        sin_blocks.append(sin_basis)
        offset += qm_rel.size

    prior_weights = problem.prior_weights if problem.prior_weights is not None else np.ones_like(problem.scales, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)
    if prior_weights.size:
        prior_weights = prior_weights / max(float(np.mean(prior_weights)), 1.0e-12)

    return _Stage2ObjectiveCache(
        center_bonds=tuple(problem.center_bonds),
        n_terms=n_terms,
        qm_rel=np.concatenate(qm_blocks) if qm_blocks else np.zeros(0, dtype=float),
        constant_rel=np.concatenate(constant_blocks) if constant_blocks else np.zeros(0, dtype=float),
        cos_basis=np.vstack(cos_blocks) if cos_blocks else np.zeros((0, n_terms), dtype=float),
        sin_basis=np.vstack(sin_blocks) if sin_blocks else np.zeros((0, n_terms), dtype=float),
        scan_caches=scan_caches,
        prior_weights=prior_weights,
        scales=np.asarray(problem.scales, dtype=float),
    )

def _stage2_cached_mm_values(
    problem: TorsionGlobalProblem,
    vector: np.ndarray,
    cache: _Stage2ObjectiveCache,
) -> np.ndarray:
    cos_coeff, sin_coeff = _global_coefficients(problem, vector)
    return cache.constant_rel + (cache.cos_basis @ cos_coeff) + (cache.sin_basis @ sin_coeff)

def _parameter_set_from_delta(problem: TorsionGlobalProblem, delta_vector: np.ndarray) -> CorrectionParameterSet:
    k_caps = _stage2_k_caps(problem, delta_vector, np.ones(_global_term_count(problem), dtype=bool))
    capped_vector = _stage2_project_vector_to_k_caps(problem, delta_vector, k_caps)
    k_values, phase_values = _split_global_vector(problem, capped_vector)

    if problem.grouped:
        reference_parameter_set = problem.reference_parameter_set if problem.reference_parameter_set is not None else problem.stage0_parameter_set
        dihedrals = list(reference_parameter_set.dihedrals)
        for center_bond in problem.center_bonds:
            for group in problem.shared_groups_map.get(center_bond, ()):
                shared_term_map: dict[int, FourierTerm] = {}
                for slot_index, slot_was_present in zip(group.slot_indices, group.existing_slot_mask):
                    if not bool(slot_was_present) and abs(float(k_values[slot_index])) <= 1.0e-10:
                        continue
                    shared_term_map[int(problem.period_orig[slot_index])] = FourierTerm(
                        kPhi=float(k_values[slot_index]),
                        period=float(problem.period_orig[slot_index]),
                        phase=float(phase_values[slot_index]),
                    )
                for dihedral_index in group.dihedral_indices:
                    dihedral = reference_parameter_set.dihedrals[dihedral_index]
                    dihedrals[dihedral_index] = type(dihedral)(
                        atoms=dihedral.atoms,
                        atom_types=dihedral.atom_types,
                        terms=_merge_template_and_frozen_terms(dihedral.terms, shared_term_map),
                    )

        return CorrectionParameterSet(
            mol2=reference_parameter_set.mol2,
            frcmod=reference_parameter_set.frcmod,
            bonds=reference_parameter_set.bonds,
            angles=reference_parameter_set.angles,
            dihedrals=dihedrals,
            impropers=reference_parameter_set.impropers,
            nonbonds=reference_parameter_set.nonbonds,
            unmatched_bonds=reference_parameter_set.unmatched_bonds,
            unmatched_angles=reference_parameter_set.unmatched_angles,
            unmatched_dihedrals=reference_parameter_set.unmatched_dihedrals,
            unmatched_impropers=reference_parameter_set.unmatched_impropers,
            unmatched_nonbonds=reference_parameter_set.unmatched_nonbonds,
        )

    replacements: dict[int, list[FourierTerm]] = {}
    for global_index, (dihedral_index, term_index) in enumerate(problem.term_paths):
        if dihedral_index not in replacements:
            dihedral = problem.stage0_parameter_set.dihedrals[dihedral_index]
            replacements[dihedral_index] = [FourierTerm(term.kPhi, term.period, term.phase) for term in dihedral.terms]
        replacements[dihedral_index][term_index] = FourierTerm(
            kPhi=float(k_values[global_index]),
            period=float(problem.period_orig[global_index]),
            phase=float(phase_values[global_index]),
        )

    dihedrals = list(problem.stage0_parameter_set.dihedrals)
    for dihedral_index, terms in replacements.items():
        dihedral = problem.stage0_parameter_set.dihedrals[dihedral_index]
        dihedrals[dihedral_index] = type(dihedral)(
            atoms=dihedral.atoms,
            atom_types=dihedral.atom_types,
            terms=terms,
        )

    return CorrectionParameterSet(
        mol2=problem.stage0_parameter_set.mol2,
        frcmod=problem.stage0_parameter_set.frcmod,
        bonds=problem.stage0_parameter_set.bonds,
        angles=problem.stage0_parameter_set.angles,
        dihedrals=dihedrals,
        impropers=problem.stage0_parameter_set.impropers,
        nonbonds=problem.stage0_parameter_set.nonbonds,
        unmatched_bonds=problem.stage0_parameter_set.unmatched_bonds,
        unmatched_angles=problem.stage0_parameter_set.unmatched_angles,
        unmatched_dihedrals=problem.stage0_parameter_set.unmatched_dihedrals,
        unmatched_impropers=problem.stage0_parameter_set.unmatched_impropers,
        unmatched_nonbonds=problem.stage0_parameter_set.unmatched_nonbonds,
    )

def apply_global_delta(problem: TorsionGlobalProblem, delta_vector: np.ndarray) -> CorrectionParameterSet:
    return _parameter_set_from_delta(problem, np.asarray(delta_vector, dtype=float))


# -----------------------------------------------------------------------------
# Shared slot labels and scan profile metrics
# -----------------------------------------------------------------------------

def extract_global_delta(problem: TorsionGlobalProblem, parameter_set: CorrectionParameterSet) -> np.ndarray:
    n_terms = _global_term_count(problem)
    orig_cos, orig_sin = _original_coefficients(problem)
    cos_coeff = np.zeros(n_terms, dtype=float)
    sin_coeff = np.zeros(n_terms, dtype=float)
    if problem.grouped:
        for center_bond in problem.center_bonds:
            for group in problem.shared_groups_map.get(center_bond, ()):
                representative = parameter_set.dihedrals[group.dihedral_indices[0]]
                used_slots: set[int] = set()
                for term in representative.terms:
                    matching_slots = [
                        slot_index
                        for slot_index, slot_period in zip(group.slot_indices, group.slot_periods)
                        if slot_index not in used_slots and abs(float(term.period) - float(slot_period)) <= 1.0e-8
                    ]
                    if not matching_slots:
                        continue
                    best_slot = min(
                        matching_slots,
                        key=lambda slot_index: abs(_normalize_phase_signed(float(term.phase) - float(problem.phase_orig[slot_index]))),
                    )
                    used_slots.add(best_slot)
                    cos_coeff[best_slot] = float(term.kPhi) * np.cos(float(term.phase))
                    sin_coeff[best_slot] = float(term.kPhi) * np.sin(float(term.phase))
        return np.concatenate([cos_coeff - orig_cos, sin_coeff - orig_sin])
    for global_index, (dihedral_index, term_index) in enumerate(problem.term_paths):
        term = parameter_set.dihedrals[dihedral_index].terms[term_index]
        cos_coeff[global_index] = float(term.kPhi) * np.cos(float(term.phase))
        sin_coeff[global_index] = float(term.kPhi) * np.sin(float(term.phase))
    return np.concatenate([cos_coeff - orig_cos, sin_coeff - orig_sin])

def _global_mm_rel_map(
    problem: TorsionGlobalProblem,
    vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    if cache is not None:
        stacked = _stage2_cached_mm_values(problem, vector, cache)
        return {
            center_bond: stacked[scan_cache.row_slice].copy()
            for center_bond, scan_cache in cache.scan_caches.items()
        }
    cos_coeff, sin_coeff = _global_coefficients(problem, vector)
    mm_rel_map: dict[tuple[int, int], np.ndarray] = {}
    for center_bond in problem.center_bonds:
        cos_basis = np.asarray(problem.centered_cos_basis_map[center_bond], dtype=float)
        sin_basis = np.asarray(problem.centered_sin_basis_map[center_bond], dtype=float)
        mm_rel_map[center_bond] = problem.constant_rel_map[center_bond] + (cos_basis @ cos_coeff) + (sin_basis @ sin_coeff)
    return mm_rel_map

def _stage2_prior_loss(
    problem: TorsionGlobalProblem,
    delta_cos: np.ndarray,
    delta_sin: np.ndarray,
    cache: _Stage2ObjectiveCache,
) -> float:
    return (
        float(
            _STAGE2_K_PRIOR_WEIGHT
            * problem.prior_weight
            * np.mean(cache.prior_weights * (((delta_cos / cache.scales) ** 2) + ((delta_sin / cache.scales) ** 2)))
        )
        if cache.n_terms
        else 0.0
    )

def _stage2_data_evaluation_and_gradient_weights(
    stacked_mm_rel: np.ndarray,
    cache: _Stage2ObjectiveCache,
    *,
    prior_loss: float,
) -> tuple[TorsionObjectiveEvaluation, np.ndarray, np.ndarray]:
    stacked_residual = stacked_mm_rel - cache.qm_rel
    per_scan_rmse: dict[tuple[int, int], float] = {}
    per_scan_data_loss: dict[tuple[int, int], float] = {}
    bucket_by_scan: dict[tuple[int, int], str] = {}
    residual_gradient_weights = np.zeros_like(stacked_residual)
    n_scans = max(len(cache.scan_caches), 1)

    for center_bond in cache.center_bonds:
        scan_cache = cache.scan_caches[center_bond]
        mm_rel = stacked_mm_rel[scan_cache.row_slice]
        qm_rel = scan_cache.qm_rel
        residual = mm_rel - qm_rel
        loss_metrics = _profile_loss_metrics(
            qm_rel,
            mm_rel,
            profile_scale=scan_cache.profile_scale,
            weights=scan_cache.weights,
        )
        per_scan_rmse[center_bond] = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
        per_scan_data_loss[center_bond] = float(loss_metrics["data_loss"])
        bucket_by_scan[center_bond] = scan_cache.scale_class
        if scan_cache.weight_sum > 0.0:
            residual_gradient_weights[scan_cache.row_slice] = (
                scan_cache.weights
                / (float(n_scans) * scan_cache.weight_sum * max(scan_cache.profile_scale, 1.0e-12) ** 2)
            )

    # Stage-2 objective:
    #   MM_s(x) = constant_s + cos_basis_s @ (orig_cos + delta_cos)
    #                      + sin_basis_s @ (orig_sin + delta_sin)
    #   data_loss_s = sum_i w_si * (MM_si - QM_si)^2 / (sum_i w_si * scale_s^2)
    #   prior_loss = lambda * prior_weight * mean_j prior_w_j *
    #                ((delta_cos_j / scale_j)^2 + (delta_sin_j / scale_j)^2)
    #   total_loss = mean_s(data_loss_s) + prior_loss
    mean_data_loss = float(np.mean(list(per_scan_data_loss.values()))) if per_scan_data_loss else 0.0
    evaluation = TorsionObjectiveEvaluation(
        total_loss=mean_data_loss + prior_loss,
        data_loss=mean_data_loss,
        prior_loss=prior_loss,
        global_rmse=float(np.sqrt(mean_data_loss)) if mean_data_loss > 0.0 else 0.0,
        per_scan_rmse=dict(per_scan_rmse),
        per_scan_data_loss=dict(per_scan_data_loss),
        bucket_by_scan=bucket_by_scan,
        objective_kind="continuous_phase_k",
    )
    return evaluation, residual_gradient_weights, stacked_residual

def _evaluate_global_continuous_phase_objective_with_gradient(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> tuple[TorsionObjectiveEvaluation, np.ndarray]:
    cache = _build_stage2_objective_cache(problem) if cache is None else cache
    raw_vector = np.asarray(delta_vector, dtype=float).reshape(-1)
    delta_cos, delta_sin = _split_global_coeff_delta(problem, raw_vector)
    stacked_mm_rel = _stage2_cached_mm_values(problem, raw_vector, cache)
    prior_loss = _stage2_prior_loss(problem, delta_cos, delta_sin, cache)
    evaluation, residual_gradient_weights, stacked_residual = _stage2_data_evaluation_and_gradient_weights(
        stacked_mm_rel,
        cache,
        prior_loss=prior_loss,
    )
    gradient_scale = 2.0 * residual_gradient_weights * stacked_residual
    grad_cos = cache.cos_basis.T @ gradient_scale
    grad_sin = cache.sin_basis.T @ gradient_scale
    if cache.n_terms:
        prior_factor = 2.0 * _STAGE2_K_PRIOR_WEIGHT * problem.prior_weight / float(cache.n_terms)
        grad_cos = grad_cos + (prior_factor * cache.prior_weights * delta_cos / (cache.scales**2))
        grad_sin = grad_sin + (prior_factor * cache.prior_weights * delta_sin / (cache.scales**2))
    return evaluation, np.concatenate([grad_cos, grad_sin])

def _evaluate_stage2_active_objective_with_gradient(
    problem: TorsionGlobalProblem,
    active_vector: np.ndarray,
    vector_base: np.ndarray,
    active_mask: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> tuple[TorsionObjectiveEvaluation, np.ndarray]:
    cache = _build_stage2_objective_cache(problem) if cache is None else cache
    active_indices = _stage2_active_indices(active_mask)
    active_values = np.asarray(active_vector, dtype=float).reshape(-1)
    if active_values.size != 2 * active_indices.size:
        raise ValueError(f"Expected {2 * active_indices.size} active coefficient deltas, got {active_values.size}.")

    base_vector = np.asarray(vector_base, dtype=float).reshape(-1)
    base_delta_cos, base_delta_sin = _split_global_coeff_delta(problem, base_vector)
    active_delta_cos = active_values[: active_indices.size]
    active_delta_sin = active_values[active_indices.size :]
    delta_cos = base_delta_cos.copy()
    delta_sin = base_delta_sin.copy()
    delta_cos[active_indices] = active_delta_cos
    delta_sin[active_indices] = active_delta_sin

    base_mm_rel = _stage2_cached_mm_values(problem, base_vector, cache)
    stacked_mm_rel = base_mm_rel
    if active_indices.size:
        stacked_mm_rel = stacked_mm_rel + (
            cache.cos_basis[:, active_indices] @ (active_delta_cos - base_delta_cos[active_indices])
        )
        stacked_mm_rel = stacked_mm_rel + (
            cache.sin_basis[:, active_indices] @ (active_delta_sin - base_delta_sin[active_indices])
        )
    prior_loss = _stage2_prior_loss(problem, delta_cos, delta_sin, cache)
    evaluation, residual_gradient_weights, stacked_residual = _stage2_data_evaluation_and_gradient_weights(
        stacked_mm_rel,
        cache,
        prior_loss=prior_loss,
    )
    gradient_scale = 2.0 * residual_gradient_weights * stacked_residual
    grad_cos = cache.cos_basis[:, active_indices].T @ gradient_scale
    grad_sin = cache.sin_basis[:, active_indices].T @ gradient_scale
    if cache.n_terms and active_indices.size:
        prior_factor = 2.0 * _STAGE2_K_PRIOR_WEIGHT * problem.prior_weight / float(cache.n_terms)
        grad_cos = grad_cos + (
            prior_factor * cache.prior_weights[active_indices] * active_delta_cos / (cache.scales[active_indices] ** 2)
        )
        grad_sin = grad_sin + (
            prior_factor * cache.prior_weights[active_indices] * active_delta_sin / (cache.scales[active_indices] ** 2)
        )
    return evaluation, np.concatenate([grad_cos, grad_sin])

def _evaluate_global_continuous_phase_objective(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> TorsionObjectiveEvaluation:
    evaluation, _gradient = _evaluate_global_continuous_phase_objective_with_gradient(problem, delta_vector, cache=cache)
    return evaluation

def evaluate_global_refit_objective(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> TorsionObjectiveEvaluation:
    return _evaluate_global_continuous_phase_objective(problem, delta_vector, cache=cache)


# -----------------------------------------------------------------------------
# Stage2 guards and coefficient optimization
# -----------------------------------------------------------------------------

def _stage2_term_profiles(
    problem: TorsionGlobalProblem,
    center_bond: tuple[int, int],
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_coeff, sin_coeff = _global_coefficients(problem, vector)
    k_values, _phase_values = _split_global_vector(problem, vector)
    start, end = problem.block_slices[center_bond]
    if end <= start:
        return np.zeros((len(problem.qm_rel_map[center_bond]), 0), dtype=float), np.zeros(0, dtype=float)
    k_block = k_values[start:end]
    cos_basis = np.asarray(problem.centered_cos_basis_map[center_bond][:, start:end], dtype=float)
    sin_basis = np.asarray(problem.centered_sin_basis_map[center_bond][:, start:end], dtype=float)
    return (
        (cos_basis * cos_coeff[start:end][np.newaxis, :]) + (sin_basis * sin_coeff[start:end][np.newaxis, :]),
        k_block,
    )

def _stage2_cancellation_metrics(
    problem: TorsionGlobalProblem,
    center_bond: tuple[int, int],
    vector: np.ndarray,
) -> tuple[float, float]:
    term_profiles, k_block = _stage2_term_profiles(problem, center_bond, vector)
    active = np.flatnonzero(np.max(np.abs(term_profiles), axis=0) > 1.0e-10) if term_profiles.size else np.asarray([], dtype=int)
    max_abs_k = float(np.max(np.abs(k_block))) if k_block.size else 0.0
    if active.size <= 1:
        return 1.0, max_abs_k
    active_profiles = term_profiles[:, active]
    individual_span = float(sum(_robust_profile_range(active_profiles[:, index]) for index in range(active_profiles.shape[1])))
    net_span = _robust_profile_range(np.sum(active_profiles, axis=1))
    return individual_span / max(net_span, _PROFILE_SCALE_FLOOR), max_abs_k

def _stage2_active_slot_labels(problem: TorsionGlobalProblem, center_bond: tuple[int, int], vector: np.ndarray) -> list[str]:
    k_values, phase_values = _split_global_vector(problem, vector)
    start, end = problem.block_slices[center_bond]
    labels: list[str] = []
    for global_index in range(start, end):
        if abs(float(k_values[global_index])) <= 1.0e-10:
            continue
        labels.append(_format_slot_label(problem.period_orig[global_index], phase_values[global_index]))
    return labels

def _stage2_candidate_guard(
    problem: TorsionGlobalProblem,
    before_vector: np.ndarray,
    candidate_vector: np.ndarray,
    tol: float,
    *,
    cache: _Stage2ObjectiveCache | None = None,
    scan_quality_by_center: dict[tuple[int, int], ScanProfileQuality] | None = None,
) -> tuple[bool, dict[str, object]]:
    before_profiles = _global_mm_rel_map(problem, before_vector, cache=cache)
    candidate_profiles = _global_mm_rel_map(problem, candidate_vector, cache=cache)
    before_eval = evaluate_global_refit_objective(problem, before_vector, cache=cache)
    candidate_eval = evaluate_global_refit_objective(problem, candidate_vector, cache=cache)
    before_k_values, _before_phase_values = _split_global_vector(problem, before_vector)
    candidate_k_values, _candidate_phase_values = _split_global_vector(problem, candidate_vector)
    per_scan: dict[str, dict[str, object]] = {}
    hard_reasons: list[str] = []
    diagnostic_flags: list[str] = []

    def add_diagnostic(flag: str) -> None:
        if flag not in diagnostic_flags:
            diagnostic_flags.append(flag)

    for center_bond in problem.center_bonds:
        qm_rel = np.asarray(problem.qm_rel_map[center_bond], dtype=float)
        scan_quality = (
            scan_quality_by_center[center_bond]
            if scan_quality_by_center is not None and center_bond in scan_quality_by_center
            else _scan_profile_quality(qm_rel)
        )
        before_mm = np.asarray(before_profiles[center_bond], dtype=float)
        candidate_mm = np.asarray(candidate_profiles[center_bond], dtype=float)
        target_like = qm_rel - np.asarray(problem.constant_rel_map.get(center_bond, np.zeros_like(qm_rel)), dtype=float)
        profile_scale = _profile_fit_scale(qm_rel, target_like)
        _unused_scale, scale_class = _profile_scale_from_arrays(qm_rel, target_like)
        stable_mask = _stable_rows_from_scan_quality(qm_rel.size, scan_quality)
        before_rmse = _rmse_from_residual(before_mm - qm_rel)
        candidate_rmse = _rmse_from_residual(candidate_mm - qm_rel)
        before_stable_rmse = _rmse_from_residual(before_mm - qm_rel, stable_mask)
        candidate_stable_rmse = _rmse_from_residual(candidate_mm - qm_rel, stable_mask)
        weights = _scan_energy_weights(qm_rel)
        before_metrics = _profile_loss_metrics(qm_rel, before_mm, profile_scale=profile_scale, weights=weights)
        candidate_metrics = _profile_loss_metrics(qm_rel, candidate_mm, profile_scale=profile_scale, weights=weights)
        cancellation_ratio, max_abs_k = _stage2_cancellation_metrics(problem, center_bond, candidate_vector)
        cancellation_cap = _STAGE2_CANCELLATION_RATIO_CAP
        max_k_cap = _stage2_max_k_cap_for_center(problem, center_bond)
        data_slack = max(float(tol), 1.0e-4, 0.05 * max(before_metrics["data_loss"], 1.0e-6))
        scan_hard_reasons: list[str] = []
        scan_diagnostics: list[str] = []
        start, end = problem.block_slices[center_bond]
        max_delta_k = (
            float(np.max(np.abs(candidate_k_values[start:end] - before_k_values[start:end])))
            if end > start
            else 0.0
        )
        has_update = max_delta_k > max(float(tol), 1.0e-8)
        rmse_gain = float(before_rmse - candidate_rmse)
        stable_rmse_gain = float(before_stable_rmse - candidate_stable_rmse)
        capped_terms = _stage2_capped_term_count(problem, candidate_vector, center_bond)
        at_cap_terms = _stage2_at_cap_term_count(problem, candidate_vector, center_bond)
        raw_scan_quality = str(scan_quality.quality)
        scan_quality_label = "geometry_jump" if raw_scan_quality in {"discontinuous", "geometry_jump"} else raw_scan_quality
        scan_quality_note = "geometry_jump" if scan_quality_label == "geometry_jump" else None
        target_signal = _robust_profile_range(target_like)
        torsion_unidentifiable = target_signal < max(0.05, 0.01 * float(profile_scale))

        if scan_quality_note is not None:
            scan_diagnostics.append("geometry_jump")
            add_diagnostic("geometry_jump")
        if torsion_unidentifiable:
            scan_diagnostics.append("torsion_unidentifiable")
            add_diagnostic("torsion_unidentifiable")
        if candidate_metrics["data_loss"] > before_metrics["data_loss"] + data_slack:
            scan_diagnostics.append("data_loss_degraded")
            add_diagnostic("data_loss_degraded")
        if cancellation_ratio > cancellation_cap:
            scan_hard_reasons.append("cancellation")
        if max_abs_k > max_k_cap:
            scan_hard_reasons.append("k_too_large")
        if at_cap_terms > 0:
            scan_diagnostics.append("at_cap")
            add_diagnostic("at_cap")
        k_efficiency_failed = (
            rmse_gain < _STAGE2_K_EFFICIENCY_SMALL_GAIN and max_delta_k > _STAGE2_K_EFFICIENCY_MAX_DELTA_K
        ) or (
            rmse_gain < _STAGE2_K_EFFICIENCY_MEDIUM_GAIN and max_abs_k > _STAGE2_K_EFFICIENCY_MAX_K
        )
        if k_efficiency_failed:
            scan_diagnostics.append("k_efficiency")
            add_diagnostic("k_efficiency")
        if torsion_unidentifiable and has_update:
            scan_hard_reasons.append("torsion_unidentifiable")
        if scan_quality_note is not None and has_update:
            scan_hard_reasons.append("geometry_jump_frozen")

        if scan_hard_reasons:
            hard_reasons.extend(f"{center_bond}:{reason}" for reason in scan_hard_reasons)
        per_scan[str(center_bond)] = {
            "scale_class": scale_class,
            "scale": float(profile_scale),
            "profile_issue": scan_quality_note,
            "scan_quality": scan_quality_label,
            "scan_quality_note": scan_quality_note,
            "max_adjacent_qm_jump": float(scan_quality.max_adjacent_qm_jump),
            "active_slots": _stage2_active_slot_labels(problem, center_bond, candidate_vector),
            "reject_reason": ",".join(scan_hard_reasons) if scan_hard_reasons else None,
            "diagnostic_flags": tuple(scan_diagnostics),
            "max_abs_k": float(max_abs_k),
            "max_k_cap": float(max_k_cap),
            "capped_terms": int(capped_terms),
            "at_cap_terms": int(at_cap_terms),
            "max_delta_k": float(max_delta_k),
            "cancellation_ratio": float(cancellation_ratio),
            "rmse_before": float(before_rmse),
            "rmse_after": float(candidate_rmse),
            "rmse_gain": float(rmse_gain),
            "stable_rmse_before": float(before_stable_rmse),
            "stable_rmse_after": float(candidate_stable_rmse),
            "stable_rmse_gain": float(stable_rmse_gain),
            "data_loss_before": float(before_metrics["data_loss"]),
            "data_loss_after": float(candidate_metrics["data_loss"]),
        }

    if not np.isfinite(candidate_eval.total_loss):
        hard_reasons.append("nonfinite_objective")
    if candidate_eval.total_loss >= before_eval.total_loss - float(tol):
        add_diagnostic("no_total_loss_gain")
    if candidate_eval.data_loss > before_eval.data_loss + max(float(tol), 0.05 * max(before_eval.data_loss, 1.0)):
        add_diagnostic("data_loss_degraded")

    summary = {
        "accepted": not hard_reasons,
        "reject_reason": ",".join(hard_reasons) if hard_reasons else None,
        "hard_reasons": tuple(hard_reasons),
        "diagnostic_flags": tuple(diagnostic_flags),
        "per_scan": per_scan,
        "total_loss_before": float(before_eval.total_loss),
        "total_loss_after": float(candidate_eval.total_loss),
        "data_loss_before": float(before_eval.data_loss),
        "data_loss_after": float(candidate_eval.data_loss),
    }
    return not hard_reasons, summary

def _stage2_apply_center_fallbacks(
    problem: TorsionGlobalProblem,
    initial_vector: np.ndarray,
    candidate_vector: np.ndarray,
    *,
    tol: float,
    cache: _Stage2ObjectiveCache | None = None,
    scan_quality_by_center: dict[tuple[int, int], ScanProfileQuality] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[str, ...]]]:
    before_profiles = _global_mm_rel_map(problem, initial_vector, cache=cache)
    candidate_profiles = _global_mm_rel_map(problem, candidate_vector, cache=cache)
    final_vector = np.asarray(candidate_vector, dtype=float).reshape(-1).copy()
    n_terms = _global_term_count(problem)
    center_status: dict[str, tuple[str, ...]] = {}
    for center_bond in problem.center_bonds:
        qm_rel = np.asarray(problem.qm_rel_map[center_bond], dtype=float)
        before_mm = np.asarray(before_profiles[center_bond], dtype=float)
        candidate_mm = np.asarray(candidate_profiles[center_bond], dtype=float)
        target_like = qm_rel - np.asarray(problem.constant_rel_map.get(center_bond, np.zeros_like(qm_rel)), dtype=float)
        profile_scale = _profile_fit_scale(qm_rel, target_like)
        weights = _scan_energy_weights(qm_rel)
        before_metrics = _profile_loss_metrics(qm_rel, before_mm, profile_scale=profile_scale, weights=weights)
        candidate_metrics = _profile_loss_metrics(qm_rel, candidate_mm, profile_scale=profile_scale, weights=weights)
        scan_quality = (
            scan_quality_by_center[center_bond]
            if scan_quality_by_center is not None and center_bond in scan_quality_by_center
            else _scan_profile_quality(qm_rel)
        )
        reasons: list[str] = []
        if scan_quality.has_geometry_jump:
            reasons.append("geometry_jump_frozen")
        else:
            improvement_tol = max(float(tol), 1.0e-10)
            if candidate_metrics["data_loss"] < before_metrics["data_loss"] - improvement_tol:
                center_status[str(center_bond)] = ("accepted",)
                continue
            reasons.append("stage2_rejected")
            if candidate_metrics["data_loss"] > before_metrics["data_loss"] + improvement_tol:
                reasons.append("data_loss_degraded")
            else:
                reasons.append("no_center_residual_gain")
        if not reasons:
            continue
        start, end = problem.block_slices[center_bond]
        final_vector[start:end] = initial_vector[start:end]
        final_vector[n_terms + start : n_terms + end] = initial_vector[n_terms + start : n_terms + end]
        center_status[str(center_bond)] = tuple(reasons)
    return final_vector, center_status

def _stage2_default_active_mask(problem: TorsionGlobalProblem) -> np.ndarray:
    n_terms = _global_term_count(problem)
    if not problem.grouped:
        return np.ones(n_terms, dtype=bool)
    active_mask = np.zeros(n_terms, dtype=bool)
    for center_bond in problem.center_bonds:
        for group in problem.shared_groups_map.get(center_bond, ()):
            for period_slots in _group_slots_by_period(group).values():
                existing_slots = [
                    int(slot_index)
                    for slot_index, slot_present in zip(group.slot_indices, group.existing_slot_mask)
                    if int(slot_index) in period_slots and bool(slot_present)
                ]
                if not existing_slots:
                    continue
                selected = max(existing_slots, key=lambda slot_index: abs(float(problem.k_orig[slot_index])))
                active_mask[selected] = True
    return active_mask

def _stage2_freeze_geometry_jump_centers(
    problem: TorsionGlobalProblem,
    active_mask: np.ndarray,
    scan_quality_by_center: dict[tuple[int, int], ScanProfileQuality],
) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=bool).copy()
    for center_bond in problem.center_bonds:
        scan_quality = scan_quality_by_center.get(center_bond)
        if scan_quality is None or not scan_quality.has_geometry_jump:
            continue
        start, end = problem.block_slices[center_bond]
        mask[start:end] = False
    return mask

def _stage2_coefficient_bounds(
    problem: TorsionGlobalProblem,
    vector_init: np.ndarray,
    active_mask: np.ndarray | None = None,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> list[tuple[float, float]]:
    vector_values = np.asarray(vector_init, dtype=float).reshape(-1)
    allowed_mask = np.ones(_global_term_count(problem), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    n_terms = _global_term_count(problem)
    orig_cos, orig_sin = _original_coefficients(problem)
    bounds = [(-np.inf, np.inf) for _ in range(_global_vector_size(problem))]
    del cache
    for center_bond in problem.center_bonds:
        max_abs_k = _FITTED_TERM_MAX_K
        start, end = problem.block_slices[center_bond]
        for global_index in range(start, end):
            if not bool(allowed_mask[global_index]):
                frozen_cos_delta = float(vector_values[global_index])
                frozen_sin_delta = float(vector_values[n_terms + global_index])
                bounds[global_index] = (frozen_cos_delta, frozen_cos_delta)
                bounds[n_terms + global_index] = (frozen_sin_delta, frozen_sin_delta)
                continue
            bounds[global_index] = (
                float(-max_abs_k - orig_cos[global_index]),
                float(max_abs_k - orig_cos[global_index]),
            )
            bounds[n_terms + global_index] = (
                float(-max_abs_k - orig_sin[global_index]),
                float(max_abs_k - orig_sin[global_index]),
            )
    return bounds

def _stage2_k_caps(
    problem: TorsionGlobalProblem,
    vector_init: np.ndarray,
    active_mask: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> np.ndarray:
    del vector_init, active_mask, cache
    caps = np.full(_global_term_count(problem), _FITTED_TERM_MAX_K, dtype=float)
    return caps

def _stage2_capped_term_count(
    problem: TorsionGlobalProblem,
    vector: np.ndarray,
    center_bond: tuple[int, int] | None = None,
    k_caps: np.ndarray | None = None,
) -> int:
    k_values, _phase_values = _split_global_vector(problem, vector)
    caps = (
        _stage2_k_caps(problem, vector, np.ones(_global_term_count(problem), dtype=bool))
        if k_caps is None
        else np.asarray(k_caps, dtype=float)
    )
    if center_bond is not None:
        start, end = problem.block_slices[center_bond]
        k_values = k_values[start:end]
        caps = caps[start:end]
    return int(np.count_nonzero(np.isfinite(caps) & (k_values > caps + 1.0e-10)))

def _stage2_at_cap_term_count(
    problem: TorsionGlobalProblem,
    vector: np.ndarray,
    center_bond: tuple[int, int] | None = None,
    k_caps: np.ndarray | None = None,
) -> int:
    k_values, _phase_values = _split_global_vector(problem, vector)
    caps = (
        _stage2_k_caps(problem, vector, np.ones(_global_term_count(problem), dtype=bool))
        if k_caps is None
        else np.asarray(k_caps, dtype=float)
    )
    if center_bond is not None:
        start, end = problem.block_slices[center_bond]
        k_values = k_values[start:end]
        caps = caps[start:end]
    finite = np.isfinite(caps)
    return int(np.count_nonzero(finite & (np.abs(k_values - caps) <= 1.0e-8)))

def _stage2_max_k_cap_for_center(
    problem: TorsionGlobalProblem,
    center_bond: tuple[int, int],
    k_caps: np.ndarray | None = None,
) -> float:
    caps = (
        _stage2_k_caps(
            problem,
            np.zeros(_global_vector_size(problem), dtype=float),
            np.ones(_global_term_count(problem), dtype=bool),
        )
        if k_caps is None
        else np.asarray(k_caps, dtype=float)
    )
    start, end = problem.block_slices[center_bond]
    finite = caps[start:end][np.isfinite(caps[start:end])]
    return float(np.max(finite)) if finite.size else float("inf")

def _stage2_project_vector_to_k_caps(
    problem: TorsionGlobalProblem,
    vector: np.ndarray,
    k_caps: np.ndarray,
) -> np.ndarray:
    raw_vector = np.asarray(vector, dtype=float).reshape(-1).copy()
    delta_cos, delta_sin = _split_global_coeff_delta(problem, raw_vector)
    orig_cos, orig_sin = _original_coefficients(problem)
    cos_coeff = orig_cos + delta_cos
    sin_coeff = orig_sin + delta_sin
    cos_coeff, sin_coeff, capped_count = _project_coefficients_to_k_caps(cos_coeff, sin_coeff, k_caps)
    if capped_count == 0:
        return raw_vector
    return np.concatenate((cos_coeff - orig_cos, sin_coeff - orig_sin))

def _optimize_stage2_coefficients(
    problem: TorsionGlobalProblem,
    vector_init: np.ndarray,
    active_mask: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    cache: _Stage2ObjectiveCache | None = None,
    on_checkpoint=None,
) -> np.ndarray:
    cache = _build_stage2_objective_cache(problem) if cache is None else cache
    vector_base = np.asarray(vector_init, dtype=float).reshape(-1).copy()
    active_mask = np.asarray(active_mask, dtype=bool)
    active_indices = _stage2_active_indices(active_mask)
    if active_indices.size == 0:
        k_caps = _stage2_k_caps(problem, vector_base, active_mask, cache=cache)
        return _stage2_project_vector_to_k_caps(problem, vector_base, k_caps)

    k_caps = _stage2_k_caps(problem, vector_base, active_mask, cache=cache)
    vector_base = _stage2_project_vector_to_k_caps(problem, vector_base, k_caps)

    def objective_and_gradient(x):
        evaluation, gradient = _evaluate_stage2_active_objective_with_gradient(
            problem,
            x,
            vector_base,
            active_mask,
            cache=cache,
        )
        if on_checkpoint is not None:
            full_vector = _scatter_stage2_active_vector(vector_base, active_mask, np.asarray(x, dtype=float).reshape(-1))
            on_checkpoint("optimizer_eval", _stage2_project_vector_to_k_caps(problem, full_vector, k_caps))
        return evaluation.total_loss, gradient

    full_bounds = _stage2_coefficient_bounds(problem, vector_base, active_mask, cache=cache)
    n_terms = _global_term_count(problem)
    active_bounds = [full_bounds[index] for index in active_indices] + [full_bounds[n_terms + index] for index in active_indices]

    result = minimize(
        fun=objective_and_gradient,
        x0=_pack_stage2_active_vector(vector_base, active_mask),
        jac=True,
        method="L-BFGS-B",
        bounds=active_bounds,
        options={"maxiter": max(int(max_iter), 1), "ftol": float(tol)},
    )
    active_candidate = np.asarray(getattr(result, "x", _pack_stage2_active_vector(vector_base, active_mask)), dtype=float).reshape(-1)
    if active_candidate.size != 2 * active_indices.size:
        return vector_base
    return _stage2_project_vector_to_k_caps(
        problem,
        _scatter_stage2_active_vector(vector_base, active_mask, active_candidate),
        k_caps,
    )


# -----------------------------------------------------------------------------
# Solver seams
# -----------------------------------------------------------------------------

def _clone_term_blocks(terms: list[list[FourierTerm]]) -> list[list[FourierTerm]]:
    return [[FourierTerm(term.kPhi, term.period, term.phase) for term in block] for block in terms]

def _terms_for_center_bond_from_delta(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    center_bond: tuple[int, int],
) -> list[list[FourierTerm]]:
    k_caps = _stage2_k_caps(problem, delta_vector, np.ones(_global_term_count(problem), dtype=bool))
    capped_vector = _stage2_project_vector_to_k_caps(problem, delta_vector, k_caps)
    k_values, phase_values = _split_global_vector(problem, capped_vector)
    if problem.grouped:
        center = normalize_center_bond(center_bond)
        reference_parameter_set = problem.reference_parameter_set if problem.reference_parameter_set is not None else problem.stage0_parameter_set
        terms_by_dihedral: dict[int, list[FourierTerm]] = {}
        for group in problem.shared_groups_map.get(center, ()):
            shared_term_map: dict[int, FourierTerm] = {}
            for slot_index, slot_was_present in zip(group.slot_indices, group.existing_slot_mask):
                if not bool(slot_was_present) and abs(float(k_values[slot_index])) <= 1.0e-10:
                    continue
                shared_term_map[int(problem.period_orig[slot_index])] = FourierTerm(
                    kPhi=float(k_values[slot_index]),
                    period=float(problem.period_orig[slot_index]),
                    phase=float(phase_values[slot_index]),
                )
            for dihedral_index in group.dihedral_indices:
                original_terms = reference_parameter_set.dihedrals[dihedral_index].terms
                terms_by_dihedral[dihedral_index] = _merge_template_and_frozen_terms(original_terms, shared_term_map)
        return [
            terms_by_dihedral[dihedral_index]
            for dihedral_index in sorted(terms_by_dihedral)
        ]

    start, end = problem.block_slices[normalize_center_bond(center_bond)]
    center_terms: list[list[FourierTerm]] = []
    current_dihedral_index: int | None = None
    current_terms: list[FourierTerm] = []
    for global_index in range(start, end):
        dihedral_index, term_index = problem.term_paths[global_index]
        dihedral = problem.stage0_parameter_set.dihedrals[dihedral_index]
        new_term = FourierTerm(
            kPhi=float(k_values[global_index]),
            period=float(problem.period_orig[global_index]),
            phase=float(phase_values[global_index]),
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


# -----------------------------------------------------------------------------
# Public algorithm entry points
# -----------------------------------------------------------------------------

def refine_torsion_scans_global(
    problem: TorsionGlobalProblem,
    delta_init: np.ndarray,
    *,
    enabled: bool,
    max_block_iter: int,
    tol: float,
) -> tuple[np.ndarray, list[TorsionRefineCycle]]:
    n_terms = _global_term_count(problem)
    vector_init = np.asarray(delta_init, dtype=float).reshape(-1).copy()
    if n_terms == 0:
        return vector_init, []
    expected_size = _global_vector_size(problem)
    if vector_init.size != expected_size:
        raise ValueError(f"Expected {expected_size} coefficient deltas, got {vector_init.size}.")
    all_k_caps = _stage2_k_caps(problem, vector_init, np.ones(n_terms, dtype=bool))
    vector_init = _stage2_project_vector_to_k_caps(problem, vector_init, all_k_caps)
    if not enabled:
        return vector_init, []

    objective_cache = _build_stage2_objective_cache(problem)
    initial_vector = vector_init.copy()
    before_eval = evaluate_global_refit_objective(problem, initial_vector, cache=objective_cache)
    max_iter = max(int(max_block_iter), int(problem.global_max_iter), 1)
    scan_quality_by_center = {
        center_bond: _scan_profile_quality(problem.qm_rel_map[center_bond])
        for center_bond in problem.center_bonds
    }
    active_mask = _stage2_freeze_geometry_jump_centers(
        problem,
        _stage2_default_active_mask(problem),
        scan_quality_by_center,
    )
    checkpoints: list[_Stage2Checkpoint] = []

    def record_checkpoint(source: str, vector: np.ndarray) -> None:
        checkpoint_vector = _stage2_project_vector_to_k_caps(
            problem,
            np.asarray(vector, dtype=float).reshape(-1).copy(),
            all_k_caps,
        )
        checkpoint_eval = evaluate_global_refit_objective(problem, checkpoint_vector, cache=objective_cache)
        hard_safe, summary = _stage2_candidate_guard(
            problem,
            initial_vector,
            checkpoint_vector,
            tol=tol,
            cache=objective_cache,
            scan_quality_by_center=scan_quality_by_center,
        )
        checkpoints.append(
            _Stage2Checkpoint(
                source=source,
                vector=checkpoint_vector,
                evaluation=checkpoint_eval,
                hard_safe=hard_safe,
                hard_reasons=tuple(summary.get("hard_reasons", ())),
                diagnostic_flags=tuple(summary.get("diagnostic_flags", ())),
                summary=summary,
            )
        )

    record_checkpoint("initial", initial_vector)
    baseline_candidate = _optimize_stage2_coefficients(
        problem,
        initial_vector,
        active_mask,
        max_iter=max_iter,
        tol=tol,
        cache=objective_cache,
        on_checkpoint=record_checkpoint,
    )
    record_checkpoint("optimizer_final", baseline_candidate)
    safe_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.hard_safe]
    best_checkpoint = (
        min(safe_checkpoints, key=lambda checkpoint: checkpoint.evaluation.total_loss)
        if safe_checkpoints
        else checkpoints[0]
    )
    final_vector = _stage2_project_vector_to_k_caps(problem, best_checkpoint.vector.copy(), all_k_caps)
    final_vector, center_status = _stage2_apply_center_fallbacks(
        problem,
        initial_vector,
        final_vector,
        tol=tol,
        cache=objective_cache,
        scan_quality_by_center=scan_quality_by_center,
    )
    center_fallbacks = {
        center: reasons
        for center, reasons in center_status.items()
        if not reasons or reasons[0] != "accepted"
    }
    final_vector = _stage2_project_vector_to_k_caps(problem, final_vector, all_k_caps)
    final_eval = evaluate_global_refit_objective(problem, final_vector, cache=objective_cache)
    _final_hard_safe, final_guard_summary = _stage2_candidate_guard(
        problem,
        initial_vector,
        final_vector,
        tol=tol,
        cache=objective_cache,
        scan_quality_by_center=scan_quality_by_center,
    )
    accepted_count = sum(1 for reasons in center_status.values() if reasons and reasons[0] == "accepted")
    rejected_count = max(len(problem.center_bonds) - accepted_count, 0)
    accepted = accepted_count > 0
    hard_reject_reasons: list[str] = []
    diagnostic_flags: list[str] = []
    for checkpoint in checkpoints:
        hard_reject_reasons.extend(checkpoint.hard_reasons)
        if checkpoint.source == "initial":
            continue
        for flag in checkpoint.diagnostic_flags:
            if flag not in diagnostic_flags:
                diagnostic_flags.append(flag)
    final_guard_summary["objective_kind"] = final_eval.objective_kind
    final_guard_summary["rolled_back"] = not accepted
    final_guard_summary["accepted"] = accepted
    final_guard_summary["best_checkpoint_source"] = best_checkpoint.source
    final_guard_summary["checkpoint_count"] = len(checkpoints)
    final_guard_summary["hard_reject_count"] = sum(1 for checkpoint in checkpoints if not checkpoint.hard_safe)
    final_guard_summary["hard_reject_reasons"] = tuple(hard_reject_reasons)
    final_guard_summary["diagnostic_flags"] = tuple(diagnostic_flags)
    final_guard_summary["center_fallbacks"] = tuple(center_fallbacks)
    final_guard_summary["center_fallback_reasons"] = center_fallbacks
    final_guard_summary["center_status"] = center_status
    per_scan_summary = final_guard_summary.get("per_scan", {})
    if isinstance(per_scan_summary, dict):
        for center_bond in problem.center_bonds:
            key = str(center_bond)
            info = per_scan_summary.get(key)
            if not isinstance(info, dict):
                continue
            reasons = center_status.get(key, ("stage2_rejected",))
            status = reasons[0] if reasons else "stage2_rejected"
            info["stage2_status"] = status
            if status != "accepted":
                info["stage2_reject_reasons"] = tuple(reasons)
    if not accepted and hard_reject_reasons and not final_guard_summary.get("reject_reason"):
        final_guard_summary["reject_reason"] = ",".join(hard_reject_reasons)
    cycles = [
        TorsionRefineCycle(
            cycle=1,
            total_loss_before=before_eval.total_loss,
            total_loss_after=final_eval.total_loss,
            global_rmse_before=before_eval.global_rmse,
            global_rmse_after=final_eval.global_rmse,
            accepted_blocks=accepted_count,
            rejected_blocks=rejected_count,
            per_scan_rmse_before=dict(before_eval.per_scan_rmse),
            per_scan_rmse_after=dict(final_eval.per_scan_rmse),
            block_reports=[],
            data_loss_before=before_eval.data_loss,
            data_loss_after=final_eval.data_loss,
            bucket_by_scan=dict(final_eval.bucket_by_scan),
            diagnostics=final_guard_summary,
        )
    ]
    return final_vector, cycles

def refine_torsion_scan(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    cycle: int,
    topology_cache=None,
    max_iter: int = 30,
    tol: float = 1.0e-4,
) -> TorsionRefineBlockReport:
    center = normalize_center_bond(center_bond)
    problem = build_global_torsion_problem(
        parameter_set,
        [center],
        {center: scan_data},
        topology_cache=topology_cache,
    )
    vector_init = np.zeros(_global_vector_size(problem), dtype=float)
    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=vector_init,
        enabled=cycle > 0,
        max_block_iter=max_iter,
        tol=tol,
    )
    before_eval = evaluate_global_refit_objective(problem, vector_init)
    after_eval = evaluate_global_refit_objective(problem, vector_final)
    if not cycles:
        terms = _terms_for_center_bond_from_delta(problem, vector_init, center)
        return TorsionRefineBlockReport(
            center_bond=center,
            accepted=False,
            iterations=0,
            rmse_before=before_eval.global_rmse,
            rmse_after=before_eval.global_rmse,
            total_loss_before=before_eval.total_loss,
            total_loss_after=before_eval.total_loss,
            data_loss_before=before_eval.data_loss,
            data_loss_after=before_eval.data_loss,
            terms_before=_clone_term_blocks(terms),
            terms_after=_clone_term_blocks(terms),
            per_scan_rmse_before=dict(before_eval.per_scan_rmse),
            per_scan_rmse_after=dict(before_eval.per_scan_rmse),
        )
    terms_before = _terms_for_center_bond_from_delta(problem, vector_init, center)
    terms_after = _terms_for_center_bond_from_delta(problem, vector_final, center)
    accepted = bool(cycles[-1].accepted_blocks > 0)
    return TorsionRefineBlockReport(
        center_bond=center,
        accepted=accepted,
        iterations=int(getattr(cycles[-1], "cycle", 1)),
        rmse_before=before_eval.global_rmse,
        rmse_after=after_eval.global_rmse,
        total_loss_before=before_eval.total_loss,
        total_loss_after=after_eval.total_loss,
        data_loss_before=before_eval.data_loss,
        data_loss_after=after_eval.data_loss,
        terms_before=_clone_term_blocks(terms_before),
        terms_after=_clone_term_blocks(terms_after),
        per_scan_rmse_before=dict(before_eval.per_scan_rmse),
        per_scan_rmse_after=dict(after_eval.per_scan_rmse),
    )
