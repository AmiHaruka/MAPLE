"""Usage: refine shared torsion scans with global Stage2 optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..readparm import CorrectionParameterSet, FourierTerm
from .records import (
    TorsionGlobalProblem,
    TorsionObjectiveEvaluation,
    TorsionRefineCycle,
)
from .config import normalize_center_bond
from .basis import (
    _normalize_phase_signed,
)
from .quality import (
    _profile_fit_scale,
    _profile_loss_metrics,
    _profile_scale_from_arrays,
    _scan_energy_weights,
)
from .stage1 import (
    _FITTED_TERM_MAX_K,
    _merge_template_and_frozen_terms,
)

_STAGE2_K_PRIOR_WEIGHT = 0.05
_STAGE2_NONFINITE_LOSS = 1.0e30


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
class _Stage2ExtraTargetCache:
    label: str
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
    target_weight: float
    source_path: str


@dataclass(frozen=True)
class _Stage2ObjectiveCache:
    center_bonds: tuple[tuple[int, int], ...]
    n_terms: int
    qm_rel: np.ndarray
    constant_rel: np.ndarray
    cos_basis: np.ndarray
    sin_basis: np.ndarray
    scan_caches: dict[tuple[int, int], _Stage2ScanCache]
    extra_target_caches: tuple[_Stage2ExtraTargetCache, ...]
    prior_weights: np.ndarray
    scales: np.ndarray

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

def _global_coefficients(problem: TorsionGlobalProblem, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_cos, delta_sin = _split_global_coeff_delta(problem, vector)
    orig_cos, orig_sin = _original_coefficients(problem)
    return orig_cos + delta_cos, orig_sin + delta_sin

def _split_global_vector(problem: TorsionGlobalProblem, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos_coeff, sin_coeff = _global_coefficients(problem, vector)
    k_values = np.hypot(cos_coeff, sin_coeff)
    phase_values = np.asarray([_normalize_phase_signed(value) for value in np.arctan2(sin_coeff, cos_coeff)], dtype=float)
    return k_values, phase_values

def _pack_stage2_k_phase(problem: TorsionGlobalProblem, delta_vector: np.ndarray) -> np.ndarray:
    k_values, phase_values = _split_global_vector(problem, delta_vector)
    return np.concatenate([k_values, phase_values])

def _delta_from_stage2_k_phase(problem: TorsionGlobalProblem, k_phase_vector: np.ndarray) -> np.ndarray:
    n_terms = _global_term_count(problem)
    values = np.asarray(k_phase_vector, dtype=float).reshape(-1)
    expected = 2 * n_terms
    if values.size != expected:
        raise ValueError(f"Expected {expected} k/phase values, got {values.size}.")
    k_values = values[:n_terms]
    phase_values = values[n_terms:]
    orig_cos, orig_sin = _original_coefficients(problem)
    cos_coeff = k_values * np.cos(phase_values)
    sin_coeff = k_values * np.sin(phase_values)
    return np.concatenate([cos_coeff - orig_cos, sin_coeff - orig_sin])

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

    extra_target_caches: list[_Stage2ExtraTargetCache] = []
    for target in problem.extra_targets:
        target_weight = float(target.weight)
        if target_weight <= 0.0:
            continue
        center_bond = normalize_center_bond(target.center_bond)
        qm_rel = np.asarray(target.qm_rel, dtype=float)
        constant_rel = np.asarray(target.constant_rel, dtype=float)
        cos_basis = np.asarray(target.cos_basis, dtype=float)
        sin_basis = np.asarray(target.sin_basis, dtype=float)
        if cos_basis.shape != (qm_rel.size, n_terms):
            raise ValueError(f"extra target {target.label!r} cos basis has shape {cos_basis.shape}, expected {(qm_rel.size, n_terms)}.")
        if sin_basis.shape != (qm_rel.size, n_terms):
            raise ValueError(f"extra target {target.label!r} sin basis has shape {sin_basis.shape}, expected {(qm_rel.size, n_terms)}.")
        if constant_rel.shape != qm_rel.shape:
            raise ValueError(f"extra target {target.label!r} constant profile must match qm_rel shape.")
        target_like = qm_rel - constant_rel
        profile_scale = _profile_fit_scale(qm_rel, target_like)
        _unused_scale, scale_class = _profile_scale_from_arrays(qm_rel, target_like)
        weights = _scan_energy_weights(qm_rel)
        weight_sum = float(np.sum(weights))
        row_slice = slice(offset, offset + qm_rel.size)
        extra_target_caches.append(
            _Stage2ExtraTargetCache(
                label=str(target.label),
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
                target_weight=target_weight,
                source_path=str(target.source_path),
            )
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
        extra_target_caches=tuple(extra_target_caches),
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
    scan_data_loss = float(np.mean(list(per_scan_data_loss.values()))) if per_scan_data_loss else 0.0
    extra_data_loss = 0.0
    # Extra Loss from ensemble targets:
    for target_cache in cache.extra_target_caches:
        mm_rel = stacked_mm_rel[target_cache.row_slice]
        qm_rel = target_cache.qm_rel
        residual = mm_rel - qm_rel
        loss_metrics = _profile_loss_metrics(
            qm_rel,
            mm_rel,
            profile_scale=target_cache.profile_scale,
            weights=target_cache.weights,
        )
        weighted_loss = float(target_cache.target_weight) / len(cache.extra_target_caches) * float(loss_metrics["data_loss"])
        extra_data_loss += weighted_loss
        if target_cache.weight_sum > 0.0 and target_cache.target_weight > 0.0:
            residual_gradient_weights[target_cache.row_slice] = (
                float(target_cache.target_weight)
                / len(cache.extra_target_caches)
                * target_cache.weights
                / (target_cache.weight_sum * max(target_cache.profile_scale, 1.0e-12) ** 2)
            )

    mean_data_loss = float(scan_data_loss + extra_data_loss)
    evaluation = TorsionObjectiveEvaluation(
        total_loss=mean_data_loss + prior_loss,
        data_loss=mean_data_loss,
        prior_loss=prior_loss,
        global_rmse=float(np.sqrt(mean_data_loss)) if mean_data_loss > 0.0 else 0.0,
        per_scan_rmse=dict(per_scan_rmse),
        per_scan_data_loss=dict(per_scan_data_loss),
        bucket_by_scan=bucket_by_scan,
        objective_kind="continuous_phase_k",
        scan_data_loss=float(scan_data_loss),
        ensemble_data_loss=float(extra_data_loss),
    )
    return evaluation, residual_gradient_weights, stacked_residual

def _evaluate_stage2_delta_objective(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> TorsionObjectiveEvaluation:
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
    return evaluation

def evaluate_global_refit_objective(
    problem: TorsionGlobalProblem,
    delta_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> TorsionObjectiveEvaluation:
    return _evaluate_stage2_delta_objective(problem, delta_vector, cache=cache)

def _evaluate_stage2_k_phase_objective_with_gradient(
    problem: TorsionGlobalProblem,
    k_phase_vector: np.ndarray,
    *,
    cache: _Stage2ObjectiveCache | None = None,
) -> tuple[TorsionObjectiveEvaluation, np.ndarray]:
    cache = _build_stage2_objective_cache(problem) if cache is None else cache
    n_terms = _global_term_count(problem)
    values = np.asarray(k_phase_vector, dtype=float).reshape(-1)
    if values.size != 2 * n_terms:
        raise ValueError(f"Expected {2 * n_terms} k/phase values, got {values.size}.")
    if not np.all(np.isfinite(values)):
        evaluation = TorsionObjectiveEvaluation(
            total_loss=_STAGE2_NONFINITE_LOSS,
            data_loss=_STAGE2_NONFINITE_LOSS,
            prior_loss=0.0,
            global_rmse=float(np.sqrt(_STAGE2_NONFINITE_LOSS)),
            per_scan_rmse={},
            objective_kind="direct_k_phase",
        )
        return evaluation, np.zeros_like(values, dtype=float)

    k_values = values[:n_terms]
    phase_values = values[n_terms:]
    cos_phase = np.cos(phase_values)
    sin_phase = np.sin(phase_values)
    cos_coeff = k_values * cos_phase
    sin_coeff = k_values * sin_phase
    orig_cos, orig_sin = _original_coefficients(problem)
    delta_cos = cos_coeff - orig_cos
    delta_sin = sin_coeff - orig_sin

    stacked_mm_rel = cache.constant_rel + (cache.cos_basis @ cos_coeff) + (cache.sin_basis @ sin_coeff)
    prior_loss = _stage2_prior_loss(problem, delta_cos, delta_sin, cache)
    evaluation, residual_gradient_weights, stacked_residual = _stage2_data_evaluation_and_gradient_weights(
        stacked_mm_rel,
        cache,
        prior_loss=prior_loss,
    )
    object.__setattr__(evaluation, "objective_kind", "direct_k_phase")
    gradient_scale = 2.0 * residual_gradient_weights * stacked_residual
    grad_cos_coeff = cache.cos_basis.T @ gradient_scale
    grad_sin_coeff = cache.sin_basis.T @ gradient_scale
    if cache.n_terms:
        prior_factor = 2.0 * _STAGE2_K_PRIOR_WEIGHT * problem.prior_weight / float(cache.n_terms)
        grad_cos_coeff = grad_cos_coeff + (prior_factor * cache.prior_weights * delta_cos / (cache.scales**2))
        grad_sin_coeff = grad_sin_coeff + (prior_factor * cache.prior_weights * delta_sin / (cache.scales**2))
    grad_k = grad_cos_coeff * cos_phase + grad_sin_coeff * sin_phase
    grad_phase = (-grad_cos_coeff * k_values * sin_phase) + (grad_sin_coeff * k_values * cos_phase)
    gradient = np.concatenate([grad_k, grad_phase])
    if not np.isfinite(evaluation.total_loss) or not np.all(np.isfinite(gradient)):
        evaluation = TorsionObjectiveEvaluation(
            total_loss=_STAGE2_NONFINITE_LOSS,
            data_loss=_STAGE2_NONFINITE_LOSS,
            prior_loss=0.0,
            global_rmse=float(np.sqrt(_STAGE2_NONFINITE_LOSS)),
            per_scan_rmse={},
            objective_kind="direct_k_phase",
        )
        return evaluation, np.zeros_like(values, dtype=float)
    return evaluation, gradient

def _stage2_k_phase_bounds(problem: TorsionGlobalProblem) -> list[tuple[float | None, float | None]]:
    n_terms = _global_term_count(problem)
    k_caps = _stage2_k_caps(problem, np.zeros(2 * n_terms, dtype=float), np.ones(n_terms, dtype=bool))
    k_bounds = [(0.0, float(cap) if np.isfinite(cap) else None) for cap in k_caps]
    phase_bounds = [(None, None) for _ in range(n_terms)]
    return k_bounds + phase_bounds

def _optimize_stage2_k_phase(
    problem: TorsionGlobalProblem,
    delta_init: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    cache: _Stage2ObjectiveCache | None = None,
) -> np.ndarray:
    cache = _build_stage2_objective_cache(problem) if cache is None else cache
    n_terms = _global_term_count(problem)
    k_caps = _stage2_k_caps(problem, delta_init, np.ones(n_terms, dtype=bool), cache=cache)
    delta_base = _stage2_project_vector_to_k_caps(problem, np.asarray(delta_init, dtype=float).reshape(-1), k_caps)
    x0 = _pack_stage2_k_phase(problem, delta_base)

    def objective_and_gradient(x):
        evaluation, gradient = _evaluate_stage2_k_phase_objective_with_gradient(problem, x, cache=cache)
        return evaluation.total_loss, gradient

    result = minimize(
        fun=objective_and_gradient,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        bounds=_stage2_k_phase_bounds(problem),
        options={"maxiter": max(int(max_iter), 1), "ftol": float(tol)},
    )
    optimized = np.asarray(getattr(result, "x", x0), dtype=float).reshape(-1)
    if optimized.size != 2 * n_terms:
        return delta_base
    if not np.all(np.isfinite(optimized)):
        return delta_base
    delta_final = _delta_from_stage2_k_phase(problem, optimized)
    return _stage2_project_vector_to_k_caps(problem, delta_final, k_caps)


# -----------------------------------------------------------------------------
# K cap helpers
# -----------------------------------------------------------------------------

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
    optimized_vector = _optimize_stage2_k_phase(
        problem,
        initial_vector,
        max_iter=max_iter,
        tol=tol,
        cache=objective_cache,
    )
    optimized_eval = evaluate_global_refit_objective(problem, optimized_vector, cache=objective_cache)
    improvement_tol = max(float(tol), 1.0e-12)
    accepted = bool(
        np.isfinite(optimized_eval.total_loss)
        and optimized_eval.total_loss < before_eval.total_loss - improvement_tol
    )
    final_vector = optimized_vector if accepted else initial_vector
    final_eval = optimized_eval if accepted else before_eval
    accepted_count = len(problem.center_bonds) if accepted else 0
    rejected_count = 0 if accepted else len(problem.center_bonds)
    final_guard_summary = {
        "objective_kind": "direct_k_phase",
        "accepted": accepted,
        "status": "accepted" if accepted else "kept_stage1",
        "rolled_back": not accepted,
        "reject_reason": None if accepted else "no_total_loss_gain",
        "initial_scan_loss": float(before_eval.scan_data_loss),
        "final_scan_loss": float(final_eval.scan_data_loss),
        "initial_ensemble_loss": float(before_eval.ensemble_data_loss),
        "final_ensemble_loss": float(final_eval.ensemble_data_loss),
        "initial_prior_loss": float(before_eval.prior_loss),
        "final_prior_loss": float(final_eval.prior_loss),
        "initial_total_loss": float(before_eval.total_loss),
        "final_total_loss": float(final_eval.total_loss),
    }
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
            data_loss_before=before_eval.data_loss,
            data_loss_after=final_eval.data_loss,
            bucket_by_scan=dict(final_eval.bucket_by_scan),
            diagnostics=final_guard_summary,
        )
    ]
    return final_vector, cycles
