"""Usage: solve problem Stage1 torsion fits and assemble problem fit reports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ..mechanics import build_mm_topology_cache, dihedral_radians, evaluate_mm_energy
from ..readparm import CorrectionParameterSet, FourierTerm
from .topology import _clone_terms, shared_group_label
from .records import (
    TorsionFitCurves,
    TorsionFitMetrics,
    TorsionFitReport,
    TorsionFitTerms,
    TorsionLocalProblem,
    TorsionScanData,
    TorsionSharedGroupReport,
)
from .config import TorsionFitParams
from .basis import (
    _build_group_spec,
    _group_slot_coefficient_basis,
    _group_slot_initial_values,
    _normalize_phase_signed,
)
from .quality import (
    _PROFILE_SCALE_FLOOR,
    _profile_fit_scale,
    _scan_energy_weights,
    _profile_loss_metrics,
    _profile_scale_from_arrays,
    _robust_profile_range,
    _scan_profile_quality,
    ScanProfileQuality,
)
from .spectral import DEFAULT_SPECTRAL_PERIODS, dominant_spectral_peaks, rank_shared_group_spectral_slots

_NEW_SLOT_MAX_MAGNITUDE_SCALE = 1.5
_STAGE1_CENTER_NEW_SLOT_LIMIT = 3
_STAGE1_ABS_RMSE_GAIN_FLOOR = 0.05
_STAGE1_LLS_PRIOR_WEIGHT = 0.01
_STAGE1_CANCELLATION_RATIO_CAP = 8.0
_FITTED_TERM_MAX_K = 3.0
_METHYL_LIKE_H_TYPES = {"hc", "h1", "h2", "h3"}
_STAGE1_SPECTRAL_COHERENCE_FLOOR = 0.15


def _project_coefficients_to_k_caps(
    cos_coeff: np.ndarray,
    sin_coeff: np.ndarray,
    k_caps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    cos_out = np.asarray(cos_coeff, dtype=float).reshape(-1).copy()
    sin_out = np.asarray(sin_coeff, dtype=float).reshape(-1).copy()
    caps = np.asarray(k_caps, dtype=float).reshape(-1)
    if cos_out.size != sin_out.size or cos_out.size != caps.size:
        raise ValueError("Coefficient and k-cap vectors must have matching sizes.")
    k_values = np.sqrt(cos_out * cos_out + sin_out * sin_out)
    over_cap = np.isfinite(caps) & (k_values > caps)
    if np.any(over_cap):
        scale = np.ones_like(k_values)
        scale[over_cap] = caps[over_cap] / k_values[over_cap]
        cos_out *= scale
        sin_out *= scale
    return cos_out, sin_out, int(np.count_nonzero(over_cap))


@dataclass
class _LocalStage1Solution:
    k_values: np.ndarray
    phase_values: np.ndarray
    cos_coeff: np.ndarray
    sin_coeff: np.ndarray
    rank: int
    dropped: int
    min_relative_sv: float
    retained_rows: np.ndarray

@dataclass
class _LocalStage1SolveCache:
    retained_rows: np.ndarray
    row_weights: np.ndarray
    profile_scale: float
    weighted_basis: np.ndarray
    weighted_residual: np.ndarray
    weighted_cos_basis: np.ndarray | None = None
    weighted_sin_basis: np.ndarray | None = None
    solutions: dict[tuple[tuple[int, ...], tuple[int, ...]], _LocalStage1Solution] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Stage2 coefficient layout and objective cache
# -----------------------------------------------------------------------------

def _local_project_coefficients_to_fitted_cap(
    cos_coeff: np.ndarray,
    sin_coeff: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    caps = np.full_like(np.asarray(cos_coeff, dtype=float), np.inf, dtype=float)
    caps[np.asarray(active_mask, dtype=bool)] = _FITTED_TERM_MAX_K
    return _project_coefficients_to_k_caps(cos_coeff, sin_coeff, caps)

def _cap_k_value(k_value: float, cap: float = _FITTED_TERM_MAX_K) -> float:
    value = float(k_value)
    if abs(value) <= float(cap):
        return value
    return float(np.copysign(float(cap), value))

def _cap_fitted_fourier_term(term: FourierTerm) -> FourierTerm:
    return FourierTerm(
        kPhi=_cap_k_value(float(term.kPhi)),
        period=float(term.period),
        phase=float(term.phase),
    )

def _cap_local_fitted_k_values(k_values: np.ndarray, active_mask: np.ndarray) -> tuple[np.ndarray, int]:
    capped = np.asarray(k_values, dtype=float).reshape(-1).copy()
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    over_cap = active & (np.abs(capped) > _FITTED_TERM_MAX_K)
    if not np.any(over_cap):
        return capped, 0
    capped[over_cap] = np.sign(capped[over_cap]) * _FITTED_TERM_MAX_K
    return capped, int(np.count_nonzero(over_cap))

def _template_slot_key(period: float, phase: float) -> int | None:
    del phase
    rounded = int(round(float(period)))
    if 1 <= rounded <= 6 and abs(float(period) - float(rounded)) <= 1.0e-8:
        return int(rounded)
    return None

def _format_slot_label(period: float, phase: float) -> str:
    phase_deg = float(np.degrees(_normalize_phase_signed(float(phase))))
    phase_text = f"{phase_deg:.0f}" if np.isclose(phase_deg, round(phase_deg), atol=1.0e-6) else f"{phase_deg:.1f}"
    return f"k{int(round(float(period)))} phase={phase_text}"

def _group_slot_label(group, slot_index: int) -> str:
    local_index = group.slot_indices.index(slot_index)
    return _format_slot_label(group.slot_periods[local_index], group.slot_phases[local_index])

def _group_active_slot_labels(group, active_mask: np.ndarray) -> list[str]:
    return [
        _format_slot_label(slot_period, slot_phase)
        for slot_index, slot_period, slot_phase in zip(group.slot_indices, group.slot_periods, group.slot_phases)
        if bool(active_mask[slot_index])
    ]

def _merge_template_and_frozen_terms(
    original_terms: list[FourierTerm],
    shared_term_map: dict[int, FourierTerm],
) -> list[FourierTerm]:
    merged_terms: list[FourierTerm] = []
    seen_template_slots: set[int] = set()
    for term in original_terms:
        slot_key = _template_slot_key(term.period, term.phase)
        if slot_key is None:
            merged_terms.append(_cap_fitted_fourier_term(term))
            continue
        if slot_key in shared_term_map and slot_key not in seen_template_slots:
            shared_term = shared_term_map[slot_key]
            merged_terms.append(_cap_fitted_fourier_term(shared_term))
            seen_template_slots.add(slot_key)
        elif slot_key not in seen_template_slots:
            merged_terms.append(_cap_fitted_fourier_term(term))
            seen_template_slots.add(slot_key)
    for slot_key in sorted(shared_term_map):
        if slot_key in seen_template_slots:
            continue
        shared_term = shared_term_map[slot_key]
        merged_terms.append(_cap_fitted_fourier_term(shared_term))
    return merged_terms

def _group_frozen_non_template_slot_labels(problem: TorsionLocalProblem, group) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for local_index in group.dihedral_indices:
        for term in problem.target_dihedrals[local_index].terms:
            if _template_slot_key(term.period, term.phase) is not None:
                continue
            label = _format_slot_label(term.period, term.phase)
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return tuple(labels)

def _local_profile_scale(problem: TorsionLocalProblem) -> tuple[float, str]:
    return _profile_scale_from_arrays(problem.qm_rel, problem.fit_target_rel)

def _local_phase_orig(problem: TorsionLocalProblem) -> np.ndarray:
    if problem.phase_orig is not None:
        return np.asarray(problem.phase_orig, dtype=float)
    if problem.shared_groups:
        return np.asarray([phase for group in problem.shared_groups for phase in group.slot_phases], dtype=float)
    return np.zeros_like(np.asarray(problem.k_orig, dtype=float))

def _local_coefficient_basis(problem: TorsionLocalProblem) -> tuple[np.ndarray, np.ndarray]:
    if problem.cos_basis is not None and problem.sin_basis is not None:
        return np.asarray(problem.cos_basis, dtype=float), np.asarray(problem.sin_basis, dtype=float)
    return np.asarray(problem.basis, dtype=float), np.zeros_like(np.asarray(problem.basis, dtype=float))

def _local_coefficients_from_k_phase(k_values: np.ndarray, phase_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_array = np.asarray(k_values, dtype=float)
    phase_array = np.asarray(phase_values, dtype=float)
    return k_array * np.cos(phase_array), k_array * np.sin(phase_array)

def _local_k_phase_from_coefficients(
    cos_coeff: np.ndarray,
    sin_coeff: np.ndarray,
    phase_orig: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_values = np.asarray(cos_coeff, dtype=float)
    sin_values = np.asarray(sin_coeff, dtype=float)
    k_values = np.hypot(cos_values, sin_values)
    phase_values = np.asarray(phase_orig, dtype=float).copy()
    nonzero = k_values > 1.0e-12
    phase_values[nonzero] = np.asarray([_normalize_phase_signed(value) for value in np.arctan2(sin_values[nonzero], cos_values[nonzero])])
    k_values[~nonzero] = 0.0
    return k_values, phase_values

def _local_torsion_profile(
    problem: TorsionLocalProblem,
    slot_values: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> np.ndarray:
    k_values = np.asarray(slot_values, dtype=float)
    if phase_values is None:
        return np.asarray(problem.basis, dtype=float) @ k_values
    cos_basis, sin_basis = _local_coefficient_basis(problem)
    cos_coeff, sin_coeff = _local_coefficients_from_k_phase(k_values, np.asarray(phase_values, dtype=float))
    return (cos_basis @ cos_coeff) + (sin_basis @ sin_coeff)

def _local_solution_residual(
    problem: TorsionLocalProblem,
    slot_values: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> np.ndarray:
    return np.asarray(problem.fit_target_rel, dtype=float) - _local_torsion_profile(problem, slot_values, phase_values)

def _local_solution_profile(
    problem: TorsionLocalProblem,
    slot_values: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> np.ndarray:
    return np.asarray(problem.mm_base_rel, dtype=float) + _local_torsion_profile(problem, slot_values, phase_values)

def _group_slots_by_period(group) -> dict[int, list[int]]:
    slots_by_period: dict[int, list[int]] = {}
    for slot_index, slot_period in zip(group.slot_indices, group.slot_periods):
        slots_by_period.setdefault(int(slot_period), []).append(int(slot_index))
    return slots_by_period

def _slot_has_existing_source(group, slot_index: int) -> bool:
    slot_source = str(group.slot_sources[group.slot_indices.index(int(slot_index))])
    return "existing" in slot_source

def _slot_has_spectral_source(group, slot_index: int) -> bool:
    slot_source = str(group.slot_sources[group.slot_indices.index(int(slot_index))])
    return "spectral" in slot_source

def _stage1_spectral_variable_mask(problem: TorsionLocalProblem, active_mask: np.ndarray) -> np.ndarray:
    variable_mask = np.zeros_like(np.asarray(active_mask, dtype=bool), dtype=bool)
    for group in problem.shared_groups:
        for slot_index in group.slot_indices:
            if bool(active_mask[slot_index]) and _slot_has_spectral_source(group, int(slot_index)):
                variable_mask[slot_index] = True
    return variable_mask

def _center_active_new_slot_count(shared_groups, active_mask: np.ndarray) -> int:
    count = 0
    for group in shared_groups:
        count += sum(
            1
            for slot_index, existing in zip(group.slot_indices, group.existing_slot_mask)
            if bool(active_mask[slot_index]) and not _slot_has_existing_source(group, int(slot_index))
        )
    return count

def _group_missing_canonical_slot_options(group, active_mask: np.ndarray) -> tuple[int, ...]:
    options: list[int] = []
    for period, period_slots in _group_slots_by_period(group).items():
        active_slot = next((slot_index for slot_index in period_slots if bool(active_mask[slot_index])), None)
        if active_slot is not None:
            continue
        canonical_slot = next(
            (
                slot_index
                for slot_index in group.slot_indices
                if slot_index in period_slots
                and _slot_has_spectral_source(group, int(slot_index))
            ),
            None,
        )
        if canonical_slot is not None:
            options.append(int(canonical_slot))
    return tuple(options)

def _apply_group_period_changes(active_mask: np.ndarray, group, selected_slots: tuple[int, ...]) -> np.ndarray:
    trial_active_mask = np.asarray(active_mask, dtype=bool).copy()
    slots_by_period = _group_slots_by_period(group)
    for selected_slot in selected_slots:
        selected_period = int(group.slot_periods[group.slot_indices.index(selected_slot)])
        for slot_index in slots_by_period[selected_period]:
            trial_active_mask[slot_index] = False
        trial_active_mask[selected_slot] = True
    return trial_active_mask

def _local_group_cancellation_ratio(
    problem: TorsionLocalProblem,
    group,
    slot_values: np.ndarray,
    active_mask: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> float:
    active_group_slots = [
        slot_index
        for slot_index in group.slot_indices
        if bool(active_mask[slot_index]) and abs(float(slot_values[slot_index])) > 1.0e-12
    ]
    if len(active_group_slots) <= 1:
        return 1.0
    if phase_values is None:
        contributions = np.asarray(problem.basis[:, active_group_slots], dtype=float) * np.asarray(slot_values[active_group_slots], dtype=float)
    else:
        cos_basis, sin_basis = _local_coefficient_basis(problem)
        phase_array = np.asarray(phase_values, dtype=float)
        slot_array = np.asarray(slot_values, dtype=float)
        cos_coeff = slot_array[active_group_slots] * np.cos(phase_array[active_group_slots])
        sin_coeff = slot_array[active_group_slots] * np.sin(phase_array[active_group_slots])
        contributions = (
            np.asarray(cos_basis[:, active_group_slots], dtype=float) * cos_coeff[np.newaxis, :]
            + np.asarray(sin_basis[:, active_group_slots], dtype=float) * sin_coeff[np.newaxis, :]
        )
    individual_span = float(sum(_robust_profile_range(contributions[:, index]) for index in range(contributions.shape[1])))
    net_span = _robust_profile_range(np.sum(contributions, axis=1))
    return individual_span / max(net_span, _PROFILE_SCALE_FLOOR)

def _stage1_scan_quality(problem: TorsionLocalProblem) -> ScanProfileQuality:
    return _scan_profile_quality(problem.qm_rel)

def _stage1_new_slot_trial(
    problem: TorsionLocalProblem,
    group,
    slot_index: int,
    current_solution: np.ndarray,
    trial_solution: np.ndarray,
    trial_active_mask: np.ndarray,
    *,
    profile_scale: float,
    current_phase: np.ndarray | None = None,
    trial_phase: np.ndarray | None = None,
) -> tuple[bool, str | None, float, dict[str, float], dict[str, float]]:
    current_profile = _local_solution_profile(problem, current_solution, current_phase)
    trial_profile = _local_solution_profile(problem, trial_solution, trial_phase)
    weights = _scan_energy_weights(problem.qm_rel)
    current_metrics = _profile_loss_metrics(problem.qm_rel, current_profile, profile_scale=profile_scale, weights=weights)
    trial_metrics = _profile_loss_metrics(problem.qm_rel, trial_profile, profile_scale=profile_scale, weights=weights)
    current_abs_rmse = current_metrics["weighted_rmse"] * float(profile_scale)
    if current_abs_rmse < _STAGE1_ABS_RMSE_GAIN_FLOOR:
        return False, "existing-fit-good", 0.0, current_metrics, trial_metrics
    score_gain = current_metrics["data_loss"] - trial_metrics["data_loss"]
    rmse_gain = current_metrics["weighted_rmse"] - trial_metrics["weighted_rmse"]
    min_score_gain = max(1.0e-4, 0.02 * max(current_metrics["data_loss"], 1.0e-6))

    if score_gain < min_score_gain and rmse_gain < np.sqrt(min_score_gain):
        return False, "data-gain-too-small", score_gain, current_metrics, trial_metrics

    new_k = abs(float(trial_solution[slot_index]))
    max_new_k = max(1.50, _NEW_SLOT_MAX_MAGNITUDE_SCALE * profile_scale)
    if new_k > max_new_k:
        return False, "k-too-large", score_gain, current_metrics, trial_metrics

    abs_rmse_gain = max(0.0, rmse_gain) * float(profile_scale)
    if abs_rmse_gain < _STAGE1_ABS_RMSE_GAIN_FLOOR:
        return False, "abs-gain-too-small", score_gain, current_metrics, trial_metrics

    if trial_phase is None:
        slot_profile = np.asarray(problem.basis[:, slot_index], dtype=float) * float(trial_solution[slot_index])
    else:
        cos_basis, sin_basis = _local_coefficient_basis(problem)
        phase_value = float(trial_phase[slot_index])
        slot_profile = float(trial_solution[slot_index]) * (
            np.asarray(cos_basis[:, slot_index], dtype=float) * np.cos(phase_value)
            + np.asarray(sin_basis[:, slot_index], dtype=float) * np.sin(phase_value)
        )
    contribution_span = _robust_profile_range(slot_profile)
    if contribution_span < max(0.02, 0.005 * profile_scale):
        return False, "new-slot-below-prune-threshold", score_gain, current_metrics, trial_metrics

    cancellation_ratio = _local_group_cancellation_ratio(problem, group, trial_solution, trial_active_mask, trial_phase)
    if cancellation_ratio > _STAGE1_CANCELLATION_RATIO_CAP:
        return False, "cancellation", score_gain, current_metrics, trial_metrics

    ranking_gain = score_gain + (0.25 * max(rmse_gain, 0.0))
    return True, None, ranking_gain, current_metrics, trial_metrics

def _relative_profile(scan_data: TorsionScanData, total_values: np.ndarray) -> np.ndarray:
    totals = np.asarray(total_values, dtype=float)
    return totals - totals[int(scan_data.ref_idx)]

def _parameter_set_relative_profile(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    *,
    topology_cache=None,
) -> np.ndarray:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    totals = np.asarray(
        [evaluate_mm_energy(atoms, parameter_set, topology_cache=cache).total for atoms in scan_data.frames],
        dtype=float,
    )
    return _relative_profile(scan_data, totals)

def _stage1_retained_rows(problem: TorsionLocalProblem, params: TorsionFitParams | None) -> tuple[np.ndarray, np.ndarray]:
    del params
    retained = np.arange(len(problem.qm_rel), dtype=int)
    weights = _scan_energy_weights(np.asarray(problem.qm_rel, dtype=float))
    return retained, np.asarray(weights, dtype=float)

def _build_local_stage1_solve_cache(
    problem: TorsionLocalProblem,
    params: TorsionFitParams | None,
) -> _LocalStage1SolveCache:
    retained_rows, row_weights = _stage1_retained_rows(problem, params)
    profile_scale = _profile_fit_scale(problem.qm_rel, problem.fit_target_rel)
    sqrt_weights = np.sqrt(row_weights)
    cos_basis, sin_basis = _local_coefficient_basis(problem)
    return _LocalStage1SolveCache(
        retained_rows=np.asarray(retained_rows, dtype=int),
        row_weights=np.asarray(row_weights, dtype=float),
        profile_scale=float(profile_scale),
        weighted_basis=np.asarray(problem.basis[retained_rows], dtype=float) * sqrt_weights[:, None] / profile_scale,
        weighted_residual=np.asarray(problem.fit_target_rel[retained_rows], dtype=float) * sqrt_weights / profile_scale,
        weighted_cos_basis=np.asarray(cos_basis[retained_rows], dtype=float) * sqrt_weights[:, None] / profile_scale,
        weighted_sin_basis=np.asarray(sin_basis[retained_rows], dtype=float) * sqrt_weights[:, None] / profile_scale,
    )

def _solve_local_stage1_active_set_with_phases(
    problem: TorsionLocalProblem,
    active_mask: np.ndarray,
    params: TorsionFitParams | None,
    *,
    solve_cache: _LocalStage1SolveCache | None = None,
    variable_phase_mask: np.ndarray | None = None,
) -> _LocalStage1Solution:
    # Stage-1 follows restrained LLS semantics: fixed-phase dihedral slots
    # solve one k column. Explicit variable slots solve cos/sin coefficients.
    active_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    variable_mask = (
        np.zeros_like(np.asarray(active_mask, dtype=bool), dtype=bool)
        if variable_phase_mask is None
        else np.asarray(variable_phase_mask, dtype=bool)
    )
    if variable_mask.shape != np.asarray(active_mask, dtype=bool).shape:
        raise ValueError("variable_phase_mask must match active_mask shape.")
    variable_mask = variable_mask & np.asarray(active_mask, dtype=bool)
    variable_indices = np.flatnonzero(variable_mask)
    active_key = (
        tuple(int(index) for index in active_indices),
        tuple(int(index) for index in variable_indices),
    )
    if solve_cache is not None and active_key in solve_cache.solutions:
        cached = solve_cache.solutions[active_key]
        return _LocalStage1Solution(
            k_values=cached.k_values.copy(),
            phase_values=cached.phase_values.copy(),
            cos_coeff=cached.cos_coeff.copy(),
            sin_coeff=cached.sin_coeff.copy(),
            rank=int(cached.rank),
            dropped=int(cached.dropped),
            min_relative_sv=float(cached.min_relative_sv),
            retained_rows=cached.retained_rows.copy(),
        )

    phase_orig = _local_phase_orig(problem)
    orig_cos, orig_sin = _local_coefficients_from_k_phase(problem.k_orig, phase_orig)
    full_cos = np.zeros_like(orig_cos)
    full_sin = np.zeros_like(orig_sin)
    fixed_indices = np.asarray([index for index in active_indices if not bool(variable_mask[index])], dtype=int)
    variable_indices = np.asarray([index for index in active_indices if bool(variable_mask[index])], dtype=int)
    retained_rows, row_weights = (
        (solve_cache.retained_rows, solve_cache.row_weights)
        if solve_cache is not None
        else _stage1_retained_rows(problem, params)
    )
    if active_indices.size == 0:
        k_values, phase_values = _local_k_phase_from_coefficients(full_cos, full_sin, phase_orig)
        empty_solution = _LocalStage1Solution(
            k_values=k_values,
            phase_values=phase_values,
            cos_coeff=full_cos,
            sin_coeff=full_sin,
            rank=0,
            dropped=0,
            min_relative_sv=0.0,
            retained_rows=np.asarray(retained_rows, dtype=int).copy(),
        )
        if solve_cache is not None:
            solve_cache.solutions[active_key] = empty_solution
        return empty_solution
    if solve_cache is None:
        profile_scale = _profile_fit_scale(problem.qm_rel, problem.fit_target_rel)
        cos_basis, sin_basis = _local_coefficient_basis(problem)
        weighted_fixed_basis = np.asarray(problem.basis[retained_rows][:, fixed_indices], dtype=float) * np.sqrt(row_weights)[:, None] / profile_scale
        weighted_variable_cos = np.asarray(cos_basis[retained_rows][:, variable_indices], dtype=float) * np.sqrt(row_weights)[:, None] / profile_scale
        weighted_variable_sin = np.asarray(sin_basis[retained_rows][:, variable_indices], dtype=float) * np.sqrt(row_weights)[:, None] / profile_scale
        weighted_residual = np.asarray(problem.fit_target_rel[retained_rows], dtype=float) * np.sqrt(row_weights) / profile_scale
    else:
        weighted_fixed_basis = np.asarray(solve_cache.weighted_basis[:, fixed_indices], dtype=float)
        weighted_variable_cos = np.asarray(solve_cache.weighted_cos_basis[:, variable_indices], dtype=float)
        weighted_variable_sin = np.asarray(solve_cache.weighted_sin_basis[:, variable_indices], dtype=float)
        weighted_residual = solve_cache.weighted_residual
    weighted_basis = np.hstack((weighted_fixed_basis, weighted_variable_cos, weighted_variable_sin))
    active_orig = np.concatenate((problem.k_orig[fixed_indices], orig_cos[variable_indices], orig_sin[variable_indices]))
    active_scales = np.concatenate((problem.scales[fixed_indices], problem.scales[variable_indices], problem.scales[variable_indices]))
    active_prior_weights = _STAGE1_LLS_PRIOR_WEIGHT * np.concatenate(
        (problem.prior_weights[fixed_indices], problem.prior_weights[variable_indices], problem.prior_weights[variable_indices])
    )
    coefficient_solution, rank, dropped, min_retained_relative_sv = _solve_local_delta(
        weighted_basis,
        weighted_residual,
        active_orig,
        active_scales,
        active_prior_weights,
        prior_count=int(active_indices.size),
    )
    n_fixed = fixed_indices.size
    n_variable = variable_indices.size
    fixed_k = coefficient_solution[:n_fixed]
    variable_cos = coefficient_solution[n_fixed : n_fixed + n_variable]
    variable_sin = coefficient_solution[n_fixed + n_variable :]
    fixed_phase = phase_orig[fixed_indices].copy()
    fixed_negative = fixed_k < 0.0
    if np.any(fixed_negative):
        fixed_k = fixed_k.copy()
        fixed_k[fixed_negative] = -fixed_k[fixed_negative]
        fixed_phase[fixed_negative] = np.asarray(
            [_normalize_phase_signed(value + np.pi) for value in fixed_phase[fixed_negative]],
            dtype=float,
        )
    full_cos[fixed_indices] = fixed_k * np.cos(fixed_phase)
    full_sin[fixed_indices] = fixed_k * np.sin(fixed_phase)
    full_cos[variable_indices] = variable_cos
    full_sin[variable_indices] = variable_sin
    full_cos, full_sin, _capped_count = _local_project_coefficients_to_fitted_cap(full_cos, full_sin, active_mask)
    k_values, phase_values = _local_k_phase_from_coefficients(full_cos, full_sin, phase_orig)
    solution = _LocalStage1Solution(
        k_values=k_values,
        phase_values=phase_values,
        cos_coeff=full_cos,
        sin_coeff=full_sin,
        rank=int(rank),
        dropped=int(dropped),
        min_relative_sv=float(min_retained_relative_sv),
        retained_rows=np.asarray(retained_rows, dtype=int).copy(),
    )
    if solve_cache is not None:
        solve_cache.solutions[active_key] = solution
    return solution

def _solve_local_stage1_active_set(
    problem: TorsionLocalProblem,
    active_mask: np.ndarray,
    params: TorsionFitParams | None,
    *,
    solve_cache: _LocalStage1SolveCache | None = None,
    variable_phase_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int, float, np.ndarray]:
    solution = _solve_local_stage1_active_set_with_phases(
        problem,
        active_mask,
        params,
        solve_cache=solve_cache,
        variable_phase_mask=variable_phase_mask,
    )
    return (
        solution.k_values.copy(),
        int(solution.rank),
        int(solution.dropped),
        float(solution.min_relative_sv),
        solution.retained_rows.copy(),
    )


# -----------------------------------------------------------------------------
# Stage1 problem least-squares fit
# -----------------------------------------------------------------------------

def _solve_local_delta(
    basis: np.ndarray,
    residual: np.ndarray,
    k_orig: np.ndarray,
    scales: np.ndarray,
    prior_weights: np.ndarray,
    *,
    prior_count: int | None = None,
) -> tuple[np.ndarray, int, int, float]:
    active_count = max(int(prior_count) if prior_count is not None else len(k_orig), 1)
    reg = np.sqrt(prior_weights / active_count) / scales
    augmented_lhs = np.vstack((basis, np.diag(reg)))
    augmented_rhs = np.concatenate((residual, reg * k_orig))
    column_norms = np.linalg.norm(augmented_lhs, axis=0)
    column_norms = np.where(column_norms > 0.0, column_norms, 1.0)
    normalized_lhs = augmented_lhs / column_norms
    try:
        normalized_solution, _residuals, rank, singular_values = np.linalg.lstsq(normalized_lhs, augmented_rhs, rcond=None)
    except np.linalg.LinAlgError:
        normalized_solution = np.linalg.pinv(normalized_lhs) @ augmented_rhs
        return normalized_solution / column_norms, int(len(normalized_solution)), 0, 1.0

    if singular_values.size == 0:
        return np.zeros_like(k_orig), 0, len(k_orig), 0.0

    solution = normalized_solution / column_norms
    min_relative_sv = float(singular_values[-1] / singular_values[0]) if singular_values[0] > 0.0 else 0.0
    return solution, int(rank), 0, min_relative_sv

def _shared_group_terms_from_slots(
    problem: TorsionLocalProblem,
    slot_values: np.ndarray,
    active_mask: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> tuple[list[list[FourierTerm]], dict[str, tuple[int, int]]]:
    fitted_terms: list[list[FourierTerm]] = [[] for _ in problem.target_dihedrals]
    diagnostics: dict[str, tuple[int, int]] = {}
    phase_array = _local_phase_orig(problem) if phase_values is None else np.asarray(phase_values, dtype=float)
    for group in problem.shared_groups:
        shared_term_map = {
            int(slot_period): FourierTerm(
                kPhi=float(slot_values[slot_index]),
                period=float(slot_period),
                phase=float(phase_array[slot_index]),
            )
            for slot_index, slot_period in zip(group.slot_indices, group.slot_periods)
            if active_mask[slot_index]
        }
        for local_index in group.dihedral_indices:
            original_terms = problem.target_dihedrals[local_index].terms
            fitted_terms[local_index] = _merge_template_and_frozen_terms(original_terms, shared_term_map)
    return fitted_terms, diagnostics

def _rebuild_local_problem(problem: TorsionLocalProblem, shared_groups) -> TorsionLocalProblem:
    cos_basis, sin_basis = _group_slot_coefficient_basis(problem.scan_data, problem.target_dihedrals, tuple(shared_groups))
    phase_orig = np.asarray([phase for group in shared_groups for phase in group.slot_phases], dtype=float)
    basis = (cos_basis * np.cos(phase_orig)[np.newaxis, :]) + (sin_basis * np.sin(phase_orig)[np.newaxis, :])
    k_orig, scales, prior_weights = _group_slot_initial_values(problem.target_dihedrals, tuple(shared_groups))
    active_mask = np.asarray(
        [existing for group in shared_groups for existing in group.existing_slot_mask],
        dtype=bool,
    )
    return TorsionLocalProblem(
        center_bond=problem.center_bond,
        scan_data=problem.scan_data,
        target_dihedrals=problem.target_dihedrals,
        representative_dihedral=problem.representative_dihedral,
        basis=basis,
        qm_rel=np.asarray(problem.qm_rel, dtype=float).copy(),
        orig_mm_rel=np.asarray(problem.orig_mm_rel, dtype=float).copy(),
        mm_zeroed_rel=np.asarray(problem.mm_base_rel, dtype=float).copy(),
        residual=np.asarray(problem.fit_target_rel, dtype=float).copy(),
        k_orig=k_orig,
        scales=scales,
        active_mask=active_mask,
        prior_weights=prior_weights,
        shared_groups=tuple(shared_groups),
        phase_orig=phase_orig,
        cos_basis=cos_basis,
        sin_basis=sin_basis,
    )

def _methyl_like_family_splits(group) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    if len(group.dihedral_indices) != 6 or group.atom_types[2] != "c3" or group.atom_types[3] not in _METHYL_LIKE_H_TYPES:
        return None
    grouped_members: dict[int, list[tuple[int, int]]] = {}
    hydrogens = {int(atoms[3]) for atoms in group.instances}
    for local_index, atoms in zip(group.dihedral_indices, group.instances):
        grouped_members.setdefault(int(atoms[0]), []).append((int(local_index), int(atoms[3])))
    if len(grouped_members) != 2 or len(hydrogens) != 3 or any({h for _, h in members} != hydrogens for members in grouped_members.values()):
        return None
    return tuple(
        (outer_atom, tuple(local_index for local_index, _ in members))
        for outer_atom, members in sorted(grouped_members.items())
    )

def _dihedral_phi_matrix(scan_data: TorsionScanData, target_dihedrals, local_indices: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((len(scan_data.frames), len(local_indices)), dtype=float)
    for frame_index, atoms in enumerate(scan_data.frames):
        positions = atoms.get_positions()
        for column_index, local_index in enumerate(local_indices):
            matrix[frame_index, column_index] = dihedral_radians(
                positions,
                *target_dihedrals[int(local_index)].atoms,
            )
    return matrix

def _representative_phi_profile(problem: TorsionLocalProblem) -> np.ndarray:
    values = np.zeros(len(problem.scan_data.frames), dtype=float)
    for frame_index, atoms in enumerate(problem.scan_data.frames):
        values[frame_index] = dihedral_radians(atoms.get_positions(), *problem.representative_dihedral)
    return values

def _spectral_slots_for_groups(
    problem: TorsionLocalProblem,
    groups,
) -> dict[str, tuple[object, ...]]:
    representative_phi = _representative_phi_profile(problem)
    peaks = dominant_spectral_peaks(
        representative_phi,
        problem.fit_target_rel,
        ref_idx=int(problem.scan_data.ref_idx),
        allowed_periods=DEFAULT_SPECTRAL_PERIODS,
    )
    if not peaks:
        return {}

    selected_by_label: dict[str, list[object]] = {group.label: [] for group in groups}
    new_candidates: list[tuple[float, object, object]] = []
    for group in groups:
        path_phi = _dihedral_phi_matrix(problem.scan_data, problem.target_dihedrals, tuple(group.dihedral_indices))
        ranked_slots = rank_shared_group_spectral_slots(
            label=group.label,
            representative_phi_values=representative_phi,
            path_phi_values=path_phi,
            peaks=peaks,
            min_coherence=_STAGE1_SPECTRAL_COHERENCE_FLOOR,
        )
        existing_periods = {int(round(float(period))) for period in group.slot_periods}
        for slot in ranked_slots:
            if int(slot.period) in existing_periods:
                selected_by_label[group.label].append(slot)
            else:
                new_candidates.append((float(slot.score), group, slot))

    for _score, group, slot in sorted(new_candidates, key=lambda item: item[0], reverse=True)[:_STAGE1_CENTER_NEW_SLOT_LIMIT]:
        selected_by_label[group.label].append(slot)

    return {
        label: tuple(slots)
        for label, slots in selected_by_label.items()
        if slots
    }

def _rebuild_groups_with_spectral_slots(problem: TorsionLocalProblem, groups) -> tuple[tuple[object, ...], bool]:
    if _stage1_scan_quality(problem).has_geometry_jump:
        return tuple(groups), False
    spectral_by_label = _spectral_slots_for_groups(problem, groups)
    if not spectral_by_label:
        return tuple(groups), False
    rebuilt_groups = []
    slot_offset = 0
    for group in groups:
        members = [
            (local_index, problem.target_dihedrals[local_index])
            for local_index in group.dihedral_indices
        ]
        rebuilt_group, slot_offset = _build_group_spec(
            group.atom_types,
            members,
            slot_offset,
            label=group.label,
            spectral_slots=spectral_by_label.get(group.label, ()),
        )
        rebuilt_groups.append(rebuilt_group)
    return tuple(rebuilt_groups), True

def _solve_local_problem_stage1(problem: TorsionLocalProblem, params: TorsionFitParams | None = None):
    initial_active_mask = np.asarray(problem.active_mask, dtype=bool).copy()
    solve_cache = _build_local_stage1_solve_cache(problem, params)
    initial_solution = _solve_local_stage1_active_set_with_phases(
        problem,
        initial_active_mask,
        params,
        solve_cache=solve_cache,
        variable_phase_mask=_stage1_spectral_variable_mask(problem, initial_active_mask),
    )
    final_active_mask = initial_active_mask.copy()
    diagnostics: dict[str, dict[str, object]] = {}
    profile_scale, scale_class = _local_profile_scale(problem)
    profile_weights = _scan_energy_weights(problem.qm_rel)
    scan_quality = _stage1_scan_quality(problem)
    accepted_labels_by_group: dict[str, list[str]] = {group.label: [] for group in problem.shared_groups}
    accepted_metrics_by_group: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    candidate_trials_by_group: dict[str, list[str]] = {group.label: [] for group in problem.shared_groups}
    geometry_jump = scan_quality.has_geometry_jump
    report_debug = bool(getattr(params, "report_debug", False))

    while (
        not geometry_jump
        and _center_active_new_slot_count(problem.shared_groups, final_active_mask) < _STAGE1_CENTER_NEW_SLOT_LIMIT
    ):
        best_trial: tuple[
            float,
            object,
            int,
            str,
            np.ndarray,
            _LocalStage1Solution,
            dict[str, float],
            dict[str, float],
        ] | None = None
        found_candidate = False
        current_solution = _solve_local_stage1_active_set_with_phases(
            problem,
            final_active_mask,
            params,
            solve_cache=solve_cache,
            variable_phase_mask=_stage1_spectral_variable_mask(problem, final_active_mask),
        )
        for group in problem.shared_groups:
            candidate_slots = _group_missing_canonical_slot_options(group, final_active_mask)
            if not candidate_slots:
                continue
            found_candidate = True
            for slot_index in candidate_slots:
                trial_active_mask = _apply_group_period_changes(final_active_mask, group, (int(slot_index),))
                trial_solution = _solve_local_stage1_active_set_with_phases(
                    problem,
                    trial_active_mask,
                    params,
                    solve_cache=solve_cache,
                    variable_phase_mask=_stage1_spectral_variable_mask(problem, trial_active_mask),
                )
                passes_gate, gate_reason, ranking_gain, before_metrics, after_metrics = _stage1_new_slot_trial(
                    problem,
                    group,
                    int(slot_index),
                    current_solution.k_values,
                    trial_solution.k_values,
                    trial_active_mask,
                    profile_scale=float(profile_scale),
                    current_phase=current_solution.phase_values,
                    trial_phase=trial_solution.phase_values,
                )
                slot_label = _group_slot_label(group, int(slot_index))
                if report_debug:
                    status = "accepted-candidate" if passes_gate else f"rejected:{gate_reason}"
                    candidate_trials_by_group[group.label].append(
                        f"{slot_label} {status} gain={float(ranking_gain):.6g} "
                        f"k={float(trial_solution.k_values[slot_index]):.6g} "
                        f"phase_deg={np.degrees(float(trial_solution.phase_values[slot_index])):.3f}"
                    )
                if not passes_gate:
                    continue
                if best_trial is None or ranking_gain > best_trial[0]:
                    best_trial = (
                        float(ranking_gain),
                        group,
                        int(slot_index),
                        slot_label,
                        trial_active_mask,
                        trial_solution,
                        before_metrics,
                        after_metrics,
                    )

        if best_trial is None:
            if not found_candidate:
                break
            break

        _gain, accepted_group, _accepted_slot_index, accepted_label, accepted_active_mask, _accepted_solution, before_metrics, after_metrics = best_trial
        final_active_mask = accepted_active_mask.copy()
        accepted_labels_by_group[accepted_group.label].append(str(accepted_label))
        accepted_metrics_by_group[accepted_group.label] = (before_metrics, after_metrics)

    final_solution = _solve_local_stage1_active_set_with_phases(
        problem,
        final_active_mask,
        params,
        solve_cache=solve_cache,
        variable_phase_mask=_stage1_spectral_variable_mask(problem, final_active_mask),
    )
    final_fitted_k = final_solution.k_values
    final_rank = final_solution.rank
    final_dropped = final_solution.dropped
    final_sv = final_solution.min_relative_sv
    retained_rows = final_solution.retained_rows
    initial_metrics = _profile_loss_metrics(
        problem.qm_rel,
        _local_solution_profile(problem, initial_solution.k_values, initial_solution.phase_values),
        profile_scale=profile_scale,
        weights=profile_weights,
    )
    final_metrics = _profile_loss_metrics(
        problem.qm_rel,
        _local_solution_profile(problem, final_fitted_k, final_solution.phase_values),
        profile_scale=profile_scale,
        weights=profile_weights,
    )
    stage1_diagnostic_flags: list[str] = ["geometry_jump"] if geometry_jump else []

    for group in problem.shared_groups:
        accepted_labels = accepted_labels_by_group.get(group.label, [])
        group_active_slots = _group_active_slot_labels(group, final_active_mask)
        before_metrics, after_metrics = accepted_metrics_by_group.get(group.label, (initial_metrics, final_metrics))
        cancellation_ratio = _local_group_cancellation_ratio(problem, group, final_fitted_k, final_active_mask, final_solution.phase_values)
        diagnostics[group.label] = {
            "rank": int(final_rank),
            "dropped": max(len(group.slot_indices) - len(group_active_slots), 0),
            "active_slots": group_active_slots,
            "accepted_slots": tuple(accepted_labels),
            "scale_class": scale_class,
            "scale": float(profile_scale),
            "new_slot_limit": int(_STAGE1_CENTER_NEW_SLOT_LIMIT),
            "cancellation_ratio": float(cancellation_ratio),
            "residual_score_before": float(before_metrics["data_loss"]),
            "residual_score_after": float(after_metrics["data_loss"]),
            "diagnostic_flags": tuple(stage1_diagnostic_flags),
            "candidate_trials": tuple(candidate_trials_by_group.get(group.label, ())) if report_debug else (),
        }

    diagnostics["_stage1"] = {
        "retained_rows": [int(index) for index in retained_rows],
        "initial_rank": int(initial_solution.rank),
        "initial_dropped": int(initial_solution.dropped),
        "initial_min_relative_sv": float(initial_solution.min_relative_sv),
        "final_rank": int(final_rank),
        "final_dropped": int(final_dropped),
        "final_min_relative_sv": float(final_sv),
        "scale_class": scale_class,
        "scale": float(profile_scale),
        "profile_issue": "geometry_jump" if geometry_jump else None,
        "scan_quality": "geometry_jump" if geometry_jump else "smooth",
        "max_jump": float(scan_quality.max_jump),
        "jump_threshold": float(scan_quality.jump_threshold),
        "phase_values": [float(value) for value in final_solution.phase_values],
        "diagnostic_flags": tuple(stage1_diagnostic_flags),
        "report_debug": report_debug,
    }
    delta = final_fitted_k - problem.k_orig
    return delta, final_active_mask, diagnostics

def _default_local_fit_solver(
    problem: TorsionLocalProblem,
    params: TorsionFitParams | None = None,
    *,
    return_problem: bool = False,
):
    replacement_groups = []
    slot_offset = 0
    applied_split = False
    for group in problem.shared_groups:
        family_splits = _methyl_like_family_splits(group)
        if family_splits is None:
            members = [
                (local_index, problem.target_dihedrals[local_index])
                for local_index in group.dihedral_indices
            ]
            rebuilt_group, slot_offset = _build_group_spec(
                group.atom_types,
                members,
                slot_offset,
                label=group.label,
            )
            replacement_groups.append(rebuilt_group)
            continue
        applied_split = True
        base_label = shared_group_label(group.atom_types)
        for outer_atom, member_indices in family_splits:
            members = [
                (local_index, problem.target_dihedrals[local_index])
                for local_index in member_indices
            ]
            rebuilt_group, slot_offset = _build_group_spec(
                group.atom_types,
                members,
                slot_offset,
                label=f"{base_label} [methyl-split outer={outer_atom}]",
            )
            replacement_groups.append(rebuilt_group)

    spectral_groups, applied_spectral = _rebuild_groups_with_spectral_slots(problem, tuple(replacement_groups))
    current_problem = (
        _rebuild_local_problem(problem, spectral_groups)
        if applied_split or applied_spectral
        else problem
    )
    current_delta, current_active_mask, current_diagnostics = _solve_local_problem_stage1(current_problem, params=params)

    if return_problem:
        return current_problem, current_delta, current_active_mask, current_diagnostics
    return current_delta, current_active_mask, current_diagnostics

def _local_fitted_terms(
    problem: TorsionLocalProblem,
    delta_kphi: np.ndarray,
    active_mask: np.ndarray,
    phase_values: np.ndarray | None = None,
) -> list[list[FourierTerm]]:
    fitted_vector = problem.k_orig + np.asarray(delta_kphi, dtype=float)
    fitted_terms, _ = _shared_group_terms_from_slots(problem, fitted_vector, active_mask, phase_values)
    return fitted_terms

def _center_torsion_relative_profile(
    scan_data: TorsionScanData,
    target_dihedrals,
    fitted_terms: list[list[FourierTerm]],
) -> np.ndarray:
    torsion_total = np.zeros(len(scan_data.frames), dtype=float)
    for frame_index, atoms in enumerate(scan_data.frames):
        positions = atoms.get_positions()
        for dihedral, terms in zip(target_dihedrals, fitted_terms):
            if not terms:
                continue
            phi = dihedral_radians(positions, *dihedral.atoms)
            for term in terms:
                torsion_total[frame_index] += float(term.kPhi) * (
                    1.0 + np.cos(float(term.period) * phi - float(term.phase))
                )
    return torsion_total - torsion_total[int(scan_data.ref_idx)]


# -----------------------------------------------------------------------------
# Report assembly
# -----------------------------------------------------------------------------

def _build_fit_report(
    problem: TorsionLocalProblem,
    delta_kphi: np.ndarray,
    active_mask: np.ndarray,
    diagnostics: dict[str, tuple[int, int]] | None = None,
    topology_cache=None,
    *,
    original_parameter_set: CorrectionParameterSet | None = None,
    stage0_parameter_set: CorrectionParameterSet | None = None,
    mm_stage2_rel: np.ndarray | None = None,
    mm_orig_rel_override: np.ndarray | None = None,
    mm_stage0_rel_override: np.ndarray | None = None,
) -> TorsionFitReport:
    diagnostics = diagnostics or {}
    stage1_diagnostics = diagnostics.get("_stage1", {}) if isinstance(diagnostics, dict) else {}
    phase_values_raw = stage1_diagnostics.get("phase_values") if isinstance(stage1_diagnostics, dict) else None
    fitted_phase_values = (
        np.asarray(phase_values_raw, dtype=float)
        if phase_values_raw is not None
        else _local_phase_orig(problem)
    )
    fitted_vector, capped_count = _cap_local_fitted_k_values(
        problem.k_orig + np.asarray(delta_kphi, dtype=float),
        active_mask,
    )
    delta_kphi = fitted_vector - problem.k_orig
    fitted_terms = _local_fitted_terms(problem, delta_kphi, active_mask, fitted_phase_values)
    residual_before = _local_solution_residual(problem, problem.k_orig, _local_phase_orig(problem)).copy()
    fitted_center_torsion_rel = _center_torsion_relative_profile(
        problem.scan_data,
        problem.target_dihedrals,
        fitted_terms,
    )
    mm_refit_rel = problem.mm_base_rel + fitted_center_torsion_rel
    residual_after = problem.qm_rel - mm_refit_rel
    rmse = float(np.sqrt(np.mean(residual_after**2))) if residual_after.size else 0.0
    mm_orig_rel = (
        np.asarray(mm_orig_rel_override, dtype=float)
        if mm_orig_rel_override is not None
        else
        _parameter_set_relative_profile(problem.scan_data, original_parameter_set, topology_cache=build_mm_topology_cache(original_parameter_set))
        if original_parameter_set is not None
        else problem.orig_mm_rel.copy()
    )
    mm_stage0_rel = (
        np.asarray(mm_stage0_rel_override, dtype=float)
        if mm_stage0_rel_override is not None
        else
        _parameter_set_relative_profile(problem.scan_data, stage0_parameter_set, topology_cache=build_mm_topology_cache(stage0_parameter_set))
        if stage0_parameter_set is not None
        else problem.orig_mm_rel.copy()
    )
    shared_groups = []
    original_phase_values = _local_phase_orig(problem)
    report_debug = bool(stage1_diagnostics.get("report_debug", False)) if isinstance(stage1_diagnostics, dict) else False
    for group in problem.shared_groups:
        original_terms = [
            FourierTerm(
                kPhi=float(problem.k_orig[slot_index]),
                period=float(slot_period),
                phase=float(original_phase_values[slot_index]),
            )
            for slot_index, slot_period, is_existing in zip(group.slot_indices, group.slot_periods, group.existing_slot_mask)
            if is_existing
        ]
        fitted_group_terms = [
                FourierTerm(
                    kPhi=_cap_k_value(float(fitted_vector[slot_index])),
                    period=float(slot_period),
                    phase=float(fitted_phase_values[slot_index]),
                )
            for slot_index, slot_period in zip(group.slot_indices, group.slot_periods)
            if active_mask[slot_index]
        ]
        group_diagnostics = diagnostics.get(group.label, {})
        if isinstance(group_diagnostics, tuple):
            rank, dropped = group_diagnostics
            diagnostic_flags = ()
            candidate_trials = ()
        else:
            rank = int(group_diagnostics.get("rank", len(fitted_group_terms)))
            dropped = int(group_diagnostics.get("dropped", 0))
            diagnostic_flags = tuple(group_diagnostics.get("diagnostic_flags", ()))
            if capped_count > 0 and "k_capped" not in diagnostic_flags:
                diagnostic_flags = diagnostic_flags + ("k_capped",)
            candidate_trials = tuple(group_diagnostics.get("candidate_trials", ())) if report_debug else ()
        frozen_non_template_slots = _group_frozen_non_template_slot_labels(problem, group)
        activated_new_slot = any(
            active_mask[slot_index] and not is_existing
            for slot_index, is_existing in zip(group.slot_indices, group.existing_slot_mask)
        )
        shared_groups.append(
            TorsionSharedGroupReport(
                label=group.label,
                atom_types=group.atom_types,
                improper=False,
                instances=list(group.instances),
                original_terms=original_terms,
                fitted_terms=fitted_group_terms,
                active_slots=tuple(f"k{int(term.period)}" for term in fitted_group_terms),
                frozen_non_template_slots=frozen_non_template_slots,
                effective_rank=rank,
                dropped_singular_directions=dropped,
                activated_new_slot=activated_new_slot,
                diagnostic_flags=diagnostic_flags,
                candidate_trials=candidate_trials,
            )
        )
    return TorsionFitReport(
        center_bond=problem.center_bond,
        target_dihedrals=[deepcopy(dihedral) for dihedral in problem.target_dihedrals],
        representative_dihedral=problem.representative_dihedral,
        scan_source_path=problem.scan_data.source_path,
        terms=TorsionFitTerms(
            original_terms=_clone_terms(problem.target_dihedrals),
            fitted_terms=fitted_terms,
            delta_kphi=np.asarray(delta_kphi, dtype=float).copy(),
            shared_groups=shared_groups,
        ),
        curves=TorsionFitCurves(
            angles_deg=np.asarray(problem.scan_data.angles_deg, dtype=float).copy(),
            qm_rel=problem.qm_rel.copy(),
            mm_orig_rel=np.asarray(mm_orig_rel, dtype=float).copy(),
            mm_stage0_rel=np.asarray(mm_stage0_rel, dtype=float).copy(),
            mm_stage1_rel=np.asarray(mm_refit_rel, dtype=float).copy(),
            mm_stage2_rel=None if mm_stage2_rel is None else np.asarray(mm_stage2_rel, dtype=float).copy(),
            mm_zeroed_rel=problem.mm_base_rel.copy(),
        ),
        metrics=TorsionFitMetrics(
            residual_before=residual_before,
            residual_after=residual_after.copy(),
            rmse=rmse,
            mae=float(np.mean(np.abs(residual_after))),
            max_abs_error=float(np.max(np.abs(residual_after))),
        ),
    )
