"""Usage: provide public torsion fitting and loss-mode workflow entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from ..mechanics import build_mm_topology_cache
from ..readparm import CorrectionParameterSet
from .topology import _clone_terms, apply_fitted_torsion, center_bond_dihedrals
from .records import TorsionEnsembleResult, TorsionFitReport, TorsionScanData, TorsionWorkflowResult
from .config import TorsionFitParams
from .basis import _MMProfileCache, build_global_torsion_problem
from .basis import build_local_torsion_problem as _build_local_problem
from .report import format_torsion_final_point_table, format_torsion_fit_report, format_torsion_stage2_lines
from .stage1 import (
    _build_fit_report,
    _default_local_fit_solver,
)
from .stage2 import (
    _build_stage2_objective_cache,
    _global_mm_rel_map,
    apply_global_delta,
    evaluate_global_refit_objective,
    refine_torsion_scans_global,
)
from .ensemble import attach_stage2_extra_targets
from .config import normalize_center_bond


def fit_torsion_scan(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
    *,
    params: TorsionFitParams | None = None,
    original_parameter_set: CorrectionParameterSet | None = None,
    stage0_parameter_set: CorrectionParameterSet | None = None,
    original_mm_rel_override: np.ndarray | None = None,
    mm_base_rel_override: np.ndarray | None = None,
    stage0_mm_rel_override: np.ndarray | None = None,
) -> TorsionFitReport:
    problem = _build_local_problem(
        scan_data,
        parameter_set,
        center_bond,
        topology_cache=topology_cache,
        mm_base_rel_override=mm_base_rel_override,
        stage0_mm_rel_override=stage0_mm_rel_override,
    )
    solver_output = _default_local_fit_solver(problem, params=params, return_problem=True)
    if isinstance(solver_output, tuple):
        solved_problem = solver_output[0]
        delta_kphi = np.asarray(solver_output[1], dtype=float)
        active_mask = np.asarray(solver_output[2], dtype=bool)
        diagnostics = solver_output[3] if len(solver_output) > 3 else {}
    else:
        solved_problem = problem
        delta_kphi = np.asarray(solver_output, dtype=float)
        active_mask = np.asarray(problem.active_mask, dtype=bool).copy()
        diagnostics = {}
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    return _build_fit_report(
        solved_problem,
        delta_kphi,
        active_mask,
        diagnostics,
        topology_cache=cache,
        original_parameter_set=original_parameter_set,
        stage0_parameter_set=stage0_parameter_set if stage0_parameter_set is not None else parameter_set,
        mm_orig_rel_override=(
            np.asarray(original_mm_rel_override, dtype=float)
            if original_mm_rel_override is not None
            else problem.orig_mm_rel if original_parameter_set is None or original_parameter_set is parameter_set else None
        ),
        mm_stage0_rel_override=problem.orig_mm_rel if stage0_parameter_set is None or stage0_parameter_set is parameter_set else None,
    )


def _stage2_fit_report_from_stage1(
    stage1_report: TorsionFitReport,
    final_parameter_set: CorrectionParameterSet,
    *,
    mm_stage2_rel: np.ndarray,
    topology_cache=None,
) -> TorsionFitReport:
    final_dihedrals = center_bond_dihedrals(final_parameter_set, stage1_report.center_bond, topology_cache=topology_cache)
    fitted_terms = _clone_terms(final_dihedrals)
    return stage1_report.with_stage2_result(fitted_terms, mm_stage2_rel)


def _fit_stage1_cycle(
    *,
    current_parameter_set: CorrectionParameterSet,
    original_parameter_set: CorrectionParameterSet,
    normalized_center_bonds: list[tuple[int, int]],
    scan_data_map,
    scan_mm_orig_rel_map,
    params: TorsionFitParams,
    profile_cache: _MMProfileCache | None = None,
    log_info: Callable[[list[str]], None] | None = None,
) -> tuple[CorrectionParameterSet, list[TorsionFitReport], dict[str, object]]:
    cache = build_mm_topology_cache(current_parameter_set)
    fit_reports: list[TorsionFitReport] = []
    stage1_parameter_set = deepcopy(current_parameter_set)
    diagnostics: dict[str, object] = {
        "solver": "local_restrained_lls",
        "center_bonds": {},
    }

    for center_bond in normalized_center_bonds:
        mm_base_rel_override = None
        stage0_mm_rel_override = None
        if profile_cache is not None:
            mm_base_rel_override = profile_cache.center_zeroed_rel(current_parameter_set, center_bond, center_bond)
            stage0_mm_rel_override = profile_cache.full_rel(current_parameter_set, center_bond)
        fit_report = fit_torsion_scan(
            scan_data_map[center_bond],
            current_parameter_set,
            center_bond,
            topology_cache=cache,
            params=params,
            original_parameter_set=original_parameter_set,
            stage0_parameter_set=current_parameter_set,
            original_mm_rel_override=scan_mm_orig_rel_map.get(center_bond),
            mm_base_rel_override=mm_base_rel_override,
            stage0_mm_rel_override=stage0_mm_rel_override,
        )
        fit_reports.append(fit_report)
        if log_info is not None:
            log_info(format_torsion_fit_report(fit_report))
        stage1_parameter_set = apply_fitted_torsion(fit_report, stage1_parameter_set)
        diagnostics["center_bonds"][str(center_bond)] = {
            "active_slots": {
                group.label: list(group.active_slots)
                for group in fit_report.terms.shared_groups
            },
        }

    return stage1_parameter_set, fit_reports, diagnostics


def _stage2_cycle_diagnostics(
    *,
    requested_cycles: int,
    refine_cycles,
    accepted_cycles: int,
    rejected_cycles: int,
    final_cycle: str,
    last_cycle_diagnostics: dict[str, object],
    initial_eval,
    final_eval,
) -> dict[str, object]:
    per_scan_guard = last_cycle_diagnostics.get("per_scan", {}) if isinstance(last_cycle_diagnostics, dict) else {}
    accepted = bool(accepted_cycles > 0)
    return {
        "solver": "continuous_phase_k_refine" if requested_cycles > 0 else "disabled",
        "requested_cycles": int(requested_cycles),
        "cycles": int(len(refine_cycles)),
        "accepted_cycles": int(accepted_cycles),
        "rejected_cycles": int(rejected_cycles),
        "accepted": accepted,
        "rolled_back": bool(rejected_cycles > 0 or not accepted),
        "final_cycle": final_cycle,
        "reject_reason": last_cycle_diagnostics.get("reject_reason") if isinstance(last_cycle_diagnostics, dict) else None,
        "objective_kind": last_cycle_diagnostics.get("objective_kind") if isinstance(last_cycle_diagnostics, dict) else getattr(final_eval, "objective_kind", None),
        "center_bonds": per_scan_guard,
        "initial_total_loss": float(initial_eval.total_loss),
        "final_total_loss": float(final_eval.total_loss),
        "initial_data_loss": float(initial_eval.data_loss),
        "final_data_loss": float(final_eval.data_loss),
        "initial_prior_loss": float(initial_eval.prior_loss),
        "final_prior_loss": float(final_eval.prior_loss),
    }


def _run_stage2_cycle(
    *,
    cycle_index: int,
    stage1_parameter_set: CorrectionParameterSet,
    fit_reports: list[TorsionFitReport],
    scan_data_map,
    params: TorsionFitParams,
    current_parameter_set: CorrectionParameterSet,
    ensemble_result: TorsionEnsembleResult | None = None,
    log_info: Callable[[list[str]], None] | None = None,
) -> tuple[CorrectionParameterSet, list[TorsionFitReport], list, object, object, dict[str, object]]:
    stage1_cache = build_mm_topology_cache(stage1_parameter_set)
    problem = build_global_torsion_problem(
        stage1_parameter_set,
        [report.center_bond for report in fit_reports],
        scan_data_map,
        topology_cache=stage1_cache,
        typed_shared=True,
        original_parameter_set=current_parameter_set,
        params=params,
        stage_mm_rel_map={
            normalize_center_bond(report.center_bond): np.asarray(report.curves.mm_stage1_rel, dtype=float)
            for report in fit_reports
            if report.curves.mm_stage1_rel is not None
        },
    )
    problem = attach_stage2_extra_targets(problem, ensemble_result, params)
    vector_init = np.zeros(2 * len(problem.k_orig), dtype=float)
    objective_cache = _build_stage2_objective_cache(problem)
    initial_eval = evaluate_global_refit_objective(problem, vector_init, cache=objective_cache)
    vector_final, refine_cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.asarray(vector_init, dtype=float),
        enabled=True,
        max_block_iter=params.refine_max_iter,
        tol=params.refine_tol,
    )
    for cycle in refine_cycles:
        cycle.cycle = int(cycle_index)
    final_parameter_set = apply_global_delta(problem, vector_final)
    final_eval = evaluate_global_refit_objective(problem, vector_final, cache=objective_cache)
    stage2_curves = _global_mm_rel_map(problem, vector_final, cache=objective_cache)
    guard_diagnostics = refine_cycles[-1].diagnostics if refine_cycles else {}
    if isinstance(guard_diagnostics, dict):
        guard_diagnostics["outer_cycle"] = int(cycle_index)
        guard_diagnostics["objective_kind"] = guard_diagnostics.get("objective_kind", final_eval.objective_kind)

    final_topology_cache = build_mm_topology_cache(final_parameter_set)
    if log_info is not None:
        log_info(format_torsion_stage2_lines(params, refine_cycles))
        log_info([f"\nFinal refined point tables after cycle {cycle_index}:\n"])
    final_fit_reports = []
    for fit_report in fit_reports:
        stage2_report = _stage2_fit_report_from_stage1(
            fit_report,
            final_parameter_set,
            mm_stage2_rel=stage2_curves[normalize_center_bond(fit_report.center_bond)],
            topology_cache=final_topology_cache,
        )
        final_fit_reports.append(stage2_report)
        if log_info is not None:
            log_info(format_torsion_final_point_table(stage2_report))
    return final_parameter_set, final_fit_reports, refine_cycles, initial_eval, final_eval, guard_diagnostics


def _accept_cycle_score(previous_score: float | None, candidate_score: float, tol: float) -> bool:
    if previous_score is None:
        return True
    threshold = max(float(tol), 1.0e-4 * max(abs(float(previous_score)), 1.0))
    return float(candidate_score) < float(previous_score) - threshold


def run_loss_mode(
    *,
    base_parameter_set: CorrectionParameterSet,
    center_bonds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    scan_data_map,
    scan_xyz_map,
    scan_mm_orig_rel_map=None,
    params: TorsionFitParams,
    topology_cache=None,
    original_parameter_set: CorrectionParameterSet | None = None,
    ensemble_result: TorsionEnsembleResult | None = None,
    log_info: Callable[[list[str]], None] | None = None,
) -> TorsionWorkflowResult:
    del topology_cache
    normalized_center_bonds = [normalize_center_bond(center_bond) for center_bond in center_bonds]
    original_parameter_set = original_parameter_set if original_parameter_set is not None else base_parameter_set
    scan_mm_orig_rel_map = scan_mm_orig_rel_map or {}
    profile_cache = _MMProfileCache(base_parameter_set, normalized_center_bonds, scan_data_map)

    stage1_parameter_set, fit_reports, stage1_diagnostics = _fit_stage1_cycle(
        current_parameter_set=base_parameter_set,
        original_parameter_set=original_parameter_set,
        normalized_center_bonds=normalized_center_bonds,
        scan_data_map=scan_data_map,
        scan_mm_orig_rel_map=scan_mm_orig_rel_map,
        params=params,
        profile_cache=profile_cache,
        log_info=log_info,
    )
    final_parameter_set = deepcopy(stage1_parameter_set)
    final_fit_reports = list(fit_reports)
    refine_cycles: list = []
    stage2_diagnostics: dict[str, object] = {
        "solver": "disabled" if params.refine_rounds <= 0 else "continuous_phase_k_refine",
        "requested_cycles": max(int(params.refine_rounds), 0),
        "cycles": 0,
        "accepted_cycles": 0,
        "rejected_cycles": 0,
        "final_cycle": "stage1 only" if params.refine_rounds <= 0 else "not run",
    }
    if params.refine_rounds > 0 and fit_reports:
        if log_info is not None:
            log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
        current_parameter_set = deepcopy(base_parameter_set)
        best_stage1_parameter_set = deepcopy(stage1_parameter_set)
        best_stage1_diagnostics = stage1_diagnostics
        best_score: float | None = None
        accepted_cycles = 0
        rejected_cycles = 0
        last_initial_eval = None
        last_final_eval = None
        last_cycle_diagnostics: dict[str, object] = {}
        for cycle_index in range(1, int(params.refine_rounds) + 1):
            if cycle_index > 1:
                stage1_parameter_set, fit_reports, stage1_diagnostics = _fit_stage1_cycle(
                    current_parameter_set=current_parameter_set,
                    original_parameter_set=original_parameter_set,
                    normalized_center_bonds=normalized_center_bonds,
                    scan_data_map=scan_data_map,
                    scan_mm_orig_rel_map=scan_mm_orig_rel_map,
                    params=params,
                    profile_cache=profile_cache,
                    log_info=log_info,
                )
            if log_info is not None:
                log_info([f"\n[Stage 2] Fast MM cycle {cycle_index}/{int(params.refine_rounds)} ...\n"])
            (
                cycle_parameter_set,
                cycle_fit_reports,
                cycle_refine_cycles,
                initial_eval,
                final_eval,
                guard_diagnostics,
            ) = _run_stage2_cycle(
                cycle_index=cycle_index,
                stage1_parameter_set=stage1_parameter_set,
                fit_reports=fit_reports,
                scan_data_map=scan_data_map,
                params=params,
                current_parameter_set=current_parameter_set,
                ensemble_result=ensemble_result,
                log_info=log_info,
            )
            has_ensemble_targets = (
                ensemble_result is not None
                and params.torsion_ensemble_weight > 0.0
                and bool(ensemble_result.frames_by_center)
            )
            candidate_score = float(final_eval.total_loss if has_ensemble_targets else final_eval.data_loss)
            last_initial_eval = initial_eval
            last_final_eval = final_eval
            last_cycle_diagnostics = guard_diagnostics if isinstance(guard_diagnostics, dict) else {}
            inner_accepted = any(int(getattr(cycle, "accepted_blocks", 0)) > 0 for cycle in cycle_refine_cycles)
            if inner_accepted and _accept_cycle_score(best_score, candidate_score, params.refine_tol):
                refine_cycles.extend(cycle_refine_cycles)
                accepted_cycles += 1
                best_score = candidate_score
                current_parameter_set = deepcopy(cycle_parameter_set)
                best_stage1_parameter_set = deepcopy(stage1_parameter_set)
                best_stage1_diagnostics = stage1_diagnostics
                final_parameter_set = deepcopy(cycle_parameter_set)
                final_fit_reports = list(cycle_fit_reports)
                continue

            rejected_cycles += 1
            for cycle in cycle_refine_cycles:
                cycle.diagnostics["outer_cycle_rejected"] = True
                cycle.diagnostics["outer_cycle_reject_reason"] = "no_fast_cycle_data_loss_gain"
            refine_cycles.extend(cycle_refine_cycles)
            break
        stage1_parameter_set = deepcopy(best_stage1_parameter_set)
        stage1_diagnostics = best_stage1_diagnostics
        if rejected_cycles and accepted_cycles > 0:
            final_cycle = f"cycle {accepted_cycles} accepted, cycle {accepted_cycles + 1} rejected"
        elif rejected_cycles:
            final_cycle = "cycle 1 rejected; keeping Stage1"
        else:
            final_cycle = f"cycle {accepted_cycles} accepted"
        if last_initial_eval is not None and last_final_eval is not None:
            stage2_diagnostics = _stage2_cycle_diagnostics(
                requested_cycles=int(params.refine_rounds),
                refine_cycles=refine_cycles,
                accepted_cycles=accepted_cycles,
                rejected_cycles=rejected_cycles,
                final_cycle=final_cycle,
                last_cycle_diagnostics=last_cycle_diagnostics,
                initial_eval=last_initial_eval,
                final_eval=last_final_eval,
            )
    elif log_info is not None:
        log_info(format_torsion_stage2_lines(params, refine_cycles))

    return TorsionWorkflowResult(
        stage1_parameter_set=stage1_parameter_set,
        final_parameter_set=final_parameter_set,
        refine_cycles=refine_cycles,
        scan_xyz=dict(scan_xyz_map),
        center_bonds=list(normalized_center_bonds),
        fit_reports=final_fit_reports,
        warnings=[],
        stage1_diagnostics=stage1_diagnostics,
        stage2_diagnostics=stage2_diagnostics,
        ensemble_xyz=dict(ensemble_result.xyz_paths) if ensemble_result is not None else {},
    )
