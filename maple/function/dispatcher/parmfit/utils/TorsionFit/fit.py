"""Usage: provide public torsion fitting and loss-mode workflow entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from ..mechanics import build_mm_topology_cache
from ..readparm import CorrectionParameterSet
from .topology import _clone_terms, apply_fitted_torsion, center_bond_dihedrals
from .records import TorsionFitReport, TorsionScanData, TorsionWorkflowResult
from .config import TorsionFitParams
from .basis import build_global_torsion_problem
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
) -> TorsionFitReport:
    problem = _build_local_problem(
        scan_data,
        parameter_set,
        center_bond,
        topology_cache=topology_cache,
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
    log_info: Callable[[list[str]], None] | None = None,
) -> TorsionWorkflowResult:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(base_parameter_set)
    normalized_center_bonds = [normalize_center_bond(center_bond) for center_bond in center_bonds]
    original_parameter_set = original_parameter_set if original_parameter_set is not None else base_parameter_set
    scan_mm_orig_rel_map = scan_mm_orig_rel_map or {}

    fit_reports: list[TorsionFitReport] = []
    stage1_parameter_set = deepcopy(base_parameter_set)
    stage1_diagnostics: dict[str, object] = {
        "solver": "local_restrained_lls",
        "center_bonds": {},
    }

    for center_bond in normalized_center_bonds:
        fit_report = fit_torsion_scan(
            scan_data_map[center_bond],
            base_parameter_set,
            center_bond,
            topology_cache=cache,
            params=params,
            original_parameter_set=original_parameter_set,
            stage0_parameter_set=base_parameter_set,
            original_mm_rel_override=scan_mm_orig_rel_map.get(center_bond),
        )
        fit_reports.append(fit_report)
        if log_info is not None:
            log_info(format_torsion_fit_report(fit_report))
        stage1_parameter_set = apply_fitted_torsion(fit_report, stage1_parameter_set)
        stage1_diagnostics["center_bonds"][str(center_bond)] = {
            "active_slots": {
                group.label: list(group.active_slots)
                for group in fit_report.terms.shared_groups
            },
        }

    final_parameter_set = deepcopy(stage1_parameter_set)
    refine_cycles = []
    stage2_diagnostics: dict[str, object] = {
        "solver": "disabled" if params.refine_rounds <= 0 else "continuous_phase_k_refine",
        "cycles": 0,
    }
    if log_info is not None:
        log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
    if params.refine_rounds > 0 and fit_reports:
        problem = build_global_torsion_problem(
            stage1_parameter_set,
            [report.center_bond for report in fit_reports],
            scan_data_map,
            topology_cache=cache,
            typed_shared=True,
            original_parameter_set=base_parameter_set,
            params=params,
            stage_mm_rel_map={
                normalize_center_bond(report.center_bond): np.asarray(report.curves.mm_stage1_rel, dtype=float)
                for report in fit_reports
                if report.curves.mm_stage1_rel is not None
            },
        )
        vector_init = np.zeros(2 * len(problem.k_orig), dtype=float)
        objective_cache = _build_stage2_objective_cache(problem)
        initial_eval = evaluate_global_refit_objective(problem, vector_init, cache=objective_cache)
        vector_final, refine_cycles = refine_torsion_scans_global(
            problem,
            delta_init=np.asarray(vector_init, dtype=float),
            enabled=params.refine_rounds > 0,
            max_block_iter=params.refine_max_iter,
            tol=params.refine_tol,
        )
        final_parameter_set = apply_global_delta(problem, vector_final)
        final_eval = evaluate_global_refit_objective(problem, vector_final, cache=objective_cache)
        stage2_curves = _global_mm_rel_map(problem, vector_final, cache=objective_cache)
        accepted = bool(refine_cycles and refine_cycles[-1].accepted_blocks > 0)
        guard_diagnostics = refine_cycles[-1].diagnostics if refine_cycles else {}
        per_scan_guard = guard_diagnostics.get("per_scan", {}) if isinstance(guard_diagnostics, dict) else {}
        stage2_diagnostics = {
            "solver": "continuous_phase_k_refine",
            "cycles": len(refine_cycles),
            "accepted": accepted,
            "rolled_back": not accepted,
            "reject_reason": guard_diagnostics.get("reject_reason") if isinstance(guard_diagnostics, dict) else None,
            "objective_kind": guard_diagnostics.get("objective_kind") if isinstance(guard_diagnostics, dict) else final_eval.objective_kind,
            "center_bonds": per_scan_guard,
            "initial_total_loss": float(initial_eval.total_loss),
            "final_total_loss": float(final_eval.total_loss),
            "initial_data_loss": float(initial_eval.data_loss),
            "final_data_loss": float(final_eval.data_loss),
            "initial_prior_loss": float(initial_eval.prior_loss),
            "final_prior_loss": float(final_eval.prior_loss),
        }
        final_topology_cache = build_mm_topology_cache(final_parameter_set)
        if log_info is not None:
            log_info(format_torsion_stage2_lines(params, refine_cycles))
            log_info(["\nFinal refined point tables:\n"])
        for fit_report in fit_reports:
            stage2_report = _stage2_fit_report_from_stage1(
                fit_report,
                final_parameter_set,
                mm_stage2_rel=stage2_curves[normalize_center_bond(fit_report.center_bond)],
                topology_cache=final_topology_cache,
            )
            if log_info is not None:
                log_info(format_torsion_final_point_table(stage2_report))
    elif log_info is not None:
        log_info(format_torsion_stage2_lines(params, refine_cycles))

    return TorsionWorkflowResult(
        stage1_parameter_set=stage1_parameter_set,
        final_parameter_set=final_parameter_set,
        refine_cycles=refine_cycles,
        scan_xyz=dict(scan_xyz_map),
        center_bonds=list(normalized_center_bonds),
        warnings=[],
        stage1_diagnostics=stage1_diagnostics,
        stage2_diagnostics=stage2_diagnostics,
    )
