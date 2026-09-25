"""Usage: provide public torsion fitting and loss-mode workflow entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from ..mechanics import build_mm_topology_cache, evaluate_mm_energy
from ..readparm import CorrectionParameterSet, Improper
from .topology import (
    _clone_terms,
    apply_fitted_improper,
    apply_fitted_torsion,
    torsion_bond_dihedrals,
    normalize_torsion_bond,
)
from .records import TorsionEnsembleResult, TorsionFitReport, TorsionScanData, TorsionWorkflowResult
from .config import TorsionFitParams
from .basis import _MMProfileCache, build_global_torsion_problem, build_improper_local_problem, build_local_torsion_problem
from .report import format_torsion_final_point_table, format_torsion_fit_report, format_torsion_stage2_lines
from .stage1 import _build_fit_report, local_fit_solver
from .stage2 import _build_stage2_objective_cache, _global_mm_rel_map, apply_global_delta, evaluate_global_refit_objective, refine_torsion_scans_global
from .ensemble import attach_stage2_extra_targets


def fit_torsion_scan(
    scan_data: TorsionScanData,
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
    topology_cache=None,
    *,
    improper_target: Improper | None = None,
    radical: bool = False,
    params: TorsionFitParams | None = None,
    original_paramset: CorrectionParameterSet | None = None,
    stage0_paramset: CorrectionParameterSet | None = None,
    original_mm_rel_override: np.ndarray | None = None,
    mm_base_rel_override: np.ndarray | None = None,
    stage0_mm_rel_override: np.ndarray | None = None,
) -> TorsionFitReport:
    if torsion_bond[0] == torsion_bond[1]:
        center = torsion_bond[0]
        stored = [improper for improper in paramset.impropers if improper.atoms[2] == center]
        target = stored[0] if stored else improper_target
        if target is None:
            raise ValueError(f"No improper instance or target was supplied for center {center}.")
        problem = build_improper_local_problem(
            scan_data,
            paramset,
            target,
            radical=radical,
            topology_cache=topology_cache,
        )
    else:
        problem = build_local_torsion_problem(
            scan_data,
            paramset,
            torsion_bond,
            topology_cache=topology_cache,
            mm_base_rel_override=mm_base_rel_override,
            stage0_mm_rel_override=stage0_mm_rel_override,
        )
    solver_output = local_fit_solver(problem, params=params, return_problem=True)
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
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(paramset)
    return _build_fit_report(
        solved_problem,
        delta_kphi,
        active_mask,
        diagnostics,
        topology_cache=cache,
        original_paramset=original_paramset,
        stage0_paramset=stage0_paramset if stage0_paramset is not None else paramset,
        mm_orig_rel_override=(
            np.asarray(original_mm_rel_override, dtype=float)
            if original_mm_rel_override is not None
            else problem.orig_mm_rel if original_paramset is None or original_paramset is paramset else None
        ),
        mm_stage0_rel_override=problem.orig_mm_rel if stage0_paramset is None or stage0_paramset is paramset else None,
    )


def _stage2_fit_report_from_stage1(
    stage1_report: TorsionFitReport,
    final_paramset: CorrectionParameterSet,
    *,
    mm_stage2_rel: np.ndarray,
    topology_cache=None,
) -> TorsionFitReport:
    final_dihedrals = torsion_bond_dihedrals(final_paramset, stage1_report.torsion_bond, topology_cache=topology_cache)
    fitted_terms = _clone_terms(final_dihedrals)
    return stage1_report.with_stage2_result(fitted_terms, mm_stage2_rel)


def _fit_stage1_cycle(
    *,
    current_paramset: CorrectionParameterSet,
    original_paramset: CorrectionParameterSet,
    normalized_torsion_bonds: list[tuple[int, int]],
    scan_data_map,
    scan_mm_orig_rel_map,
    params: TorsionFitParams,
    profile_cache: _MMProfileCache | None = None,
    improper_targets: dict[tuple[int, int], Improper] | None = None,
    radical_centers: tuple[int, ...] = (),
    log_info: Callable[[list[str]], None] | None = None,
) -> tuple[CorrectionParameterSet, list[TorsionFitReport], dict[str, object]]:
    cache = build_mm_topology_cache(current_paramset)
    fit_reports: list[TorsionFitReport] = []
    stage1_paramset = deepcopy(current_paramset)
    diagnostics: dict[str, object] = {
        "solver": "local_restrained_lls",
        "torsion_bonds": {},
    }

    for torsion_bond in normalized_torsion_bonds:
        improper = torsion_bond[0] == torsion_bond[1]
        mm_base_rel_override = None
        stage0_mm_rel_override = None
        if not improper and profile_cache is not None:
            mm_base_rel_override = profile_cache.center_zeroed_rel(current_paramset, torsion_bond, torsion_bond)
            stage0_mm_rel_override = profile_cache.full_rel(current_paramset, torsion_bond)
        fit_report = fit_torsion_scan(
            scan_data_map[torsion_bond],
            current_paramset,
            torsion_bond,
            topology_cache=cache,
            improper_target=improper_targets.get(torsion_bond) if improper else None,
            radical=improper and int(torsion_bond[0]) in radical_centers,
            params=params,
            original_paramset=original_paramset,
            stage0_paramset=current_paramset,
            original_mm_rel_override=scan_mm_orig_rel_map.get(torsion_bond),
            mm_base_rel_override=mm_base_rel_override,
            stage0_mm_rel_override=stage0_mm_rel_override,
        )
        fit_reports.append(fit_report)
        if log_info is not None:
            log_info(format_torsion_fit_report(fit_report))
        if improper:
            stage1_paramset = apply_fitted_improper(fit_report, stage1_paramset)
        else:
            stage1_paramset = apply_fitted_torsion(fit_report, stage1_paramset)
        diagnostics["torsion_bonds"][str(torsion_bond)] = {
            "active_slots": {
                group.label: list(group.active_slots)
                for group in fit_report.terms.shared_groups
            },
        }

    return stage1_paramset, fit_reports, diagnostics


def _stage2_cycle_diagnostics(
    *,
    requested_rounds: int,
    refine_cycles,
    initial_eval,
    final_eval,
) -> dict[str, object]:
    accepted_cycles = sum(1 for cycle in refine_cycles if cycle.diagnostics.get("status") == "accepted")
    rejected_cycles = sum(1 for cycle in refine_cycles if cycle.diagnostics.get("status") != "accepted")
    last_accepted = 0
    for cycle in refine_cycles:
        if cycle.diagnostics.get("status") == "accepted":
            last_accepted = int(cycle.cycle)
    accepted = bool(last_accepted)
    return {
        "solver": "direct_k_phase" if requested_rounds > 0 else "disabled",
        "requested_fast_cycles": int(requested_rounds),
        "requested_cycles": int(requested_rounds),
        "requested_rounds": int(requested_rounds),
        "fast_cycles": int(len(refine_cycles)),
        "cycles": int(len(refine_cycles)),
        "rounds": int(len(refine_cycles)),
        "accepted_cycles": int(accepted_cycles),
        "rejected_cycles": int(rejected_cycles),
        "best_round": int(last_accepted),
        "accepted": accepted,
        "rolled_back": not accepted,
        "final_cycle": (
            f"cycle {last_accepted} accepted"
            if accepted and rejected_cycles == 0
            else f"cycle {last_accepted} accepted, cycle {refine_cycles[-1].cycle} rejected"
            if accepted and refine_cycles
            else "kept Stage1"
        ),
        "initial_total_loss": float(initial_eval.total_loss),
        "final_total_loss": float(final_eval.total_loss),
        "initial_data_loss": float(initial_eval.data_loss),
        "final_data_loss": float(final_eval.data_loss),
        "initial_scan_loss": float(getattr(initial_eval, "scan_data_loss", 0.0)),
        "final_scan_loss": float(getattr(final_eval, "scan_data_loss", 0.0)),
        "initial_ensemble_loss": float(getattr(initial_eval, "ensemble_data_loss", 0.0)),
        "final_ensemble_loss": float(getattr(final_eval, "ensemble_data_loss", 0.0)),
        "initial_prior_loss": float(initial_eval.prior_loss),
        "final_prior_loss": float(final_eval.prior_loss),
    }


def _stage2_improper_report_from_stage1(
    stage1_report: TorsionFitReport,
    final_paramset: CorrectionParameterSet,
    scan_data: TorsionScanData,
    *,
    topology_cache=None,
) -> TorsionFitReport:
    center = stage1_report.torsion_bond[0]
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(final_paramset)
    instances = [improper for improper in final_paramset.impropers if improper.atoms[2] == center]
    fitted_terms = _clone_terms(instances)
    mm_stage2_total = np.asarray(
        [evaluate_mm_energy(atoms, final_paramset, topology_cache=cache).total for atoms in scan_data.frames],
        dtype=float,
    )
    mm_stage2_rel = mm_stage2_total - mm_stage2_total[int(scan_data.ref_idx)]
    return stage1_report.with_stage2_result(fitted_terms, mm_stage2_rel)


def _run_stage2_refinement(
    *,
    stage1_paramset: CorrectionParameterSet,
    fit_reports: list[TorsionFitReport],
    scan_data_map,
    params: TorsionFitParams,
    current_paramset: CorrectionParameterSet,
    ensemble_result: TorsionEnsembleResult | None = None,
    radical_centers: tuple[int, ...] = (),
    log_info: Callable[[list[str]], None] | None = None,
) -> tuple[CorrectionParameterSet, list[TorsionFitReport], list, object, object, dict[str, object]]:
    stage1_cache = build_mm_topology_cache(stage1_paramset)
    problem = build_global_torsion_problem(
        stage1_paramset,
        [report.torsion_bond for report in fit_reports],
        scan_data_map,
        topology_cache=stage1_cache,
        typed_shared=True,
        original_paramset=current_paramset,
        params=params,
        stage_mm_rel_map={
            normalize_torsion_bond(report.torsion_bond): np.asarray(report.curves.mm_stage1_rel, dtype=float)
            for report in fit_reports
            if report.curves.mm_stage1_rel is not None
        },
        radical_centers=radical_centers,
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
    final_paramset = apply_global_delta(problem, vector_final)
    final_eval = evaluate_global_refit_objective(problem, vector_final, cache=objective_cache)
    stage2_curves = _global_mm_rel_map(problem, vector_final, cache=objective_cache)
    diagnostics = _stage2_cycle_diagnostics(
        requested_rounds=int(params.refine_rounds),
        refine_cycles=refine_cycles,
        initial_eval=initial_eval,
        final_eval=final_eval,
    )

    final_topology_cache = build_mm_topology_cache(final_paramset)
    if log_info is not None:
        log_info(format_torsion_stage2_lines(params, refine_cycles))
        log_info(["\nFinal refined point tables after Stage 2:\n"])
    final_fit_reports = []
    for fit_report in fit_reports:
        if fit_report.torsion_bond[0] == fit_report.torsion_bond[1]:
            stage2_report = _stage2_improper_report_from_stage1(
                fit_report,
                final_paramset,
                scan_data_map[normalize_torsion_bond(fit_report.torsion_bond)],
                topology_cache=final_topology_cache,
            )
        else:
            stage2_report = _stage2_fit_report_from_stage1(
                fit_report,
                final_paramset,
                mm_stage2_rel=stage2_curves[normalize_torsion_bond(fit_report.torsion_bond)],
                topology_cache=final_topology_cache,
            )
        final_fit_reports.append(stage2_report)
        if log_info is not None:
            log_info(format_torsion_final_point_table(stage2_report))
    return final_paramset, final_fit_reports, refine_cycles, initial_eval, final_eval, diagnostics


def run_loss_mode(
    *,
    base_paramset: CorrectionParameterSet,
    torsion_bonds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    scan_data_map,
    scan_xyz_map,
    scan_mm_orig_rel_map=None,
    params: TorsionFitParams,
    topology_cache=None,
    original_paramset: CorrectionParameterSet | None = None,
    ensemble_result: TorsionEnsembleResult | None = None,
    improper_targets: dict[tuple[int, int], Improper] | None = None,
    radical_centers: tuple[int, ...] = (),
    log_info: Callable[[list[str]], None] | None = None,
) -> TorsionWorkflowResult:
    del topology_cache
    normalized_torsion_bonds = [normalize_torsion_bond(torsion_bond) for torsion_bond in torsion_bonds]
    original_paramset = original_paramset if original_paramset is not None else base_paramset
    scan_mm_orig_rel_map = scan_mm_orig_rel_map or {}
    improper_targets = improper_targets or {}
    profile_cache = _MMProfileCache(
        base_paramset,
        [bond for bond in normalized_torsion_bonds if not bond[0] == bond[1]],
        scan_data_map,
    )

    stage1_paramset, fit_reports, stage1_diagnostics = _fit_stage1_cycle(
        current_paramset=base_paramset,
        original_paramset=original_paramset,
        normalized_torsion_bonds=normalized_torsion_bonds,
        scan_data_map=scan_data_map,
        scan_mm_orig_rel_map=scan_mm_orig_rel_map,
        params=params,
        profile_cache=profile_cache,
        improper_targets=improper_targets,
        radical_centers=radical_centers,
        log_info=log_info,
    )
    final_paramset = deepcopy(stage1_paramset)
    final_fit_reports = list(fit_reports)
    refine_cycles: list = []
    stage2_diagnostics: dict[str, object] = {
        "solver": "disabled" if params.refine_rounds <= 0 else "direct_k_phase",
        "requested_fast_cycles": max(int(params.refine_rounds), 0),
        "requested_cycles": max(int(params.refine_rounds), 0),
        "requested_rounds": max(int(params.refine_rounds), 0),
        "fast_cycles": 0,
        "cycles": 0,
        "rounds": 0,
        "accepted_cycles": 0,
        "rejected_cycles": 0,
        "final_cycle": "stage1 only" if params.refine_rounds <= 0 else "not run",
    }
    if params.refine_rounds > 0 and fit_reports:
        if log_info is not None:
            log_info([f"\n[Stage 2] Running direct k/phase fast cycles ({int(params.refine_rounds)} max cycles)...\n"])
        current_paramset = base_paramset
        current_stage1_paramset = stage1_paramset
        current_fit_reports = fit_reports
        current_stage1_diagnostics = stage1_diagnostics
        best_stage1_paramset = stage1_paramset
        best_stage1_diagnostics = stage1_diagnostics
        best_eval = None
        initial_eval = None
        improvement_tol = max(float(params.refine_tol), 1.0e-12)

        for cycle_index in range(1, int(params.refine_rounds) + 1):
            if cycle_index > 1:
                current_stage1_paramset, current_fit_reports, current_stage1_diagnostics = _fit_stage1_cycle(
                    current_paramset=current_paramset,
                    original_paramset=original_paramset,
                    normalized_torsion_bonds=normalized_torsion_bonds,
                    scan_data_map=scan_data_map,
                    scan_mm_orig_rel_map=scan_mm_orig_rel_map,
                    params=params,
                    profile_cache=profile_cache,
                    improper_targets=improper_targets,
                    radical_centers=radical_centers,
                    log_info=log_info,
                )
            (
                candidate_paramset,
                candidate_fit_reports,
                cycle_reports,
                cycle_initial_eval,
                cycle_final_eval,
                _cycle_diagnostics,
            ) = _run_stage2_refinement(
                stage1_paramset=current_stage1_paramset,
                fit_reports=current_fit_reports,
                scan_data_map=scan_data_map,
                params=params,
                current_paramset=current_paramset,
                ensemble_result=ensemble_result,
                radical_centers=radical_centers,
                log_info=log_info,
            )
            if initial_eval is None:
                initial_eval = cycle_initial_eval
            if best_eval is None:
                best_eval = cycle_initial_eval

            improved = bool(
                np.isfinite(cycle_final_eval.total_loss)
                and cycle_final_eval.total_loss < best_eval.total_loss - improvement_tol
                and cycle_reports
                and cycle_reports[-1].diagnostics.get("status") == "accepted"
            )
            for cycle_report in cycle_reports:
                cycle_report.cycle = int(cycle_index)
                cycle_report.diagnostics["cycle_index"] = int(cycle_index)
                cycle_report.diagnostics["cycle_total_loss_before"] = float(best_eval.total_loss)
                cycle_report.diagnostics["cycle_total_loss_after"] = float(cycle_final_eval.total_loss)
                cycle_report.diagnostics["cycle_data_loss_before"] = float(best_eval.data_loss)
                cycle_report.diagnostics["cycle_data_loss_after"] = float(cycle_final_eval.data_loss)
                if improved:
                    cycle_report.diagnostics["status"] = "accepted"
                else:
                    cycle_report.diagnostics["status"] = "rejected"
                    cycle_report.accepted_blocks = 0
                    cycle_report.rejected_blocks = len(normalized_torsion_bonds)
            refine_cycles.extend(cycle_reports)

            if improved:
                final_paramset = candidate_paramset
                final_fit_reports = candidate_fit_reports
                best_stage1_paramset = current_stage1_paramset
                best_stage1_diagnostics = current_stage1_diagnostics
                current_paramset = candidate_paramset
                best_eval = cycle_final_eval
                continue
            break

        stage1_paramset = best_stage1_paramset
        stage1_diagnostics = best_stage1_diagnostics
        stage2_diagnostics = _stage2_cycle_diagnostics(
            requested_rounds=int(params.refine_rounds),
            refine_cycles=refine_cycles,
            initial_eval=initial_eval if initial_eval is not None else best_eval,
            final_eval=best_eval if best_eval is not None else initial_eval,
        )
    elif log_info is not None:
        log_info(format_torsion_stage2_lines(params, refine_cycles))

    return TorsionWorkflowResult(
        stage1_paramset=stage1_paramset,
        final_paramset=final_paramset,
        refine_cycles=refine_cycles,
        scan_xyz=dict(scan_xyz_map),
        torsion_bonds=list(normalized_torsion_bonds),
        fit_reports=final_fit_reports,
        warnings=[],
        stage1_diagnostics=stage1_diagnostics,
        stage2_diagnostics=stage2_diagnostics,
        ensemble_xyz=dict(ensemble_result.xyz_paths) if ensemble_result is not None else {},
    )
