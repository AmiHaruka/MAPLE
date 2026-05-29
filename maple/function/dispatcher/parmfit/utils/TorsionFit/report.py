"""Usage: format torsion fitting reports and stage summaries."""

from __future__ import annotations

from math import degrees

import numpy as np

from .records import TorsionFitReport, TorsionRefineCycle
from .config import TorsionFitParams


def format_torsion_fit_report(report: TorsionFitReport) -> list[str]:
    terms = report.terms
    curves = report.curves
    metrics = report.metrics
    absolute = report.absolute
    lines = [
        "\n",
        "=" * 92 + "\n",
        "Parmfit Torsion Scan Fit".center(92) + "\n",
        "=" * 92 + "\n",
        f"Center bond: {report.center_bond}\n",
        f"Representative dihedral: {report.representative_dihedral}\n",
        f"Scan xyz: {report.scan_source_path}\n",
        f"Target proper count: {len(report.target_dihedrals)}\n",
        "\n",
        "Target proper order:\n",
    ]

    for index, dihedral in enumerate(report.target_dihedrals, start=1):
        lines.append(f"  {index:>2d}. atoms={dihedral.atoms} types={dihedral.atom_types}\n")

    if terms.shared_groups:
        lines.append("\nUnique dihedral groups:\n")
        for index, group in enumerate(terms.shared_groups, start=1):
            lines.append(
                f"  {index:>2d}. {group.label}  terms={len(group.fitted_terms)}  shared_instances={len(group.instances)}\n"
            )
            instances = ", ".join(str(instance) for instance in group.instances)
            lines.append(f"      instances: {instances}\n")
            if "[methyl-split outer=" in group.label:
                lines.append("      sharing_override: methyl-like split applied\n")
            active_slots = getattr(group, "active_slots", ())
            if active_slots:
                lines.append(f"      active_slots: {', '.join(active_slots)}\n")
            frozen_non_template_slots = getattr(group, "frozen_non_template_slots", ())
            if frozen_non_template_slots:
                lines.append(f"      frozen_non_template: {', '.join(frozen_non_template_slots)}\n")
            lines.append(
                "      "
                f"effective_rank: {getattr(group, 'effective_rank', 0)}  "
                f"dropped_singular_directions: {getattr(group, 'dropped_singular_directions', 0)}\n"
            )
            lines.append(
                "      "
                f"activated_new_slot: {str(bool(getattr(group, 'activated_new_slot', False))).lower()}\n"
            )
            diagnostic_flags = getattr(group, "diagnostic_flags", ())
            if diagnostic_flags:
                lines.append(f"      diagnostics: {', '.join(diagnostic_flags)}\n")
            candidate_trials = getattr(group, "candidate_trials", ())
            if candidate_trials:
                lines.append("      candidate_trials:\n")
                for trial in candidate_trials:
                    lines.append(f"        {trial}\n")
            for term_index, (old_term, new_term) in enumerate(zip(group.original_terms, group.fitted_terms), start=1):
                lines.append(
                    "      "
                    f"{term_index:>2d}: "
                    f"kPhi {old_term.kPhi:.6f} -> {new_term.kPhi:.6f}  "
                    f"n={new_term.period:.3f}  phase={degrees(new_term.phase):.3f}\n"
                )
    else:
        lines.append("\nStage-1 delta initializer:\n")
        delta_str = ", ".join(f"{value:.6f}" for value in terms.delta_kphi)
        lines.append(f"  group={tuple(range(1, len(report.target_dihedrals) + 1))} delta_kPhi=[{delta_str}]\n")

        lines.append("\nStage-1 fitted kPhi per proper:\n")
        for index, (old_terms, new_terms) in enumerate(zip(terms.original_terms, terms.fitted_terms), start=1):
            for term_index, (old_term, new_term) in enumerate(zip(old_terms, new_terms), start=1):
                lines.append(
                    "  "
                    f"{index:>2d}.{term_index}: "
                    f"kPhi {old_term.kPhi:.6f} -> {new_term.kPhi:.6f}  "
                    f"n={new_term.period:.3f}  phase={degrees(new_term.phase):.3f}\n"
                )

    if absolute is not None:
        lines.extend(
            [
                "\n",
                "Absolute point table:\n",
                " angle_deg      QM_abs      orig_MM_total    torsion_delta    MM_refit_total     offset_K    residual_after_abs\n",
            ]
        )
        for angle_deg, qm_kcal, orig_mm_total, torsion_fit, mm_refit_total, residual_after_abs in zip(
            curves.angles_deg,
            absolute.qm_kcal,
            absolute.orig_mm_total,
            absolute.torsion_fit,
            absolute.mm_refit_total,
            absolute.residual_after_abs,
        ):
            lines.append(
                f"{float(angle_deg):10.4f}  "
                f"{float(qm_kcal):10.6f}  "
                f"{float(orig_mm_total):13.6f}  "
                f"{float(torsion_fit):13.6f}  "
                f"{float(mm_refit_total):14.6f}  "
                f"{float(absolute.offset_k):11.6f}  "
                f"{float(residual_after_abs):18.6f}\n"
            )
    elif (
        curves.qm_rel is not None
        and curves.mm_orig_rel is not None
        and curves.mm_stage0_rel is not None
        and curves.mm_stage1_rel is not None
    ):
        lines.extend(
            [
                "\n",
                "Relative scan point table:\n",
                " angle_deg      QM_ref      MM_orig      MM_stage0    MM_stage1    MM_stage2\n",
            ]
        )
        mm_stage2 = curves.mm_stage2_rel
        if mm_stage2 is None:
            mm_stage2 = np.full_like(np.asarray(curves.mm_stage1_rel, dtype=float), np.nan, dtype=float)
        for angle_deg, qm_rel, mm_orig_rel, mm_stage0_rel, mm_stage1_rel, mm_stage2_rel in zip(
            curves.angles_deg,
            curves.qm_rel,
            curves.mm_orig_rel,
            curves.mm_stage0_rel,
            curves.mm_stage1_rel,
            mm_stage2,
        ):
            stage2_text = "NA" if np.isnan(float(mm_stage2_rel)) else f"{float(mm_stage2_rel):11.6f}"
            lines.append(
                f"{float(angle_deg):10.4f}  "
                f"{float(qm_rel):10.6f}  "
                f"{float(mm_orig_rel):11.6f}  "
                f"{float(mm_stage0_rel):11.6f}  "
                f"{float(mm_stage1_rel):11.6f}  "
                f"{stage2_text}\n"
            )

    lines.append("\n")
    if absolute is not None:
        before_sse = float(np.sum(np.asarray(metrics.residual_before, dtype=float) ** 2))
        after_sse = float(np.sum(np.asarray(absolute.residual_after_abs, dtype=float) ** 2))
        lines.append(f"Absolute objective (offset SSE): {before_sse:.6f} -> {after_sse:.6f}\n")
        lines.append(f"Absolute RMSE:          {metrics.rmse:.6f}\n")
        lines.append(f"MAE:                   {metrics.mae:.6f}\n")
        lines.append(f"MaxAbsError:           {metrics.max_abs_error:.6f}\n")
    else:
        lines.extend(
            [
                f"Local initializer RMSE: {metrics.rmse:.6f}\n",
                f"MAE:                   {metrics.mae:.6f}\n",
                f"MaxAbsError:           {metrics.max_abs_error:.6f}\n",
            ]
        )
    return lines


def format_torsion_final_point_table(report: TorsionFitReport) -> list[str]:
    curves = report.curves
    metrics = report.metrics
    if (
        curves.qm_rel is None
        or curves.mm_orig_rel is None
        or curves.mm_stage0_rel is None
        or curves.mm_stage1_rel is None
        or curves.mm_stage2_rel is None
    ):
        return []
    lines = [
        "\n",
        "=" * 92 + "\n",
        "Parmfit Torsion Stage-2 Final Fit".center(92) + "\n",
        "=" * 92 + "\n",
        f"Center bond: {report.center_bond}\n",
        f"Representative dihedral: {report.representative_dihedral}\n",
        f"Scan xyz: {report.scan_source_path}\n",
        "\n",
        "Relative scan point table:\n",
        " angle_deg      QM_ref      MM_orig      MM_stage0    MM_stage1    MM_stage2\n",
    ]
    for angle_deg, qm_rel, mm_orig_rel, mm_stage0_rel, mm_stage1_rel, mm_stage2_rel in zip(
        curves.angles_deg,
        curves.qm_rel,
        curves.mm_orig_rel,
        curves.mm_stage0_rel,
        curves.mm_stage1_rel,
        curves.mm_stage2_rel,
    ):
        lines.append(
            f"{float(angle_deg):10.4f}  "
            f"{float(qm_rel):10.6f}  "
            f"{float(mm_orig_rel):11.6f}  "
            f"{float(mm_stage0_rel):11.6f}  "
            f"{float(mm_stage1_rel):11.6f}  "
            f"{float(mm_stage2_rel):11.6f}\n"
        )
    lines.extend(
        [
            "\n",
            f"Final refined RMSE:      {metrics.rmse:.6f}\n",
            f"MAE:                     {metrics.mae:.6f}\n",
            f"MaxAbsError:             {metrics.max_abs_error:.6f}\n",
        ]
    )
    return lines


def format_torsion_stage1_lines(
    params: TorsionFitParams,
    warnings: list[str],
    *,
    has_center_bonds: bool = True,
) -> list[str]:
    lines = ["\n"]
    if not params.enabled:
        lines.append("torsion state:   disabled by parmfit(torsionfit=false)\n")
        return lines
    if warnings:
        lines.append("warnings:\n")
        for warning in warnings:
            lines.append(f"  - {warning}\n")
    if not has_center_bonds:
        lines.append("torsion state:   no eligible center bonds were selected\n")
    return lines


def format_torsion_stage2_lines(
    params: TorsionFitParams,
    refine_reports: list[TorsionRefineCycle],
    *,
    has_center_bonds: bool = True,
) -> list[str]:
    lines = ["\n"]
    if not params.enabled:
        lines.append("refine state:     disabled because torsion fitting is disabled\n")
        return lines
    if params.refine_rounds <= 0:
        lines.append("refine state:     not requested\n")
        return lines
    if not has_center_bonds:
        lines.append("refine state:     no eligible center bonds were selected in stage 1\n")
        return lines
    if not refine_reports:
        lines.append("refine state:     no refinement cycles were executed\n")
    else:
        selected_cycles: list[TorsionRefineCycle] = []
        for cycle in refine_reports:
            if cycle.cycle == 1 or (cycle.cycle - 1) % 20 == 0:
                selected_cycles.append(cycle)
        if refine_reports and refine_reports[-1] not in selected_cycles:
            selected_cycles.append(refine_reports[-1])
        for cycle in selected_cycles:
            lines.extend(format_torsion_refine_cycle(cycle))
    return lines


def format_torsion_refine_cycle(report: TorsionRefineCycle) -> list[str]:
    headline = (
        f"\ncycle {report.cycle}: accepted={report.accepted_blocks} rejected={report.rejected_blocks}  "
        f"objective {report.total_loss_before:.6f} -> {report.total_loss_after:.6f}  "
        f"data_loss {report.data_loss_before:.6f} -> {report.data_loss_after:.6f}  "
        f"global weighted energy RMSE {report.global_rmse_before:.6f} -> {report.global_rmse_after:.6f}"
    )
    lines = [headline + "\n"]
    lines.append("  per-scan RMSE:\n")
    for center_bond in report.per_scan_rmse_before:
        lines.append(
            f"    {center_bond}: {report.per_scan_rmse_before[center_bond]:.6f} -> "
            f"{report.per_scan_rmse_after[center_bond]:.6f}\n"
        )
    per_scan_diagnostics = report.diagnostics.get("per_scan", {}) if isinstance(report.diagnostics, dict) else {}
    if per_scan_diagnostics:
        lines.append("  per-scan diagnostics:\n")
        for center_bond in report.per_scan_rmse_before:
            info = per_scan_diagnostics.get(str(center_bond))
            if not isinstance(info, dict):
                continue
            lines.append(
                f"    {center_bond}: "
                f"status={info.get('stage2_status') or 'unknown'}  "
                f"profile_issue={info.get('profile_issue') or info.get('scan_quality_note') or 'none'}  "
                f"max_adjacent_qm_jump={float(info.get('max_adjacent_qm_jump', 0.0)):.6f}  "
                f"diagnostics={','.join(info.get('diagnostic_flags', ())) or 'none'}  "
                f"max_k={float(info.get('max_abs_k', 0.0)):.6f}  "
                f"max_k_cap={float(info.get('max_k_cap', 0.0)):.6f}  "
                f"capped_terms={int(info.get('capped_terms', 0))}  "
                f"at_cap_terms={int(info.get('at_cap_terms', 0))}  "
                f"max_delta_k={float(info.get('max_delta_k', 0.0)):.6f}  "
                f"rmse_gain={float(info.get('rmse_gain', 0.0)):.6f}\n"
            )
    return lines
