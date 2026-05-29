"""Usage: format correction summaries and parameter change reports."""

from __future__ import annotations

import os
from math import degrees, isclose, sqrt
from typing import Optional

from ..utils.readparm import Angle, Bond, CorrectionParameterSet, Dihedral, FourierTerm, Improper, Nonbond
from .artifacts import CorrectionWorkflowResult
from .config import CorrectionConfig


SECTION_WIDTH = 108
RESULT_WIDTH = 70
REPORT_FLOAT_TOL = 1.0e-10


def stage_lines(message: str) -> list[str]:
    return [f"{message}\n"]


def summary_lines(
    config: CorrectionConfig,
    parameter_set: CorrectionParameterSet,
    auto_frcmod_path: str | None = None,
) -> list[str]:
    summary = {
        "bonds": len(parameter_set.bonds),
        "angles": len(parameter_set.angles),
        "dihedrals": len(parameter_set.dihedrals),
        "impropers": len(parameter_set.impropers),
        "nonbonds": len(parameter_set.nonbonds),
        "unmatched_bonds": len(parameter_set.unmatched_bonds),
        "unmatched_angles": len(parameter_set.unmatched_angles),
        "unmatched_dihedrals": len(parameter_set.unmatched_dihedrals),
        "unmatched_impropers": len(parameter_set.unmatched_impropers),
        "unmatched_nonbonds": len(parameter_set.unmatched_nonbonds),
    }
    torsion = config.torsion
    lines = [
        "\n",
        "=" * RESULT_WIDTH + "\n",
        "Parmfit Correction Setup".center(RESULT_WIDTH) + "\n",
        "=" * RESULT_WIDTH + "\n",
        f"Input mol2:        {_relative_path(config.mol2)}\n",
        f"Initial frcmod:    {_relative_path(auto_frcmod_path) if auto_frcmod_path else 'not generated'}\n",
        f"Topology atoms:    {len(parameter_set.mol2.atoms)}\n",
        f"Topology bonds:    {len(parameter_set.mol2.bonds)}\n",
        f"TorsionFit:        {'enabled' if torsion.enabled else 'disabled'}\n",
        f"Torsion backend:   {torsion.backend}\n",
        f"Torsion constraint:{torsion.constraint_mode}\n",
        _torsion_ensemble_setup_line(torsion),
        f"Scan grid:         {torsion.torsion_step_deg:.4f} deg x {torsion.torsion_steps} steps\n",
        f"Stage2 refine:     cycles={torsion.refine_rounds}, block_max_iter={torsion.refine_max_iter}, tol={torsion.refine_tol:.6g}\n",
        (
            "Center bonds:      auto-select non-ring center bonds with proper torsions\n"
            if torsion.center_bonds is None
            else f"Center bonds:      {list(torsion.center_bonds)}\n"
        ),
        "\n",
        "Assigned terms:\n",
        f"  bonds:      {summary['bonds']} (unmatched: {summary['unmatched_bonds']})\n",
        f"  angles:     {summary['angles']} (unmatched: {summary['unmatched_angles']})\n",
        f"  dihedrals:  {summary['dihedrals']} (unmatched: {summary['unmatched_dihedrals']})\n",
        f"  impropers:  {summary['impropers']} (unmatched: {summary['unmatched_impropers']})\n",
        f"  nonbonds:   {summary['nonbonds']} (unmatched: {summary['unmatched_nonbonds']})\n",
        "=" * RESULT_WIDTH + "\n",
    ]
    return lines


def _torsion_ensemble_setup_line(torsion) -> str:
    if not getattr(torsion, "torsion_ensemble", False):
        return "Torsion ensemble:false\n"
    return (
        "Torsion ensemble:"
        f"enabled, ratio={float(torsion.torsion_ensemble_ratio):.3f}, "
        f"weight={float(torsion.torsion_ensemble_weight):.3f}\n"
    )


def parameter_change_lines(
    title: str,
    original_result: CorrectionParameterSet,
    corrected_result: CorrectionParameterSet,
    *,
    sections: tuple[str, ...],
) -> list[str]:
    lines = [
        "\n",
        "=" * SECTION_WIDTH + "\n",
        title.center(SECTION_WIDTH) + "\n",
        "=" * SECTION_WIDTH + "\n",
        "Display convention: left = before this stage | right = after this stage\n",
        "Only changed force-field items are shown below.\n",
        "\n",
    ]
    changed_sections = 0
    for section in sections:
        section_lines = _section_change_lines(section, original_result, corrected_result)
        if section_lines:
            changed_sections += 1
            lines.extend(section_lines)
    if changed_sections == 0:
        lines.append("none\n")
    return lines


def has_parameter_changes(
    original_result: CorrectionParameterSet,
    corrected_result: CorrectionParameterSet,
    *,
    sections: tuple[str, ...],
) -> bool:
    return any(
        _section_change_count(section, original_result, corrected_result) > 0
        for section in sections
    )


def correction_result_lines(config: CorrectionConfig, result: CorrectionWorkflowResult) -> list[str]:
    mseminario_counts = _change_counts(result.initial_parameter_set, result.stage0_parameter_set)
    torsion_counts = _change_counts(result.stage0_parameter_set, result.final_parameter_set)
    final_counts = _change_counts(result.initial_parameter_set, result.final_parameter_set)
    other_changes = final_counts["impropers"] + final_counts["nonbonds"]
    warnings = _dedupe_warnings([*result.torsion.warnings, *result.gromacs.warnings])
    lines = [
        "\n",
        "=" * RESULT_WIDTH + "\n",
        "PARMFIT CORRECTION RESULT".center(RESULT_WIDTH) + "\n",
        "=" * RESULT_WIDTH + "\n",
        "Status: completed\n",
        f"Input mol2: {_relative_path(config.mol2)}\n",
        "\n",
        "Main products:\n",
        f"  Amber mol2:   {_relative_path(result.amber.mol2)}\n",
        f"  Amber frcmod: {_relative_path(result.amber.frcmod)}\n",
        f"  GROMACS top:  {_relative_path(result.gromacs.top)}\n",
        f"  GROMACS gro:  {_relative_path(result.gromacs.gro)}\n",
    ]
    if result.stage_timings:
        lines.extend(["\n", "Stage timing:\n"])
        for name, elapsed in sorted(result.stage_timings, key=lambda item: item[1], reverse=True):
            suffix = "   slowest" if elapsed == max(value for _, value in result.stage_timings) else ""
            lines.append(f"  {name:<30} {_format_duration(elapsed):>9}{suffix}\n")
    lines.extend(
        [
            "\n",
            "Parameter changes:\n",
            f"  bonds changed by mSeminario:     {mseminario_counts['bonds']}\n",
            f"  angles changed by mSeminario:    {mseminario_counts['angles']}\n",
            f"  dihedrals changed by TorsionFit: {torsion_counts['dihedrals']}\n",
            f"  other changes:                   {other_changes if other_changes else 'none'}\n",
        ]
    )
    lines.extend(_torsion_energy_trace_lines(config, result))
    lines.extend(["\n", "TorsionFit:\n"])
    if not config.torsion.enabled:
        lines.append("  state: disabled by parmfit(torsionfit=false)\n")
    elif not result.torsion.center_bonds:
        lines.append("  state: no fittable center bonds\n")
    else:
        lines.append("  state: enabled\n")
        lines.append(f"  center bonds: {_format_center_bonds(result.torsion.center_bonds)}\n")
        if result.torsion.scan_xyz:
            lines.append("  scan files:\n")
            for center_bond, path in sorted(result.torsion.scan_xyz.items()):
                lines.append(f"    {center_bond}: {_relative_path(path)}\n")
    lines.extend(_torsion_refine_round_lines(config, result))
    lines.extend(_stage2_debug_lines(config, result))

    lines.extend(["\n", "Warnings:\n"])
    if warnings:
        for warning in warnings:
            lines.append(f"  {warning}\n")
    else:
        lines.append("  none\n")
    lines.append("=" * RESULT_WIDTH + "\n")
    return lines


def _torsion_refine_round_lines(config: CorrectionConfig, result: CorrectionWorkflowResult) -> list[str]:
    if not config.torsion.enabled:
        return [
            "  stage1: not run\n",
            "  stage2: not run\n",
            "  final parameters: mSeminario result\n",
        ]
    if not result.torsion.center_bonds:
        return [
            "  stage1: not run because no fittable center bonds\n",
            "  stage2: not run\n",
            "  final parameters: mSeminario result\n",
        ]

    requested_rounds = max(int(config.torsion.refine_rounds), 0)
    if requested_rounds <= 0:
        return [
            "  stage1: completed\n",
            "  stage2: disabled by torsion_refine_rounds=0\n",
            "  final parameters: Stage1 result\n",
        ]

    diagnostics = result.torsion.stage2_diagnostics if isinstance(result.torsion.stage2_diagnostics, dict) else {}
    accepted_rounds = int(diagnostics.get("accepted_cycles", _accepted_refine_rounds(result)))
    rejected_rounds = int(diagnostics.get("rejected_cycles", _rejected_refine_rounds(result)))
    lines = [
        "  stage1: completed\n",
        f"  stage2: enabled, requested_rounds={requested_rounds}\n",
        f"  refine rounds: accepted={accepted_rounds}, rejected={rejected_rounds}\n",
    ]
    if accepted_rounds > 0 and rejected_rounds > 0:
        lines.append(
            f"  final parameters: round {accepted_rounds} accepted; round {accepted_rounds + 1} rejected, keeping round {accepted_rounds}\n"
        )
    elif accepted_rounds > 0:
        lines.append(f"  final parameters: round {accepted_rounds} accepted\n")
    elif result.torsion.refine_cycles:
        lines.append("  final parameters: Stage1 result; round 1 rejected\n")
    else:
        lines.append("  final parameters: Stage1 result; Stage2 produced no accepted round\n")
    return lines


def _accepted_refine_rounds(result: CorrectionWorkflowResult) -> int:
    return sum(1 for cycle in result.torsion.refine_cycles if int(cycle.accepted_blocks) > 0)


def _rejected_refine_rounds(result: CorrectionWorkflowResult) -> int:
    return sum(1 for cycle in result.torsion.refine_cycles if int(cycle.rejected_blocks) > 0)


def _stage2_debug_lines(config: CorrectionConfig, result: CorrectionWorkflowResult) -> list[str]:
    if not getattr(config.torsion, "report_debug", False):
        return []
    if not result.torsion.refine_cycles:
        return []
    diagnostics = result.torsion.refine_cycles[-1].diagnostics
    if not isinstance(diagnostics, dict):
        return []
    required = ("initial_total_loss", "final_total_loss")
    if not all(key in diagnostics for key in required):
        return []
    lines = ["  Stage2 loss:\n"]
    lines.append(
        f"    scan:      {float(diagnostics.get('initial_scan_loss', 0.0)):.6f} -> {float(diagnostics.get('final_scan_loss', 0.0)):.6f}\n"
    )
    lines.append(
        f"    ensemble:  {float(diagnostics.get('initial_ensemble_loss', 0.0)):.6f} -> {float(diagnostics.get('final_ensemble_loss', 0.0)):.6f}\n"
    )
    lines.append(
        f"    prior:     {float(diagnostics.get('initial_prior_loss', 0.0)):.6f} -> {float(diagnostics.get('final_prior_loss', 0.0)):.6f}\n"
    )
    lines.append(
        f"    total:     {float(diagnostics.get('initial_total_loss', 0.0)):.6f} -> {float(diagnostics.get('final_total_loss', 0.0)):.6f}\n"
    )
    lines.append(f"    status:    {diagnostics.get('status', 'unknown')}\n")
    return lines


def _torsion_energy_trace_lines(config: CorrectionConfig, result: CorrectionWorkflowResult) -> list[str]:
    lines = ["\n", "Torsion energy trace:\n"]
    if not config.torsion.enabled:
        return lines + ["  disabled by parmfit(torsionfit=false)\n"]
    if not result.torsion.center_bonds:
        return lines + ["  no fittable center bonds\n"]

    reports = list(getattr(result.torsion, "fit_reports", []) or [])
    if not reports:
        return lines + ["  not available\n"]

    wrote_table = False
    for report in reports:
        report_lines = _torsion_fit_report_energy_trace_lines(report)
        if report_lines:
            wrote_table = True
            lines.extend(report_lines)
    if not wrote_table:
        lines.append("  not available\n")
    return lines


def _torsion_fit_report_energy_trace_lines(report) -> list[str]:
    curves = getattr(report, "curves", None)
    if curves is None:
        return []
    if (
        curves.qm_rel is None
        or curves.mm_orig_rel is None
        or curves.mm_stage0_rel is None
        or curves.mm_stage1_rel is None
    ):
        return []

    angles = list(curves.angles_deg)
    mlip_ref = list(curves.qm_rel)
    orig_ref = list(curves.mm_orig_rel)
    stage0_ref = list(curves.mm_stage0_rel)
    stage1_ref = list(curves.mm_stage1_rel)
    stage2_ref = None if curves.mm_stage2_rel is None else list(curves.mm_stage2_rel)
    row_count = min(len(angles), len(mlip_ref), len(orig_ref), len(stage0_ref), len(stage1_ref))
    if row_count == 0:
        return []
    if stage2_ref is not None:
        row_count = min(row_count, len(stage2_ref))

    lines = [
        f"  \ncenter bond {report.center_bond}:\n",
        f"    scan xyz: {_relative_path(getattr(report, 'scan_source_path', None))}\n",
        "    angle_deg    MLIP_ref     orig_ref    stage0_ref    stage1_ref    stage2_ref\n",
    ]
    for index in range(row_count):
        stage2_text = "NA" if stage2_ref is None else f"{float(stage2_ref[index]):10.6f}"
        lines.append(
            f"    {float(angles[index]):9.4f}  "
            f"{float(mlip_ref[index]):10.6f}  "
            f"{float(orig_ref[index]):10.6f}  "
            f"{float(stage0_ref[index]):10.6f}  "
            f"{float(stage1_ref[index]):10.6f}  "
            f"{stage2_text:>10s}\n"
        )
    if stage2_ref is not None:
        mae, rmse = _mae_rmse(mlip_ref[:row_count], stage2_ref[:row_count])
        lines.append(f"    MLIP_ref vs stage2_final: MAE = {mae:.6f} kcal/mol, RMSE = {rmse:.6f} kcal/mol\n")
    return lines


def _mae_rmse(reference: list[float], predicted: list[float]) -> tuple[float, float]:
    residuals = [float(lhs) - float(rhs) for lhs, rhs in zip(reference, predicted)]
    if not residuals:
        return 0.0, 0.0
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = sqrt(sum(value * value for value in residuals) / len(residuals))
    return float(mae), float(rmse)


def _relative_path(path: str | None) -> str:
    if not path:
        return "NA"
    try:
        return os.path.relpath(os.fspath(path), os.getcwd())
    except ValueError:
        return os.fspath(path)


def _format_duration(seconds: float) -> str:
    seconds_i = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(seconds_i, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _format_center_bonds(center_bonds: list[tuple[int, int]]) -> str:
    if not center_bonds:
        return "none"
    return ", ".join(str(tuple(center_bond)) for center_bond in center_bonds)


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in warnings:
        normalized = str(warning).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _section_change_lines(
    section: str,
    original_result: CorrectionParameterSet,
    corrected_result: CorrectionParameterSet,
) -> list[str]:
    if section == "bonds":
        return _bond_lines(original_result.bonds, corrected_result.bonds)
    if section == "angles":
        return _angle_lines(original_result.angles, corrected_result.angles)
    if section == "dihedrals":
        return _dihedral_lines(original_result.dihedrals, corrected_result.dihedrals)
    if section == "impropers":
        return _improper_lines(original_result.impropers, corrected_result.impropers)
    if section == "nonbonds":
        return _nonbond_lines(original_result.nonbonds, corrected_result.nonbonds)
    raise ValueError(f"Unknown correction parameter section: {section}")


def _change_counts(
    original_result: CorrectionParameterSet,
    corrected_result: CorrectionParameterSet,
) -> dict[str, int]:
    return {
        section: _section_change_count(section, original_result, corrected_result)
        for section in ("bonds", "angles", "dihedrals", "impropers", "nonbonds")
    }


def _section_change_count(
    section: str,
    original_result: CorrectionParameterSet,
    corrected_result: CorrectionParameterSet,
) -> int:
    if section == "bonds":
        return _bond_change_count(original_result.bonds, corrected_result.bonds)
    if section == "angles":
        return _angle_change_count(original_result.angles, corrected_result.angles)
    if section == "dihedrals":
        return _torsion_change_count(original_result.dihedrals, corrected_result.dihedrals)
    if section == "impropers":
        return _torsion_change_count(original_result.impropers, corrected_result.impropers)
    if section == "nonbonds":
        return _nonbond_change_count(original_result.nonbonds, corrected_result.nonbonds)
    raise ValueError(f"Unknown correction parameter section: {section}")


def _bond_change_count(old_bonds: list[Bond], new_bonds: list[Bond]) -> int:
    _validate_paired_lengths("bonds", old_bonds, new_bonds)
    return sum(1 for old_bond, new_bond in zip(old_bonds, new_bonds) if _bond_changed(old_bond, new_bond))


def _angle_change_count(old_angles: list[Angle], new_angles: list[Angle]) -> int:
    _validate_paired_lengths("angles", old_angles, new_angles)
    return sum(1 for old_angle, new_angle in zip(old_angles, new_angles) if _angle_changed(old_angle, new_angle))


def _torsion_change_count(old_items: list[Dihedral] | list[Improper], new_items: list[Dihedral] | list[Improper]) -> int:
    _validate_paired_lengths("torsions", old_items, new_items)
    count = 0
    for old_item, new_item in zip(old_items, new_items):
        n_terms = max(len(old_item.terms), len(new_item.terms), 1)
        changed = False
        for term_index in range(n_terms):
            old_term = old_item.terms[term_index] if term_index < len(old_item.terms) else None
            new_term = new_item.terms[term_index] if term_index < len(new_item.terms) else None
            if _term_changed(old_term, new_term):
                changed = True
                break
        if changed:
            count += 1
    return count


def _nonbond_change_count(old_nonbonds: list[Nonbond], new_nonbonds: list[Nonbond]) -> int:
    _validate_paired_lengths("nonbonds", old_nonbonds, new_nonbonds)
    return sum(
        1
        for old_nonbond, new_nonbond in zip(old_nonbonds, new_nonbonds)
        if _nonbond_changed(old_nonbond, new_nonbond)
    )


def _section_header(title: str) -> list[str]:
    return [
        "\n",
        "-" * SECTION_WIDTH + "\n",
        title.center(SECTION_WIDTH) + "\n",
        "-" * SECTION_WIDTH + "\n",
    ]


def _validate_paired_lengths(name: str, old_items: list, new_items: list) -> None:
    if len(old_items) != len(new_items):
        raise ValueError(
            f"Correction report cannot align {name}: {len(old_items)} original entries vs {len(new_items)} new entries."
        )


def _format_atoms(atoms: tuple[int, ...]) -> str:
    return "(" + ",".join(str(atom) for atom in atoms) + ")"


def _format_types(atom_types: tuple[str, ...]) -> str:
    return "-".join(atom_types)


def _format_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def _format_angle_deg(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{degrees(float(value)):.4f}"


def _format_bond_side(bond: Bond) -> str:
    return f"k={_format_float(bond.kBond)}  r={_format_float(bond.rEq)}"


def _format_angle_side(angle: Angle) -> str:
    return f"k={_format_float(angle.kTheta)}  theta={_format_angle_deg(angle.thetaEq)}"


def _format_term_side(term: Optional[FourierTerm]) -> str:
    if term is None:
        return "k=NA  n=NA  phase=NA"
    return (
        f"k={_format_float(term.kPhi)}  "
        f"n={_format_float(term.period, 3)}  "
        f"phase={_format_angle_deg(term.phase)}"
    )


def _format_nonbond_side(nonbond: Nonbond) -> str:
    return (
        f"q={_format_float(nonbond.charge)}  "
        f"rmin/2={_format_float(nonbond.rmin_half)}  "
        f"eps={_format_float(nonbond.epsilon)}"
    )


def _close_float(left: Optional[float], right: Optional[float], tol: float = REPORT_FLOAT_TOL) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)


def _bond_changed(old_bond: Bond, new_bond: Bond) -> bool:
    return not (
        _close_float(old_bond.kBond, new_bond.kBond)
        and _close_float(old_bond.rEq, new_bond.rEq)
    )


def _angle_changed(old_angle: Angle, new_angle: Angle) -> bool:
    return not (
        _close_float(old_angle.kTheta, new_angle.kTheta)
        and _close_float(old_angle.thetaEq, new_angle.thetaEq)
    )


def _term_changed(old_term: Optional[FourierTerm], new_term: Optional[FourierTerm]) -> bool:
    if old_term is None and new_term is None:
        return False
    if old_term is None or new_term is None:
        return True
    return not (
        _close_float(old_term.kPhi, new_term.kPhi)
        and _close_float(old_term.period, new_term.period)
        and _close_float(old_term.phase, new_term.phase)
    )


def _nonbond_changed(old_nonbond: Nonbond, new_nonbond: Nonbond) -> bool:
    return not (
        _close_float(old_nonbond.charge, new_nonbond.charge)
        and _close_float(old_nonbond.rmin_half, new_nonbond.rmin_half)
        and _close_float(old_nonbond.epsilon, new_nonbond.epsilon)
    )


def _bond_lines(old_bonds: list[Bond], new_bonds: list[Bond]) -> list[str]:
    _validate_paired_lengths("bonds", old_bonds, new_bonds)
    rows: list[str] = []
    for old_bond, new_bond in zip(old_bonds, new_bonds):
        if not _bond_changed(old_bond, new_bond):
            continue
        label = (
            f"{_format_atoms(new_bond.atoms):<16} "
            f"{_format_types(new_bond.atom_types):<18}"
        )
        rows.append(
            f"{label}{_format_bond_side(old_bond):<34} | \t {_format_bond_side(new_bond):<34}\n"
        )
    return _section_header("BONDS") + rows if rows else []


def _angle_lines(old_angles: list[Angle], new_angles: list[Angle]) -> list[str]:
    _validate_paired_lengths("angles", old_angles, new_angles)
    rows: list[str] = []
    for old_angle, new_angle in zip(old_angles, new_angles):
        if not _angle_changed(old_angle, new_angle):
            continue
        label = (
            f"{_format_atoms(new_angle.atoms):<16} "
            f"{_format_types(new_angle.atom_types):<24}"
        )
        rows.append(
            f"{label}{_format_angle_side(old_angle):<36} | \t {_format_angle_side(new_angle):<36}\n"
        )
    return _section_header("ANGLES") + rows if rows else []


def _dihedral_lines(old_dihedrals: list[Dihedral], new_dihedrals: list[Dihedral]) -> list[str]:
    _validate_paired_lengths("dihedrals", old_dihedrals, new_dihedrals)
    rows: list[str] = []
    for old_dihedral, new_dihedral in zip(old_dihedrals, new_dihedrals):
        n_terms = max(len(old_dihedral.terms), len(new_dihedral.terms), 1)
        for term_index in range(n_terms):
            old_term = old_dihedral.terms[term_index] if term_index < len(old_dihedral.terms) else None
            new_term = new_dihedral.terms[term_index] if term_index < len(new_dihedral.terms) else None
            if not _term_changed(old_term, new_term):
                continue
            label = (
                f"{_format_atoms(new_dihedral.atoms):<18} "
                f"{_format_types(new_dihedral.atom_types):<26} "
                f"term={term_index + 1:<2}"
            )
            rows.append(
                f"{label}{_format_term_side(old_term):<34} |  {_format_term_side(new_term):<34}\n"
            )
    return _section_header("DIHEDRALS") + rows if rows else []


def _improper_lines(old_impropers: list[Improper], new_impropers: list[Improper]) -> list[str]:
    _validate_paired_lengths("impropers", old_impropers, new_impropers)
    rows: list[str] = []
    for old_improper, new_improper in zip(old_impropers, new_impropers):
        n_terms = max(len(old_improper.terms), len(new_improper.terms), 1)
        for term_index in range(n_terms):
            old_term = old_improper.terms[term_index] if term_index < len(old_improper.terms) else None
            new_term = new_improper.terms[term_index] if term_index < len(new_improper.terms) else None
            if not _term_changed(old_term, new_term):
                continue
            label = (
                f"{_format_atoms(new_improper.atoms):<18} "
                f"{_format_types(new_improper.atom_types):<26} "
                f"term={term_index + 1:<2}"
            )
            rows.append(
                f"{label}{_format_term_side(old_term):<34} |  {_format_term_side(new_term):<34}\n"
            )
    return _section_header("IMPROPERS") + rows if rows else []


def _nonbond_lines(old_nonbonds: list[Nonbond], new_nonbonds: list[Nonbond]) -> list[str]:
    _validate_paired_lengths("nonbonds", old_nonbonds, new_nonbonds)
    rows: list[str] = []
    for old_nonbond, new_nonbond in zip(old_nonbonds, new_nonbonds):
        if not _nonbond_changed(old_nonbond, new_nonbond):
            continue
        label = f"atom={new_nonbond.atom:<4d} {new_nonbond.atom_type:<12}"
        rows.append(
            f"{label}{_format_nonbond_side(old_nonbond):<42} |  {_format_nonbond_side(new_nonbond):<42}\n"
        )
    return _section_header("NONBONDS") + rows if rows else []
