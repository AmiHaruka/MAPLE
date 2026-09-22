"""Usage: format MetalAA workflow status lines and the MetalAA result summary."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from ..structure import get_resid_label

from .config import MetalAbinitioConfig
from .artifacts import MetalArtifacts, MetalAtomTypeRow


def format_metal_start_lines(
    config: MetalAbinitioConfig,
    *,
    large_charge: int,
    large_mult: int,
) -> list[str]:
    return [
        f"  ion_resid: {config.ion_resid}\n",
        f"  add_resid: {config.add_resid}\n",
        f"  cluster_cutoff: {config.cluster_cutoff:.2f} A\n",
        f"  donor_cutoff: {config.donor_cutoff:.2f} A\n",
        f"  water ff: {config.wat_ff}\n",
        f"  protein ff: {config.pro_ff}\n",
        f"  ion ff: {config.ion_ff}\n",
        f"  ion_charges: {config.ion_charges}\n",
        f"  ion_mult: {config.ion_mult}\n",
        f"  ncaa_resids: {config.ncaa_resids}\n",
        f"  large model charge/mult: {large_charge} {large_mult}\n",
        f"  chgmod: {config.resp.chgmod}\n",
        f"  fixchg_resids: {config.resp.fixchg_resids}\n",
        f"  QM method: {config.resp.qm.theory}/{config.resp.qm.basis}\n",
    ]


def format_metal_final_lines(
    *,
    artifacts: MetalArtifacts,
    atom_type_rows: list[MetalAtomTypeRow] | None = None,
    ion_frcmods: list[str] | None = None,
    metal_formal_charge: int | None = None,
    metal_fitted_charge: float | None = None,
    bonded_warning: Optional[str] = None,
    external_residues: list[str] | None = None,
    stage_timings: list[tuple[str, float]] | None = None,
) -> list[str]:
    del artifacts, atom_type_rows, ion_frcmods, metal_formal_charge
    del metal_fitted_charge, bonded_warning, external_residues, stage_timings
    return ["  [MetalAA] route completed; final summary follows.\n"]


def format_pdb_read_diagnostics(
    diagnostics,
    *,
    target_label: str | None = None,
    target_charge: tuple[str, int] | None = None,
) -> list[str]:
    lines = [
        "  PDB residue templates: "
        f"matched={diagnostics.matched}, backbone_only={diagnostics.backbone_only}, unmatched={diagnostics.unmatched}\n"
    ]
    backbone_only_residues = getattr(diagnostics, "backbone_only_residues", ())
    not_matched_residues = getattr(diagnostics, "not_matched_residues", ())
    if backbone_only_residues:
        entries = [
            f"{label} ({'target' if target_label is not None and label == target_label else 'also note'})"
            for label in backbone_only_residues
        ]
        lines.append("    non-standard (matched backbone only): " + ", ".join(entries) + "\n")
    if not_matched_residues:
        lines.append("    not matched (seemingly not protein): " + ", ".join(not_matched_residues) + "\n")
        lines.append(
            "    declare them via lig_resids (or ncaa_resids) to build parameters as small molecules\n"
        )
    renamed_residues = getattr(diagnostics, "renamed_residues", ())
    if renamed_residues:
        lines.append("    template overrides (renamed): " + ", ".join(renamed_residues) + "\n")
    if target_charge is not None:
        charge_label, charge = target_charge
        lines.append(f"  PDB target charge: {charge_label} = {charge} (from your input)\n")
    return lines


@dataclass(frozen=True)
class TLeapSummary:
    status: str
    errors: int | None = None
    warnings: int | None = None
    notes: int | None = None
    checks: tuple[str, ...] = ()


def _safe_get(mapping: dict | None, key: str, default: str = "not written") -> str:
    if not mapping:
        return default
    value = mapping.get(key)
    return str(value) if value else default


def _artifact_files(result) -> dict:
    artifacts = getattr(result, "artifacts", None)
    if artifacts is not None:
        return dict(getattr(artifacts, "files", {}) or {})
    return dict(getattr(result, "files", {}) or {})


def _labels(residues: Iterable[dict] | None) -> str:
    labels = [get_resid_label(residue) for residue in residues or []]
    return ", ".join(labels) if labels else "none"


def _deduped(values: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _relative_path(path: str, *, base_dir: str | None = None) -> str:
    text = str(path or "").strip()
    if not text or text == "not written":
        return text or "not written"
    if not os.path.isabs(text):
        return text
    base = os.path.abspath(base_dir or os.getcwd())
    rel = os.path.relpath(text, base)
    if rel == "." or rel.startswith(".." + os.sep) or rel == "..":
        return text
    return rel


def _path_list(values: Iterable[str] | None) -> str:
    items = [_relative_path(value) for value in values or [] if str(value).strip()]
    return ", ".join(items) if items else "none"


def _duration(seconds: float) -> str:
    total = int(round(max(float(seconds), 0.0)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def _timing_label(label: str) -> str:
    mapping = {
        "RESP fitting": "RESP fitting",
        "Hessian/mSeminario/frcmod export": "Hessian + mSeminario",
        "Hessian/Seminario/frcmod export": "Hessian + Seminario",
        "model optimization": "model optimization",
        "site export": "site export",
        "tleap validation": "tleap validation",
    }
    return mapping.get(str(label), str(label))


def _display_timings(result) -> list[tuple[str, float]]:
    raw = [(str(label), float(seconds)) for label, seconds in getattr(result, "stage_timings", []) or []]
    suppressed = {"large RESP", "Hessian evaluation"}
    combined: dict[str, float] = {}
    for label, seconds in raw:
        if label in suppressed:
            continue
        display = _timing_label(label)
        combined[display] = max(combined.get(display, 0.0), float(seconds))
    return sorted(combined.items(), key=lambda item: item[1], reverse=True)


def _timing_lines(result) -> list[str]:
    timings = _display_timings(result)
    lines = ["\nStage timing:\n"]
    if not timings:
        return lines + ["  none\n"]
    label_width = max(len(label) for label, _seconds in timings) + 2
    for label, seconds in timings:
        lines.append(f"  {label:<{label_width}} {_duration(seconds):>8s}\n")
    return lines


def _tleap_output_path(files: dict) -> str | None:
    tleap_input = str(files.get("tleap_input") or "").strip()
    if not tleap_input:
        return None
    root, _ext = os.path.splitext(tleap_input)
    return root + ".out"


def _parse_tleap_summary(path: str | None) -> TLeapSummary:
    if not path or not os.path.exists(path):
        return TLeapSummary(status="not executed")
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    match = re.search(r"Errors\s*=\s*(\d+);\s*Warnings\s*=\s*(\d+);\s*Notes\s*=\s*(\d+)", text)
    errors = warnings = notes = None
    if match:
        errors, warnings, notes = (int(match.group(index)) for index in range(1, 4))

    checks: list[str] = []
    for long_bond in re.finditer(
        r"There is a bond of\s+([0-9.]+)\s+angstroms between\s+(\S+)\s+and\s+(\S+)\s+atoms",
        text,
    ):
        distance, left, right = long_bond.groups()
        checks.append(f"long bond {left}-{right} = {float(distance):.3f} A")
    if "FATAL" in text:
        checks.append("FATAL message present in tleap output")
    if "Could not find" in text and "parameter" in text:
        checks.append("missing parameter message present in tleap output")
    return_code = re.search(r"MAPLE_TLEAP_RETURN_CODE\s*=\s*(\d+)", text)
    if return_code:
        checks.append(f"tleap return code {int(return_code.group(1))}")
    status = "executed" if match else "output found"
    return TLeapSummary(status=status, errors=errors, warnings=warnings, notes=notes, checks=tuple(_deduped(checks)))


def _tleap_status_lines(summary: TLeapSummary) -> list[str]:
    lines = ["\nTleap status:\n"]
    if summary.status == "not executed":
        return lines + ["  input written, not executed\n"]
    if summary.errors is not None:
        lines.append(f"  Errors: {summary.errors}\n")
    if summary.warnings is not None:
        lines.append(f"  Warnings: {summary.warnings}\n")
    if summary.notes is not None:
        lines.append(f"  Notes: {summary.notes}\n")
    if summary.errors is None and summary.warnings is None and summary.notes is None:
        lines.append(f"  {summary.status}\n")
    for check in summary.checks:
        lines.append(f"  Check: {check}\n")
    return lines


def _warning_lines(warnings: Iterable[str] | None) -> list[str]:
    lines = ["\nWarnings:\n"]
    items = _deduped(warnings)
    if items:
        lines.extend(f"  {item}\n" for item in items)
    else:
        lines.append("  none\n")
    return lines


def _next_step_lines(files: dict) -> list[str]:
    tleap_input = _safe_get(files, "tleap_input", "")
    if not tleap_input:
        return ["\nNext step:\n", "  none\n"]
    workdir = os.path.dirname(tleap_input) or "."
    return [
        "\nNext step:\n",
        f"  cd {_relative_path(workdir)}\n",
        f"  tleap -s -f {os.path.basename(tleap_input)} |tee {os.path.basename(tleap_input).replace('.in', '.out')}\n",
    ]


def _metal_warnings(result) -> list[str]:
    selection = getattr(result, "selection", None)
    site_model = getattr(result, "site_model", {}) or {}
    warnings: list[str] = []
    warnings.extend(getattr(selection, "warnings", []) or [])
    warnings.extend(site_model.get("warnings", []) if isinstance(site_model, dict) else [])
    bonded_warning = getattr(result, "bonded_warning", None)
    if bonded_warning:
        warnings.append(str(bonded_warning))
    return warnings


def _atom_type_lines(rows) -> list[str]:
    items = list(rows or [])
    if not items:
        return []
    lines = ["\nRenamed atom types:\n", "  residue      atom  old   new   charge\n"]
    for row in items:
        lines.append(f"  {row.residue:<12s} {row.atom_name:<5s} {row.old_type:<5s} {row.new_type:<5s} {row.charge: .6f}\n")
    return lines


def _header() -> list[str]:
    return [
        "\n",
        "=" * 70 + "\n",
        "PARMFIT METALAA RESULT".center(70) + "\n",
        "=" * 70 + "\n",
        "Status: completed\n",
    ]


def _footer() -> list[str]:
    return ["=" * 70 + "\n"]


def _pro_ff_label(config) -> str:
    return getattr(config, "pro_ff", "ff14SB")


def format_metal_summary(*, target_label: str, config, result) -> list[str]:
    files = _artifact_files(result)
    artifacts = getattr(result, "artifacts", None)
    selection = getattr(result, "selection", None)
    large_model = getattr(result, "large_model", {}) or {}
    site_typing = getattr(result, "site_typing", None)
    mol2_files = getattr(artifacts, "mol2_files", {}) if artifacts is not None else {}

    lines = _header()
    lines.extend(
        [
            "Route: MetalAA\n",
            f"Target: {target_label}\n",
            f"Protein ff: {_pro_ff_label(config)}\n",
            f"Ions charge: {getattr(config, 'ion_charges', 'unknown')}\n",
            f"Large model charge/mult: {large_model.get('charge', 'unknown')} {large_model.get('mult', 'unknown')}\n",
            f"Core residues: {_labels(getattr(selection, 'core_residues', []))}\n",
            "\nMain products:\n",
            f"  final frcmod: {_relative_path(_safe_get(files, 'frcmod'))}\n",
            f"  tleap input: {_relative_path(_safe_get(files, 'tleap_input'))}\n",
            f"  tleap PDB:   {_relative_path(_safe_get(files, 'tleap_pdb'))}\n",
            f"  mol2 files:  {_path_list(mol2_files.values() if isinstance(mol2_files, dict) else [])}\n",
        ]
    )
    lines.extend(_atom_type_lines(getattr(site_typing, "atom_type_rows", [])))
    lines.extend(_timing_lines(result))
    lines.extend(_next_step_lines(files))
    lines.extend(_tleap_status_lines(_parse_tleap_summary(_tleap_output_path(files))))
    lines.extend(_warning_lines(_metal_warnings(result)))
    lines.extend(_footer())
    return lines
