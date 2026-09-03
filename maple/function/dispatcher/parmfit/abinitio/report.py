"""Usage: format user-readable summaries for parmfit abinitio routes."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..utils.structure import get_resid_label

_WIDTH = 70


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


def _names(values: Iterable[str] | None) -> str:
    items = _deduped(values)
    return ", ".join(items) if items else "none"


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
        "large RESP/Gaussian ESP": "large RESP/Gaussian ESP",
        "Hessian/mSeminario/frcmod export": "Hessian + mSeminario",
        "Hessian/Seminario/frcmod export": "Hessian + Seminario",
        "large optimization": "large optimization",
        "site export": "site export",
        "site selection/model build": "site selection/model build",
        "model preparation/reference optimization": "model preparation + reference",
        "charge fitting": "charge fitting",
        "AmberTools template build": "AmberTools template build",
        "mSeminario setup/Hessian": "Hessian + mSeminario",
        "Seminario setup/Hessian": "Hessian + Seminario",
        "TorsionFit": "TorsionFit",
        "final export": "final export",
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


def _ncaa_prom_note_lines(config) -> list[str]:
    prom = getattr(config, "prom", "ff14SB")
    if prom != "ff19SB":
        return []
    return [
        "\nff19SB note:\n",
        "  NCAA boundary exact terms follow ff19SB atom types.\n",
        "  CMAP terms are not generated for the NCAA residue; inspect tleap output before production MD.\n",
    ]


def _header() -> list[str]:
    return [
        "\n",
        "=" * _WIDTH + "\n",
        "PARMFIT ABINITIO RESULT".center(_WIDTH) + "\n",
        "=" * _WIDTH + "\n",
        "Status: completed\n",
    ]


def _footer() -> list[str]:
    return ["=" * _WIDTH + "\n"]


def _prom_label(config) -> str:
    return getattr(config, "prom", "ff14SB")


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


def _format_metal_summary(*, target_label: str, config, result) -> list[str]:
    files = _artifact_files(result)
    artifacts = getattr(result, "artifacts", None)
    selection = getattr(result, "selection", None)
    large_model = getattr(result, "large_model", {}) or {}
    site_typing = getattr(result, "site_typing", None)
    mol2_files = getattr(artifacts, "mol2_files", {}) if artifacts is not None else {}
    oxidation = getattr(config, "oxy", None)
    if oxidation is None:
        oxidation = getattr(config, "charge", "unknown")

    lines = _header()
    lines.extend(
        [
            "Route: MetalAA\n",
            f"Target: {target_label}\n",
            f"Protein model: {_prom_label(config)}\n",
            f"Charge/mult/oxidation: {getattr(config, 'charge', 'unknown')} {getattr(config, 'mult', 'unknown')} {oxidation}\n",
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


def _format_ncaa_summary(*, target_label: str, config, result) -> list[str]:
    files = _artifact_files(result)
    identity = getattr(result, "identity", None)
    charge_result = getattr(result, "charge_result", None)
    refined_prepin = _safe_get(files, "refined_prepin")
    refined_frcmod = _safe_get(files, "refined_frcmod")
    charge_method = getattr(charge_result, "method", "unknown")
    charge_detail = str(getattr(charge_result, "detail", "") or "")
    if charge_method in {"model", "antechamber"} and charge_detail:
        charge_method = f"{charge_method} ({charge_detail})"

    lines = _header()
    lines.extend(
        [
            "Route: NCAA\n",
            f"Target: {target_label}\n",
            f"Residue name: {getattr(config, 'rn', 'unknown')}\n",
            f"Chirality: {getattr(identity, 'chirality', 'unknown')}\n",
            f"Protein model: {_prom_label(config)}\n",
            f"Charge/mult: {getattr(config, 'charge', 'unknown')} {getattr(config, 'mult', 'unknown')}\n",
            f"Charge fitting: {charge_method}\n",
            *(
                [f"Charge level: {charge_detail}\n"]
                if getattr(charge_result, "method", None) == "resp"
                else []
            ),
            f"Target charge: {getattr(charge_result, 'target_charge', 'unknown')}\n",
            f"Actual charge: {float(getattr(charge_result, 'actual_charge', float('nan'))):.8f}\n",
            f"Charge MOL2: {_relative_path(getattr(charge_result, 'work_mol2', 'not written'))}\n",
            "\nMain products:\n",
            f"  refined prepin: {_relative_path(refined_prepin)}\n",
            f"  refined frcmod: {_relative_path(refined_frcmod)}\n",
            f"  tleap PDB:      {_relative_path(_safe_get(files, 'tleap_pdb'))}\n",
            f"  tleap input:    {_relative_path(_safe_get(files, 'tleap_input'))}\n",
        ]
    )
    lines.extend(_timing_lines(result))
    lines.extend(_next_step_lines(files))
    lines.extend(_tleap_status_lines(_parse_tleap_summary(_tleap_output_path(files))))
    lines.extend(_ncaa_prom_note_lines(config))
    lines.extend(_warning_lines([]))
    lines.extend(_footer())
    return lines


def format_abinitio_summary(*, route: str, target_label: str, target_kind: str, config, result) -> list[str]:
    del target_kind
    if route == "MetalAA":
        return _format_metal_summary(target_label=target_label, config=config, result=result)
    if route == "NCAA":
        return _format_ncaa_summary(target_label=target_label, config=config, result=result)
    raise ValueError(f"Unsupported abinitio summary route {route!r}.")
