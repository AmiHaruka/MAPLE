"""Usage: run the torsion scan and fitting workflow."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Callable

import numpy as np
from ase import Atoms

from ..mechanics import build_mm_topology_cache, evaluate_mm_energy
from ..readparm import CorrectionParameterSet
from ..runtime import parmfit_work_prefix
from ..Scan import run_silent_scan
from .topology import (
    representative_dihedral_for_center_bond,
    resolve_torsion_center_bonds,
)
from .fit import run_loss_mode
from .records import TorsionScanData, TorsionScanRuntime, TorsionWorkflowResult
from .ensemble import build_torsion_local_ensemble
from .scanio import read_scan_xyz
from .config import TorsionFitParams, normalize_center_bond
from .report import format_torsion_stage1_lines, format_torsion_stage2_lines


_LOSS_MODE_MM_ORIG_FILTER_KCAL = 50.0


def _scan_output_paths(output: str, center_bond: tuple[int, int]) -> tuple[str, str]:
    center = normalize_center_bond(center_bond)
    prefix = f"{parmfit_work_prefix(output, 'torsionfit')}_torsionfit_{center[0]}-{center[1]}"
    return prefix + ".out", prefix + "_scan_final.xyz"


def _expected_scan_angles(
    atoms: Atoms,
    params: TorsionFitParams,
    representative_dihedral: tuple[int, int, int, int],
) -> np.ndarray:
    atom_indices = [int(atom) - 1 for atom in representative_dihedral]
    initial_angle = float(atoms.get_dihedral(*atom_indices))
    step = float(params.torsion_step_deg)
    return np.asarray([initial_angle + step * idx for idx in range(int(params.torsion_steps) + 1)], dtype=float)


def _read_cached_scan_if_valid(
    scan_xyz: str,
    atoms: Atoms,
    params: TorsionFitParams,
    representative_dihedral: tuple[int, int, int, int],
) -> TorsionScanData | None:
    if not os.path.isfile(scan_xyz):
        return None
    try:
        scan_data = read_scan_xyz(scan_xyz)
    except Exception:
        return None

    expected_count = int(params.torsion_steps) + 1
    if len(scan_data.frames) != expected_count:
        return None

    expected_symbols = atoms.get_chemical_symbols()
    for frame in scan_data.frames:
        if len(frame) != len(expected_symbols) or frame.get_chemical_symbols() != expected_symbols:
            return None

    expected_angles = _expected_scan_angles(atoms, params, representative_dihedral)
    cached_angles = np.asarray(scan_data.angles_deg, dtype=float)
    if cached_angles.shape != expected_angles.shape:
        return None
    if not np.allclose(cached_angles, expected_angles, rtol=0.0, atol=1.0e-2):
        return None
    return scan_data


def _run_center_bond_scan(
    atoms: Atoms,
    output: str,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    center_bond: tuple[int, int],
    representative_dihedral: tuple[int, int, int, int],
) -> str:
    scan_output, _ = _scan_output_paths(output, center_bond)
    result = run_silent_scan(
        output=scan_output,
        atoms=atoms,
        constraints=[[*representative_dihedral, params.torsion_step_deg, params.torsion_steps]],
        params=runtime.to_scan_params(),
        method=runtime.backend,
        constraint_mode=runtime.constraint_mode,
    )
    return result.xyz_path


def _filtered_scan_data(scan_data: TorsionScanData, keep_mask: np.ndarray) -> TorsionScanData:
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != len(scan_data.frames):
        raise ValueError("Scan-point filter mask must match the scan length.")
    if np.all(mask):
        return scan_data

    qm_hartree = np.asarray(scan_data.qm_hartree, dtype=float)[mask]
    qm_kcal = np.asarray(scan_data.qm_kcal, dtype=float)[mask]
    angles_deg = np.asarray(scan_data.angles_deg, dtype=float)[mask]
    frames = [frame for frame, keep in zip(scan_data.frames, mask) if keep]
    ref_idx = int(np.argmin(qm_kcal))
    return TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_hartree,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path=scan_data.source_path,
        ref_idx=ref_idx,
        qm_rel=qm_kcal - qm_kcal[ref_idx],
    )


def _filter_loss_mode_scan_points(
    scan_data: TorsionScanData,
    original_parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    *,
    topology_cache=None,
) -> tuple[TorsionScanData, str | None, np.ndarray]:
    mm_orig_total = np.asarray(
        [evaluate_mm_energy(atoms, original_parameter_set, topology_cache=topology_cache).total for atoms in scan_data.frames],
        dtype=float,
    )
    mm_orig_filter_rel = mm_orig_total - float(np.min(mm_orig_total))
    keep_mask = np.asarray(mm_orig_filter_rel <= _LOSS_MODE_MM_ORIG_FILTER_KCAL, dtype=bool)
    if np.all(keep_mask):
        mm_orig_report_rel = mm_orig_total - mm_orig_total[int(scan_data.ref_idx)]
        return scan_data, None, mm_orig_report_rel

    dropped_angles = np.asarray(scan_data.angles_deg, dtype=float)[~keep_mask]
    filtered = _filtered_scan_data(scan_data, keep_mask)
    filtered_mm_orig_total = mm_orig_total[keep_mask]
    mm_orig_report_rel = filtered_mm_orig_total - filtered_mm_orig_total[int(filtered.ref_idx)]
    warning = (
        f"loss scan filter: center bond {normalize_center_bond(center_bond)} dropped "
        f"{int(np.count_nonzero(~keep_mask))} point(s) with MM_orig_rel > {_LOSS_MODE_MM_ORIG_FILTER_KCAL:.1f} "
        f"at angles [{', '.join(f'{float(angle):.4f}' for angle in dropped_angles)}]"
    )
    return filtered, warning, mm_orig_report_rel


def run_torsion_workflow(
    *,
    atoms: Atoms,
    output: str,
    parameter_set: CorrectionParameterSet,
    original_parameter_set: CorrectionParameterSet | None = None,
    params: TorsionFitParams,
    runtime: TorsionScanRuntime,
    center_bond_filter: Callable[[tuple[int, int]], bool] | None = None,
    mobile_atoms=None,
    log_info: Callable[[list[str]], None] | None = None,
) -> TorsionWorkflowResult:
    base_parameter_set = deepcopy(parameter_set)
    original_parameter_set = deepcopy(original_parameter_set) if original_parameter_set is not None else deepcopy(base_parameter_set)
    if not params.enabled:
        if log_info is not None:
            log_info(format_torsion_stage1_lines(params, []))
            log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
            log_info(format_torsion_stage2_lines(params, []))
        return TorsionWorkflowResult(
            stage1_parameter_set=None,
            final_parameter_set=deepcopy(base_parameter_set),
        )

    topology_cache = build_mm_topology_cache(base_parameter_set)
    center_bonds, warnings = resolve_torsion_center_bonds(
        base_parameter_set,
        params,
        topology_cache=topology_cache,
    )
    if center_bond_filter is not None:
        center_bonds = [bond for bond in center_bonds if center_bond_filter(normalize_center_bond(bond))]
    if not center_bonds:
        empty_result = deepcopy(base_parameter_set)
        if log_info is not None:
            log_info(format_torsion_stage1_lines(params, warnings, has_center_bonds=False))
            log_info(["\n[Stage 2] Running stage-2 global torsion refinement...\n"])
            log_info(format_torsion_stage2_lines(params, [], has_center_bonds=False))
        return TorsionWorkflowResult(
            stage1_parameter_set=empty_result,
            final_parameter_set=deepcopy(empty_result),
            center_bonds=[],
            warnings=warnings,
        )

    scan_data_map = {}
    scan_xyz_map: dict[tuple[int, int], str] = {}
    scan_mm_orig_rel_map: dict[tuple[int, int], np.ndarray] = {}
    original_topology_cache = build_mm_topology_cache(original_parameter_set)

    for center_bond in center_bonds:
        representative = representative_dihedral_for_center_bond(
            base_parameter_set,
            center_bond,
            topology_cache=topology_cache,
        )
        _, scan_xyz = _scan_output_paths(output, center_bond)
        scan_data = _read_cached_scan_if_valid(scan_xyz, atoms, params, representative.atoms)
        if scan_data is not None:
            if log_info is not None:
                log_info([f"reuse existing torsion scan xyz: {scan_xyz}\n"])
        else:
            scan_xyz = _run_center_bond_scan(
                atoms,
                output,
                params,
                runtime,
                center_bond,
                representative.atoms,
            )
            scan_data = read_scan_xyz(scan_xyz)
        scan_data, warning, mm_orig_rel = _filter_loss_mode_scan_points(
            scan_data,
            original_parameter_set,
            center_bond,
            topology_cache=original_topology_cache,
        )
        if warning is not None:
            warnings.append(warning)
        scan_data_map[normalize_center_bond(center_bond)] = scan_data
        scan_xyz_map[normalize_center_bond(center_bond)] = scan_xyz
        scan_mm_orig_rel_map[normalize_center_bond(center_bond)] = np.asarray(mm_orig_rel, dtype=float)

    ensemble_result = None
    if params.torsion_ensemble:
        ensemble_result = build_torsion_local_ensemble(
            atoms=atoms,
            parameter_set=base_parameter_set,
            center_bonds=center_bonds,
            scan_data_map=scan_data_map,
            params=params,
            output=output,
            mobile_atoms=mobile_atoms,
            log_info=log_info,
        )
        warnings.extend(ensemble_result.warnings)

    if log_info is not None and warnings:
        log_info(format_torsion_stage1_lines(params, warnings))

    loss_kwargs = {
        "base_parameter_set": base_parameter_set,
        "original_parameter_set": original_parameter_set,
        "center_bonds": center_bonds,
        "scan_data_map": scan_data_map,
        "scan_xyz_map": scan_xyz_map,
        "scan_mm_orig_rel_map": scan_mm_orig_rel_map,
        "params": params,
        "topology_cache": topology_cache,
        "log_info": log_info,
    }
    if ensemble_result is not None:
        loss_kwargs["ensemble_result"] = ensemble_result
    result = run_loss_mode(**loss_kwargs)

    result.warnings = list(warnings)
    result.center_bonds = [normalize_center_bond(bond) for bond in center_bonds]
    result.scan_xyz = dict(scan_xyz_map)
    result.ensemble_xyz = dict(ensemble_result.xyz_paths) if ensemble_result is not None else {}
    return result
