"""Usage: run the correction parameter refinement workflow."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter
from typing import Callable

from ase import Atoms

from ..utils.mSeminario import apply_mseminario
from ..utils.readparm import CorrectionParameterSet
from ..utils.runtime import get_cartesian_hessian, run_silent_lbfgs
from ..utils.Scan.optimizer import LBFGS
from ..utils.TorsionFit import TorsionScanRuntime, run_torsion_workflow
from .artifacts import CorrectionWorkflowResult, export_amber, export_gromacs
from .config import CorrectionConfig
from .parameters import build_parameter_set
from .report import correction_result_lines, has_parameter_changes, parameter_change_lines, stage_lines, summary_lines


@contextmanager
def _timed_stage(name: str, timings: list[tuple[str, float]]):
    start = perf_counter()
    try:
        yield
    finally:
        timings.append((name, perf_counter() - start))


def run_geometry_optimization(atoms: Atoms, output: str, config: CorrectionConfig) -> LBFGS:
    return run_silent_lbfgs(atoms, output=output, params=config.lbfgs)


def run_mseminario_stage(
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
    config: CorrectionConfig,
    apply_mseminario_fn=apply_mseminario,
) -> CorrectionParameterSet:
    stage0_parameter_set = deepcopy(parameter_set)
    hessian = get_cartesian_hessian(atoms)
    apply_mseminario_fn(
        atoms,
        hessian,
        stage0_parameter_set.bonds,
        stage0_parameter_set.angles,
        config.vib_scale,
    )
    return stage0_parameter_set


def build_torsion_runtime(config: CorrectionConfig) -> TorsionScanRuntime:
    return TorsionScanRuntime(
        max_iter=int(config.scan_opt.max_iter),
        memory=int(config.scan_opt.memory),
        curvature=float(config.scan_opt.curvature),
        max_step=float(config.scan_opt.max_step),
        backend=config.torsion.backend,
        constraint_mode=config.torsion.constraint_mode,
    )


def run_correction_workflow(
    *,
    output: str,
    atoms: Atoms,
    config: CorrectionConfig,
    log_info: Callable[[list[str]], None],
    apply_mseminario_fn=apply_mseminario,
    torsion_workflow_fn=run_torsion_workflow,
) -> CorrectionWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    log_info(stage_lines("[Correction] initial parameter assignment ..."))
    with _timed_stage("initial parameter assignment", stage_timings):
        initial_result, auto_frcmod_path = build_parameter_set(output, atoms, config)
    original_result = deepcopy(initial_result)
    log_info(summary_lines(config, original_result, auto_frcmod_path))

    log_info(stage_lines("[Correction] geometry optimization ..."))
    with _timed_stage("geometry optimization", stage_timings):
        optimizer = run_geometry_optimization(atoms, output, config)
    if not optimizer.converged:
        raise RuntimeError(f"LBFGS did not converge within {optimizer.params.max_iter} iterations.")

    log_info(stage_lines("[Correction] Hessian + mSeminario ..."))
    with _timed_stage("Hessian + mSeminario", stage_timings):
        stage0_result = run_mseminario_stage(
            atoms,
            original_result,
            config,
            apply_mseminario_fn=apply_mseminario_fn,
        )
    log_info(
        parameter_change_lines(
            "mSeminario bond/angle changes",
            original_result,
            stage0_result,
            sections=("bonds", "angles"),
        )
    )

    log_info(stage_lines("[Correction] TorsionFit ..."))
    with _timed_stage("TorsionFit", stage_timings):
        torsion = torsion_workflow_fn(
            atoms=atoms,
            output=output,
            parameter_set=stage0_result,
            original_parameter_set=original_result,
            params=config.torsion,
            runtime=build_torsion_runtime(config),
            log_info=None,
        )
    final_parameter_set = deepcopy(torsion.final_parameter_set)
    log_info(
        parameter_change_lines(
            "TorsionFit dihedral changes",
            stage0_result,
            final_parameter_set,
            sections=("dihedrals",),
        )
    )
    if has_parameter_changes(original_result, final_parameter_set, sections=("impropers", "nonbonds")):
        log_info(
            parameter_change_lines(
                "Other final changes",
                original_result,
                final_parameter_set,
                sections=("impropers", "nonbonds"),
            )
        )

    log_info(stage_lines("[Correction] export GROMACS ..."))
    with _timed_stage("export GROMACS", stage_timings):
        gromacs = export_gromacs(output, atoms, final_parameter_set)

    log_info(stage_lines("[Correction] export Amber ..."))
    with _timed_stage("export Amber", stage_timings):
        amber = export_amber(output, config, final_parameter_set)

    result = CorrectionWorkflowResult(
        initial_parameter_set=original_result,
        stage0_parameter_set=stage0_result,
        final_parameter_set=final_parameter_set,
        torsion=torsion,
        gromacs=gromacs,
        amber=amber,
        auto_frcmod=auto_frcmod_path,
        stage_timings=stage_timings,
    )
    log_info(correction_result_lines(config, result))
    return result
