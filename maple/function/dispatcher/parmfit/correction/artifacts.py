"""Usage: hold correction workflow results and write correction artifact files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ase import Atoms

from ..utils.outputparm import write_amber_files, write_gromacs_files
from ..utils.readparm import CorrectionParameterSet
from ..utils.runtime import parmfit_output_dir
from ..utils.TorsionFit import TorsionWorkflowResult
from .config import CorrectionConfig


@dataclass(frozen=True)
class GromacsExportResult:
    top: str
    gro: str
    warnings: list[str] = field(default_factory=list)
    omitted_counts: dict = field(default_factory=dict)
    written_sections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AmberExportResult:
    mol2: str
    frcmod: str


@dataclass(frozen=True)
class CorrectionWorkflowResult:
    initial_parameter_set: CorrectionParameterSet
    stage0_parameter_set: CorrectionParameterSet
    final_parameter_set: CorrectionParameterSet
    torsion: TorsionWorkflowResult
    gromacs: GromacsExportResult
    amber: AmberExportResult
    auto_frcmod: str
    stage_timings: list[tuple[str, float]] = field(default_factory=list)


def export_gromacs(
    output: str,
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
) -> GromacsExportResult:
    base = os.path.splitext(os.path.basename(output))[0]
    gromacs_top, gromacs_gro, gromacs_meta = write_gromacs_files(
        parameter_set,
        atoms,
        os.path.join(parmfit_output_dir(output), base),
    )
    return GromacsExportResult(
        top=gromacs_top,
        gro=gromacs_gro,
        warnings=list(gromacs_meta.get("warnings", [])),
        omitted_counts=dict(gromacs_meta.get("omitted_counts", {})),
        written_sections=list(gromacs_meta.get("written_sections", [])),
    )


def export_amber(output: str, config: CorrectionConfig, parameter_set: CorrectionParameterSet) -> AmberExportResult:
    base = os.path.splitext(os.path.basename(output))[0]
    maple_mol2, maple_frcmod = write_amber_files(
        parameter_set,
        config.mol2,
        os.path.join(parmfit_output_dir(output), base),
    )
    return AmberExportResult(mol2=maple_mol2, frcmod=maple_frcmod)
