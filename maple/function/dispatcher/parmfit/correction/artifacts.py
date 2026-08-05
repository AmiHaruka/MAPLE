"""Usage: hold correction workflow results and write correction artifact files."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from ase import Atoms

from ..utils.outputparm import format_corr_tleap, write_amber_files, write_gromacs_files
from ..utils.readparm import CorrectionParameterSet
from ..utils.runtime import parmfit_output_dir
from ..utils.TorsionFit import TorsionWorkflowResult
from ..utils.chargefit import ChargeFitResult
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
    tleap_in: str = ""


@dataclass(frozen=True)
class CorrectionWorkflowResult:
    init_parmset: CorrectionParameterSet
    stage0_parmset: CorrectionParameterSet
    final_parmset: CorrectionParameterSet
    torsion: TorsionWorkflowResult
    gromacs: GromacsExportResult
    amber: AmberExportResult
    init_frcmod: str
    stage_timings: list[tuple[str, float]] = field(default_factory=list)
    charge_result: ChargeFitResult | None = None
    charge_timing: float | None = None
    mlip_stage0_parmset: CorrectionParameterSet | None = None
    mlip_final_parmset: CorrectionParameterSet | None = None
    mlip_torsion: TorsionWorkflowResult | None = None
    mlip_gromacs: GromacsExportResult | None = None
    mlip_amber: AmberExportResult | None = None
    mlip_charge_result: ChargeFitResult | None = None
    mlip_charge_timing: float | None = None


def export_gromacs(
    output: str,
    atoms: Atoms,
    parmset: CorrectionParameterSet,
    *,
    output_suffix: str = "",
) -> GromacsExportResult:
    base = os.path.splitext(os.path.basename(output))[0]
    output_base = os.path.join(parmfit_output_dir(output), base + str(output_suffix))
    gromacs_top, gromacs_gro, gromacs_meta = write_gromacs_files(
        parmset,
        atoms,
        output_base,
    )
    return GromacsExportResult(
        top=gromacs_top,
        gro=gromacs_gro,
        warnings=list(gromacs_meta.get("warnings", [])),
        omitted_counts=dict(gromacs_meta.get("omitted_counts", {})),
        written_sections=list(gromacs_meta.get("written_sections", [])),
    )


def export_amber(
    output: str,
    atoms: Atoms,
    config: CorrectionConfig,
    parmset: CorrectionParameterSet,
    *,
    use_refined_parameters: bool = True,
    source_frcmod: str = "",
    output_suffix: str = "",
) -> AmberExportResult:
    base = os.path.splitext(os.path.basename(output))[0]
    output_base = os.path.join(parmfit_output_dir(output), base + str(output_suffix))
    if not use_refined_parameters:
        maple_mol2 = output_base + "_maple.mol2"
        maple_frcmod = output_base + "_maple.frcmod"
        maple_tleap = output_base + "_maple_tleap.in"
        shutil.copyfile(config.mol2, maple_mol2)
        shutil.copyfile(source_frcmod, maple_frcmod)
        # Types are still the stock gaff2 ones here, so the script needs no
        # addAtomTypes block -- but it is written all the same, so every Amber
        # export ships the same set of files.
        gas_base = os.path.basename(output_base) + "_maple_gas"
        with open(maple_tleap, "w") as handle:
            handle.writelines(
                format_corr_tleap(
                    [],
                    mol2_name=os.path.basename(maple_mol2),
                    frcmod_name=os.path.basename(maple_frcmod),
                    prmtop_name=gas_base + ".prmtop",
                    inpcrd_name=gas_base + ".inpcrd",
                )
            )
        return AmberExportResult(mol2=maple_mol2, frcmod=maple_frcmod, tleap_in=maple_tleap)
    maple_mol2, maple_frcmod, maple_tleap = write_amber_files(
        parmset,
        atoms,
        config.mol2,
        output_base,
    )
    return AmberExportResult(mol2=maple_mol2, frcmod=maple_frcmod, tleap_in=maple_tleap)
