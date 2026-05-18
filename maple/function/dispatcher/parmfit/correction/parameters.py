"""Usage: build correction parameter sets from user mol2 and AmberTools frcmod."""

from __future__ import annotations

import os
import shutil

from ase import Atoms

from ..utils import interface as amber_interface
from ..utils.readparm import CorrectionParameterSet, build_correction_parameter_set
from ..utils.runtime import parmfit_output_dir
from .config import CorrectionConfig


def build_parameter_set(output: str, atoms: Atoms, config: CorrectionConfig) -> tuple[CorrectionParameterSet, str]:
    base = os.path.splitext(os.path.basename(output))[0]
    workdir = os.path.dirname(config.mol2)
    auto_frcmod = amber_interface.run_parmchk2(
        os.path.basename(config.mol2),
        {"residue_name": f"{base}_original"},
        True,
        workdir,
    ).frcmod_path
    final_auto_frcmod = os.path.join(parmfit_output_dir(output), f"{base}_original.frcmod")
    shutil.move(auto_frcmod, final_auto_frcmod)
    return build_correction_parameter_set(atoms, config.mol2, final_auto_frcmod), final_auto_frcmod
