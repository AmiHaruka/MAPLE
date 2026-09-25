"""Usage: build correction parameter sets from user mol2 and AmberTools frcmod."""

from __future__ import annotations

import os
import shutil

from ase import Atoms

from .. import interface
from ..readparm import CorrectionParameterSet, build_correction_paramset
from ..runtime import parmfit_output_dir
from .config import CorrectionConfig


def build_init_parmset(output: str, atoms: Atoms, config: CorrectionConfig) -> tuple[CorrectionParameterSet, str]:
    base = os.path.splitext(os.path.basename(output))[0]
    workdir = os.path.dirname(config.mol2)
    init_frcmod = interface.run_parmchk2(
        os.path.basename(config.mol2),
        {"residue_name": f"{base}_original"},
        True,
        workdir,
    ).frcmod_path
    final_init_frcmod = os.path.join(parmfit_output_dir(output), f"{base}_original.frcmod")
    shutil.move(init_frcmod, final_init_frcmod)
    return build_correction_paramset(atoms, config.mol2, final_init_frcmod, config.torsion.p_thresh), final_init_frcmod
