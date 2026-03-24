from __future__ import annotations

from typing import Optional

from ase import Atoms

from ...jobABC import JobABC
from ..utils.readparm import parse_frcmod, parse_mol2

from maple.function.timer import timer


class Abinitio(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}

    def run(self):
        with timer("Parmfit abinitio"):
            info = ["\n", "=" * 70 + "\n", "Parmfit Abinitio".center(70) + "\n", "=" * 70 + "\n"]

            mol2_path = self.params.get("mol2")
            frcmod_path = self.params.get("frcmod")

            if mol2_path:
                mol2 = parse_mol2(mol2_path)
                info.append(f"Validated mol2 topology: {mol2_path} ({len(mol2.atoms)} atoms)\n")

            if frcmod_path:
                frcmod = parse_frcmod(frcmod_path)
                info.append(
                    "Validated frcmod template set: "
                    f"{frcmod_path} ({len(frcmod.bond_params)} bonds, "
                    f"{len(frcmod.angle_params)} angles, "
                    f"{len(frcmod.dihedral_params)} dihedrals, "
                    f"{len(frcmod.improper_params)} impropers, "
                    f"{len(frcmod.nonbond_params)} nonbond types)\n"
                )

            info.append(
                "Abinitio parameter fitting workflow is not implemented yet; "
                "the dispatcher currently validates optional auxiliary inputs and preserves "
                "the MAPLE inp geometry as the source for future Hessian-based fitting.\n"
            )
            self.log_info(info)
