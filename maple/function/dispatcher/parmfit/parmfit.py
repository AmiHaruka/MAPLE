from __future__ import annotations

import os
from typing import Optional

from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer

class Parmfit(JobABC):
    def __init__(
        self,
        output: str,
        atoms: Atoms,
        params: Optional[dict] = None,
        method: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}
        self.method = (method or self.params.get("method") or "correction").lower()
        self.extra = extra if extra is not None else {}

    def run(self):
        with timer("Parmfit optimization"):
            self._normalize_paths()

            if self.method == "abinitio":
                parts = self.params.get("cmo", "0 1").split()
                if len(parts) not in {2, 3}:
                    raise ValueError("cmo must be '<charge> <mult>' or '<charge> <mult> <oxy>'.")
                charge, mult = map(int, parts[:2])
                self.atoms.info["charge"], self.atoms.info["mult"], self.atoms.info["spin"] = charge, mult, (mult - 1) / 2
                if len(parts) == 3:
                    self.atoms.info["oxy"] = int(parts[2])
                else:
                    self.atoms.info.pop("oxy", None)
                from .abinitio import Abinitio

                parmfit = Abinitio(output=self.output, atoms=self.atoms, params=self.params)
                return parmfit.run()
            elif self.method == "correction":
                from .correction import Correction

                parmfit = Correction(output=self.output, atoms=self.atoms, params=self.params)
                return parmfit.run()
            else:
                raise NotImplementedError(f"Other parmfit strategy '{self.method}' not implemented yet.")

    def _normalize_paths(self) -> None:
        if self.method == "correction":
            mol2_path = self.params.get("mol2")
            if not mol2_path:
                raise ValueError(
                    "parmfit(method=correction) requires the 'mol2' input file."
                )
            normalized = os.path.abspath(mol2_path)
            if not os.path.isfile(normalized):
                raise ValueError(f"parmfit input file not found for 'mol2': {mol2_path}")
            self.params["mol2"] = normalized
            return

        if self.method == "abinitio":
            pdb_path = self.params.get("pdb")
            if not pdb_path:
                raise ValueError("parmfit(method=abinitio) requires the 'pdb' input file.")
            normalized = os.path.abspath(pdb_path)
            if not os.path.isfile(normalized):
                raise ValueError(f"parmfit input file not found for 'pdb': {pdb_path}")
            self.params["pdb"] = normalized
