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
        method: Optional[str] = "correction",
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
                from .abinitio import Abinitio

                parmfit = Abinitio(output=self.output, atoms=self.atoms, params=self.params)
                parmfit.run()
            elif self.method == "correction":
                from .correction import Correction

                parmfit = Correction(output=self.output, atoms=self.atoms, params=self.params)
                parmfit.run()
            else:
                raise NotImplementedError(f"Parmfit strategy '{self.method}' not implemented yet.")

    def _normalize_paths(self) -> None:
        base_dir = os.getcwd()
        input_path = self.extra.get("input_path")
        if isinstance(input_path, str):
            base_dir = os.path.dirname(os.path.abspath(input_path))

        for key in ("mol2", "frcmod"):
            path = self.params.get(key)
            if path is None:
                continue
            if not isinstance(path, str):
                raise ValueError(f"parmfit '{key}' must be a file path string.")
            if not os.path.isabs(path):
                path = os.path.abspath(os.path.join(base_dir, path))
            self.params[key] = path

        if self.method == "correction":
            missing = [key for key in ("mol2", "frcmod") if not self.params.get(key)]
            if missing:
                raise ValueError(
                    "parmfit(method=correction) requires the following file inputs: "
                    + ", ".join(missing)
                )

        for key in ("mol2", "frcmod"):
            path = self.params.get(key)
            if path and not os.path.isfile(path):
                raise ValueError(f"parmfit input file not found for '{key}': {path}")
