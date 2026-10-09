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
        self.method = (method or self.params.get("method") or ("" if self.params.get("input") else "refinement")).lower()
        self.extra = extra if extra is not None else {}

    def run(self):
        with timer("Parmfit optimization"):
            self._load_external_config()
            if not self.method:
                raise ValueError("parmfit requires method=refinement|ncaa|metalaa in #parmfit(...) when input=<config> is used.")
            self._normalize_paths()

            if self.method.lower() == "ncaa":
                from .utils.NCAA import NCAA

                parmfit = NCAA(output=self.output, atoms=self.atoms, params=self.params)
                return parmfit.run()
            elif self.method.lower() == "metalaa":
                from .utils.MetalAA import MetalAA

                parmfit = MetalAA(output=self.output, atoms=self.atoms, params=self.params)
                return parmfit.run()
            elif self.method.lower() == "refinement":
                # Old setting
                #self._apply_cmo(use_oxy=False, info_fallback=True)
                self._prepare_refinement_mol2()
                from .utils.Refinement import Refinement

                parmfit = Refinement(output=self.output, atoms=self.atoms, params=self.params)
                return parmfit.run()
            else:
                raise NotImplementedError(f"Other parmfit strategy '{self.method}' not implemented yet.")


#####################
###  Load Helpers ###
#####################

    def _apply_cmo(self, *, use_oxy: bool, info_fallback: bool) -> None:
        explicit = self.params.get("cmo") or None
        if explicit is None:
            charge = int(self.atoms.info.get("charge", 0)) if info_fallback else 0
            mult = int(self.atoms.info.get("mult", 1)) if info_fallback else 1
            parts = (str(charge), str(mult))
        else:
            parts = str(explicit).split()
            if len(parts) not in {2, 3}:
                raise ValueError("cmo must be '<charge> <mult>' or '<charge> <mult> <oxy>'.")
            charge, mult = map(int, parts[:2])

        self.atoms.info["charge"] = charge
        self.atoms.info["mult"] = mult
        self.atoms.info["spin"] = (mult - 1) / 2
        if use_oxy and len(parts) == 3:
            self.atoms.info["oxy"] = int(parts[2])
        else:
            self.atoms.info.pop("oxy", None)

    def _load_external_config(self) -> None:
        config_ref = self.params.get("input")
        if not config_ref:
            return

        from maple.function.read.filereader.parmfit_reader import ParmfitReader

        input_path = self.extra.get("input_path") if isinstance(self.extra, dict) else None
        base_dir = os.path.dirname(os.path.abspath(input_path)) if input_path else os.getcwd()
        config_path = ParmfitReader.resolve_path(str(config_ref), base_dir=base_dir)
        loaded = ParmfitReader(config_path)

        runtime_params = {}
        if "pdb" in self.params:
            runtime_params["pdb"] = self.params["pdb"]
        self.params = dict(loaded)
        self.params.update(runtime_params)
        self.params.pop("input", None)

    def _input_base_dir(self) -> str:
        input_path = self.extra.get("input_path") if isinstance(self.extra, dict) else None
        if input_path:
            return os.path.dirname(os.path.abspath(input_path))
        return os.getcwd()

    def _resolve_input_file(self, key: str) -> None:
        path = self.params.get(key)
        if not path:
            return
        normalized = path if os.path.isabs(path) else os.path.join(self._input_base_dir(), path)
        normalized = os.path.abspath(normalized)
        if not os.path.isfile(normalized):
            if self.method == "refinement":
                raise ValueError(
                    f"parmfit(method=refinement) requires the following file inputs: {key} "
                    f"(not found: {path})"
                )
            raise ValueError(f"parmfit input file not found for '{key}': {path}")
        self.params[key] = normalized

    def _normalize_paths(self) -> None:
        if self.method == "refinement":
            if self.params.get("mol2"):
                self._resolve_input_file("mol2")
            return

        if self.method in ("ncaa", "metalaa"):
            pdb_path = self.params.get("pdb")
            if not pdb_path:
                raise ValueError("parmfit(method=ncaa/metalaa) requires a PDB block: PDB <path>.")
            self._resolve_input_file("pdb")

    def _prepare_refinement_mol2(self) -> None:
        """Auto-generate the GAFF2 mol2 via antechamber when mol2= is not given."""
        if self.params.get("mol2"):
            return
        if str(self.params.get("chg_fit", "none")).strip().lower() in ("", "none"):
            self.params["chg_fit"] = "abcg2"
        from ase.io import write as ase_write

        from .utils.interface import run_antechamber
        from .utils.runtime import parmfit_output_dir

        workdir = parmfit_output_dir(self.output)
        stem = os.path.splitext(os.path.basename(self.output))[0]
        ase_write(os.path.join(workdir, "ante.pdb"), self.atoms)
        charge_file = os.path.join(workdir, "ante.chg")
        with open(charge_file, "w", encoding="utf-8") as handle:
            for _ in self.atoms:
                handle.write("0.000000\n")

        run_antechamber(
            "ante.pdb",
            {
                "residue_name": "LIG",
                "net_charge": int(self.atoms.info.get("charge", 0)),
                "multiplicity": int(self.atoms.info.get("mult", 1)),
            },
            workdir,
            input_format="pdb",
            output_format="mol2",
            charge_mode="rc",
            charge_file=charge_file,
        )
        mol2_path = os.path.join(workdir, f"{stem}_ff.mol2")
        os.replace(os.path.join(workdir, "LIG.mol2"), mol2_path)
        self.params["mol2"] = mol2_path

