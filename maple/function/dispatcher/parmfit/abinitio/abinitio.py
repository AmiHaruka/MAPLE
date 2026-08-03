from __future__ import annotations

from typing import Optional

from ase import Atoms

from ...jobABC import JobABC
from .report import format_abinitio_summary, format_pdb_read_diagnostics
from ..utils.structure import classify_kind, get_resid_label
from ..utils.context import find_unique_residue
from ..utils.MetalAA import parse_metal_abinitio_config, run_metal_abinitio
from ..utils.NCAA import parse_ncaa_abinitio_config, run_ncaa_abinitio

from maple.function.timer import timer
from maple.function.read.filereader.pdb_reader import read_pdb_result


class Abinitio(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}

    def run(self):
        with timer("Parmfit abinitio"):
            info = ["\n", "=" * 70 + "\n", "Parmfit Abinitio".center(70) + "\n", "=" * 70 + "\n"]
            raw = dict(self.params or {})
            pdb = raw["pdb"].strip()
            target = raw["target"].strip()

            charge = self.atoms.info.get("charge")
            mult = self.atoms.info.get("mult")
            if charge is None or mult is None:
                raise ValueError("Ab initio parmfit requires charge and multiplicity on source atoms.")
            oxy = self.atoms.info.get("oxy")

            altloc_selectors = [target, *raw.get("keep", "").split(), *raw.get("add_resid", "").split()]
            pdb_result = read_pdb_result(
                pdb,
                prom=raw.get("prom", "ff14SB"),
                altloc_selectors=altloc_selectors,
            )
            structure = pdb_result.structure
            target_residue = find_unique_residue(structure, target, label="Target residue")
            target_kind = classify_kind(target_residue)
            target_charge = None
            if target_kind == "protein":
                target_residue["net_charge"] = int(charge)
                target_charge = (get_resid_label(target_residue), int(charge))

            info.append(f"Target: {get_resid_label(target_residue)} ({target_kind})\n")
            info.extend(format_pdb_read_diagnostics(pdb_result.diagnostics, target_charge=target_charge))
            self.log_info(info)

            if target_kind == "ion":
                config = parse_metal_abinitio_config(
                    raw,
                    pdb_path=pdb,
                    target=target,
                    charge=int(charge),
                    mult=int(mult),
                    oxy=None if oxy is None else int(oxy),
                    target_residue=target_residue,
                )
                result = run_metal_abinitio(
                    output=self.output,
                    source_atoms=self.atoms,
                    structure=structure,
                    config=config,
                    log_info=self.log_info,
                )
                self.log_info(
                    format_abinitio_summary(
                        route="MetalAA",
                        target_label=get_resid_label(target_residue),
                        target_kind=target_kind,
                        config=config,
                        result=result,
                    )
                )
                return result

            if target_kind == "protein":
                config = parse_ncaa_abinitio_config(
                    raw,
                    pdb_path=pdb,
                    target=target,
                    charge=int(charge),
                    mult=int(mult),
                )
                result = run_ncaa_abinitio(
                    output=self.output,
                    source_atoms=self.atoms,
                    structure=structure,
                    target_residue=target_residue,
                    config=config,
                    log_info=self.log_info,
                )
                self.log_info(
                    format_abinitio_summary(
                        route="NCAA",
                        target_label=get_resid_label(target_residue),
                        target_kind=target_kind,
                        config=config,
                        result=result,
                    )
                )
                return result

            raise NotImplementedError(
                f"Unsupported abinitio parmfit target kind '{target_kind}' for residue {get_resid_label(target_residue)}."
            )
