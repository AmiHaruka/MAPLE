from __future__ import annotations

from typing import Optional

from ase import Atoms

from ....jobABC import JobABC
from ...runconfig import as_tracked, write_parmfit_run
from ..context import find_unique_residue
from ..structure import get_resid_label
from .config import parse_metal_abinitio_config
from .report import format_metal_summary, format_pdb_read_diagnostics
from .workflow import run_metal_abinitio

from maple.function.timer import timer
from maple.function.read.filereader.pdb_reader import read_pdb_result


def _split_entries(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [token.strip() for token in str(value).split(",") if token.strip()]


class MetalAA(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}

    def run(self):
        with timer("Parmfit MetalAA"):
            info = ["\n", "=" * 70 + "\n", "Parmfit MetalAA".center(70) + "\n", "=" * 70 + "\n"]
            raw = as_tracked(self.params)
            pdb = str(raw.get("pdb", "")).strip()
            target = str(raw.get("ion_resids", "")).strip()
            if not pdb:
                raise ValueError("parmfit(method=metalaa) requires a PDB block: PDB <path>.")
            if not target:
                raise ValueError("parmfit(method=metalaa) requires an ion residue selector, e.g. ion_resids=A301.")

            altloc_selectors = [
                target,
                *_split_entries(raw.get("keep", "")),
                *_split_entries(raw.get("add_resid", "")),
                *_split_entries(raw.get("ncaa_resids", "")),
                *_split_entries(raw.get("lig_resids", "")),
            ]
            pdb_result = read_pdb_result(
                pdb,
                pro_ff=raw.get("pro_ff", "ff14SB"),
                altloc_selectors=altloc_selectors,
            )
            structure = pdb_result.structure
            target_residue = find_unique_residue(structure, target, label="Target residue")
            target_label = get_resid_label(target_residue)

            info.append(f"Target: {target_label} (ion)\n")
            info.extend(
                format_pdb_read_diagnostics(
                    pdb_result.diagnostics,
                    target_label=target_label,
                    target_charge=None,
                )
            )
            self.log_info(info)

            config = parse_metal_abinitio_config(
                raw,
                pdb_path=pdb,
                ion_resids=target,
                target_residue=target_residue,
            )
            write_parmfit_run(
                self.output,
                "metalaa",
                raw,
                warn=lambda message: self.log_info([f"WARNING: {message}\n"]),
            )
            result = run_metal_abinitio(
                output=self.output,
                source_atoms=self.atoms,
                structure=structure,
                config=config,
                log_info=self.log_info,
            )
            self.log_info(
                format_metal_summary(
                    target_label=get_resid_label(target_residue),
                    config=config,
                    result=result,
                )
            )
            return result
