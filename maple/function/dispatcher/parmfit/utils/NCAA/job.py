from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from ase import Atoms

from ....jobABC import JobABC
from ...runconfig import as_tracked, write_parmfit_run
from .. import interface
from ..amber_templates import required_template_leaprcs
from ..context import find_unique_residue
from ..outputparm import allocate_maple_atom_types
from ..structure import get_resid_key, get_resid_label
from .artifacts import write_ncaa_tleap_input, write_ncaa_tleap_pdb
from .config import parse_ncaa_abinitio_config
from .report import format_ncaa_summary, format_pdb_read_diagnostics
from .workflow import run_ncaa_abinitio

from maple.function.timer import timer
from maple.function.read.filereader.pdb_reader import read_pdb_result


def _split_entries(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [token.strip() for token in str(value).split(",") if token.strip()]


class NCAA(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.params = params if params is not None else {}

    def run(self):
        with timer("Parmfit NCAA"):
            info = ["\n", "=" * 70 + "\n", "Parmfit NCAA".center(70) + "\n", "=" * 70 + "\n"]
            raw = as_tracked(self.params)
            pdb = str(raw.get("pdb", "")).strip()
            if not pdb:
                raise ValueError("parmfit(method=ncaa) requires a PDB block: PDB <path>.")
            ncaa_resids = _split_entries(raw.get("ncaa_resids", ""))
            if not ncaa_resids:
                raise ValueError(
                    "parmfit(method=ncaa) requires ncaa_resids=... (one residue selector per entry, comma-separated)."
                )
            config = parse_ncaa_abinitio_config(raw, pdb_path=pdb)
            selectors = list(config.ncaa_resids)
            charges = list(config.ncaa_charges)
            rns = list(config.ncaa_resnames)
            multis = list(config.ncaa_mults)

            altloc_selectors = [*selectors, *_split_entries(raw.get("keep", "")), *_split_entries(raw.get("add_resid", ""))]
            pdb_result = read_pdb_result(
                pdb,
                pro_ff=raw.get("pro_ff", "ff14SB"),
                altloc_selectors=altloc_selectors,
            )
            structure = pdb_result.structure
            residues = [
                find_unique_residue(structure, selector, label="Target residue")
                for selector in selectors
            ]
            for index, residue in enumerate(residues):
                residue["net_charge"] = charges[index]

            issued_types: set[str] = set()
            for residue in residues:
                maple_types = allocate_maple_atom_types(len(residue["atoms"]), issued_types)
                issued_types.update(maple_types.values())
                for atom_offset, atom in enumerate(residue["atoms"], start=1):
                    atom["maple_type"] = maple_types[atom_offset]

            labels = [get_resid_label(residue) for residue in residues]
            info.append(f"Target: {', '.join(labels)} (protein)\n")
            info.extend(
                format_pdb_read_diagnostics(
                    pdb_result.diagnostics,
                    target_label=labels[0] if len(labels) == 1 else None,
                    target_charge=(labels[0], charges[0]) if len(labels) == 1 else None,
                )
            )
            if len(residues) > 1:
                info.append(
                    "  PDB target charges: "
                    + ", ".join(f"{label} = {charge}" for label, charge in zip(labels, charges))
                    + " (from your input)\n"
                )
            self.log_info(info)

            write_parmfit_run(
                self.output,
                "ncaa",
                raw,
                warn=lambda message: self.log_info([f"WARNING: {message}\n"]),
            )

            multi = len(residues) > 1
            if config.torsion_bonds_per_residue is not None and len(config.torsion_bonds_per_residue) != len(residues):
                raise ValueError(
                    f"torsion-bonds groups ({len(config.torsion_bonds_per_residue)}) must match "
                    f"residue count ({len(residues)})."
                )
            results = []
            built = []
            final_config = config
            for index, residue in enumerate(residues):
                tag = f"{residue['resname'].upper()}{residue['resseq']}"
                rn = rns[index] if index < len(rns) else tag
                mult = multis[index] if index < len(multis) else 1
                torsion = config.torsion
                if config.torsion_bonds_per_residue is not None:
                    torsion = replace(torsion, torsion_bonds=config.torsion_bonds_per_residue[index])
                residue_config = replace(
                    config,
                    target=selectors[index],
                    res_charge=charges[index],
                    res_spin_multi=mult,
                    rn=rn,
                    torsion=torsion,
                )
                work_dir = f"ncaa/{tag}"
                result = run_ncaa_abinitio(
                    output=self.output,
                    source_atoms=self.atoms,
                    structure=structure,
                    target_residue=residue,
                    config=residue_config,
                    log_info=self.log_info,
                    work_dir=work_dir,
                    tleap_validation=not multi,
                )
                results.append(result)
                built.append((labels[index], rn))
                final_config = residue_config

            final_result = results[-1]
            if multi:
                rename_map = {
                    get_resid_key(residue): built[index][1]
                    for index, residue in enumerate(residues)
                }
                write_ncaa_tleap_pdb(
                    final_result.artifacts.tleap_pdb,
                    structure=structure,
                    target_residue=residues[-1],
                    rename_map=rename_map,
                )
                write_ncaa_tleap_input(
                    final_result.artifacts.tleap_input,
                    amber=final_result.artifacts.amber,
                    atom_type_rows=final_result.atom_type_rows,
                    prepared_pdb_name=os.path.basename(final_result.artifacts.tleap_pdb),
                    base=os.path.splitext(os.path.basename(self.output))[0],
                    pro_ff=final_config.pro_ff,
                    wat_ff=final_config.wat_ff,
                    template_leaprcs=required_template_leaprcs(structure["residues"], final_config.pro_ff),
                    extra_templates=[result.artifacts.amber for result in results[:-1]],
                    extra_type_rows=[row for result in results[:-1] for row in result.atom_type_rows],
                )
                self.log_info(["  [NCAA] tleap validation ...\n"])
                interface.run_tleap(
                    final_result.artifacts.tleap_input,
                    workdir=os.path.dirname(final_result.artifacts.tleap_input) or ".",
                )
            self.log_info(
                format_ncaa_summary(
                    target_label=get_resid_label(residues[-1]),
                    config=final_config,
                    result=final_result,
                )
            )
            if multi:
                self.log_info(
                    ["  NCAA residues built: " + ", ".join(f"{label} -> {rn}" for label, rn in built) + "\n"]
                )
            return final_result
