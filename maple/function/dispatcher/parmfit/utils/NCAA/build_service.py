"""Usage: build one declared non-standard residue's parameters (pluggable NCAA building service).

Runs the native NCAA parameterization chain for a single residue: model construction
(capped for backbone residues, bare otherwise), frozen-skeleton optimization, RESP
charge fitting against the declared net charge, antechamber typing, and an
intermediate-layer parmchk2 frcmod. The refined (MAPLE) parameter layer is never
consumed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .. import interface
from ..chgfit import ChargeFitConfig, apply_model_charges, fit_multiconformer_charges
from ..context import find_prev_next_peptide_residues
from ..model import infer_bond_pairs
from ..readparm import Mol2Topology, parse_mol2
from ..runtime import parmfit_workdir
from .models import build_capped_ncaa_model, build_ncaa_sidechain_relax_indices, optimize_capped_confs

_BACKBONE_ATOM_NAMES = {"N", "CA", "C"}


@dataclass(frozen=True)
class ResidueBuildResult:
    residue_key: tuple[str, int, str]
    rn: str
    atom_types: dict[str, str]
    atom_charges: dict[str, float]
    frcmod_path: str
    typed_mol2_path: str


def _residue_has_backbone(residue: dict) -> bool:
    names = {atom["name"] for atom in residue["atoms"]}
    return _BACKBONE_ATOM_NAMES <= names


def _slice_typed_residue(typed_mol2: Mol2Topology, atom_names: set[str]) -> Mol2Topology:
    residue_atoms = [atom for atom in typed_mol2.atoms if atom.name in atom_names]
    kept_ids = {atom.atom_id for atom in residue_atoms}
    id_to_index = {atom.atom_id: index for index, atom in enumerate(residue_atoms)}
    residue_bonds = [
        bond for bond in typed_mol2.bonds if bond.atom1 in kept_ids and bond.atom2 in kept_ids
    ]
    adjacency: dict[int, set[int]] = {}
    for bond in residue_bonds:
        left = id_to_index.get(bond.atom1)
        right = id_to_index.get(bond.atom2)
        if left is None or right is None:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return Mol2Topology(atoms=residue_atoms, bonds=residue_bonds, id_to_index=id_to_index, adjacency=adjacency)


def build_residue_parameters(
    structure: dict,
    residue: dict,
    *,
    output: str,
    rn: str,
    net_charge: int,
    multiplicity: int,
    charge_fit: ChargeFitConfig,
    source_atoms,
    pro_ff: str = "ff14SB",
    tag: str,
    opt_max_iter: int = 256,
    opt_max_step: float = 0.2,
) -> ResidueBuildResult:
    residue_key = (residue["chain"], residue["resseq"], residue.get("icode", ""))
    has_backbone = _residue_has_backbone(residue)
    workdir = parmfit_workdir(output, f"ncaa/{tag}")

    if has_backbone:
        prev_residue, next_residue = find_prev_next_peptide_residues(structure, residue)
        model = build_capped_ncaa_model(residue, rn, prev_residue=prev_residue, next_residue=next_residue)
        frozen_indices = build_ncaa_sidechain_relax_indices(model)
    else:
        from ..structure import copy_residue

        model = {
            "name": "ncaa_bare_model",
            "target_key": residue_key,
            "residues": [copy_residue(residue, resname=rn)],
        }
        frozen_indices = None
    model["charge"] = int(net_charge)
    model["mult"] = int(multiplicity)

    conformer = optimize_capped_confs(
        model,
        source_atoms=source_atoms,
        output=os.path.join(workdir, f"{tag}.build"),
        max_iter=opt_max_iter,
        max_step=opt_max_step,
        frozen_indices=frozen_indices,
    )
    model = conformer.model

    charge_result = fit_multiconformer_charges(
        output=output,
        conformers=[("ref", model)],
        representative_model=model,
        residue_key=model["target_key"],
        bond_pairs=infer_bond_pairs(model),
        total_charge=int(net_charge),
        multiplicity=int(multiplicity),
        config=charge_fit,
        source_atoms=source_atoms,
        pro_ff=pro_ff,
        workflow=f"ncaa/{tag}",
    )

    interface.run_antechamber(
        os.path.basename(charge_result.work_mol2),
        {"residue_name": rn, "net_charge": int(net_charge), "multiplicity": int(multiplicity)},
        workdir,
        input_format="mol2",
        output_format="mol2",
        charge_mode="rc",
        charge_file=os.path.basename(charge_result.files["target_chg"]),
    )
    typed_mol2_path = os.path.join(workdir, f"{rn}.mol2")
    typed_mol2 = parse_mol2(typed_mol2_path)

    charged_model = apply_model_charges(model, charge_result.charges)
    atom_charges: dict[str, float] = {}
    atom_names: set[str] = set()
    for built_residue in charged_model["residues"]:
        if built_residue.get("resname", "").upper() != rn.upper():
            continue
        for atom in built_residue["atoms"]:
            atom_charges[atom["name"]] = float(atom["charge"])
            atom_names.add(atom["name"])

    residue_typed = _slice_typed_residue(typed_mol2, atom_names)
    atom_types = {atom.name: atom.atom_type for atom in residue_typed.atoms}

    parmchk_result = interface.run_parmchk2(
        os.path.basename(typed_mol2_path),
        {"residue_name": rn},
        True,
        workdir,
    )
    frcmod_path = parmchk_result.frcmod_path
    if has_backbone:
        interface.patch_frcmod_crossterms(frcmod_path)

    return ResidueBuildResult(
        residue_key=residue_key,
        rn=rn,
        atom_types=atom_types,
        atom_charges=atom_charges,
        frcmod_path=frcmod_path,
        typed_mol2_path=typed_mol2_path,
    )
