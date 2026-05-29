"""Usage: build and write NCAA Amber templates, remapped parameters, and artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
import math
import os
import re
import shutil

from .. import interface as amber_interface
from ..ionparams import infer_ion_frcmod_name
from ..model import write_model_pdb
from ..outputparm import allocate_maple_atom_types, format_tleap_add_atom_types_lines
from ..readparm import Angle, Bond, Dihedral, Improper, Nonbond, CorrectionParameterSet, FrcmodDB, Mol2Atom, Mol2Topology
from ..runtime import MultiRespPipelineResult, parmfit_output_dir, parmfit_workdir
from ..structure import copy_residue, covalent_cutoff, get_atom_xyz, get_resid_key, residue_sort_key, search_atom
from .config import NCAAAbinitioConfig
from .models import NCAAConformer, build_residue_local_adjacency, infer_mainchain_names, infer_terminal_omit_names


# ---------------------------------------------------------------------------
# Records and Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NCAAAmberArtifacts:
    capped_mol2: str
    gaff2_mol2: str
    ac: str
    mc: str
    prepin: str
    refined_prepin: str
    res: str
    newpdb: str
    frcmod: str
    refined_frcmod: str


@dataclass(frozen=True)
class NCAAArtifacts:
    amber: NCAAAmberArtifacts
    target_capped_pdb: str
    tleap_pdb: str
    tleap_input: str
    conformer_capped_pdbs: dict[str, str]

    @property
    def files(self) -> dict[str, str]:
        return {
            "target_capped_pdb": self.target_capped_pdb,
            "tleap_pdb": self.tleap_pdb,
            "tleap_input": self.tleap_input,
            "capped_mol2": self.amber.capped_mol2,
            "gaff2_mol2": self.amber.gaff2_mol2,
            "ac": self.amber.ac,
            "mc": self.amber.mc,
            "prepin": self.amber.prepin,
            "refined_prepin": self.amber.refined_prepin,
            "res": self.amber.res,
            "newpdb": self.amber.newpdb,
            "frcmod": self.amber.frcmod,
            "refined_frcmod": self.amber.refined_frcmod,
            **{f"{label}_capped_pdb": path for label, path in self.conformer_capped_pdbs.items()},
        }


@dataclass(frozen=True)
class NCAAAmberBuildBundle:
    workdir: str
    final_dir: str
    interface_cfg: dict[str, object]
    mainchain_path: str
    typed_mol2_result: object
    antechamber_result: object
    prepgen_result: object
    parmchk_result: object


class AtomTypeRow(NamedTuple):
    atom_name: str
    element: str
    old_type: str
    maple_type: str
    resp_charge: float


@dataclass(frozen=True)
class NCAAExportBundle:
    amber: NCAAAmberArtifacts
    atom_type_rows: list[AtomTypeRow]
    artifacts: NCAAArtifacts


@dataclass(frozen=True)
class _MapleResidueMapping:
    prepin_lines: list[str]
    prepin_atom_names: list[str]
    atom_type_rows: list[AtomTypeRow]
    name_to_global_index: dict[str, int]
    global_to_local_index: dict[int, int]
    global_to_maple_type: dict[int, str]
    maple_mass_params: dict[str, float]


# ---------------------------------------------------------------------------
# AmberTools Build
# ---------------------------------------------------------------------------


def _run_ambertools_build(
    *,
    output: str,
    representative_model: dict,
    charged_residue: dict,
    resp_result: MultiRespPipelineResult,
    config: NCAAAbinitioConfig,
) -> NCAAAmberBuildBundle:
    workdir = parmfit_workdir(output, "ncaa")
    final_dir = parmfit_output_dir(output)
    interface_cfg = {
        "residue_name": config.rn,
        "net_charge": config.charge,
    }
    source_mol2 = os.path.basename(resp_result.files["mol2"])
    target_chg = os.path.basename(resp_result.resp_files["target_chg"])
    typed_mol2_result = amber_interface.run_antechamber(
        source_mol2,
        interface_cfg,
        workdir,
        input_format="mol2",
        output_format="mol2",
        charge_mode="rc",
        charge_file=target_chg,
    )
    antechamber_result = amber_interface.run_antechamber(
        source_mol2,
        interface_cfg,
        workdir,
        input_format="mol2",
        charge_mode="rc",
        charge_file=target_chg,
    )

    segment_sizes = representative_model["segment_sizes"]
    ac_names = amber_interface.read_ac_names(antechamber_result.ac_path)
    n_ace = segment_sizes["ace"]
    n_res = segment_sizes["residue"]
    ace_names = ac_names[:n_ace]
    nme_names = ac_names[n_ace + n_res :]

    mainchain_path = os.path.join(workdir, f"{config.rn}.mc")
    amber_interface.write_mainchain_mc(
        mainchain_path,
        ace_names,
        nme_names,
        head="N",
        tail="C",
        mainchain=infer_mainchain_names(charged_residue),
        charge=config.charge,
        extra_omit_names=infer_terminal_omit_names(charged_residue),
    )

    prepgen_result = amber_interface.run_prepgen(
        os.path.basename(antechamber_result.ac_path),
        os.path.basename(mainchain_path),
        interface_cfg,
        workdir,
    )
    parmchk_result = amber_interface.run_parmchk2(
        os.path.basename(prepgen_result.prepin_path),
        interface_cfg,
        False,
        workdir,
    )
    return NCAAAmberBuildBundle(
        workdir=workdir,
        final_dir=final_dir,
        interface_cfg=interface_cfg,
        mainchain_path=mainchain_path,
        typed_mol2_result=typed_mol2_result,
        antechamber_result=antechamber_result,
        prepgen_result=prepgen_result,
        parmchk_result=parmchk_result,
    )


def _materialize_amber_artifacts(
    *,
    build: NCAAAmberBuildBundle,
    resp_result: MultiRespPipelineResult,
    config: NCAAAbinitioConfig,
) -> NCAAAmberArtifacts:
    final_prepin = os.path.join(build.final_dir, f"{config.rn}.prepin")
    final_frcmod = os.path.join(build.final_dir, f"{config.rn}.frcmod")
    shutil.copyfile(build.prepgen_result.prepin_path, final_prepin)
    shutil.copyfile(build.parmchk_result.frcmod_path, final_frcmod)
    return NCAAAmberArtifacts(
        capped_mol2=resp_result.files["mol2"],
        gaff2_mol2=build.typed_mol2_result.ac_path,
        ac=build.antechamber_result.ac_path,
        mc=build.mainchain_path,
        prepin=final_prepin,
        refined_prepin=os.path.join(build.final_dir, f"{config.rn}_maple.prepin"),
        res=build.prepgen_result.res_path,
        newpdb=build.prepgen_result.newpdb_path,
        frcmod=final_frcmod,
        refined_frcmod=os.path.join(build.final_dir, f"{config.rn}_maple.frcmod"),
    )


def build_ncaa_amber_artifacts(
    *,
    output: str,
    representative_model: dict,
    charged_residue: dict,
    resp_result: MultiRespPipelineResult,
    config: NCAAAbinitioConfig,
) -> NCAAAmberArtifacts:
    build = _run_ambertools_build(
        output=output,
        representative_model=representative_model,
        charged_residue=charged_residue,
        resp_result=resp_result,
        config=config,
    )
    return _materialize_amber_artifacts(
        build=build,
        resp_result=resp_result,
        config=config,
    )


# ---------------------------------------------------------------------------
# Deployment Files
# ---------------------------------------------------------------------------


def export_ncaa_artifacts(
    output: str,
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    conformers: list[NCAAConformer],
    config: NCAAAbinitioConfig,
    atom_type_rows: list[AtomTypeRow],
    structure: dict,
    target_residue: dict,
) -> NCAAArtifacts:
    workdir = parmfit_workdir(output, "ncaa")
    base = os.path.splitext(os.path.basename(output))[0]
    artifacts = NCAAArtifacts(
        amber=amber,
        target_capped_pdb=os.path.join(workdir, f"{base}_capped_target.pdb"),
        tleap_pdb=os.path.join(parmfit_output_dir(output), f"{base}_ncaa_tleap.pdb"),
        tleap_input=os.path.join(parmfit_output_dir(output), f"{base}_ncaa_tleap.in"),
        conformer_capped_pdbs={
            conformer.label: os.path.join(workdir, f"{base}_{conformer.label}_capped_opt.pdb")
            for conformer in conformers
        },
    )
    for conformer in conformers:
        write_model_pdb(artifacts.conformer_capped_pdbs[conformer.label], conformer.model)
    write_model_pdb(artifacts.target_capped_pdb, representative_model)
    write_ncaa_tleap_pdb(
        artifacts.tleap_pdb,
        structure=structure,
        target_residue=target_residue,
        rn=config.rn,
    )
    write_ncaa_tleap_input(
        artifacts.tleap_input,
        amber=amber,
        atom_type_rows=atom_type_rows,
        prepared_pdb=os.path.basename(artifacts.tleap_pdb),
        base=base,
        prom=config.prom,
        watm=config.watm,
        ionm=config.ionm,
    )
    return artifacts


def write_ncaa_tleap_pdb(
    path: str,
    *,
    structure: dict,
    target_residue: dict,
    rn: str,
) -> str:
    target_key = get_resid_key(target_residue)
    residues = []
    for new_resseq, residue in enumerate(sorted(structure["residues"], key=residue_sort_key), start=1):
        copied = copy_residue(residue, resname=rn) if get_resid_key(residue) == target_key else copy_residue(residue)
        copied["resseq"] = new_resseq
        residues.append(copied)
    write_model_pdb(path, {"name": "ncaa_tleap_model", "residues": residues})
    return path


def write_ncaa_tleap_input(
    path: str,
    *,
    amber: NCAAAmberArtifacts,
    atom_type_rows: list["AtomTypeRow"],
    prepared_pdb: str,
    base: str,
    prom: str = "ff14SB",
    watm: str = "tip3p",
    ionm: str = "12_6",
) -> None:
    lines = [
        f"source leaprc.protein.{prom}\n",
        "source leaprc.gaff2\n",
        f"source leaprc.water.{watm}\n",
    ]
    lines.extend(
        format_tleap_add_atom_types_lines(
            [(row.atom_name, row.element, row.old_type, row.maple_type) for row in atom_type_rows]
        )
    )
    lines.extend(
        [
            f"loadamberprep {os.path.basename(amber.refined_prepin)}\n",
            f"loadamberparams {os.path.basename(amber.refined_frcmod)}\n",
            f"loadamberparams {infer_ion_frcmod_name(watm=watm, ionm=ionm, residue='Na')}\n",
            f"mol = loadpdb {prepared_pdb}\n",
            "check mol\n",
            "charge mol\n",
            f"solvatebox mol {'SPCBOX' if watm == 'spce' else watm.upper() + 'BOX'} 10.0\n",
            "addions mol Na+ 0\n",
            "addions mol Cl- 0\n",
            f"savepdb mol {base}_ncaa_solvated.pdb\n",
            f"saveamberparm mol {base}_ncaa.prmtop {base}_ncaa.inpcrd\n",
            "quit\n",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


# ---------------------------------------------------------------------------
# FRCMOD Merge and Remap
# ---------------------------------------------------------------------------


_PREPIN_ATOM_RE = re.compile(r"^(\s*\d+\s+\S+\s+)(\S+)(\s+.*)$")


def _replace_prepin_atom_type(raw: str, new_type: str) -> str:
    match = _PREPIN_ATOM_RE.match(raw.rstrip("\n"))
    if match is None:
        raise ValueError(f"Could not rewrite prepin atom type for line: {raw.rstrip()}")
    prefix, old_type, suffix = match.groups()
    return f"{prefix}{new_type:<{len(old_type)}}{suffix}\n"


def _insert_frcmod_section_lines(frcmod_path: str, extra_sections: dict[str, list[str]]) -> None:
    if not any(extra_sections.values()):
        return

    with open(frcmod_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    section_end: dict[str, int] = {}
    current: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}:
            current = stripped
        elif stripped == "" and current is not None:
            section_end[current] = index
            current = None

    output: list[str] = []
    for index, line in enumerate(lines):
        for section in ("BOND", "ANGLE", "DIHE"):
            if index == section_end.get(section):
                output.extend(extra_sections.get(section, ()))
        output.append(line)

    with open(frcmod_path, "w", encoding="utf-8") as handle:
        handle.writelines(output)


def _prepin_atom_rows(prepin_lines: list[str]) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    for line_index, raw in enumerate(prepin_lines):
        parts = raw.split()
        if len(parts) < 11 or parts[1] == "DUMM":
            continue
        try:
            int(parts[0])
            float(parts[10])
        except (ValueError, IndexError):
            continue
        rows.append((line_index, parts))
    return rows


def _build_maple_residue_mapping(
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    charged_residue: dict,
    parameter_set: CorrectionParameterSet,
) -> _MapleResidueMapping:
    with open(amber.prepin, "r", encoding="utf-8", errors="replace") as handle:
        prepin_lines = handle.readlines()
    prepin_atom_rows = _prepin_atom_rows(prepin_lines)
    prepin_atom_names = [parts[1] for _, parts in prepin_atom_rows]

    residue_atoms = sorted(charged_residue["atoms"], key=lambda atom: atom["serial"])
    residue_atom_names = [atom["name"] for atom in residue_atoms]
    if Counter(prepin_atom_names) != Counter(residue_atom_names):
        raise ValueError(
            "prepgen produced an invalid NCAA template: prepin atom names do not match the target residue atoms."
        )
    residue_atom_by_name = {atom["name"]: atom for atom in residue_atoms}
    residue_start = int(representative_model["segment_sizes"]["ace"]) + 1
    name_to_global_index = {
        atom["name"]: residue_start + offset
        for offset, atom in enumerate(residue_atoms)
    }

    existing_types = {atom.atom_type for atom in parameter_set.mol2.atoms}
    maple_types = allocate_maple_atom_types(len(prepin_atom_rows), existing_types)
    global_to_local_index: dict[int, int] = {}
    global_to_maple_type: dict[int, str] = {}
    maple_mass_params: dict[str, float] = {}
    atom_type_rows: list[AtomTypeRow] = []

    for local_index, (line_index, parts) in enumerate(prepin_atom_rows, start=1):
        name = parts[1]
        global_index = name_to_global_index[name]
        old_type = parameter_set.nonbonds[global_index - 1].atom_type
        resp_charge = float(parameter_set.nonbonds[global_index - 1].charge)
        maple_type = maple_types[local_index]
        prepin_lines[line_index] = _replace_prepin_atom_type(prepin_lines[line_index], maple_type)
        atom_type_rows.append(AtomTypeRow(name, residue_atom_by_name[name]["element"], old_type, maple_type, resp_charge))
        global_to_local_index[global_index] = local_index
        global_to_maple_type[global_index] = maple_type
        maple_mass_params[maple_type] = parameter_set.frcmod.mass_params[old_type]

    return _MapleResidueMapping(
        prepin_lines=prepin_lines,
        prepin_atom_names=prepin_atom_names,
        atom_type_rows=atom_type_rows,
        name_to_global_index=name_to_global_index,
        global_to_local_index=global_to_local_index,
        global_to_maple_type=global_to_maple_type,
        maple_mass_params=maple_mass_params,
    )


def _build_residue_parameter_set(
    parameter_set: CorrectionParameterSet,
    mapping: _MapleResidueMapping,
) -> CorrectionParameterSet:
    residue_indices = set(mapping.global_to_local_index)
    residue_mol2_atoms = [
        Mol2Atom(
            atom_id=local_index,
            name=name,
            atom_type=mapping.global_to_maple_type[mapping.name_to_global_index[name]],
            charge=parameter_set.nonbonds[mapping.name_to_global_index[name] - 1].charge,
        )
        for local_index, name in enumerate(mapping.prepin_atom_names, start=1)
    ]
    residue_nonbonds = [
        Nonbond(
            atom=mapping.global_to_local_index[global_index],
            atom_type=mapping.global_to_maple_type[global_index],
            charge=parameter_set.nonbonds[global_index - 1].charge,
            rmin_half=parameter_set.nonbonds[global_index - 1].rmin_half,
            epsilon=parameter_set.nonbonds[global_index - 1].epsilon,
        )
        for global_index in sorted(mapping.global_to_local_index, key=mapping.global_to_local_index.get)
    ]
    residue_bonds = [
        Bond(
            atoms=tuple(mapping.global_to_local_index[index] for index in bond.atoms),
            atom_types=tuple(mapping.global_to_maple_type[index] for index in bond.atoms),
            kBond=bond.kBond,
            rEq=bond.rEq,
        )
        for bond in parameter_set.bonds
        if set(bond.atoms).issubset(residue_indices)
    ]
    residue_angles = [
        Angle(
            atoms=tuple(mapping.global_to_local_index[index] for index in angle.atoms),
            atom_types=tuple(mapping.global_to_maple_type[index] for index in angle.atoms),
            kTheta=angle.kTheta,
            thetaEq=angle.thetaEq,
        )
        for angle in parameter_set.angles
        if set(angle.atoms).issubset(residue_indices)
    ]
    residue_dihedrals = [
        Dihedral(
            atoms=tuple(mapping.global_to_local_index[index] for index in dihedral.atoms),
            atom_types=tuple(mapping.global_to_maple_type[index] for index in dihedral.atoms),
            terms=list(dihedral.terms),
        )
        for dihedral in parameter_set.dihedrals
        if set(dihedral.atoms).issubset(residue_indices)
    ]
    residue_impropers = [
        Improper(
            atoms=tuple(mapping.global_to_local_index[index] for index in improper.atoms),
            atom_types=tuple(mapping.global_to_maple_type[index] for index in improper.atoms),
            terms=list(improper.terms),
        )
        for improper in parameter_set.impropers
        if set(improper.atoms).issubset(residue_indices)
    ]
    return CorrectionParameterSet(
        mol2=Mol2Topology(
            atoms=residue_mol2_atoms,
            bonds=[],
            id_to_index={atom.atom_id: atom.atom_id for atom in residue_mol2_atoms},
            adjacency={},
        ),
        frcmod=FrcmodDB(mass_params=dict(mapping.maple_mass_params)),
        bonds=residue_bonds,
        angles=residue_angles,
        dihedrals=residue_dihedrals,
        impropers=residue_impropers,
        nonbonds=residue_nonbonds,
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def write_ncaa_amber_files(
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    charged_residue: dict,
    final_parameter_set: CorrectionParameterSet,
    structure: dict,
    target_residue: dict,
    prom: str = "ff14SB",
) -> list[AtomTypeRow]:
    mapping = _build_maple_residue_mapping(
        amber=amber,
        representative_model=representative_model,
        charged_residue=charged_residue,
        parameter_set=final_parameter_set,
    )
    with open(amber.refined_prepin, "w", encoding="utf-8") as handle:
        handle.writelines(mapping.prepin_lines)

    residue_parameter_set = _build_residue_parameter_set(final_parameter_set, mapping)
    amber_interface.write_refined_frcmod(
        residue_parameter_set,
        amber.refined_frcmod,
        mass_params=mapping.maple_mass_params,
        remark="REMARK MAPLE ncaa refined frcmod",
    )
    extra_sections = _generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=mapping.name_to_global_index,
        global_to_maple_type=mapping.global_to_maple_type,
        structure=structure,
        target_residue=target_residue,
        prom=prom,
    )
    _insert_frcmod_section_lines(
        amber.refined_frcmod,
        extra_sections,
    )
    return mapping.atom_type_rows


# ---------------------------------------------------------------------------
# Peptide Boundary Terms
# ---------------------------------------------------------------------------


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        ordered.append(line)
    return ordered


def _fmt_bond(left: str, right: str, k: float, req: float, note: str) -> str:
    return f"{left:<2s}-{right:<2s}     {k:7.3f}    {req:6.4f}      {note}\n"


def _fmt_angle(left: str, center: str, right: str, k: float, theta: float, note: str) -> str:
    return f"{left:<2s}-{center:<2s}-{right:<2s}     {k:7.3f}    {theta:7.3f}      {note}\n"


def _fmt_dihedral(
    left: str,
    center1: str,
    center2: str,
    right: str,
    terms: list[tuple[int, float, float, float]],
    note: str,
) -> list[str]:
    return [
        f"{left:<2s}-{center1:<2s}-{center2:<2s}-{right:<2s}     {idivf:1d}     {kphi:7.4f}    {phase:8.3f}    {period:6.3f}      {note}\n"
        for idivf, kphi, phase, period in terms
    ]


def _neighbor_residue(structure: dict, target_residue: dict, offset: int) -> dict | None:
    target_key = get_resid_key(target_residue)
    residues = structure.get("residues", [])
    target_index = next(
        (index for index, residue in enumerate(residues) if get_resid_key(residue) == target_key),
        None,
    )
    if target_index is None:
        return None
    neighbor_index = target_index + offset
    if neighbor_index < 0 or neighbor_index >= len(residues):
        return None
    neighbor = residues[neighbor_index]
    if neighbor.get("chain") != target_residue.get("chain"):
        return None
    return neighbor


def _atoms_are_bonded(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return False
    distance = math.dist(get_atom_xyz(left), get_atom_xyz(right))
    return distance <= covalent_cutoff(left, right)


def _previous_peptide_residue(structure: dict, target_residue: dict) -> dict | None:
    previous = _neighbor_residue(structure, target_residue, -1)
    if previous is None:
        return None
    if _atoms_are_bonded(search_atom(previous, "C"), search_atom(target_residue, "N")):
        return previous
    return None


def _next_peptide_residue(structure: dict, target_residue: dict) -> dict | None:
    next_residue = _neighbor_residue(structure, target_residue, 1)
    if next_residue is None:
        return None
    if _atoms_are_bonded(search_atom(target_residue, "C"), search_atom(next_residue, "N")):
        return next_residue
    return None


def _prom_boundary_type(structure: dict, residue: dict, atom_name: str, *, prom: str = "ff14SB") -> str:
    atom = search_atom(residue, atom_name)
    if atom is None:
        raise ValueError(f"Could not find atom {atom_name!r} in NCAA boundary residue.")
    is_n_terminal = _previous_peptide_residue(structure, residue) is None
    is_c_terminal = _next_peptide_residue(structure, residue) is None
    if atom_name == "N":
        return "N3" if is_n_terminal else "N"
    if atom_name == "CA":
        if is_n_terminal or is_c_terminal:
            return "CX"
        return "XC" if prom == "ff19SB" else "CX"
    if atom_name == "C":
        return "C"
    if atom_name in {"O", "OXT"}:
        return "O"
    if residue.get("resname") == "PRO" and atom_name in {"CD", "CG", "CB"}:
        return "CT"
    if atom["element"] == "H":
        return "H" if atom_name in {"H", "HN"} else "H1"
    if atom["element"] == "C":
        return "CT"
    return atom["element"]


def _target_ca_neighbor_names(
    *,
    residue_atoms: list[dict],
    adjacency: dict[int, set[int]],
    ca_local: int,
    excluded: set[str],
) -> list[str]:
    names: list[str] = []
    for neighbor in sorted(adjacency[ca_local]):
        atom = residue_atoms[neighbor - 1]
        if atom["name"] in excluded or atom["name"] == "O":
            continue
        names.append(atom["name"])
    return names


def _next_n_substituent_types(structure: dict, next_residue: dict, *, prom: str) -> list[str]:
    if next_residue.get("resname") == "PRO":
        atom_names = ["CA", "CD"]
    else:
        atom_names = ["H", "HN", "CA"]
    types: list[str] = []
    for atom_name in atom_names:
        if search_atom(next_residue, atom_name) is not None:
            types.append(_prom_boundary_type(structure, next_residue, atom_name, prom=prom))
    return _dedupe_lines(types)


# AutoNACC-style exact peptide-boundary terms: fill matching gaps, not fitted torsions.
_C_N_XC_X_TERMS = [(6, 0.0, 0.0, 2.0)]
_N_XC_C_N_TERMS = [
    (1, 0.00, 0.0, -4.0),
    (1, 0.55, 180.0, -3.0),
    (1, 1.58, 180.0, -2.0),
    (1, 0.45, 180.0, 1.0),
]
_CT_XC_C_N_TERMS = [
    (1, 0.00, 0.0, -4.0),
    (1, 0.40, 0.0, -3.0),
    (1, 0.20, 0.0, -2.0),
    (1, 0.20, 0.0, 1.0),
]
_AMIDE_X_C_N_X_TERMS = [(4, 10.00, 180.0, 2.0)]
_AMIDE_O_C_N_H_TERMS = [
    (1, 2.50, 180.0, -2.0),
    (1, 2.00, 0.0, 1.0),
]


def _psi_terms_for_left_type(left_type: str) -> list[tuple[int, float, float, float]]:
    if left_type == "N":
        return _N_XC_C_N_TERMS
    return _CT_XC_C_N_TERMS


def _generate_maple_crossterms(
    *,
    residue: dict,
    name_to_global_index: dict[str, int],
    global_to_maple_type: dict[int, str],
    structure: dict,
    target_residue: dict,
    prom: str = "ff14SB",
) -> dict[str, list[str]]:
    residue_atoms, local_index_by_name, adjacency = build_residue_local_adjacency(residue)

    n_local = local_index_by_name["N"]
    ca_local = local_index_by_name["CA"]

    def maple_type(name: str) -> str:
        return global_to_maple_type[name_to_global_index[name]]

    n_maple = maple_type("N")
    ca_maple = maple_type("CA")
    c_maple = maple_type("C")
    o_maple = maple_type("O")

    n_hydrogens = sorted(
        residue_atoms[neighbor - 1]["name"]
        for neighbor in adjacency[n_local]
        if residue_atoms[neighbor - 1]["element"] == "H"
    )
    ca_neighbors_from_head = _target_ca_neighbor_names(
        residue_atoms=residue_atoms,
        adjacency=adjacency,
        ca_local=ca_local,
        excluded={"N"},
    )
    ca_neighbors_from_tail = _target_ca_neighbor_names(
        residue_atoms=residue_atoms,
        adjacency=adjacency,
        ca_local=ca_local,
        excluded={"C"},
    )

    previous_residue = _previous_peptide_residue(structure, target_residue)
    next_residue = _next_peptide_residue(structure, target_residue)

    bond_lines: list[str] = []
    angle_lines: list[str] = []
    dihe_lines: list[str] = []

    # Head-side peptide boundary.
    if previous_residue is not None:
        note = f"{prom}/gaff2 peptide boundary"
        prev_ca_type = _prom_boundary_type(structure, previous_residue, "CA", prom=prom)
        prev_c_type = _prom_boundary_type(structure, previous_residue, "C", prom=prom)
        prev_o_type = _prom_boundary_type(structure, previous_residue, "O", prom=prom)
        bond_lines.append(_fmt_bond(prev_c_type, n_maple, 490.0, 1.3350, note))
        angle_lines.extend(
            [
                _fmt_angle(prev_o_type, prev_c_type, n_maple, 80.0, 122.9, note),
                _fmt_angle(prev_ca_type, prev_c_type, n_maple, 70.0, 116.6, note),
                _fmt_angle(prev_c_type, n_maple, ca_maple, 50.0, 121.9, note),
            ]
        )
        dihe_lines.extend(_fmt_dihedral(prev_o_type, prev_c_type, n_maple, ca_maple, _AMIDE_X_C_N_X_TERMS, note))
        dihe_lines.extend(_fmt_dihedral(prev_ca_type, prev_c_type, n_maple, ca_maple, _AMIDE_X_C_N_X_TERMS, note))
        for hydrogen_name in n_hydrogens:
            hydrogen_maple = maple_type(hydrogen_name)
            angle_lines.append(_fmt_angle(prev_c_type, n_maple, hydrogen_maple, 50.0, 120.0, note))
            dihe_lines.extend(_fmt_dihedral(prev_o_type, prev_c_type, n_maple, hydrogen_maple, _AMIDE_O_C_N_H_TERMS, note))
            dihe_lines.extend(_fmt_dihedral(prev_ca_type, prev_c_type, n_maple, hydrogen_maple, _AMIDE_X_C_N_X_TERMS, note))
        for neighbor_name in ca_neighbors_from_head:
            neighbor_maple = maple_type(neighbor_name)
            dihe_lines.extend(_fmt_dihedral(prev_c_type, n_maple, ca_maple, neighbor_maple, _C_N_XC_X_TERMS, note))

    # Tail-side peptide boundary.
    if next_residue is not None:
        note = f"{prom}/gaff2 peptide boundary"
        next_n_type = _prom_boundary_type(structure, next_residue, "N", prom=prom)
        next_substituent_types = _next_n_substituent_types(structure, next_residue, prom=prom)
        bond_lines.append(_fmt_bond(c_maple, next_n_type, 490.0, 1.3350, note))
        angle_lines.extend(
            [
                _fmt_angle(o_maple, c_maple, next_n_type, 80.0, 122.9, note),
                _fmt_angle(ca_maple, c_maple, next_n_type, 70.0, 116.6, note),
            ]
        )
        for next_type in next_substituent_types:
            angle_lines.append(_fmt_angle(c_maple, next_n_type, next_type, 50.0, 120.0 if next_type == "H" else 121.9, note))
            terms = _AMIDE_O_C_N_H_TERMS if next_type == "H" else _AMIDE_X_C_N_X_TERMS
            dihe_lines.extend(_fmt_dihedral(o_maple, c_maple, next_n_type, next_type, terms, note))
            dihe_lines.extend(_fmt_dihedral(ca_maple, c_maple, next_n_type, next_type, _AMIDE_X_C_N_X_TERMS, note))
        for neighbor_name in ca_neighbors_from_tail:
            neighbor_maple = maple_type(neighbor_name)
            terms = _psi_terms_for_left_type("N" if neighbor_name == "N" else "CT")
            dihe_lines.extend(_fmt_dihedral(neighbor_maple, ca_maple, c_maple, next_n_type, terms, note))

    return {
        "BOND": _dedupe_lines(bond_lines),
        "ANGLE": _dedupe_lines(angle_lines),
        "DIHE": _dedupe_lines(dihe_lines),
    }


# ---------------------------------------------------------------------------
# Public Writers
# ---------------------------------------------------------------------------


def build_ncaa_export_bundle(
    output: str,
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    charged_residue: dict,
    conformers: list[NCAAConformer],
    final_parameter_set: CorrectionParameterSet,
    config: NCAAAbinitioConfig,
    structure: dict,
    target_residue: dict,
) -> NCAAExportBundle:
    atom_type_rows = write_ncaa_amber_files(
        amber=amber,
        representative_model=representative_model,
        charged_residue=charged_residue,
        final_parameter_set=final_parameter_set,
        structure=structure,
        target_residue=target_residue,
        prom=config.prom,
    )
    artifacts = export_ncaa_artifacts(
        output,
        amber=amber,
        representative_model=representative_model,
        conformers=conformers,
        config=config,
        atom_type_rows=atom_type_rows,
        structure=structure,
        target_residue=target_residue,
    )
    return NCAAExportBundle(
        amber=amber,
        atom_type_rows=atom_type_rows,
        artifacts=artifacts,
    )
