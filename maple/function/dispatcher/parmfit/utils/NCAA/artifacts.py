"""Usage: build and write NCAA Amber templates, remapped parameters, and artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
import os
import re
import shutil

from .. import interface as amber_interface
from ..model import write_model_pdb
from ..outputparm import allocate_maple_atom_types, format_tleap_add_atom_types_lines
from ..readparm import Angle, Bond, Dihedral, Improper, Nonbond, CorrectionParameterSet, FrcmodDB, Mol2Atom, Mol2Topology
from ..runtime import MultiRespPipelineResult, parmfit_output_dir, parmfit_workdir
from .config import NCAAAbinitioConfig
from .models import NCAAConformer, build_residue_local_adjacency, infer_mainchain_names, infer_terminal_omit_names


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
    tleap_input: str
    conformer_capped_pdbs: dict[str, str]

    @property
    def files(self) -> dict[str, str]:
        return {
            "target_capped_pdb": self.target_capped_pdb,
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


def export_ncaa_artifacts(
    output: str,
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    conformers: list[NCAAConformer],
    config: NCAAAbinitioConfig,
    atom_type_rows: list[AtomTypeRow],
) -> NCAAArtifacts:
    workdir = parmfit_workdir(output, "ncaa")
    base = os.path.splitext(os.path.basename(output))[0]
    artifacts = NCAAArtifacts(
        amber=amber,
        target_capped_pdb=os.path.join(workdir, f"{base}_capped_target.pdb"),
        tleap_input=os.path.join(parmfit_output_dir(output), f"{base}_ncaa_tleap.in"),
        conformer_capped_pdbs={
            conformer.label: os.path.join(workdir, f"{base}_{conformer.label}_capped_opt.pdb")
            for conformer in conformers
        },
    )
    for conformer in conformers:
        write_model_pdb(artifacts.conformer_capped_pdbs[conformer.label], conformer.model)
    write_model_pdb(artifacts.target_capped_pdb, representative_model)
    write_ncaa_tleap_input(
        artifacts.tleap_input,
        amber=amber,
        atom_type_rows=atom_type_rows,
        prepared_pdb=os.path.basename(config.pdb_path),
        base=base,
    )
    return artifacts


def write_ncaa_tleap_input(
    path: str,
    *,
    amber: NCAAAmberArtifacts,
    atom_type_rows: list["AtomTypeRow"],
    prepared_pdb: str,
    base: str,
) -> None:
    lines = [
        "source leaprc.protein.ff19SB\n",
        "source leaprc.gaff2\n",
        "source leaprc.water.opc\n",
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
            "loadamberparams frcmod.ionslm_hfe_opc\n",
            f"mol = loadpdb {prepared_pdb}\n",
            "check mol\n",
            "charge mol\n",
            "solvatebox mol OPCBOX 10.0\n",
            "addions mol Na+ 0\n",
            "addions mol Cl- 0\n",
            f"savepdb mol {base}_ncaa_solvated.pdb\n",
            f"saveamberparm mol {base}_ncaa.prmtop {base}_ncaa.inpcrd\n",
            "quit\n",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


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


def _generate_maple_crossterms(
    *,
    residue: dict,
    name_to_global_index: dict[str, int],
    global_to_maple_type: dict[int, str],
) -> dict[str, list[str]]:
    residue_atoms, local_index_by_name, adjacency = build_residue_local_adjacency(residue)

    n_local = local_index_by_name["N"]
    ca_local = local_index_by_name["CA"]
    c_local = local_index_by_name["C"]
    o_local = local_index_by_name["O"]

    def maple_type(name: str) -> str:
        return global_to_maple_type[name_to_global_index[name]]

    n_maple = maple_type("N")
    ca_maple = maple_type("CA")
    c_maple = maple_type("C")
    o_maple = maple_type("O")

    n_hydrogens = [
        residue_atoms[neighbor - 1]["name"]
        for neighbor in adjacency[n_local]
        if residue_atoms[neighbor - 1]["element"] == "H"
    ]
    ca_heavy_neighbors = [
        residue_atoms[neighbor - 1]["name"]
        for neighbor in adjacency[ca_local]
        if residue_atoms[neighbor - 1]["element"] != "H" and residue_atoms[neighbor - 1]["name"] not in {"N", "C", "O"}
    ]
    ca_hydrogens = [
        residue_atoms[neighbor - 1]["name"]
        for neighbor in adjacency[ca_local]
        if residue_atoms[neighbor - 1]["element"] == "H"
    ]

    bond_lines = [
        f"C -{n_maple}     490.000    1.3350      ff14SB/gaff2 peptide bond\n",
        f"{c_maple}-N      490.000    1.3350      ff14SB/gaff2 peptide bond\n",
    ]

    angle_lines = [
        f"O -C -{n_maple:<2s}      80.000    122.900      ff14SB(O,C)/gaff2(ns)\n",
        f"CX-C -{n_maple:<2s}      70.000    116.600      ff14SB(CX,C)/gaff2(ns)\n",
        f"C -{n_maple:<2s}-{ca_maple:<2s}      50.000    121.900      ff14SB(C)/gaff2(ns,c3)\n",
        f"{o_maple:<2s}-{c_maple:<2s}-N       80.000    122.900      gaff2(o,c)/ff14SB(N)\n",
        f"{c_maple:<2s}-N -CX      50.000    121.900      gaff2(c)/ff14SB(N,CX)\n",
        f"{ca_maple:<2s}-{c_maple:<2s}-N       70.000    116.600      gaff2(c3,c)/ff14SB(N)\n",
    ]
    for hydrogen_name in n_hydrogens:
        angle_lines.append(
            f"C -{n_maple:<2s}-{global_to_maple_type[name_to_global_index[hydrogen_name]]:<2s}      50.000    120.000      ff14SB(C)/gaff2(ns,hn)\n"
        )
    angle_lines.append(f"{c_maple:<2s}-N -H       80.000    122.900      gaff2(c)/ff14SB(N,H)\n")

    dihe_lines = [
        f"O -C -{n_maple:<2s}-{ca_maple:<2s}     4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"CX-C -{n_maple:<2s}-{ca_maple:<2s}     4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"{o_maple:<2s}-{c_maple:<2s}-N -H      1      2.500     180.000    -2.000      ff14SB/gaff2\n",
        f"{o_maple:<2s}-{c_maple:<2s}-N -H      1      2.000       0.000     1.000      ff14SB/gaff2\n",
        f"{o_maple:<2s}-{c_maple:<2s}-N -CX     4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"{ca_maple:<2s}-{c_maple:<2s}-N -H      4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"{ca_maple:<2s}-{c_maple:<2s}-N -CX     4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"{c_maple:<2s}-N -CX-C      4     10.000     180.000     2.000      ff14SB/gaff2\n",
        f"{c_maple:<2s}-N -CX-H1     4     10.000     180.000     2.000      ff14SB/gaff2\n",
    ]
    for hydrogen_name in n_hydrogens:
        hydrogen_maple = global_to_maple_type[name_to_global_index[hydrogen_name]]
        dihe_lines.extend(
            [
                f"O -C -{n_maple:<2s}-{hydrogen_maple:<2s}     1      2.500     180.000    -2.000      ff14SB/gaff2\n",
                f"O -C -{n_maple:<2s}-{hydrogen_maple:<2s}     1      2.000       0.000     1.000      ff14SB/gaff2\n",
                f"CX-C -{n_maple:<2s}-{hydrogen_maple:<2s}     4     10.000     180.000     2.000      ff14SB/gaff2\n",
            ]
        )
    for neighbor_name in ca_heavy_neighbors:
        neighbor_maple = global_to_maple_type[name_to_global_index[neighbor_name]]
        dihe_lines.append(
            f"C -{n_maple:<2s}-{ca_maple:<2s}-{neighbor_maple:<2s}     6      0.000       0.000     2.000      ff14SB/gaff2\n"
        )
    for hydrogen_name in ca_hydrogens:
        hydrogen_maple = global_to_maple_type[name_to_global_index[hydrogen_name]]
        dihe_lines.append(
            f"C -{n_maple:<2s}-{ca_maple:<2s}-{hydrogen_maple:<2s}     6      0.000       0.000     2.000      ff14SB/gaff2\n"
        )

    def _dedupe(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            ordered.append(line)
        return ordered

    return {
        "BOND": _dedupe(bond_lines),
        "ANGLE": _dedupe(angle_lines),
        "DIHE": _dedupe(dihe_lines),
    }


def _build_maple_residue_mapping(
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    charged_residue: dict,
    parameter_set: CorrectionParameterSet,
) -> _MapleResidueMapping:
    with open(amber.prepin, "r", encoding="utf-8", errors="replace") as handle:
        prepin_lines = handle.readlines()
    prepin_atom_rows: list[tuple[int, list[str]]] = []
    for line_index, raw in enumerate(prepin_lines):
        parts = raw.split()
        if len(parts) < 11 or parts[1] == "DUMM":
            continue
        try:
            int(parts[0])
            float(parts[10])
        except (ValueError, IndexError):
            continue
        prepin_atom_rows.append((line_index, parts))
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
    )
    _insert_frcmod_section_lines(
        amber.refined_frcmod,
        extra_sections,
    )
    return mapping.atom_type_rows


def build_ncaa_export_bundle(
    output: str,
    *,
    amber: NCAAAmberArtifacts,
    representative_model: dict,
    charged_residue: dict,
    conformers: list[NCAAConformer],
    final_parameter_set: CorrectionParameterSet,
    config: NCAAAbinitioConfig,
) -> NCAAExportBundle:
    atom_type_rows = write_ncaa_amber_files(
        amber=amber,
        representative_model=representative_model,
        charged_residue=charged_residue,
        final_parameter_set=final_parameter_set,
    )
    artifacts = export_ncaa_artifacts(
        output,
        amber=amber,
        representative_model=representative_model,
        conformers=conformers,
        config=config,
        atom_type_rows=atom_type_rows,
    )
    return NCAAExportBundle(
        amber=amber,
        atom_type_rows=atom_type_rows,
        artifacts=artifacts,
    )
