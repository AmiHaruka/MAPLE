"""Usage: run the MetalAA abinitio parameterization workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from .. import interface
from ..Seminario import apply_seminario
from ..mSeminario import apply_mseminario
from ..model import build_bond_angle_terms, flatten_model_atoms, infer_bond_pairs, model_to_atoms, update_model_from_atoms
from ..QMInterface import build_qm_reference_runner
from ..runtime import copy_thresholds, get_cartesian_hessian, optimize_model_geometry, parmfit_work_prefix, run_resp_pipeline
from ..structure import (
    CHARGED_STANDARD_RESIDUES,
    METAL_SITE_DONOR_ELEMENTS,
    get_atom_xyz,
    get_resid_key,
    get_resid_label,
    residue_sort_key,
)
from .charges import infer_model_charge, project_resp_charges_onto_site_model
from .config import MetalAbinitioConfig
from .recognize import MetalSiteSelection, apply_cfmol2_templates, build_cofactor_orig_frcmods, find_metal_site_core
from .artifacts import MetalArtifacts, MetalSiteTyping, plan_metal_artifacts, write_large_pdb, write_site_frcmod, write_site_model_files
from .models import MetalModelBundle, build_metal_model_bundle, build_metal_site_model
from .report import format_metal_final_lines, format_metal_start_lines


@contextmanager
def _timed_stage(stage_timings: list[tuple[str, float]], label: str):
    start = perf_counter()
    try:
        yield
    finally:
        stage_timings.append((label, perf_counter() - start))


@dataclass(frozen=True)
class MetalWorkflowResult:
    selection: MetalSiteSelection
    large_model: dict
    site_model: dict
    artifacts: MetalArtifacts
    site_typing: MetalSiteTyping
    bonded_warning: Optional[str] = None
    stage_timings: list[tuple[str, float]] = field(default_factory=list)

    @property
    def core_info(self) -> dict:
        return self.selection.to_dict()

    @property
    def files(self) -> dict[str, str]:
        return self.artifacts.files

    @property
    def resp_files(self) -> dict[str, str]:
        return self.artifacts.resp_files


@dataclass(frozen=True)
class _OptimizedCore:
    selection: MetalSiteSelection
    core_residues: list[dict]
    donor_atoms: dict[tuple[str, int, str], list[str]]
    target_key: tuple[str, int, str]
    target_residue: dict
    metal_atom: dict


@dataclass(frozen=True)
class _RespProblem:
    bond_pairs: list[tuple[int, int]]
    charge_groups: list[tuple[list[int], float]]


@dataclass(frozen=True)
class _SiteExport:
    site_pdb_path: str
    site_mol2_path: str
    site_typing: MetalSiteTyping


@dataclass(frozen=True)
class _BondedResult:
    frcmod_path: str
    bonded_warning: Optional[str]


def _annotate_target_ion_formal_charge(model: dict, target_key: tuple[str, int, str] | None, formal_charge: int) -> None:
    for residue in model["residues"]:
        if residue.get("kind") == "ion" and get_resid_key(residue) == target_key:
            residue["formal_charge"] = int(formal_charge)


def _atom_entries_by_residue(model: dict) -> dict[tuple[str, int, str], list[tuple[int, dict]]]:
    grouped: dict[tuple[str, int, str], list[tuple[int, dict]]] = {}
    for atom_index, (residue, atom) in enumerate(flatten_model_atoms(model), start=1):
        grouped.setdefault(get_resid_key(residue), []).append((atom_index, atom))
    return grouped


def _single_atom_residue_atom(residue: dict, *, label: str) -> dict:
    if len(residue.get("atoms", [])) != 1:
        raise ValueError(f"{label} must be a single-atom residue.")
    return residue["atoms"][0]


def _bond_pairs_from_names(
    name_pairs,
    entries: list[tuple[int, dict]],
    *,
    label: str,
) -> list[tuple[int, int]]:
    index_by_name = {atom["name"]: atom_index for atom_index, atom in entries}
    pairs: set[tuple[int, int]] = set()
    for left_name, right_name in name_pairs:
        try:
            left_index = index_by_name[left_name]
            right_index = index_by_name[right_name]
        except KeyError as exc:
            raise ValueError(f"{label} cfmol2 bond references unknown atom {exc.args[0]!r}.") from exc
        pairs.add(tuple(sorted((left_index, right_index))))
    return sorted(pairs)


def _remap_terms_to_site_model(large_model: dict, site_model: dict, bond_terms, angle_terms):
    large_to_site_index: dict[int, int] = {}
    site_atoms_by_residue = _atom_entries_by_residue(site_model)
    for residue_key, large_entries in _atom_entries_by_residue(large_model).items():
        site_entries = site_atoms_by_residue.get(residue_key)
        if site_entries is None:
            continue
        if len(large_entries) != len(site_entries):
            raise ValueError(
                f"MetalAA term remap atom count mismatch for residue {residue_key}: "
                f"large_model has {len(large_entries)} atoms, site_model has {len(site_entries)}."
            )
        large_names = [atom["name"] for _atom_index, atom in large_entries]
        site_names = [atom["name"] for _atom_index, atom in site_entries]
        if large_names != site_names:
            raise ValueError(
                f"MetalAA term remap atom order mismatch for residue {residue_key}: "
                f"large_model atoms {large_names}, site_model atoms {site_names}."
            )
        for (large_index, _large_atom), (site_index, _site_atom) in zip(large_entries, site_entries, strict=True):
            large_to_site_index[large_index] = site_index

    def mapped_atoms(atom_indices: tuple[int, ...]) -> tuple[int, ...] | None:
        mapped: list[int] = []
        for atom_index in atom_indices:
            site_index = large_to_site_index.get(atom_index)
            if site_index is None:
                return None
            mapped.append(site_index)
        return tuple(mapped)

    mapped_bonds = []
    for bond in bond_terms:
        atoms = mapped_atoms(bond.atoms)
        if atoms is None:
            continue
        mapped_bonds.append(
            type(bond)(
                atoms=atoms,
                atom_types=bond.atom_types,
                kBond=bond.kBond,
                rEq=bond.rEq,
            )
        )

    mapped_angles = []
    for angle in angle_terms:
        atoms = mapped_atoms(angle.atoms)
        if atoms is None:
            continue
        mapped_angles.append(
            type(angle)(
                atoms=atoms,
                atom_types=angle.atom_types,
                kTheta=angle.kTheta,
                thetaEq=angle.thetaEq,
            )
        )
    return mapped_bonds, mapped_angles


def _external_residue_labels(structure: dict, managed_keys: set[tuple[str, int, str]]) -> list[str]:
    labels: list[str] = []
    for residue in sorted(structure["residues"], key=residue_sort_key):
        if residue.get("kind") not in {"ligand", "cofactor"}:
            continue
        if get_resid_key(residue) in managed_keys:
            continue
        labels.append(get_resid_label(residue))
    return labels


def _prepare_metal_large_model(structure: dict, config: MetalAbinitioConfig) -> tuple[MetalSiteSelection, MetalModelBundle]:
    selection = find_metal_site_core(
        structure,
        target_residue=config.target_residue,
        add_resid=config.add_resid,
        set_bonded=config.set_bonded,
        donor_cutoff=config.donor_cutoff,
    )
    bundle = build_metal_model_bundle(
        structure,
        target=config.target,
        add_resid=config.add_resid,
        cluster_cutoff=config.cluster_cutoff,
        donor_cutoff=config.donor_cutoff,
        selection=selection,
    )
    metal_formal_charge = config.oxy if config.oxy is not None else config.charge
    _annotate_target_ion_formal_charge(bundle.large_model, bundle.large_model.get("target_key"), metal_formal_charge)
    bundle.large_charge = infer_model_charge(config.charge, bundle.large_model)
    bundle.large_mult = config.mult
    bundle.large_model["charge"] = bundle.large_charge
    bundle.large_model["mult"] = bundle.large_mult
    return selection, bundle


def _reselect_optimized_core(
    *,
    bundle: MetalModelBundle,
    selection: MetalSiteSelection,
    config: MetalAbinitioConfig,
) -> _OptimizedCore:
    target_key = bundle.large_model.get("target_key")
    residues_by_key = {get_resid_key(residue): residue for residue in bundle.large_model["residues"]}
    target_residue = residues_by_key.get(target_key)
    if target_key is None or target_residue is None:
        raise ValueError("Optimized large_model does not contain the target metal residue.")
    metal_atom = _single_atom_residue_atom(target_residue, label="MetalAA target metal residue")
    metal_xyz = get_atom_xyz(metal_atom)
    manual_core_keys = {get_resid_key(residue) for residue in selection.manual_core_residues}
    donor_atoms: dict[tuple[str, int, str], list[str]] = {}
    auto_core_keys: set[tuple[str, int, str]] = set()
    if config.set_bonded:
        for residue_key, atom_names in selection.donor_atoms.items():
            residue = residues_by_key.get(residue_key)
            if residue is None:
                raise ValueError(f"Explicit MetalAA donor residue {residue_key!r} is absent from optimized large_model.")
            residue_atom_names = {atom["name"] for atom in residue["atoms"]}
            missing = sorted(set(atom_names) - residue_atom_names)
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"Explicit MetalAA donor atom(s) missing after optimization for {residue_key!r}: {joined}.")
            donor_atoms[residue_key] = list(atom_names)
            if residue_key not in manual_core_keys:
                auto_core_keys.add(residue_key)
    else:
        for residue in sorted(bundle.large_model["residues"], key=residue_sort_key):
            residue_key = get_resid_key(residue)
            if residue_key == target_key or residue.get("kind") in {"ion", "cap", "small_model"}:
                continue
            typed_nonprotein = residue.get("kind") in {"ligand", "cofactor"} and bool(residue.get("_cfmol2_path"))
            if residue.get("kind") != "protein" and residue_key not in manual_core_keys and not typed_nonprotein:
                continue
            donor_names: list[str] = []
            for atom in residue["atoms"]:
                if atom["element"].upper() not in METAL_SITE_DONOR_ELEMENTS:
                    continue
                delta = get_atom_xyz(atom) - metal_xyz
                if float(np.sqrt(np.dot(delta, delta))) <= config.donor_cutoff:
                    donor_names.append(atom["name"])
            if donor_names:
                donor_atoms[residue_key] = sorted(set(donor_names))
                if (residue.get("kind") == "protein" or typed_nonprotein) and residue_key not in manual_core_keys:
                    auto_core_keys.add(residue_key)

    core_keys = {target_key} | manual_core_keys | auto_core_keys
    donor_atoms = {
        residue_key: atom_names
        for residue_key, atom_names in donor_atoms.items()
        if residue_key in core_keys
    }
    core_residues = [
        residue
        for residue in sorted(bundle.large_model["residues"], key=residue_sort_key)
        if get_resid_key(residue) in core_keys
    ]
    updated_selection = MetalSiteSelection(
        target=target_residue,
        core_residues=core_residues,
        auto_core_residues=[
            residue
            for residue in core_residues
            if get_resid_key(residue) in auto_core_keys
        ],
        manual_core_residues=[
            residue
            for residue in core_residues
            if get_resid_key(residue) in manual_core_keys
        ],
        donor_atoms=donor_atoms,
        warnings=list(selection.warnings),
        donor_cutoff=config.donor_cutoff,
    )
    bundle.selection = updated_selection
    bundle.large_model["core_keys"] = [get_resid_key(residue) for residue in core_residues]
    bundle.large_model["donor_atoms"] = dict(donor_atoms)
    return _OptimizedCore(
        selection=updated_selection,
        core_residues=core_residues,
        donor_atoms=donor_atoms,
        target_key=target_key,
        target_residue=target_residue,
        metal_atom=metal_atom,
    )


def _build_large_resp_problem(bundle: MetalModelBundle, core: _OptimizedCore) -> _RespProblem:
    flattened_large = flatten_model_atoms(bundle.large_model)
    index_by_residue_atom = {
        (get_resid_key(residue), atom["name"]): atom_index
        for atom_index, (residue, atom) in enumerate(flattened_large, start=1)
    }
    large_entries_by_residue = _atom_entries_by_residue(bundle.large_model)
    core_keys = {get_resid_key(residue) for residue in core.core_residues}
    charge_groups: list[tuple[list[int], float]] = []
    for residue in sorted(bundle.large_model["residues"], key=residue_sort_key):
        residue_key = get_resid_key(residue)
        if residue_key in core_keys:
            continue
        atom_indices = [atom_index for atom_index, _atom in large_entries_by_residue[residue_key]]
        if "formal_charge" in residue:
            target_charge = int(residue["formal_charge"])
        elif residue.get("kind") in {"ligand", "cofactor"}:
            raise ValueError(
                f"Non-site residue {get_resid_label(residue)}:{residue['resname']} "
                "requires a ligand/NCAA template or an explicit formal_charge."
            )
        else:
            target_charge = CHARGED_STANDARD_RESIDUES.get(residue["resname"].upper(), 0)
        charge_groups.append((atom_indices, float(target_charge)))

    metal_index = index_by_residue_atom[(core.target_key, core.metal_atom["name"])]
    metal_indices = {
        atom_index
        for atom_index, (residue, _atom) in enumerate(flattened_large, start=1)
        if get_resid_key(residue) == core.target_key
    }
    donor_metal_pairs: set[tuple[int, int]] = set()
    for donor_key, atom_names in core.donor_atoms.items():
        for atom_name in atom_names:
            donor_index = index_by_residue_atom.get((donor_key, atom_name))
            if donor_index is not None:
                donor_metal_pairs.add(tuple(sorted((donor_index, metal_index))))

    large_bond_pairs_set: set[tuple[int, int]] = set()
    typed_cofactor_atom_indices: set[int] = set()
    for residue in bundle.large_model["residues"]:
        name_pairs = residue.get("_cfmol2_bond_name_pairs")
        if not name_pairs:
            continue
        entries = large_entries_by_residue[get_resid_key(residue)]
        typed_cofactor_atom_indices.update(atom_index for atom_index, _atom in entries)
        large_bond_pairs_set.update(
            _bond_pairs_from_names(
                name_pairs,
                entries,
                label=get_resid_label(residue),
            )
        )
    for left, right in infer_bond_pairs(bundle.large_model, source_structure=bundle.large_model):
        pair = tuple(sorted((left, right)))
        if typed_cofactor_atom_indices.intersection(pair):
            continue
        if metal_indices.intersection(pair):
            continue
        large_bond_pairs_set.add(pair)
    large_bond_pairs_set.update(donor_metal_pairs)
    return _RespProblem(
        bond_pairs=sorted(large_bond_pairs_set),
        charge_groups=charge_groups,
    )


def _export_metal_site_model(
    *,
    structure: dict,
    bundle: MetalModelBundle,
    core: _OptimizedCore,
    resp_result,
    artifacts: MetalArtifacts,
    config: MetalAbinitioConfig,
) -> _SiteExport:
    bundle.site_model = build_metal_site_model(
        bundle.large_model,
        core.core_residues,
        donor_atoms=core.donor_atoms,
    )
    metal_formal_charge = config.oxy if config.oxy is not None else config.charge
    _annotate_target_ion_formal_charge(bundle.site_model, bundle.site_model.get("target_key"), metal_formal_charge)
    bundle.site_model, charge_warnings = project_resp_charges_onto_site_model(bundle.site_model, resp_result.model)
    site_charge = infer_model_charge(config.charge, bundle.site_model)
    resp_charge_sum = sum(
        float(atom.get("charge", 0.0))
        for residue in bundle.site_model["residues"]
        for atom in residue["atoms"]
    )
    if abs(resp_charge_sum - float(site_charge)) > 1.0e-4:
        raise ValueError(
            f"Site-model RESP charge {resp_charge_sum:.6f} does not match "
            f"integer target {site_charge:d}."
        )
    bundle.site_model["charge"] = site_charge
    bundle.site_model["mult"] = config.mult
    bundle.site_model["warnings"] = list(bundle.large_model.get("warnings", [])) + charge_warnings
    artifacts.files["gaussian_input"] = resp_result.files["gaussian_input"]
    artifacts.resp_files.clear()
    artifacts.resp_files.update(resp_result.resp_files)
    site_pdb_path, site_mol2_path, site_typing = write_site_model_files(
        artifacts,
        structure=structure,
        site_model=bundle.site_model,
        watm=config.watm,
        ionm=config.ionm,
        prom=config.prom,
        cofactor_frcmods=artifacts.cofactor_frcmods,
        cofactor_frcmod_by_residue=artifacts.cofactor_frcmod_by_residue,
    )
    return _SiteExport(
        site_pdb_path=site_pdb_path,
        site_mol2_path=site_mol2_path,
        site_typing=site_typing,
    )


def _export_metal_bonded_frcmod(
    *,
    source_atoms,
    bundle: MetalModelBundle,
    resp_problem: _RespProblem,
    artifacts: MetalArtifacts,
    site_typing: MetalSiteTyping,
    stage_timings: list[tuple[str, float]],
    bonded_method: str,
    output: str,
    qm_hessian=None,
    qm_runner=None,
    vib_scale: float = 1.0,
) -> _BondedResult:
    large_atoms = model_to_atoms(bundle.large_model, charge=bundle.large_charge, mult=bundle.large_mult)
    copy_thresholds(source_atoms, large_atoms)
    large_atoms.calc = source_atoms.calc
    if qm_runner is None and not hasattr(large_atoms.calc, "get_hessian"):
        raise ValueError(
            "Attached calculator does not provide get_hessian(), which is required for metal bond/angle fitting."
        )

    with _timed_stage(stage_timings, "Hessian evaluation"):
        if qm_hessian is not None:
            hessian = qm_hessian
        elif qm_runner is not None:
            qm_freq = qm_runner.opt_frequency(
                large_atoms,
                f"{parmfit_work_prefix(output, 'qm')}_metal_large",
            )
            if qm_freq.hessian is None:
                raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
            hessian = qm_freq.hessian
        else:
            hessian = get_cartesian_hessian(large_atoms)
    large_bond_terms, large_angle_terms = build_bond_angle_terms(
        bundle.large_model,
        source_structure=bundle.large_model,
        bond_pairs=resp_problem.bond_pairs,
    )
    label = "Seminario" if bonded_method == "seminario" else "mSeminario"
    apply_bonded = apply_seminario if bonded_method == "seminario" else apply_mseminario
    bonded_warning: Optional[str] = None
    try:
        apply_bonded(large_atoms, hessian, large_bond_terms, large_angle_terms, vib_scale)
    except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
        bonded_warning = (
            f"{label} fitting could not determine all metal-related bond/angle force constants "
            f"from the supplied Hessian: {exc}"
        )
    bond_terms, angle_terms = _remap_terms_to_site_model(
        bundle.large_model,
        bundle.site_model,
        large_bond_terms,
        large_angle_terms,
    )
    frcmod_path = write_site_frcmod(
        artifacts,
        site_model=bundle.site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        site_typing=site_typing,
    )
    return _BondedResult(frcmod_path=frcmod_path, bonded_warning=bonded_warning)


def run_metal_abinitio(
    *,
    output: str,
    source_atoms,
    structure: dict,
    config: MetalAbinitioConfig,
    log_info: Callable[[list], None],
) -> MetalWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    qm_runner = build_qm_reference_runner(config.qm)
    artifacts = plan_metal_artifacts(output)
    cofactor_templates = apply_cfmol2_templates(structure, config.cfmol2)
    cofactor_frcmod_by_residue = build_cofactor_orig_frcmods(output, cofactor_templates)
    artifacts.cofactor_frcmod_by_residue.clear()
    artifacts.cofactor_frcmod_by_residue.update(cofactor_frcmod_by_residue)
    artifacts.cofactor_frcmods.clear()
    artifacts.cofactor_frcmods.extend(cofactor_frcmod_by_residue.values())
    with _timed_stage(stage_timings, "site selection/model build"):
        selection, bundle = _prepare_metal_large_model(structure, config)

    log_info(format_metal_start_lines(config, large_charge=bundle.large_charge, large_mult=bundle.large_mult))
    write_large_pdb(artifacts, bundle.large_model, optimized=False)
    log_info(["  [MetalAA] large-model input written.\n"])
    qm_large_hessian = None

    log_info(["  [MetalAA] MLIP large-model optimization ...\n"])
    with _timed_stage(stage_timings, "large optimization"):
        bundle.large_model = optimize_model_geometry(
            bundle.large_model,
            output=output,
            source_atoms=source_atoms,
            max_iter=config.opt_max_iter,
            max_step=config.opt_max_step,
            failure_message="Metal-site optimization did not converge for large_model.",
        )
        if qm_runner is not None:
            log_info(["  [MetalAA] QM reference optimization ...\n"])
            qm_atoms = model_to_atoms(bundle.large_model, charge=bundle.large_charge, mult=bundle.large_mult)
            qm_result = qm_runner.opt_frequency(qm_atoms, f"{parmfit_work_prefix(output, 'qm')}_metal_large")
            qm_large_hessian = qm_result.hessian
            if qm_large_hessian is None:
                raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
            bundle.large_model = update_model_from_atoms(bundle.large_model, qm_result.atoms)
    metal_formal_charge = config.oxy if config.oxy is not None else config.charge
    _annotate_target_ion_formal_charge(bundle.large_model, bundle.large_model.get("target_key"), metal_formal_charge)
    write_large_pdb(artifacts, bundle.large_model, optimized=True)

    core = _reselect_optimized_core(
        bundle=bundle,
        selection=selection,
        config=config,
    )
    selection = core.selection
    resp_problem = _build_large_resp_problem(bundle, core)

    log_info(["  [MetalAA] large-model RESP ...\n"])
    resp_wfn = None
    if (
        qm_runner is not None
        and config.resp.qm.backend == "gaussian"
        and config.qm.qm_engine in {"gaussian", "g16", "g09"}
    ):
        opt_theory, opt_basis = (part.strip() for part in config.qm.opt_level.strip().split("/", 1))
        if config.resp.qm.theory == opt_theory and config.resp.qm.basis == opt_basis:
            resp_wfn = getattr(qm_runner, "last_wfn_path", None)
    with _timed_stage(stage_timings, "large RESP"):
        with _timed_stage(stage_timings, "large RESP/Gaussian ESP"):
            resp_result = run_resp_pipeline(
                output=output,
                model=bundle.large_model,
                bond_pairs=resp_problem.bond_pairs,
                total_charge=bundle.large_charge,
                multiplicity=bundle.large_mult,
                chgmod=config.resp.chgmod,
                fixchg_resids=config.resp.fixchg_resids,
                qm=config.resp.qm,
                label="metal_large_resp",
                watm=config.watm,
                prom=config.prom,
                charge_groups=resp_problem.charge_groups,
                wfn_path=resp_wfn,
            )

    with _timed_stage(stage_timings, "site export"):
        site_export = _export_metal_site_model(
            structure=structure,
            bundle=bundle,
            core=core,
            resp_result=resp_result,
            artifacts=artifacts,
            config=config,
        )
    log_info(["  [MetalAA] site files written.\n"])

    bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
    hessian_source = "QM Hessian" if qm_large_hessian is not None else "MLIP Hessian"
    log_info([f"  [MetalAA] {hessian_source} + {bonded_label} + final frcmod ...\n"])
    with _timed_stage(stage_timings, f"Hessian/{bonded_label}/frcmod export"):
        bonded = _export_metal_bonded_frcmod(
            source_atoms=source_atoms,
            bundle=bundle,
            resp_problem=resp_problem,
            artifacts=artifacts,
            site_typing=site_export.site_typing,
            stage_timings=stage_timings,
            bonded_method=config.bonded,
            output=output,
            qm_hessian=qm_large_hessian,
            qm_runner=qm_runner,
            vib_scale=config.vib_scale,
        )
    log_info(["  [MetalAA] final parameter files written.\n"])
    log_info(["  [MetalAA] running tleap validation ...\n"])
    with _timed_stage(stage_timings, "tleap validation"):
        interface.run_tleap(
            artifacts.files["tleap_input"],
            workdir=os.path.dirname(artifacts.files["tleap_input"]) or ".",
        )
    log_info(
        format_metal_final_lines(
            artifacts=artifacts,
            atom_type_rows=site_export.site_typing.atom_type_rows,
            ion_frcmods=site_export.site_typing.ion_frcmods,
            metal_formal_charge=site_export.site_typing.metal_formal_charge,
            metal_fitted_charge=site_export.site_typing.metal_fitted_charge,
            bonded_warning=bonded.bonded_warning,
            external_residues=_external_residue_labels(
                structure,
                {get_resid_key(residue) for residue in bundle.selection.core_residues},
            ),
            stage_timings=stage_timings,
        )
    )

    return MetalWorkflowResult(
        selection=selection,
        large_model=bundle.large_model,
        site_model=bundle.site_model,
        artifacts=artifacts,
        site_typing=site_export.site_typing,
        bonded_warning=bonded.bonded_warning,
        stage_timings=list(stage_timings),
    )
