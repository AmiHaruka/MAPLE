"""Usage: run the MetalAA abinitio parameterization workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from .. import interface as amber_interface
from ..mSeminario import apply_mseminario
from ..model import build_bond_angle_terms, flatten_model_atoms, infer_bond_pairs, model_to_atoms
from ..runtime import copy_thresholds, get_cartesian_hessian, optimize_model_geometry, run_resp_pipeline
from ..structure import (
    CHARGED_STANDARD_RESIDUES,
    METAL_SITE_DONOR_ELEMENTS,
    get_atom_xyz,
    get_resid_key,
    get_resid_label,
    residue_sort_key,
)
from .charges import infer_large_model_charge, project_resp_charges_onto_site_model
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
    mseminario_warning: Optional[str] = None
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
class MetalOptimizedCore:
    selection: MetalSiteSelection
    final_core_residues: list[dict]
    optimized_donor_atoms: dict[tuple[str, int, str], list[str]]
    target_key: tuple[str, int, str]
    target_residue: dict
    metal_atom: dict


@dataclass(frozen=True)
class MetalRespProblem:
    bond_pairs: list[tuple[int, int]]
    charge_groups: list[tuple[list[int], float]]


@dataclass(frozen=True)
class MetalDeploymentResult:
    site_pdb_path: str
    site_mol2_path: str
    site_typing: MetalSiteTyping


@dataclass(frozen=True)
class MetalSeminarioResult:
    frcmod_path: str
    mseminario_warning: Optional[str]


def _annotate_target_ion_formal_charge(model: dict, target_key: tuple[str, int, str] | None, formal_charge: int) -> None:
    for residue in model["residues"]:
        if residue.get("kind") == "ion" and get_resid_key(residue) == target_key:
            residue["formal_charge"] = int(formal_charge)


def _remap_terms_to_site_model(large_model: dict, site_model: dict, bond_terms, angle_terms):
    large_atoms = [atom for _residue, atom in flatten_model_atoms(large_model)]
    serial_to_site_index = {
        int(atom["serial"]): atom_index
        for atom_index, (_residue, atom) in enumerate(flatten_model_atoms(site_model), start=1)
    }

    def mapped_atoms(atom_indices: tuple[int, ...]) -> tuple[int, ...] | None:
        mapped: list[int] = []
        for atom_index in atom_indices:
            serial = int(large_atoms[atom_index - 1]["serial"])
            site_index = serial_to_site_index.get(serial)
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
    bundle.large_charge = infer_large_model_charge(config.charge, bundle.large_model)
    bundle.large_mult = config.mult
    bundle.large_model["charge"] = bundle.large_charge
    bundle.large_model["mult"] = bundle.large_mult
    return selection, bundle


def _reselect_optimized_core(
    *,
    bundle: MetalModelBundle,
    selection: MetalSiteSelection,
    config: MetalAbinitioConfig,
) -> MetalOptimizedCore:
    target_key = bundle.large_model.get("target_key")
    residues_by_key = {get_resid_key(residue): residue for residue in bundle.large_model["residues"]}
    target_residue = residues_by_key.get(target_key)
    if target_key is None or target_residue is None:
        raise ValueError("Optimized large_model does not contain the target metal residue.")
    metal_atom = sorted(target_residue["atoms"], key=lambda atom: atom["serial"])[0]
    metal_xyz = get_atom_xyz(metal_atom)
    manual_core_keys = {get_resid_key(residue) for residue in selection.manual_core_residues}
    optimized_donor_atoms: dict[tuple[str, int, str], list[str]] = {}
    optimized_auto_core_keys: set[tuple[str, int, str]] = set()
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
            optimized_donor_atoms[residue_key] = sorted(set(donor_names))
            if (residue.get("kind") == "protein" or typed_nonprotein) and residue_key not in manual_core_keys:
                optimized_auto_core_keys.add(residue_key)

    final_core_keys = {target_key} | manual_core_keys | optimized_auto_core_keys
    optimized_donor_atoms = {
        residue_key: atom_names
        for residue_key, atom_names in optimized_donor_atoms.items()
        if residue_key in final_core_keys
    }
    final_core_residues = [
        residue
        for residue in sorted(bundle.large_model["residues"], key=residue_sort_key)
        if get_resid_key(residue) in final_core_keys
    ]
    optimized_selection = MetalSiteSelection(
        target=target_residue,
        core_residues=final_core_residues,
        auto_core_residues=[
            residue
            for residue in final_core_residues
            if get_resid_key(residue) in optimized_auto_core_keys
        ],
        manual_core_residues=[
            residue
            for residue in final_core_residues
            if get_resid_key(residue) in manual_core_keys
        ],
        donor_atoms=optimized_donor_atoms,
        warnings=list(selection.warnings),
        donor_cutoff=config.donor_cutoff,
    )
    bundle.selection = optimized_selection
    bundle.large_model["core_keys"] = [get_resid_key(residue) for residue in final_core_residues]
    bundle.large_model["donor_atoms"] = dict(optimized_donor_atoms)
    return MetalOptimizedCore(
        selection=optimized_selection,
        final_core_residues=final_core_residues,
        optimized_donor_atoms=optimized_donor_atoms,
        target_key=target_key,
        target_residue=target_residue,
        metal_atom=metal_atom,
    )


def _build_large_resp_problem(bundle: MetalModelBundle, core: MetalOptimizedCore) -> MetalRespProblem:
    flattened_large = flatten_model_atoms(bundle.large_model)
    index_by_residue_atom = {
        (get_resid_key(residue), atom["name"]): atom_index
        for atom_index, (residue, atom) in enumerate(flattened_large, start=1)
    }
    large_serial_to_index = {
        int(atom["serial"]): atom_index
        for atom_index, (_residue, atom) in enumerate(flattened_large, start=1)
    }
    final_core_keys = {get_resid_key(residue) for residue in core.final_core_residues}
    charge_groups: list[tuple[list[int], float]] = []
    for residue in sorted(bundle.large_model["residues"], key=residue_sort_key):
        residue_key = get_resid_key(residue)
        if residue_key in final_core_keys:
            continue
        atom_indices = [large_serial_to_index[int(atom["serial"])] for atom in residue["atoms"]]
        if "formal_charge" in residue:
            target_charge = int(residue["formal_charge"])
        elif residue.get("kind") in {"ligand", "cofactor"}:
            raise ValueError(
                f"Non-deployment residue {get_resid_label(residue)}:{residue['resname']} "
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
    for donor_key, atom_names in core.optimized_donor_atoms.items():
        for atom_name in atom_names:
            donor_index = index_by_residue_atom.get((donor_key, atom_name))
            if donor_index is not None:
                donor_metal_pairs.add(tuple(sorted((donor_index, metal_index))))

    large_bond_pairs_set: set[tuple[int, int]] = set()
    for left, right in infer_bond_pairs(bundle.large_model, source_structure=bundle.large_model):
        pair = tuple(sorted((left, right)))
        if metal_indices.intersection(pair):
            continue
        large_bond_pairs_set.add(pair)
    large_bond_pairs_set.update(donor_metal_pairs)
    return MetalRespProblem(
        bond_pairs=sorted(large_bond_pairs_set),
        charge_groups=charge_groups,
    )


def _deploy_metal_site_model(
    *,
    structure: dict,
    bundle: MetalModelBundle,
    core: MetalOptimizedCore,
    resp_result,
    artifacts: MetalArtifacts,
    config: MetalAbinitioConfig,
) -> MetalDeploymentResult:
    bundle.site_model = build_metal_site_model(
        bundle.large_model,
        core.final_core_residues,
        donor_atoms=core.optimized_donor_atoms,
    )
    metal_formal_charge = config.oxy if config.oxy is not None else config.charge
    _annotate_target_ion_formal_charge(bundle.site_model, bundle.site_model.get("target_key"), metal_formal_charge)
    bundle.site_model, charge_warnings = project_resp_charges_onto_site_model(bundle.site_model, resp_result.model)
    deployment_charge = infer_large_model_charge(config.charge, bundle.site_model)
    actual_deployment_charge = sum(
        float(atom.get("charge", 0.0))
        for residue in bundle.site_model["residues"]
        for atom in residue["atoms"]
    )
    if abs(actual_deployment_charge - float(deployment_charge)) > 1.0e-4:
        raise ValueError(
            f"Deployment RESP charge {actual_deployment_charge:.6f} does not match "
            f"integer target {deployment_charge:d}."
        )
    bundle.site_model["charge"] = deployment_charge
    bundle.site_model["mult"] = config.mult
    bundle.site_model["warnings"] = list(bundle.large_model.get("warnings", [])) + charge_warnings
    artifacts.files["gaussian_input"] = resp_result.files["gaussian_input"]
    artifacts.resp_files.clear()
    artifacts.resp_files.update(resp_result.resp_files)
    site_pdb_path, site_mol2_path, site_typing = write_site_model_files(
        artifacts,
        structure=structure,
        site_model=bundle.site_model,
        watm=config.resp.watm,
        ionm=config.ionm,
        cofactor_frcmods=artifacts.cofactor_frcmods,
    )
    return MetalDeploymentResult(
        site_pdb_path=site_pdb_path,
        site_mol2_path=site_mol2_path,
        site_typing=site_typing,
    )


def _export_metal_mseminario_frcmod(
    *,
    output: str,
    source_atoms,
    bundle: MetalModelBundle,
    resp_problem: MetalRespProblem,
    artifacts: MetalArtifacts,
    site_typing: MetalSiteTyping,
    stage_timings: list[tuple[str, float]],
) -> MetalSeminarioResult:
    large_atoms = model_to_atoms(bundle.large_model, charge=bundle.large_charge, mult=bundle.large_mult)
    copy_thresholds(source_atoms, large_atoms)
    large_atoms.calc = source_atoms.calc
    if not hasattr(large_atoms.calc, "get_hessian"):
        raise ValueError("Attached calculator does not provide get_hessian(), which is required for metal bond/angle fitting.")

    with _timed_stage(stage_timings, "Hessian evaluation"):
        hessian = get_cartesian_hessian(large_atoms)
    large_bond_terms, large_angle_terms = build_bond_angle_terms(
        bundle.large_model,
        source_structure=bundle.large_model,
        bond_pairs=resp_problem.bond_pairs,
    )
    mseminario_warning: Optional[str] = None
    try:
        apply_mseminario(large_atoms, hessian, large_bond_terms, large_angle_terms)
    except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
        mseminario_warning = (
            "Modified Seminario fitting could not determine all metal-related bond/angle force constants "
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
        typing=site_typing,
    )
    return MetalSeminarioResult(frcmod_path=frcmod_path, mseminario_warning=mseminario_warning)


def run_metal_abinitio(
    *,
    output: str,
    source_atoms,
    structure: dict,
    config: MetalAbinitioConfig,
    log_info: Callable[[list], None],
) -> MetalWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    artifacts = plan_metal_artifacts(output)
    cofactor_templates = apply_cfmol2_templates(structure, config.cfmol2)
    artifacts.cofactor_frcmods.clear()
    artifacts.cofactor_frcmods.extend(build_cofactor_orig_frcmods(output, cofactor_templates))
    with _timed_stage(stage_timings, "site selection/model build"):
        selection, bundle = _prepare_metal_large_model(structure, config)

    log_info(format_metal_start_lines(config, large_charge=bundle.large_charge, large_mult=bundle.large_mult))
    write_large_pdb(artifacts, bundle.large_model, optimized=False)
    log_info(["  [MetalAA] large-model input written.\n"])

    log_info(["  [MetalAA] large-model optimization ...\n"])
    with _timed_stage(stage_timings, "large optimization"):
        bundle.large_model = optimize_model_geometry(
            bundle.large_model,
            output=output,
            source_atoms=source_atoms,
            max_iter=config.opt_max_iter,
            max_step=config.opt_max_step,
            failure_message="Metal-site optimization did not converge for large_model.",
        )
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
                watm=config.resp.watm,
                charge_groups=resp_problem.charge_groups,
            )

    with _timed_stage(stage_timings, "site deployment export"):
        deployment = _deploy_metal_site_model(
            structure=structure,
            bundle=bundle,
            core=core,
            resp_result=resp_result,
            artifacts=artifacts,
            config=config,
        )
    log_info(["  [MetalAA] site deployment files written.\n"])

    log_info(["  [MetalAA] Hessian + mSeminario + final frcmod ...\n"])
    with _timed_stage(stage_timings, "Hessian/mSeminario/frcmod export"):
        seminario = _export_metal_mseminario_frcmod(
            output=output,
            source_atoms=source_atoms,
            bundle=bundle,
            resp_problem=resp_problem,
            artifacts=artifacts,
            site_typing=deployment.site_typing,
            stage_timings=stage_timings,
        )
    log_info(["  [MetalAA] final parameter files written.\n"])
    log_info(["  [MetalAA] running tleap validation ...\n"])
    with _timed_stage(stage_timings, "tleap validation"):
        amber_interface.run_tleap(
            artifacts.files["tleap_input"],
            workdir=os.path.dirname(artifacts.files["tleap_input"]) or ".",
        )
    log_info(
        format_metal_final_lines(
            artifacts=artifacts,
            atom_type_rows=deployment.site_typing.atom_type_rows,
            ion_frcmods=deployment.site_typing.ion_frcmods,
            metal_formal_charge=deployment.site_typing.metal_formal_charge,
            metal_fitted_charge=deployment.site_typing.metal_fitted_charge,
            mseminario_warning=seminario.mseminario_warning,
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
        site_typing=deployment.site_typing,
        mseminario_warning=seminario.mseminario_warning,
        stage_timings=list(stage_timings),
    )
