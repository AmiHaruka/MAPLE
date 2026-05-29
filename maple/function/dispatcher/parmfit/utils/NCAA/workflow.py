"""Usage: run the NCAA abinitio parameterization workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Callable

from .. import interface as amber_interface
from ..context import find_prev_next_peptide_residues, find_residue_by_key
from ..Seminario import apply_seminario
from ..mSeminario import apply_mseminario
from ..model import infer_bond_pairs, model_to_atoms
from ..readparm import CorrectionParameterSet, build_correction_parameter_set
from ..runtime import (
    copy_thresholds,
    get_cartesian_hessian,
    parmfit_work_prefix,
    run_multiconformer_resp,
)
from ..structure import copy_residue
from ..TorsionFit import TorsionScanRuntime, TorsionWorkflowResult, run_torsion_workflow
from .artifacts import NCAAArtifacts, build_ncaa_amber_artifacts, build_ncaa_export_bundle
from .config import NCAAAbinitioConfig
from .models import NCAAConformer, NCAAIdentity, build_capped_ncaa_model, build_ncaa_center_bond_filter, build_ncaa_sidechain_relax_indices, build_resp_conformers_from_reference, identity_ncaa, optimize_capped_reference, warn_capped_proton_transfer
from .report import format_ncaa_final_lines, format_ncaa_start_lines


@contextmanager
def _timed_stage(stage_timings: list[tuple[str, float]], label: str):
    start = perf_counter()
    try:
        yield
    finally:
        stage_timings.append((label, perf_counter() - start))


@dataclass(frozen=True)
class NCAAWorkflowResult:
    identity: NCAAIdentity
    representative: NCAAConformer
    conformers: list[NCAAConformer]
    parameter_set: CorrectionParameterSet
    torsion: TorsionWorkflowResult
    artifacts: NCAAArtifacts
    resp_files: dict[str, str]
    representative_model: dict
    residue_model: dict
    stage_timings: list[tuple[str, float]] = field(default_factory=list)

    @property
    def chirality(self) -> str:
        return self.identity.chirality

    @property
    def representative_conformer(self) -> str:
        return self.representative.label

    @property
    def files(self) -> dict[str, str]:
        return self.artifacts.files


@dataclass(frozen=True)
class NCAAModelBundle:
    identity: NCAAIdentity
    representative: NCAAConformer
    conformers: list[NCAAConformer]
    sidechain_relax_indices: tuple[int, ...]


def _prepare_ncaa_models(
    *,
    output: str,
    source_atoms,
    structure: dict,
    target_residue: dict,
    config: NCAAAbinitioConfig,
) -> NCAAModelBundle:
    identity = identity_ncaa(target_residue)
    prev_residue, next_residue = find_prev_next_peptide_residues(structure, target_residue)
    capped_model = build_capped_ncaa_model(
        target_residue,
        config.rn,
        prev_residue=prev_residue,
        next_residue=next_residue,
    )
    capped_model["charge"] = config.charge
    capped_model["mult"] = config.mult
    sidechain_relax_indices = build_ncaa_sidechain_relax_indices(capped_model)
    sidechain_relax_set = set(sidechain_relax_indices)
    atom_count = sum(len(residue["atoms"]) for residue in capped_model["residues"])
    frozen_indices = tuple(
        index - 1
        for index in range(1, atom_count + 1)
        if index not in sidechain_relax_set
    )
    work_prefix = parmfit_work_prefix(output, "ncaa")
    representative = optimize_capped_reference(
        capped_model,
        source_atoms=source_atoms,
        output=f"{work_prefix}_reference.out",
        frozen_indices=frozen_indices,
    )
    conformers = build_resp_conformers_from_reference(
        representative.model,
        chirality=identity.chirality,
        source_atoms=source_atoms,
        output_base=work_prefix,
    )
    warn_capped_proton_transfer([representative, *conformers])
    return NCAAModelBundle(
        identity=identity,
        representative=representative,
        conformers=conformers,
        sidechain_relax_indices=sidechain_relax_indices,
    )

def _refine_ncaa_parameters(
    *,
    output: str,
    source_atoms,
    representative_model: dict,
    typed_mol2_path: str,
    frcmod_path: str,
    config: NCAAAbinitioConfig,
    sidechain_relax_indices: tuple[int, ...],
    stage_timings: list[tuple[str, float]],
) -> tuple[CorrectionParameterSet, TorsionWorkflowResult]:
    representative_atoms = model_to_atoms(
        representative_model,
        charge=representative_model.get("charge"),
        mult=representative_model.get("mult"),
    )
    representative_atoms.calc = source_atoms.calc
    copy_thresholds(source_atoms, representative_atoms)
    bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
    apply_bonded = apply_seminario if config.bonded == "seminario" else apply_mseminario
    with _timed_stage(stage_timings, f"{bonded_label} setup/Hessian"):
        amber_interface.patch_frcmod_crossterms(frcmod_path)
        stage0_result = build_correction_parameter_set(representative_atoms, typed_mol2_path, frcmod_path)

        hessian = get_cartesian_hessian(representative_atoms)
        apply_bonded(
            representative_atoms,
            hessian,
            stage0_result.bonds,
            stage0_result.angles,
            config.vib_scale,
        )

    with _timed_stage(stage_timings, "TorsionFit"):
        torsion_kwargs = {
            "atoms": representative_atoms,
            "output": output,
            "parameter_set": stage0_result,
            "params": config.torsion,
            "runtime": TorsionScanRuntime(
                max_iter=256,
                memory=int(max(config.resp.qm.mem, 1)),
                curvature=0.6,
                max_step=0.2,
                backend=config.torsion.backend,
                constraint_mode=config.torsion.constraint_mode,
            ),
            "center_bond_filter": build_ncaa_center_bond_filter(representative_model),
            "mobile_atoms": sidechain_relax_indices,
        }
        torsion = run_torsion_workflow(**torsion_kwargs)
    return torsion.final_parameter_set, torsion


def run_ncaa_abinitio(
    *,
    output: str,
    source_atoms,
    structure: dict,
    target_residue: dict,
    config: NCAAAbinitioConfig,
    log_info: Callable[[list], None],
) -> NCAAWorkflowResult:
    stage_timings: list[tuple[str, float]] = []
    log_info(["  [NCAA] model preparation + reference optimization ...\n"])
    with _timed_stage(stage_timings, "model preparation/reference optimization"):
        prepared = _prepare_ncaa_models(
            output=output,
            source_atoms=source_atoms,
            structure=structure,
            target_residue=target_residue,
            config=config,
        )
    log_info(
        format_ncaa_start_lines(
            config=config,
            identity=prepared.identity,
        )
    )

    log_info(["  [NCAA] multiconformer RESP ...\n"])
    with _timed_stage(stage_timings, "multiconformer RESP"):
        resp_result = run_multiconformer_resp(
            output=output,
            conformers=[(conformer.label, conformer.model) for conformer in prepared.conformers],
            representative_model=prepared.representative.model,
            residue_key=prepared.identity.residue_key,
            bond_pairs=infer_bond_pairs(prepared.representative.model),
            total_charge=config.charge,
            multiplicity=config.mult,
            qm=config.resp.qm,
            prom=config.prom,
        )
    representative_model = resp_result.model
    charged_residue = find_residue_by_key(
        representative_model,
        prepared.identity.residue_key,
        label="Representative NCAA target residue",
    )
    residue_model = {"name": "ncaa_residue_model", "residues": [copy_residue(charged_residue)]}

    log_info(["  [NCAA] AmberTools template build ...\n"])
    with _timed_stage(stage_timings, "AmberTools template build"):
        amber_artifacts = build_ncaa_amber_artifacts(
            output=output,
            representative_model=representative_model,
            charged_residue=charged_residue,
            resp_result=resp_result,
            config=config,
        )
    bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
    log_info([f"  [NCAA] Hessian + {bonded_label} + TorsionFit ...\n"])
    parameter_set, torsion = _refine_ncaa_parameters(
        output=output,
        source_atoms=source_atoms,
        representative_model=representative_model,
        typed_mol2_path=amber_artifacts.gaff2_mol2,
        frcmod_path=amber_artifacts.frcmod,
        config=config,
        sidechain_relax_indices=prepared.sidechain_relax_indices,
        stage_timings=stage_timings,
    )
    log_info(["  [NCAA] writing refined templates + tleap input ...\n"])
    with _timed_stage(stage_timings, "final export"):
        export_bundle = build_ncaa_export_bundle(
            output,
            amber=amber_artifacts,
            representative_model=representative_model,
            charged_residue=charged_residue,
            conformers=prepared.conformers,
            final_parameter_set=parameter_set,
            config=config,
            structure=structure,
            target_residue=target_residue,
        )

    log_info(["  [NCAA] tleap validation ...\n"])
    with _timed_stage(stage_timings, "tleap validation"):
        amber_interface.run_tleap(
            export_bundle.artifacts.tleap_input,
            workdir=os.path.dirname(export_bundle.artifacts.tleap_input) or ".",
        )

    log_info(
        format_ncaa_final_lines(
            artifacts=export_bundle.artifacts,
            atom_type_rows=export_bundle.atom_type_rows,
            stage_timings=stage_timings,
        )
    )

    return NCAAWorkflowResult(
        identity=prepared.identity,
        representative=prepared.representative,
        conformers=prepared.conformers,
        parameter_set=parameter_set,
        torsion=torsion,
        artifacts=export_bundle.artifacts,
        resp_files=resp_result.resp_files,
        representative_model=representative_model,
        residue_model=residue_model,
        stage_timings=list(stage_timings),
    )
