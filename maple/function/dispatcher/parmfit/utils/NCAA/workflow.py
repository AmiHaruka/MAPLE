"""Usage: run the NCAA abinitio parameterization workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Callable

from .. import interface
from ..chargefit import (
    ChargeFitResult,
    apply_model_charges,
    fit_multiconformer_charges,
)
from ..context import find_prev_next_peptide_residues, find_residue_by_key
from ..Seminario import apply_seminario
from ..mSeminario import apply_mseminario
from ..model import infer_bond_pairs, model_to_atoms, update_model_from_atoms
from ..QMInterface import build_qm_reference_runner
from ..readparm import CorrectionParameterSet, build_correction_parameter_set
from ..runtime import (
    copy_thresholds,
    get_cartesian_hessian,
    parmfit_work_prefix,
)
from ..structure import copy_residue
from ..TorsionFit import TorsionScanRuntime, TorsionWorkflowResult, run_torsion_workflow
from .artifacts import NCAAArtifacts, build_ncaa_amber_artifacts, build_ncaa_export_bundle
from .config import NCAAAbinitioConfig
from .models import NCAAConformer, NCAAIdentity, build_capped_ncaa_model, build_charge_conformers, build_ncaa_center_bond_filter, build_ncaa_sidechain_relax_indices, identity_ncaa, optimize_capped_confs, warn_capped_proton_transfer
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
    charge_result: ChargeFitResult
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
    qm_hessian: object | None = None


def _prepare_ncaa_models(
    *,
    output: str,
    source_atoms,
    structure: dict,
    target_residue: dict,
    config: NCAAAbinitioConfig,
    qm_runner=None,
    log_info: Callable[[list], None] | None = None,
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
    representative = optimize_capped_confs(
        capped_model,
        source_atoms=source_atoms,
        output=f"{work_prefix}_reference.out",
        frozen_indices=frozen_indices,
        max_iter=config.opt_max_iter,
        max_step=config.opt_max_step,
    )
    qm_hessian = None
    needs_qm_reference = config.bonded != "none" or bool(config.torsion.enabled)
    if qm_runner is not None and needs_qm_reference:
        if log_info is not None:
            log_info(["  [NCAA] QM reference optimization ...\n"])
        qm_atoms = model_to_atoms(representative.model, charge=config.charge, mult=config.mult)
        # The MLIP preoptimization preserves peptide context with a frozen
        # backbone. Relax all coordinates for a stationary-point QM Hessian.
        if config.bonded != "none":
            qm_result = qm_runner.opt_frequency(
                qm_atoms,
                f"{work_prefix}_reference_qm",
            )
            qm_hessian = qm_result.hessian
            if qm_hessian is None:
                raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
        else:
            qm_result = qm_runner.optimize(
                qm_atoms,
                f"{work_prefix}_reference_qm",
            )
        representative = NCAAConformer(
            label=representative.label,
            phi_deg=representative.phi_deg,
            psi_deg=representative.psi_deg,
            energy=qm_result.energy_hartree,
            model=update_model_from_atoms(representative.model, qm_result.atoms),
        )
    conformers = build_charge_conformers(
        representative.model,
        chirality=identity.chirality,
        source_atoms=source_atoms,
        output_base=work_prefix,
        max_iter=config.opt_max_iter,
        max_step=config.opt_max_step,
    )
    warn_capped_proton_transfer([representative, *conformers])
    return NCAAModelBundle(
        identity=identity,
        representative=representative,
        conformers=conformers,
        sidechain_relax_indices=sidechain_relax_indices,
        qm_hessian=qm_hessian,
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
    qm_hessian=None,
    qm_runner=None,
) -> tuple[CorrectionParameterSet, TorsionWorkflowResult]:
    representative_atoms = model_to_atoms(
        representative_model,
        charge=representative_model.get("charge"),
        mult=representative_model.get("mult"),
    )
    representative_atoms.calc = source_atoms.calc
    copy_thresholds(source_atoms, representative_atoms)
    bonded_enabled = config.bonded != "none"
    torsion_enabled = bool(config.torsion.enabled)
    bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
    stage_label = f"{bonded_label} setup/Hessian" if bonded_enabled else "parameter setup"
    with _timed_stage(stage_timings, stage_label):
        if bonded_enabled or torsion_enabled:
            interface.patch_frcmod_crossterms(frcmod_path)
        stage0_result = build_correction_parameter_set(representative_atoms, typed_mol2_path, frcmod_path)

        if bonded_enabled:
            apply_bonded = apply_seminario if config.bonded == "seminario" else apply_mseminario
            if qm_hessian is not None:
                hessian = qm_hessian
            elif qm_runner is not None:
                qm_freq = qm_runner.opt_frequency(
                    representative_atoms,
                    f"{parmfit_work_prefix(output, 'qm')}_ncaa_reference",
                )
                if qm_freq.hessian is None:
                    raise ValueError("QM opt-frequency job did not provide a Cartesian Hessian.")
                hessian = qm_freq.hessian
            else:
                hessian = get_cartesian_hessian(representative_atoms)
            apply_bonded(
                representative_atoms,
                hessian,
                stage0_result.bonds,
                stage0_result.angles,
                config.vib_scale,
            )

    if torsion_enabled:
        with _timed_stage(stage_timings, "TorsionFit"):
            torsion_kwargs = {
                "atoms": representative_atoms,
                "output": output,
                "parameter_set": stage0_result,
                "params": config.torsion,
                "runtime": TorsionScanRuntime(
                    max_iter=config.opt_max_iter,
                    memory=int(max(config.qm.qm_mem, 1)),
                    curvature=0.6,
                    max_step=config.opt_max_step,
                    backend=config.torsion.backend,
                    constraint_mode=config.torsion.constraint_mode,
                ),
                "center_bond_filter": build_ncaa_center_bond_filter(representative_model),
                "mobile_atoms": sidechain_relax_indices,
            }
            if qm_runner is not None:
                torsion_kwargs["qm_runner"] = qm_runner
            torsion = run_torsion_workflow(**torsion_kwargs)
    else:
        torsion = TorsionWorkflowResult(
            stage1_parameter_set=None,
            final_parameter_set=stage0_result,
        )
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
    qm_runner = build_qm_reference_runner(config.qm)
    log_info(["  [NCAA] MLIP model preparation + reference optimization ...\n"])
    with _timed_stage(stage_timings, "model preparation/reference optimization"):
        prepared = _prepare_ncaa_models(
            output=output,
            source_atoms=source_atoms,
            structure=structure,
            target_residue=target_residue,
            config=config,
            qm_runner=qm_runner,
            log_info=log_info,
        )
    log_info(
        format_ncaa_start_lines(
            config=config,
            identity=prepared.identity,
        )
    )

    log_info([f"  [NCAA] atomic charge fitting ({config.charge_fit.method}) ...\n"])
    resp_wfn = None
    if (
        config.charge_fit.method == "resp"
        and config.charge_fit.qm is not None
        and qm_runner is not None
        and config.qm.qm_engine in {"gaussian", "g16", "g09"}
    ):
        opt_theory, opt_basis = (part.strip() for part in config.qm.opt_level.strip().split("/", 1))
        if (
            config.charge_fit.qm.theory == opt_theory
            and config.charge_fit.qm.basis == opt_basis
        ):
            resp_wfn = getattr(qm_runner, "last_wfn_path", None)
    with _timed_stage(stage_timings, "charge fitting"):
        charge_result = fit_multiconformer_charges(
            output=output,
            conformers=[(conformer.label, conformer.model) for conformer in prepared.conformers],
            representative_model=prepared.representative.model,
            residue_key=prepared.identity.residue_key,
            bond_pairs=infer_bond_pairs(prepared.representative.model),
            total_charge=config.charge,
            multiplicity=config.mult,
            config=config.charge_fit,
            source_atoms=source_atoms,
            prom=config.prom,
            wfn_path=resp_wfn,
        )
    representative_model = apply_model_charges(
        prepared.representative.model,
        charge_result.charges,
    )
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
            charge_result=charge_result,
            config=config,
        )
    if config.bonded == "none":
        log_info(["  [NCAA] bond/angle skipped ...\n"])
    else:
        bonded_label = "Seminario" if config.bonded == "seminario" else "mSeminario"
        hessian_source = "QM Hessian" if qm_runner is not None else "MLIP Hessian"
        log_info([f"  [NCAA] {hessian_source} + {bonded_label} ...\n"])
    if config.torsion.enabled:
        log_info(["  [NCAA] TorsionFit ...\n"])
        if qm_runner is not None:
            qm_mode = int(getattr(config.qm, "qm_mode", 2))
            log_info([f"  [NCAA] Using QM reference data for TorsionFit (mode={qm_mode}) ...\n"])
    else:
        log_info(["  [NCAA] TorsionFit skipped ...\n"])
    parameter_set, torsion = _refine_ncaa_parameters(
        output=output,
        source_atoms=source_atoms,
        representative_model=representative_model,
        typed_mol2_path=amber_artifacts.gaff2_mol2,
        frcmod_path=amber_artifacts.frcmod,
        config=config,
        sidechain_relax_indices=prepared.sidechain_relax_indices,
        stage_timings=stage_timings,
        qm_hessian=prepared.qm_hessian,
        qm_runner=qm_runner,
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
        interface.run_tleap(
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
        charge_result=charge_result,
        representative_model=representative_model,
        residue_model=residue_model,
        stage_timings=list(stage_timings),
    )
