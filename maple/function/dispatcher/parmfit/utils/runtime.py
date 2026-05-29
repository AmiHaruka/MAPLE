"""Usage: manage parmfit workdirs, optimization, Hessian, and RESP runtime calls."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
import shutil
import subprocess as sp
from typing import Optional

import numpy as np
from ase import Atoms

from . import interface as qm_interface
from . import resp as resp_utils
from .interface import QMMethod
from .Scan.optimizer import LBFGS, LBFGSParams


MEDIUM_THRESHOLDS = {
    "f_max_th": 0.00285,
    "f_rms_th": 0.00190,
    "dp_max_th": 0.00315,
    "dp_rms_th": 0.00210,
}

# =============================================================================
# Prepare workdir
# =============================================================================


def parmfit_workdir(output: str, workflow: str) -> str:
    path = os.path.join(parmfit_output_dir(output), workflow)
    os.makedirs(path, exist_ok=True)
    return path


def parmfit_output_dir(output: str) -> str:
    base = os.path.splitext(os.path.abspath(output))[0]
    path = f"{base}_work"
    os.makedirs(path, exist_ok=True)
    return path


def parmfit_work_prefix(output: str, workflow: str) -> str:
    base_name = os.path.splitext(os.path.basename(output))[0]
    return os.path.join(parmfit_workdir(output, workflow), base_name)

# =============================================================================
# MAPLE generic runtime utilities
# =============================================================================

def to_f64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)


def get_forces(atoms: Atoms) -> np.ndarray:
    if hasattr(atoms, "get_forces"):
        try:
            return np.asarray(atoms.get_forces(), dtype=float)
        except AttributeError:
            pass
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "get_forces"):
        return np.asarray(atoms.calc.get_forces(atoms), dtype=float)
    raise ValueError("Silent runtime requires force evaluation from atoms or atoms.calc.")


def get_potential_energy(atoms: Atoms, force_consistent: bool = True) -> float:
    if hasattr(atoms, "get_potential_energy"):
        try:
            return float(atoms.get_potential_energy(force_consistent=force_consistent))
        except AttributeError:
            pass
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "get_potential_energy"):
        return float(atoms.calc.get_potential_energy(atoms, force_consistent=force_consistent))
    raise ValueError("Silent runtime requires energy evaluation from atoms or atoms.calc.")


def get_cartesian_hessian(atoms: Atoms) -> np.ndarray:
    hessian = to_f64(atoms.calc.get_hessian(atoms))
    if hessian.ndim == 3 and hessian.shape[0] == 1:
        hessian = hessian[0]
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"Hessian must be square 2D, got shape {hessian.shape}")
    return hessian


def copy_thresholds(source_atoms, target_atoms) -> None:
    for attr, default in MEDIUM_THRESHOLDS.items():
        setattr(target_atoms, attr, float(getattr(source_atoms, attr, default)))


def silent_lbfgs_params(params: LBFGSParams | None = None) -> LBFGSParams:
    resolved = deepcopy(params) if params is not None else LBFGSParams()
    resolved.write_traj = False
    resolved.verbose = 0
    resolved.use_projection = False
    resolved.use_line_search = False
    return resolved


SilentLBFGS = LBFGS


def _require_lbfgs_thresholds(atoms: Atoms) -> None:
    missing = [
        attr
        for attr in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
        if not hasattr(atoms, attr)
    ]
    if missing:
        raise ValueError(f"LBFGS thresholds are missing on atoms: {', '.join(missing)}.")


def run_silent_lbfgs(
    atoms: Atoms,
    *,
    output: str,
    params: LBFGSParams | None = None,
) -> LBFGS:
    _require_lbfgs_thresholds(atoms)
    optimizer = LBFGS(atoms=atoms, output=output, params=silent_lbfgs_params(params))
    optimizer.run()
    return optimizer


def optimize_atoms_geometry(
    atoms: Atoms,
    *,
    output: str,
    max_iter: int = 256,
    max_step: float = 0.2,
    failure_message: str | None = None,
) -> Atoms:
    params = silent_lbfgs_params()
    params.max_iter = int(max_iter)
    params.max_step = float(max_step)
    optimizer = run_silent_lbfgs(atoms, output=output, params=params)
    if not optimizer.converged:
        raise RuntimeError(failure_message or "Geometry optimization did not converge.")
    return atoms


def optimize_model_geometry(
    model: dict,
    *,
    output: str,
    source_atoms: Atoms,
    calculator=None,
    max_iter: int = 256,
    max_step: float = 0.2,
    failure_message: str | None = None,
) -> dict:
    from .model import model_to_atoms, update_model_from_atoms

    atoms = model_to_atoms(model, charge=model.get("charge"), mult=model.get("mult"))
    copy_thresholds(source_atoms, atoms)
    atoms.calc = source_atoms.calc if calculator is None else calculator
    optimize_atoms_geometry(
        atoms,
        output=output,
        max_iter=max_iter,
        max_step=max_step,
        failure_message=failure_message or f"Geometry optimization did not converge for {model.get('name', 'model')}.",
    )
    return update_model_from_atoms(model, atoms)



# =============================================================================
# Resp charge fitting pipeline utilities
# =============================================================================

@dataclass
class RespPipelineResult:
    model: dict
    files: dict[str, str]
    resp_files: dict[str, str]
    decision: QMMethod


@dataclass
class MultiRespPipelineResult:
    model: dict
    files: dict[str, str]
    resp_files: dict[str, str]
    conformers: dict[str, dict[str, object]]
    decision: QMMethod


def _resolve_external_binary(name: str, candidates: Optional[tuple[str, ...]] = None) -> str:
    for candidate in candidates or (name,):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    expected = ", ".join(candidates or (name,))
    raise RuntimeError(f"Required external program {name!r} was not found. Expected one of: {expected}.")


def _run_external_command(args: list[str], *, cwd: str) -> None:
    result = sp.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"return code {result.returncode}"
        raise RuntimeError(f"External command failed: {' '.join(args)}\n{detail}")


def _resp_paths(output: str, *, label: str = "metal_site_resp") -> dict[str, str]:
    workdir = parmfit_workdir(output, "metalaa")
    work_prefix = parmfit_work_prefix(output, "metalaa")
    return {
        "workdir": workdir,
        "gaussian_input": f"{work_prefix}_{label}.gjf",
        "gaussian_log": f"{work_prefix}_{label}.log",
        "esp": os.path.join(workdir, f"{label}.esp"),
        "resp1_in": os.path.join(workdir, "resp1.in"),
        "resp1_out": os.path.join(workdir, "resp1.out"),
        "resp1_pch": os.path.join(workdir, "resp1.pch"),
        "resp1_chg": os.path.join(workdir, "resp1.chg"),
        "resp1_calc_esp": os.path.join(workdir, "resp1_calc.esp"),
        "resp2_in": os.path.join(workdir, "resp2.in"),
        "resp2_out": os.path.join(workdir, "resp2.out"),
        "resp2_pch": os.path.join(workdir, "resp2.pch"),
        "resp2_chg": os.path.join(workdir, "resp2.chg"),
        "resp2_calc_esp": os.path.join(workdir, "resp2_calc.esp"),
        "mol2": os.path.join(workdir, f"{label}.mol2"),
    }


def run_espgen(log_file: str, esp_file: str) -> None:
    espgen_cmd = _resolve_external_binary("espgen")
    _run_external_command([espgen_cmd, "-i", log_file, "-o", esp_file], cwd=os.path.dirname(os.path.abspath(log_file)))
    if not os.path.isfile(esp_file):
        raise FileNotFoundError(f"espgen did not create the expected ESP file: {esp_file}")


def run_resp_stage(
    *,
    workdir: str,
    input_path: str,
    output_path: str,
    punch_path: str,
    charge_path: str,
    esp_path: str,
    calc_esp_path: str,
    qin_path: Optional[str] = None,
) -> None:
    resp_cmd = _resolve_external_binary("resp")
    args = [
        resp_cmd,
        "-O",
        "-i",
        os.path.basename(input_path),
        "-o",
        os.path.basename(output_path),
        "-p",
        os.path.basename(punch_path),
        "-t",
        os.path.basename(charge_path),
        "-e",
        os.path.basename(esp_path),
        "-s",
        os.path.basename(calc_esp_path),
    ]
    if qin_path is not None:
        args.extend(["-q", os.path.basename(qin_path)])
    _run_external_command(args, cwd=workdir)
    for required in (output_path, punch_path, charge_path):
        if not os.path.isfile(required):
            raise FileNotFoundError(f"RESP did not create the expected file: {required}")


def run_resp_pipeline(
    *,
    output: str,
    model: dict,
    bond_pairs: list[tuple[int, int]],
    total_charge: int,
    multiplicity: int,
    chgmod: int,
    qm: QMMethod,
    fixchg_resids: list[str] | None = None,
    label: str = "metal_site_resp",
    watm: str | None = None,
    prom: str = "ff14SB",
    charge_groups: list[tuple[list[int], float]] | None = None,
) -> RespPipelineResult:
    paths = _resp_paths(output, label=label)
    os.makedirs(paths["workdir"], exist_ok=True)

    qm_interface.prepare_gaussian_esp_input(
        paths["gaussian_input"],
        model,
        total_charge=total_charge,
        multiplicity=multiplicity,
        decision=qm,
        watm=watm,
    )
    gaussian_log = qm_interface.run_gaussian(paths["gaussian_input"], qm)
    paths["gaussian_log"] = gaussian_log

    run_espgen(paths["gaussian_log"], paths["esp"])
    resp_inputs = resp_utils.write_resp_input_files(
        paths["workdir"],
        model,
        total_charge=total_charge,
        chgmod=chgmod,
        fixchg_resids=fixchg_resids,
        charge_groups=charge_groups,
        prom=prom,
    )
    paths["resp1_in"] = resp_inputs.resp1_in
    paths["resp2_in"] = resp_inputs.resp2_in

    run_resp_stage(
        workdir=paths["workdir"],
        input_path=paths["resp1_in"],
        output_path=paths["resp1_out"],
        punch_path=paths["resp1_pch"],
        charge_path=paths["resp1_chg"],
        esp_path=paths["esp"],
        calc_esp_path=paths["resp1_calc_esp"],
    )
    run_resp_stage(
        workdir=paths["workdir"],
        input_path=paths["resp2_in"],
        output_path=paths["resp2_out"],
        punch_path=paths["resp2_pch"],
        charge_path=paths["resp2_chg"],
        esp_path=paths["esp"],
        calc_esp_path=paths["resp2_calc_esp"],
        qin_path=paths["resp1_chg"],
    )

    charges = resp_utils.read_resp_charges(paths["resp2_chg"])
    charged_model = resp_utils.apply_resp_charges(model, charges)
    resp_utils.write_resp_mol2(paths["mol2"], charged_model, bond_pairs, prom=prom)

    return RespPipelineResult(
        model=charged_model,
        files={
            "gaussian_input": paths["gaussian_input"],
            "mol2": paths["mol2"],
        },
        resp_files={
            "gaussian_log": paths["gaussian_log"],
            "esp": paths["esp"],
            "resp1_in": paths["resp1_in"],
            "resp1_out": paths["resp1_out"],
            "resp1_pch": paths["resp1_pch"],
            "resp1_chg": paths["resp1_chg"],
            "resp1_calc_esp": paths["resp1_calc_esp"],
            "resp2_in": paths["resp2_in"],
            "resp2_out": paths["resp2_out"],
            "resp2_pch": paths["resp2_pch"],
            "resp2_chg": paths["resp2_chg"],
            "resp2_calc_esp": paths["resp2_calc_esp"],
        },
        decision=qm,
    )


def _multiconformer_resp_paths(output: str, labels: list[str]) -> dict[str, str]:
    base = os.path.splitext(output)[0]
    base_name = os.path.basename(base)
    workdir = parmfit_workdir(output, "ncaa")
    work_prefix = parmfit_work_prefix(output, "ncaa")
    paths = {
        "workdir": workdir,
        "all_esp": os.path.join(workdir, f"{base_name}_all.esp"),
        "resp1_in": os.path.join(workdir, f"{base_name}_resp1.in"),
        "resp1_out": os.path.join(workdir, f"{base_name}_resp1.out"),
        "resp1_pch": os.path.join(workdir, f"{base_name}_resp1.pch"),
        "resp1_chg": os.path.join(workdir, f"{base_name}_resp1.chg"),
        "resp1_calc_esp": os.path.join(workdir, f"{base_name}_resp1_calc.esp"),
        "resp2_in": os.path.join(workdir, f"{base_name}_resp2.in"),
        "resp2_out": os.path.join(workdir, f"{base_name}_resp2.out"),
        "resp2_pch": os.path.join(workdir, f"{base_name}_resp2.pch"),
        "resp2_chg": os.path.join(workdir, f"{base_name}_resp2.chg"),
        "resp2_calc_esp": os.path.join(workdir, f"{base_name}_resp2_calc.esp"),
        "target_chg": os.path.join(workdir, f"{base_name}_target.chg"),
        "mol2": os.path.join(workdir, f"{base_name}_capped.mol2"),
    }
    for label in labels:
        paths[f"{label}_gaussian_input"] = f"{work_prefix}_{label}_resp.gjf"
        paths[f"{label}_esp"] = os.path.join(workdir, f"{base_name}_{label}.esp")
    return paths


def run_multiconformer_resp(
    *,
    output: str,
    conformers: list[tuple[str, dict]],
    representative_model: dict,
    residue_key: tuple[str, int, str],
    bond_pairs: list[tuple[int, int]],
    total_charge: int,
    multiplicity: int,
    qm: QMMethod,
    prom: str = "ff14SB",
) -> MultiRespPipelineResult:
    if not conformers:
        raise ValueError("Multiconformer RESP requires at least one conformer.")

    labels = [label for label, _ in conformers]
    paths = _multiconformer_resp_paths(output, labels)
    os.makedirs(paths["workdir"], exist_ok=True)

    conformer_outputs: dict[str, dict[str, object]] = {}
    esp_files: list[str] = []
    models = [model for _, model in conformers]
    for label, model in conformers:
        gaussian_input = paths[f"{label}_gaussian_input"]
        qm_interface.prepare_gaussian_esp_input(
            gaussian_input,
            model,
            total_charge=total_charge,
            multiplicity=multiplicity,
            decision=qm,
            title=f"MAPLE NCAA {label} RESP",
        )
        gaussian_log = qm_interface.run_gaussian(gaussian_input, qm)
        esp_path = paths[f"{label}_esp"]
        run_espgen(gaussian_log, esp_path)
        conformer_outputs[label] = {
            "gaussian_input": gaussian_input,
            "gaussian_log": gaussian_log,
            "esp": esp_path,
        }
        esp_files.append(esp_path)

    resp_utils.merge_esp_files(esp_files, paths["all_esp"])
    resp_inputs = resp_utils.write_multiconformer_resp_input_files(
        paths["workdir"],
        models,
        labels=labels,
        total_charge=total_charge,
        residue_key=residue_key,
    )
    paths["resp1_in"] = resp_inputs.resp1_in
    paths["resp2_in"] = resp_inputs.resp2_in

    run_resp_stage(
        workdir=paths["workdir"],
        input_path=paths["resp1_in"],
        output_path=paths["resp1_out"],
        punch_path=paths["resp1_pch"],
        charge_path=paths["resp1_chg"],
        esp_path=paths["all_esp"],
        calc_esp_path=paths["resp1_calc_esp"],
    )
    run_resp_stage(
        workdir=paths["workdir"],
        input_path=paths["resp2_in"],
        output_path=paths["resp2_out"],
        punch_path=paths["resp2_pch"],
        charge_path=paths["resp2_chg"],
        esp_path=paths["all_esp"],
        calc_esp_path=paths["resp2_calc_esp"],
        qin_path=paths["resp1_chg"],
    )

    charges = resp_utils.read_resp_charges(paths["resp2_chg"])
    n_atoms = sum(len(residue["atoms"]) for residue in representative_model["residues"])
    if len(charges) == n_atoms:
        reference_charges = charges
    elif len(charges) % n_atoms == 0:
        reference_charges = charges[:n_atoms]
    else:
        raise ValueError(
            f"RESP returned {len(charges)} charges for a representative model with {n_atoms} atoms."
        )
    with open(paths["target_chg"], "w", encoding="utf-8") as handle:
        handle.write(" ".join(f"{charge:.10f}" for charge in reference_charges))
        handle.write("\n")
    charged_model = resp_utils.apply_resp_charges(representative_model, reference_charges)
    resp_utils.write_resp_mol2(paths["mol2"], charged_model, bond_pairs, prom=prom)

    return MultiRespPipelineResult(
        model=charged_model,
        files={"mol2": paths["mol2"]},
        resp_files={
            "all_esp": paths["all_esp"],
            "resp1_in": paths["resp1_in"],
            "resp1_out": paths["resp1_out"],
            "resp1_pch": paths["resp1_pch"],
            "resp1_chg": paths["resp1_chg"],
            "resp1_calc_esp": paths["resp1_calc_esp"],
            "resp2_in": paths["resp2_in"],
            "resp2_out": paths["resp2_out"],
            "resp2_pch": paths["resp2_pch"],
            "resp2_chg": paths["resp2_chg"],
            "resp2_calc_esp": paths["resp2_calc_esp"],
            "target_chg": paths["target_chg"],
            **{
                f"{label}_gaussian_input": data["gaussian_input"]
                for label, data in conformer_outputs.items()
            },
            **{
                f"{label}_gaussian_log": data["gaussian_log"]
                for label, data in conformer_outputs.items()
            },
            **{
                f"{label}_esp": data["esp"]
                for label, data in conformer_outputs.items()
            },
        },
        conformers=conformer_outputs,
        decision=qm,
    )
