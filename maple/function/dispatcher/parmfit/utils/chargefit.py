"""Shared atomic charge-source adapters for ParmFit workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import os
from pathlib import Path

import numpy as np

from . import interface, resp, runtime
from .interface import QMMethod
from .mlip_tools import (
    calculator_atomic_charges,
    release_charge_calculator_cache,
    resolve_charge_calculator,
    resolve_charge_model_class,
)
from .model import flatten_model_atoms, model_to_atoms
from .mol2_tools import write_updated_mol2
from .readparm import CorrectionParameterSet, parse_mol2
from .runtime import parmfit_workdir, run_espgen, run_resp_stage
from .structure import get_resid_key


@dataclass
class ChargeFitConfig:
    method: str = "none"
    level: str = "HF/6-31G(d)"
    route: str = ""
    qm: QMMethod | None = None


@dataclass(frozen=True)
class ChargeFitResult:
    method: str
    charges: np.ndarray
    target_charge: int
    actual_charge: float
    work_mol2: str
    detail: str = ""
    stderr: str = ""
    files: dict[str, str] = field(default_factory=dict)


def build_charge_fit_config(
    raw: dict,
    *,
    default_method: str,
    default_level: str,
    default_nproc: int,
    default_mem: int,
) -> ChargeFitConfig:
    method_value = str(raw.get("chg_fit", default_method)).strip()
    method_key = method_value.lower()
    method = method_key if method_key in {"none", "resp"} else method_value
    config = ChargeFitConfig(
        method=method,
        level=str(raw.get("chg_level", default_level)).strip(),
        route=str(raw.get("chg_route", "")).strip(),
    )
    if method != "resp":
        return config

    backend = str(raw.get("resp_backend", "gaussian")).strip().lower()
    if backend != "gaussian":
        raise ValueError(
            f"Unsupported RESP backend {backend!r}; expected one of 'gaussian'."
        )
    if "/" not in config.level:
        raise ValueError(
            f"RESP level {config.level!r} must use METHOD/BASIS syntax."
        )
    theory, basis = (part.strip() for part in config.level.split("/", 1))
    if not theory or not basis:
        raise ValueError(
            f"RESP level {config.level!r} must use METHOD/BASIS syntax."
        )
    config.qm = interface.set_method(
        {
            "theory": theory,
            "basis": basis,
            "route": config.route,
            "nproc": raw.get("qm_nproc", default_nproc),
            "mem": raw.get("qm_mem", default_mem),
        },
        backend="gaussian",
    )
    return config


def validated_charges(charges, atom_count: int) -> np.ndarray:
    values = np.asarray(charges, dtype=float).reshape(-1)
    if values.size != atom_count:
        raise ValueError(
            f"Atomic charges contain {values.size} values for {atom_count} atoms."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Atomic charges must be finite.")
    return values


def apply_atomic_charges(
    parameter_set: CorrectionParameterSet,
    charges,
) -> CorrectionParameterSet:
    """Return a parameter-set copy carrying the fitted atomic charges."""
    values = validated_charges(charges, len(parameter_set.mol2.atoms))
    updated = deepcopy(parameter_set)
    updated.mol2.atoms = [
        replace(atom, charge=float(values[index]))
        for index, atom in enumerate(updated.mol2.atoms)
    ]
    for nonbond in updated.nonbonds:
        nonbond.charge = float(values[nonbond.atom - 1])
    return updated


def build_resp_model(
    atoms,
    mol2_path: str,
    *,
    total_charge: int,
    multiplicity: int,
) -> tuple[dict, list[tuple[int, int]]]:
    """Build the one-residue model expected by the RESP writers."""
    topology = parse_mol2(mol2_path)
    if len(topology.atoms) != len(atoms):
        raise ValueError(
            f"MOL2 contains {len(topology.atoms)} atoms but the charge target has {len(atoms)}."
        )
    positions = np.asarray(atoms.get_positions(), dtype=float)
    symbols = atoms.get_chemical_symbols()
    model_atoms = []
    for index, (mol2_atom, symbol, xyz) in enumerate(
        zip(topology.atoms, symbols, positions, strict=True),
        start=1,
    ):
        model_atoms.append(
            {
                "serial": index,
                "name": mol2_atom.name,
                "element": str(symbol),
                "xyz": np.asarray(xyz, dtype=float),
                "charge": mol2_atom.charge,
                "atom_type": mol2_atom.atom_type,
            }
        )
    residue = {
        "resname": "MOL",
        "chain": "_",
        "resseq": 1,
        "icode": "",
        "kind": "ligand",
        "atoms": model_atoms,
        "coords": positions.copy(),
    }
    bond_pairs = [
        (topology.id_to_index[bond.atom1], topology.id_to_index[bond.atom2])
        for bond in topology.bonds
    ]
    return (
        {
            "name": Path(mol2_path).stem,
            "charge": int(total_charge),
            "mult": int(multiplicity),
            "residues": [residue],
        },
        bond_pairs,
    )


def _resp_paths(workdir: str) -> dict[str, str]:
    root = Path(workdir)
    return {
        "gaussian_input": str(root / "resp.gjf"),
        "esp": str(root / "resp.esp"),
        "resp1_out": str(root / "resp1.out"),
        "resp1_pch": str(root / "resp1.pch"),
        "resp1_chg": str(root / "resp1.chg"),
        "resp1_calc_esp": str(root / "resp1_calc.esp"),
        "resp2_out": str(root / "resp2.out"),
        "resp2_pch": str(root / "resp2.pch"),
        "resp2_chg": str(root / "resp2.chg"),
        "resp2_calc_esp": str(root / "resp2_calc.esp"),
    }


def _fit_resp_charges(
    atoms,
    mol2_path: str,
    workdir: str,
    *,
    total_charge: int,
    multiplicity: int,
    config: ChargeFitConfig,
) -> tuple[np.ndarray, dict[str, str]]:
    if config.qm is None:
        raise ValueError("RESP charge fitting requires a Gaussian QM configuration.")
    model, bond_pairs = build_resp_model(
        atoms,
        mol2_path,
        total_charge=total_charge,
        multiplicity=multiplicity,
    )
    paths = _resp_paths(workdir)
    interface.prepare_gaussian_esp_input(
        paths["gaussian_input"],
        model,
        total_charge=total_charge,
        multiplicity=multiplicity,
        decision=config.qm,
        title="MAPLE Correction RESP",
    )
    paths["gaussian_log"] = interface.run_gaussian(
        paths["gaussian_input"],
        config.qm,
    )
    run_espgen(paths["gaussian_log"], paths["esp"])
    inputs = resp.write_resp_input_files(
        workdir,
        model,
        total_charge=total_charge,
        chgmod=0,
        bond_pairs=bond_pairs,
    )
    paths["resp1_in"] = inputs.resp1_in
    paths["resp2_in"] = inputs.resp2_in
    run_resp_stage(
        workdir=workdir,
        input_path=paths["resp1_in"],
        output_path=paths["resp1_out"],
        punch_path=paths["resp1_pch"],
        charge_path=paths["resp1_chg"],
        esp_path=paths["esp"],
        calc_esp_path=paths["resp1_calc_esp"],
    )
    run_resp_stage(
        workdir=workdir,
        input_path=paths["resp2_in"],
        output_path=paths["resp2_out"],
        punch_path=paths["resp2_pch"],
        charge_path=paths["resp2_chg"],
        esp_path=paths["esp"],
        calc_esp_path=paths["resp2_calc_esp"],
        qin_path=paths["resp1_chg"],
    )
    charges = validated_charges(
        resp.read_resp_charges(paths["resp2_chg"]),
        len(atoms),
    )
    return charges, paths


def fit_molecule_charges(
    *,
    output: str,
    atoms,
    source_mol2: str,
    config: ChargeFitConfig,
    route: str,
    calculator_cache: dict | None = None,
) -> ChargeFitResult:
    """Fit one molecule and materialize its token-preserving charged MOL2."""
    target_charge = int(atoms.info.get("charge", 0))
    multiplicity = int(atoms.info.get("mult", 1))
    raw_method = str(config.method).strip()
    method_key = raw_method.lower()
    geometry_mol2 = source_mol2
    files: dict[str, str] = {"input_mol2": source_mol2}
    stderr = ""

    if method_key == "none":
        topology = parse_mol2(source_mol2)
        charges = np.asarray([atom.charge for atom in topology.atoms], dtype=float)
        method = "input"
        detail = "none"
    else:
        # parmfit_workdir creates the directory, so only ask for it on the paths
        # that actually write into it.
        workdir = parmfit_workdir(output, "chargefit")
        geometry_mol2 = write_updated_mol2(
            source_mol2,
            os.path.join(workdir, f"{route}_input.mol2"),
            positions=atoms.get_positions(),
        )
        files["input_mol2"] = geometry_mol2

    if method_key == "resp":
        charges, resp_files = _fit_resp_charges(
            atoms,
            geometry_mol2,
            workdir,
            total_charge=target_charge,
            multiplicity=multiplicity,
            config=config,
        )
        files.update(resp_files)
        method = "resp"
        detail = config.level
    elif method_key != "none":
        try:
            active_calculator = getattr(atoms, "calc", None)
            canonical, calculator_class = resolve_charge_model_class(
                raw_method,
                device=getattr(active_calculator, "device", None),
                output=output,
            )
        except ValueError:
            charges, antechamber_mol2, stderr = runtime.run_antechamber_charge_method(
                geometry_mol2,
                workdir,
                method=raw_method,
                total_charge=target_charge,
                multiplicity=multiplicity,
            )
            files["antechamber_mol2"] = antechamber_mol2
            method = "antechamber"
            detail = raw_method
        else:
            if "charges" not in tuple(
                getattr(calculator_class, "implemented_properties", ())
            ):
                raise ValueError(
                    f"MAPLE model {canonical!r} does not provide the ASE 'charges' property."
                )
            if (
                target_charge != 0 or multiplicity != 1
            ) and not getattr(calculator_class, "SUPPORTS_CHARGE_MULT", False):
                raise ValueError(
                    f"MAPLE model {canonical!r} cannot evaluate charges for "
                    f"charge={target_charge}, multiplicity={multiplicity}."
                )
            owns_cache = calculator_cache is None
            cache = {} if owns_cache else calculator_cache
            calculator = cache.get(canonical)
            if calculator is None:
                calculator, canonical = resolve_charge_calculator(
                    canonical,
                    atoms=atoms,
                    output=output,
                )
                cache[canonical] = calculator
            try:
                charges = calculator_atomic_charges(
                    calculator,
                    atoms,
                    model_name=canonical,
                )
            finally:
                calculator = None
                if owns_cache:
                    release_charge_calculator_cache(
                        cache,
                        active_calculators=(active_calculator,),
                    )
            method = "model"
            detail = canonical

    charges = validated_charges(charges, len(atoms))
    if method_key == "none":
        # Nothing was fitted, and the exporters take the geometry from `atoms`,
        # so a rewritten copy of the input would carry no information.
        charged_mol2 = source_mol2
    else:
        charged_mol2 = write_updated_mol2(
            source_mol2,
            os.path.join(workdir, f"{route}_charged.mol2"),
            positions=atoms.get_positions(),
            charges=charges,
        )
        files["charged_mol2"] = charged_mol2
    return ChargeFitResult(
        method=method,
        charges=charges,
        target_charge=target_charge,
        actual_charge=float(charges.sum()),
        work_mol2=charged_mol2,
        detail=detail,
        stderr=stderr,
        files=files,
    )


def apply_model_charges(model: dict, charges) -> dict:
    values = validated_charges(charges, len(flatten_model_atoms(model)))
    updated = deepcopy(model)
    charge_index = 0
    for residue in updated["residues"]:
        for atom in sorted(residue["atoms"], key=lambda item: item["serial"]):
            atom["charge"] = float(values[charge_index])
            charge_index += 1
    return updated


def _write_multiconformer_charge_outputs(
    *,
    output: str,
    representative_model: dict,
    bond_pairs: list[tuple[int, int]],
    charges: np.ndarray,
    prom: str,
) -> tuple[str, str]:
    workdir = parmfit_workdir(output, "ncaa")
    base_name = Path(output).stem
    target_chg = os.path.join(workdir, f"{base_name}_target.chg")
    work_mol2 = os.path.join(workdir, f"{base_name}_capped.mol2")
    with open(target_chg, "w", encoding="utf-8") as handle:
        handle.write(" ".join(f"{charge:.10f}" for charge in charges))
        handle.write("\n")
    charged_model = apply_model_charges(representative_model, charges)
    resp.write_resp_mol2(
        work_mol2,
        charged_model,
        bond_pairs,
        prom=prom,
    )
    return os.path.abspath(work_mol2), os.path.abspath(target_chg)


def fit_multiconformer_charges(
    *,
    output: str,
    conformers: list[tuple[str, dict]],
    representative_model: dict,
    residue_key: tuple[str, int, str],
    bond_pairs: list[tuple[int, int]],
    total_charge: int,
    multiplicity: int,
    config: ChargeFitConfig,
    source_atoms,
    prom: str,
    wfn_path: str | None = None,
) -> ChargeFitResult:
    """Fit conformer charges and apply their per-atom mean to a representative."""
    representative_atoms = flatten_model_atoms(representative_model)
    atom_count = len(representative_atoms)
    residue_indices = [
        index
        for index, (residue, _atom) in enumerate(representative_atoms)
        if get_resid_key(residue) == residue_key
    ]
    raw_method = str(config.method).strip()
    method_key = raw_method.lower()
    files: dict[str, str] = {}
    stderr_lines: list[str] = []

    if method_key == "resp":
        if config.qm is None:
            raise ValueError("RESP charge fitting requires a Gaussian QM configuration.")
        resp_result = runtime.run_multiconformer_resp(
            output=output,
            conformers=conformers,
            representative_model=representative_model,
            residue_key=residue_key,
            bond_pairs=bond_pairs,
            total_charge=total_charge,
            multiplicity=multiplicity,
            qm=config.qm,
            prom=prom,
            wfn_path=wfn_path,
        )
        charges = validated_charges(
            [
                atom["charge"]
                for _residue, atom in flatten_model_atoms(resp_result.model)
            ],
            atom_count,
        )
        files.update(resp_result.files)
        files.update(resp_result.resp_files)
        return ChargeFitResult(
            method="resp",
            charges=charges,
            target_charge=int(total_charge),
            actual_charge=float(charges[residue_indices].sum()),
            work_mol2=resp_result.files["mol2"],
            detail=config.level,
            files=files,
        )

    active_calculator = getattr(source_atoms, "calc", None)
    try:
        canonical, calculator_class = resolve_charge_model_class(
            raw_method,
            device=getattr(active_calculator, "device", None),
            output=output,
        )
    except ValueError:
        charge_sets = []
        for label, model in conformers:
            workdir = parmfit_workdir(
                output,
                os.path.join("ncaa", str(label)),
            )
            input_mol2 = os.path.join(workdir, "input.mol2")
            resp.write_resp_mol2(
                input_mol2,
                model,
                bond_pairs,
                prom=prom,
            )
            values, charged_mol2, stderr = runtime.run_antechamber_charge_method(
                input_mol2,
                workdir,
                method=raw_method,
                total_charge=total_charge,
                multiplicity=multiplicity,
            )
            charge_sets.append(validated_charges(values, atom_count))
            files[f"{label}_input_mol2"] = input_mol2
            files[f"{label}_antechamber_mol2"] = charged_mol2
            if stderr.strip():
                stderr_lines.append(stderr.strip())
        method = "antechamber"
        detail = raw_method
    else:
        if "charges" not in tuple(
            getattr(calculator_class, "implemented_properties", ())
        ):
            raise ValueError(
                f"MAPLE model {canonical!r} does not provide the ASE 'charges' property."
            )
        if (
            total_charge != 0 or multiplicity != 1
        ) and not getattr(calculator_class, "SUPPORTS_CHARGE_MULT", False):
            raise ValueError(
                f"MAPLE model {canonical!r} cannot evaluate charges for "
                f"charge={total_charge}, multiplicity={multiplicity}."
            )
        cache: dict = {}
        charge_sets = []
        try:
            calculator = None
            for _label, model in conformers:
                atoms = model_to_atoms(
                    model,
                    charge=total_charge,
                    mult=multiplicity,
                )
                atoms.calc = active_calculator
                if calculator is None:
                    calculator, canonical = resolve_charge_calculator(
                        canonical,
                        atoms=atoms,
                        output=output,
                    )
                    cache[canonical] = calculator
                charge_sets.append(
                    calculator_atomic_charges(
                        calculator,
                        atoms,
                        model_name=canonical,
                    )
                )
        finally:
            calculator = None
            release_charge_calculator_cache(
                cache,
                active_calculators=(active_calculator,),
            )
        method = "model"
        detail = canonical

    charges = np.mean(np.stack(charge_sets, axis=0), axis=0)
    charges = validated_charges(charges, atom_count)
    mainchain_indices = [
        index
        for index in residue_indices
        if representative_atoms[index][1]["name"] in {"N", "CA", "C"}
    ]
    charge_delta = float(total_charge) - float(charges[residue_indices].sum())
    if abs(charge_delta) > 1.0e-4:
        charges[mainchain_indices] += charge_delta / len(mainchain_indices)
    work_mol2, target_chg = _write_multiconformer_charge_outputs(
        output=output,
        representative_model=representative_model,
        bond_pairs=bond_pairs,
        charges=charges,
        prom=prom,
    )
    files["mol2"] = work_mol2
    files["target_chg"] = target_chg
    return ChargeFitResult(
        method=method,
        charges=charges,
        target_charge=int(total_charge),
        actual_charge=float(charges[residue_indices].sum()),
        work_mol2=work_mol2,
        detail=detail,
        stderr="\n".join(stderr_lines),
        files=files,
    )
