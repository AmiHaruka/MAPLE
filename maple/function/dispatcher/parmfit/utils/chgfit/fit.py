"""Usage: fit atomic charges: RESP (Gaussian ESP), MAPLE models, and antechamber methods.

The whole charge-fitting subsystem lives here: the RESP stage runner, the
multiconformer pipeline, the antechamber/mlip dispatch, and the RESP input
formats with the reference-charge library.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import os
import shutil
import subprocess as sp

import numpy as np

from .. import interface
from ..amber_templates import load_amber_template_registry
from ..context import find_unique_residue
from ..interface import QMMethod
from ..mlip_tools import (
    calculator_atomic_charges,
    release_charge_calculator_cache,
    resolve_charge_calculator,
    resolve_charge_model_class,
)
from ..model import flatten_model_atoms, infer_bond_pairs, model_to_atoms
from ..mol2_tools import write_updated_mol2
from ..readparm import CorrectionParameterSet, parse_mol2
from ..runtime import parmfit_workdir, parmfit_work_prefix
from ..structure import get_resid_key
from typing import Optional


RESP_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
    "X": 0,
    "HE": 2,
    "He": 2,
    "LI": 3,
    "Li": 3,
    "BE": 4,
    "Be": 4,
    "B": 5,
    "NE": 10,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "AL": 13,
    "Al": 13,
    "SI": 14,
    "Si": 14,
    "Cl": 17,
    "AR": 18,
    "Ar": 18,
    "Ca": 20,
    "SC": 21,
    "Sc": 21,
    "TI": 22,
    "Ti": 22,
    "V": 23,
    "CR": 24,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "GA": 31,
    "Ga": 31,
    "GE": 32,
    "Ge": 32,
    "AS": 33,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "KR": 36,
    "Kr": 36,
    "RB": 37,
    "Rb": 37,
    "SR": 38,
    "Sr": 38,
    "Y": 39,
    "ZR": 40,
    "Zr": 40,
    "NB": 41,
    "Nb": 41,
    "MO": 42,
    "Mo": 42,
    "TC": 43,
    "Tc": 43,
    "RU": 44,
    "Ru": 44,
    "RH": 45,
    "Rh": 45,
    "PD": 46,
    "Pd": 46,
    "AG": 47,
    "Ag": 47,
    "CD": 48,
    "Cd": 48,
    "IN": 49,
    "In": 49,
    "SN": 50,
    "Sn": 50,
    "SB": 51,
    "Sb": 51,
    "TE": 52,
    "Te": 52,
    "XE": 54,
    "Xe": 54,
    "CS": 55,
    "Cs": 55,
    "BA": 56,
    "Ba": 56,
    "LA": 57,
    "La": 57,
    "CE": 58,
    "Ce": 58,
    "PR": 59,
    "Pr": 59,
    "ND": 60,
    "Nd": 60,
    "PM": 61,
    "Pm": 61,
    "SM": 62,
    "Sm": 62,
    "EU": 63,
    "Eu": 63,
    "GD": 64,
    "Gd": 64,
    "TB": 65,
    "Tb": 65,
    "DY": 66,
    "Dy": 66,
    "HO": 67,
    "Ho": 67,
    "ER": 68,
    "Er": 68,
    "TM": 69,
    "Tm": 69,
    "YB": 70,
    "Yb": 70,
    "LU": 71,
    "Lu": 71,
    "HF": 72,
    "Hf": 72,
    "TA": 73,
    "Ta": 73,
    "W": 74,
    "RE": 75,
    "Re": 75,
    "OS": 76,
    "Os": 76,
    "IR": 77,
    "Ir": 77,
    "PT": 78,
    "Pt": 78,
    "AU": 79,
    "Au": 79,
    "HG": 80,
    "Hg": 80,
    "TL": 81,
    "Tl": 81,
    "PB": 82,
    "Pb": 82,
    "BI": 83,
    "Bi": 83,
    "PO": 84,
    "Po": 84,
    "AT": 85,
    "At": 85,
    "RN": 86,
    "Rn": 86,
    "FR": 87,
    "Fr": 87,
    "RA": 88,
    "Ra": 88,
    "AC": 89,
    "Ac": 89,
    "TH": 90,
    "Th": 90,
    "PA": 91,
    "Pa": 91,
    "U": 92,
    "NP": 93,
    "Np": 93,
    "PU": 94,
    "Pu": 94,
}

_BACKBONE_FIXED_ATOMS = {
    0: set(),
    1: {"CA", "N", "C", "O", "OXT"},
    2: {"CA", "H", "HA", "N", "C", "O", "OXT"},
    3: {"CA", "H", "HA", "N", "C", "O", "CB", "OXT"},
}

_ATOM_NAME_ALIASES = {
    "HN": ("H",),
    "CMA": ("CH3",),
    "CAC": ("C",),
    "OAC": ("O",),
    "H1A": ("H1",),
    "H2A": ("H2",),
    "H3A": ("H3",),
    "NNM": ("N",),
    "CNM": ("C",),
    "HNM": ("H",),
    "H1M": ("H1",),
    "H2M": ("H2",),
    "H3M": ("H3",),
}

from .config import ChargeFitConfig


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
        "esp": str(root / "esp"),
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
    inputs = write_resp_input_files(
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
        read_resp_charges(paths["resp2_chg"]),
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
            charges, antechamber_mol2, stderr = run_antechamber_charge_method(
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
    pro_ff: str,
    workflow: str = "ncaa",
) -> tuple[str, str]:
    workdir = parmfit_workdir(output, workflow)
    base_name = Path(output).stem
    target_chg = os.path.join(workdir, f"{base_name}_target.chg")
    work_mol2 = os.path.join(workdir, f"{base_name}_capped.mol2")
    with open(target_chg, "w", encoding="utf-8") as handle:
        handle.write(" ".join(f"{charge:.10f}" for charge in charges))
        handle.write("\n")
    charged_model = apply_model_charges(representative_model, charges)
    write_resp_mol2(
        work_mol2,
        charged_model,
        bond_pairs,
        pro_ff=pro_ff,
    )
    return os.path.abspath(work_mol2), os.path.abspath(target_chg)



def run_mlip_charge_method(
    conformers: list[tuple[str, dict]],
    *,
    calculator_class: type,
    canonical: str,
    active_calculator,
    total_charge: int,
    multiplicity: int,
    output: str,
) -> tuple[list[np.ndarray], str]:
    """Evaluate MAPLE-model charges for every conformer, sharing one resolved
    calculator across the conformer set; returns (charge sets, model name)."""
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
    return charge_sets, canonical





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
    pro_ff: str,
    wfn_path: str | None = None,
    workflow: str = "ncaa",
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
        resp_result = run_multiconformer_resp(
            output=output,
            conformers=conformers,
            representative_model=representative_model,
            residue_key=residue_key,
            bond_pairs=bond_pairs,
            total_charge=total_charge,
            multiplicity=multiplicity,
            qm=config.qm,
            pro_ff=pro_ff,
            wfn_path=wfn_path,
            workflow=workflow,
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
                os.path.join(workflow, str(label)),
            )
            input_mol2 = os.path.join(workdir, "input.mol2")
            write_resp_mol2(
                input_mol2,
                model,
                bond_pairs,
                pro_ff=pro_ff,
            )
            values, charged_mol2, stderr = run_antechamber_charge_method(
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
        charge_sets, canonical = run_mlip_charge_method(
            conformers,
            calculator_class=calculator_class,
            canonical=canonical,
            active_calculator=active_calculator,
            total_charge=total_charge,
            multiplicity=multiplicity,
            output=output,
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
        # residues without backbone atoms (ligands) spread the delta over all
        # of their own atoms instead
        spread_indices = mainchain_indices or residue_indices
        charges[spread_indices] += charge_delta / len(spread_indices)
    work_mol2, target_chg = _write_multiconformer_charge_outputs(
        output=output,
        representative_model=representative_model,
        bond_pairs=bond_pairs,
        charges=charges,
        pro_ff=pro_ff,
        workflow=workflow,
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



# ---------------------------------------------------------------------------
# RESP input formats, stage runners, and the reference-charge library
# (formerly utils/resp.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RespInputFiles:
    resp1_in: str
    resp2_in: str



def _flatten_model_atoms(model: dict) -> list[tuple[dict, dict]]:
    return flatten_model_atoms(model)



def _flattened_atoms_and_adjacency(
    model: dict,
    bond_pairs: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[dict, dict]], list[dict], dict[int, set[int]]]:
    flattened = _flatten_model_atoms(model)
    atoms = [atom for _, atom in flattened]
    adjacency: dict[int, set[int]] = {index: set() for index in range(1, len(atoms) + 1)}
    for left, right in bond_pairs if bond_pairs is not None else infer_bond_pairs(model):
        adjacency[left].add(right)
        adjacency[right].add(left)
    return flattened, atoms, adjacency



def _candidate_atom_names(atom_name: str) -> list[str]:
    names = [atom_name]
    for alias in _ATOM_NAME_ALIASES.get(atom_name, ()):
        if alias not in names:
            names.append(alias)
    return names



def _library_residue_names(resname: str, category: str) -> list[str]:
    upper = resname.upper()
    names: list[str] = []
    if category == "nterm":
        base = upper[1:] if upper.startswith("N") and len(upper) > 1 else upper
        names.extend([f"N{base}", upper, base])
    elif category == "cterm":
        base = upper[1:] if upper.startswith("C") and len(upper) > 1 else upper
        names.extend([f"C{base}", upper, base])
    else:
        names.append(upper)
        if upper.startswith(("N", "C")) and len(upper) > 1:
            names.append(upper[1:])

    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered



@lru_cache(maxsize=4)
def load_reference_charge_library(pro_ff: str = "ff14SB") -> dict[str, dict[str, dict[str, tuple[str, float]]]]:
    library: dict[str, dict[str, dict[str, tuple[str, float]]]] = {
        "internal": {},
        "nterm": {},
        "cterm": {},
    }
    for template in load_amber_template_registry(pro_ff).templates:
        if template.category not in library:
            continue
        library[template.category][template.name] = {
            atom.name: (atom.amber_type, atom.charge)
            for atom in template.atoms
        }
    return library



def _reference_category(residue: dict, library: dict[str, dict[str, dict[str, tuple[str, float]]]]) -> str | None:
    if residue.get("kind") != "protein":
        return None
    resname = residue["resname"].upper()
    if residue.get("_prev_peptide_key") is None and residue.get("_next_peptide_key") is not None:
        return "nterm" if any(name in library["nterm"] for name in _library_residue_names(resname, "nterm")) else None
    if residue.get("_next_peptide_key") is None and residue.get("_prev_peptide_key") is not None:
        return "cterm" if any(name in library["cterm"] for name in _library_residue_names(resname, "cterm")) else None
    if residue.get("_prev_peptide_key") is not None and residue.get("_next_peptide_key") is not None:
        return "internal" if any(name in library["internal"] for name in _library_residue_names(resname, "internal")) else None
    return None



def _lookup_reference_entry(
    residue: dict,
    atom: dict,
    library: dict[str, dict[str, dict[str, tuple[str, float]]]],
) -> tuple[str, float] | None:
    category = _reference_category(residue, library)
    if category is None:
        return None
    for residue_name in _library_residue_names(residue["resname"], category):
        residue_entries = library[category].get(residue_name)
        if residue_entries is None:
            continue
        for atom_name in _candidate_atom_names(atom["name"]):
            entry = residue_entries.get(atom_name)
            if entry is not None:
                return entry
    return None



def lookup_standard_residue_entry(
    *,
    resname: str,
    atom_name: str,
    category: str,
    pro_ff: str = "ff14SB",
) -> tuple[str, float] | None:
    library = load_reference_charge_library(pro_ff)
    for residue_name in _library_residue_names(resname, category):
        residue_entries = library.get(category, {}).get(residue_name)
        if residue_entries is None:
            continue
        for candidate_name in _candidate_atom_names(atom_name):
            entry = residue_entries.get(candidate_name)
            if entry is not None:
                return entry
    return None



def lookup_standard_atom_entry(
    residue: dict,
    atom: dict,
    *,
    category: str | None = None,
    pro_ff: str = "ff14SB",
) -> tuple[str, float] | None:
    library = load_reference_charge_library(pro_ff)
    if category is not None:
        return lookup_standard_residue_entry(
            resname=residue["resname"],
            atom_name=atom["name"],
            category=category,
            pro_ff=pro_ff,
        )
    return _lookup_reference_entry(residue, atom, library)



def _resolve_fixchg_residue_keys(model: dict, selectors: list[str] | None) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for selector in selectors or []:
        residue = find_unique_residue(model, selector, label="fixchg residue")
        keys.add(get_resid_key(residue))
    return keys



def _backbone_atom_matches(atom_name: str, allowed: set[str]) -> bool:
    if atom_name in allowed:
        return True
    if atom_name == "HN" and "H" in allowed:
        return True
    return False



def collect_fixed_charge_constraints(
    model: dict,
    *,
    chgmod: int,
    fixchg_resids: list[str] | None = None,
    pro_ff: str = "ff14SB",
) -> dict[int, float]:
    library = load_reference_charge_library(pro_ff)
    fixed_keys = _resolve_fixchg_residue_keys(model, fixchg_resids)
    constraints: dict[int, float] = {}
    allowed_backbone_names = _BACKBONE_FIXED_ATOMS[int(chgmod)]

    for atom_index, (residue, atom) in enumerate(_flatten_model_atoms(model), start=1):
        residue_key = get_resid_key(residue)
        matched_entry = None
        if residue.get("kind") == "protein" and atom.get("amber_type") and "charge" in atom:
            matched_entry = (str(atom["amber_type"]), float(atom["charge"]))
        entry = matched_entry or _lookup_reference_entry(residue, atom, library)

        if residue_key in fixed_keys:
            if entry is None:
                raise ValueError(
                    f"fixchg residue {residue['chain']}{residue['resseq']}{residue['icode']}:{residue['resname']} "
                    "does not have a supported amber reference charge definition."
                )
            constraints[atom_index] = float(entry[1])
            continue

        if not allowed_backbone_names:
            continue
        category = residue.get("template_category") or _reference_category(residue, library)
        if category != "internal":
            continue
        if not _backbone_atom_matches(atom.get("role", atom["name"]), allowed_backbone_names):
            continue
        if entry is None:
            continue
        constraints[atom_index] = float(entry[1])

    return constraints



def build_stage2_equivalence_map(
    model: dict,
    *,
    fixed_charge_indices: set[int] | None = None,
    bond_pairs: list[tuple[int, int]] | None = None,
) -> dict[int, int]:
    fixed_charge_indices = fixed_charge_indices or set()
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model, bond_pairs=bond_pairs)

    # Stage 2 reads the stage-1 charges with iqopt=2. Negative ivary values
    # retain those charges; only aliphatic CH2/CH3 groups are refitted.
    ivary: dict[int, int] = {index: -1 for index in range(1, len(atoms) + 1)}
    for carbon_index, atom in enumerate(atoms, start=1):
        if atom["element"] != "C":
            continue
        hydrogens = sorted(
            neighbor
            for neighbor in adjacency[carbon_index]
            if atoms[neighbor - 1]["element"] == "H" and neighbor not in fixed_charge_indices
        )
        if len(hydrogens) not in {2, 3}:
            continue
        if carbon_index not in fixed_charge_indices:
            ivary[carbon_index] = 0
        root = hydrogens[0]
        ivary[root] = 0
        for hydrogen in hydrogens[1:]:
            ivary[hydrogen] = root
    return ivary



def _resp_header(total_charge: int, n_atoms: int, *, stage: int) -> str:
    qwt = "0.00050" if stage == 1 else "0.00100"
    lines = [
        "Resp charges for organic molecule",
        " ",
        " &cntrl",
        " ",
        " nmol = 1,",
        " ihfree = 1,",
        " ioutopt = 1,",
    ]
    if stage == 2:
        lines.append(" iqopt = 2,")
    lines.extend(
        [
            f" qwt = {qwt},",
            " ",
            " &end",
            "    1.0",
            "Resp charges for organic molecule",
            f"{int(total_charge):5d} {int(n_atoms):4d}",
        ]
    )
    return "\n".join(lines) + "\n"



def _write_charge_constraints(
    handle,
    constraints: dict[int, float],
    charge_groups: list[tuple[list[int], float]] | None = None,
) -> None:
    for atom_indices, target_charge in charge_groups or []:
        _write_group_constraint(handle, [(1, atom_index) for atom_index in atom_indices], target_charge)
    for atom_index in sorted(constraints):
        handle.write(f"{1:5d}{constraints[atom_index]:10.5f}\n")
        handle.write(f"{1:5d}{atom_index:5d}\n")
    handle.write("\n\n")



def _write_group_constraint(handle, atom_pairs: list[tuple[int, int]], target_charge: float) -> None:
    handle.write(f"{len(atom_pairs):5d}{float(target_charge):10.5f}\n")
    values: list[int] = []
    for molecule_index, atom_index in atom_pairs:
        values.extend((molecule_index, atom_index))
    for start in range(0, len(values), 16):
        handle.write("".join(f"{value:5d}" for value in values[start : start + 16]) + "\n")



def _write_interstructure_equivalencing(handle, atom_groups: list[list[tuple[int, int]]]) -> None:
    for atom_group in atom_groups:
        handle.write(f"{len(atom_group):5d}\n")
        values: list[int] = []
        for molecule_index, atom_index in atom_group:
            values.extend((molecule_index, atom_index))
        for start in range(0, len(values), 16):
            handle.write("".join(f"{value:5d}" for value in values[start : start + 16]) + "\n")
    handle.write("\n")



def merge_esp_files(esp_files: list[str], output_path: str) -> str:
    with open(output_path, "wb") as out_handle:
        for esp_file in esp_files:
            with open(esp_file, "rb") as in_handle:
                out_handle.write(in_handle.read())
    return output_path



def _model_signature(model: dict) -> list[tuple[str, str, str]]:
    return [
        (residue["resname"], atom["name"], atom["element"])
        for residue, atom in _flatten_model_atoms(model)
    ]



def _group_equivalent_hydrogens(model: dict) -> dict[int, int]:
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model)

    ivary: dict[int, int] = {index: 0 for index in range(1, len(atoms) + 1)}
    for heavy_index, atom in enumerate(atoms, start=1):
        if atom["element"] != "C":
            continue
        hydrogens = sorted(
            neighbor for neighbor in adjacency[heavy_index] if atoms[neighbor - 1]["element"] == "H"
        )
        if len(hydrogens) not in {2, 3}:
            continue
        root = hydrogens[0]
        for hydrogen in hydrogens[1:]:
            ivary[hydrogen] = root
    return ivary



def _free_stage2_atoms(model: dict) -> tuple[set[int], dict[int, int]]:
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model)

    free_atoms: set[int] = set()
    hydrogen_roots: dict[int, int] = {}
    for carbon_index, atom in enumerate(atoms, start=1):
        if atom["element"] != "C":
            continue
        hydrogens = sorted(
            neighbor for neighbor in adjacency[carbon_index] if atoms[neighbor - 1]["element"] == "H"
        )
        if len(hydrogens) not in {2, 3}:
            continue
        free_atoms.add(carbon_index)
        root = hydrogens[0]
        free_atoms.update(hydrogens)
        hydrogen_roots[root] = 0
        for hydrogen in hydrogens[1:]:
            hydrogen_roots[hydrogen] = root
    return free_atoms, hydrogen_roots



def _write_multiconformer_resp_stage(
    handle,
    *,
    models: list[dict],
    ivary_blocks: list[dict[int, int]],
    residue_indices: list[int],
    total_charge: int,
    qwt: str,
    include_iqopt: bool,
) -> None:
    n_models = len(models)
    handle.write("Resp charges for organic molecule\n \n &cntrl\n \n")
    handle.write(f" nmol = {n_models},\n")
    handle.write(" ihfree = 1,\n")
    handle.write(" ioutopt = 1,\n")
    if include_iqopt:
        handle.write(" iqopt = 2,\n")
    handle.write(f" qwt = {qwt},\n \n &end\n")
    for model_index, model in enumerate(models, start=1):
        handle.write("    1.0\n")
        handle.write(f"{model.get('label', f'mol{model_index}')}\n")
        flattened = _flatten_model_atoms(model)
        handle.write(f"{int(model.get('charge', 0)):5d} {len(flattened):4d}\n")
        for atom_index, (_, atom) in enumerate(flattened, start=1):
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{ivary_blocks[model_index - 1][atom_index]:5d}\n")
        handle.write("\n")

    for model_index in range(1, n_models + 1):
        _write_group_constraint(
            handle,
            [(model_index, atom_index) for atom_index in residue_indices],
            float(total_charge),
        )
    handle.write("\n")

    atom_groups: list[list[tuple[int, int]]] = []
    for atom_index in range(1, len(_flatten_model_atoms(models[0])) + 1):
        if include_iqopt and ivary_blocks[0][atom_index] == -1:
            continue
        atom_groups.append([(model_index, atom_index) for model_index in range(1, n_models + 1)])
    _write_interstructure_equivalencing(handle, atom_groups)



def write_multiconformer_resp_input_files(
    workdir: str,
    models: list[dict],
    *,
    labels: list[str] | None = None,
    total_charge: int,
    residue_key: tuple[str, int, str],
) -> RespInputFiles:
    if not models:
        raise ValueError("Multiconformer RESP requires at least one model.")

    signature = _model_signature(models[0])
    for model in models[1:]:
        if _model_signature(model) != signature:
            raise ValueError("All multiconformer RESP models must have identical residue/atom ordering.")

    n_models = len(models)
    base_stage1 = _group_equivalent_hydrogens(models[0])
    stage1_blocks = [dict(base_stage1) for _ in range(n_models)]

    free_atoms, hydrogen_roots = _free_stage2_atoms(models[0])
    base_stage2 = {index: -1 for index in range(1, len(signature) + 1)}
    for atom_index in sorted(free_atoms):
        base_stage2[atom_index] = hydrogen_roots.get(atom_index, 0)
    stage2_blocks = [dict(base_stage2) for _ in range(n_models)]

    residue_indices = [
        atom_index
        for atom_index, (residue, _) in enumerate(_flatten_model_atoms(models[0]), start=1)
        if get_resid_key(residue) == residue_key
    ]
    if not residue_indices:
        raise ValueError("Residue charge constraint requires at least one target residue atom.")

    resolved_labels = labels or [f"conf{index}" for index in range(1, n_models + 1)]
    if len(resolved_labels) != n_models:
        raise ValueError("Multiconformer RESP labels must match the number of models.")
    labeled_models = []
    for model, label in zip(models, resolved_labels, strict=True):
        labeled_models.append(dict(model, label=label))

    resp1_in = Path(workdir) / "resp1.in"
    resp2_in = Path(workdir) / "resp2.in"

    with open(resp1_in, "w", encoding="utf-8") as handle:
        _write_multiconformer_resp_stage(
            handle,
            models=labeled_models,
            ivary_blocks=stage1_blocks,
            residue_indices=residue_indices,
            total_charge=total_charge,
            qwt="0.00050",
            include_iqopt=False,
        )

    with open(resp2_in, "w", encoding="utf-8") as handle:
        _write_multiconformer_resp_stage(
            handle,
            models=labeled_models,
            ivary_blocks=stage2_blocks,
            residue_indices=residue_indices,
            total_charge=total_charge,
            qwt="0.00100",
            include_iqopt=True,
        )

    return RespInputFiles(resp1_in=str(resp1_in), resp2_in=str(resp2_in))



def write_resp_input_files(
    workdir,
    model,
    *,
    total_charge,
    chgmod,
    fixchg_resids=None,
    charge_groups=None,
    pro_ff="ff14SB",
    bond_pairs=None,
    fixed_charges=None,
):
    flattened = _flatten_model_atoms(model)
    constraints = collect_fixed_charge_constraints(
        model,
        chgmod=chgmod,
        fixchg_resids=fixchg_resids,
        pro_ff=pro_ff,
    )
    for atom_index, charge in (fixed_charges or {}).items():
        constraints[int(atom_index)] = float(charge)
    ivary_stage2 = build_stage2_equivalence_map(
        model,
        fixed_charge_indices=set(constraints),
        bond_pairs=bond_pairs or [],
    )
    resp1_in = Path(workdir) / "resp1.in"
    resp2_in = Path(workdir) / "resp2.in"
    with open(resp1_in, "w", encoding="utf-8") as handle:
        handle.write(_resp_header(total_charge, len(flattened), stage=1))
        for _, atom in flattened:
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported RESP atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{0:5d}\n")
        _write_charge_constraints(handle, constraints, charge_groups)
    with open(resp2_in, "w", encoding="utf-8") as handle:
        handle.write(_resp_header(total_charge, len(flattened), stage=2))
        for atom_index, (_, atom) in enumerate(flattened, start=1):
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported RESP atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{ivary_stage2[atom_index]:5d}\n")
        _write_charge_constraints(handle, constraints, charge_groups)
    return RespInputFiles(resp1_in=str(resp1_in), resp2_in=str(resp2_in))





def read_resp_charges(path: str) -> list[float]:
    charges: list[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for token in line.split():
                charges.append(float(token))
    return charges



def apply_resp_charges(model: dict, charges: list[float]) -> dict:
    return apply_model_charges(model, charges)



def _default_atom_type(atom: dict) -> str:
    return atom["element"].lower()



def _mol2_atom_type(
    residue: dict,
    atom: dict,
    library: dict[str, dict[str, dict[str, tuple[str, float]]]],
) -> str:
    explicit = atom.get("atom_type")
    if explicit:
        return str(explicit)
    if residue.get("kind") == "protein" and atom.get("amber_type"):
        return str(atom["amber_type"])
    entry = _lookup_reference_entry(residue, atom, library)
    if entry is not None:
        return entry[0].strip() or _default_atom_type(atom)
    return _default_atom_type(atom)



def write_resp_mol2(
    path: str,
    model: dict,
    bond_pairs: list[tuple[int, int]],
    *,
    atom_type_overrides: dict[int, str] | None = None,
    pro_ff: str = "ff14SB",
) -> None:
    library = load_reference_charge_library(pro_ff)
    flattened = _flatten_model_atoms(model)
    molecule_name = Path(path).stem
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("@<TRIPOS>MOLECULE\n")
        handle.write(f"{molecule_name}\n")
        handle.write(f"{len(flattened):5d}{len(bond_pairs):6d}{1:6d}{0:6d}{0:6d}\n")
        handle.write("SMALL\n")
        handle.write("USER_CHARGES\n\n\n")
        handle.write("@<TRIPOS>ATOM\n")
        for atom_index, (residue, atom) in enumerate(flattened, start=1):
            x, y, z = atom["xyz"]
            charge = float(atom.get("charge", 0.0))
            atom_type = (
                atom_type_overrides[atom_index]
                if atom_type_overrides is not None and atom_index in atom_type_overrides
                else _mol2_atom_type(residue, atom, library)
            )
            handle.write(
                f"{atom_index:7d} {atom['name']:<4s} {x:10.4f}{y:10.4f}{z:10.4f} "
                f"{atom_type:<4s} {1:6d} {residue['resname']:<4s} {charge:12.6f}\n"
            )
        handle.write("@<TRIPOS>BOND\n")
        for bond_id, (left, right) in enumerate(bond_pairs, start=1):
            handle.write(f"{bond_id:6d}{left:5d}{right:5d}{1:2d}\n")
        handle.write("@<TRIPOS>SUBSTRUCTURE\n")
        handle.write(f"{1:6d} {molecule_name:<4s} {1:8d} TEMP              0 ****  ****    0 ROOT\n")




# ---------------------------------------------------------------------------
# RESP stage runners (formerly utils/runtime.py)
# ---------------------------------------------------------------------------


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



def run_antechamber_charge_method(
    input_mol2: str,
    workdir: str,
    *,
    method: str,
    total_charge: int,
    multiplicity: int,
) -> tuple[np.ndarray, str, str]:
    """Run an arbitrary Antechamber charge method and return its MOL2 charges."""
    executable = shutil.which("antechamber")
    if executable is None:
        raise RuntimeError("Required external program 'antechamber' was not found.")
    work_path = Path(workdir)
    work_path.mkdir(parents=True, exist_ok=True)
    output_name = "antechamber_charged.mol2"
    args = [
        executable,
        "-i",
        os.path.abspath(input_mol2),
        "-fi",
        "mol2",
        "-o",
        output_name,
        "-fo",
        "mol2",
        "-c",
        str(method),
        "-nc",
        str(int(total_charge)),
        "-m",
        str(int(multiplicity)),
        "-at",
        "gaff2",
        "-seq",
        "n",
        "-pf",
        "y",
    ]
    completed = sp.run(
        args,
        cwd=str(work_path),
        capture_output=True,
        text=True,
    )
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        detail = stderr.strip() or (completed.stdout or "").strip()
        raise RuntimeError(f"antechamber charge fitting failed:\n{detail}")
    output_path = work_path / output_name
    if not output_path.is_file():
        raise FileNotFoundError(
            f"antechamber did not create the expected MOL2 file: {output_path}"
        )
    topology = parse_mol2(str(output_path))
    charges = np.asarray([atom.charge for atom in topology.atoms], dtype=float)
    return charges, str(output_path.resolve()), stderr



def _site_resp_paths(output: str, *, label: str = "metal_site_resp") -> dict[str, str]:
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
    output,
    model,
    *,
    bond_pairs,
    total_charge,
    multiplicity,
    chgmod,
    qm,
    fixchg_resids=None,
    label="metal_site_resp",
    wat_ff=None,
    pro_ff="ff14SB",
    charge_groups=None,
    wfn_path=None,
    fixed_charges=None,
):
    paths = _site_resp_paths(output, label=label)
    os.makedirs(paths["workdir"], exist_ok=True)
    interface.prepare_gaussian_esp_input(
        paths["gaussian_input"],
        model,
        total_charge=total_charge,
        multiplicity=multiplicity,
        decision=qm,
        wat_ff=wat_ff,
        wfn_path=wfn_path,
    )
    gaussian_log = interface.run_gaussian(paths["gaussian_input"], qm)
    paths["gaussian_log"] = gaussian_log
    run_espgen(paths["gaussian_log"], paths["esp"])
    resp_inputs = write_resp_input_files(
        paths["workdir"],
        model,
        total_charge=total_charge,
        chgmod=chgmod,
        fixchg_resids=fixchg_resids,
        charge_groups=charge_groups,
        pro_ff=pro_ff,
        fixed_charges=fixed_charges,
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
    charges = read_resp_charges(paths["resp2_chg"])
    charged_model = apply_resp_charges(model, charges)
    write_resp_mol2(paths["mol2"], charged_model, bond_pairs, pro_ff=pro_ff)
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
def _multiconformer_resp_paths(output: str, labels: list[str], workflow: str = "ncaa") -> dict[str, str]:
    base = os.path.splitext(output)[0]
    base_name = os.path.basename(base)
    workdir = parmfit_workdir(output, workflow)
    work_prefix = parmfit_work_prefix(output, workflow)
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
    pro_ff: str = "ff14SB",
    wfn_path: str | None = None,
    workflow: str = "ncaa",
) -> MultiRespPipelineResult:
    if not conformers:
        raise ValueError("Multiconformer RESP requires at least one conformer.")

    labels = [label for label, _ in conformers]
    paths = _multiconformer_resp_paths(output, labels, workflow=workflow)
    os.makedirs(paths["workdir"], exist_ok=True)

    conformer_outputs: dict[str, dict[str, str]] = {}
    esp_files: list[str] = []
    models = [model for _, model in conformers]
    for label, model in conformers:
        gaussian_input = paths[f"{label}_gaussian_input"]
        interface.prepare_gaussian_esp_input(
            gaussian_input,
            model,
            total_charge=total_charge,
            multiplicity=multiplicity,
            decision=qm,
            title=f"MAPLE {label} RESP",
            wfn_path=wfn_path,
        )
        gaussian_log = interface.run_gaussian(gaussian_input, qm)
        esp_path = paths[f"{label}_esp"]
        run_espgen(gaussian_log, esp_path)
        conformer_outputs[label] = {
            "gaussian_input": gaussian_input,
            "gaussian_log": gaussian_log,
            "esp": esp_path,
        }
        esp_files.append(esp_path)
    merge_esp_files(esp_files, paths["all_esp"])
    resp_inputs = write_multiconformer_resp_input_files(
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
    charges = read_resp_charges(paths["resp2_chg"])
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
        handle.write(" ".join(str(value) for value in reference_charges))
        handle.write("\n")
    charged_model = apply_resp_charges(representative_model, reference_charges)
    write_resp_mol2(paths["mol2"], charged_model, bond_pairs, pro_ff=pro_ff)
    return MultiRespPipelineResult(
        model=charged_model,
        conformers=conformer_outputs,
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
        decision=qm,
    )

