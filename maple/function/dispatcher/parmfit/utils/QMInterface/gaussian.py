"""Gaussian-backed QM reference jobs for ParmFit."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess as sp
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from .calculator import QMReferenceConfig, QMReferenceResult

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR2_TO_HARTREE_PER_ANG2 = 1.0 / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM)

_TASK_SUFFIX = {
    "opt_frequency": "optfreq",
    "constrained_opt": "constr_opt",
    "gradient": "force",
    "sp": "sp",
    "opt": "opt",
    "freq": "freq",
}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", "", "none"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def build_qm_reference_config(raw_params: dict | None) -> QMReferenceConfig:
    raw = dict(raw_params or {})
    iqm = _coerce_bool(raw.get("iqm", False))
    engine = str(raw.get("qm_engine", "g16")).strip().lower()
    if iqm and engine not in {"gaussian", "g16", "g09", "orca"}:
        raise NotImplementedError(f"Unsupported QM reference engine {engine!r}.")
    opt_level = str(raw.get("opt_level", "B3LYP/def2-SVP")).strip()
    sp_level = str(raw.get("sp_level", "")).strip() or opt_level
    for level in (opt_level, sp_level):
        # I might not consider supporting semi-empirical methods like AM1, PM3, PM6, etc.
        if "/" not in level:
            raise ValueError(f"QM level {level!r} must use METHOD/BASIS syntax.")
        theory, basis = (part.strip() for part in level.split("/", 1))
        if not theory or not basis:
            raise ValueError(f"QM level {level!r} must use METHOD/BASIS syntax.")
    qm_mode = int(raw.get("qm_mode", 1))
    if qm_mode not in {1, 2, 3}:
        raise ValueError("qm_mode must be 1, 2, or 3.")
    return QMReferenceConfig(
        iqm=iqm,
        qm_engine=engine,
        opt_level=opt_level,
        sp_level=sp_level,
        opt_route=str(raw.get("opt_route", "")).strip(),
        sp_route=str(raw.get("sp_route", "")).strip(),
        qm_nproc=int(raw.get("qm_nproc", 8)),
        qm_mem=int(raw.get("qm_mem", 24)),
        qm_mode=qm_mode,
        qm_compare=_coerce_bool(raw.get("qm_compare", False)),
    )


def _task_route(task: str, *, has_frozen_atoms: bool = False) -> str:
    task = str(task).strip().lower()
    if task == "sp":
        return ""
    if task == "opt_frequency":
        return "Opt=ModRedundant Freq" if has_frozen_atoms else "Opt Freq"
    if task == "freq":
        return "Freq"
    if task == "gradient":
        return "Force"
    if task == "constrained_opt":
        return "Opt=ModRedundant"
    if task == "opt":
        return "Opt=ModRedundant" if has_frozen_atoms else "Opt"
    raise ValueError(f"Unknown Gaussian QM task {task!r}.")


def _route_line(
    config: QMReferenceConfig,
    task: str,
    *,
    has_frozen_atoms: bool = False,
    wfn_path: str | os.PathLike[str] | None = None,
) -> str:
    task_name = str(task).strip().lower()
    level = config.sp_level if task_name == "sp" else config.opt_level
    theory, basis = (part.strip() for part in str(level).strip().split("/", 1))
    parts = [f"{theory}/{basis}"]
    route = config.sp_route if task_name == "sp" else config.opt_route
    if route:
        parts.append(route)
    task_route = _task_route(task, has_frozen_atoms=has_frozen_atoms)
    if task_route:
        parts.append(task_route)
    if wfn_path is not None:
        parts.append("Guess=Read")
    return "#P " + " ".join(parts)


def write_gaussian_input(
    path: str | os.PathLike[str],
    atoms: Atoms,
    config: QMReferenceConfig,
    *,
    task: str,
    charge: int = 0,
    multiplicity: int = 1,
    torsion: tuple[int, int, int, int] | None = None,
    torsion_angle_deg: float | None = None,
    frozen_indices: Iterable[int] | None = None,
    wfn_path: str | os.PathLike[str] | None = None,
    title: str = "MAPLE ParmFit QM reference",
) -> str:
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frozen = tuple(int(index) for index in (frozen_indices or ()))
    route_line = _route_line(config, task, has_frozen_atoms=bool(frozen), wfn_path=wfn_path)
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    chk_name = Path(path).with_suffix(".chk").name
    with open(path, "w", encoding="utf-8") as handle:
        if wfn_path is not None:
            handle.write(f"%oldchk={os.fspath(wfn_path)}\n")
        handle.write(f"%chk={chk_name}\n")
        handle.write(f"%nproc={int(config.qm_nproc)}\n")
        handle.write(f"%mem={int(config.qm_mem)}GB\n")
        handle.write(f"{route_line}\n\n")
        handle.write(f"{title}\n\n")
        handle.write(f"{int(charge)} {int(multiplicity)}\n")
        for symbol, xyz in zip(symbols, positions, strict=True):
            handle.write(f" {symbol:<2s} {float(xyz[0]):16.8f} {float(xyz[1]):16.8f} {float(xyz[2]):16.8f}\n")
        handle.write("\n")
        if torsion is not None:
            if torsion_angle_deg is None:
                raise ValueError("torsion_angle_deg is required when torsion is provided.")
            a, b, c, d = (int(value) for value in torsion)
            handle.write(f"D {a:d} {b:d} {c:d} {d:d} {float(torsion_angle_deg):.6f} F\n")
        for index in frozen:
            handle.write(f"X {index + 1:d} F\n")
        if torsion is not None or frozen:
            handle.write("\n")
    return path


def _resolve_gaussian_command(engine: str) -> str:
    if engine in {"g16", "g09"}:
        resolved = shutil.which(engine)
        if resolved:
            return resolved
        raise RuntimeError(f"Gaussian executable {engine!r} was not found.")
    for candidate in ("g16", "g09"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Gaussian executable was not found. Expected one of: g16, g09.")


def run_gaussian_input(path: str | os.PathLike[str], config: QMReferenceConfig) -> str:
    path = Path(path)
    command = _resolve_gaussian_command(config.qm_engine)
    result = sp.run([command, path.name], cwd=str(path.parent), capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"return code {result.returncode}").strip()
        raise RuntimeError(f"Gaussian execution failed for {path}: {detail}")
    log_path = path.with_suffix(".log")
    if not log_path.is_file():
        out_path = path.with_suffix(".out")
        if out_path.is_file():
            return str(out_path)
        raise FileNotFoundError(f"Gaussian did not create {log_path} or {out_path}.")
    return str(log_path)


def run_gaussian_formchk(chk_path: str | os.PathLike[str], fchk_path: str | os.PathLike[str]) -> str:
    chk_path = Path(chk_path)
    fchk_path = Path(fchk_path)
    command = shutil.which("formchk")
    if command is None:
        raise RuntimeError("Gaussian formchk executable was not found; cannot extract Cartesian Hessian from checkpoint.")
    result = sp.run([command, chk_path.name, fchk_path.name], cwd=str(chk_path.parent), capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"return code {result.returncode}").strip()
        raise RuntimeError(f"Gaussian formchk failed for {chk_path}: {detail}")
    if not fchk_path.is_file():
        raise FileNotFoundError(f"Gaussian formchk did not create {fchk_path}.")
    return str(fchk_path)


def _symbol_from_atomic_number(atomic_number: int) -> str:
    atomic_number = int(atomic_number)
    if atomic_number <= 0 or atomic_number >= len(chemical_symbols):
        raise ValueError(f"Unsupported atomic number in Gaussian log: {atomic_number}")
    symbol = chemical_symbols[atomic_number]
    if not symbol:
        raise ValueError(f"Unsupported atomic number in Gaussian log: {atomic_number}")
    return str(symbol)


def _parse_energy(lines: list[str]) -> float:
    energy = None
    for line in lines:
        if "SCF Done:" in line and "=" in line:
            energy = float(line.split("=", 1)[1].split()[0].replace("D", "E"))
    if energy is None:
        raise ValueError("Gaussian log does not contain an SCF Done energy.")
    return float(energy)


def _parse_orientation(lines: list[str]) -> Atoms:
    start = None
    for index, line in enumerate(lines):
        if "Standard orientation:" in line or "Input orientation:" in line:
            start = index
    if start is None:
        raise ValueError("Gaussian log does not contain an orientation block.")

    dashed = []
    for index in range(start + 1, len(lines)):
        if set(lines[index].strip()) == {"-"}:
            dashed.append(index)
            if len(dashed) == 2:
                data_start = index + 1
                break
    else:
        raise ValueError("Gaussian orientation block is incomplete.")

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for line in lines[data_start:]:
        if set(line.strip()) == {"-"}:
            break
        parts = line.split()
        if len(parts) < 6:
            continue
        symbols.append(_symbol_from_atomic_number(int(parts[1])))
        positions.append((float(parts[3]), float(parts[4]), float(parts[5])))
    if not symbols:
        raise ValueError("Gaussian orientation block has no atoms.")
    return Atoms(symbols=symbols, positions=positions)


def _infer_lower_triangle_dimension(value_count: int) -> int:
    ndim = int((np.sqrt(8 * int(value_count) + 1) - 1) / 2)
    if ndim * (ndim + 1) // 2 != int(value_count):
        raise ValueError(f"Packed Hessian length {value_count} is not a triangular matrix size.")
    return ndim


def _fchk_number_of_atoms(lines: list[str]) -> int | None:
    for line in lines:
        if line.startswith("Number of atoms"):
            parts = line.split()
            if parts:
                return int(parts[-1])
    return None


def _fchk_array_header_count(line: str) -> int | None:
    match = re.search(r"\bN\s*=\s*(\d+)", line)
    return int(match.group(1)) if match else None


def parse_gaussian_fchk_hessian(path: str | os.PathLike[str], *, expected_atoms: int | None = None) -> np.ndarray:
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    atom_count = _fchk_number_of_atoms(lines)
    if expected_atoms is not None and atom_count is not None and atom_count != int(expected_atoms):
        raise ValueError(f"Gaussian fchk atom count {atom_count} does not match {expected_atoms} atoms.")

    start = None
    expected_values = None
    for index, line in enumerate(lines):
        if line.startswith("Cartesian Force Constants"):
            start = index + 1
            expected_values = _fchk_array_header_count(line)
            break
    if start is None or expected_values is None:
        raise ValueError(f"Gaussian fchk does not contain Cartesian Force Constants: {path}")

    values: list[float] = []
    for line in lines[start:]:
        for token in line.replace("D", "E").split():
            try:
                values.append(float(token))
            except ValueError:
                continue
            if len(values) == expected_values:
                break
        if len(values) == expected_values:
            break
    if len(values) != expected_values:
        raise ValueError(f"Gaussian fchk Cartesian Force Constants are incomplete: read {len(values)} of {expected_values}.")

    ndim = 3 * int(expected_atoms) if expected_atoms is not None else None
    if ndim is None and atom_count is not None:
        ndim = 3 * int(atom_count)
    inferred_ndim = _infer_lower_triangle_dimension(expected_values)
    if ndim is None:
        ndim = inferred_ndim
    if inferred_ndim != ndim:
        raise ValueError(f"Gaussian fchk Hessian dimension {inferred_ndim} does not match expected dimension {ndim}.")

    hessian = np.zeros((ndim, ndim), dtype=float)
    cursor = 0
    for row in range(ndim):
        for column in range(row + 1):
            value = values[cursor]
            hessian[row, column] = value
            hessian[column, row] = value
            cursor += 1
    return hessian * HARTREE_PER_BOHR2_TO_HARTREE_PER_ANG2


def _load_gaussian_frequency_hessian(log_path: str | os.PathLike[str], input_path: str | os.PathLike[str], natoms: int) -> np.ndarray:
    del log_path
    input_path = Path(input_path)
    fchk_path = input_path.with_suffix(".fchk")
    if fchk_path.is_file():
        return parse_gaussian_fchk_hessian(fchk_path, expected_atoms=natoms)

    chk_path = input_path.with_suffix(".chk")
    if chk_path.is_file():
        try:
            run_gaussian_formchk(chk_path, fchk_path)
            return parse_gaussian_fchk_hessian(fchk_path, expected_atoms=natoms)
        except Exception as formchk_error:
            raise RuntimeError(
                f"Gaussian frequency Hessian requires a readable fchk file; formchk failed for {chk_path}."
            ) from formchk_error

    raise FileNotFoundError(
        f"Gaussian frequency Hessian requires {fchk_path.name} or checkpoint {chk_path.name}; "
        "ordinary Gaussian logs are not accepted as Hessian sources."
    )


def parse_gaussian_log(
    path: str | os.PathLike[str],
    *,
    input_path: str | os.PathLike[str] | None = None,
    expect_hessian: bool = False,
) -> QMReferenceResult:
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    text = "".join(lines)
    if "Normal termination" not in text or "Error termination" in text:
        raise RuntimeError(f"Gaussian job did not terminate normally: {path}")
    atoms = _parse_orientation(lines)
    energy = _parse_energy(lines)
    if expect_hessian:
        raise ValueError(
            "Gaussian Hessian parsing from log is disabled; parse Cartesian Force Constants from a formatted checkpoint."
        )
    return QMReferenceResult(
        atoms=atoms,
        energy_hartree=float(energy),
        input_path=os.fspath(input_path) if input_path is not None else str(Path(path).with_suffix(".gjf")),
        log_path=path,
        hessian=None,
    )


def parse_gaussian_force_log(
    path: str | os.PathLike[str],
    *,
    expected_atoms: int | None = None,
) -> tuple[float, np.ndarray]:
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    text = "".join(lines)
    if "Normal termination" not in text or "Error termination" in text:
        raise RuntimeError(f"Gaussian force job did not terminate normally: {path}")
    energy = _parse_energy(lines)
    block_start = None
    for index, line in enumerate(lines):
        if "Forces (Hartrees/Bohr)" in line:
            block_start = index
    if block_start is None:
        raise ValueError(f"Gaussian log does not contain a Forces (Hartrees/Bohr) block: {path}")

    forces: list[tuple[float, float, float]] = []
    collecting = False
    for line in lines[block_start + 1:]:
        stripped = line.strip()
        if not stripped:
            if collecting:
                break
            continue
        if set(stripped) == {"-"}:
            if collecting:
                break
            continue
        parts = stripped.replace("D", "E").split()
        if len(parts) < 5:
            continue
        try:
            int(parts[0])
            int(parts[1])
            row = (float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        forces.append(row)
        collecting = True
    if expected_atoms is not None and len(forces) != int(expected_atoms):
        raise ValueError(f"Gaussian force block reports {len(forces)} atoms but {expected_atoms} were expected.")
    if not forces:
        raise ValueError(f"Gaussian force block is empty: {path}")
    return float(energy), np.asarray(forces, dtype=float) / BOHR_TO_ANGSTROM


class GaussianReferenceRunner:
    def __init__(self, config: QMReferenceConfig):
        self.config = config

    def _charge_mult(self, atoms: Atoms, charge: int | None, multiplicity: int | None) -> tuple[int, int]:
        resolved_charge = int(atoms.info.get("charge", 0) if charge is None else charge)
        resolved_mult = int(atoms.info.get("mult", 1) if multiplicity is None else multiplicity)
        return resolved_charge, resolved_mult

    def _run(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        task: str,
        charge: int | None = None,
        multiplicity: int | None = None,
        torsion: tuple[int, int, int, int] | None = None,
        torsion_angle_deg: float | None = None,
        frozen_indices: Iterable[int] | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
        expect_hessian: bool = False,
    ) -> QMReferenceResult:
        resolved_charge, resolved_mult = self._charge_mult(atoms, charge, multiplicity)
        input_path = f"{prefix}_{_TASK_SUFFIX.get(task, task)}.gjf"
        write_gaussian_input(
            input_path,
            atoms,
            self.config,
            task=task,
            charge=resolved_charge,
            multiplicity=resolved_mult,
            torsion=torsion,
            torsion_angle_deg=torsion_angle_deg,
            frozen_indices=frozen_indices,
            wfn_path=wfn_path,
        )
        log_path = run_gaussian_input(input_path, self.config)
        wfn_path = str(Path(input_path).with_suffix(".chk"))
        self.last_wfn_path = wfn_path
        if not expect_hessian:
            result = parse_gaussian_log(log_path, input_path=input_path)
            return QMReferenceResult(
                atoms=result.atoms,
                energy_hartree=result.energy_hartree,
                input_path=result.input_path,
                log_path=result.log_path,
                hessian=result.hessian,
                wfn_path=wfn_path,
            )
        result = parse_gaussian_log(log_path, input_path=input_path)
        hessian = _load_gaussian_frequency_hessian(log_path, input_path, len(result.atoms))
        return QMReferenceResult(
            atoms=result.atoms,
            energy_hartree=result.energy_hartree,
            input_path=result.input_path,
            log_path=result.log_path,
            hessian=hessian,
            wfn_path=wfn_path,
        )

    def optimize(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        frozen_indices: Iterable[int] | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> QMReferenceResult:
        return self._run(
            atoms,
            prefix,
            task="opt",
            charge=charge,
            multiplicity=multiplicity,
            frozen_indices=frozen_indices,
            wfn_path=wfn_path,
        )

    def opt_frequency(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        frozen_indices: Iterable[int] | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> QMReferenceResult:
        return self._run(
            atoms,
            prefix,
            task="opt_frequency",
            charge=charge,
            multiplicity=multiplicity,
            frozen_indices=frozen_indices,
            wfn_path=wfn_path,
            expect_hessian=True,
        )

    def constrained_torsion_opt(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        torsion: tuple[int, int, int, int],
        torsion_angle_deg: float,
        charge: int | None = None,
        multiplicity: int | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> QMReferenceResult:
        return self._run(
            atoms,
            prefix,
            task="constrained_opt",
            charge=charge,
            multiplicity=multiplicity,
            torsion=torsion,
            torsion_angle_deg=torsion_angle_deg,
            wfn_path=wfn_path,
        )

    def single_point(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> QMReferenceResult:
        return self._run(
            atoms,
            prefix,
            task="sp",
            charge=charge,
            multiplicity=multiplicity,
            wfn_path=wfn_path,
        )

    def frequency(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> QMReferenceResult:
        return self._run(
            atoms,
            prefix,
            task="freq",
            charge=charge,
            multiplicity=multiplicity,
            wfn_path=wfn_path,
            expect_hessian=True,
        )

    def gradient(
        self,
        atoms: Atoms,
        prefix: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        wfn_path: str | os.PathLike[str] | None = None,
    ) -> tuple[float, np.ndarray, str]:
        resolved_charge, resolved_mult = self._charge_mult(atoms, charge, multiplicity)
        input_path = f"{prefix}_{_TASK_SUFFIX['gradient']}.gjf"
        write_gaussian_input(
            input_path,
            atoms,
            self.config,
            task="gradient",
            charge=resolved_charge,
            multiplicity=resolved_mult,
            wfn_path=wfn_path,
        )
        log_path = run_gaussian_input(input_path, self.config)
        energy, forces = parse_gaussian_force_log(log_path, expected_atoms=len(atoms))
        wfn_path = str(Path(input_path).with_suffix(".chk"))
        self.last_wfn_path = wfn_path
        return energy, forces, wfn_path


def build_qm_reference_runner(config: QMReferenceConfig):
    if not config.iqm:
        return None
    if config.qm_engine == "orca":
        from .orca import ORCAReferenceRunner

        return ORCAReferenceRunner(config)
    return GaussianReferenceRunner(config)
