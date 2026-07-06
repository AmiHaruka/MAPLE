"""ORCA-backed QM reference jobs for ParmFit."""

from __future__ import annotations

import os
import math
from pathlib import Path
import shutil
import subprocess as sp
from typing import Iterable

import numpy as np
from ase import Atoms

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


def _task_route(task: str) -> str:
    task = str(task).strip().lower()
    if task == "sp":
        return ""
    if task == "opt_frequency":
        return "Opt Freq"
    if task == "freq":
        return "Freq"
    if task == "gradient":
        return "EnGrad"
    if task == "constrained_opt":
        return "Opt"
    if task == "opt":
        return "Opt"
    raise ValueError(f"Unknown ORCA QM task {task!r}.")


def _orca_keywords(config: QMReferenceConfig, task: str, *, wfn_path: str | os.PathLike[str] | None = None) -> str:
    task_name = str(task).strip().lower()
    level = config.sp_level if task_name == "sp" else config.opt_level
    # I might not consider supporting semi-empirical methods like AM1, PM3, PM6, etc.
    # Even xTB series methods: why using xTB for QM reference?
    theory, basis = (part.strip() for part in str(level).strip().split("/", 1))
    parts = [theory, basis]
    route = config.sp_route if task_name == "sp" else config.opt_route
    if route:
        parts.append(route)
    task_route = _task_route(task)
    if task_route:
        parts.append(task_route)
    if wfn_path is not None:
        parts.append("MOREAD")
    return "! " + " ".join(part for part in parts if part)


def _wrap_orca_constraint_angle(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def write_orca_input(
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
) -> str:
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frozen = tuple(int(index) for index in (frozen_indices or ()))
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{_orca_keywords(config, task, wfn_path=wfn_path)}\n")
        handle.write(f"%pal nprocs {int(config.qm_nproc)} end\n")
        maxcore = max(1, int(math.ceil(float(config.qm_mem) * 1024.0 / max(int(config.qm_nproc), 1))))
        handle.write(f"%maxcore {maxcore:d}\n")
        if wfn_path is not None:
            handle.write(f'%moinp "{os.fspath(wfn_path)}"\n')
        if torsion is not None or frozen:
            handle.write("%geom\n")
            handle.write("  Constraints\n")
            if torsion is not None:
                if torsion_angle_deg is None:
                    raise ValueError("torsion_angle_deg is required when torsion is provided.")
                zero_based = tuple(int(value) - 1 for value in torsion)
                wrapped_angle = _wrap_orca_constraint_angle(float(torsion_angle_deg))
                handle.write(
                    "    { D "
                    + " ".join(str(value) for value in zero_based)
                    + f" {wrapped_angle:.6f} C }}\n"
                )
            for index in frozen:
                handle.write(f"    {{ C {int(index):d} C }}\n")
            handle.write("  end\n")
            handle.write("end\n")
        handle.write(f"* xyz {int(charge)} {int(multiplicity)}\n")
        for symbol, xyz in zip(symbols, positions, strict=True):
            handle.write(f"  {symbol:<2s} {float(xyz[0]):16.8f} {float(xyz[1]):16.8f} {float(xyz[2]):16.8f}\n")
        handle.write("*\n")
    return path


def _resolve_orca_command() -> str:
    resolved = shutil.which("orca")
    if resolved:
        return resolved
    raise RuntimeError("ORCA executable was not found. Expected command: orca.")


def run_orca_input(path: str | os.PathLike[str], config: QMReferenceConfig) -> str:
    del config
    path = Path(path)
    command = _resolve_orca_command()
    result = sp.run([command, path.name], cwd=str(path.parent), capture_output=True, text=True)
    out_path = path.with_suffix(".out")
    if result.stdout:
        out_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"return code {result.returncode}").strip()
        raise RuntimeError(f"ORCA execution failed for {path}: {detail}")
    if not out_path.is_file():
        raise FileNotFoundError(f"ORCA did not create output for {path}.")
    return str(out_path)


def _parse_orca_energy(lines: list[str]) -> float:
    energy = None
    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[-1].replace("D", "E"))
    if energy is None:
        raise ValueError("ORCA output does not contain a final single point energy.")
    return float(energy)


def _parse_orca_coordinates(lines: list[str]) -> Atoms:
    start = None
    for index, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper():
            start = index
    if start is None:
        raise ValueError("ORCA output does not contain Cartesian coordinates.")

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    collecting = False
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if collecting:
                break
            continue
        if set(stripped) == {"-"}:
            continue
        parts = stripped.split()
        if len(parts) < 4:
            if collecting:
                break
            continue
        try:
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            if collecting:
                break
            continue
        symbols.append(parts[0])
        positions.append(xyz)
        collecting = True
    if not symbols:
        raise ValueError("ORCA Cartesian coordinate block has no atoms.")
    return Atoms(symbols=symbols, positions=positions)


def parse_orca_output(
    path: str | os.PathLike[str],
    *,
    input_path: str | os.PathLike[str] | None = None,
    hessian: np.ndarray | None = None,
) -> QMReferenceResult:
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    text = "".join(lines)
    if "ORCA TERMINATED NORMALLY" not in text or "error termination" in text.lower():
        raise RuntimeError(f"ORCA job did not terminate normally: {path}")
    atoms = _parse_orca_coordinates(lines)
    energy = _parse_orca_energy(lines)
    return QMReferenceResult(
        atoms=atoms,
        energy_hartree=energy,
        input_path=os.fspath(input_path) if input_path is not None else str(Path(path).with_suffix(".inp")),
        log_path=path,
        hessian=hessian,
    )


def _next_orca_engrad_value(lines: list[str], start: int) -> tuple[int, str]:
    cursor = start
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        cursor += 1
        if not stripped or stripped.startswith("#"):
            continue
        return cursor, stripped.replace("D", "E")
    raise ValueError("Unexpected end of ORCA engrad file.")


def parse_orca_engrad(
    path: str | os.PathLike[str],
    *,
    expected_atoms: int | None = None,
) -> tuple[float, np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    natoms = None
    energy = None
    gradients: list[float] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        if "number of atoms" in lower:
            _, value = _next_orca_engrad_value(lines, index + 1)
            natoms = int(value.split()[0])
        elif "current total energy" in lower:
            _, value = _next_orca_engrad_value(lines, index + 1)
            energy = float(value.split()[0])
        elif "current gradient" in lower:
            if natoms is None:
                raise ValueError("ORCA engrad gradient appears before the atom count.")
            cursor = index + 1
            gradients = []
            while len(gradients) < 3 * natoms:
                cursor, value = _next_orca_engrad_value(lines, cursor)
                gradients.append(float(value.split()[0]))
            break

    if natoms is None:
        raise ValueError(f"ORCA engrad file does not report atom count: {path}")
    if expected_atoms is not None and natoms != int(expected_atoms):
        raise ValueError(f"ORCA engrad reports {natoms} atoms but {expected_atoms} were expected.")
    if energy is None:
        raise ValueError(f"ORCA engrad file does not report total energy: {path}")
    if len(gradients) != 3 * natoms:
        raise ValueError(f"ORCA engrad gradient block is incomplete: {path}")
    gradient = np.asarray(gradients, dtype=float).reshape((natoms, 3))
    return float(energy), -gradient / BOHR_TO_ANGSTROM


def _parse_orca_hessian_atom_count(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip().lower() == "$atoms":
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor >= len(lines):
                raise ValueError("ORCA hessian $atoms block is missing its atom count.")
            return int(lines[cursor].split()[0])
    return None


def parse_orca_hessian(path: str | os.PathLike[str], *, expected_atoms: int | None = None) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    atom_count = _parse_orca_hessian_atom_count(lines)
    if expected_atoms is not None and atom_count is not None and atom_count != int(expected_atoms):
        raise ValueError(f"ORCA hessian atoms block reports {atom_count} atoms but {expected_atoms} were expected.")
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "$hessian":
            start = index + 1
            break
    if start is None:
        raise ValueError(f"ORCA hessian file does not contain a $hessian block: {path}")
    cursor = start
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("ORCA hessian block is missing its dimension.")
    ndim = int(lines[cursor].split()[0])
    if expected_atoms is not None and ndim != 3 * int(expected_atoms):
        raise ValueError(f"ORCA hessian dimension {ndim} does not match {expected_atoms} atoms.")
    if atom_count is not None and ndim != 3 * int(atom_count):
        raise ValueError(f"ORCA hessian dimension {ndim} does not match atoms block count {atom_count}.")
    cursor += 1
    hessian = np.zeros((ndim, ndim), dtype=float)
    filled = np.zeros((ndim, ndim), dtype=bool)
    columns: list[int] = []
    for line in lines[cursor:]:
        parts = line.replace("D", "E").split()
        if not parts:
            continue
        if parts[0].lower().startswith("$"):
            break
        if all(part.lstrip("-").isdigit() for part in parts):
            columns = [int(part) for part in parts]
            continue
        try:
            row = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if row < 0 or row >= ndim:
            continue
        for column, value in zip(columns, values, strict=False):
            if column < 0 or column >= ndim:
                continue
            hessian[row, column] = value
            filled[row, column] = True
    if not np.all(filled):
        read_values = int(np.count_nonzero(filled))
        raise ValueError(f"ORCA hessian block is incomplete: read {read_values} of {ndim * ndim} matrix values.")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-8):
        max_diff = float(np.max(np.abs(hessian - hessian.T)))
        raise ValueError(f"ORCA hessian matrix is not symmetric; max absolute difference is {max_diff:.3e}.")
    return hessian * HARTREE_PER_BOHR2_TO_HARTREE_PER_ANG2


class ORCAReferenceRunner:
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
        input_path = f"{prefix}_{_TASK_SUFFIX.get(task, task)}.inp"
        write_orca_input(
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
        output_path = run_orca_input(input_path, self.config)
        hessian = None
        if expect_hessian:
            hessian_path = str(Path(input_path).with_suffix(".hess"))
            if not os.path.isfile(hessian_path):
                raise FileNotFoundError(f"ORCA frequency job did not create {hessian_path}.")
            hessian = parse_orca_hessian(hessian_path, expected_atoms=len(atoms))
        result = parse_orca_output(output_path, input_path=input_path, hessian=hessian)
        wfn_path = str(Path(input_path).with_suffix(".gbw"))
        self.last_wfn_path = wfn_path
        return QMReferenceResult(
            atoms=result.atoms,
            energy_hartree=result.energy_hartree,
            input_path=result.input_path,
            log_path=result.log_path,
            hessian=result.hessian,
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
        input_path = f"{prefix}_{_TASK_SUFFIX['gradient']}.inp"
        write_orca_input(
            input_path,
            atoms,
            self.config,
            task="gradient",
            charge=resolved_charge,
            multiplicity=resolved_mult,
            wfn_path=wfn_path,
        )
        run_orca_input(input_path, self.config)
        engrad_path = str(Path(input_path).with_suffix(".engrad"))
        if not os.path.isfile(engrad_path):
            raise FileNotFoundError(f"ORCA gradient job did not create {engrad_path}.")
        energy, forces = parse_orca_engrad(engrad_path, expected_atoms=len(atoms))
        wfn_path = str(Path(input_path).with_suffix(".gbw"))
        self.last_wfn_path = wfn_path
        return energy, forces, wfn_path
