"""Convert MAPLE text output to a Gaussian-like log file.

The goal is compatibility with post-processing tools that only understand the
common Gaussian log layout.  This module intentionally does *not* run quantum
chemistry; it rewrites MAPLE coordinates, energies, frequencies, and basic
thermochemistry into Gaussian-shaped sections.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    from maple import __version__ as _VERSION
except Exception:  # pragma: no cover - importlib fallback is packaging-only
    _VERSION = "0.1.2"


EV_PER_HARTREE = 27.211386245988
KCAL_PER_HARTREE = 627.5094740631
BOHR_PER_ANGSTROM = 1.8897261246257702

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

ENERGY_RE = re.compile(
    rf"\b(?:Final\s+Energy|Energy)\s*[:=]\s*({FLOAT_PATTERN})"
    r"(?:\s*(Hartree|Eh|Ha|a\.?u\.?|eV))?",
    re.IGNORECASE,
)
ITERATION_RE = re.compile(r"\bIteration:\s*(\d+)\b", re.IGNORECASE)
SCAN_RE = re.compile(
    r"\bScanning combination\s+(\d+)/(\d+)(?::\s*\[([^\]]*)\])?",
    re.IGNORECASE,
)
MODEL_RE = re.compile(r"\bGlobal parameter:\s*model\s*=\s*([^\s]+)", re.IGNORECASE)
TASK_RE = re.compile(r"\bTask set to\s*'([^']+)'", re.IGNORECASE)
CONFIG_TASK_RE = re.compile(r"^\s*Task:\s*([A-Za-z0-9_+-]+)\s*$")
CONFIG_METHOD_RE = re.compile(r"^\s*method\s*:\s*([A-Za-z0-9_+-]+)\s*$", re.IGNORECASE)
CHARGE_MULT_RE = re.compile(r"^\s*([+-]?\d+)\s+(\d+)\s*$")
ATOM_RE = re.compile(
    rf"^\s*(?:(\d+)\s+)?([A-Z][a-z]?)\s+"
    rf"({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*$"
)
ATOM_WITH_EXTRAS_RE = re.compile(
    rf"^\s*(?:(\d+)\s+)?([A-Z][a-z]?)\s+"
    rf"({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})"
    rf"(?:\s+{FLOAT_PATTERN}){{0,3}}\s*$"
)
FREQ_RE = re.compile(rf"^\s*\d+\s*:\s*({FLOAT_PATTERN})\s+cm\*\*-1", re.IGNORECASE)
THERMO_RE = re.compile(rf"\.\.\.\s*({FLOAT_PATTERN})\s+kcal/mol", re.IGNORECASE)
MD_THERMO_RE = re.compile(
    rf"^\s*(\d+)\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+"
    rf"({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*$"
)
SCAN_DIRECTIVE_RE = re.compile(
    rf"^\s*S\s+((?:\d+\s+){{1,4}})({FLOAT_PATTERN})\s+(\d+)\s*$",
    re.IGNORECASE,
)

ELEMENTS = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]
ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(ELEMENTS) if number}


@dataclass(frozen=True)
class AtomRecord:
    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class EnergyRecord:
    value: float
    unit: str | None


@dataclass(frozen=True)
class ConvergenceRecord:
    maximum_force: tuple[float, float, bool] | None = None
    rms_force: tuple[float, float, bool] | None = None
    maximum_displacement: tuple[float, float, bool] | None = None
    rms_displacement: tuple[float, float, bool] | None = None


@dataclass(frozen=True)
class ScanDirective:
    atoms: tuple[int, ...]
    step: float
    steps: int


@dataclass
class Frame:
    atoms: list[AtomRecord]
    energy: EnergyRecord | None = None
    label: str | None = None
    convergence: ConvergenceRecord | None = None
    scan_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class MdThermoRecord:
    step: int
    time_fs: float
    temperature_k: float
    kinetic_hartree: float
    potential_hartree: float
    total_hartree: float


@dataclass
class MapleOutput:
    source: Path
    task: str | None = None
    method: str | None = None
    model: str | None = None
    charge: int = 0
    multiplicity: int = 1
    initial_atoms: list[AtomRecord] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=list)
    normal_modes: dict[int, list[float]] = field(default_factory=dict)
    thermo_kcal: dict[str, float] = field(default_factory=dict)
    sequence_kind: str | None = None
    md_thermo: list[MdThermoRecord] = field(default_factory=list)
    scan_directives: list[ScanDirective] = field(default_factory=list)


def _atomic_number(symbol: str) -> int:
    try:
        return ATOMIC_NUMBERS[symbol]
    except KeyError as exc:
        raise ValueError(f"Unknown element symbol in MAPLE geometry: {symbol}") from exc


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", "*", "="}


def _parse_atom_line(line: str) -> AtomRecord | None:
    match = ATOM_WITH_EXTRAS_RE.match(line)
    if not match:
        return None

    symbol = match.group(2)
    if symbol not in ATOMIC_NUMBERS:
        return None

    return AtomRecord(
        symbol=symbol,
        x=float(match.group(3)),
        y=float(match.group(4)),
        z=float(match.group(5)),
    )


def _parse_coordinate_block(lines: Sequence[str], header_index: int) -> tuple[list[AtomRecord], int]:
    atoms: list[AtomRecord] = []
    index = header_index + 1

    while index < len(lines):
        stripped = lines[index].strip()
        atom = _parse_atom_line(lines[index])

        if atom is not None:
            atoms.append(atom)
            index += 1
            continue

        if atoms:
            if stripped == "" or _is_separator(stripped):
                index += 1
                continue
            break

        # Preamble inside MAPLE coordinate sections.
        if (
            stripped == ""
            or _is_separator(stripped)
            or stripped.lower().startswith("group ")
            or stripped.lower().startswith("center ")
            or stripped.lower().startswith("number ")
            or stripped.lower().startswith("atom ")
        ):
            index += 1
            continue

        # Frequency outputs may print a Coordinates heading without a geometry
        # table; do not scan into unrelated numeric tables.
        if (
            "frequency" in stripped.lower()
            or "thermochemistry" in stripped.lower()
            or "normal modes" in stripped.lower()
            or stripped.lower().startswith("starting ")
        ):
            break

        index += 1

    return atoms, index


def _parse_xyz(path: Path) -> list[AtomRecord]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")

    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {path}") from exc

    atoms: list[AtomRecord] = []
    for line in lines[2 : 2 + atom_count]:
        atom = _parse_atom_line(line)
        if atom is None:
            raise ValueError(f"Invalid XYZ coordinate line in {path}: {line}")
        atoms.append(atom)

    return atoms


def _parse_xyz_trajectory(path: Path) -> list[Frame]:
    lines = path.read_text().splitlines()
    frames: list[Frame] = []
    index = 0

    while index < len(lines):
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            break

        try:
            atom_count = int(lines[index].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count in {path}: {lines[index]}") from exc

        if index + 1 >= len(lines):
            break

        comment = lines[index + 1].strip()
        energy_match = ENERGY_RE.search(comment)
        energy = None
        if energy_match:
            energy = EnergyRecord(float(energy_match.group(1)), energy_match.group(2))
        scan_values = _parse_scan_values(comment)

        atoms: list[AtomRecord] = []
        for line in lines[index + 2 : index + 2 + atom_count]:
            atom = _parse_atom_line(line)
            if atom is None:
                raise ValueError(f"Invalid XYZ coordinate line in {path}: {line}")
            atoms.append(atom)

        if len(atoms) != atom_count:
            raise ValueError(f"XYZ atom count mismatch in {path}: expected {atom_count}, got {len(atoms)}")

        frames.append(
            Frame(
                atoms=atoms,
                energy=energy,
                label=comment or f"Frame {len(frames) + 1}",
                scan_values=scan_values,
            )
        )
        index += atom_count + 2

    return frames


def _parse_scan_values(text: str) -> tuple[float, ...]:
    match = SCAN_RE.search(text)
    if not match or not match.group(3):
        return ()
    values: list[float] = []
    for token in match.group(3).split(","):
        try:
            values.append(float(token.strip()))
        except ValueError:
            return ()
    return tuple(values)


def _parse_md_thermo(path: Path) -> list[MdThermoRecord]:
    records: list[MdThermoRecord] = []
    if not path.exists():
        return records

    for line in path.read_text(errors="replace").splitlines():
        match = MD_THERMO_RE.match(line)
        if not match:
            continue
        records.append(
            MdThermoRecord(
                step=int(match.group(1)),
                time_fs=float(match.group(2)),
                temperature_k=float(match.group(3)),
                kinetic_hartree=float(match.group(4)),
                potential_hartree=float(match.group(5)),
                total_hartree=float(match.group(6)),
            )
        )
    return records


def _load_irc_frames(source: Path) -> list[Frame]:
    forward_path = source.with_name(f"{source.stem}_forward.xyz")
    backward_path = source.with_name(f"{source.stem}_backward.xyz")
    if forward_path.exists() and backward_path.exists():
        forward = _parse_xyz_trajectory(forward_path)
        backward = _parse_xyz_trajectory(backward_path)
        if forward and backward:
            frames = list(reversed(backward))
            frames.extend(forward)
            for index, frame in enumerate(frames, start=1):
                frame.label = f"IRC point {index}"
            return frames

    full_path = source.with_name(f"{source.stem}_full.xyz")
    if full_path.exists():
        frames = _parse_xyz_trajectory(full_path)
        for index, frame in enumerate(frames, start=1):
            frame.label = f"IRC point {index}"
        return frames

    return []


def _resolve_geometry_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(base_dir / path)

    candidates.append(base_dir / path.name)

    marker = "/example/"
    raw_text = raw_path.replace("\\", "/")
    if marker in raw_text:
        rel_after_example = raw_text.split(marker, 1)[1]
        for parent in [base_dir, *base_dir.parents]:
            if parent.name == "example":
                candidates.append(parent / rel_after_example)
                break

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last resort for stale absolute paths in shipped example inputs.
    for parent in [base_dir, *base_dir.parents]:
        if parent.name == "example":
            matches = list(parent.rglob(path.name))
            if matches:
                return matches[0]
            break

    return candidates[0]


def _parse_inp_geometry(path: Path) -> tuple[list[AtomRecord], int, int]:
    lines = path.read_text().splitlines()
    atoms: list[AtomRecord] = []
    charge = 0
    multiplicity = 1
    in_coordinates = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if in_coordinates and atoms:
                break
            continue

        if stripped.startswith("#"):
            if not in_coordinates:
                continue
            break

        if stripped.upper().startswith("XYZ "):
            tokens = stripped.split()
            if len(tokens) >= 4 and re.fullmatch(r"[+-]?\d+", tokens[1]) and re.fullmatch(r"\d+", tokens[2]):
                charge = int(tokens[1])
                multiplicity = int(tokens[2])
                xyz_path_text = tokens[3]
            elif len(tokens) >= 2:
                xyz_path_text = tokens[1]
            else:
                raise ValueError(f"Invalid XYZ reference in {path}: {stripped}")
            xyz_path = _resolve_geometry_path(xyz_path_text, path.parent)
            return _parse_xyz(xyz_path), charge, multiplicity

        charge_mult = CHARGE_MULT_RE.match(stripped)
        if charge_mult and not atoms:
            charge = int(charge_mult.group(1))
            multiplicity = int(charge_mult.group(2))
            in_coordinates = True
            continue

        atom = _parse_atom_line(stripped)
        if atom is not None:
            atoms.append(atom)
            in_coordinates = True
            continue

        if in_coordinates:
            break

    return atoms, charge, multiplicity


def _parse_scan_directives(path: Path) -> list[ScanDirective]:
    directives: list[ScanDirective] = []
    if not path.exists():
        return directives

    for line in path.read_text(errors="replace").splitlines():
        match = SCAN_DIRECTIVE_RE.match(line)
        if not match:
            continue
        atoms = tuple(int(token) for token in match.group(1).split())
        directives.append(
            ScanDirective(
                atoms=atoms,
                step=float(match.group(2)),
                steps=int(match.group(3)),
            )
        )
    return directives


def _load_sidecar_geometry(output_path: Path) -> tuple[list[AtomRecord], int, int]:
    for suffix in (".xyz", ".inp"):
        candidate = output_path.with_suffix(suffix)
        if not candidate.exists():
            continue
        if suffix == ".xyz":
            return _parse_xyz(candidate), 0, 1
        return _parse_inp_geometry(candidate)

    return [], 0, 1


def _parse_frequency_values(lines: Iterable[str]) -> list[float]:
    frequencies: list[float] = []
    in_section = False

    for line in lines:
        upper = line.upper()
        if "VIBRATIONAL FREQUENCIES" in upper:
            in_section = True
            continue
        if in_section and ("THERMOCHEMISTRY" in upper or "NORMAL MODES" in upper):
            break
        if not in_section:
            continue

        match = FREQ_RE.match(line)
        if match:
            frequencies.append(float(match.group(1)))

    return frequencies


def _parse_thermochemistry(lines: Iterable[str]) -> dict[str, float]:
    values: dict[str, float] = {}

    for line in lines:
        match = THERMO_RE.search(line)
        if not match:
            continue

        value = float(match.group(1))
        lowered = line.lower()
        if "zero point energy" in lowered:
            values["zero_point"] = value
        elif "total thermal correction" in lowered:
            values["thermal_energy"] = value
        elif "total enthalpy correction" in lowered:
            values["enthalpy"] = value
        elif "gibbs free energy" in lowered:
            values["gibbs"] = value

    return values


def _parse_normal_modes(lines: Sequence[str]) -> dict[int, list[float]]:
    modes: dict[int, list[float]] = {}
    in_section = False
    active_columns: list[int] = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if "NORMAL MODES" in upper:
            in_section = True
            continue
        if not in_section:
            continue

        if not stripped:
            active_columns = []
            continue
        if stripped.startswith("These modes") or stripped.startswith("M(") or stripped.startswith("Thus,"):
            continue
        if stripped.startswith("Program started") or stripped.startswith("="):
            break

        tokens = stripped.split()
        if tokens and all(token.isdigit() for token in tokens):
            active_columns = [int(token) for token in tokens]
            for mode_index in active_columns:
                modes.setdefault(mode_index, [])
            continue

        if active_columns and tokens and tokens[0].isdigit():
            values = tokens[1:]
            if len(values) < len(active_columns):
                continue
            for mode_index, token in zip(active_columns, values):
                try:
                    modes.setdefault(mode_index, []).append(float(token))
                except ValueError:
                    pass

    return modes


def _parse_convergence(lines: Sequence[str], energy_line_index: int) -> ConvergenceRecord | None:
    values: dict[str, tuple[float, float, bool]] = {}
    patterns = {
        "maximum_force": re.compile(
            rf"^\s*Maximum\s+Force:\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+(Yes|No)\b",
            re.IGNORECASE,
        ),
        "rms_force": re.compile(
            rf"^\s*RMS\s+Force:\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+(Yes|No)\b",
            re.IGNORECASE,
        ),
        "maximum_displacement": re.compile(
            rf"^\s*Maximum\s+Displacement:\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+(Yes|No)\b",
            re.IGNORECASE,
        ),
        "rms_displacement": re.compile(
            rf"^\s*RMS\s+Displacement:\s*({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+(Yes|No)\b",
            re.IGNORECASE,
        ),
    }

    for line in lines[energy_line_index + 1 : energy_line_index + 8]:
        for key, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                values[key] = (
                    float(match.group(1)),
                    float(match.group(2)),
                    match.group(3).lower() == "yes",
                )

    if not values:
        return None

    return ConvergenceRecord(
        maximum_force=values.get("maximum_force"),
        rms_force=values.get("rms_force"),
        maximum_displacement=values.get("maximum_displacement"),
        rms_displacement=values.get("rms_displacement"),
    )


def parse_maple_output(path: str | Path) -> MapleOutput:
    source = Path(path)
    lines = source.read_text(errors="replace").splitlines()
    parsed = MapleOutput(source=source)

    sidecar_atoms, sidecar_charge, sidecar_multiplicity = _load_sidecar_geometry(source)
    parsed.charge = sidecar_charge
    parsed.multiplicity = sidecar_multiplicity
    parsed.scan_directives = _parse_scan_directives(source.with_suffix(".inp"))

    current_atoms: list[AtomRecord] = []
    current_label: str | None = None
    current_scan_values: tuple[float, ...] = ()

    index = 0
    while index < len(lines):
        line = lines[index]

        model_match = MODEL_RE.search(line)
        if model_match:
            parsed.model = model_match.group(1)

        task_match = TASK_RE.search(line) or CONFIG_TASK_RE.search(line)
        if task_match:
            parsed.task = task_match.group(1).lower()

        method_match = CONFIG_METHOD_RE.search(line)
        if method_match:
            parsed.method = method_match.group(1).lower()

        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            current_label = f"Iteration {iteration_match.group(1)}"

        scan_match = SCAN_RE.search(line)
        if scan_match:
            current_label = f"Scan {scan_match.group(1)} of {scan_match.group(2)}"
            current_scan_values = _parse_scan_values(line)

        if line.strip() == "Coordinates":
            atoms, next_index = _parse_coordinate_block(lines, index)
            if atoms:
                current_atoms = atoms
                if not parsed.initial_atoms:
                    parsed.initial_atoms = atoms
                index = next_index
                continue

        energy_match = ENERGY_RE.search(line)
        if energy_match:
            atoms_for_frame = current_atoms or sidecar_atoms
            if atoms_for_frame:
                frame = Frame(
                    atoms=list(atoms_for_frame),
                    energy=EnergyRecord(
                        value=float(energy_match.group(1)),
                        unit=energy_match.group(2),
                    ),
                    label=current_label or f"Frame {len(parsed.frames) + 1}",
                    convergence=_parse_convergence(lines, index),
                    scan_values=current_scan_values,
                )
                if not parsed.frames or parsed.frames[-1].atoms != frame.atoms or parsed.frames[-1].energy != frame.energy:
                    parsed.frames.append(frame)

        index += 1

    if not parsed.initial_atoms:
        parsed.initial_atoms = sidecar_atoms

    irc_frames = _load_irc_frames(source)
    if irc_frames and len(irc_frames) > len(parsed.frames):
        parsed.frames = irc_frames
        parsed.initial_atoms = irc_frames[0].atoms
        parsed.task = "irc"
        parsed.sequence_kind = "irc"

    trajectory_candidates = [
        (source.with_name(f"{source.stem}_md_traj.xyz"), "md", "md"),
        (source.with_name(f"{source.stem}_autoneb_global_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_autoneb_path_root_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_nebts_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_stringts_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_cineb_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_gsm_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_mep.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_image_traj.xyz"), "ts", "reaction_path"),
        (source.with_name(f"{source.stem}_dimer_traj.xyz"), "ts", "ts_optimization"),
        (source.with_name(f"{source.stem}_prfo_traj.xyz"), "ts", "ts_optimization"),
        (source.with_name(f"{source.stem}_gsm_relax_traj.xyz"), "ts", "ts_optimization"),
    ]
    for xyz_traj, task_name, sequence_kind in trajectory_candidates:
        if not xyz_traj.exists():
            continue
        frames = _parse_xyz_trajectory(xyz_traj)
        if sequence_kind == "ts_optimization" and parsed.frames:
            parsed.task = task_name
            parsed.sequence_kind = sequence_kind
            break
        if frames and (
            sequence_kind == "reaction_path"
            or len(frames) > len(parsed.frames)
            or not parsed.frames
        ):
            parsed.frames = frames
            parsed.initial_atoms = frames[0].atoms
            parsed.task = task_name
            parsed.sequence_kind = sequence_kind
            break

    if parsed.sequence_kind == "md":
        parsed.md_thermo = _parse_md_thermo(source.with_name(f"{source.stem}_md_thermo.dat"))

    if not parsed.frames and parsed.initial_atoms:
        parsed.frames.append(Frame(atoms=list(parsed.initial_atoms), label="Frame 1"))

    parsed.frequencies = _parse_frequency_values(lines)
    parsed.normal_modes = _parse_normal_modes(lines)
    parsed.thermo_kcal = _parse_thermochemistry(lines)

    charge_match = re.search(r"\bCharge\s*=\s*([+-]?\d+)\s+Multiplicity\s*=\s*(\d+)\b", "\n".join(lines))
    if charge_match:
        parsed.charge = int(charge_match.group(1))
        parsed.multiplicity = int(charge_match.group(2))

    return parsed


def _canonical_energy_unit(unit: str | None) -> str | None:
    if unit is None:
        return None

    normalized = unit.lower().replace(".", "")
    if normalized in {"hartree", "eh", "ha", "au", "a u"}:
        return "hartree"
    if normalized == "ev":
        return "ev"
    return None


def _energy_to_hartree(
    energy: EnergyRecord,
    policy: str,
    atom_count: int | None = None,
    model: str | None = None,
) -> float:
    if policy == "hartree":
        return energy.value
    if policy == "ev":
        return energy.value / EV_PER_HARTREE
    if policy != "auto":
        raise ValueError(f"Unsupported energy unit policy: {policy}")

    unit = _canonical_energy_unit(energy.unit)
    if unit == "ev":
        return energy.value / EV_PER_HARTREE
    if unit is None and atom_count and model and model.lower() == "uma" and abs(energy.value) / atom_count > 5.0:
        return energy.value / EV_PER_HARTREE
    return energy.value


def _frame_energy_hartree(frame: Frame, policy: str, model: str | None = None) -> float | None:
    if frame.energy is None:
        return None
    return _energy_to_hartree(frame.energy, policy, len(frame.atoms), model)


def _result_kind(parsed: MapleOutput) -> str:
    if parsed.sequence_kind == "md":
        return "md"
    if parsed.sequence_kind in {"irc", "reaction_path"}:
        return "irc"
    if parsed.task == "irc":
        return "irc"
    if parsed.task == "freq" or parsed.frequencies:
        return "freq"
    if parsed.task == "ts":
        return "ts_opt"
    if parsed.task == "scan":
        return "scan"
    if parsed.task == "opt" or len(parsed.frames) > 1:
        return "opt"
    return "sp"


def _irc_max_points_per_path(frames: Sequence[Frame]) -> int:
    coords = _reaction_coordinates(frames)
    return max(
        sum(1 for coord in coords if coord > 0),
        sum(1 for coord in coords if coord < 0),
        1,
    )


def _irc_transition_state_index(frames: Sequence[Frame]) -> int:
    if not frames:
        return 0
    return max(
        range(len(frames)),
        key=lambda index: _frame_energy_hartree(frames[index], "auto") or float("-inf"),
    )


def _route_for(parsed: MapleOutput) -> str:
    kind = _result_kind(parsed)
    max_points = max(len(parsed.frames), 1)

    if kind == "freq":
        return "#P HF/Gen Freq NoSymm"
    if kind == "irc":
        return f"#P HF/Gen IRC=(CalcFC,MaxPoints={_irc_max_points_per_path(parsed.frames)},Report) NoSymm"
    if kind == "md":
        return f"#P HF/Gen ADMP(MaxPoints={max_points}) NoSymm"
    if kind == "ts_opt":
        # GaussView has no separate TS-result viewer: TS searches are read via
        # the Optimization result path.  Keeping the route as a plain Opt avoids
        # GaussView's stricter TS preprocessor expectations while preserving
        # the MAPLE TS optimization trajectory, energies, and convergence data.
        return "#P HF/Gen Opt NoSymm"
    if kind == "scan":
        return "#P HF/Gen Opt=ModRedundant NoSymm"
    if kind == "opt":
        return "#P HF/Gen Opt NoSymm"
    return "#P HF/Gen SP NoSymm"


def _format_gaussian_preamble(
    parsed: MapleOutput,
    route: str,
    alpha_electrons: int,
    beta_electrons: int,
) -> list[str]:
    title = f"MAPLE converted {parsed.task or 'unknown'} result"
    return [
        f"! This file was generated by maple2gaussian version {_VERSION}",
        f"! Source program: MAPLE task={parsed.task or 'unknown'} method={parsed.method or 'unknown'} model={parsed.model or 'unknown'}",
        "! Energy unit policy: auto",
        "",
        " Entering Gaussian System, Link 0=g16",
        f" Input={parsed.source.name}",
        f" Output={parsed.source.with_suffix('.log').name}",
        " Initial command:",
        f" /usr/local/g16/l1.exe \"{parsed.source.with_suffix('.gjf').name}\" -scrdir=\"/tmp\"",
        " Default CPUs for threads: 1",
        " Entering Link 1 = /usr/local/g16/l1.exe PID=         1.",
        " ------------------------------------------------------------",
        f" {route}",
        " ------------------------------------------------------------",
        title,
        "",
        f" Charge = {parsed.charge:4d} Multiplicity = {parsed.multiplicity}",
        "0 basis functions",
        f"{alpha_electrons} alpha electrons",
        f"{beta_electrons} beta electrons",
        "",
    ]


def _format_orientation(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        "                        Standard orientation:",
        "---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        "---------------------------------------------------------------------",
    ]
    for number, atom in enumerate(atoms, start=1):
        lines.append(
            f"{number:7d}{_atomic_number(atom.symbol):11d}{0:12d}"
            f"{atom.x:16.6f}{atom.y:12.6f}{atom.z:12.6f}"
        )
    lines.append("---------------------------------------------------------------------")
    return lines


def _format_zmatrix_orientation(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        "                         Z-Matrix orientation:                         ",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for number, atom in enumerate(atoms, start=1):
        lines.append(
            f"{number:7d}{_atomic_number(atom.symbol):11d}{0:12d}"
            f"{atom.x:16.6f}{atom.y:12.6f}{atom.z:12.6f}"
        )
    lines.append(" ---------------------------------------------------------------------")
    return lines


def _format_input_orientation(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        "                         Input orientation:",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for number, atom in enumerate(atoms, start=1):
        lines.append(
            f"{number:7d}{_atomic_number(atom.symbol):11d}{0:12d}"
            f"{atom.x:16.6f}{atom.y:12.6f}{atom.z:12.6f}"
        )
    lines.append(" ---------------------------------------------------------------------")
    return lines


def _format_stoichiometry(atoms: Sequence[AtomRecord]) -> str:
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1

    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
    if "H" in counts:
        ordered_symbols.append("H")
    ordered_symbols.extend(symbol for symbol in sorted(counts) if symbol not in {"C", "H"})

    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}"
        for symbol in ordered_symbols
    )


def _atom_distance(first: AtomRecord, second: AtomRecord) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def _vector_between(first: AtomRecord, second: AtomRecord) -> tuple[float, float, float]:
    return (first.x - second.x, first.y - second.y, first.z - second.z)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _bond_angle(first: AtomRecord, center: AtomRecord, third: AtomRecord) -> float:
    first_vector = _vector_between(first, center)
    third_vector = _vector_between(third, center)
    denominator = _norm(first_vector) * _norm(third_vector)
    if denominator < 1.0e-12:
        return 90.0
    cosine = max(-1.0, min(1.0, _dot(first_vector, third_vector) / denominator))
    return math.degrees(math.acos(cosine))


def _dihedral_angle(
    first: AtomRecord,
    second: AtomRecord,
    third: AtomRecord,
    fourth: AtomRecord,
) -> float:
    b0 = _vector_between(first, second)
    b1 = _vector_between(third, second)
    b2 = _vector_between(fourth, third)
    b1_norm = _norm(b1)
    if b1_norm < 1.0e-12:
        return 0.0
    b1_unit = (b1[0] / b1_norm, b1[1] / b1_norm, b1[2] / b1_norm)
    v = tuple(b0[index] - _dot(b0, b1_unit) * b1_unit[index] for index in range(3))
    w = tuple(b2[index] - _dot(b2, b1_unit) * b1_unit[index] for index in range(3))
    if _norm(v) < 1.0e-12 or _norm(w) < 1.0e-12:
        return 0.0
    x_value = _dot(v, w)
    y_value = _dot(_cross(b1_unit, v), w)
    return math.degrees(math.atan2(y_value, x_value))


def _format_distance_matrix(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = ["                    Distance matrix (angstroms):"]
    chunk_size = 5
    for start in range(0, len(atoms), chunk_size):
        stop = min(start + chunk_size, len(atoms))
        lines.append("             " + "".join(f"{column + 1:11d}" for column in range(start, stop)))
        for row in range(start, len(atoms)):
            end_column = min(stop, row + 1)
            if start >= end_column:
                continue
            distances = [
                _atom_distance(atoms[row], atoms[column])
                for column in range(start, end_column)
            ]
            lines.append(
                f"{row + 1:5d}  {atoms[row].symbol:<2s}"
                + "".join(f"{distance:11.6f}" for distance in distances)
            )
    return lines


def _format_irc_l202_block(atoms: Sequence[AtomRecord], next_link: int = 301) -> list[str]:
    lines = _format_input_orientation(atoms)
    lines.extend(_format_distance_matrix(atoms))
    lines.extend(
        [
            " Symmetry turned off by external request.",
            f" Stoichiometry    {_format_stoichiometry(atoms)}",
            " Framework group  C1[X]",
            f" Deg. of freedom {max(3 * len(atoms) - 6, 0):5d}",
            " Full point group                 C1      NOp   1",
            " Rotational constants (GHZ):           0.0000000           0.0000000           0.0000000",
            " Leave Link  202 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
            f" (Enter /usr/local/g16/l{next_link}.exe)",
        ]
    )
    return lines


def _format_opt_final_l202_block(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        " Leave Link  103 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l202.exe)",
    ]
    lines.extend(_format_input_orientation(atoms))
    lines.extend(_format_distance_matrix(atoms))
    lines.extend(
        [
            f" Stoichiometry    {_format_stoichiometry(atoms)}",
            " Framework group  C1[X]",
            f" Deg. of freedom {max(3 * len(atoms) - 6, 0):5d}",
            " Full point group                 C1      NOp   1",
        ]
    )
    lines.extend(_format_orientation(atoms))
    lines.extend(
        [
            " Rotational constants (GHZ):           0.0000000           0.0000000           0.0000000",
            " Leave Link  202 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
            " (Enter /usr/local/g16/l601.exe)",
        ]
    )
    return lines


def _format_irc_pre_scf_links() -> list[str]:
    return [
        " Standard basis: Gen (5D, 7F)",
        " Leave Link  301 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l502.exe)",
    ]


def _format_irc_enter_l123() -> list[str]:
    return [
        " Leave Link  502 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l716.exe)",
        " Internal  Forces:  Max     0.000000000 RMS     0.000000000",
        " Leave Link  716 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l123.exe)",
    ]


def _format_irc_current_structure(atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        "                   CURRENT STRUCTURE",
        "               Cartesian Coordinates (Ang):                ",
        " ---------------------------------------------------------------------",
        " Center     Atomic                     Coordinates (Angstroms)",
        " Number     Number                        X           Y           Z",
        " ---------------------------------------------------------------------",
    ]
    for number, atom in enumerate(atoms, start=1):
        lines.append(
            f"{number:7d}{_atomic_number(atom.symbol):11d}"
            f"{atom.x:28.6f}{atom.y:12.6f}{atom.z:12.6f}"
        )
    lines.append(" ---------------------------------------------------------------------")
    return lines


def _electron_counts(atoms: Sequence[AtomRecord], charge: int, multiplicity: int) -> tuple[int, int, int]:
    total = sum(_atomic_number(atom.symbol) for atom in atoms) - charge
    unpaired = max(multiplicity - 1, 0)
    alpha = (total + unpaired) // 2
    beta = total - alpha
    return total, alpha, beta


def _format_convergence(step_number: int, convergence: ConvergenceRecord | None) -> list[str]:
    line = " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad"
    lines = [
        "",
        line,
        f" Step number {step_number:3d}",
        "         Item               Value     Threshold  Converged?",
    ]

    def add(label: str, item: tuple[float, float, bool] | None) -> None:
        if item is None:
            value, threshold, converged = 0.0, 0.0, True
        else:
            value, threshold, converged = item
        lines.append(f" {label:<22s}{value:12.6f} {threshold:12.6f}     {'YES' if converged else 'NO'}")

    add("Maximum Force", convergence.maximum_force if convergence else None)
    add("RMS     Force", convergence.rms_force if convergence else None)
    add("Maximum Displacement", convergence.maximum_displacement if convergence else None)
    add("RMS     Displacement", convergence.rms_displacement if convergence else None)
    lines.append(line)
    return lines


def _format_ts_completion() -> list[str]:
    return [
        " Optimization completed.",
        "    -- Stationary point found.",
        " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
    ]


def _render_ts_minimal_optimization(
    parsed: MapleOutput,
    energy_unit: str,
    alpha_electrons: int,
    beta_electrons: int,
) -> str:
    # GaussView accepts this small optimization-shaped subset for TS PRFO/Dimer
    # trajectories.  The more Gaussian-like route/link/footer scaffolding used
    # for normal optimizations trips GaussView's TS PreprocessFile() path on
    # these converted logs even when the file ends with Normal termination.
    lines = [
        f"! This file was generated by maple2gaussian version {_VERSION}",
        "! MAPLE transition-state optimization converted to a minimal Gaussian-like trajectory",
        f"! Source program: MAPLE task={parsed.task or 'unknown'} method={parsed.method or 'unknown'} model={parsed.model or 'unknown'}",
        "",
        f" Charge = {parsed.charge:4d} Multiplicity = {parsed.multiplicity}",
        "0 basis functions",
        f"{alpha_electrons} alpha electrons",
        f"{beta_electrons} beta electrons",
        "GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
        "GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
    ]

    for frame_index, frame in enumerate(parsed.frames, start=1):
        lines.extend(_format_orientation(frame.atoms))
        energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
        if energy is not None:
            lines.extend(["", f" SCF Done:  E(theory) = {energy:.9f}"])
        lines.extend(_format_convergence(frame_index, frame.convergence))
        if frame_index == len(parsed.frames):
            lines.extend(["", " Optimization completed.", "    -- Stationary point found."])
        lines.append("")

    lines.extend([" Normal termination of Gaussian", ""])
    return "\n".join(lines)


def _render_scan_minimal(
    parsed: MapleOutput,
    energy_unit: str,
    alpha_electrons: int,
    beta_electrons: int,
) -> str:
    route = "#P HF/STO-3G Opt=Z-Matrix NoSymm"
    lines = _format_scan_preamble(parsed, route)
    lines.extend(_format_scan_symbolic_zmatrix(parsed))
    lines.append("")
    lines.extend(_format_scan_l101_to_l103(parsed))
    lines.extend(_format_scan_initial_parameters(parsed))

    total_frames = len(parsed.frames)
    for frame_index, frame in enumerate(parsed.frames, start=1):
        lines.extend(_format_scan_l202_block(parsed, frame.atoms))
        energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
        if energy is not None:
            lines.extend(["", f" SCF Done:  E(RHF) = {energy:.9f}     A.U. after    1 cycles"])
        lines.extend(_format_scan_post_scf_links())
        lines.extend(_format_scan_step(frame_index, total_frames, frame))
        lines.extend(_format_scan_optimized_parameters(frame, include_gradgrad=frame_index < total_frames))
        if frame_index < total_frames:
            lines.append("")

    lines.extend(_format_optimized_scan_summary(parsed, energy_unit))
    if parsed.frames:
        lines.extend(_format_scan_l202_block(parsed, parsed.frames[-1].atoms, next_link=601))
    lines.extend(_format_scan_l9999_entry())
    lines.extend(_format_scan_archive_record(parsed, route, energy_unit))
    lines.extend(
        [
            "",
            " MAPLE CONVERTED SCAN OUTPUT.",
            " Job cpu time:       0 days  0 hours  0 minutes  0.0 seconds.",
            " Elapsed time:       0 days  0 hours  0 minutes  0.0 seconds.",
            " File lengths (MBytes):  RWF=      1 Int=      0 D2E=      0 Chk=      1 Scr=      1",
            " Normal termination of Gaussian 16 at Sat May 23 00:00:00 2026.",
            "",
        ]
    )
    return "\n".join(lines)


def _irc_path_number(reaction_coordinate: float) -> int:
    return 2 if reaction_coordinate < 0 else 1


def _irc_path_name(path_number: int) -> str:
    return "REVERSE" if path_number == 2 else "FORWARD"


def _irc_point_number(reaction_coordinate: float) -> int:
    if abs(reaction_coordinate) < 1.0e-8:
        return 0
    return int(round(abs(reaction_coordinate) / 0.1))


def _format_irc_point(
    reaction_coordinate: float,
    atoms: Sequence[AtomRecord],
    next_reaction_coordinate: float | None,
    completed_point_count: int,
    next_display_point_number: int | None,
) -> list[str]:
    path_number = _irc_path_number(reaction_coordinate)
    path_point_number = _irc_point_number(reaction_coordinate)
    change_coordinate = 0.0 if path_point_number == 0 else min(abs(reaction_coordinate), 0.1)

    lines = [
        "",
        " IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC",
        f" Pt {completed_point_count:2d} Step number   1 out of a maximum of  20",
        " Using modified Bulirsch-Stoer corrector integration.",
        " SUMMARY OF CORRECTOR INTEGRATION:",
        "   Predictor End Point Energy =     0.000000",
        "   Old End Point Energy       =     0.000000",
        "   Corrected End Point Energy =     0.000000",
        "   Predictor End-Start Dist.  =     0.100000",
        "   Old End-Start Dist.        =     0.100000",
        "   New End-Start Dist.        =     0.100000",
        "   New End-Old End Dist.      =     0.000000",
        " CORRECTOR INTEGRATION CONVERGENCE:",
        "   Recorrection delta-x convergence threshold:    0.010000",
        "   Delta-x Convergence Met",
        f" Point Number: {path_point_number:3d}          Path Number:   {path_number}",
    ]
    lines.extend(_format_irc_current_structure(atoms))
    lines.extend(
        [
            "   CHANGE IN THE REACTION COORDINATE =    " + f"{change_coordinate:8.5f}",
            "   NET REACTION COORDINATE UP TO THIS POINT =    " + f"{abs(reaction_coordinate):8.5f}",
            f"  # OF POINTS ALONG THE PATH = {completed_point_count:3d}",
            "  # OF STEPS =   1",
            "",
        ]
    )

    if next_reaction_coordinate is None:
        lines.extend(
            [
                " Maximum number of steps reached.",
                f" Calculation of {_irc_path_name(path_number)} path complete.",
            ]
        )
        return lines

    next_path_number = _irc_path_number(next_reaction_coordinate)
    if next_path_number != path_number:
        lines.extend(
            [
                " Maximum number of steps reached.",
                f" Calculation of {_irc_path_name(path_number)} path complete.",
                f" Beginning calculation of the {_irc_path_name(next_path_number)} path.",
                " ",
            ]
        )

    lines.extend(
        [
            " Calculating another point on the path.",
            f" Point Number {next_display_point_number or path_point_number + 1:2d} in {_irc_path_name(next_path_number)} path direction.",
            " Using LQA Reaction Path Following.",
            " IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC",
            " Leave Link  123 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
            " (Enter /usr/local/g16/l202.exe)",
        ]
    )
    return lines


def _reaction_coordinates(frames: Sequence[Frame]) -> list[float]:
    if not frames:
        return []
    max_index = _irc_transition_state_index(frames)
    return [(index - max_index) * 0.1 for index in range(len(frames))]


def _format_irc_summary(parsed: MapleOutput, energy_unit: str) -> list[str]:
    coords = _reaction_coordinates(parsed.frames)
    energies = [_frame_energy_hartree(frame, energy_unit, parsed.model) for frame in parsed.frames]
    reference = max((energy for energy in energies if energy is not None), default=0.0)
    lines = [
        "",
        " Reaction path calculation complete.",
        " ",
        f" Energies reported relative to the TS energy of {reference:17.6f}",
        " --------------------------------------------------------------------------",
        "    Summary of reaction path following",
        " --------------------------------------------------------------------------",
        "                        Energy    RxCoord",
    ]
    for index, (energy, coord) in enumerate(zip(energies, coords), start=1):
        relative_energy = (energy - reference) if energy is not None else 0.0
        lines.append(f" {index:3d} {relative_energy:26.5f} {coord:9.5f}")
    lines.extend(
        [
            " --------------------------------------------------------------------------",
            " ",
            f" Total number of points: {max(len(parsed.frames) - 1, 0):20d}",
            f" Total number of gradient calculations: {max(len(parsed.frames) - 1, 0):5d}",
            " Total number of Hessian calculations:      0",
            " IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC",
        ]
    )
    return lines


def _format_irc_preamble(parsed: MapleOutput) -> list[str]:
    max_points = _irc_max_points_per_path(parsed.frames)
    return [
        " IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC",
        " ------------------------------------------------------------------------",
        " INPUT DATA FOR L123",
        " ------------------------------------------------------------------------",
        " GENERAL PARAMETERS:",
        " Follow reaction path in both directions.",
        f" Maximum points per path      = {max_points:3d}",
        " Step size                    =   0.100 bohr",
        " Integration scheme           = HPC",
        " Initial Hessian              = ReadFC from chk",
        " Hessian evaluation           = All updating",
        " Hessian updating method      = Bofill",
        " ------------------------------------------------------------------------",
        "         ******** Start new reaction path calculation ********",
        " Current Structure is TS -> form Hessian eigenvectors.",
        "                           Diagonalizing Hessian.",
        " Point Number:   0          Path Number:   1",
        " Calculating another point on the path.",
        " Point Number  1 in FORWARD path direction.",
        " Using LQA Reaction Path Following.",
        " IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC",
        " Leave Link  123 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l202.exe)",
        "",
    ]


def _format_fortran_float(value: float) -> str:
    return f"{value: 19.12E}".replace("E", "D")


def _format_md_preamble(parsed: MapleOutput) -> list[str]:
    step_size = 0.0
    if len(parsed.md_thermo) >= 2:
        step_size = parsed.md_thermo[1].time_fs - parsed.md_thermo[0].time_fs
    elif parsed.md_thermo:
        step_size = parsed.md_thermo[0].time_fs

    return [
        " TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ",
        "-------------------------------------------------------------------",
        "INPUT DATA FOR L121",
        "-------------------------------------------------------------------",
        "General parameters:",
        f"Max. points for each Traj.   = {len(parsed.frames):6d}",
        "Total Number of Trajectories =      1",
        f"Trajectory Step Size          = {step_size:6.3f} femtosec",
        "Sampling parameters:",
        "Vib Energy Sampling Option   = MAPLE trajectory",
        "Rot Energy Sampling Option   = MAPLE trajectory",
        "---------------------------------------------------------------------",
        "",
    ]


def _format_md_step(frame_index: int, frame: Frame, thermo: MdThermoRecord | None, energy_unit: str, model: str | None) -> list[str]:
    time_fs = thermo.time_fs if thermo else float(frame_index - 1)
    kinetic = thermo.kinetic_hartree if thermo else 0.0
    potential = thermo.potential_hartree if thermo else (_frame_energy_hartree(frame, energy_unit, model) or 0.0)
    total = thermo.total_hartree if thermo else potential + kinetic

    lines = [
        "",
        "TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ",
        "",
        f"ADMP step {frame_index - 1:6d}",
        "Frobenius norm of [F,P] =     0.000000D+00",
        "",
        f"Summary information for step {frame_index - 1:6d}",
        f"Time (fs) {time_fs:12.6f}",
        f"EKin       = {kinetic:14.7f}; EPot   = {potential:14.7f}; ETot   = {total:14.7f}",
        f"ETot-EKinP = {total:14.7f}",
        f"Total energy {total:16.9E} A.U.".replace("E", "D"),
        "Total angular momentum   0.000000D+00 h-bar",
        "Cartesian coordinates:",
    ]
    for index, atom in enumerate(frame.atoms, start=1):
        lines.append(
            f"I={index:5d} X={_format_fortran_float(atom.x * BOHR_PER_ANGSTROM)}"
            f" Y={_format_fortran_float(atom.y * BOHR_PER_ANGSTROM)}"
            f" Z={_format_fortran_float(atom.z * BOHR_PER_ANGSTROM)}"
        )
    lines.append("MW Cartesian velocity:")
    for index, _atom in enumerate(frame.atoms, start=1):
        lines.append(
            f"I={index:5d} X={_format_fortran_float(0.0)}"
            f" Y={_format_fortran_float(0.0)}"
            f" Z={_format_fortran_float(0.0)}"
        )
    lines.append("TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ-TRJ")
    return lines


def _format_md_summary(parsed: MapleOutput, energy_unit: str) -> list[str]:
    lines = [
        "",
        "Trajectory summary for trajectory     1",
        f"Energy/gradient evaluations          {len(parsed.frames):4d}",
        "Hessian evaluations                     0",
        "Trajectory summary",
        "Time (fs)  Kinetic (au)  Potent (au)   Delta E (au)  Delta A (h-bar)",
    ]

    if parsed.md_thermo:
        initial_total = parsed.md_thermo[0].total_hartree
        for record in parsed.md_thermo[: len(parsed.frames)]:
            lines.append(
                f"{record.time_fs:9.6f} {record.kinetic_hartree:11.7f}"
                f" {record.potential_hartree:13.7f}"
                f" {record.total_hartree - initial_total:12.7f}"
                "     0.0000000000000000"
            )
        return lines

    initial_energy = _frame_energy_hartree(parsed.frames[0], energy_unit, parsed.model) or 0.0
    for index, frame in enumerate(parsed.frames):
        potential = _frame_energy_hartree(frame, energy_unit, parsed.model) or initial_energy
        lines.append(
            f"{index:9.6f} {0.0:11.7f} {potential:13.7f}"
            f" {potential - initial_energy:12.7f}     0.0000000000000000"
        )
    return lines


def _convergence_all_yes(convergence: ConvergenceRecord | None) -> bool:
    if convergence is None:
        return False

    items = (
        convergence.maximum_force,
        convergence.rms_force,
        convergence.maximum_displacement,
        convergence.rms_displacement,
    )
    return all(item is not None and item[2] for item in items)


def _mode_components(mode_values: Sequence[float], atom_count: int) -> list[tuple[float, float, float]]:
    components: list[tuple[float, float, float]] = []
    for atom_index in range(atom_count):
        base = atom_index * 3
        if base + 2 < len(mode_values):
            components.append((mode_values[base], mode_values[base + 1], mode_values[base + 2]))
        else:
            components.append((0.0, 0.0, 0.0))
    return components


def _expected_vibrational_mode_count(atom_count: int) -> int:
    if atom_count <= 1:
        return 0
    if atom_count == 2:
        return 1
    return 3 * atom_count - 6


def _frequency_entries_for_gaussian(
    parsed: MapleOutput,
    atom_count: int,
) -> list[tuple[int, float]]:
    entries = list(enumerate(parsed.frequencies))
    expected_count = _expected_vibrational_mode_count(atom_count)
    if not entries:
        return entries

    vibrational_entries = [
        entry
        for entry in entries
        if abs(entry[1]) >= 5.0
    ]
    if vibrational_entries:
        entries = vibrational_entries

    if expected_count > 0:
        entries = entries[:expected_count]

    return entries


def _format_frequency_blocks(parsed: MapleOutput) -> list[str]:
    if not parsed.frequencies:
        return []

    atoms = parsed.frames[-1].atoms if parsed.frames else parsed.initial_atoms
    if not atoms:
        return []

    frequency_entries = _frequency_entries_for_gaussian(parsed, len(atoms))

    lines = [
        "",
        " Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
        " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
        " incident light, reduced masses (AMU), force constants (mDyne/A),",
        " and normal coordinates:",
    ]

    for start in range(0, len(frequency_entries), 3):
        entries = frequency_entries[start : start + 3]
        freqs = [entry[1] for entry in entries]
        mode_numbers = list(range(start + 1, start + 1 + len(entries)))
        reduced_masses = [1.0] * len(freqs)
        force_constants = [
            (abs(freq) / 5140.48) ** 2 * reduced_mass
            for freq, reduced_mass in zip(freqs, reduced_masses)
        ]
        zero_values = [0.0] * len(freqs)

        lines.append("".join(f"{number:23d}" for number in mode_numbers))
        lines.append("".join(f"{'A':>23s}" for _ in freqs))

        def gaussian_values(label: str, values: Sequence[float]) -> str:
            if not values:
                return label
            return label + f"{values[0]:12.4f}" + "".join(f"{value:23.4f}" for value in values[1:])

        lines.append(gaussian_values(" Frequencies --", freqs))
        lines.append(gaussian_values(" Red. masses --", reduced_masses))
        lines.append(gaussian_values(" Frc consts  --", force_constants))
        lines.append(gaussian_values(" IR Inten    --", zero_values))
        lines.append(gaussian_values(" Raman Activ --", zero_values))
        lines.append(gaussian_values(" Depolar (P) --", zero_values))
        lines.append(gaussian_values(" Depolar (U) --", zero_values))
        lines.append("  Atom  AN      X      Y      Z" + "        X      Y      Z" * (len(freqs) - 1))

        mode_components = [
            _mode_components(parsed.normal_modes.get(original_index, []), len(atoms))
            for original_index, _frequency in entries
        ]
        for atom_index, atom in enumerate(atoms):
            row = f"{atom_index + 1:6d}{_atomic_number(atom.symbol):4d}"
            for components in mode_components:
                x, y, z = components[atom_index]
                row += f"{x:8.2f}{y:7.2f}{z:7.2f}"
            lines.append(row)

    return lines


def _format_thermochemistry(parsed: MapleOutput) -> list[str]:
    if not parsed.thermo_kcal:
        return []

    def hartree(name: str) -> float:
        return parsed.thermo_kcal[name] / KCAL_PER_HARTREE

    lines = [
        "",
        " -------------------",
        " - Thermochemistry -",
        " -------------------",
        " Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.",
    ]
    if "zero_point" in parsed.thermo_kcal:
        lines.append(f" Zero-point correction= {hartree('zero_point'):12.6f} (Hartree/Particle)")
    if "thermal_energy" in parsed.thermo_kcal:
        lines.append(f" Thermal correction to Energy= {hartree('thermal_energy'):12.6f}")
    if "enthalpy" in parsed.thermo_kcal:
        lines.append(f" Thermal correction to Enthalpy= {hartree('enthalpy'):12.6f}")
    if "gibbs" in parsed.thermo_kcal:
        lines.append(f" Thermal correction to Gibbs Free Energy= {hartree('gibbs'):12.6f}")
    return lines


def _scan_atoms(parsed: MapleOutput) -> Sequence[AtomRecord]:
    if parsed.frames:
        return parsed.frames[0].atoms
    return parsed.initial_atoms


def _scan_length_variable_map(parsed: MapleOutput) -> dict[int, tuple[int, str, float, int, float]]:
    frames = [frame for frame in parsed.frames if frame.energy is not None]
    first_values = frames[0].scan_values if frames else ()
    mapping: dict[int, tuple[int, str, float, int, float]] = {}
    for index, directive in enumerate(parsed.scan_directives, start=1):
        if len(directive.atoms) != 2:
            continue
        target = max(directive.atoms)
        reference = min(directive.atoms)
        if target <= 1 or reference == target:
            continue
        initial_value = first_values[index - 1] if index - 1 < len(first_values) else 0.0
        mapping[target] = (
            reference,
            _scan_definition_label(index),
            initial_value,
            directive.steps,
            directive.step,
        )

    if not mapping and first_values and len(_scan_atoms(parsed)) >= 2:
        for index, initial_value in enumerate(first_values, start=1):
            target = min(index + 1, len(_scan_atoms(parsed)))
            if target <= 1:
                continue
            if len(frames) > 1 and index - 1 < len(frames[1].scan_values):
                step = frames[1].scan_values[index - 1] - initial_value
            else:
                step = 0.0
            mapping[target] = (
                target - 1,
                _scan_definition_label(index),
                initial_value,
                max(len(frames) - 1, 0),
                step,
            )
    return mapping


def _previous_reference(target: int, excluded: set[int], fallback: int) -> int:
    for candidate in range(target - 1, 0, -1):
        if candidate not in excluded:
            return candidate
    return fallback


def _zmatrix_variable_records(parsed: MapleOutput) -> list[tuple[str, float, tuple[int, float] | None]]:
    atoms = _scan_atoms(parsed)
    scan_lengths = _scan_length_variable_map(parsed)
    records: list[tuple[str, float, tuple[int, float] | None]] = []

    def add_length(index: int, reference: int, name: str, scan: tuple[int, float] | None = None) -> None:
        value = scan_lengths[index][2] if index in scan_lengths and name == scan_lengths[index][1] else _atom_distance(atoms[index - 1], atoms[reference - 1])
        records.append((name, value, scan))

    if len(atoms) >= 2:
        scan_info = scan_lengths.get(2)
        reference = scan_info[0] if scan_info else 1
        add_length(2, reference, scan_info[1] if scan_info else "ZR2", (scan_info[3], scan_info[4]) if scan_info else None)
    if len(atoms) >= 3:
        scan_info = scan_lengths.get(3)
        length_reference = scan_info[0] if scan_info else 2
        angle_reference = _previous_reference(3, {length_reference}, 1)
        add_length(3, length_reference, scan_info[1] if scan_info else "ZR3", (scan_info[3], scan_info[4]) if scan_info else None)
        records.append(("ZA3", _bond_angle(atoms[2], atoms[length_reference - 1], atoms[angle_reference - 1]), None))
    for index in range(4, len(atoms) + 1):
        scan_info = scan_lengths.get(index)
        length_reference = scan_info[0] if scan_info else index - 1
        angle_reference = _previous_reference(index, {length_reference}, index - 2)
        dihedral_reference = _previous_reference(index, {length_reference, angle_reference}, index - 3)
        add_length(index, length_reference, scan_info[1] if scan_info else f"ZR{index}", (scan_info[3], scan_info[4]) if scan_info else None)
        records.append((f"ZA{index}", _bond_angle(atoms[index - 1], atoms[length_reference - 1], atoms[angle_reference - 1]), None))
        records.append((f"ZD{index}", _dihedral_angle(atoms[index - 1], atoms[length_reference - 1], atoms[angle_reference - 1], atoms[dihedral_reference - 1]), None))
    return records


def _zmatrix_body_lines(parsed: MapleOutput) -> list[str]:
    atoms = _scan_atoms(parsed)
    if not atoms:
        return []
    scan_lengths = _scan_length_variable_map(parsed)
    lines = [f" {atoms[0].symbol}"]
    if len(atoms) >= 2:
        scan_info = scan_lengths.get(2)
        reference = scan_info[0] if scan_info else 1
        length_name = scan_info[1] if scan_info else "ZR2"
        lines.append(f" {atoms[1].symbol:<2s}                 {reference:d}    {length_name}")
    if len(atoms) >= 3:
        scan_info = scan_lengths.get(3)
        length_reference = scan_info[0] if scan_info else 2
        angle_reference = _previous_reference(3, {length_reference}, 1)
        length_name = scan_info[1] if scan_info else "ZR3"
        lines.append(f" {atoms[2].symbol:<2s}                 {length_reference:d}    {length_name:<7s}  {angle_reference:d}    ZA3")
    for index in range(4, len(atoms) + 1):
        atom = atoms[index - 1]
        scan_info = scan_lengths.get(index)
        length_reference = scan_info[0] if scan_info else index - 1
        angle_reference = _previous_reference(index, {length_reference}, index - 2)
        dihedral_reference = _previous_reference(index, {length_reference, angle_reference}, index - 3)
        length_name = scan_info[1] if scan_info else f"ZR{index}"
        lines.append(
            f" {atom.symbol:<2s}                 {length_reference:d}    {length_name:<7s}"
            f"  {angle_reference:d}    ZA{index:<4d}     {dihedral_reference:d}    ZD{index:<4d}     0"
        )
    return lines


def _zmatrix_archive_fields(parsed: MapleOutput) -> list[str]:
    atoms = _scan_atoms(parsed)
    if not atoms:
        return []
    scan_lengths = _scan_length_variable_map(parsed)
    fields = [atoms[0].symbol]
    if len(atoms) >= 2:
        scan_info = scan_lengths.get(2)
        reference = scan_info[0] if scan_info else 1
        fields.append(f"{atoms[1].symbol},{reference:d},{scan_info[1] if scan_info else 'ZR2'}")
    if len(atoms) >= 3:
        scan_info = scan_lengths.get(3)
        length_reference = scan_info[0] if scan_info else 2
        angle_reference = _previous_reference(3, {length_reference}, 1)
        length_name = scan_info[1] if scan_info else "ZR3"
        fields.append(f"{atoms[2].symbol},{length_reference:d},{length_name},{angle_reference:d},ZA3")
    for index in range(4, len(atoms) + 1):
        atom = atoms[index - 1]
        scan_info = scan_lengths.get(index)
        length_reference = scan_info[0] if scan_info else index - 1
        angle_reference = _previous_reference(index, {length_reference}, index - 2)
        dihedral_reference = _previous_reference(index, {length_reference, angle_reference}, index - 3)
        length_name = scan_info[1] if scan_info else f"ZR{index:d}"
        fields.append(
            f"{atom.symbol},{length_reference:d},{length_name},{angle_reference:d},ZA{index:d},{dihedral_reference:d},ZD{index:d},0"
        )
    return fields


def _zmatrix_numeric_rows(parsed: MapleOutput, atoms: Sequence[AtomRecord]) -> list[tuple[int, int | None, float | None, int | None, float | None, int | None, float | None]]:
    scan_lengths = _scan_length_variable_map(parsed)
    rows: list[tuple[int, int | None, float | None, int | None, float | None, int | None, float | None]] = []
    if not atoms:
        return rows
    rows.append((1, None, None, None, None, None, None))
    if len(atoms) >= 2:
        scan_info = scan_lengths.get(2)
        reference = scan_info[0] if scan_info else 1
        rows.append((2, reference, _atom_distance(atoms[1], atoms[reference - 1]), None, None, None, None))
    if len(atoms) >= 3:
        scan_info = scan_lengths.get(3)
        length_reference = scan_info[0] if scan_info else 2
        angle_reference = _previous_reference(3, {length_reference}, 1)
        rows.append(
            (
                3,
                length_reference,
                _atom_distance(atoms[2], atoms[length_reference - 1]),
                angle_reference,
                _bond_angle(atoms[2], atoms[length_reference - 1], atoms[angle_reference - 1]),
                None,
                None,
            )
        )
    for index in range(4, len(atoms) + 1):
        scan_info = scan_lengths.get(index)
        length_reference = scan_info[0] if scan_info else index - 1
        angle_reference = _previous_reference(index, {length_reference}, index - 2)
        dihedral_reference = _previous_reference(index, {length_reference, angle_reference}, index - 3)
        rows.append(
            (
                index,
                length_reference,
                _atom_distance(atoms[index - 1], atoms[length_reference - 1]),
                angle_reference,
                _bond_angle(atoms[index - 1], atoms[length_reference - 1], atoms[angle_reference - 1]),
                dihedral_reference,
                _dihedral_angle(atoms[index - 1], atoms[length_reference - 1], atoms[angle_reference - 1], atoms[dihedral_reference - 1]),
            )
        )
    return rows


def _format_scan_preamble(parsed: MapleOutput, route: str) -> list[str]:
    return [
        " Entering Gaussian System, Link 0=g16",
        f" Input={parsed.source.name}",
        f" Output={parsed.source.with_suffix('.log').name}",
        " Initial command:",
        f" /usr/local/g16/l1.exe \"{parsed.source.with_suffix('.gjf').name}\" -scrdir=\"/tmp\"",
        " Default is to use a total of   1 processors:",
        "                                1 via shared-memory",
        " Entering Link 1 = /usr/local/g16/l1.exe PID=         1.",
        " ******************************************",
        " Gaussian 16:  ES64L-G16RevA.03 25-Dec-2016",
        "                23-May-2026",
        " ******************************************",
        " --------------------------------",
        f" {route}",
        " --------------------------------",
        " 1/10=7,18=40,38=1/1,3;",
        " 2/12=2,15=1,17=6,18=5,29=3,40=1/2;",
        " 3/6=3,11=9,25=1,30=1,71=1/1,2,3;",
        " 4//1;",
        " 5/5=2,38=5/2;",
        " 6/7=2,8=2,9=2,10=2,28=1/1;",
        " 7/29=1,30=1/1,2,3,16;",
        " 1/10=7,18=40/3(2);",
        " 2/15=1,29=3/2;",
        " 99//99;",
        " 2/15=1,29=3/2;",
        " 3/6=3,11=9,25=1,30=1,71=1/1,2,3;",
        " 4/5=5,16=3,69=1/1;",
        " 5/5=2,38=5/2;",
        " 7/30=1/1,2,3,16;",
        " 1/18=40/3(-5);",
        " 2/15=1,29=3/2;",
        " 6/7=2,8=2,9=2,10=2,19=2,28=1/1;",
        " 99/9=1/99;",
        " Leave Link    1 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l101.exe)",
        " ----------------------------------------",
        f"MAPLE scan converted from {parsed.source.name}",
        f"maple2gaussian version {_VERSION}; task={parsed.task or 'unknown'} method={parsed.method or 'unknown'} model={parsed.model or 'unknown'}",
        " ----------------------------------------",
    ]


def _format_scan_l101_to_l103(parsed: MapleOutput) -> list[str]:
    atom_count = len(_scan_atoms(parsed))
    return [
        " ITRead=  0  0",
        " MicOpt= -1 -1",
        f" NAtoms={atom_count:7d} NQM={atom_count:9d} NQMF=       0 NMMI=      0 NMMIF=      0",
        "               NMic=       0 NMicF=      0.",
        " Leave Link  101 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l103.exe)",
        "",
    ]


def _format_scan_zmatrix_table(parsed: MapleOutput, atoms: Sequence[AtomRecord]) -> list[str]:
    lines = [
        " ---------------------------------------------------------------------------------------------------",
        "                           Z-MATRIX (ANGSTROMS AND DEGREES)",
        "   CD    Cent   Atom    N1       Length/X        N2       Alpha/Y        N3        Beta/Z          J",
        " ---------------------------------------------------------------------------------------------------",
    ]
    variable_index = 1
    for row in _zmatrix_numeric_rows(parsed, atoms):
        atom_index, length_reference, length, angle_reference, angle, dihedral_reference, dihedral = row
        atom = atoms[atom_index - 1]
        line = f"{atom_index:7d}{atom_index:7d}  {atom.symbol:<2s}"
        if length_reference is not None and length is not None:
            line += f"{length_reference:9d}{length:11.6f}({variable_index:6d})"
            variable_index += 1
        if angle_reference is not None and angle is not None:
            line += f"{angle_reference:8d}{angle:10.3f}({variable_index:6d})"
            variable_index += 1
        if dihedral_reference is not None and dihedral is not None:
            line += f"{dihedral_reference:8d}{dihedral:10.3f}({variable_index:6d})      0"
            variable_index += 1
        lines.append(line)
    lines.append(" ---------------------------------------------------------------------------------------------------")
    return lines


def _format_scan_l202_block(parsed: MapleOutput, atoms: Sequence[AtomRecord], next_link: int = 502) -> list[str]:
    lines = [
        " Leave Link  103 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l202.exe)",
    ]
    lines.extend(_format_scan_zmatrix_table(parsed, atoms))
    lines.extend(_format_zmatrix_orientation(atoms))
    lines.extend(
        [
            " Symmetry turned off by external request.",
            f" Stoichiometry    {_format_stoichiometry(atoms)}",
            " Framework group  C1[X]",
            f" Deg. of freedom {max(3 * len(atoms) - 6, 0):5d}",
            " Full point group                 C1      NOp   1",
            " Rotational constants (GHZ):           0.0000000           0.0000000           0.0000000",
            " Leave Link  202 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
            f" (Enter /usr/local/g16/l{next_link}.exe)",
        ]
    )
    return lines


def _format_scan_post_scf_links() -> list[str]:
    return [
        " Leave Link  502 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l716.exe)",
        " Internal  Forces:  Max     0.000000000 RMS     0.000000000",
        " Leave Link  716 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l103.exe)",
    ]


def _format_scan_l9999_entry() -> list[str]:
    return [
        " Copying SCF densities to generalized density rwf, IOpCl= 0 IROHF=0.",
        " Leave Link  601 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
        " (Enter /usr/local/g16/l9999.exe)",
    ]


def _scan_value_count(parsed: MapleOutput) -> int:
    return max(len(parsed.scan_directives), max((len(frame.scan_values) for frame in parsed.frames), default=0))


def _scan_values_for_archive(parsed: MapleOutput) -> list[str]:
    return [
        f"{name}={value:.6f},s,{scan[0]},{scan[1]:.6f}"
        for name, value, scan in _zmatrix_variable_records(parsed)
        if scan is not None
    ]


def _format_scan_symbolic_zmatrix(parsed: MapleOutput) -> list[str]:
    atoms = _scan_atoms(parsed)
    if not atoms:
        return []
    lines = [
        " Symbolic Z-matrix:",
        f" Charge = {parsed.charge:2d} Multiplicity = {parsed.multiplicity}",
    ]
    lines.extend(_zmatrix_body_lines(parsed))
    lines.append("      Variables:")
    for name, value, scan in _zmatrix_variable_records(parsed):
        if scan is None:
            lines.append(f" {name:<8s} {value:14.8f}")
        else:
            lines.append(f" {name:<8s} {value:14.8f} Scan {scan[0]:5d} {scan[1]:9.4f}")
    return lines


def _format_scan_variable_table(parsed: MapleOutput) -> list[str]:
    value_count = max((len(frame.scan_values) for frame in parsed.frames), default=0)
    count = max(len(parsed.scan_directives), value_count)
    if count == 0:
        return []

    first_values = parsed.frames[0].scan_values if parsed.frames else ()
    lines = [
        " Variable       Value     No. Steps Step-Size",
        " -------- ----------- --------- ---------",
    ]
    for index in range(1, count + 1):
        directive = parsed.scan_directives[index - 1] if index - 1 < len(parsed.scan_directives) else None
        value = first_values[index - 1] if index - 1 < len(first_values) else 0.0
        step_count = directive.steps if directive else max(len(parsed.frames) - 1, 0)
        if directive:
            step_size = directive.step
        elif len(parsed.frames) > 1 and index - 1 < len(parsed.frames[1].scan_values):
            step_size = parsed.frames[1].scan_values[index - 1] - value
        else:
            step_size = 0.0
        lines.append(f" {index:8d} {value:11.6f} {step_count:9d} {step_size:9.4f}")
    total_points = len([frame for frame in parsed.frames if frame.energy is not None]) or len(parsed.frames)
    lines.append(f" A total of {total_points:d} points will be computed.")
    return lines


def _format_scan_summary(parsed: MapleOutput, energy_unit: str) -> list[str]:
    frames = [frame for frame in parsed.frames if frame.energy is not None]
    if not frames:
        return []

    value_count = max((len(frame.scan_values) for frame in frames), default=0)
    value_labels = [_scan_definition_label(index) for index in range(1, value_count + 1)]
    header = " N"
    separator = " ----"
    for label in value_labels:
        header += f"  {label:>9s}"
        separator += "  ---------"
    header += "          SCF"
    separator += "  -----------"

    lines = [
        "",
        " Scan completed.",
        "",
        " Summary of the potential surface scan:",
        header,
        separator,
    ]
    for index, frame in enumerate(frames, start=1):
        energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
        if energy is None:
            continue
        row = f" {index:4d}"
        for value_index in range(value_count):
            value = frame.scan_values[value_index] if value_index < len(frame.scan_values) else float(index)
            row += f"  {value:9.4f}"
        row += f"  {energy:11.5f}"
        lines.append(row)
    lines.append(separator)
    return lines


def _format_optimized_scan_summary(parsed: MapleOutput, energy_unit: str) -> list[str]:
    frames = [frame for frame in parsed.frames if frame.energy is not None]
    if not frames:
        return []

    value_count = max((len(frame.scan_values) for frame in frames), default=0)
    value_labels = [_scan_definition_label(index) for index in range(1, value_count + 1)]
    lines = [" Summary of Optimized Potential Surface Scan"]
    columns_per_block = 5

    for start in range(0, len(frames), columns_per_block):
        block = frames[start : start + columns_per_block]
        indices = list(range(start + 1, start + len(block) + 1))
        lines.append(" " * 24 + "".join(f"{index:13d}" for index in indices))

        energy_row = "     EIGENVALUES -- "
        for frame in block:
            energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
            energy_row += f"{(energy if energy is not None else 0.0):13.5f}"
        lines.append(energy_row)

        for value_index, label in enumerate(value_labels):
            row = f" {label:>16s}"
            for absolute_index, frame in zip(indices, block):
                value = frame.scan_values[value_index] if value_index < len(frame.scan_values) else float(absolute_index)
                row += f"{value:13.5f}"
            lines.append(row)
    atom_index, displacement = _largest_scan_displacement(parsed)
    lines.append(f" Largest change from initial coordinates is atom {atom_index:4d} {displacement:11.3f} Angstoms.")
    lines.append(" GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad")
    lines.append("")
    return lines


def _largest_scan_displacement(parsed: MapleOutput) -> tuple[int, float]:
    if len(parsed.frames) < 2:
        return 1, 0.0
    first_atoms = parsed.frames[0].atoms
    last_atoms = parsed.frames[-1].atoms
    if not first_atoms or len(first_atoms) != len(last_atoms):
        return 1, 0.0
    best_index = 1
    best_displacement = 0.0
    for index, (first, last) in enumerate(zip(first_atoms, last_atoms), start=1):
        displacement = _atom_distance(first, last)
        if displacement > best_displacement:
            best_index = index
            best_displacement = displacement
    return best_index, best_displacement


def _wrap_archive_record(record: str) -> list[str]:
    return [" " + record[index : index + 78] for index in range(0, len(record), 78)]


def _wrap_archive_fields(fields: Sequence[str]) -> list[str]:
    return _wrap_archive_record("\\".join(fields))


def _format_scan_archive_record(parsed: MapleOutput, route: str, energy_unit: str) -> list[str]:
    frames = [frame for frame in parsed.frames if frame.energy is not None]
    atoms = frames[0].atoms if frames else parsed.frames[0].atoms if parsed.frames else parsed.initial_atoms
    if not atoms:
        return []

    formula = _format_stoichiometry(atoms)
    coordinate_fields = _zmatrix_archive_fields(parsed)
    variable_fields = [
        f"{name}={value:.8f}"
        for name, value, scan in _zmatrix_variable_records(parsed)
        if scan is None
    ]
    scan_fields = _scan_values_for_archive(parsed)
    energies = [
        _frame_energy_hartree(frame, energy_unit, parsed.model)
        for frame in frames
    ]
    energy_values = ",".join(f"{energy:.10g}" for energy in energies if energy is not None)
    rmsd_values = ",".join("0.000e+00" for _frame in frames)

    fields = [
        "1",
        "1",
        "GINC-MAPLE",
        "Scan",
        "RHF",
        "STO-3G",
        formula,
        "MAPLE",
        "23-May-2026",
        "1",
        "",
        route,
        "",
        f"MAPLE scan converted from {parsed.source.name}",
        "",
        f"{parsed.charge},{parsed.multiplicity}",
        *coordinate_fields,
        "",
        *variable_fields,
        *scan_fields,
        r"Version=ES64L-G16RevA.03",
    ]
    if energy_values:
        fields.append(f"HF={energy_values}")
    if rmsd_values:
        fields.append(f"RMSD={rmsd_values}")
    fields.append(r"PG=C01 [X]\\@")
    return _wrap_archive_fields(fields)


def _scan_definition_label(index: int) -> str:
    return f"R{index}"


def _scan_definition(directive: ScanDirective, index: int) -> str:
    if len(directive.atoms) == 2:
        return f"R({directive.atoms[0]},{directive.atoms[1]})"
    if len(directive.atoms) == 3:
        return f"A({directive.atoms[0]},{directive.atoms[1]},{directive.atoms[2]})"
    if len(directive.atoms) >= 4:
        return f"D({directive.atoms[0]},{directive.atoms[1]},{directive.atoms[2]},{directive.atoms[3]})"
    return f"X({index})"


def _format_scan_modredundant_section(parsed: MapleOutput) -> list[str]:
    if not parsed.scan_directives:
        return []

    lines = [" The following ModRedundant input section has been read:"]
    for directive in parsed.scan_directives:
        type_code = "B" if len(directive.atoms) == 2 else "A" if len(directive.atoms) == 3 else "D"
        atoms = " ".join(str(atom) for atom in directive.atoms)
        lines.append(f" {type_code} {atoms} S {directive.steps:4d} {directive.step:10.4f}")
    lines.append("")
    return lines


def _format_scan_initial_parameters(parsed: MapleOutput) -> list[str]:
    value_count = max((len(frame.scan_values) for frame in parsed.frames), default=0)
    if not parsed.scan_directives and value_count == 0:
        return []

    first_values = parsed.frames[0].scan_values if parsed.frames else ()
    lines = [
        " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
        " Berny optimization.",
        " Initialization pass.",
        "                       ----------------------------",
        "                       !    Initial Parameters    !",
        "                       ! (Angstroms and Degrees)  !",
        " ----------------------                            ----------------------",
        " !      Name          Value   Derivative information (Atomic Units)     !",
        " ------------------------------------------------------------------------",
    ]
    count = max(len(parsed.scan_directives), value_count)
    for index in range(1, count + 1):
        value = first_values[index - 1] if index - 1 < len(first_values) else 0.0
        lines.append(f" !       {_scan_definition_label(index):<8s}{value:10.5f}      Scan                                      !")
    lines.extend(
        [
            " ------------------------------------------------------------------------",
            " Trust Radius=3.00D-01 FncErr=1.00D-07 GrdErr=1.00D-07 EigMax=2.50D+02 EigMin=1.00D-04",
            f" Number of optimizations in scan= {len(parsed.frames):5d}",
            " Number of steps in this run=     20 maximum allowed number of steps=    100.",
            " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
            "",
        ]
    )
    return lines


def _format_scan_optimized_parameters(frame: Frame, include_gradgrad: bool = True) -> list[str]:
    if not frame.scan_values:
        return []
    lines = [
        "                       ----------------------------",
        "                       !   Optimized Parameters   !",
        "                       ! (Angstroms and Degrees)  !",
        " ----------------------                            ----------------------",
        " !      Name          Value   Derivative information (Atomic Units)     !",
        " ------------------------------------------------------------------------",
    ]
    for index, value in enumerate(frame.scan_values, start=1):
        lines.append(f" !       {_scan_definition_label(index):<8s}{value:10.5f}      -DE/DX =    0.0                           !")
    lines.extend([" ------------------------------------------------------------------------", " Lowest energy point so far.  Saving SCF results."])
    if include_gradgrad:
        lines.append("GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad")
    return lines


def _format_scan_step(frame_index: int, frame_count: int, frame: Frame) -> list[str]:
    lines = [
        "GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad",
        " Berny optimization.",
        " Search for a local minimum.",
        f" Step number   1 out of a maximum of   20 on scan point {frame_index:5d} out of {frame_count:5d}",
        " All quantities printed in internal units (Hartrees-Bohrs-Radians)",
        "         Item               Value     Threshold  Converged?",
    ]

    def add(label: str, item: tuple[float, float, bool] | None) -> None:
        if item is None:
            value, threshold, converged = 0.0, 0.0, True
        else:
            value, threshold, converged = item
        lines.append(f" {label:<22s}{value:12.6f} {threshold:12.6f}     {'YES' if converged else 'NO'}")

    convergence = frame.convergence
    add("Maximum Force", convergence.maximum_force if convergence else None)
    add("RMS     Force", convergence.rms_force if convergence else None)
    add("Maximum Displacement", convergence.maximum_displacement if convergence else None)
    add("RMS     Displacement", convergence.rms_displacement if convergence else None)
    lines.extend(
        [
            " Predicted change in Energy=-0.000000D+00",
            " Optimization completed.",
            "    -- Stationary point found.",
        ]
    )
    return lines


def _format_gaussian_footer(kind: str) -> list[str]:
    lines = [""]
    if kind == "irc":
        lines.extend(
            [
                " Leave Link  601 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
                " (Enter /usr/local/g16/l9999.exe)",
                "",
                " This type of calculation cannot be archived.",
                "",
                "",
                " IT IS A SIMPLE TASK TO MAKE THINGS COMPLEX,",
                " BUT A COMPLEX TASK TO MAKE THEM SIMPLE.",
            ]
        )
    else:
        lines.extend(
            [
                " Leave Link  601 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
                " (Enter /usr/local/g16/l9999.exe)",
                "",
            ]
        )
    lines.extend(
        [
            " Job cpu time:       0 days  0 hours  0 minutes  0.0 seconds.",
            " Elapsed time:       0 days  0 hours  0 minutes  0.0 seconds.",
            " File lengths (MBytes):  RWF=      1 Int=      0 D2E=      0 Chk=      1 Scr=      1",
            " Normal termination of Gaussian 16 at Sat May 23 00:00:00 2026.",
            "",
        ]
    )
    return lines


def render_gaussian_log(parsed: MapleOutput, energy_unit: str = "auto") -> str:
    if not parsed.frames and not parsed.initial_atoms:
        raise ValueError(
            "MAPLE output does not include coordinates; provide a same-stem .inp or .xyz sidecar"
        )

    atoms_for_counts = parsed.frames[0].atoms if parsed.frames else parsed.initial_atoms
    _total_electrons, alpha_electrons, beta_electrons = _electron_counts(
        atoms_for_counts,
        parsed.charge,
        parsed.multiplicity,
    )
    kind = _result_kind(parsed)
    if kind == "ts_opt":
        return _render_ts_minimal_optimization(
            parsed,
            energy_unit,
            alpha_electrons,
            beta_electrons,
        )
    if kind == "scan":
        return _render_scan_minimal(
            parsed,
            energy_unit,
            alpha_electrons,
            beta_electrons,
        )

    route = _route_for(parsed)
    lines = _format_gaussian_preamble(parsed, route, alpha_electrons, beta_electrons)

    if kind == "md":
        lines.extend(_format_md_preamble(parsed))
    if kind == "irc":
        transition_state_frame = parsed.frames[_irc_transition_state_index(parsed.frames)]
        lines.extend(_format_irc_l202_block(transition_state_frame.atoms))
        lines.extend(_format_irc_pre_scf_links())
        transition_state_energy = _frame_energy_hartree(transition_state_frame, energy_unit, parsed.model)
        if transition_state_energy is not None:
            lines.extend(
                [
                    "",
                    f" SCF Done:  E(theory) = {transition_state_energy:.9f}     A.U. after    0 cycles",
                ]
            )
        lines.append("")
        lines.extend(_format_irc_preamble(parsed))

    irc_coordinates = _reaction_coordinates(parsed.frames) if kind == "irc" else []
    frame_indices = list(range(len(parsed.frames)))
    if kind == "irc":
        frame_indices = sorted(
            [index for index in frame_indices if abs(irc_coordinates[index]) >= 1.0e-8],
            key=lambda index: (
                1
                if irc_coordinates[index] > 0
                else 2,
                abs(irc_coordinates[index]),
            ),
        )

    for frame_index, original_frame_index in enumerate(frame_indices, start=1):
        frame = parsed.frames[original_frame_index]
        reaction_coordinate = irc_coordinates[original_frame_index] if kind == "irc" else None
        if kind == "irc":
            lines.extend(_format_irc_l202_block(frame.atoms))
            lines.extend(_format_irc_pre_scf_links())
            energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
            if energy is not None:
                lines.extend(
                    [
                        "",
                        f" SCF Done:  E(theory) = {energy:.9f}     A.U. after    0 cycles",
                    ]
                )
            lines.extend(_format_irc_enter_l123())

            if reaction_coordinate is None or abs(reaction_coordinate) < 1.0e-8:
                lines.append("")
                continue

            nonzero_frame_indices = [
                index
                for index in frame_indices
                if abs(irc_coordinates[index]) >= 1.0e-8
            ]
            nonzero_position = nonzero_frame_indices.index(original_frame_index)
            next_reaction_coordinate = None
            next_display_point_number = None
            if nonzero_position + 1 < len(nonzero_frame_indices):
                next_original_frame_index = nonzero_frame_indices[nonzero_position + 1]
                next_reaction_coordinate = irc_coordinates[next_original_frame_index]
                next_display_point_number = _irc_point_number(next_reaction_coordinate)
            lines.extend(
                _format_irc_point(
                    reaction_coordinate,
                    frame.atoms,
                    next_reaction_coordinate,
                    nonzero_position + 1,
                    next_display_point_number,
                )
            )
            lines.append("")
            continue

        lines.extend(_format_orientation(frame.atoms))
        energy = _frame_energy_hartree(frame, energy_unit, parsed.model)
        if energy is not None:
            lines.extend(
                [
                    "",
                    f" SCF Done:  E(theory) = {energy:.9f}     A.U. after    0 cycles",
                ]
            )

        if (kind in {"opt", "scan"} and len(parsed.frames) > 1) or kind == "ts_opt":
            lines.extend(_format_convergence(frame_index, frame.convergence))
            if frame_index == len(parsed.frames) and kind == "ts_opt":
                lines.extend(_format_ts_completion())
            elif frame_index == len(parsed.frames) and _convergence_all_yes(frame.convergence):
                lines.extend([" Optimization completed.", " -- Stationary point found."])
            lines.append("")
        elif kind == "md":
            thermo = parsed.md_thermo[frame_index - 1] if frame_index <= len(parsed.md_thermo) else None
            lines.extend(_format_md_step(frame_index, frame, thermo, energy_unit, parsed.model))
            lines.append("")

    if kind == "irc":
        lines.extend(_format_irc_summary(parsed, energy_unit))
        if frame_indices:
            lines.extend(
                [
                    " Leave Link  123 at Sat May 23 00:00:00 2026, MaxMem=   104857600 cpu:               0.0 elap:               0.0",
                    " (Enter /usr/local/g16/l202.exe)",
                ]
            )
            lines.extend(_format_irc_l202_block(parsed.frames[frame_indices[-1]].atoms, next_link=601))
    if kind == "md":
        lines.extend(_format_md_summary(parsed, energy_unit))
    if (kind in {"opt", "scan"} and len(parsed.frames) > 1) or kind == "ts_opt":
        lines.extend(_format_opt_final_l202_block(parsed.frames[-1].atoms))
    if kind == "scan":
        lines.extend(_format_scan_summary(parsed, energy_unit))
    lines.extend(_format_frequency_blocks(parsed))
    lines.extend(_format_thermochemistry(parsed))
    lines.extend(_format_gaussian_footer(kind))
    return "\n".join(lines)


def convert_file(input_file: str | Path, output_file: str | Path | None = None, energy_unit: str = "auto") -> Path:
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path.with_suffix(".log")
    parsed = parse_maple_output(input_path)
    output_path.write_text(render_gaussian_log(parsed, energy_unit=energy_unit), newline="\r\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maple2gs",
        description="Convert MAPLE output to a Gaussian-like log for post-processing tools.",
    )
    parser.add_argument("input_file", help="MAPLE .out file")
    parser.add_argument("-o", "--output", help="Output Gaussian-like .log file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = convert_file(args.input_file, args.output, energy_unit="auto")
    except Exception as exc:
        print(f"maple2gs: error: {exc}", file=sys.stderr)
        return 1

    print(f"Gaussian-like log written to: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
