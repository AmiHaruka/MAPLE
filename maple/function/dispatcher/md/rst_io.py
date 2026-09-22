"""Versioned restart checkpoint I/O for molecular dynamics.

RST v2 stores Cartesian positions and velocities at full text round-trip
precision, the complete 3x3 cell, RNG state, and a checksummed state contract.
The contract binds an exact continuation to its PES/electronic state, dynamics
parameters, masses, and ASE constraints. RST v1 remains readable only so the
logger can expose it through explicit new-run ``load_state`` semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

RST_HEADER_V1 = "MAPLE_RST_V1"
RST_HEADER_V2 = "MAPLE_RST_V2"
RST_HEADER = RST_HEADER_V2


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Checkpoint metadata contains unsupported value {type(value).__name__}")


def constraint_identity(atoms) -> list[dict[str, Any]]:
    """Return the canonical ASE constraint definition stored by RST v2."""
    definitions = []
    for constraint in atoms.constraints:
        if not hasattr(constraint, "todict"):
            raise ValueError(
                f"Constraint {type(constraint).__name__} cannot be represented in an exact restart"
            )
        definitions.append(_jsonable(constraint.todict()))
    return definitions


def get_rng_state_hex(rng: np.random.Generator) -> str:
    """Serialize a NumPy generator state as hex-encoded canonical JSON."""
    return _canonical_json(rng.bit_generator.state).encode().hex()


def restore_rng_from_hex(rng: np.random.Generator, hex_str: str) -> None:
    """Restore a NumPy generator state written by :func:`get_rng_state_hex`."""
    state = json.loads(bytes.fromhex(hex_str).decode())
    rng.bit_generator.state = state


def _state_contract(atoms, pes_identity, dynamics_parameters) -> dict[str, Any]:
    if pes_identity is None:
        raise ValueError("RST v2 requires a PES/electronic-state identity")
    if dynamics_parameters is None:
        raise ValueError("RST v2 requires dynamics parameters")
    return {
        "pes_identity": _jsonable(pes_identity),
        "dynamics_parameters": _jsonable(dynamics_parameters),
        "masses": _jsonable(np.asarray(atoms.get_masses(), dtype=np.float64)),
        "constraints": constraint_identity(atoms),
    }


def _serialize_rst(
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
    velocity_representation=None,
    pes_identity=None,
    dynamics_parameters=None,
):
    """Serialize and validate an exact RST v2 checkpoint."""
    velocities = np.asarray(velocities, dtype=np.float64)
    expected_shape = (len(atoms), 3)
    if velocities.shape != expected_shape:
        raise ValueError(f"Velocities must have shape {expected_shape}, got {velocities.shape}")

    contract = _state_contract(atoms, pes_identity, dynamics_parameters)
    contract_json = _canonical_json(contract)
    contract_digest = hashlib.sha256(contract_json.encode()).hexdigest()
    cell = np.asarray(atoms.cell.array, dtype=np.float64).reshape(9)
    pbc_flags = ["T" if flag else "F" for flag in atoms.pbc]

    lines = [
        f"{RST_HEADER_V2}\n",
        f"natoms = {len(atoms)}\n",
        f"step = {int(step)}\n",
        f"time = {float(step) * float(timestep):.17g}\n",
        f"ensemble = {ensemble}\n",
        f"timestep = {float(timestep):.17g}\n",
        f"energy = {float(energy):.17g}\n",
        f"state_contract = {contract_json}\n",
        f"state_contract_sha256 = {contract_digest}\n",
        "cell = " + " ".join(f"{value:.17g}" for value in cell) + "\n",
        f"pbc = {' '.join(pbc_flags)}\n",
    ]
    if velocity_representation is not None:
        lines.append(f"velocity_representation = {velocity_representation}\n")
    if rng_state is not None:
        lines.append(f"rng_state = {rng_state}\n")

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    for symbol, pos, vel in zip(atoms.get_chemical_symbols(), positions, velocities):
        values = " ".join(f"{value:.17g}" for value in np.concatenate((pos, vel)))
        lines.append(f"{symbol} {values}\n")
    lines.append("END_RST\n")
    return "".join(lines).encode("utf-8")


def _write_temp_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return Path(name)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = _write_temp_bytes(path, data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_rst(
    path,
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
    velocity_representation=None,
    pes_identity=None,
    dynamics_parameters=None,
):
    """Atomically write a validated RST v2 checkpoint."""
    data = _serialize_rst(
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=energy,
        rng_state=rng_state,
        velocity_representation=velocity_representation,
        pes_identity=pes_identity,
        dynamics_parameters=dynamics_parameters,
    )
    _atomic_write_bytes(Path(path), data)


def _parse_json_field(header: dict[str, str], key: str, path: Path) -> Any:
    try:
        return json.loads(header[key])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or missing {key} in {path}") from exc


def _parse_rst_text(text: str, path: Path) -> dict:
    """Parse and validate checkpoint text captured from one immutable snapshot."""
    lines = text.splitlines()
    if not lines or lines[0].strip() not in {RST_HEADER_V1, RST_HEADER_V2}:
        raise ValueError(f"Not a valid MAPLE RST file: {path}")
    version = 2 if lines[0].strip() == RST_HEADER_V2 else 1
    if lines[-1].strip() != "END_RST":
        raise ValueError(f"Missing END_RST in {path}")

    header: dict[str, str] = {}
    atom_lines = []
    for line in lines[1:-1]:
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
        elif line.strip():
            atom_lines.append(line)

    required = ["natoms", "step", "time", "ensemble", "timestep", "energy"]
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(f"Missing required RST header fields: {', '.join(missing)}")

    try:
        natoms = int(header["natoms"])
        step = int(header["step"])
        time = float(header["time"])
        timestep = float(header["timestep"])
        energy = float(header["energy"])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric RST header field in {path}") from exc
    if natoms < 0 or len(atom_lines) != natoms:
        raise ValueError(
            f"Atom count mismatch inside RST file: expected {natoms}, found {len(atom_lines)}"
        )

    symbols, positions, velocities = [], [], []
    for idx, line in enumerate(atom_lines, 1):
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"Invalid atom line {idx} in {path}: {line!r}")
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid atom line {idx} in {path}: {line!r}") from exc
        symbols.append(parts[0])
        positions.append(values[:3])
        velocities.append(values[3:])

    positions_array = np.asarray(positions, dtype=np.float64).reshape(natoms, 3)
    velocities_array = np.asarray(velocities, dtype=np.float64).reshape(natoms, 3)
    numeric_header = np.asarray([time, timestep, energy], dtype=np.float64)
    if not np.all(np.isfinite(numeric_header)) or timestep <= 0 or step < 0:
        raise ValueError(f"Invalid non-finite or non-positive RST state in {path}")
    if not np.all(np.isfinite(positions_array)) or not np.all(np.isfinite(velocities_array)):
        raise ValueError(f"Invalid non-finite atomic state in {path}")
    if not np.isclose(time, step * timestep, rtol=0.0, atol=1e-10):
        raise ValueError(f"RST time is inconsistent with step and timestep in {path}")

    cell = None
    if "cell" in header:
        try:
            cell = [float(value) for value in header["cell"].split()]
        except ValueError as exc:
            raise ValueError(f"Invalid cell line in {path}") from exc
        expected_cell_values = 9 if version == 2 else 6
        if len(cell) != expected_cell_values:
            raise ValueError(f"Invalid cell line in {path}")
        if version == 2:
            cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)

    pbc = None
    if "pbc" in header:
        flags = header["pbc"].split()
        if len(flags) != 3 or any(flag not in {"T", "F"} for flag in flags):
            raise ValueError(f"Invalid pbc line in {path}")
        pbc = [flag == "T" for flag in flags]

    contract = None
    if version == 2:
        if cell is None or pbc is None:
            raise ValueError(f"RST v2 requires full cell and pbc fields in {path}")
        contract = _parse_json_field(header, "state_contract", path)
        if not isinstance(contract, dict):
            raise ValueError(f"Invalid state_contract in {path}")
        required_contract = {"pes_identity", "dynamics_parameters", "masses", "constraints"}
        if required_contract - contract.keys():
            missing_contract = ", ".join(sorted(required_contract - contract.keys()))
            raise ValueError(f"Missing RST state contract fields: {missing_contract}")
        expected_digest = hashlib.sha256(_canonical_json(contract).encode()).hexdigest()
        if header.get("state_contract_sha256") != expected_digest:
            raise ValueError(f"RST state contract checksum mismatch in {path}")
        masses = np.asarray(contract["masses"], dtype=np.float64)
        if masses.shape != (natoms,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError(f"Invalid masses in RST state contract: {path}")
        if not isinstance(contract["constraints"], list):
            raise ValueError(f"Invalid constraints in RST state contract: {path}")

    if "rng_state" in header:
        try:
            rng_state = json.loads(bytes.fromhex(header["rng_state"]).decode())
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid rng_state in {path}") from exc
        if not isinstance(rng_state, dict) or "bit_generator" not in rng_state:
            raise ValueError(f"Invalid rng_state in {path}")

    return {
        "version": version,
        "natoms": natoms,
        "step": step,
        "time": time,
        "ensemble": header["ensemble"],
        "timestep": timestep,
        "energy": energy,
        "rng_state": header.get("rng_state"),
        "velocity_representation": header.get("velocity_representation", "standard"),
        "symbols": symbols,
        "positions": positions_array,
        "velocities": velocities_array,
        "cell": cell,
        "pbc": pbc,
        "pes_identity": None if contract is None else contract["pes_identity"],
        "dynamics_parameters": None if contract is None else contract["dynamics_parameters"],
        "masses": None if contract is None else np.asarray(contract["masses"], dtype=np.float64),
        "constraints": None if contract is None else contract["constraints"],
    }


def read_rst_bytes(data: bytes, path: str | Path = "<memory>") -> dict:
    """Parse an RST checkpoint from an already captured byte string."""
    path = Path(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"RST checkpoint is not valid UTF-8: {path}") from exc
    return _parse_rst_text(text, path)


def read_rst(path) -> dict:
    """Read one immutable RST byte snapshot and validate it."""
    path = Path(path)
    return read_rst_bytes(path.read_bytes(), path)


def rotate_rst_checkpoint(
    rst_path,
    rst_prev_path,
    atoms,
    velocities,
    step,
    timestep,
    ensemble,
    energy,
    rng_state=None,
    velocity_representation=None,
    pes_identity=None,
    dynamics_parameters=None,
):
    """Rotate the current runtime checkpoint and write a new RST v2 file."""
    rst_path = Path(rst_path)
    rst_prev_path = Path(rst_prev_path)
    data = _serialize_rst(
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=energy,
        rng_state=rng_state,
        velocity_representation=velocity_representation,
        pes_identity=pes_identity,
        dynamics_parameters=dynamics_parameters,
    )
    new_checkpoint = _write_temp_bytes(rst_path, data)
    previous_checkpoint = None
    try:
        if rst_path.exists() and rst_path.stat().st_size > 0:
            previous_checkpoint = _write_temp_bytes(
                rst_prev_path, rst_path.read_bytes()
            )
            os.replace(previous_checkpoint, rst_prev_path)
            previous_checkpoint = None
        os.replace(new_checkpoint, rst_path)
        new_checkpoint = None
    finally:
        if new_checkpoint is not None:
            new_checkpoint.unlink(missing_ok=True)
        if previous_checkpoint is not None:
            previous_checkpoint.unlink(missing_ok=True)


def promote_rst_checkpoint(
    source, rst_path, rst_prev_path, *, source_bytes: bytes | None = None
) -> None:
    """Promote captured, validated bytes to the canonical checkpoint path."""
    source = Path(source)
    rst_path = Path(rst_path)
    rst_prev_path = Path(rst_prev_path)
    data = source.read_bytes() if source_bytes is None else source_bytes
    if (
        source.resolve() == rst_path.resolve()
        and rst_path.exists()
        and rst_path.read_bytes() == data
    ):
        return

    promoted = _write_temp_bytes(rst_path, data)
    previous = None
    try:
        source_is_previous = source.resolve() == rst_prev_path.resolve()
        if rst_path.exists() and not source_is_previous:
            previous = _write_temp_bytes(rst_prev_path, rst_path.read_bytes())
        os.replace(promoted, rst_path)
        promoted = None
        if previous is not None:
            os.replace(previous, rst_prev_path)
            previous = None
    finally:
        if promoted is not None:
            promoted.unlink(missing_ok=True)
        if previous is not None:
            previous.unlink(missing_ok=True)
