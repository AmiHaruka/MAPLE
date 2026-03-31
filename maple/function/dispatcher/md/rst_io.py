import json
from pathlib import Path

import numpy as np
from ase.cell import Cell


RST_HEADER = "MAPLE_RST_V1"



def get_rng_state_hex(rng: np.random.Generator) -> str:
    state_json = json.dumps(rng.bit_generator.state, sort_keys=True)
    return state_json.encode().hex()



def restore_rng_from_hex(rng: np.random.Generator, hex_str: str) -> None:
    state = json.loads(bytes.fromhex(hex_str).decode())
    rng.bit_generator.state = state



def write_rst(path, atoms, velocities, step, timestep, ensemble, energy, rng_state=None):
    path = Path(path)
    expected_shape = (len(atoms), 3)
    if np.shape(velocities) != expected_shape:
        raise ValueError(
            "Velocities must have shape "
            f"{expected_shape}, got {np.shape(velocities)}"
        )
    time_fs = step * timestep
    cell_line = ""
    pbc_line = ""
    if any(atoms.pbc):
        cell = atoms.cell.cellpar()
        cell_line = (
            f"cell = {cell[0]:.10f} {cell[1]:.10f} {cell[2]:.10f} "
            f"{cell[3]:.10f} {cell[4]:.10f} {cell[5]:.10f}\n"
        )
        pbc_flags = ["T" if flag else "F" for flag in atoms.pbc]
        pbc_line = f"pbc = {' '.join(pbc_flags)}\n"

    lines = [
        f"{RST_HEADER}\n",
        f"natoms = {len(atoms)}\n",
        f"step = {step}\n",
        f"time = {time_fs:.10f}\n",
        f"ensemble = {ensemble}\n",
        f"timestep = {timestep:.10f}\n",
        f"energy = {energy:.10f}\n",
    ]
    if rng_state is not None:
        lines.append(f"rng_state = {rng_state}\n")
    if cell_line:
        lines.append(cell_line)
    if pbc_line:
        lines.append(pbc_line)

    for symbol, pos, vel in zip(atoms.get_chemical_symbols(), atoms.get_positions(), velocities):
        lines.append(
            f"{symbol:<2s} {pos[0]: .10f} {pos[1]: .10f} {pos[2]: .10f}"
            f" {vel[0]: .10e} {vel[1]: .10e} {vel[2]: .10e}\n"
        )
    lines.append("END_RST\n")
    path.write_text("".join(lines))



def read_rst(path):
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != RST_HEADER:
        raise ValueError(f"Not a valid MAPLE RST file: {path}")
    if not lines or lines[-1].strip() != "END_RST":
        raise ValueError(f"Missing END_RST in {path}")

    header = {}
    atom_lines = []
    for line in lines[1:-1]:
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
        elif line.strip():
            atom_lines.append(line)

    required_fields = ["natoms", "step", "time", "ensemble", "timestep", "energy"]
    missing_fields = [field for field in required_fields if field not in header]
    if missing_fields:
        missing_list = ", ".join(missing_fields)
        raise ValueError(f"Missing required RST header fields: {missing_list}")

    natoms = int(header["natoms"])
    if len(atom_lines) != natoms:
        raise ValueError(
            f"Atom count mismatch inside RST file: expected {natoms}, found {len(atom_lines)}"
        )

    symbols = []
    positions = []
    velocities = []
    for idx, line in enumerate(atom_lines, 1):
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"Invalid atom line {idx} in {path}: {line!r}")
        symbol = parts[0]
        xyz = [float(x) for x in parts[1:4]]
        vel = [float(x) for x in parts[4:7]]
        symbols.append(symbol)
        positions.append(xyz)
        velocities.append(vel)

    cell = None
    if "cell" in header:
        cell = [float(x) for x in header["cell"].split()]
        if len(cell) != 6:
            raise ValueError(f"Invalid cell line in {path}")

    pbc = None
    if "pbc" in header:
        pbc = [flag == "T" for flag in header["pbc"].split()]
        if len(pbc) != 3:
            raise ValueError(f"Invalid pbc line in {path}")

    return {
        "natoms": natoms,
        "step": int(header["step"]),
        "time": float(header["time"]),
        "ensemble": header["ensemble"],
        "timestep": float(header["timestep"]),
        "energy": float(header["energy"]),
        "rng_state": header.get("rng_state"),
        "symbols": symbols,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "cell": cell,
        "pbc": pbc,
    }


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
):
    rst_path = Path(rst_path)
    rst_prev_path = Path(rst_prev_path)

    if rst_path.exists() and rst_path.stat().st_size > 0:
        if rst_prev_path.exists():
            rst_prev_path.unlink()
        rst_path.replace(rst_prev_path)

    write_rst(
        rst_path,
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=energy,
        rng_state=rng_state,
    )
