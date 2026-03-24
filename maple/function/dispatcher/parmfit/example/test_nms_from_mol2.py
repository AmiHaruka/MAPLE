from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_THRESHOLDS = {
    "f_max_th": 0.00285,
    "f_rms_th": 0.00190,
    "dp_max_th": 0.00315,
    "dp_rms_th": 0.00210,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple NMS workflow from a mol2 file: read, optimize, Hessian, sample.",
    )
    parser.add_argument("mol2", type=Path, help="Input Tripos mol2 file with coordinates.")
    parser.add_argument("--model", default="aimnet2nse", help="MAPLE model name. Default: aimnet2")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge. Default: 0")
    parser.add_argument("--mult", type=int, default=1, help="Spin multiplicity. Default: 1")
    parser.add_argument("--samples", type=int, default=50, help="Number of NMS conformers. Default: 50")
    parser.add_argument("--temperature", type=float, default=0.00000001, help="Sampling temperature in K")
    parser.add_argument("--start-mode", type=int, default=7, help="1-based normal mode index to start from")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--distribution",
        default="classical",
        choices=("classical", "wigner"),
        help="Normal-coordinate sampling distribution. Default: classical",
    )
    parser.add_argument("--max-iter", type=int, default=256, help="Maximum LBFGS iterations")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output log path. Default: <mol2_stem>_nms.out",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        default=None,
        help="Output XYZ ensemble path. Default: <mol2_stem>_nms.xyz",
    )
    parser.add_argument("--verbose", type=int, default=1, help="LBFGS verbosity: 0 or 1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ase import Atoms  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("This script requires ASE to read/write atomic structures.") from exc

    from maple.function.calculator.set_calculator import SetClaculator
    from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS
    from maple.function.dispatcher.parmfit.utils.nma import compute_normal_modes, sample_normal_modes

    mol2_path = args.mol2.resolve()
    if not mol2_path.is_file():
        raise FileNotFoundError(f"mol2 file not found: {mol2_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{args.device}' but CUDA is not available.")

    base = mol2_path.with_suffix("")
    output_path = args.output.resolve() if args.output is not None else base.with_name(base.name + "_nms.out")
    xyz_path = args.xyz.resolve() if args.xyz is not None else base.with_name(base.name + "_nms.xyz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    atoms = load_mol2_atoms(mol2_path)
    atoms.info["charge"] = args.charge
    atoms.info["mult"] = args.mult
    apply_default_thresholds(atoms)

    calculator = SetClaculator(
        device=torch.device(args.device),
        model=args.model,
        output=str(output_path),
        atoms=atoms,
        d4=False,
        implicit="None",
        solvent="None",
    ).set_calculator()
    atoms.calc = calculator

    optimizer = LBFGS(
        atoms,
        output=str(output_path),
        paras={"opt": {"max_iter": args.max_iter, "verbose": args.verbose, "write_traj": False}},
    )
    atoms = optimizer.run()

    hessian_cart = atoms.calc.get_hessian(atoms)
    normal_modes = compute_normal_modes(atoms, hessian_cart)
    samples = sample_normal_modes(
        atoms,
        normal_modes,
        n_samples=args.samples,
        temperature=args.temperature,
        start_mode=args.start_mode,
        random_seed=args.seed,
        distribution=args.distribution,
    )

    write_xyz_trajectory(xyz_path, samples.samples)
    log_lines = [
        "\n",
        "=" * 70 + "\n",
        "Simple NMS Sampling".center(70) + "\n",
        "=" * 70 + "\n",
        f"mol2:          {mol2_path}\n",
        f"model:         {args.model}\n",
        f"device:        {args.device}\n",
        f"charge/mult:   {args.charge}/{args.mult}\n",
        f"samples:       {args.samples}\n",
        f"temperature:   {args.temperature:.2f} K\n",
        f"distribution:  {args.distribution}\n",
        f"start_mode:    {args.start_mode}\n",
        f"sampled modes: {samples.mode_indices}\n",
        f"ensemble xyz:  {xyz_path}\n",
        "=" * 70 + "\n",
    ]
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.writelines(log_lines)

    print(f"NMS complete. Wrote {args.samples} conformers to {xyz_path}")
    print(f"Log written to {output_path}")
    return 0


def load_mol2_atoms(path: Path) -> Atoms:
    from ase import Atoms

    symbols: list[str] = []
    positions: list[list[float]] = []
    charges: list[float] = []
    in_atom_section = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("@<TRIPOS>"):
                in_atom_section = stripped.upper() == "@<TRIPOS>ATOM"
                continue
            if not in_atom_section:
                continue

            parts = raw.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid mol2 atom line in {path}: {raw.rstrip()}")
            atom_name = parts[1]
            x, y, z = map(float, parts[2:5])
            atom_type = parts[5]
            charge = float(parts[8]) if len(parts) >= 9 else 0.0

            symbols.append(guess_symbol(atom_name, atom_type))
            positions.append([x, y, z])
            charges.append(charge)

    if not symbols:
        raise ValueError(f"mol2 file has no @<TRIPOS>ATOM section with coordinates: {path}")

    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    if hasattr(atoms, "set_initial_charges"):
        atoms.set_initial_charges(np.asarray(charges, dtype=float))
    return atoms


def guess_symbol(atom_name: str, atom_type: str) -> str:
    from ase.data import chemical_symbols

    allowed = sorted((symbol for symbol in chemical_symbols if symbol), key=len, reverse=True)
    candidates = [atom_type.split(".", 1)[0], atom_name]
    for raw in candidates:
        letters = re.sub(r"[^A-Za-z]", "", raw)
        if not letters:
            continue
        normalized = letters[0].upper() + letters[1:].lower()
        for symbol in allowed:
            if normalized.startswith(symbol):
                return symbol
        single = normalized[0]
        if single in allowed:
            return single
    raise ValueError(f"Failed to infer element symbol from mol2 atom '{atom_name}' / '{atom_type}'.")


def apply_default_thresholds(atoms: Atoms) -> None:
    for key, value in DEFAULT_THRESHOLDS.items():
        setattr(atoms, key, value)


def write_xyz_trajectory(path: Path, atoms_list: list[Atoms]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for index, atoms in enumerate(atoms_list):
            handle.write(f"{len(atoms)}\n")
            handle.write(f"NMS sample {index}\n")
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")


if __name__ == "__main__":
    raise SystemExit(main())
