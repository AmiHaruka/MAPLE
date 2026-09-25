#!/usr/bin/env python3
"""Run required real-checkpoint AIMNet2 parity and write a release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from ase import Atoms

import maple
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNet2Calculator,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _structures(model_name: str) -> list[Atoms]:
    if model_name == "aimnet2nse":
        return [
            Atoms(
                "O",
                positions=[[0.0, 0.0, 0.0]],
                info={"charge": 0, "mult": 3},
            ),
            Atoms(
                "OH",
                positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]],
                info={"charge": 0, "mult": 2},
            ),
            Atoms(
                "O2",
                positions=[[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]],
                info={"charge": 0, "mult": 3},
            ),
        ]
    return [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        ),
        Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ),
    ]


def _verify_model(
    model_name: str,
    checkpoint: Path,
    device: torch.device,
    atol: float,
) -> dict:
    atoms_list = _structures(model_name)
    calc = AIMNet2Calculator(
        device=device,
        model=model_name,
        model_path=str(checkpoint),
        coulomb_method="simple",
    )
    if not calc.supports_batch_energy_forces:
        raise RuntimeError(
            f"{model_name} checkpoint is not recognized by the batch manifest"
        )

    # Warm both paths before timing.
    calc.calculate(atoms_list[0], properties=("energy", "forces"))
    calc.calculate_many(atoms_list[:1], properties=("energy", "forces"))
    _synchronize(device)

    start = time.perf_counter()
    single_energies = []
    single_forces = []
    for atoms in atoms_list:
        calc.calculate(atoms, properties=("energy", "forces"))
        single_energies.append(float(calc.results["energy"]))
        single_forces.append(
            np.asarray(calc.results["forces"], dtype=np.float64).copy()
        )
    _synchronize(device)
    sequential_seconds = time.perf_counter() - start

    start = time.perf_counter()
    batched = calc.calculate_many(
        atoms_list,
        properties=("energy", "forces"),
    )
    _synchronize(device)
    batch_seconds = time.perf_counter() - start
    energy_only = calc.calculate_many(atoms_list, properties=("energy",))

    energy_error = float(
        np.max(
            np.abs(
                np.asarray(batched.energies)
                - np.asarray(single_energies)
            )
        )
    )
    energy_only_error = float(
        np.max(
            np.abs(
                np.asarray(energy_only.energies)
                - np.asarray(single_energies)
            )
        )
    )
    batch_forces = batched.forces
    if batch_forces is None:
        raise RuntimeError(
            f"{model_name} batch result omitted requested forces"
        )
    force_error = float(
        max(
            np.max(np.abs(batch_force - single_force))
            for batch_force, single_force in zip(
                batch_forces,
                single_forces,
            )
        )
    )
    passed = (
        energy_error <= atol
        and energy_only_error <= atol
        and force_error <= atol
    )
    return {
        "model": model_name,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "input_dtype": str(calc.input_dtype),
        "num_charge_channels": calc.num_charge_channels,
        "structures": len(atoms_list),
        "energy_max_abs_error_hartree": energy_error,
        "energy_only_max_abs_error_hartree": energy_only_error,
        "force_max_abs_error_hartree_per_angstrom": force_error,
        "absolute_tolerance": atol,
        "sequential_seconds": sequential_seconds,
        "batch_seconds": batch_seconds,
        "speedup": (
            sequential_seconds / batch_seconds
            if batch_seconds > 0.0
            else None
        ),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "MAPLE_RELEASE_CHECKPOINT_DIR",
                "maple/function/calculator/model",
            )
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("MAPLE_RELEASE_DEVICE", "cuda"),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=float(os.environ.get("MAPLE_BATCH_RELEASE_ATOL", "5e-7")),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/aimnet2-batch-release.json"),
    )
    args = parser.parse_args()

    if not np.isfinite(args.atol) or args.atol <= 0.0:
        parser.error("--atol must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA release verification was requested but is unavailable")

    checkpoints = {
        name: args.checkpoint_dir / f"{name}.pt"
        for name in ("aimnet2", "aimnet2nse")
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        parser.error(
            "required release checkpoints are missing: " + ", ".join(missing)
        )

    results = [
        _verify_model(name, path, device, args.atol)
        for name, path in checkpoints.items()
    ]
    artifact = {
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "maple_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "maple_version": maple.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "models": results,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
