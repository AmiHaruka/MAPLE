#!/usr/bin/env python3
"""Evidence-producing scalar/batch parity panel for local MAPLE checkpoints.

This is a verification tool, not a scientific-admission gate.  All numerical
bounds are frozen below before any model is loaded.  The tool never downloads
a checkpoint; UMA is deliberately opt-in because its smallest local checkpoint
is about 1.1 GB and its CPU runtime can be substantial.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from ase import Atoms

import maple


# Preregistered absolute bounds in MAPLE's returned units (Ha and Ha/Angstrom).
# UMA has no repository-validated bound, so its measurements are report-only.
FROZEN_BOUNDS: dict[str, dict[str, float] | None] = {
    "aimnet2": {"energy": 5e-7, "forces": 5e-7},
    "aimnet2nse": {"energy": 5e-7, "forces": 5e-7},
    "ani2x": {"energy": 5e-7, "forces": 5e-7},
    "maceoff23m": {"energy": 1e-9, "forces": 1e-9},
    "uma": None,
}
DEFAULT_MODELS = tuple(FROZEN_BOUNDS)
CHECKPOINT_NAMES = {
    "aimnet2": "aimnet2.pt",
    "aimnet2nse": "aimnet2nse.pt",
    "ani2x": "ani2x.pt",
    "maceoff23m": "maceoff23m.pt",
    "uma": "uma-s-1p1.pt",
}
COMMON_SOURCE_FILES = (
    "maple/function/calculator/_batch_types.py",
    "maple/function/calculator/_batch_utils.py",
    "maple/function/calculator/_batch_eval.py",
    "maple/function/calculator/calculator_base.py",
)
MODEL_SOURCE_FILES = {
    "aimnet2": ("maple/function/calculator/aimnet/_aimnet2_calculator.py",),
    "aimnet2nse": ("maple/function/calculator/aimnet/_aimnet2_calculator.py",),
    "ani2x": ("maple/function/calculator/ani/_ani_calculator.py",),
    "maceoff23m": (
        "maple/function/calculator/mace/_mace_calculator.py",
        "maple/function/calculator/mace/_batch_graph.py",
        "maple/function/calculator/mace/_common.py",
    ),
    "uma": ("maple/function/calculator/uma/_uma_calculator.py",),
}


class NativeBatchUnavailableError(RuntimeError):
    """The loaded checkpoint/calculator does not admit native E/F batching."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(*args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return b"unknown"


def provenance() -> dict[str, Any]:
    source = Path(__file__).resolve()
    diff = _git_bytes("diff", "--binary", "HEAD", "--")
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "git_base": _git_bytes("rev-parse", "HEAD").decode().strip(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_dirty": bool(_git_bytes("status", "--porcelain").strip()),
    }


def source_manifest(models: Iterable[str]) -> dict[str, str]:
    """Hash the exact shared and backend sources relevant to this run."""
    paths = set(COMMON_SOURCE_FILES)
    for model in models:
        paths.update(MODEL_SOURCE_FILES[model])
    manifest = {}
    for relative in sorted(paths):
        path = Path(relative)
        manifest[relative] = sha256_file(path) if path.is_file() else "MISSING"
    return manifest


def _benzene_cluster() -> Atoms:
    """Return five separated benzene fragments as one 60-atom molecule."""
    symbols: list[str] = []
    positions: list[list[float]] = []
    for fragment in range(5):
        origin = np.array([20.0 * fragment, 0.0, 0.0])
        for radius, symbol in ((1.397, "C"), (2.487, "H")):
            for index in range(6):
                angle = index * np.pi / 3.0
                positions.append(
                    (origin + [radius * np.cos(angle), radius * np.sin(angle), 0.0]).tolist()
                )
                symbols.append(symbol)
    return Atoms(symbols, positions=positions)


def neutral_structures() -> list[Atoms]:
    return [
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        Atoms("CH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]),
        Atoms("NH3", positions=[[0, 0, 0.12], [0.94, 0, -0.28], [-0.47, 0.81, -0.28], [-0.47, -0.81, -0.28]]),
        Atoms("C2H6", positions=[[-0.77, 0, 0], [0.77, 0, 0], [-1.15, 1.0, 0], [-1.15, -0.5, 0.87], [-1.15, -0.5, -0.87], [1.15, -1.0, 0], [1.15, 0.5, 0.87], [1.15, 0.5, -0.87]]),
        Atoms("COH2", positions=[[0, 0, 0], [1.22, 0, 0], [-0.58, 0.94, 0], [-0.58, -0.94, 0]]),
        Atoms("COH4", positions=[[0, 0, 0], [1.43, 0, 0], [1.82, 0.9, 0], [-0.4, 0.95, 0], [-0.4, -0.48, 0.82], [-0.4, -0.48, -0.82]]),
        _benzene_cluster(),
    ]


def nse_structures() -> list[Atoms]:
    return [
        Atoms("O", positions=[[0, 0, 0]], info={"charge": 0, "mult": 3}),
        Atoms("OH", positions=[[0, 0, 0], [0.97, 0, 0]], info={"charge": 0, "mult": 2}),
        Atoms("NH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], info={"charge": 1, "mult": 1}),
    ]


def _identity(calc: Any) -> Any:
    getter = getattr(calc, "get_pes_identity", None)
    value = getter() if callable(getter) else getattr(calc, "maple_pes_identity", None)
    return copy.deepcopy(value)


def _sequential(calc: Any, structures: Iterable[Atoms], energy_kind: str) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    energies, forces = [], []
    for atoms in structures:
        calc.calculate(atoms, properties=(energy_kind, "forces"))
        energies.append(float(calc.results[energy_kind]))
        forces.append(np.asarray(calc.results["forces"], dtype=np.float64).copy())
    return np.asarray(energies), tuple(forces)


def _errors(result: Any, reference: tuple[np.ndarray, tuple[np.ndarray, ...]], energy_kind: str, structures: list[Atoms]) -> tuple[float, float]:
    result.validate_against(structures, (energy_kind, "forces"))
    if result.energy_kind != energy_kind:
        raise AssertionError(f"returned energy_kind={result.energy_kind!r}, expected {energy_kind!r}")
    energy_error = float(np.max(np.abs(np.asarray(result.energies) - reference[0])))
    force_error = float(max(np.max(np.abs(got - want)) for got, want in zip(result.forces, reference[1])))
    return energy_error, force_error


def _observed_calculate_many(
    calc: Any,
    structures: list[Atoms],
    properties: tuple[str, str],
) -> tuple[Any, int]:
    """Call native batching while counting any hidden scalar fallback calls."""
    original = calc.calculate
    instance_dict = getattr(calc, "__dict__", {})
    had_instance_override = "calculate" in instance_dict
    prior_instance_value = instance_dict.get("calculate")
    scalar_calls = 0

    def observed_calculate(*args, **kwargs):
        nonlocal scalar_calls
        scalar_calls += 1
        return original(*args, **kwargs)

    calc.calculate = observed_calculate
    try:
        result = calc.calculate_many(structures, properties=properties)
    finally:
        if had_instance_override:
            calc.calculate = prior_instance_value
        else:
            del calc.calculate
    return result, scalar_calls


def _chunked(calc: Any, structures: list[Atoms], properties: tuple[str, str], size: int) -> Any:
    from maple.function.calculator._batch_types import BatchResult

    parts = []
    observations = []
    for start in range(0, len(structures), size):
        selected = structures[start:start + size]
        part, scalar_calls = _observed_calculate_many(calc, selected, properties)
        # Validate every native return before concatenation.  In particular,
        # do not overwrite and thereby conceal a wrong energy_kind label.
        part.validate_against(selected, properties)
        parts.append(part)
        observations.append(
            {"structures": len(selected), "scalar_calculate_calls": scalar_calls}
        )
    result = BatchResult(
        energies=np.concatenate([part.energies for part in parts]),
        energy_kind=properties[0],
        forces=tuple(force for part in parts for force in part.forces),
    )
    return result, observations


def run_parity_panel(calc: Any, model_name: str, structures: list[Atoms], bounds: dict[str, float] | None) -> dict[str, Any]:
    """Measure fixed cases.  The supplied bounds must predate all execution."""
    if getattr(calc, "supports_batch_energy_forces", None) is not True:
        raise NativeBatchUnavailableError(
            f"{model_name} loaded calculator/checkpoint does not explicitly "
            "declare supports_batch_energy_forces=True"
        )
    initial_identity = _identity(calc)
    measurements: list[dict[str, Any]] = []
    subsets = {
        "B1": structures[:1],
        "B2": structures[:2],
        "B8": structures[:8],
        "reversed": list(reversed(structures[:8])),
        "mixed_size": [structures[0], structures[1], structures[-1]],
    }
    for energy_kind in ("energy", "free_energy"):
        reference = _sequential(calc, structures[:8], energy_kind)
        by_id = {id(at): i for i, at in enumerate(structures[:8])}
        for case, selected in subsets.items():
            indices = [by_id[id(at)] for at in selected]
            expected = (reference[0][indices], tuple(reference[1][i] for i in indices))
            result, scalar_calls = _observed_calculate_many(
                calc, selected, (energy_kind, "forces")
            )
            energy_error, force_error = _errors(result, expected, energy_kind, selected)
            observations = [
                {
                    "structures": len(selected),
                    "scalar_calculate_calls": scalar_calls,
                }
            ]
            measurements.append(
                _measurement(
                    case,
                    energy_kind,
                    len(selected),
                    energy_error,
                    force_error,
                    bounds,
                    observations,
                )
            )
        for chunk_size in (1, 2, 8):
            result, observations = _chunked(
                calc, structures[:8], (energy_kind, "forces"), chunk_size
            )
            energy_error, force_error = _errors(result, reference, energy_kind, structures[:8])
            measurements.append(
                _measurement(
                    f"chunk_{chunk_size}",
                    energy_kind,
                    len(reference[0]),
                    energy_error,
                    force_error,
                    bounds,
                    observations,
                )
            )

    final_identity = _identity(calc)
    identity_preserved = initial_identity == final_identity
    bounded_pass = bounds is not None and identity_preserved and all(m["status"] == "PASS" for m in measurements)
    return {
        "model": model_name,
        "status": "PASS" if bounded_pass else ("UNQUALIFIED" if bounds is None and identity_preserved else "FAIL"),
        "qualification": "bounded" if bounds is not None else "report_only_no_repository_validated_bound",
        "bounds": bounds,
        "structure_atom_counts": [len(atoms) for atoms in structures],
        "largest_structure_note": (
            "five benzene fragments separated by 20 Angstrom within one "
            "nonperiodic Atoms object; backend long-range handling retained"
            if structures and len(structures[-1]) == 60
            else None
        ),
        "identity_preserved": identity_preserved,
        "initial_identity": initial_identity,
        "measurements": measurements,
    }


def _measurement(
    case: str,
    energy_kind: str,
    count: int,
    energy_error: float,
    force_error: float,
    bounds: dict[str, float] | None,
    observations: list[dict[str, int]],
) -> dict[str, Any]:
    # B1 calls are reported but not treated as proof of multi-structure native
    # execution. Every part with B>1 must avoid the scalar calculate method.
    checked = [item for item in observations if item["structures"] > 1]
    native_no_serial_fallback = all(
        item["scalar_calculate_calls"] == 0 for item in checked
    )
    if bounds is None:
        status = "UNQUALIFIED"
    else:
        status = (
            "PASS"
            if energy_error <= bounds["energy"]
            and force_error <= bounds["forces"]
            and native_no_serial_fallback
            else "FAIL"
        )
    return {
        "case": case,
        "energy_kind": energy_kind,
        "structures": count,
        "energy_max_abs_error_hartree": energy_error,
        "force_max_abs_error_hartree_per_angstrom": force_error,
        "native_fallback_observations": observations,
        "native_fallback_checked": bool(checked),
        "native_no_serial_fallback": native_no_serial_fallback,
        "status": status,
    }


def _load_calculator(
    model: str,
    checkpoint: Path,
    device: torch.device,
    coulomb: str = "simple",
) -> Any:
    if model in {"aimnet2", "aimnet2nse"}:
        from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
        return AIMNet2Calculator(
            device=device,
            model=model,
            model_path=str(checkpoint),
            coulomb_method=coulomb,
        )
    if model == "ani2x":
        from maple.function.calculator.ani._ani_calculator import ANICalculator
        return ANICalculator(device=device, model=model, model_path=str(checkpoint))
    if model == "maceoff23m":
        from maple.function.calculator.mace._mace_calculator import MACECalculator
        return MACECalculator(device=device, model=model, model_path=str(checkpoint))
    if model == "uma":
        from maple.function.calculator.uma._uma_calculator import UMACalculator
        return UMACalculator(device=device, model="uma", size="uma-s-1p1", checkpoint_path=str(checkpoint), task="omol", inference_settings="default")
    raise ValueError(f"unknown model {model!r}")


def unavailable(model: str, reason: str, checkpoint: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"model": model, "status": "UNAVAILABLE", "reason": reason, "bounds": FROZEN_BOUNDS[model]}
    if checkpoint is not None:
        record["checkpoint"] = str(checkpoint)
    return record


def _uma_runtime_checkpoint_record(
    source_digest: str,
    calc: Any,
) -> dict[str, Any]:
    """Describe UMA's derived compatibility artifact without rewriting it."""
    from fairchem.core._config import CACHE_DIR

    compat_path = (
        Path(CACHE_DIR)
        / "maple_compat"
        / f"uma-s-1p1-v2-{source_digest}.pt"
    )
    identity = _identity(calc) or {}
    fingerprint = identity.get("model_fingerprint", {})
    return {
        "path": str(compat_path.resolve()),
        "exists": compat_path.is_file(),
        "sha256": fingerprint.get("digest"),
        "derivation": "UMA v2 compatibility copy derived from immutable local source checkpoint",
        "source_checkpoint_mutated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("maple/function/calculator/model"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/batch-backend-parity.json"))
    parser.add_argument("--include-uma", action="store_true", help="explicitly permit loading/running the large local UMA checkpoint")
    parser.add_argument(
        "--coulomb",
        choices=("simple", "dsf"),
        default="simple",
        help="AIMNet2 Coulomb implementation; ignored by other backends",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero unless every selected backend has bounded PASS status",
    )
    args = parser.parse_args(argv)

    # Prevent dependency code from silently fetching absent assets.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    records = []
    for model in dict.fromkeys(args.models):
        checkpoint = args.checkpoint_dir / CHECKPOINT_NAMES[model]
        if model == "uma" and not args.include_uma:
            records.append(unavailable(model, "not run: UMA requires explicit --include-uma resource opt-in", checkpoint))
            continue
        if not checkpoint.is_file():
            records.append(unavailable(model, "local checkpoint is missing; downloads are disabled", checkpoint))
            continue
        started = time.perf_counter()
        try:
            calc = _load_calculator(model, checkpoint, device, args.coulomb)
            structures = neutral_structures()
            result = run_parity_panel(calc, model, structures, FROZEN_BOUNDS[model])
            if model == "aimnet2nse":
                result["charged_open_shell_panel"] = run_parity_panel(calc, model, nse_structures(), FROZEN_BOUNDS[model])
                if result["charged_open_shell_panel"]["status"] != "PASS":
                    result["status"] = "FAIL"
            result.update({
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "device": str(device),
                "dtype": str(getattr(calc, "dtype", getattr(calc, "input_dtype", "backend_managed"))),
                "coulomb": args.coulomb if model.startswith("aimnet2") else None,
                "elapsed_seconds": time.perf_counter() - started,
            })
            if model == "uma":
                result["runtime_checkpoint"] = _uma_runtime_checkpoint_record(
                    result["checkpoint_sha256"],
                    calc,
                )
            records.append(result)
        except NativeBatchUnavailableError as exc:
            record = unavailable(model, str(exc), checkpoint)
            record.update(
                {
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            records.append(record)
        except (ImportError, MemoryError) as exc:
            record = unavailable(
                model,
                f"local runtime resource unavailable: {type(exc).__name__}: {exc}",
                checkpoint,
            )
            record.update(
                {
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            records.append(record)
        except Exception as exc:  # record negative evidence rather than skip-to-green
            records.append({
                "model": model,
                "status": "FAIL",
                "bounds": FROZEN_BOUNDS[model],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            })

    statuses = [record["status"] for record in records]
    if any(status == "FAIL" for status in statuses):
        status = "FAIL"
    elif any(status in {"UNAVAILABLE", "UNQUALIFIED"} for status in statuses):
        status = "INCOMPLETE"
    else:
        status = "PASS"
    packages = {}
    for package in ("ase", "numpy", "torch", "fairchem-core"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    artifact = {
        "schema_version": 1,
        "scope": "engineering parity only; not scientific admission or a benchmark",
        "status": status,
        "frozen_bounds": FROZEN_BOUNDS,
        "provenance": provenance(),
        "source_manifest_sha256": source_manifest(args.models),
        "runtime": {"maple": maple.__version__, "python": sys.version, "platform": platform.platform(), "device": str(device), "packages": packages},
        "models": records,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True, default=str))
    return 1 if status == "FAIL" or (args.require_complete and status != "PASS") else 0


if __name__ == "__main__":
    raise SystemExit(main())
