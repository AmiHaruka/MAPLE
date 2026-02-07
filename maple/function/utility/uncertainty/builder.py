import os
from typing import Any

from .config import parse_ensemble_config
from .ensemble_calculator import EnsembleCalculator


def _log_lines(output: str, lines) -> None:
    with open(output, "a") as f:
        for line in lines:
            f.write(line)


def wrap_with_ensemble(
    primary_calc,
    ensemble_raw: Any,
    calculator_factory,
    output: str,
    jobtype: str,
):
    """
    Build and apply uncertainty ensemble wrapper if requested.

    Rules in this version:
      - Custom model paths are supported for MACE and AIMNet2 families.
      - models[0] is fixed as primary calculator.
      - models[1:] are observers.
      - freq job skips ensemble evaluation.
      - not supported for md tasks.
    """
    cfg = parse_ensemble_config(ensemble_raw)
    if not cfg.active:
        return primary_calc

    if str(jobtype).lower() in ("freq","md"):
        _log_lines(output, ["[UNC] ensemble active but skipped for freq/md task.\n"])
        return primary_calc

    model_paths = [os.path.abspath(p) for p in cfg.models]

    # Build primary from models[0]
    primary_path = model_paths[0]
    primary = calculator_factory.build_calculator_from_path(primary_path)

    observers = []
    labels = []
    dropped = []

    for i, path in enumerate(model_paths[1:], start=2):
        label = f"model{i}:{os.path.basename(path)}"
        try:
            obs = calculator_factory.build_calculator_from_path(path)
            observers.append(obs)
            labels.append(label)
        except Exception as exc:
            dropped.append(f"[UNC][WARN] dropped observer at build: {label} reason={exc}\n")

    if dropped:
        _log_lines(output, dropped)

    if len(observers) == 0:
        _log_lines(
            output,
            [
                "[UNC][WARN] ensemble enabled but all observer models failed; "
                "falling back to primary only.\n"
            ],
        )
        return primary

    _log_lines(
        output,
        [
            "[UNC] ensemble enabled with {} models (1 primary + {} observers).\n".format(
                1 + len(observers), len(observers)
            )
        ],
    )
    return EnsembleCalculator(
        primary_calc=primary,
        observer_calcs=observers,
        output=output,
        observer_labels=labels,
        on_error=cfg.on_error,
    )
