from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EnsembleConfig:
    active: bool = False
    models: List[str] = None
    on_error: str = "warn"

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = []


def parse_ensemble_config(raw: Any) -> EnsembleConfig:
    """
    Parse raw `ensemble` setting from CommandControl params.

    Accepted input:
      - None -> inactive config
      - dict with keys: active, models, on_error
      - bool  -> active flag only
    """
    if raw is None:
        return EnsembleConfig()

    if isinstance(raw, bool):
        return EnsembleConfig(active=raw)

    if not isinstance(raw, dict):
        raise ValueError("ensemble must be a dict, bool, or omitted.")

    active = bool(raw.get("active", False))
    on_error = str(raw.get("on_error", "warn")).strip().lower()
    models_raw = raw.get("models", "")

    if isinstance(models_raw, list):
        models = [str(x).strip() for x in models_raw if str(x).strip()]
    else:
        models = [x.strip() for x in str(models_raw).split(";") if x.strip()]

    cfg = EnsembleConfig(active=active, models=models, on_error=on_error)
    _validate_ensemble_config(cfg)
    return cfg


def _validate_ensemble_config(cfg: EnsembleConfig) -> None:
    if cfg.on_error not in ("warn",):
        raise ValueError("ensemble.on_error only supports 'warn' in this version.")

    if not cfg.active:
        return

    if len(cfg.models) < 2:
        raise ValueError(
            "ensemble.active=true requires at least 2 model paths in ensemble.models."
        )

