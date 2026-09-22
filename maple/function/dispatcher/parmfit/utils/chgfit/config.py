"""Usage: parse the user-facing charge-fitting options (chg_fit/chg_level/chg_route)."""

from __future__ import annotations

from dataclasses import dataclass

from .. import interface
from ..interface import QMMethod


@dataclass
class ChargeFitConfig:
    method: str = "none"
    level: str = "HF/6-31G(d)"
    route: str = ""
    qm: QMMethod | None = None



def build_charge_fit_config(
    raw: dict,
    *,
    default_method: str,
    default_level: str,
    default_nproc: int,
    default_mem: int,
) -> ChargeFitConfig:
    method_value = str(raw.get("chg_fit", default_method)).strip()
    method_key = method_value.lower()
    method = method_key if method_key in {"none", "resp"} else method_value
    config = ChargeFitConfig(
        method=method,
        level=str(raw.get("chg_level", default_level)).strip(),
        route=str(raw.get("chg_route", "")).strip(),
    )
    if method != "resp":
        return config

    backend = str(raw.get("resp_backend", "gaussian")).strip().lower()
    if backend != "gaussian":
        raise ValueError(
            f"Unsupported RESP backend {backend!r}; expected one of 'gaussian'."
        )
    if "/" not in config.level:
        raise ValueError(
            f"RESP level {config.level!r} must use METHOD/BASIS syntax."
        )
    theory, basis = (part.strip() for part in config.level.split("/", 1))
    if not theory or not basis:
        raise ValueError(
            f"RESP level {config.level!r} must use METHOD/BASIS syntax."
        )
    config.qm = interface.set_method(
        {
            "theory": theory,
            "basis": basis,
            "route": config.route,
            "nproc": raw.get("qm_nproc", default_nproc),
            "mem": raw.get("qm_mem", default_mem),
        },
        backend="gaussian",
    )
    return config

