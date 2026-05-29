"""Usage: parse user-facing torsion fitting options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def normalize_center_bond(center_bond: tuple[int, int]) -> tuple[int, int]:
    return _pair(*center_bond)


@dataclass
class TorsionFitParams:
    enabled: bool = True
    center_bonds: Optional[tuple[tuple[int, int], ...]] = None
    torsion_steps: int = 36
    torsion_step_deg: float = field(init=False)
    backend: str = "cgbs"   #lbfgs/cgws/cgbs
    constraint_mode: str = "projected"  #fixinternals/projected

    refine_rounds: int = 2
    refine_max_iter: int = 100
    refine_tol: float = 1.0e-6
    report_debug: bool = False
    torsion_ensemble: bool = True
    torsion_ensemble_ratio: float = 0.3
    torsion_ensemble_weight: float = 0.50

    def __post_init__(self):
        self._refresh_derived()

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def _refresh_derived(self):
        self.torsion_steps = int(self.torsion_steps)
        if self.torsion_steps <= 0:
            raise ValueError("torsion_steps must be a positive integer.")
        self.backend = str(self.backend).strip().lower()
        if self.backend not in {"lbfgs", "cgws", "cgbs"}:
            raise ValueError(f"backend must be one of {{'lbfgs', 'cgws', 'cgbs'}}, got {self.backend!r}.")
        self.constraint_mode = str(self.constraint_mode).strip().lower()
        if self.constraint_mode not in {"fixinternals", "projected"}:
            raise ValueError(
                f"constraint_mode must be one of {{'fixinternals', 'projected'}}, got {self.constraint_mode!r}."
            )
        self.refine_rounds = int(self.refine_rounds)
        self.refine_max_iter = int(self.refine_max_iter)
        self.refine_tol = float(self.refine_tol)
        self.report_debug = bool(self.report_debug)
        self.torsion_ensemble = bool(self.torsion_ensemble)
        self.torsion_ensemble_ratio = float(self.torsion_ensemble_ratio)
        if self.torsion_ensemble_ratio <= 0.0:
            raise ValueError("torsion_ensemble_ratio must be positive.")
        self.torsion_ensemble_weight = float(self.torsion_ensemble_weight)
        if self.torsion_ensemble_weight < 0.0:
            raise ValueError("torsion_ensemble_weight must be non-negative.")
        self.torsion_step_deg = 360.0 / float(self.torsion_steps)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if token in {"false", "f", "no", "n", "0", "off"}:
            return False
    raise ValueError(f"Invalid torsionfit boolean value: {value!r}")


def _coerce_int(value) -> int:
    return int(value)


def _coerce_float(value) -> float:
    return float(value)


def _parse_center_bonds(value) -> Optional[tuple[tuple[int, int], ...]]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    entries = value if isinstance(value, (list, tuple)) else str(value).split(",")
    bonds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        if isinstance(entry, str):
            token = entry.strip()
            if not token:
                continue
            left, right = token.split("-", 1)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            left, right = entry
        else:
            raise ValueError(f"Invalid torsion center bond entry: {entry!r}")
        bond = normalize_center_bond((int(left), int(right)))
        if bond[0] == bond[1]:
            raise ValueError(f"torsion center bond cannot be self-referential: {entry!r}")
        if bond not in seen:
            seen.add(bond)
            bonds.append(bond)
    return tuple(bonds) if bonds else None


def build_torsion_fit_params(paras: Optional[dict]) -> TorsionFitParams:
    params = paras if isinstance(paras, dict) else {}
    root = params
    for alias in ("correction", "parmfit"):
        if isinstance(params.get(alias), dict):
            root = params[alias]
            break
    source = root
    torsion = TorsionFitParams()

    if "torsionfit" in source:
        torsion.enabled = _coerce_bool(source["torsionfit"])
    if "torsion_bonds" in source:
        torsion.center_bonds = _parse_center_bonds(source["torsion_bonds"])
    if "torsion_steps" in source:
        torsion.torsion_steps = _coerce_int(source["torsion_steps"])
    if "backend" in source:
        torsion.backend = source["backend"]
    if "constraint_mode" in source:
        torsion.constraint_mode = source["constraint_mode"]
    if "torsion_refine_rounds" in source:
        torsion.refine_rounds = _coerce_int(source["torsion_refine_rounds"])
    if "torsion_refine_max_iter" in source:
        torsion.refine_max_iter = _coerce_int(source["torsion_refine_max_iter"])
    if "torsion_refine_tol" in source:
        torsion.refine_tol = _coerce_float(source["torsion_refine_tol"])
    if "report_debug" in source:
        torsion.report_debug = _coerce_bool(source["report_debug"])
    if "torsion_ensemble" in source:
        torsion.torsion_ensemble = _coerce_bool(source["torsion_ensemble"])
    if "torsion_ensemble_ratio" in source:
        torsion.torsion_ensemble_ratio = _coerce_float(source["torsion_ensemble_ratio"])
    if "torsion_ensemble_weight" in source:
        torsion.torsion_ensemble_weight = _coerce_float(source["torsion_ensemble_weight"])

    torsion._refresh_derived()
    return torsion
