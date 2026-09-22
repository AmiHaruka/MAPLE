"""Usage: parse user-facing torsion fitting options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ...runconfig import as_tracked


TORSIONFIT_CANONICAL_PERIODS = (1, 2, 3, 4)


@dataclass
class TorsionFitParams:
    enabled: bool = True
    torsion_bonds: Optional[tuple[tuple[int, int], ...]] = None
    radical_center: tuple[int, ...] | None = None
    p_thresh: float = 30.0
    torsion_steps: int = 36
    torsion_step_deg: float = field(init=False)
    backend: str = "cgbs"   #lbfgs/cgws/cgbs
    constraint_mode: str = "projected"  #fixinternals/projected

    refine_rounds: int = 3
    refine_max_iter: int = 256
    refine_tol: float = 1.0e-6
    stage1_weights: bool = True
    report_debug: bool = False
    torsion_ensemble: bool = False
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
        self.stage1_weights = bool(self.stage1_weights)
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


def _split_entries(value):
    """Shared front half of the entry-list parsers: None/empty -> None, else list/tuple or comma split."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value if isinstance(value, (list, tuple)) else str(value).split(",")


def _parse_torsion_bonds(value) -> Optional[tuple[tuple[int, int], ...]]:
    entries = _split_entries(value)
    if entries is None:
        return None

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
            raise ValueError(f"Invalid torsion bond entry: {entry!r}")
        left_index, right_index = int(left), int(right)
        bond = (left_index, right_index) if left_index < right_index else (right_index, left_index)
        if bond[0] == bond[1]:
            raise ValueError(f"torsion bond cannot be self-referential: {entry!r}")
        if bond not in seen:
            seen.add(bond)
            bonds.append(bond)
    return tuple(bonds) if bonds else None


def _parse_int_entries(value) -> Optional[tuple[int, ...]]:
    entries = _split_entries(value)
    if entries is None:
        return None
    centers = tuple(int(str(entry).strip()) for entry in entries if str(entry).strip())
    return centers or None


def build_torsion_fit_params(paras: Optional[dict]) -> TorsionFitParams:
    root = as_tracked(paras)
    root.set_group("TorsionFit", "Torsion parameter fitting switches and scan controls")
    for alias in ("corr", "correction", "parmfit"):
        if alias in root and isinstance(root[alias], dict):
            # Nested API payloads are plain dicts; wrap so the note-carrying get works.
            root = as_tracked(root[alias])
            break
    torsion = TorsionFitParams()

    torsion.enabled = _coerce_bool(root.get("torsionfit", torsion.enabled))
    torsion.torsion_bonds = _parse_torsion_bonds(root.get("torsion_bonds", torsion.torsion_bonds))
    torsion.radical_center = _parse_int_entries(root.get("radical_center", torsion.radical_center))
    torsion.p_thresh = float(root.get("p_thresh", torsion.p_thresh))
    torsion.torsion_steps = int(root.get("torsion_steps", torsion.torsion_steps))
    torsion.backend = root.get("backend", torsion.backend)
    torsion.constraint_mode = root.get("constraint_mode", torsion.constraint_mode)
    torsion.refine_rounds = int(root.get("torsion_refine_rounds", torsion.refine_rounds))
    torsion.refine_max_iter = int(root.get("torsion_refine_max_iter", torsion.refine_max_iter))
    torsion.refine_tol = float(root.get("torsion_refine_tol", torsion.refine_tol))
    torsion.stage1_weights = _coerce_bool(root.get("stage1_weights", torsion.stage1_weights))
    torsion.report_debug = _coerce_bool(root.get("report_debug", torsion.report_debug))
    torsion.torsion_ensemble = _coerce_bool(root.get("torsion_ensemble", torsion.torsion_ensemble))
    torsion.torsion_ensemble_ratio = float(root.get("torsion_ensemble_ratio", torsion.torsion_ensemble_ratio))
    torsion.torsion_ensemble_weight = float(root.get("torsion_ensemble_weight", torsion.torsion_ensemble_weight))

    torsion._refresh_derived()
    return torsion
