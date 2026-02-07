import copy
import hashlib
from typing import List, Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .logger import format_uncertainty_lines
from .metrics import energy_stats, force_stats


class EnsembleCalculator(Calculator):
    """
    ASE calculator wrapper:
      - primary calculator provides official MAPLE results
      - observer calculators are used for uncertainty estimation only
    """

    def __init__(
        self,
        primary_calc: Calculator,
        observer_calcs: List[Calculator],
        output: str,
        observer_labels: Optional[List[str]] = None,
        on_error: str = "warn",
    ) -> None:
        super().__init__()
        self.primary_calc = primary_calc
        self.observer_calcs = list(observer_calcs)
        self.output = output
        self.on_error = str(on_error).lower()

        self.observer_labels = observer_labels or [
            f"model{i+2}" for i in range(len(self.observer_calcs))
        ]
        if len(self.observer_labels) != len(self.observer_calcs):
            self.observer_labels = [
                f"model{i+2}" for i in range(len(self.observer_calcs))
            ]

        self.dropped_messages: List[str] = []
        self.eval_counter = 0
        self._last_sig: Optional[str] = None
        self.implemented_properties = getattr(
            self.primary_calc, "implemented_properties", ["energy", "forces", "free_energy"]
        )

    def _log_info(self, lines: List[str]) -> None:
        with open(self.output, "a") as f:
            for line in lines:
                f.write(line)

    def _make_sig(self, atoms, props: List[str]) -> str:
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        payload = pos.tobytes() + ("|".join(sorted(props))).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        props = list(properties) if properties is not None else ["energy"]
        super().calculate(atoms, props, system_changes)

        # 1) primary evaluation
        self.primary_calc.calculate(atoms, props, system_changes)
        self.results = copy.deepcopy(self.primary_calc.results)

        # Determine whether uncertainty evaluation should run for this call
        needs_energy = any(p in props for p in ("energy", "free_energy", "forces"))
        needs_forces = "forces" in props
        if (not needs_energy and not needs_forces) or len(self.observer_calcs) == 0:
            return

        # Optional log de-dup for identical geometry + properties
        sig = self._make_sig(atoms, props)
        if sig == self._last_sig:
            return
        self._last_sig = sig

        obs_props = ["energy"]
        if needs_forces:
            obs_props.append("forces")

        primary_energy = self.results.get("energy", None)
        primary_forces = self.results.get("forces", None)

        obs_energies: List[float] = []
        obs_forces: List[np.ndarray] = []
        active_labels: List[str] = []
        keep_calcs: List[Calculator] = []
        keep_labels: List[str] = []
        dropped_local: List[str] = []

        # 2) observer evaluation
        for calc, label in zip(self.observer_calcs, self.observer_labels):
            try:
                calc.calculate(atoms, obs_props, system_changes)
                e = calc.results.get("energy", None)
                if e is not None:
                    obs_energies.append(float(e))
                if needs_forces and ("forces" in calc.results):
                    obs_forces.append(np.asarray(calc.results["forces"], dtype=np.float64))
                keep_calcs.append(calc)
                keep_labels.append(label)
                active_labels.append(label)
            except Exception as exc:
                msg = f"dropped observer: {label} reason={exc}"
                dropped_local.append(msg)
                self.dropped_messages.append(msg)

        # Keep only active observers after this call
        self.observer_calcs = keep_calcs
        self.observer_labels = keep_labels

        # 3) metrics
        e_stat = None
        f_stat = None
        if primary_energy is not None and len(obs_energies) > 0:
            e_stat = energy_stats(float(primary_energy), obs_energies)
        if needs_forces and primary_forces is not None and len(obs_forces) > 0:
            f_stat = force_stats(np.asarray(primary_forces, dtype=np.float64), obs_forces)

        self.eval_counter += 1
        lines = format_uncertainty_lines(
            eval_id=self.eval_counter,
            props=obs_props,
            active_models=1 + len(active_labels),
            dropped_count=len(self.dropped_messages),
            energy=e_stat,
            force=f_stat,
            observer_labels=active_labels,
            dropped_messages=dropped_local,
        )
        self._log_info(lines)

    # Delegate Hessian-style calls directly to primary calculator.
    def get_hessian(self, atoms=None, **kwargs):
        if hasattr(self.primary_calc, "get_hessian"):
            return self.primary_calc.get_hessian(atoms, **kwargs)
        raise AttributeError("Primary calculator does not support get_hessian().")

    def __getattr__(self, name):
        primary = self.__dict__.get("primary_calc", None)
        if primary is not None and hasattr(primary, name):
            return getattr(primary, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
