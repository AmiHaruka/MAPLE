from __future__ import annotations

from pathlib import Path

import numpy as np

from ..calculator_base import CalcABC, EV2HARTREE, register_calculator
from ..electronic_state import (
    attach_calculator_identity,
    solvation_identity_settings,
    validate_electronic_state,
)


_FENNOL_MODEL_FILES = {
    "fennix-bio1s": "fennix-bio1S-finetuneIons.fnx",
    "fennix-bio1m": "fennix-bio1M-finetuneIons.fnx",
}

@register_calculator
class FeNNolCalculator(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy", "hessian"]

    MODEL_NAMES = ("fennol", "fennix-bio1s", "fennix-bio1m")
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    LOCAL_MODEL_FILENAMES = _FENNOL_MODEL_FILES
    OPTION_KEYS = ("use_float64",)
    MODEL_PATH_OPTION = "model_path"

    @staticmethod
    def model_filename(model: str) -> str:
        key = str(model).strip().lower()
        try:
            return _FENNOL_MODEL_FILES[key]
        except KeyError as exc:
            raise ValueError(f"Unknown built-in FeNNol model name: {model!r}") from exc

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        kwargs = {"model_path": resolved_model_path}
        use_float64 = model_options.get("use_float64")
        if use_float64 is not None:
            kwargs["use_float64"] = str(use_float64).strip().lower() in {"1", "true", "yes", "on",}
        return kwargs

    @staticmethod
    def _load_runtime(model_path: str, *, use_float64: bool):
        from ._runtime import load_fennol_runtime

        return load_fennol_runtime(model_path, use_float64=use_float64)

    def __init__(
        self,
        device=None,
        model: str = "fennix-bio1s",
        model_path: str | None = None,
        implicit: str = "none",
        solvent: str = "none",
        use_float64: bool = False,
    ):
        super().__init__()
        self.device = device
        if model_path is None:
            if str(model).strip().lower() == "fennol":
                raise ValueError("Generic #model=fennol requires model_path=/path/to/model.fnx.")
            model_dir = Path(__file__).resolve().parents[1] / "model"
            model_path = str(model_dir / self.model_filename(model))
        self.model_path = str(model_path)
        self.runtime = self._load_runtime(self.model_path, use_float64=use_float64)
        self.hessian = "analytic"
        attach_calculator_identity(
            self,
            backend=str(model).lower(),
            checkpoint_path=self.model_path,
            relevant_settings={
                "use_float64": bool(use_float64),
                **solvation_identity_settings(implicit, solvent),
            },
        )
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _runtime_unit(self) -> str:
        return getattr(self.runtime, "energy_unit", self.MODEL_ENERGY_UNIT)

    @staticmethod
    def validate_electronic_state_request(charge, multiplicity, settings) -> None:
        if multiplicity != 1:
            raise ValueError(
                "FeNNol accepts total charge but this MAPLE runtime has no spin/multiplicity "
                "input; use mult=1."
            )

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = self._normalize_properties(properties)
        if system_changes is None:
            atoms = super().calculate(atoms, properties)
        else:
            atoms = super().calculate(atoms, properties, system_changes)
        total_charge = self._total_charge_from_atoms(atoms)

        needs_forces = "forces" in properties
        needs_hessian = "hessian" in properties
        if needs_forces:
            energy, forces, _ = self.runtime.energy_forces(atoms, total_charge)
        else:
            energy, _ = self.runtime.energy(atoms, total_charge)
            forces = None

        hessian = None
        if needs_hessian:
            hessian = np.asarray(self.get_hessian(atoms), dtype=float)

        self._finalize_results(atoms, energy=energy, forces=forces, hessian=hessian, unit=self._runtime_unit())

    def _analytic_hessian(self, atoms) -> np.ndarray:
        hessian = np.asarray(
            self.runtime.hessian(atoms, self._total_charge_from_atoms(atoms)),
            dtype=float,
        )
        return hessian * (EV2HARTREE if self._runtime_unit() == "eV" else 1.0)

    def get_hvp(self, atoms, n: np.ndarray):
        if getattr(self, "solvent_correction", None) is not None:
            return super().get_hvp(atoms, n)
        self._reject_unsupported_pbc(atoms)
        validate_electronic_state(atoms, self)

        if hasattr(n, "detach"):
            vector = n.detach().cpu().numpy()
        else:
            vector = np.asarray(n, dtype=float)
        hvp_ev, forces_ev, energy_ev = self.runtime.hvp(
            atoms,
            self._total_charge_from_atoms(atoms),
            vector,
        )

        import torch

        scale = EV2HARTREE if self._runtime_unit() == "eV" else 1.0
        return (
            torch.as_tensor(np.asarray(hvp_ev, dtype=float) * scale, dtype=torch.float64),
            torch.as_tensor(np.asarray(forces_ev, dtype=float) * scale, dtype=torch.float64),
            torch.as_tensor(float(energy_ev) * scale, dtype=torch.float64),
        )
