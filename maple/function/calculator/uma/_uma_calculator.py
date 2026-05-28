from __future__ import annotations

import importlib
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core._config import CACHE_DIR
    from fairchem.core.calculate.ase_calculator import AtomicData, FAIRChemCalculator, UMATask
    from fairchem.core.units.mlip_unit import load_predict_unit
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

from ..calculator_base import (
    EV2HARTREE,
    init_implicit_solvent,
    numerical_hessian_from_atoms,
    register_calculator,
)


UMA_DEFAULT_SIZE = "uma-s-1p1"
UMA_MODELS_MAP = {
    "uma": UMA_DEFAULT_SIZE,
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
}

UMA_FALLBACK_HF_MODELS = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
SUPPORTED_UMA_INFERENCE = {"default", "turbo"}

# GPU default and CPU fallback for FAIR Chemistry's inference path. "default"
# is the general-purpose backend; "turbo" speeds up fixed-composition workloads
# (NEB / TS / freq) on CUDA but routes through Triton kernels unavailable on
# CPU. Users override via `model=uma(...,inference=turbo)`; CPU coerces to
# default regardless.
UMA_INFERENCE_SETTINGS = "default"
UMA_CPU_INFERENCE_SETTINGS = "default"


@register_calculator
class UMACalculator(FAIRChemCalculator):
    """UMA calculator with MAPLE-specific unit conversion and Hessian support.

    Does NOT inherit CalcABC — UMA already extends third-party FAIRChemCalculator.
    Satisfies the MAPLE calculator protocol via attribute presence (the six
    MODEL_* class attrs + calculate + get_hessian + get_hvp where required).

    Explicit `task=` is respected. When `task` is omitted, MAPLE keeps the
    historical convenience behavior of inferring `omol` for non-periodic
    systems and `omat` for periodic systems.
    """

    MODEL_NAMES = ("uma",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False

    # Legacy attribute kept until SetClaculator stops reading it (commit 6).
    supported_hessian_modes = SUPPORTED_HESSIAN_MODES

    @staticmethod
    def _normalize_device(device):
        # FAIR Chemistry's MLIP unit accepts only "cpu" or "cuda".
        # Keep UMA's historical behavior: CUDA-like requests use the CUDA
        # backend token, while other strings fall back to CPU.
        device_name = str(device).lower()
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _build_predictor(
        checkpoint: str,
        overrides,
        device: str,
        checkpoint_path=None,
        inference_settings=None,
    ):
        device = UMACalculator._normalize_device(device)
        # Turbo selects FAIR Chemistry's fast GPU execution path; CPU uses the
        # general-purpose backend to avoid Triton GPU kernels on CPU tensors.
        if device == "cpu":
            if inference_settings == "turbo":
                warnings.warn(
                    "UMA 'turbo' inference requires CUDA; falling back to 'default' on CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            inference_settings = UMA_CPU_INFERENCE_SETTINGS
        elif inference_settings is None:
            inference_settings = UMA_INFERENCE_SETTINGS

        if checkpoint_path and os.path.isfile(checkpoint_path):
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_path)
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )

        if checkpoint in pretrained_mlip.available_models:
            return pretrained_mlip.get_predict_unit(
                checkpoint,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )

        if os.path.isfile(checkpoint):
            compat_path = UMACalculator._prepare_compat_checkpoint(Path(checkpoint).stem, checkpoint)
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )

        if checkpoint in UMA_FALLBACK_HF_MODELS:
            checkpoint_file = hf_hub_download(
                repo_id="facebook/UMA",
                subfolder="checkpoints",
                filename=f"{checkpoint}.pt",
                cache_dir=CACHE_DIR,
            )
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_file)
            atom_refs = OmegaConf.load(
                hf_hub_download(
                    repo_id="facebook/UMA",
                    subfolder="references",
                    filename="iso_atom_elem_refs.yaml",
                    cache_dir=CACHE_DIR,
                )
            )
            form_elem_refs = OmegaConf.load(
                hf_hub_download(
                    repo_id="facebook/UMA",
                    subfolder="references",
                    filename="form_elem_refs.yaml",
                    cache_dir=CACHE_DIR,
                )
            )["refs"]
            return load_predict_unit(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
            )

        raise ValueError(
            f"Unsupported UMA checkpoint '{checkpoint}'. "
            f"Built-in models: {sorted(pretrained_mlip.available_models)}"
        )

    @staticmethod
    def _prepare_compat_checkpoint(checkpoint_name: str, checkpoint_path: str) -> str:
        compat_dir = Path(CACHE_DIR) / "maple_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        compat_path = compat_dir / f"{checkpoint_name}.pt"

        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = raw_checkpoint.model_config
        model_config.pop("model_id", None)
        model_config.pop("supports_single_atoms", None)

        backbone = model_config.get("backbone", {})
        backbone.pop("charge_balanced_channels", None)
        backbone.pop("composition_dropout", None)
        dataset_mapping = backbone.pop("dataset_mapping", None)
        if dataset_mapping is not None and "dataset_list" not in backbone:
            backbone["dataset_list"] = list(dataset_mapping)

        for head_config in model_config.get("heads", {}).values():
            head_mapping = head_config.pop("dataset_mapping", None)
            if head_mapping is not None and "dataset_names" not in head_config:
                head_config["dataset_names"] = list(head_mapping)

        torch.save(raw_checkpoint, compat_path)
        return str(compat_path)

    def __init__(
        self,
        device,
        model: str = "uma",
        overrides=None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        task=None,
        size=None,
        checkpoint_path=None,
        inference_settings=None,
    ):
        if size is not None:
            size = str(size).lower()
        checkpoint = UMA_MODELS_MAP.get(size, size) if size else UMA_MODELS_MAP.get(model, UMA_DEFAULT_SIZE)

        if task is not None:
            task = str(task).lower()
            if task not in SUPPORTED_UMA_TASKS:
                raise ValueError(f"Unsupported UMA task: '{task}'. Supported: {sorted(SUPPORTED_UMA_TASKS)}")

        if inference_settings is not None:
            inference_settings = str(inference_settings).lower()
            if inference_settings not in SUPPORTED_UMA_INFERENCE:
                raise ValueError(
                    f"Unsupported UMA inference mode: '{inference_settings}'. "
                    f"Supported: {sorted(SUPPORTED_UMA_INFERENCE)}"
                )

        device = self._normalize_device(device)

        if not importlib.util.find_spec("fairchem"):
            raise ImportError("fairchem-core is not installed. Please install it first.")

        predictor = self._build_predictor(
            checkpoint,
            overrides,
            device,
            checkpoint_path=checkpoint_path,
            inference_settings=inference_settings,
        )
        super().__init__(predict_unit=predictor, task_name=task or "omol")

        self.device = torch.device(device)
        self._predictor_unit = predictor
        self._auto_task = task is None
        self.hessian = "numerical"

        # Shared helper sets self.solvent_correction (and self.chargecalc when
        # applicable); identical contract to CalcABC.implicit_solv_init.
        init_implicit_solvent(self, implicit, solvent, self.device)

    def _set_task_from_atoms(self, atoms: Atoms) -> None:
        if not self._auto_task:
            return

        task = "omat" if any(atoms.pbc) else "omol"
        if task == self.task_name:
            return

        self._task = UMATask(task)
        if self._predictor_unit.inference_settings.external_graph_gen:
            r_edges, max_neigh = True, 300
        else:
            r_edges, max_neigh = False, None

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=task,
            r_edges=r_edges,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius=6.0,
        )
        self.task_name = task

    def get_hessian(self, atoms: Atoms, delta: float = 0.002) -> np.ndarray:
        """Numerical-only Hessian via shared finite-difference helper."""
        return numerical_hessian_from_atoms(self, atoms, delta)

    def calculate(self, atoms, properties=None, system_changes=None):
        self._set_task_from_atoms(atoms)

        atoms.info["spin"] = int(atoms.info.get("mult", 1))
        atoms.info["charge"] = int(atoms.info.get("charge", 0))

        super().calculate(atoms, properties, system_changes)

        # eV → Hartree: UMA's MODEL_ENERGY_UNIT is 'eV'; equivalent to the
        # _finalize_results unit step but inlined because UMA does not inherit
        # CalcABC.
        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        # Implicit-solvent correction in Hartree (mirrors _finalize_results).
        if self.solvent_correction is not None:
            atoms.atomic_charges = self.chargecalc(atoms)
            solvent_energy, solvent_force = self.solvent_correction.get_energy_and_force(atoms)
            if "energy" in self.results:
                self.results["energy"] += solvent_energy.item()
            if "free_energy" in self.results:
                self.results["free_energy"] += solvent_energy.item()
            if "forces" in self.results:
                self.results["forces"] += solvent_force.detach().cpu().numpy()

        return self.results
