import importlib
import os
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


EV2HARTREE = 1.0 / 27.211386245988

UMA_DEFAULT_SIZE = "uma-s-1p1"
UMA_MODELS_MAP = {
    "uma": UMA_DEFAULT_SIZE,
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
}

UMA_FALLBACK_HF_MODELS = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}

# FAIR Chemistry's turbo mode is optimized for repeated evaluations on a
# fixed-composition system, which matches MAPLE's NEB/TS/freq workloads.
UMA_INFERENCE_SETTINGS = "turbo"


class UMACalculator(FAIRChemCalculator):
    """
    UMA calculator with MAPLE-specific unit conversion and Hessian support.

    Explicit `task=` is respected. When `task` is omitted, MAPLE keeps the
    historical convenience behavior of inferring `omol` for non-periodic
    systems and `omat` for periodic systems.
    """

    supported_hessian_modes = ("numerical",)

    @staticmethod
    def _build_predictor(
        checkpoint: str,
        overrides: dict | None,
        device: str,
        checkpoint_path: str | None = None,
    ):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_path)
            return load_predict_unit(
                compat_path,
                inference_settings=UMA_INFERENCE_SETTINGS,
                overrides=overrides,
                device=device,
            )

        if checkpoint in pretrained_mlip.available_models:
            return pretrained_mlip.get_predict_unit(
                checkpoint,
                inference_settings=UMA_INFERENCE_SETTINGS,
                overrides=overrides,
                device=device,
            )

        if os.path.isfile(checkpoint):
            compat_path = UMACalculator._prepare_compat_checkpoint(Path(checkpoint).stem, checkpoint)
            return load_predict_unit(
                compat_path,
                inference_settings=UMA_INFERENCE_SETTINGS,
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
                inference_settings=UMA_INFERENCE_SETTINGS,
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
        device: torch.device,
        model: str = "uma",
        overrides: dict | None = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = "none",
        task: str | None = None,
        size: str | None = None,
        checkpoint_path: str | None = None,
    ):
        if size is not None:
            size = str(size).lower()
        checkpoint = UMA_MODELS_MAP.get(size, size) if size else UMA_MODELS_MAP.get(model, UMA_DEFAULT_SIZE)

        if task is not None:
            task = str(task).lower()
            if task not in SUPPORTED_UMA_TASKS:
                raise ValueError(f"Unsupported UMA task: '{task}'. Supported: {sorted(SUPPORTED_UMA_TASKS)}")

        device = str(device)
        device = "cuda" if device.startswith("cuda") else "cpu"

        if not importlib.util.find_spec("fairchem"):
            raise ImportError("fairchem-core is not installed. Please install it first.")

        predictor = self._build_predictor(
            checkpoint,
            overrides,
            device,
            checkpoint_path=checkpoint_path,
        )
        super().__init__(predict_unit=predictor, task_name=task or "omol")

        self.device = torch.device(device)
        self._predictor_unit = predictor
        self._auto_task = task is None
        self.hessian = "numerical"

        if implicit == "gbsa" and solvent != "none":
            from ..extra_correction import GBSA, QEqTorch

            self.solvent_correction = GBSA(solvent=solvent, device=self.device)
            self.chargecalc = QEqTorch(device=self.device)
        else:
            self.solvent_correction = None

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

    def get_energy(self, atoms: Atoms) -> torch.Tensor:
        self.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy_value = self.results["energy"]

        if self.solvent_correction:
            energy_value += self.solvent_correction.get_energy(atoms)

        return torch.tensor(energy_value, dtype=torch.float32, device=self.device)

    def get_hessian(
        self,
        atoms: Atoms,
        delta: float = 0.002,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        from ase.constraints import FixAtoms

        n_atoms = len(atoms)
        pos0 = atoms.get_positions()
        fixed = {
            i
            for constraint in atoms.constraints
            if isinstance(constraint, FixAtoms)
            for i in constraint.get_indices()
        }
        movable = [i for i in range(n_atoms) if i not in fixed]

        if not movable:
            return torch.zeros((3 * n_atoms, 3 * n_atoms), dtype=dtype, device=self.device)

        hessian = torch.zeros((3 * n_atoms, 3 * n_atoms), dtype=dtype, device=self.device)

        def eval_force(positions: np.ndarray) -> torch.Tensor:
            atoms_tmp = atoms.copy()
            atoms_tmp.set_positions(positions)
            self.calculate(atoms_tmp, properties=["forces"], system_changes=all_changes)
            return torch.tensor(self.results["forces"], dtype=dtype, device=self.device)

        for atom_index in movable:
            for axis in range(3):
                pos_p = pos0.copy()
                pos_p[atom_index, axis] += delta
                force_p = eval_force(pos_p)

                pos_m = pos0.copy()
                pos_m[atom_index, axis] -= delta
                force_m = eval_force(pos_m)

                row = 3 * atom_index + axis
                hessian[row, :] = (-(force_p - force_m) / (2.0 * delta)).reshape(-1)

        return hessian

    def calculate(self, atoms, properties=None, system_changes=None):
        self._set_task_from_atoms(atoms)

        atoms.info["spin"] = int(atoms.info.get("mult", 1))
        atoms.info["charge"] = int(atoms.info.get("charge", 0))

        super().calculate(atoms, properties, system_changes)

        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        if self.solvent_correction:
            atoms.atomic_charges = self.chargecalc(atoms)
            solvent_energy, solvent_force = self.solvent_correction.get_energy_and_force(atoms)
            if "energy" in self.results:
                self.results["energy"] += solvent_energy.item()
            if "free_energy" in self.results:
                self.results["free_energy"] += solvent_energy.item()
            if "forces" in self.results:
                self.results["forces"] += solvent_force.detach().cpu().numpy()

        return self.results
