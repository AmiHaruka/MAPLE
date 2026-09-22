from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core._config import CACHE_DIR
    from fairchem.core.calculate.ase_calculator import (
        AtomicData,
        FAIRChemCalculator,
        UMATask,
    )
    try:
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    except ImportError:
        atomicdata_list_to_batch = None
    from fairchem.core.units.mlip_unit import load_predict_unit
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
)
from ..calculator_base import (
    EV2HARTREE,
    init_implicit_solvent,
    register_calculator,
    reject_implicit_solvent_derivatives,
)
from ..electronic_state import (
    attach_calculator_identity,
    checkpoint_fingerprint,
    requested_electronic_state,
    solvation_identity_settings,
    validate_electronic_state,
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
PERIODIC_UMA_TASKS = SUPPORTED_UMA_TASKS - {"omol"}
SUPPORTED_UMA_INFERENCE = {"default", "turbo"}

# GPU default and CPU fallback for FAIR Chemistry's inference path. "default"
# is the general-purpose backend; "turbo" speeds up fixed-composition workloads
# (NEB / TS / freq) on CUDA but routes through Triton kernels unavailable on
# CPU. Users override via `model=uma(...,inference=turbo)`; CPU coerces to
# default regardless.
UMA_INFERENCE_SETTINGS = "default"
UMA_CPU_INFERENCE_SETTINGS = "default"
UMA_COMPAT_SCHEMA = "v2"


def _omega_mapping(value, name: str) -> dict[Any, Any]:
    plain = OmegaConf.to_container(value, resolve=True)
    if not isinstance(plain, dict):
        raise TypeError(f"UMA {name} must be a mapping.")
    return dict(plain)


def _reference_values(value) -> dict[Any, Any]:
    mapping = _omega_mapping(value, "form-element reference data")
    refs = mapping.get("refs")
    if not isinstance(refs, dict):
        raise TypeError("UMA form-element reference data must contain a 'refs' mapping.")
    return dict(refs)


@dataclass(frozen=True)
class _LoadedPredictor:
    predictor: Any
    checkpoint_path: str
    model_fingerprint: dict[str, str]
    inference: str
    reference_energies: dict[str, object | None]


def _load_with_fingerprint(path: str, **kwargs) -> tuple[Any, dict[str, str]]:
    """Load an immutable resolved artifact and freeze its content identity."""
    fingerprint = checkpoint_fingerprint(path)
    return load_predict_unit(path, **kwargs), fingerprint


@register_calculator
class UMACalculator(FAIRChemCalculator):
    """UMA calculator with MAPLE-specific unit conversion and Hessian support.

    Does NOT inherit CalcABC — UMA already extends third-party FAIRChemCalculator.
    Satisfies the MAPLE calculator protocol via attribute presence (class
    capability attrs + calculate + get_hessian + get_hvp where required).

    Explicit `task=` is respected. When `task` is omitted, MAPLE only infers
    `omol` for non-periodic systems; periodic UMA requires an explicit FAIR-Chem
    task because `pbc -> omat` is too broad for production use.
    """

    MODEL_NAMES = ("uma",)
    MODEL_ENERGY_UNIT = "eV"
    SUPPORTED_HESSIAN_MODES = ("numerical",)
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = True
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = (
        'task',
        'size',
        'checkpoint_path',
        'inference',
        'overrides',
        'batch_size',
        'path_batch_size',
    )
    MODEL_PATH_OPTION = 'checkpoint_path'
    supports_batch_energy_forces = True
    batch_memory_model = 'disconnected_graph'
    auto_batch_hard_cap = 8
    auto_path_batch_cap = 8

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {
            'task': options.get('task'),
            'size': options.get('size'),
            'checkpoint_path': options.get('checkpoint_path') or resolved_model_path,
            'inference_settings': options.get('inference'),
            'overrides': options.get('overrides'),
            'batch_size': options.get('batch_size'),
            'path_batch_size': options.get('path_batch_size'),
        }

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
            predictor, fingerprint = _load_with_fingerprint(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
            return _LoadedPredictor(
                predictor, compat_path, fingerprint, inference_settings,
                {"atom_refs": None, "form_elem_refs": None},
            )

        if checkpoint in pretrained_mlip.available_models:
            resolved_path = pretrained_mlip.pretrained_checkpoint_path_from_name(checkpoint)
            atom_refs = _omega_mapping(
                pretrained_mlip.get_reference_energies(checkpoint, "atom_refs"),
                "atomic reference data",
            )
            form_elem_refs = None
            if checkpoint in UMA_FALLBACK_HF_MODELS:
                form_elem_refs = _reference_values(
                    pretrained_mlip.get_reference_energies(checkpoint, "form_elem_refs")
                )
            predictor, fingerprint = _load_with_fingerprint(
                resolved_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
            )
            return _LoadedPredictor(
                predictor,
                resolved_path,
                fingerprint,
                inference_settings,
                {"atom_refs": atom_refs, "form_elem_refs": form_elem_refs},
            )

        if os.path.isfile(checkpoint):
            compat_path = UMACalculator._prepare_compat_checkpoint(Path(checkpoint).stem, checkpoint)
            predictor, fingerprint = _load_with_fingerprint(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
            return _LoadedPredictor(
                predictor, compat_path, fingerprint, inference_settings,
                {"atom_refs": None, "form_elem_refs": None},
            )

        if checkpoint in UMA_FALLBACK_HF_MODELS:
            checkpoint_file = hf_hub_download(
                repo_id="facebook/UMA",
                subfolder="checkpoints",
                filename=f"{checkpoint}.pt",
                cache_dir=CACHE_DIR,
            )
            compat_path = UMACalculator._prepare_compat_checkpoint(checkpoint, checkpoint_file)
            atom_refs = _omega_mapping(
                OmegaConf.load(
                    hf_hub_download(
                        repo_id="facebook/UMA",
                        subfolder="references",
                        filename="iso_atom_elem_refs.yaml",
                        cache_dir=CACHE_DIR,
                    )
                ),
                "atomic reference data",
            )
            form_elem_refs = _reference_values(
                OmegaConf.load(
                    hf_hub_download(
                        repo_id="facebook/UMA",
                        subfolder="references",
                        filename="form_elem_refs.yaml",
                        cache_dir=CACHE_DIR,
                    )
                )
            )
            predictor, fingerprint = _load_with_fingerprint(
                compat_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
            )
            return _LoadedPredictor(
                predictor,
                compat_path,
                fingerprint,
                inference_settings,
                {"atom_refs": atom_refs, "form_elem_refs": form_elem_refs},
            )

        raise ValueError(
            f"Unsupported UMA checkpoint '{checkpoint}'. "
            f"Built-in models: {sorted(pretrained_mlip.available_models)}"
        )

    @staticmethod
    def _prepare_compat_checkpoint(checkpoint_name: str, checkpoint_path: str) -> str:
        compat_dir = Path(CACHE_DIR) / "maple_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(checkpoint_path)
        with source_path.open("rb") as source:
            source_stat = os.fstat(source.fileno())
            source_identity = (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
            )
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            final_stat = os.fstat(source.fileno())
            if source_identity != (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
            ):
                raise RuntimeError("UMA checkpoint changed while its content identity was read.")

            safe_name = Path(str(checkpoint_name)).name
            compat_path = compat_dir / (
                f"{safe_name}-{UMA_COMPAT_SCHEMA}-{digest.hexdigest()}.pt"
            )
            if compat_path.is_file():
                return str(compat_path)

            source.seek(0)
            raw_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            loaded_stat = os.fstat(source.fileno())
            if source_identity != (
                loaded_stat.st_dev,
                loaded_stat.st_ino,
                loaded_stat.st_size,
                loaded_stat.st_mtime_ns,
            ):
                raise RuntimeError("UMA checkpoint changed while it was loaded.")
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

            descriptor, temp_name = tempfile.mkstemp(
                dir=compat_dir, prefix=f".{safe_name}-", suffix=".tmp"
            )
            os.close(descriptor)
            temp_path = Path(temp_name)
            try:
                # Passing a file object keeps the temporary pathname out of
                # PyTorch's ZIP archive metadata, so equivalent compatibility
                # artifacts have a stable content fingerprint across caches.
                with temp_path.open("wb") as temp_file:
                    torch.save(raw_checkpoint, temp_file)
                try:
                    os.link(temp_path, compat_path)
                except FileExistsError:
                    pass
            finally:
                temp_path.unlink(missing_ok=True)
            return str(compat_path)

    def __init__(
        self,
        device,
        model: str = "uma",
        overrides=None,
        implicit: Literal["gbsa", "none"] = "none",
        solvent: str = "none",
        task=None,
        size=None,
        checkpoint_path=None,
        inference_settings=None,
        batch_size=None,
        path_batch_size=None,
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

        loaded = self._build_predictor(
            checkpoint,
            overrides,
            device,
            checkpoint_path=checkpoint_path,
            inference_settings=inference_settings,
        )
        predictor = loaded.predictor
        super().__init__(predict_unit=predictor, task_name=task or "omol")

        self.device = torch.device(device)
        self._predictor_unit = predictor
        self._auto_task = task is None
        self.hessian = "numerical"
        self.batch_size = batch_size
        self.path_batch_size = path_batch_size
        self._maple_inference_settings = inference_settings

        attach_calculator_identity(
            self,
            backend=f"uma:{checkpoint}",
            model_fingerprint=loaded.model_fingerprint,
            relevant_settings={
                "task": task or "omol",
                "inference": loaded.inference,
                "overrides": overrides or {},
                "reference_energies": loaded.reference_energies,
                **solvation_identity_settings(implicit, solvent),
            },
        )

        # Shared helper sets self.solvent_correction (and self.chargecalc when
        # applicable); identical contract to CalcABC.implicit_solv_init.
        init_implicit_solvent(self, implicit, solvent, self.device)

    def _set_task_from_atoms(self, atoms: Atoms) -> None:
        if not self._auto_task:
            return

        if any(atoms.pbc):
            raise ValueError(
                "UMA periodic calculations require an explicit FAIR-Chem task "
                "(for example task=omat, oc20, oc22, oc25, omc, or odac). "
                "MAPLE no longer silently maps every periodic system to task='omat'."
            )

        task = "omol"
        if task == self.task_name:
            return

        self._task = UMATask(task)
        self._task_name = task
        identity = getattr(self, "maple_pes_identity", None)
        if identity is not None:
            identity["relevant_settings"]["task"] = task
        self.implemented_properties = [
            task_obj.property for task_obj in self.predictor.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append("free_energy")

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
            target_dtype=self._predictor_unit.inference_settings.base_precision_dtype,
        )

    def _validate_task_atoms_compatibility(self, atoms: Atoms) -> None:
        if not any(atoms.pbc):
            return

        if self.task_name == "omol":
            raise ValueError(
                "UMA task='omol' is molecular and must not be used with periodic atoms. "
                "Use an explicit periodic/domain task such as task=omat, oc20, oc22, "
                "oc25, omc, or odac after confirming the system domain."
            )

        if self.task_name not in PERIODIC_UMA_TASKS:
            raise ValueError(
                f"UMA task='{self.task_name}' is not registered as a MAPLE periodic task; "
                f"supported periodic tasks: {sorted(PERIODIC_UMA_TASKS)}."
            )

    def electronic_state_settings(self) -> dict[str, str]:
        return {"task": self.task_name}

    @staticmethod
    def validate_electronic_state_request(charge, multiplicity, settings) -> None:
        task = settings.get("task") or "omol"
        if task != "omol" and (charge != 0 or multiplicity != 1):
            raise ValueError(
                f"UMA task='{task}' does not support the requested electronic state "
                f"(charge={charge}, mult={multiplicity}); use task='omol' for "
                "charge/spin-dependent molecular calculations."
            )

    def _validate_charge_spin_task_compatibility(self, atoms: Atoms) -> tuple[int, int]:
        charge, mult = validate_electronic_state(atoms, self)
        has_charge = charge != 0
        has_open_shell = mult != 1

        if self.task_name == "omol":
            if has_charge or has_open_shell:
                message = (
                    "UMA omol charged/open-shell inputs are passed through to FAIR-Chem, "
                    "but MAPLE has not yet accepted golden numerical tolerances for these "
                    "states; compare against FAIR-Chem/reference calculations before "
                    "production use."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            return charge, mult

        return charge, mult

    def get_hessian(self, atoms: Atoms, delta: float = 0.002) -> np.ndarray:
        """Numerical-only Hessian via shared finite-difference helper."""
        from .._batch_eval import FDHessianEvaluator

        self._set_task_from_atoms(atoms)
        self._validate_task_atoms_compatibility(atoms)
        self._validate_charge_spin_task_compatibility(atoms)
        return FDHessianEvaluator(
            self,
            fd_batch_size=getattr(self, "fd_batch_size", None),
        ).hessian(atoms, delta)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate non-periodic UMA structures through FAIR-Chem's batch data path.

        Native batching is limited to the validated molecular/no-solvent path.
        Periodic structures, single atoms, unavailable FAIR-Chem batch helpers,
        and experimental implicit solvent all fall back to ``calculate(...)`` so
        UMA's task/PBC/charge/spin/unit semantics stay identical to the
        single-structure calculator.
        """
        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if (
            atomicdata_list_to_batch is None
            or atoms_list_has_pbc(atoms_list)
            or getattr(self, "solvent_correction", None) is not None
            or any(len(at) == 1 for at in atoms_list)
            or self._maple_inference_settings == "turbo"
            or any(
                requested_electronic_state(at) != (0, 1)
                for at in atoms_list
            )
        ):
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )

        calc_atoms_list = []
        for at in atoms_list:
            self._set_task_from_atoms(at)
            self._validate_task_atoms_compatibility(at)
            charge, mult = self._validate_charge_spin_task_compatibility(at)

            calc_atoms = at.copy()
            calc_atoms.info["spin"] = mult
            calc_atoms.info["charge"] = charge
            self._check_atoms_pbc(calc_atoms)
            calc_atoms_list.append(calc_atoms)

        data_list = [self.a2g(at) for at in calc_atoms_list]
        batch = atomicdata_list_to_batch(data_list)
        pred = self.predictor.predict(batch)

        energies = None
        if want_energy:
            energies = (
                pred["energy"].detach().cpu().numpy().astype(np.float64)
                * EV2HARTREE
            )

        forces_list = None
        if want_forces:
            forces_all = (
                pred["forces"].detach().cpu().numpy().astype(np.float64)
                * EV2HARTREE
            )
            batch_index = batch.batch.detach().cpu().numpy()
            forces_list = [
                forces_all[batch_index == i]
                for i in range(len(atoms_list))
            ]

        return BatchResult(
            energies=energies,
            forces=forces_list,
        ).validate_against(atoms_list, request)

    def calculate(self, atoms, properties=None, system_changes=None):
        properties = reject_implicit_solvent_derivatives(self, properties)
        system_changes = all_changes if system_changes is None else system_changes

        if atoms is None:
            atoms = getattr(self, "atoms", None)
        if atoms is None:
            raise ValueError("UMACalculator.calculate requires an Atoms object.")

        requested = {str(prop).lower() for prop in properties}
        if requested & {"stress", "stresses", "virial", "virials"}:
            raise NotImplementedError(
                "UMA stress/virial output is not unit-converted by MAPLE yet; "
                "request energy/forces only until stress units are validated."
            )

        self._set_task_from_atoms(atoms)
        self._validate_task_atoms_compatibility(atoms)
        charge, mult = self._validate_charge_spin_task_compatibility(atoms)

        calc_atoms = atoms.copy()
        calc_atoms.info["spin"] = mult
        calc_atoms.info["charge"] = charge

        super().calculate(calc_atoms, properties, system_changes)

        # eV → Hartree: UMA's MODEL_ENERGY_UNIT is 'eV'; equivalent to the
        # _finalize_results unit step but inlined because UMA does not inherit
        # CalcABC.
        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        # Experimental implicit solvation is energy-only. Derivative requests
        # have already failed via reject_implicit_solvent_derivatives().
        if self.solvent_correction is not None:
            calc_atoms.atomic_charges = self.chargecalc(calc_atoms, total_charge=float(charge))
            solvent_energy, _ = self.solvent_correction.get_energy(calc_atoms)
            if "energy" in self.results:
                self.results["energy"] += solvent_energy.item()
            if "free_energy" in self.results:
                self.results["free_energy"] += solvent_energy.item()

        return self.results
