import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
import torch
from ase import Atoms

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator


IMPLEMENTATION_MODELS = [
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "aimnet2",
    "aimnet2nse",
    "uma",
    "maceomol",
    "macepols",
    "macepolm",
    "macepoll",
]

MODEL_NAME_TO_FILE = {
    "ani2x": "ani2x.pt",
    "ani1x": "ani1x.pt",
    "ani1ccx": "ani1ccx.pt",
    "ani1xnr": "ani1xnr.pt",
    "aimnet2": "aimnet2.pt",
    "aimnet2nse": "aimnet2nse.pt",
    "maceoff23m": "maceoff23m.pt",
    "egret": "egret1s.pt",
}

HF_REPO_ID = "Wayne7815/MAPLE_models"

MODEL_HESSIAN_SUPPORT = {
    "ani2x": ("analytic", "numerical"),
    "ani1x": ("analytic", "numerical"),
    "ani1ccx": ("analytic", "numerical"),
    "ani1xnr": ("analytic", "numerical"),
    "maceoff23s": ("analytic", "numerical"),
    "maceoff23m": ("analytic", "numerical"),
    "maceoff23l": ("analytic", "numerical"),
    "egret": ("analytic", "numerical"),
    "aimnet2": ("analytic", "numerical"),
    "aimnet2nse": ("analytic", "numerical"),
    "uma": ("numerical",),
    "maceomol": ("analytic", "numerical"),
    "macepols": ("analytic", "numerical"),
    "macepolm": ("analytic", "numerical"),
    "macepoll": ("analytic", "numerical"),
}

UNSUPPORTED_CHARGE_MULT_MODELS = {
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "maceomol",
}


def _model_download_url(filename: str) -> str:
    return f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}"


class SetClaculator:
    def __init__(
        self,
        device: torch.device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = "None",
        solvent: str = "None",
        model_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent
        self.model_options = model_options or {}
        self._model_error_logged = False

    def _model_dir(self) -> Path:
        return Path(__file__).parent / "model"

    def _model_dir_description(self) -> str:
        model_dir = self._model_dir()
        resolved_dir = model_dir.resolve()
        if resolved_dir != model_dir:
            return f"{model_dir} (resolved: {resolved_dir})"
        return str(model_dir)

    def _log_model_error(self, message: str) -> None:
        self._model_error_logged = True
        self.log_error(message)

    def _validate_requested_hessian_mode(self) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        declared = MODEL_HESSIAN_SUPPORT.get(self.model)
        if declared is not None and mode not in declared:
            supported_text = ", ".join(sorted(declared))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        supported = getattr(calculator, "supported_hessian_modes", None)
        if supported is not None and mode not in supported:
            supported_text = ", ".join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

        if not hasattr(calculator, "hessian"):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def _coerce_uma_inference_for_device(
        self, inference: Optional[str], device_name: str
    ) -> Optional[str]:
        if inference == "turbo" and device_name == "cpu":
            self.log_info([
                "\n [WARNING] UMA inference='turbo' requires CUDA; "
                "falling back to 'default' on CPU.\n"
            ])
            return "default"
        return inference

    def _ensure_model_file(self, model_name: str) -> Optional[Path]:
        filename = MODEL_NAME_TO_FILE.get(model_name)
        if filename is None:
            return None

        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        url = _model_download_url(filename)
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f" [INFO] Downloading from: {url}\n",
            ]
        )

        temp_path = model_path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url) as response:
                size = response.headers.get("Content-Length")
                if size:
                    self.log_info([f" [INFO] File size: {int(size) / 1024 / 1024:.1f} MB\n"])
                with open(temp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.rename(model_path)
            self.log_info([f" [INFO] Download complete: {model_path}\n"])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Download failed for model '{model_name}' (HTTP {exc.code}): {url}\n"
                f"       MAPLE model directory: {self._model_dir_description()}"
            )
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Network error while downloading model '{model_name}': {exc.reason}\n"
                f"       MAPLE model directory: {self._model_dir_description()}"
            )
            raise RuntimeError(f"Failed to download model '{model_name}': {exc.reason}") from exc

        return model_path

    def _local_model_file(self, filename: str) -> Optional[Path]:
        model_path = self._model_dir() / filename
        return model_path if model_path.exists() else None

    def _require_local_model_file(self, model_name: str, filename: Optional[str] = None) -> Path:
        filename = filename or f"{model_name}.pt"
        model_path = self._model_dir() / filename
        if model_path.exists():
            return model_path

        message = (
            f"Model file '{filename}' for '{model_name}' was not found locally.\n"
            f"       MAPLE model directory searched: {self._model_dir_description()}\n"
            f"       Expected file path: {model_path}\n"
            f"This model is not available from {HF_REPO_ID}; install the backend-specific model "
            "file or pass an explicit model_path when supported."
        )
        self._log_model_error(message)
        raise FileNotFoundError(message)

    def _warn_charge_mult(self) -> None:
        if self.atoms is None:
            return

        has_charge = self.atoms.info.get("charge", 0) != 0
        has_mult = self.atoms.info.get("mult", 1) != 1
        if (has_charge or has_mult) and self.model in UNSUPPORTED_CHARGE_MULT_MODELS:
            self.log_info(
                [
                    f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n",
                    f"           charge={self.atoms.info.get('charge', 0)}, ",
                    f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n",
                    "           Models with charge/mult support: aimnet2, aimnet2nse, uma, macepols/m/l\n",
                ]
            )

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        model = self.model

        if model in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
            self._ensure_model_file(model)
            calculator = ANICalculator(
                model=model,
                d4=self.d4,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"maceoff23s", "maceoff23m", "maceoff23l", "egret"}:
            model_path = self._ensure_model_file(model)
            if model_path is None:
                model_path = self._require_local_model_file(model)
            calculator = MACECalculator(
                model=model,
                model_path=str(model_path),
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"aimnet2", "aimnet2nse"}:
            self._ensure_model_file(model)
            from .aimnet._aimnet2_calculator import AIMNet2Calculator

            calculator = AIMNet2Calculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model == "uma":
            from .uma._uma_calculator import UMACalculator, UMA_DEFAULT_SIZE, UMA_FALLBACK_HF_MODELS

            uma_task = self.model_options.get("task")
            uma_size = self.model_options.get("size")
            uma_inference = self.model_options.get("inference")
            effective_device = UMACalculator._normalize_device(self.device)
            uma_inference = self._coerce_uma_inference_for_device(
                uma_inference, effective_device
            )

            checkpoint_path = self.model_options.get("checkpoint_path") or self.model_options.get("model_path")
            effective_size = str(uma_size).lower() if uma_size else UMA_DEFAULT_SIZE
            if checkpoint_path is None and effective_size in UMA_FALLBACK_HF_MODELS:
                local_checkpoint = self._local_model_file(f"{effective_size}.pt")
                if local_checkpoint is not None:
                    checkpoint_path = str(local_checkpoint)

            calculator = UMACalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
                task=uma_task,
                size=uma_size,
                checkpoint_path=checkpoint_path,
                inference_settings=uma_inference,
            )
        elif model == "maceomol":
            self._require_local_model_file(model)
            from .mace._mace_general_calculator import MACEModelCalculator

            calculator = MACEModelCalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"macepols", "macepolm", "macepoll"}:
            from .mace._macepol_calculator import MACEPolCalculator

            model_path = self.model_options.get("model_path")
            if model_path is None:
                model_path = str(self._require_local_model_file(model))

            calculator = MACEPolCalculator(
                model=model,
                device=self.device,
                model_path=model_path,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        else:
            raise ValueError(f"Model '{model}' is not implemented yet.")

        self._apply_hessian_mode(calculator)
        return calculator

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        try:
            if self.model not in IMPLEMENTATION_MODELS:
                self._log_model_error(f"Unsupported model: {self.model}")
                raise ValueError(f"Unsupported model: '{self.model}'.")

            self._validate_requested_hessian_mode()

            if self.d4 and self.model not in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
                self.log_info([f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"])

            calculator = self._build_calculator()
            self._warn_charge_mult()
            return calculator
        except Exception as exc:
            if not self._model_error_logged:
                self._log_model_error(
                    f"Failed to initialize model '{self.model}'.\n"
                    f"       MAPLE model directory: {self._model_dir_description()}\n"
                    f"       Error: {type(exc).__name__}: {exc}"
                )
            raise

    def log_error(self, error_message: str) -> None:
        with open(self.output, "a") as handle:
            handle.write(f"ERROR: {error_message.rstrip()}\n")

    def log_info(self, info_message: list) -> None:
        with open(self.output, "a") as handle:
            for line in info_message:
                handle.write(line)
