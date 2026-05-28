from __future__ import annotations

import importlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
from ase import Atoms

from .calculator_base import (
    get_registered_calculator,
    import_calculator_plugin,
    load_calculator_plugins_from_env,
)


HF_REPO_ID = 'Wayne7815/MAPLE_models'


# Static seed map: builtin name → module that registers the calculator class.
# External users add backends via input-header `module=` or the
# MAPLE_CALCULATOR_PLUGINS env var; this dict is for shipped backends only.
_BUILTIN_NAME_TO_MODULE = {
    'ani2x': 'maple.function.calculator.ani._ani_calculator',
    'ani1x': 'maple.function.calculator.ani._ani_calculator',
    'ani1ccx': 'maple.function.calculator.ani._ani_calculator',
    'ani1xnr': 'maple.function.calculator.ani._ani_calculator',
    'aimnet2': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2nse': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'maceoff23s': 'maple.function.calculator.mace._mace_calculator',
    'maceoff23m': 'maple.function.calculator.mace._mace_calculator',
    'maceoff23l': 'maple.function.calculator.mace._mace_calculator',
    'egret': 'maple.function.calculator.mace._mace_calculator',
    'maceomol': 'maple.function.calculator.mace._mace_general_calculator',
    'macepols': 'maple.function.calculator.mace._macepol_calculator',
    'macepolm': 'maple.function.calculator.mace._macepol_calculator',
    'macepoll': 'maple.function.calculator.mace._macepol_calculator',
    'uma': 'maple.function.calculator.uma._uma_calculator',
}


_plugins_loaded_from_env = False


def _model_download_url(filename: str) -> str:
    return f'https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}'


class SetCalculator:
    def __init__(
        self,
        device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = 'None',
        solvent: str = 'None',
        model_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = str(model).lower()
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent
        self.model_options = model_options or {}
        self._model_error_logged = False

    def _model_dir(self) -> Path:
        return Path(__file__).parent / 'model'

    def _model_dir_description(self) -> str:
        model_dir = self._model_dir()
        resolved_dir = model_dir.resolve()
        if resolved_dir != model_dir:
            return f'{model_dir} (resolved: {resolved_dir})'
        return str(model_dir)

    def _log_model_error(self, message: str) -> None:
        self._model_error_logged = True
        self.log_error(message)

    def _discover_calculator_class(self, name: str):
        """Resolve `name` to a registered calculator class.

        Plug-in discovery layers:
        1. Explicit `module=` in model_options (run import_calculator_plugin).
        2. _BUILTIN_NAME_TO_MODULE seed for shipped backends.
        3. MAPLE_CALCULATOR_PLUGINS env var (loaded once per process).
        """
        global _plugins_loaded_from_env
        if not _plugins_loaded_from_env:
            load_calculator_plugins_from_env()
            _plugins_loaded_from_env = True

        normalized = name.lower()
        module_override = self.model_options.get('module')
        if module_override:
            import_calculator_plugin(str(module_override))
        else:
            builtin_module = _BUILTIN_NAME_TO_MODULE.get(normalized)
            if builtin_module is not None:
                importlib.import_module(builtin_module)

        try:
            return get_registered_calculator(normalized)
        except KeyError:
            raise ValueError(f"Unsupported model: '{name}'.")

    def _validate_against_class(self, cls) -> None:
        """Pre-instantiation gates: hessian mode, charge/mult, d4."""
        mode = self.model_options.get('hessian')
        if mode is not None:
            mode = str(mode).lower()
            if mode not in cls.SUPPORTED_HESSIAN_MODES:
                supported_text = ', '.join(sorted(cls.SUPPORTED_HESSIAN_MODES))
                raise ValueError(
                    f"Model '{self.model}' does not support hessian='{mode}'. "
                    f'Supported modes: {supported_text}'
                )

        if self.atoms is not None:
            has_charge = self.atoms.info.get('charge', 0) != 0
            has_mult = self.atoms.info.get('mult', 1) != 1
            if (has_charge or has_mult) and not cls.SUPPORTS_CHARGE_MULT:
                self.log_info(
                    [
                        f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n",
                        f"           charge={self.atoms.info.get('charge', 0)}, ",
                        f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n",
                        '           Models with charge/mult support: aimnet2, aimnet2nse, uma, macepols/m/l\n',
                    ]
                )

        if self.d4:
            import inspect

            ctor_params = inspect.signature(cls.__init__).parameters
            if 'd4' not in ctor_params:
                self.log_info(
                    [f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"]
                )

    def _resolve_model_path(self, cls, name: str) -> Optional[Path]:
        """Resolve checkpoint path per class attrs.

        - If `cls.CHECKPOINT_FILENAME` maps the name → ensure (download or local).
        - Else if `cls.REQUIRES_LOCAL_MODEL_FILE` is True → require local.
        - Else → no path (backend looks up its own default).
        """
        if cls.CHECKPOINT_FILENAME and name in cls.CHECKPOINT_FILENAME:
            filename = cls.CHECKPOINT_FILENAME[name]
            return self._ensure_model_file(filename, name)
        if cls.REQUIRES_LOCAL_MODEL_FILE:
            return self._require_local_model_file(name)
        return None

    def _coerce_uma_inference_for_device(self, inference, device_name: str):
        if inference == 'turbo' and device_name == 'cpu':
            self.log_info(
                [
                    "\n [WARNING] UMA inference='turbo' requires CUDA; "
                    "falling back to 'default' on CPU.\n"
                ]
            )
            return 'default'
        return inference

    def _ensure_model_file(self, filename: str, model_name: str) -> Path:
        """Ensure a HF-hosted checkpoint is on disk; download if missing."""
        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        url = _model_download_url(filename)
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f' [INFO] Downloading from: {url}\n',
            ]
        )

        temp_path = model_path.with_suffix('.tmp')
        try:
            with urllib.request.urlopen(url) as response:
                size = response.headers.get('Content-Length')
                if size:
                    self.log_info([f' [INFO] File size: {int(size) / 1024 / 1024:.1f} MB\n'])
                with open(temp_path, 'wb') as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.rename(model_path)
            self.log_info([f' [INFO] Download complete: {model_path}\n'])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Download failed for model '{model_name}' (HTTP {exc.code}): {url}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Network error while downloading model '{model_name}': {exc.reason}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': {exc.reason}") from exc

        return model_path

    def _local_model_file(self, filename: str) -> Optional[Path]:
        model_path = self._model_dir() / filename
        return model_path if model_path.exists() else None

    def _require_local_model_file(self, model_name: str, filename: Optional[str] = None) -> Path:
        filename = filename or f'{model_name}.pt'
        model_path = self._model_dir() / filename
        if model_path.exists():
            return model_path

        message = (
            f"Model file '{filename}' for '{model_name}' was not found locally.\n"
            f'       MAPLE model directory searched: {self._model_dir_description()}\n'
            f'       Expected file path: {model_path}\n'
            f'This model is not available from {HF_REPO_ID}; install the backend-specific model '
            'file or pass an explicit model_path when supported.'
        )
        self._log_model_error(message)
        raise FileNotFoundError(message)

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        name = self.model

        cls = self._discover_calculator_class(name)
        self._validate_against_class(cls)
        # Lowercase keys for case-insensitive lookup inside build_kwargs_from_options.
        options = {str(k).lower(): v for k, v in self.model_options.items()}
        options.setdefault('d4', self.d4)
        # Allow input header to override the auto-resolved model path.
        if cls.REQUIRES_LOCAL_MODEL_FILE and options.get('model_path'):
            resolved_model_path = Path(str(options['model_path']))
        else:
            resolved_model_path = self._resolve_model_path(cls, name)

        # UMA's `size`/`checkpoint_path` resolution is special: when no checkpoint
        # is given and the requested size has a known HF fallback, prefer a
        # locally-cached MAPLE copy if present. Handle the coercion here so that
        # UMACalculator only receives clean kwargs.
        if cls.__name__ == 'UMACalculator':
            from .uma._uma_calculator import UMACalculator, UMA_DEFAULT_SIZE, UMA_FALLBACK_HF_MODELS

            inference = options.get('inference')
            effective_device = UMACalculator._normalize_device(self.device)
            options['inference'] = self._coerce_uma_inference_for_device(inference, effective_device)

            checkpoint_path = options.get('checkpoint_path') or options.get('model_path')
            effective_size = str(options.get('size')).lower() if options.get('size') else UMA_DEFAULT_SIZE
            if checkpoint_path is None and effective_size in UMA_FALLBACK_HF_MODELS:
                local_checkpoint = self._local_model_file(f'{effective_size}.pt')
                if local_checkpoint is not None:
                    checkpoint_path = str(local_checkpoint)
            options['checkpoint_path'] = checkpoint_path

        resolved_path_str = str(resolved_model_path) if resolved_model_path is not None else None
        kwargs = cls.build_kwargs_from_options(name, options, resolved_model_path=resolved_path_str)

        calculator = cls(
            device=self.device,
            model=name,
            implicit=self.implicit,
            solvent=self.solvent,
            **kwargs,
        )

        self._apply_hessian_mode(calculator)
        return calculator

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get('hessian')
        if mode is None:
            return

        mode = str(mode).lower()
        supported = type(calculator).SUPPORTED_HESSIAN_MODES
        if mode not in supported:
            supported_text = ', '.join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f'Supported modes: {supported_text}'
            )

        if not hasattr(calculator, 'hessian'):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        try:
            calculator = self._build_calculator()
            return calculator
        except Exception as exc:
            if not self._model_error_logged:
                self._log_model_error(
                    f"Failed to initialize model '{self.model}'.\n"
                    f'       MAPLE model directory: {self._model_dir_description()}\n'
                    f'       Error: {type(exc).__name__}: {exc}'
                )
            raise

    def log_error(self, error_message: str) -> None:
        with open(self.output, 'a') as handle:
            handle.write(f'ERROR: {error_message.rstrip()}\n')

    def log_info(self, info_message: list) -> None:
        with open(self.output, 'a') as handle:
            for line in info_message:
                handle.write(line)


# Historical misspelling kept as an alias for one release.
SetClaculator = SetCalculator
