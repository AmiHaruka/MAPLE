
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import fields
from difflib import get_close_matches
from typing import Tuple


class JobABC(ABC):

    # Command-level keys which may accompany a flat method parameter mapping.
    # They are consumed by the dispatcher/calculator setup, not method
    # dataclasses.  Nested method mappings do not need these exceptions.
    TASK_ROUTING_PARAM_KEYS = (
        "method", "model", "model_options", "device", "gpuid", "d4",
        "pbc", "solv", "level",
    )

    def __init__(self, output: str):
        self.output = output

    @abstractmethod
    def run(self):
        pass

    # ==============================================
    # Parameter handling utilities
    # ==============================================

    @staticmethod
    def _lower_keys(d: dict) -> dict:
        """Return a copy of dict with all string keys lowercased."""
        if not isinstance(d, dict):
            return {}
        return {(k.lower() if isinstance(k, str) else k): v for k, v in d.items()}

    @staticmethod
    def _select_subdict(paras: dict, name_aliases: Tuple[str, ...]) -> dict:
        """Extract a sub-dict using aliases (e.g., 'neb', 'NEB')."""
        if not isinstance(paras, dict):
            return {}
        low = JobABC._lower_keys(paras)
        for alias in name_aliases:
            key = alias.lower()
            if key in low and isinstance(low[key], dict):
                return low[key]
        return low

    @staticmethod
    def _effective_param_dict(
        paras: dict,
        aliases: Tuple[str, ...],
    ) -> dict:
        """Return the flat-first, selected-nested-wins parameter mapping."""
        if not isinstance(paras, dict):
            return {}
        low = JobABC._lower_keys(paras)
        nested_key = next(
            (
                alias.lower()
                for alias in aliases
                if isinstance(low.get(alias.lower()), dict)
            ),
            None,
        )
        if nested_key is None:
            if all(isinstance(key, str) and key == key.lower() for key in paras):
                return paras
            return low
        effective = {
            key: value for key, value in low.items() if key != nested_key
        }
        effective.update(JobABC._lower_keys(low[nested_key]))
        return effective

    @staticmethod
    def _select_refinement_params(
        paras: dict,
        method: str,
        *,
        forwarded_keys: tuple[str, ...] | None = None,
    ) -> dict:
        """Isolate explicit refinement options from the parent method budget.

        A nested ``#prfo(...)``/``#dimer(...)`` block is authoritative.  Only
        named historical forwarding keys are copied from a flat parent mapping;
        parent fields such as ``max_iter`` must not silently redefine the
        refinement optimizer's independent budget.
        """
        if not isinstance(paras, dict):
            return {}
        if forwarded_keys is None:
            forwarded_keys = (
                ("rigid_symmetry",) if method.lower() == "prfo" else ()
            )
        low = JobABC._lower_keys(paras)
        selected = low.get(method.lower())
        if (
            isinstance(selected, dict)
            and set(low) == {method.lower()}
        ):
            # Already an isolated explicit refinement mapping; preserve object
            # identity for callers/tests that treat it as immutable config.
            return paras
        options = JobABC._lower_keys(selected) if isinstance(selected, dict) else {}
        for key in forwarded_keys:
            normalized = key.lower()
            if normalized in low and normalized not in options:
                options[normalized] = low[normalized]
        return {method.lower(): options} if options else {}

    @staticmethod
    def _update_dataclass_from_dict(
        dc_obj,
        d: dict,
        *,
        strict: bool = False,
        context: str | None = None,
        allowed_keys: tuple[str, ...] = (),
    ):
        """Update a dataclass instance from a dict (case-insensitive keys).

        ``strict`` is opt-in so legacy jobs retain their existing parameter
        handling.  Strict callers may name routing or downstream-refinement
        keys in ``allowed_keys``; those keys are accepted but are not assigned
        to the dataclass.
        """
        if not isinstance(d, dict):
            return dc_obj
        low = JobABC._lower_keys(d)
        fld_names = {f.name.lower(): f.name for f in fields(dc_obj)}
        allowed = {key.lower() for key in allowed_keys}
        if strict:
            valid = set(fld_names) | allowed
            for key in low:
                if key in valid:
                    continue
                label = context or type(dc_obj).__name__
                message = f"Unknown {label} parameter: '{key}'."
                match = get_close_matches(key, sorted(valid), n=1, cutoff=0.72)
                if match:
                    message += f" Did you mean '{match[0]}'?"
                raise ValueError(message)
        for k_low, v in low.items():
            if k_low in fld_names:
                setattr(dc_obj, fld_names[k_low], v)
        return dc_obj

    @classmethod
    def _validate_nested_dataclass_params(
        cls,
        paras: dict,
        key: str,
        params_class,
        *,
        context: str,
    ) -> None:
        """Validate one explicitly selected downstream method block."""
        if not isinstance(paras, dict):
            return
        low = cls._lower_keys(paras)
        if key not in low:
            return
        nested = low[key]
        if not isinstance(nested, dict):
            raise ValueError(
                f"{context} options must use a nested parameter mapping."
            )
        cls._update_dataclass_from_dict(
            params_class(),
            nested,
            strict=True,
            context=context,
        )

    def _init_params(
        self,
        params_class,
        paras: dict,
        aliases: Tuple[str, ...],
        *,
        strict: bool = False,
        context: str | None = None,
        allowed_keys: tuple[str, ...] = (),
    ):
        """Initialize params dataclass from external dict."""
        params = params_class()
        if isinstance(paras, dict):
            low = self._lower_keys(paras)
            nested_key = next(
                (
                    alias.lower()
                    for alias in aliases
                    if isinstance(low.get(alias.lower()), dict)
                ),
                None,
            )
            if nested_key is not None:
                # Validate the command-level mapping too: otherwise adding a
                # method-specific block could hide a typo beside it. Apply
                # flat values first and let the explicit nested block win.
                top_level = {
                    key: value for key, value in low.items() if key != nested_key
                }
                self._update_dataclass_from_dict(
                    params,
                    top_level,
                    strict=strict,
                    context=context,
                    allowed_keys=allowed_keys,
                )
                sub_dict = low[nested_key]
                nested_allowed = (
                    allowed_keys if nested_key == "ts" or not strict else ()
                )
            else:
                sub_dict = low
                nested_allowed = allowed_keys
            self._update_dataclass_from_dict(
                params,
                sub_dict,
                strict=strict,
                context=context,
                allowed_keys=nested_allowed,
            )
        return params

    # ==============================================
    # Logging utilities
    # ==============================================

    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: Iterable[str]) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: Text fragments to write in order.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")
