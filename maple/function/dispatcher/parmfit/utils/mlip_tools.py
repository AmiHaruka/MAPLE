"""Utilities for evaluating optional MAPLE model properties inside workflows."""

from __future__ import annotations

import gc

import numpy as np

from maple.function.calculator.set_calculator import SetCalculator


def resolve_charge_model_class(
    model_name: str,
    *,
    device=None,
    output: str,
) -> tuple[str, type]:
    """Resolve a charge model through ParmFit's existing SetCalculator integration."""
    resolver = SetCalculator(
        device=device,
        model=model_name,
        output=output,
    )
    calculator_class = resolver._discover_calculator_class(resolver.model)
    return resolver.model, calculator_class


def resolve_charge_calculator(
    model_name: str,
    *,
    atoms,
    output: str,
):
    """Return a calculator for ``model_name``, reusing the active model when possible."""
    active_calculator = getattr(atoms, "calc", None)
    device = (
        getattr(active_calculator, "device", None)
        if active_calculator is not None
        else None
    )
    canonical, _ = resolve_charge_model_class(
        model_name,
        device=device,
        output=output,
    )
    main_name = (
        getattr(active_calculator, "model_name", None)
        if active_calculator is not None
        else None
    )
    if (
        active_calculator is not None
        and main_name is not None
        and str(main_name).strip().lower() == canonical
    ):
        return active_calculator, canonical

    if device is None:
        raise ValueError(
            f"Cannot initialize charge model {canonical!r}: "
            "the active calculator does not provide a device."
        )

    calculator = SetCalculator(
        device=device,
        model=canonical,
        output=output,
        atoms=atoms,
    ).set_calculator()
    return calculator, canonical


def calculator_atomic_charges(calculator, atoms, *, model_name: str) -> np.ndarray:
    """Evaluate the standard ASE ``charges`` property with minimal shape checks."""
    properties = tuple(getattr(calculator, "implemented_properties", ()))
    if "charges" not in properties:
        raise ValueError(
            f"MAPLE model {model_name!r} does not provide the ASE 'charges' property."
        )
    charges = np.asarray(calculator.get_property("charges", atoms), dtype=float).reshape(-1)
    if charges.size != len(atoms):
        raise ValueError(
            f"MAPLE model {model_name!r} returned {charges.size} charges for {len(atoms)} atoms."
        )
    if not np.all(np.isfinite(charges)):
        raise ValueError(f"MAPLE model {model_name!r} returned non-finite charges.")
    return charges.copy()


def release_auxiliary_calculator_memory(device) -> None:
    """Release unreachable model objects and return unused CUDA blocks."""
    gc.collect()
    if device is None or not str(device).lower().startswith("cuda"):
        return

    import torch

    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def release_charge_calculator_cache(
    calculators: dict,
    *,
    active_calculators,
) -> None:
    """Drop auxiliary charge models while preserving calculators owned by Atoms."""
    active_ids = {
        id(calculator)
        for calculator in active_calculators
        if calculator is not None
    }
    auxiliary_devices = {
        str(getattr(calculator, "device", None)): getattr(calculator, "device", None)
        for calculator in calculators.values()
        if id(calculator) not in active_ids
    }
    calculators.clear()
    for device in auxiliary_devices.values():
        release_auxiliary_calculator_memory(device)
