"""Shared helpers for calculator metadata validation."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any


def coerce_int_metadata(
    value: Any,
    name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Return ``value`` as an int, rejecting lossy scientific metadata casts."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got boolean {value!r}.")

    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real):
        as_float = float(value)
        if not as_float.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        parsed = int(as_float)
    elif isinstance(value, str):
        stripped = value.strip()
        digits = stripped[1:] if stripped.startswith(("+", "-")) else stripped
        if not digits or not digits.isdigit():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        parsed = int(stripped)
    else:
        raise ValueError(f"{name} must be an integer, got {value!r}.")

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {parsed}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {parsed}.")
    return parsed


def integer_info(
    atoms,
    key: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Read ``atoms.info[key]`` as validated integer metadata."""
    if key not in atoms.info:
        return int(default)
    return coerce_int_metadata(
        atoms.info[key],
        f"atoms.info[{key!r}]",
        min_value=min_value,
        max_value=max_value,
    )
