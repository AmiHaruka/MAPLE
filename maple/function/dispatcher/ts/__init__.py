"""Transition-state dispatcher exports with optional dependencies kept lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = ("TransitionState",)


if TYPE_CHECKING:
    from .ts import TransitionState


def __getattr__(name: str):
    if name != "TransitionState":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .ts import TransitionState

    globals()[name] = TransitionState
    return TransitionState


def __dir__():
    return sorted(set(globals()) | set(__all__))
