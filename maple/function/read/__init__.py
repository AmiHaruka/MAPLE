"""Public input helpers, imported lazily to keep optional ML stacks optional."""

from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = ("InputReader", "ExplicitSolv")


if TYPE_CHECKING:
    from .input_reader import InputReader
    from .post_process.explicit_solvent.solvate import ExplicitSolv


def __getattr__(name: str):
    if name == "InputReader":
        from .input_reader import InputReader

        value = InputReader
    elif name == "ExplicitSolv":
        from .post_process.explicit_solvent.solvate import ExplicitSolv

        value = ExplicitSolv
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
