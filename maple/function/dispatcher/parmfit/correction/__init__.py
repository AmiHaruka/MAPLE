"""Usage: expose the correction dispatcher API."""

__all__ = ["Correction"]


def __getattr__(name: str):
    if name == "Correction":
        from .correction import Correction

        return Correction
    raise AttributeError(name)
