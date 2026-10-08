"""Usage: expose the refinement dispatcher API."""

__all__ = ["Refinement"]


def __getattr__(name: str):
    if name == "Refinement":
        from .job import Refinement

        return Refinement
    raise AttributeError(name)
