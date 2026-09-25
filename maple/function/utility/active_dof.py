"""Active Cartesian degrees of freedom for supported ASE constraints."""

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def active_atom_mask(atoms: Atoms) -> np.ndarray:
    """Return active atoms, supporting only whole-atom ``FixAtoms`` constraints."""
    active = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NotImplementedError(
                "Only ASE FixAtoms constraints are supported; "
                f"got {type(constraint).__name__}"
            )
        active[np.asarray(constraint.get_indices(), dtype=int)] = False
    return active


def active_dof_mask(atoms: Atoms) -> np.ndarray:
    """Return a flattened boolean mask for the active Cartesian coordinates."""
    return np.repeat(active_atom_mask(atoms), 3)
