"""Mass-weighted rigid-body geometry classification."""

import numpy as np
from ase import Atoms


def mass_weighted_rigid_basis(atoms: Atoms) -> np.ndarray:
    """Return an orthonormal rigid-motion basis for the system geometry."""
    masses = np.asarray(atoms.get_masses(), dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    n_atoms = len(atoms)
    include_rotations = not np.any(atoms.get_pbc())
    candidates = np.zeros((3 * n_atoms, 6 if include_rotations else 3))

    sqrt_masses = np.sqrt(masses)
    for atom_index, sqrt_mass in enumerate(sqrt_masses):
        block = slice(3 * atom_index, 3 * atom_index + 3)
        candidates[block, :3] = sqrt_mass * np.eye(3)

    if include_rotations and n_atoms:
        center = np.average(positions, axis=0, weights=masses)
        axes = np.eye(3)
        for atom_index, (position, sqrt_mass) in enumerate(
            zip(positions - center, sqrt_masses)
        ):
            block = slice(3 * atom_index, 3 * atom_index + 3)
            for axis_index, axis in enumerate(axes):
                candidates[block, 3 + axis_index] = (
                    sqrt_mass * np.cross(axis, position)
                )

    if not candidates.size:
        return np.empty((3 * n_atoms, 0))

    # Translation and rotation are orthogonal when rotations are constructed
    # about the centre of mass.  Factor them separately so the returned basis
    # retains that physical partition; callers can then remove rotation while
    # preserving an existing centre-of-mass velocity without forming the
    # ill-conditioned inertia-tensor Gram matrix.
    translation_scale = np.sqrt(np.sum(masses))
    translation_vectors = candidates[:, :3] / translation_scale
    if not include_rotations:
        return translation_vectors

    rotation_candidates = candidates[:, 3:]
    rotation_candidates -= translation_vectors @ (
        translation_vectors.T @ rotation_candidates
    )
    rotation_vectors, singular_values, _ = np.linalg.svd(
        rotation_candidates, full_matrices=False
    )
    if not singular_values.size:
        return translation_vectors
    largest_scale = max(translation_scale, singular_values[0])
    tolerance = (
        max(candidates.shape)
        * np.finfo(candidates.dtype).eps
        * largest_scale
    )
    rotations = rotation_vectors[:, singular_values > tolerance]
    combined = np.column_stack((translation_vectors, rotations))
    # SVD has selected the rank; QR jointly orthogonalizes only retained directions.
    return np.linalg.qr(combined, mode="reduced")[0]


def project_rigid_body_velocities(
    atoms: Atoms,
    velocities: np.ndarray,
    *,
    remove_translation: bool,
    remove_rotation: bool,
) -> np.ndarray:
    """Project velocities in the mass metric onto the requested internal space."""
    values = np.asarray(velocities, dtype=float)
    if values.shape != (len(atoms), 3):
        raise ValueError(
            f"Expected velocities with shape {(len(atoms), 3)}, got {values.shape}"
        )
    if not remove_translation and not remove_rotation:
        return values.copy()

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if np.any(masses <= 0.0):
        raise ValueError("Rigid-body velocity projection requires positive masses")

    basis = mass_weighted_rigid_basis(atoms)
    n_translation = min(3, basis.shape[1])
    selected: list[np.ndarray] = []
    if remove_translation:
        selected.append(basis[:, :n_translation])
    if remove_rotation and not np.any(atoms.get_pbc()):
        selected.append(basis[:, n_translation:])
    if not selected:
        return values.copy()

    projector_basis = np.column_stack(selected)
    weighted = (np.sqrt(masses)[:, np.newaxis] * values).reshape(-1)
    weighted -= projector_basis @ (projector_basis.T @ weighted)
    return weighted.reshape((-1, 3)) / np.sqrt(masses)[:, np.newaxis]


def rotational_dof(atoms: Atoms) -> int:
    """Return the rigid rotational dimension: 0, 2, or 3."""
    if np.any(atoms.get_pbc()):
        return 0
    return max(mass_weighted_rigid_basis(atoms).shape[1] - 3, 0)


def is_linear_geometry(atoms: Atoms) -> bool:
    """Return whether the geometry has exactly two rigid rotations."""
    return rotational_dof(atoms) == 2
