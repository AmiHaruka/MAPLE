"""Usage: fit bond, angle, and improper parameters with the Seminario method."""

from itertools import product
from math import acos, pi

import numpy as np
from ase import Atoms

from .readparm import Angle, Bond, FourierTerm, Improper


HARTREE_TO_KCAL_MOL = 627.509474
LINEAR_TOL = 1.0e-8


def apply_seminario(
    atoms: Atoms,
    hessian_cart: np.ndarray,
    bonds: list[Bond],
    angles: list[Angle],
    vibrational_scaling: float = 1.0,
    impropers: list[Improper] | None = None,
) -> tuple[list[Bond], list[Angle], list[Improper]]:
    """
    Fill bond, angle, and improper instances using the Seminario method.
    """
    impropers = [] if impropers is None else impropers
    hessian_input = np.asarray(hessian_cart, dtype=float)
    expected_shape = (3 * len(atoms), 3 * len(atoms))
    if hessian_input.shape != expected_shape:
        raise ValueError(
            f"Hessian shape {hessian_input.shape} does not match expected {expected_shape} for {len(atoms)} atoms."
        )

    hessian = hessian_input * HARTREE_TO_KCAL_MOL
    positions = np.asarray(atoms.get_positions(), dtype=float)
    scaling_sq = float(vibrational_scaling) ** 2
    eig_cache = _build_block_eigen_cache(hessian, bonds, angles, impropers)

    for bond in bonds:
        i, j = bond.atoms
        bond.rEq = _bond_length(positions, i, j)
        k_ij = _bond_force_constant(i, j, positions, eig_cache)
        k_ji = _bond_force_constant(j, i, positions, eig_cache)
        bond.kBond = max(float(np.real((k_ij + k_ji) * 0.25) * scaling_sq), 0.0)

    for angle in angles:
        i, j, k = angle.atoms
        angle.thetaEq = _angle_value(positions, i, j, k)
        k_theta = _angle_force_constant(i, j, k, positions, eig_cache)
        angle.kTheta = max(float(np.real(k_theta * 0.5) * scaling_sq), 0.0)

    for improper in impropers:
        i, j, k, l = improper.atoms
        k_phi = _improper_force_constant(i, j, k, l, positions, eig_cache)
        improper.terms = [
            FourierTerm(
                kPhi=max(k_phi * scaling_sq, 0.0),
                period=2.0,
                phase=pi,
            )
        ]

    return bonds, angles, impropers


def _build_block_eigen_cache(
    hessian: np.ndarray,
    bonds: list[Bond],
    angles: list[Angle],
    impropers: list[Improper],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    pairs: set[tuple[int, int]] = set()
    for bond in bonds:
        i, j = bond.atoms
        pairs.add((i, j))
        pairs.add((j, i))
    for angle in angles:
        i, j, k = angle.atoms
        pairs.add((i, j))
        pairs.add((k, j))
    for improper in impropers:
        i, j, center, l = improper.atoms
        pairs.add((center, i))
        pairs.add((center, j))
        pairs.add((center, l))

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i, j in pairs:
        block = -_hessian_block(hessian, i, j)
        cache[(i, j)] = np.linalg.eig(block)
    return cache


def _hessian_block(hessian: np.ndarray, i: int, j: int) -> np.ndarray:
    i0 = 3 * (i - 1)
    j0 = 3 * (j - 1)
    return hessian[i0:i0 + 3, j0:j0 + 3]


def _bond_length(positions: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions[j - 1] - positions[i - 1]))


def _angle_value(positions: np.ndarray, i: int, j: int, k: int) -> float:
    u_ji = _unit_vector(positions[i - 1] - positions[j - 1])
    u_jk = _unit_vector(positions[k - 1] - positions[j - 1])
    cosine = float(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0))
    return float(acos(cosine))


def _improper_value(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    b0 = positions[j - 1] - positions[i - 1]
    b1 = positions[k - 1] - positions[j - 1]
    b2 = positions[l - 1] - positions[k - 1]
    u_jk = _unit_vector(b1)
    u_pa = _unit_vector(b0 - np.dot(b0, u_jk) * u_jk)
    u_pc = _unit_vector(b2 - np.dot(b2, u_jk) * u_jk)
    angle = float(np.arctan2(np.dot(np.cross(u_jk, u_pa), u_pc), np.dot(u_pa, u_pc)))
    return ((angle + 2.0 * pi) % (2.0 * pi)) - pi


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-16:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _bond_force_constant(
    atom_a: int,
    atom_b: int,
    positions: np.ndarray,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> complex:
    eigenvalues, eigenvectors = eig_cache[(atom_a, atom_b)]
    u_ab = _unit_vector(positions[atom_a - 1] - positions[atom_b - 1])
    return _seminario_sum(u_ab, eigenvalues, eigenvectors)


def _angle_force_constant(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    positions: np.ndarray,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> complex:
    bond_length_ab = _bond_length(positions, atom_a, atom_b)
    bond_length_cb = _bond_length(positions, atom_c, atom_b)
    u_ab = _unit_vector(positions[atom_a - 1] - positions[atom_b - 1])
    u_cb = _unit_vector(positions[atom_c - 1] - positions[atom_b - 1])

    if abs(float(np.linalg.norm(u_cb - u_ab))) < 0.01 or (1.99 < abs(float(np.linalg.norm(u_cb - u_ab))) < 2.01):
        return _angle_force_constant_linear(atom_a, atom_b, atom_c, positions, bond_length_ab, bond_length_cb, eig_cache)

    try:
        u_n = _unit_normal(u_cb, u_ab)
        u_pa = _unit_vector(np.cross(u_n, u_ab))
        u_pc = _unit_vector(np.cross(u_cb, u_n))
    except ValueError:
        return _angle_force_constant_linear(atom_a, atom_b, atom_c, positions, bond_length_ab, bond_length_cb, eig_cache)

    return _angle_force_constant_from_normals(
        atom_a,
        atom_b,
        atom_c,
        bond_length_ab,
        bond_length_cb,
        eig_cache,
        u_pa,
        u_pc,
    )


def _unit_normal(u_cb: np.ndarray, u_ab: np.ndarray) -> np.ndarray:
    cross = np.cross(u_cb, u_ab)
    norm = np.linalg.norm(cross)
    if norm < LINEAR_TOL:
        raise ValueError("Angle vectors are linearly dependent.")
    return cross / norm


def _angle_force_constant_from_normals(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    bond_length_ab: float,
    bond_length_cb: float,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    u_pa: np.ndarray,
    u_pc: np.ndarray,
) -> complex:
    eigenvalues_ab, eigenvectors_ab = eig_cache[(atom_a, atom_b)]
    eigenvalues_cb, eigenvectors_cb = eig_cache[(atom_c, atom_b)]
    sum_first = _seminario_sum(u_pa, eigenvalues_ab, eigenvectors_ab)
    sum_second = _seminario_sum(u_pc, eigenvalues_cb, eigenvectors_cb)
    springs = (1.0 / (bond_length_ab**2 * sum_first)) + (1.0 / (bond_length_cb**2 * sum_second))
    return 1.0 / springs


def _angle_force_constant_linear(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    positions: np.ndarray,
    bond_length_ab: float,
    bond_length_cb: float,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    n_theta: int = 18,
    n_phi: int = 36,
) -> complex:
    u_ab = _unit_vector(positions[atom_a - 1] - positions[atom_b - 1])
    u_cb = _unit_vector(positions[atom_c - 1] - positions[atom_b - 1])
    k_values: list[complex] = []
    for theta_idx, phi_idx in product(range(n_theta), range(n_phi)):
        theta = np.pi * (theta_idx + 0.5) / n_theta
        phi = 2.0 * np.pi * phi_idx / n_phi
        u_n = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=float,
        )
        try:
            u_pa = _unit_vector(np.cross(u_n, u_ab))
            u_pc = _unit_vector(np.cross(u_cb, u_n))
        except ValueError:
            continue
        k_values.append(
            _angle_force_constant_from_normals(
                atom_a,
                atom_b,
                atom_c,
                bond_length_ab,
                bond_length_cb,
                eig_cache,
                u_pa,
                u_pc,
            )
        )
    if not k_values:
        raise ValueError("Failed to construct a valid normal for a linear angle.")
    return complex(np.mean(k_values))


def _improper_force_constant(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    atom_d: int,
    positions: np.ndarray,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> float:
    v_ab = positions[atom_b - 1] - positions[atom_a - 1]
    v_ad = positions[atom_d - 1] - positions[atom_a - 1]
    u_n = _unit_vector(np.cross(v_ab, v_ad))

    total = 0.0 + 0.0j
    for outer in (atom_a, atom_b, atom_d):
        eigenvalues, eigenvectors = eig_cache[(atom_c, outer)]
        total += _seminario_sum(u_n, eigenvalues, eigenvectors)
    k_h = float(np.real(total * 0.5))

    eps = 1.0e-4
    displaced = positions.copy()
    displaced[atom_c - 1] = positions[atom_c - 1] + eps * u_n
    phi_plus = _improper_value(displaced, atom_a, atom_b, atom_c, atom_d)
    displaced[atom_c - 1] = positions[atom_c - 1] - eps * u_n
    phi_minus = _improper_value(displaced, atom_a, atom_b, atom_c, atom_d)
    # a planar reference can sit on the +/-pi branch cut, so re-wrap the difference
    slope = (((phi_plus - phi_minus + pi) % (2.0 * pi)) - pi) / (2.0 * eps)
    # K(1 - cos 2phi) ~ 2 K phi^2 must reproduce k_h h^2 at the reference
    return k_h / (2.0 * slope * slope)


def _seminario_sum(vector: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> complex:
    value = 0.0 + 0.0j
    for idx in range(3):
        value += eigenvalues[idx] * abs(np.dot(eigenvectors[:, idx], vector))
    return value
