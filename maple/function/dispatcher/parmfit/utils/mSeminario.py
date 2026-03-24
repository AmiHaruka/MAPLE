from itertools import product
from math import acos

import numpy as np
from ase import Atoms

from .parm import Angle, Bond


HARTREE_TO_KCAL_MOL = 627.509474
LINEAR_TOL = 1.0e-8


def apply_mseminario(
    atoms: Atoms,
    hessian_cart: np.ndarray,
    bonds: list[Bond],
    angles: list[Angle],
    vibrational_scaling: float = 1.0,
) -> tuple[list[Bond], list[Angle]]:
    """
    Fill bond and angle instances using the Modified Seminario method.
    """
    hessian = hessian_cart * HARTREE_TO_KCAL_MOL
    positions = np.asarray(atoms.get_positions(), dtype=float)
    scaling_sq = float(vibrational_scaling) ** 2
    eig_cache = _build_block_eigen_cache(hessian, bonds, angles)
    angle_scalings = _angle_scaling_factors(positions, angles)

    for bond in bonds:
        i, j = bond.atoms
        bond.rEq = _bond_length(positions, i, j)
        k_ij = _bond_force_constant(i, j, positions, eig_cache)
        k_ji = _bond_force_constant(j, i, positions, eig_cache)
        bond.kBond = max(float(np.real((k_ij + k_ji) * 0.5) * scaling_sq), 0.0)

    for angle_index, angle in enumerate(angles):
        i, j, k = angle.atoms
        scale_ij, scale_kj = angle_scalings[angle_index]
        angle.thetaEq = _angle_value(positions, i, j, k)
        theta_ijk = _angle_force_constant(i, j, k, positions, eig_cache, scale_ij, scale_kj)
        theta_kji = _angle_force_constant(k, j, i, positions, eig_cache, scale_kj, scale_ij)
        angle.kTheta = max(float(np.real((theta_ijk + theta_kji) * 0.5) * scaling_sq), 0.0)

    return bonds, angles


def _build_block_eigen_cache(
    hessian: np.ndarray,
    bonds: list[Bond],
    angles: list[Angle],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    pairs: set[tuple[int, int]] = set()
    for bond in bonds:
        i, j = bond.atoms
        pairs.add((i, j))
        pairs.add((j, i))
    for angle in angles:
        i, j, k = angle.atoms
        pairs.add((i, j))
        pairs.add((j, i))
        pairs.add((k, j))
        pairs.add((j, k))

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i, j in pairs:
        block = _hessian_block(hessian, i, j)
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
    u_ab = _unit_vector(positions[atom_b - 1] - positions[atom_a - 1])
    value = 0.0 + 0.0j
    for idx in range(3):
        projection = abs(np.dot(u_ab, eigenvectors[:, idx]))
        value += eigenvalues[idx] * projection
    return -0.5 * value


def _u_pa(atom_a: int, atom_b: int, atom_c: int, positions: np.ndarray) -> np.ndarray:
    u_ab = _unit_vector(positions[atom_b - 1] - positions[atom_a - 1])
    u_cb = _unit_vector(positions[atom_b - 1] - positions[atom_c - 1])
    u_n = _unit_normal(u_cb, u_ab)
    return _unit_vector(np.cross(u_n, u_ab))


def _unit_normal(u_cb: np.ndarray, u_ab: np.ndarray) -> np.ndarray:
    cross = np.cross(u_cb, u_ab)
    norm = np.linalg.norm(cross)
    if norm < LINEAR_TOL:
        raise ValueError("Angle vectors are linearly dependent.")
    return cross / norm


def _angle_force_constant(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    positions: np.ndarray,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    scaling_1: float,
    scaling_2: float,
) -> float:
    bond_length_ab = _bond_length(positions, atom_a, atom_b)
    bond_length_bc = _bond_length(positions, atom_b, atom_c)

    u_ab = _unit_vector(positions[atom_b - 1] - positions[atom_a - 1])
    u_cb = _unit_vector(positions[atom_b - 1] - positions[atom_c - 1])

    if abs(float(np.linalg.norm(u_cb - u_ab))) < 0.01 or (1.99 < abs(float(np.linalg.norm(u_cb - u_ab))) < 2.01):
        return _angle_force_constant_linear(
            atom_a,
            atom_b,
            atom_c,
            positions,
            bond_length_ab,
            bond_length_bc,
            eig_cache,
        )

    try:
        u_n = _unit_normal(u_cb, u_ab)
        u_pa = _unit_vector(np.cross(u_n, u_ab))
        u_pc = _unit_vector(np.cross(u_cb, u_n))
        return _angle_force_constant_from_normals(
            atom_a,
            atom_b,
            atom_c,
            bond_length_ab,
            bond_length_bc,
            eig_cache,
            u_pa,
            u_pc,
            scaling_1,
            scaling_2,
        )
    except ValueError:
        return _angle_force_constant_linear(
            atom_a,
            atom_b,
            atom_c,
            positions,
            bond_length_ab,
            bond_length_bc,
            eig_cache,
        )


def _angle_force_constant_from_normals(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    bond_length_ab: float,
    bond_length_bc: float,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    u_pa: np.ndarray,
    u_pc: np.ndarray,
    scaling_1: float,
    scaling_2: float,
) -> float:
    eigenvalues_ab, eigenvectors_ab = eig_cache[(atom_a, atom_b)]
    eigenvalues_cb, eigenvectors_cb = eig_cache[(atom_c, atom_b)]

    sum_first = 0.0 + 0.0j
    sum_second = 0.0 + 0.0j
    for idx in range(3):
        eig_ab = eigenvectors_ab[:, idx]
        eig_cb = eigenvectors_cb[:, idx]
        sum_first += eigenvalues_ab[idx] * abs(_complex_dot(u_pa, eig_ab))
        sum_second += eigenvalues_cb[idx] * abs(_complex_dot(u_pc, eig_cb))

    sum_first /= float(scaling_1)
    sum_second /= float(scaling_2)

    springs = (1.0 / (bond_length_ab**2 * sum_first)) + (1.0 / (bond_length_bc**2 * sum_second))
    k_theta = 1.0 / springs
    return float(abs(-0.5 * k_theta))


def _angle_force_constant_linear(
    atom_a: int,
    atom_b: int,
    atom_c: int,
    positions: np.ndarray,
    bond_length_ab: float,
    bond_length_bc: float,
    eig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    n_theta: int = 18,
    n_phi: int = 36,
) -> float:
    u_ab = _unit_vector(positions[atom_b - 1] - positions[atom_a - 1])
    u_cb = _unit_vector(positions[atom_b - 1] - positions[atom_c - 1])
    k_values: list[float] = []
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
                bond_length_bc,
                eig_cache,
                u_pa,
                u_pc,
                1.0,
                1.0,
            )
        )
    if not k_values:
        raise ValueError("Failed to construct a valid normal for a linear angle.")
    return float(np.mean(k_values))


def _complex_dot(vector: np.ndarray, eigenvector: np.ndarray) -> complex:
    value = 0.0 + 0.0j
    for idx in range(3):
        value += vector[idx] * np.conjugate(eigenvector[idx])
    return value


def _angle_scaling_factors(positions: np.ndarray, angles: list[Angle]) -> list[tuple[float, float]]:
    central_map: dict[int, list[tuple[int, int, int]]] = {}
    for index, angle in enumerate(angles):
        left, center, right = angle.atoms
        central_map.setdefault(center, []).append((left, right, index))
        central_map.setdefault(center, []).append((right, left, index))

    ordered_scalings: list[list[float]] = [[] for _ in angles]

    for center, entries in central_map.items():
        entries = sorted(entries, key=lambda item: item[0])
        u_pa_vectors: list[np.ndarray | None] = []
        for left, right, _ in entries:
            try:
                u_pa_vectors.append(_u_pa(left, center, right, positions))
            except ValueError:
                u_pa_vectors.append(None)
        scaling_by_entry: list[tuple[float, int]] = []

        for idx, entry in enumerate(entries):
            left_atom = entry[0]
            if u_pa_vectors[idx] is None:
                scaling_by_entry.append((1.0, entry[2]))
                continue
            contributions = []

            offset = idx + 1
            while offset < len(entries) and entries[offset][0] == left_atom:
                if u_pa_vectors[offset] is not None:
                    contributions.append(abs(np.dot(u_pa_vectors[idx], u_pa_vectors[offset])) ** 2)
                offset += 1

            offset = idx - 1
            while offset >= 0 and entries[offset][0] == left_atom:
                if u_pa_vectors[offset] is not None:
                    contributions.append(abs(np.dot(u_pa_vectors[idx], u_pa_vectors[offset])) ** 2)
                offset -= 1

            if contributions:
                scaling = 1.0 + float(np.mean(contributions))
            else:
                scaling = 1.0
            scaling_by_entry.append((scaling, entry[2]))

        for scaling, angle_index in scaling_by_entry:
            ordered_scalings[angle_index].append(scaling)

    result: list[tuple[float, float]] = []
    for scaling_pair in ordered_scalings:
        if len(scaling_pair) != 2:
            raise ValueError("Each angle must receive exactly two Modified Seminario scaling factors.")
        result.append((float(scaling_pair[0]), float(scaling_pair[1])))
    return result
