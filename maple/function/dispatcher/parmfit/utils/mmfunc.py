from __future__ import annotations

import numpy as np


_TOL = 1.0e-12


def _positions_array(positions) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Positions must have shape (N, 3), got {array.shape}")
    return array


def _point(positions: np.ndarray, index: int) -> np.ndarray:
    if index < 1 or index > len(positions):
        raise IndexError(f"Atom index out of range for 1-based indexing: {index}")
    return positions[index - 1]


def distance_angstrom(positions, i: int, j: int) -> float:
    array = _positions_array(positions)
    rij = _point(array, j) - _point(array, i)
    return float(np.linalg.norm(rij))


def angle_radians(positions, i: int, j: int, k: int) -> float:
    array = _positions_array(positions)
    v1 = _point(array, i) - _point(array, j)
    v2 = _point(array, k) - _point(array, j)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= _TOL or n2 <= _TOL:
        raise ValueError(f"Cannot compute angle for degenerate geometry: {(i, j, k)}")
    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine))


def dihedral_radians(positions, i: int, j: int, k: int, l: int) -> float:
    array = _positions_array(positions)
    p0 = _point(array, i)
    p1 = _point(array, j)
    p2 = _point(array, k)
    p3 = _point(array, l)

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm <= _TOL:
        raise ValueError(f"Cannot compute dihedral for degenerate central bond: {(i, j, k, l)}")
    b1_hat = b1 / b1_norm

    v = b0 - np.dot(b0, b1_hat) * b1_hat
    w = b2 - np.dot(b2, b1_hat) * b1_hat

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm <= _TOL or w_norm <= _TOL:
        raise ValueError(f"Cannot compute dihedral for collinear geometry: {(i, j, k, l)}")

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_hat, v), w)
    angle = float(np.arctan2(y, x))
    if angle <= -np.pi:
        angle += 2.0 * np.pi
    return angle
