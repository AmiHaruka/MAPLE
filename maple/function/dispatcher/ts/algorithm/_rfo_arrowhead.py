"""Extremal eigenvalues of a real symmetric arrowhead, without dense matrices."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class ArrowheadSolverError(RuntimeError):
    """An extremal root could not be bracketed or resolved in float64."""


def prepare_extremal_root(diagonal, coupling, *, maximum=False):
    """Prepare the fixed spectrum for repeated coupling-scale root queries.

    Zero couplings are deflated exactly. The remaining lowest root lies below
    every active pole and is the unique zero of -x - sum(b²/(d-x)). Scaling
    keeps the bracket and secular equation in a representable numeric range.
    Repeated poles need no special matrix decomposition.
    """
    d = np.asarray(diagonal, dtype=np.float64)
    b = np.asarray(coupling, dtype=np.float64)
    if d.ndim != 1 or b.shape != d.shape:
        raise ValueError('Arrowhead diagonal and coupling must be matching vectors')
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(b)):
        raise ArrowheadSolverError('Non-finite arrowhead input')
    if maximum:
        minimum = prepare_extremal_root(-d, b)
        return lambda coupling_scale=1.0: -minimum(coupling_scale)
    # Deflation precedes scaling: an uncoupled enormous pole must not erase
    # the active block or change which couplings are considered nonzero.
    active = b != 0.0
    decoupled = min(0.0, float(np.min(d[~active]))) if np.any(~active) else 0.0
    d, original_b = d[active], b[active]
    max_d = float(np.max(np.abs(d))) if d.size else 0.0
    max_b = float(np.max(np.abs(original_b))) if d.size else 0.0
    min_b = float(np.min(np.abs(original_b))) if d.size else 0.0
    min_d = float(np.min(d)) if d.size else 0.0

    def solve(coupling_scale=1.0):
        factor = float(coupling_scale)
        if not np.isfinite(factor) or factor < 1.0:
            raise ArrowheadSolverError('Invalid augmented coupling scale')
        if not d.size:
            return decoupled
        scale = max(max_d, max_b * factor)
        if not np.isfinite(scale) or min_b * factor / scale == 0.0:
            raise ArrowheadSolverError('Unrepresentable active coupling scale')
        b_original = original_b * factor
        ds, b = d / scale, b_original / scale

        def finish(root):
            if not np.isfinite(root):
                raise ArrowheadSolverError('Extremal eigenvalue exceeds float64 range')
            if root == 0.0:
                raise ArrowheadSolverError('Active secular root lost to scaling underflow')
            return min(decoupled, float(root))

        if d.size == 1:
            # Multiply one factor in original units, avoiding b_scaled**2
            # underflow when the physical root is still representable.
            length = float(np.hypot(ds[0], 2.0 * b[0]))
            root = (0.5 * (ds[0] - length) * scale if ds[0] <= 0.0
                    else -b_original[0] * ((2.0 * b[0]) / (length + ds[0])))
            return finish(root)

        pole = min(0.0, min_d / scale)
        lower = float(np.nextafter(pole - np.hypot.reduce(b), -np.inf))
        upper = float(np.nextafter(pole, -np.inf))

        def secular(x):
            return -x - float(np.dot(b, b / (ds - x)))

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            f_lower, f_upper = secular(lower), secular(upper)
            if np.isnan(f_lower) or np.isnan(f_upper) or f_lower < 0.0:
                raise ArrowheadSolverError('Invalid extremal secular bracket')
            if f_upper > 0.0:
                # The active root and pole round to the same float. Do not wander to
                # another secular interval; the pole is the representable extremum.
                root = pole
            else:
                try:
                    root = brentq(secular, lower, upper,
                                  xtol=np.nextafter(0.0, 1.0),
                                  rtol=4.0 * np.finfo(float).eps, maxiter=256)
                except (ValueError, RuntimeError) as exc:
                    raise ArrowheadSolverError('Extremal secular root did not converge') from exc
        if not np.isfinite(root) or not lower <= root <= pole:
            raise ArrowheadSolverError('Extremal root left its pole interval')
        return finish(float(root) * scale)

    return solve


def extremal_root(diagonal, coupling, *, maximum=False) -> float:
    """Evaluate one arrowhead root; iterative callers can reuse preparation."""
    return prepare_extremal_root(diagonal, coupling, maximum=maximum)()
