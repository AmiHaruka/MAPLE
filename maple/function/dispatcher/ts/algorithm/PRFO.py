# -*- coding: utf-8 -*-
"""
PRFO (Partitioned Rational Function Optimization) implementation for TS search.
Features:
- Augmented-Hessian RS-P-RFO with a shared trust restriction
- Mass-weighted coordinates for step computation
- Symmetric TS model-quality trust adaptation
- Mode-following for transition state optimization
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Optional

import numpy as np
from ase import Atoms

from .logger import log_info
from ._pdb_compat import require_shared_pdb_writer
from ...jobABC import JobABC
from maple.function.read.filereader.pdb_reader import write_pdb, write_pdb_model
from maple.function.utility.rigid_body import mass_weighted_rigid_basis


def _finite_positive_real(value) -> float:
    """Accept only finite positive real scalars, never booleans or strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError("expected a real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("expected a finite positive real scalar")
    return result

# =============================================================================
# ------------------------------ Utilities ------------------------------------
# =============================================================================

def to_numpy_f64(x):
    """Convert input (numpy/torch/list/scalar) to float64 numpy array or float."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            return arr.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

def vec1d(x, n_expected=None):
    """Convert to float64 1D vector and optionally check length."""
    v = to_numpy_f64(x).reshape(-1)
    if n_expected is not None and v.size != n_expected:
        raise ValueError(f"Expected size {n_expected}, got {v.size}")
    return v


def _ts_step_quality(
    actual_change: float,
    predicted_change: float,
) -> tuple[float | None, float]:
    """Return ``rho`` and symmetric TS quality ``Q = 1 - |rho - 1|``."""
    actual_change = float(actual_change)
    predicted_change = float(predicted_change)
    if (
        not np.isfinite(actual_change)
        or not np.isfinite(predicted_change)
        or abs(predicted_change) <= 1.0e-16
    ):
        return None, float("-inf")
    rho = actual_change / predicted_change
    if not np.isfinite(rho):
        return None, float("-inf")
    return rho, 1.0 - abs(rho - 1.0)


def write_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, 
              iteration: Optional[int] = None):
    """
    Write a single geometry to XYZ file.
    
    Parameters
    ----------
    filename : str
        Output file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int, optional
        Iteration number to include in comment line
    """
    if atoms.info.get("pdb_template"):
        write_pdb(filename, atoms, atoms.info["pdb_template"])
        return
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        
        # Build comment line
        comment_parts = []
        if iteration is not None:
            comment_parts.append(f"Iteration {iteration}")
        if energy is not None:
            comment_parts.append(f"Energy = {energy:.10f}")
        
        if comment_parts:
            f.write("  ".join(comment_parts) + "\n")
        else:
            f.write("TS optimization\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def append_xyz_trajectory(filename: str, atoms: Atoms, energy: Optional[float] = None,
                         iteration: int = 0):
    """
    Append a geometry to XYZ trajectory file.
    
    Parameters
    ----------
    filename : str
        Output trajectory file path
    atoms : Atoms
        ASE Atoms object
    energy : float, optional
        Energy value to include in comment line
    iteration : int
        Iteration number
    """
    # Remove file if first iteration
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    if atoms.info.get("pdb_template"):
        remark = f"Iteration {iteration}"
        if energy is not None:
            remark += f"  Energy = {energy:.10f}"
        with open(filename, "a", encoding="utf-8") as f:
            write_pdb_model(
                f,
                atoms,
                atoms.info["pdb_template"],
                model_index=iteration + 1,
                remark=remark,
            )
        return
    
    pos = to_numpy_f64(atoms.get_positions())
    symbols = atoms.get_chemical_symbols()
    
    with open(filename, "a") as f:
        f.write(f"{len(symbols)}\n")
        
        if energy is not None:
            f.write(f"Iteration {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iteration {iteration}\n")
        
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------- PRFO Core -----------------------------------
# =============================================================================

def prfo_step(H, g, is_ts=False, target_mode=None, trust_radius=0.2,
              evals_eps=1e-10, mu_margin=1e-8, max_bisect_it=60,
              pre_eig=None):
    """
    Restricted-step partitioned RFO step in the coordinates of ``H`` and ``g``.

    The target TS mode uses the highest root of its one-dimensional augmented
    Hessian (maximization); the complementary subspace uses the lowest root
    (minimization).  A shared generalized-eigenproblem scale ``alpha >= 1``
    restricts the combined step to the trust radius.

    Parameters
    ----------
    H : (n,n) array-like
        Hessian matrix
    g : (n,) array-like
        Gradient vector (NOT forces)
    is_ts : bool
        TS mode if True (enables partition into uphill/downhill subspaces)
    target_mode : int or None
        Uphill mode index in TS mode (defaults to most negative)
    trust_radius : float
        Trust radius bound measured in coordinates of H,g
    evals_eps : float
        Eigenvalue regularization threshold
    mu_margin : float
        Retained for call compatibility; RS-P-RFO uses ``alpha`` bisection.
    max_bisect_it : int
        Maximum bisection iterations
    pre_eig : optional tuple
        Pre-computed eigendecomposition (w, V, g_proj) to reuse

    Returns
    -------
    s : (n,) array
        Step in the SAME coordinates as H,g
    """
    H = to_numpy_f64(H)
    if H.ndim == 3 and H.shape[0] == 1:
        H = H[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square 2D, got shape={H.shape}")
    n = H.shape[0]

    g = vec1d(g, n)
    trust_radius = float(trust_radius)
    evals_eps = float(evals_eps)
    if not np.all(np.isfinite(H)):
        raise FloatingPointError("P-RFO Hessian contains non-finite values")
    if not np.all(np.isfinite(g)):
        raise FloatingPointError("P-RFO gradient contains non-finite values")
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("P-RFO trust radius must be finite and positive")
    if not np.isfinite(evals_eps) or evals_eps <= 0.0:
        raise ValueError("P-RFO eigenvalue threshold must be finite and positive")

    # Eigendecomposition (or reuse)
    if pre_eig is not None:
        w, V, gp = pre_eig
        w = vec1d(w, n)
        V = to_numpy_f64(V)
        gp = vec1d(gp, n)
        if V.shape != (n, n):
            raise ValueError("pre_eig V has wrong shape.")
        if (
            not np.all(np.isfinite(w))
            or not np.all(np.isfinite(V))
            or not np.all(np.isfinite(gp))
        ):
            raise FloatingPointError(
                "P-RFO precomputed eigendecomposition contains non-finite values"
            )
    else:
        w, V = np.linalg.eigh(H)
        gp = V.T @ g

    # Partition into the maximized target mode and minimized complement.
    if is_ts:
        if target_mode is None:
            j = int(np.argmin(w))
        else:
            j = int(target_mode)
        if j < 0 or j >= n:
            raise ValueError(f"target_mode={j} is outside [0, {n})")
        maximize_idx = np.asarray([j], dtype=int)
        minimize_idx = np.asarray(
            [index for index in range(n) if index != j],
            dtype=int,
        )
    else:
        maximize_idx = np.zeros(0, dtype=int)
        minimize_idx = np.arange(n, dtype=int)

    def augmented_subspace_step(
        indices: np.ndarray,
        *,
        maximize: bool,
        alpha: float,
    ) -> np.ndarray:
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        curvatures = w[indices]
        gradients = gp[indices]
        inv_sqrt_alpha = 1.0 / np.sqrt(alpha)
        augmented = np.zeros(
            (indices.size + 1, indices.size + 1),
            dtype=np.float64,
        )
        augmented[0, 1:] = gradients * inv_sqrt_alpha
        augmented[1:, 0] = gradients * inv_sqrt_alpha
        augmented[1:, 1:] = np.diag(curvatures / alpha)
        roots = np.linalg.eigvalsh(augmented)
        root = float(roots[-1] if maximize else roots[0])
        denominator = curvatures - alpha * root
        replacement = np.where(
            denominator < 0.0,
            -evals_eps,
            evals_eps,
        )
        denominator = np.where(
            np.abs(denominator) < evals_eps,
            replacement,
            denominator,
        )
        return -gradients / denominator

    def step_for_alpha(alpha: float) -> np.ndarray:
        projected = np.zeros(n, dtype=np.float64)
        if maximize_idx.size:
            projected[maximize_idx] = augmented_subspace_step(
                maximize_idx,
                maximize=True,
                alpha=alpha,
            )
        if minimize_idx.size:
            projected[minimize_idx] = augmented_subspace_step(
                minimize_idx,
                maximize=False,
                alpha=alpha,
            )
        step = V @ projected
        if not np.all(np.isfinite(step)):
            raise FloatingPointError("P-RFO step solver produced non-finite values")
        return step

    unrestricted = step_for_alpha(1.0)
    if float(np.linalg.norm(unrestricted)) <= trust_radius:
        return unrestricted

    alpha_lo = 1.0
    alpha_hi = 2.0
    restricted = step_for_alpha(alpha_hi)
    brackets = 0
    while (
        float(np.linalg.norm(restricted)) > trust_radius
        and brackets < max_bisect_it
    ):
        alpha_lo = alpha_hi
        alpha_hi *= 2.0
        restricted = step_for_alpha(alpha_hi)
        brackets += 1
    if float(np.linalg.norm(restricted)) > trust_radius:
        raise RuntimeError("P-RFO could not bracket the restricted step")

    for _ in range(max_bisect_it):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        candidate = step_for_alpha(alpha_mid)
        norm = float(np.linalg.norm(candidate))
        if norm > trust_radius:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
            restricted = candidate
        if abs(norm - trust_radius) <= 1.0e-10 * max(1.0, trust_radius):
            restricted = candidate
            break

    return restricted

def calculate_Hessian(atoms: Atoms):
    """
    Calculate Hessian matrix from calculator.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with calculator attached
    
    Returns
    -------
    H : array
        Hessian matrix in float64
    """
    calc = atoms.calc
    H = calc.get_hessian(atoms)
    return to_numpy_f64(H)


def _rigid_internal_complement(atoms: Atoms) -> np.ndarray:
    """Orthonormal MW internal coordinates for a free, non-periodic molecule.

    The rigid rank is geometry dependent (three translations plus two or
    three rotations), so it must not be replaced with an unconditional six-DOF
    subtraction.  Complete QR of the existing orthonormal rigid basis supplies
    its complement without forming an ill-conditioned explicit projector.
    """
    rigid = mass_weighted_rigid_basis(atoms)
    n_dof = 3 * len(atoms)
    if rigid.ndim != 2 or rigid.shape[0] != n_dof:
        raise ValueError("Rigid-body basis has an invalid shape")
    if rigid.shape[1] >= n_dof:
        raise NotImplementedError(
            "Free-molecule PRFO requires at least one internal degree of freedom"
        )
    complete_basis = np.linalg.qr(rigid, mode="complete")[0]
    return complete_basis[:, rigid.shape[1]:]


def _resolve_prfo_coordinate_space(
    atoms: Atoms, requested: str, calculator=None
) -> str:
    """Resolve one shared scalar/batch TS symmetry contract, failing closed.

    Geometry does not prove rigid invariance: a laboratory-frame field can
    break it. Unknown backends require an explicit assertion by the caller.
    """
    requested = str(requested).lower()
    if requested not in {"auto", "free_molecule", "cartesian_external"}:
        raise ValueError(
            "PRFO rigid_symmetry must be 'auto', 'free_molecule', or "
            "'cartesian_external'"
        )
    if np.any(atoms.get_pbc()):
        raise NotImplementedError(
            "PRFO periodic TS search has no validated symmetry/curvature "
            "contract and fails closed."
        )
    backend = atoms.calc if calculator is None else calculator
    capability = getattr(backend, "rigid_body_invariant", None)
    if capability is not None and not isinstance(capability, (bool, np.bool_)):
        raise TypeError("Calculator rigid_body_invariant must be boolean")
    declared = None if capability is None else bool(capability)
    if requested == "auto":
        if declared is None:
            raise NotImplementedError(
                "PRFO cannot infer rigid-body symmetry from this calculator. "
                "Set paras={'prfo': {'rigid_symmetry': 'free_molecule'}} "
                "only for a free rigid-invariant PES, or use "
                "'cartesian_external' for an explicitly laboratory-frame "
                "PES; unknown symmetry fails closed."
            )
        requested = "free_molecule" if declared else "cartesian_external"
    elif declared is False and requested == "free_molecule":
        raise ValueError(
            "Calculator is not rigid-body invariant; free-molecule "
            "projection would change its physical problem."
        )
    elif declared is True and requested == "cartesian_external":
        raise ValueError(
            "Calculator is rigid-body invariant; cartesian_external "
            "could mistake rigid curvature for an internal TS mode."
        )
    return requested


def _bfgs_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BFGS update of Hessian approximation."""
    H = to_numpy_f64(H)
    s = vec1d(s)
    y = vec1d(y, s.size)
    ys = float(y.dot(s))
    if ys <= 1e-12:
        return H
    Hs = H.dot(s)
    sHs = float(s.dot(Hs))
    if sHs <= 1e-12:
        return H
    H_new = H + np.outer(y, y) / ys - np.outer(Hs, Hs) / sHs
    return 0.5 * (H_new + H_new.T)

def _bofill_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bofill = mixed MS/SR1 + PSB.
    If the SR1/MS denominator is unsafe, fall back to pure PSB.
    """
    H = to_numpy_f64(H)
    s = vec1d(s)
    y = vec1d(y, s.size)

    s2 = float(np.dot(s, s))
    y2 = float(np.dot(y, y))
    if s2 <= 1e-16 or y2 <= 1e-16:
        return H

    # Residual: z = Δg - H Δx
    z = y - H.dot(s)
    z2 = float(np.dot(z, z))
    sz = float(np.dot(s, z))

    # PSB term is safe as long as s2 is nonzero, already guaranteed above.
    psb = (
        (np.outer(z, s) + np.outer(s, z)) / s2
        - (sz / (s2 * s2)) * np.outer(s, s)
    )

    # if s·z is too small, do NOT divide by it; use pure PSB.
    use_sr1 = (abs(sz) > 1e-8) and (z2 > 1e-16)

    if use_sr1:
        ms = np.outer(z, z) / sz
        if s2 > 1e-16 and z2 > 1e-16:
            ratio = (sz * sz) / (s2 * z2)
            phi = float(np.clip(1.0 - ratio, 0.0, 1.0))
        else:
            phi = 1.0
    else:
        ms = np.zeros_like(H)
        phi = 1.0

    H_new = H + (1.0 - phi) * ms + phi * psb
    return 0.5 * (H_new + H_new.T)

# =============================================================================
# ------------------------------- PRFO Parameters -----------------------------
# =============================================================================

@dataclass
class PRFOParams:
    """Parameters for PRFO transition state search."""
    
    # Optimization control
    max_iter: int = 256                    # Maximum optimization iterations
    
    # Trust region parameters
    trust_radius: float = 0.2              # Initial trust radius (in MW coords)
    trust_min: float = 1e-3                # Minimum trust radius
    trust_max: float = 1.0                 # Maximum trust radius
    
    # TS step-quality thresholds Q = 1 - |actual/predicted - 1|
    eta_reject: float = 0.0
    eta_shrink: float = 0.5
    eta_expand: float = 0.75
    
    # PRFO step parameters
    evals_eps: float = 1e-10               # Eigenvalue regularization threshold
    mu_margin: float = 1e-8                # Safety margin for bisection
    max_bisect_it: int = 60                # Maximum bisection iterations
    recalc: int = 1                        # Exact Hessian recalculation interval
    hessian_update: str = "bofill"         # Working Hessian update: bofill or bfgs
    inertia_threshold: float = 1e-6        # Significant negative MW Hessian mode
    stagnation_threshold: float = 1e-12    # Zero-step / no-progress threshold
    rigid_symmetry: str = "auto"            # free_molecule | cartesian_external | auto
    
    # Convergence thresholds (should be set from atoms object)
    f_max_th: float = 9.5e-3               # Maximum force threshold (Eh/Angstrom)
    f_rms_th: float = 5e-3                 # RMS force threshold (Eh/Angstrom)
    dp_max_th: float = 1.8e-3              # Maximum displacement threshold (Angstrom)
    dp_rms_th: float = 1.2e-3              # RMS displacement threshold (Angstrom)


class PRFOStatus(str, Enum):
    """Optimization status; none of these values certifies a transition state."""

    GEOMETRY_CONVERGED = "geometry_converged"
    FAILED_NONFINITE = "failed_nonfinite"
    FAILED_BACKEND = "failed_backend"
    FAILED_HESSIAN = "failed_hessian"
    FAILED_WRONG_INERTIA = "failed_wrong_inertia"
    FAILED_STAGNATION = "failed_stagnation"
    FAILED_STEP_SOLVER = "failed_step_solver"
    FAILED_MODE_TRACKING = "failed_mode_tracking"
    FAILED_MAXITER = "failed_maxiter"


@dataclass(frozen=True)
class PRFOResult:
    """A geometry candidate is not a frequency/IRC-verified TS.

    ``negative_modes`` counts the selected physical space: internal modes for
    a rigid-invariant free molecule, full Cartesian modes for an explicitly
    non-invariant laboratory-frame problem.
    """

    atoms: Atoms
    status: PRFOStatus
    iterations: int
    structure_path: str
    negative_modes: int | None = None
    detail: str = ""

    @property
    def geometry_converged(self) -> bool:
        return self.status is PRFOStatus.GEOMETRY_CONVERGED


class PRFOConvergenceError(RuntimeError):
    def __init__(self, result: PRFOResult):
        super().__init__(
            "PRFO did not produce a geometry-converged first-order-saddle "
            "candidate: "
            f"status={result.status.value}, iterations={result.iterations}, "
            f"diagnostic={result.structure_path}"
        )
        self.result = result


# =============================================================================
# ------------------------------- PRFO Class ----------------------------------
# =============================================================================

class PRFO(JobABC):
    """
    Transition-state-candidate search using restricted-step partitioned RFO.

    The optimization is performed in mass-weighted coordinates for the step
    computation and trust-region enforcement, while geometry updates are done
    in Cartesian coordinates.
    """

    def __init__(self,
                 output: str,
                 atoms: Atoms,
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        require_shared_pdb_writer([atoms], "PRFO")
        if bool(getattr(atoms, "constraints", None)):
            raise NotImplementedError(
                "PRFO does not yet project fixed/constrained degrees of "
                "freedom from the Hessian; constrained TS search fails closed."
            )

        # Initialize params from paras dict
        self.params = self._init_params(PRFOParams, paras, ("prfo", "PRFO", "ts"))
        self.params.recalc = int(self.params.recalc)
        self.params.max_iter = int(self.params.max_iter)
        self.params.max_bisect_it = int(self.params.max_bisect_it)
        self.params.hessian_update = str(self.params.hessian_update).lower()
        if self.params.hessian_update not in {"bofill", "bfgs"}:
            raise ValueError("Hessian update method must be 'bofill' or 'bfgs'.")

        # Override convergence thresholds from atoms if available
        for attr in ('f_max_th', 'f_rms_th', 'dp_max_th', 'dp_rms_th'):
            if hasattr(atoms, attr):
                setattr(self.params, attr, getattr(atoms, attr))
        self._validate_params()
        self._coordinate_space = self._resolve_coordinate_space(atoms)
        if self._coordinate_space == "free_molecule" and len(atoms) < 2:
            raise NotImplementedError(
                "Free-molecule PRFO requires at least one internal degree "
                "of freedom."
            )

        # Mode tracking
        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None
        self.result: PRFOResult | None = None

    def _validate_params(self) -> None:
        if self.params.max_iter < 0:
            raise ValueError("PRFO max_iter must be a non-negative integer")
        if self.params.recalc <= 0:
            raise ValueError("PRFO recalc must be a positive integer")
        if self.params.max_bisect_it <= 0:
            raise ValueError("PRFO max_bisect_it must be a positive integer")

        positive = (
            "trust_radius",
            "trust_min",
            "trust_max",
            "evals_eps",
            "mu_margin",
            "inertia_threshold",
            "stagnation_threshold",
            "f_max_th",
            "f_rms_th",
            "dp_max_th",
            "dp_rms_th",
        )
        for name in positive:
            try:
                value = _finite_positive_real(getattr(self.params, name))
            except ValueError as exc:
                raise ValueError(
                    f"PRFO {name} must be a finite positive real scalar"
                ) from exc
            setattr(self.params, name, value)
        if not (
            self.params.trust_min
            <= self.params.trust_radius
            <= self.params.trust_max
        ):
            raise ValueError(
                "PRFO trust radii must satisfy trust_min <= trust_radius <= trust_max"
            )
        qualities = (
            float(self.params.eta_reject),
            float(self.params.eta_shrink),
            float(self.params.eta_expand),
        )
        if (
            not np.all(np.isfinite(qualities))
            or not (
                0.0
                <= qualities[0]
                <= qualities[1]
                <= qualities[2]
                <= 1.0
            )
        ):
            raise ValueError(
                "PRFO quality thresholds must be finite and ordered as "
                "0 <= eta_reject <= eta_shrink <= eta_expand <= 1"
            )
        (
            self.params.eta_reject,
            self.params.eta_shrink,
            self.params.eta_expand,
        ) = qualities

    def _resolve_coordinate_space(self, atoms: Atoms) -> str:
        return _resolve_prfo_coordinate_space(atoms, self.params.rigid_symmetry)
    
    def atoms_to_xyz(self, atoms: Atoms) -> str:
        """Convert Atoms object to XYZ format string."""
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            lines.append(f"{s:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
        return "\n".join(lines) + "\n"
    
    def _terminal_result(
        self,
        *,
        status: PRFOStatus,
        iteration: int,
        energy: float | None,
        negative_modes: int | None,
        detail: str,
    ) -> PRFOResult:
        base, _ = os.path.splitext(self.output)
        extension = (
            ".pdb" if self.atoms.info.get("pdb_template") else ".xyz"
        )
        suffix = {
            PRFOStatus.GEOMETRY_CONVERGED: "_prfo_ts_candidate",
            PRFOStatus.FAILED_NONFINITE: "_prfo_nonfinite",
            PRFOStatus.FAILED_BACKEND: "_prfo_backend_failure",
            PRFOStatus.FAILED_HESSIAN: "_prfo_hessian_failure",
            PRFOStatus.FAILED_WRONG_INERTIA: "_prfo_wrong_inertia",
            PRFOStatus.FAILED_STAGNATION: "_prfo_stagnation",
            PRFOStatus.FAILED_STEP_SOLVER: "_prfo_step_failure",
            PRFOStatus.FAILED_MODE_TRACKING: "_prfo_mode_failure",
            PRFOStatus.FAILED_MAXITER: "_prfo_unconverged",
        }[status]
        structure_path = base + suffix + extension
        safe_energy = (
            float(energy)
            if energy is not None and np.isfinite(float(energy))
            else None
        )
        write_xyz(
            structure_path,
            self.atoms,
            energy=safe_energy,
            iteration=iteration,
        )
        if status is PRFOStatus.GEOMETRY_CONVERGED:
            heading = "Geometry and first-order inertia converged"
            qualification = (
                "This is a first-order-saddle candidate, not a "
                "frequency/IRC-verified transition state."
            )
        else:
            heading = f"PRFO terminated: {status.value}"
            qualification = "No transition-state candidate was produced."
        log_info(
            [
                "\n\n" + "-" * 70 + "\n",
                f"{heading.center(70)}\n",
                f"{detail}\n",
                f"{qualification}\n",
                f"Diagnostic structure: {structure_path}\n",
            ],
            self.output,
        )
        result = PRFOResult(
            atoms=self.atoms,
            status=status,
            iterations=int(iteration),
            structure_path=structure_path,
            negative_modes=negative_modes,
            detail=detail,
        )
        self.result = result
        return result
    
    def log_iteration(self, iteration: int, atoms: Atoms, E: float,
                     model_change: float, actual_change: float,
                     rho: Optional[float], quality: float, trust_radius: float,
                     norm_mw: float, on_boundary: bool):
        """
        Log detailed information for current iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        atoms : Atoms
            Current geometry
        E : float
            Current energy
        model_change : float
            Predicted energy change by model
        actual_change : float
            Actual energy change
        rho : float or None
            Model agreement ratio
        quality : float
            Symmetric TS model quality, ``1 - abs(rho - 1)``
        trust_radius : float
            Current trust radius
        norm_mw : float
            Step norm in MW coordinates
        on_boundary : bool
            Whether step is on trust radius boundary
        """
        iter_str = f"Iteration: {iteration}"
        info_message = ['\n' + '-' * 70 + '\n', f'{iter_str.center(70)}\n\n']
        
        # Coordinates section
        info_message.append(f'\n{"Coordinates".center(70)}\n')
        info_message.append('-' * 70 + '\n')
        for atom_index, atom in enumerate(atoms):
            element_type = atom.symbol
            coord = atom.position
            info_message.append(
                f"{atom_index:<4} {element_type:<2} "
                f"{coord[0]:>20.4f} {coord[1]:>20.4f} {coord[2]:>20.4f}\n"
            )
        
        # Energy and convergence
        info_message.append(
            f"\n\nEnergy:                {E:>12.6f} "
            "Convergence criteria  Is converged \n"
        )
        
        # Force convergence
        if atoms.max_f > self.params.f_max_th:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Force:         {atoms.max_f:>12.6f} "
                f"{self.params.f_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_f > self.params.f_rms_th:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Force:             {atoms.rms_f:>12.6f} "
                f"{self.params.f_rms_th:>12.6f}                Yes\n"
            )
        
        # Displacement convergence
        if atoms.max_dp > self.params.dp_max_th:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"Maximum Displacement:  {atoms.max_dp:>12.6f} "
                f"{self.params.dp_max_th:>12.6f}                Yes\n"
            )
        
        if atoms.rms_dp > self.params.dp_rms_th:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                No\n"
            )
        else:
            info_message.append(
                f"RMS Displacement:      {atoms.rms_dp:>12.6f} "
                f"{self.params.dp_rms_th:>12.6f}                Yes\n"
            )
        
        # Trust region info
        rho_str = f"{rho:.3f}" if rho is not None else "nan"
        info_message.append(
            f"\nModel change: {model_change: .6e}  "
            f"Actual change: {actual_change: .6e}  "
            f"rho: {rho_str}  Q: {quality:.3f}\n"
        )
        info_message.append(
            f"Trust radius (MW): {trust_radius: .6f}  "
            f"Step norm (MW): {norm_mw: .6f}  "
            f"On boundary: {on_boundary}\n"
        )
        
        log_info(info_message, self.output)

    def update_mode_tracking(self, w_mw: np.ndarray, V_mw: np.ndarray,
                           gp_mw: np.ndarray) -> int:
        """
        Update mode-following for transition state search.
        
        Parameters
        ----------
        w_mw : array
            Eigenvalues in the selected mass-weighted physical space
        V_mw : array
            Physical eigenvectors embedded in full mass-weighted Cartesian
            coordinates, so overlap remains meaningful as the basis changes
        gp_mw : array
            Projected gradient in eigen-basis
        
        Returns
        -------
        int
            Index of tracked mode
        """
        # Rigid modes have already been removed for free molecules.  Keep the
        # existing curvature significance threshold for genuine soft modes.
        significant = np.flatnonzero(
            np.abs(w_mw) > self.params.inertia_threshold
        )
        if significant.size == 0:
            raise RuntimeError(
                "No significant Hessian mode is available for tracking"
            )
        if self.tracked_mode_vec_mw is None:
            # Start on the lowest significant curvature. Subsequent
            # iterations preserve mode identity by overlap.
            self.tracked_mode_idx = int(
                significant[np.argmin(w_mw[significant])]
            )
            self.tracked_mode_vec_mw = V_mw[:, self.tracked_mode_idx].copy()
        else:
            # Follow the significant mode with maximum overlap.
            overlaps = np.abs(V_mw.T @ self.tracked_mode_vec_mw)
            self.tracked_mode_idx = int(
                significant[np.argmax(overlaps[significant])]
            )
            
            # Keep consistent sign to avoid flips
            sign_align = np.sign(np.dot(V_mw[:, self.tracked_mode_idx],
                                       self.tracked_mode_vec_mw))
            if sign_align == 0.0:
                sign_align = 1.0
            self.tracked_mode_vec_mw = \
                V_mw[:, self.tracked_mode_idx] * sign_align
        
        return self.tracked_mode_idx
    
    def run(self) -> Atoms:
        """Run PRFO and return a geometry-converged TS candidate.

        This compatibility wrapper preserves the historical ``Atoms`` return
        type while failing closed on max-iteration termination. Call
        :meth:`run_result` when the structured status is needed.
        """
        result = self.run_result()
        if not result.geometry_converged:
            raise PRFOConvergenceError(result)
        return result.atoms

    def run_result(self) -> PRFOResult:
        """Run RS-P-RFO and return a structured, fail-closed status."""
        sys.setrecursionlimit(1000)

        atoms = self.atoms
        trust_radius = self.params.trust_radius

        iteration = 0

        # Setup trajectory file
        base, _ = os.path.splitext(self.output)
        ext = ".pdb" if atoms.info.get("pdb_template") else ".xyz"
        traj_file = base + "_prfo_traj" + ext
        # Log header
        info_message = [
            "\nStarting Transition State Search (TS) with RS-PRFO...\n",
            f"Hessian recalc interval: {self.params.recalc}; "
            f"update method: {self.params.hessian_update}\n",
            "Trust quality Q=1-|actual/predicted-1|: "
            f"reject<={self.params.eta_reject}, "
            f"shrink<{self.params.eta_shrink}, "
            f"expand>={self.params.eta_expand}\n",
            f"Force convergence thresholds: "
            f"f_max={self.params.f_max_th:.6f}, "
            f"f_rms={self.params.f_rms_th:.6f}\n",
            f"Displacement diagnostics (not success criteria): "
            f"dp_max={self.params.dp_max_th:.6f}, "
            f"dp_rms={self.params.dp_rms_th:.6f}\n",
            f"Coordinate-space contract: {self._coordinate_space}\n",
        ]
        log_info(info_message, self.output)

        H_work = None
        last_max_dp = 0.0
        last_rms_dp = 0.0

        while True:
            X = to_numpy_f64(atoms.get_positions()).reshape(-1, 3)
            if not np.all(np.isfinite(X)):
                return self._terminal_result(
                    status=PRFOStatus.FAILED_NONFINITE,
                    iteration=iteration,
                    energy=None,
                    negative_modes=None,
                    detail="Current coordinates contain non-finite values.",
                )
            try:
                E_old = float(
                    atoms.get_potential_energy(force_consistent=True)
                )
                F_cart = to_numpy_f64(atoms.get_forces())
            except Exception as exc:
                return self._terminal_result(
                    status=PRFOStatus.FAILED_BACKEND,
                    iteration=iteration,
                    energy=None,
                    negative_modes=None,
                    detail=f"Energy/force backend failed: {type(exc).__name__}: {exc}",
                )
            if not np.isfinite(E_old) or not np.all(np.isfinite(F_cart)):
                return self._terminal_result(
                    status=PRFOStatus.FAILED_NONFINITE,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=None,
                    detail="Current energy or force contains non-finite values.",
                )
            expected_force_shape = (len(atoms), 3)
            if F_cart.shape != expected_force_shape:
                return self._terminal_result(
                    status=PRFOStatus.FAILED_BACKEND,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=None,
                    detail=(
                        f"Current force shape is {F_cart.shape}; expected "
                        f"{expected_force_shape}."
                    ),
                )
            g_cart = vec1d(-F_cart)
            atoms.max_f = float(np.max(np.abs(F_cart)))
            atoms.rms_f = float(np.sqrt(np.mean(F_cart * F_cart)))
            atoms.max_dp = last_max_dp
            atoms.rms_dp = last_rms_dp
            forces_converged = (
                atoms.max_f <= self.params.f_max_th
                and atoms.rms_f <= self.params.f_rms_th
            )

            need_recalc = (
                forces_converged
                or H_work is None
                or (iteration % self.params.recalc == 0)
            )
            if need_recalc:
                try:
                    H_cart = to_numpy_f64(calculate_Hessian(atoms))
                except Exception as exc:
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_HESSIAN,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=None,
                        detail=f"Hessian backend failed: {type(exc).__name__}: {exc}",
                    )
                if H_cart.ndim == 3 and H_cart.shape[0] == 1:
                    H_cart = H_cart[0]
                if (
                    H_cart.ndim != 2
                    or H_cart.shape[0] != H_cart.shape[1]
                    or H_cart.shape[0] != g_cart.size
                ):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_HESSIAN,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=None,
                        detail=(
                            f"Hessian shape {H_cart.shape} is incompatible with "
                            f"gradient size {g_cart.size}."
                        ),
                    )
                if not np.all(np.isfinite(H_cart)):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=None,
                        detail="Hessian contains non-finite values.",
                    )
                H_cart = 0.5 * (H_cart + H_cart.T)
                H_work = H_cart.copy()
            else:
                H_cart = H_work

            n3 = H_cart.shape[0]
            masses = to_numpy_f64(atoms.get_masses())
            if (
                not np.all(np.isfinite(masses))
                or np.any(masses <= 0.0)
            ):
                return self._terminal_result(
                    status=PRFOStatus.FAILED_HESSIAN,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=None,
                    detail="Atomic masses must be finite and positive.",
                )
            D = vec1d(1.0 / np.sqrt(np.repeat(masses, 3)), n3)
            g_mw = vec1d(D * g_cart, n3)
            H_mw = (D[:, None] * H_cart) * D[None, :]
            try:
                basis = (
                    _rigid_internal_complement(atoms)
                    if self._coordinate_space == "free_molecule"
                    else np.eye(n3, dtype=np.float64)
                )
                H_physical = basis.T @ H_mw @ basis
                H_physical = 0.5 * (H_physical + H_physical.T)
                g_physical = basis.T @ g_mw
                w_physical, V_physical = np.linalg.eigh(H_physical)
            except (ValueError, np.linalg.LinAlgError, NotImplementedError) as exc:
                return self._terminal_result(
                    status=PRFOStatus.FAILED_HESSIAN,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=None,
                    detail=f"Physical Hessian eigensystem failed: {exc}",
                )
            n_physical = H_physical.shape[0]
            gp_physical = vec1d(V_physical.T @ g_physical, n_physical)
            V_modes_mw = basis @ V_physical
            if (
                not np.all(np.isfinite(H_physical))
                or not np.all(np.isfinite(g_physical))
                or not np.all(np.isfinite(w_physical))
                or not np.all(np.isfinite(V_physical))
                or not np.all(np.isfinite(gp_physical))
                or not np.all(np.isfinite(V_modes_mw))
            ):
                return self._terminal_result(
                    status=PRFOStatus.FAILED_HESSIAN,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=None,
                    detail="Hessian eigendecomposition contains non-finite values.",
                )
            negative_modes = int(
                np.count_nonzero(w_physical < -self.params.inertia_threshold)
            )

            if forces_converged:
                if negative_modes == 1:
                    return self._terminal_result(
                        status=PRFOStatus.GEOMETRY_CONVERGED,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Forces and first-order inertia converged.",
                    )
                return self._terminal_result(
                    status=PRFOStatus.FAILED_WRONG_INERTIA,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=negative_modes,
                    detail=(
                        "Geometry converged but the mass-weighted physical "
                        "Hessian has "
                        f"{negative_modes} significant negative modes; expected 1."
                    ),
                )

            if iteration >= self.params.max_iter:
                return self._terminal_result(
                    status=PRFOStatus.FAILED_MAXITER,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=negative_modes,
                    detail=(
                        "PRFO exhausted its accepted-step budget before force "
                        "and first-order-inertia convergence."
                    ),
                )

            try:
                tracked_mode_idx = self.update_mode_tracking(
                    w_physical,
                    V_modes_mw,
                    gp_physical,
                )
            except Exception as exc:
                return self._terminal_result(
                    status=PRFOStatus.FAILED_MODE_TRACKING,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=negative_modes,
                    detail=f"Mode tracking failed: {type(exc).__name__}: {exc}",
                )

            accepted = False
            max_attempts = 8
            attempts = 0
            while not accepted and attempts < max_attempts:
                attempts += 1
                try:
                    s_physical = prfo_step(
                        H=H_physical,
                        g=g_physical,
                        is_ts=True,
                        target_mode=tracked_mode_idx,
                        trust_radius=trust_radius,
                        evals_eps=self.params.evals_eps,
                        mu_margin=self.params.mu_margin,
                        max_bisect_it=self.params.max_bisect_it,
                        pre_eig=(w_physical, V_physical, gp_physical),
                    )
                    s_mw = basis @ s_physical
                except Exception as exc:
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_STEP_SOLVER,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail=f"P-RFO step solver failed: {type(exc).__name__}: {exc}",
                    )
                norm_mw = float(np.linalg.norm(s_mw))
                if not np.isfinite(norm_mw) or not np.all(np.isfinite(s_mw)):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="P-RFO step contains non-finite values.",
                    )
                if norm_mw <= self.params.stagnation_threshold:
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_STAGNATION,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="P-RFO produced a zero step before force convergence.",
                    )
                on_boundary = (
                    abs(norm_mw - trust_radius)
                    <= 1e-6 * max(1.0, trust_radius)
                )
                s_cart = vec1d(D * s_mw, n3)
                if not np.all(np.isfinite(s_cart)):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Cartesian P-RFO step contains non-finite values.",
                    )
                Hs = H_cart @ s_cart
                model_change = float(g_cart.dot(s_cart) + 0.5 * s_cart.dot(Hs))
                if not np.isfinite(model_change):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Predicted P-RFO model change is non-finite.",
                    )
                X_new = X.reshape(-1, 3) + s_cart.reshape(-1, 3)
                if not np.all(np.isfinite(X_new)):
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Trial coordinates contain non-finite values.",
                    )
                atoms.set_positions(X_new)
                try:
                    E_new = float(
                        atoms.get_potential_energy(force_consistent=True)
                    )
                except Exception as exc:
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_BACKEND,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail=f"Trial energy failed: {type(exc).__name__}: {exc}",
                    )
                if not np.isfinite(E_new):
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Trial energy is non-finite.",
                    )
                actual_change = float(E_new - E_old)
                if not np.isfinite(actual_change):
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Actual trial energy change is non-finite.",
                    )
                rho, quality = _ts_step_quality(
                    actual_change,
                    model_change,
                )
                if quality <= self.params.eta_reject:
                    atoms.set_positions(X)
                    trust_radius = max(
                        self.params.trust_min,
                        0.5 * min(trust_radius, norm_mw),
                    )
                    continue

                accepted = True
                if quality < self.params.eta_shrink:
                    trust_radius = max(
                        self.params.trust_min,
                        0.5 * min(trust_radius, norm_mw),
                    )
                elif quality >= self.params.eta_expand and on_boundary:
                    trust_radius = min(
                        self.params.trust_max,
                        np.sqrt(2.0) * trust_radius,
                    )
                try:
                    F_new = to_numpy_f64(atoms.get_forces())
                except Exception as exc:
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_BACKEND,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail=f"Trial force failed: {type(exc).__name__}: {exc}",
                    )
                if not np.all(np.isfinite(F_new)):
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_NONFINITE,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Trial force contains non-finite values.",
                    )
                if F_new.shape != expected_force_shape:
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_BACKEND,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail=(
                            f"Trial force shape is {F_new.shape}; expected "
                            f"{expected_force_shape}."
                        ),
                    )
                g_new_cart = vec1d(-F_new, n3)
                y_cart = g_new_cart - g_cart
                if self.params.hessian_update == "bfgs":
                    H_work = _bfgs_update(H_work, s_cart, y_cart)
                else:
                    H_work = _bofill_update(H_work, s_cart, y_cart)
                if not np.all(np.isfinite(H_work)):
                    atoms.set_positions(X)
                    return self._terminal_result(
                        status=PRFOStatus.FAILED_HESSIAN,
                        iteration=iteration,
                        energy=E_old,
                        negative_modes=negative_modes,
                        detail="Hessian update produced non-finite values.",
                    )

                dof = s_cart.size
                last_max_dp = float(np.max(np.abs(s_cart)))
                last_rms_dp = float(np.sqrt(np.sum(s_cart * s_cart) / dof))
                atoms.max_dp = last_max_dp
                atoms.rms_dp = last_rms_dp
                atoms.max_f = float(np.max(np.abs(F_new)))
                atoms.rms_f = float(np.sqrt(np.mean(F_new * F_new)))
                self.log_iteration(iteration + 1, atoms, E_new,
                                 model_change, actual_change, rho, quality,
                                 trust_radius, norm_mw, on_boundary)
                append_xyz_trajectory(traj_file, atoms, energy=E_new,
                                    iteration=iteration + 1)

            if not accepted:
                atoms.set_positions(X)
                return self._terminal_result(
                    status=PRFOStatus.FAILED_STAGNATION,
                    iteration=iteration,
                    energy=E_old,
                    negative_modes=negative_modes,
                    detail=(
                        f"No acceptable step after {max_attempts} trust-radius "
                        "attempts."
                    ),
                )
            iteration += 1
