# -*- coding: utf-8 -*-
"""
Dimer implementation with:
- Minimum-mode following (rotation: minimize kappa = n^T H n)
- Translation with parallel flip once kappa < 0
- Explicit analytic-HVP or central-force-difference evaluator selection
- Trust-radius / max-step control
- Detailed human-readable logging and XYZ outputs per iteration
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, List, Tuple

import numpy as np
from ase import Atoms

from .logger import log_info
from ._artifacts import archive_artifacts
from ...jobABC import JobABC
from maple.function.calculator._batch_eval import HVPEvaluator, energy_forces_one
from maple.function.read.filereader.pdb_reader import write_pdb_model, write_pdb_trajectory
from maple.function.utility.rigid_body import mass_weighted_rigid_basis

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

def write_xyz(filename: str, images: List[Atoms], energies: Optional[List[float]] = None):
    """
    Write a multi-frame XYZ trajectory. If energies given, write in comment line.
    """
    if images and images[0].info.get("pdb_template"):
        write_pdb_trajectory(filename, images, energies=energies)
        return
    with open(filename, "w") as f:
        for i, at in enumerate(images):
            pos = to_numpy_f64(at.get_positions())
            symbols = at.get_chemical_symbols()
            f.write(f"{len(symbols)}\n")
            if energies is not None:
                f.write(f"Image {i}  Energy = {energies[i]:.10f}\n")
            else:
                f.write(f"Image {i}\n")
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

def write_all_images_xyz(filename: str, atoms: Atoms, energy: Optional[float] = None, iteration: int = 0):
    """
    Append current single structure to an xyz trajectory file (for Dimer debug).
    Each block corresponds to one iteration of Dimer optimization.
    """
    if iteration == 0 and os.path.exists(filename):
        os.remove(filename)
    if atoms.info.get("pdb_template"):
        remark = f"Iter {iteration}"
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
            f.write(f"Iter {iteration}  Energy = {energy:.10f}\n")
        else:
            f.write(f"Iter {iteration}\n")
        for s, (x, y, z) in zip(symbols, pos):
            f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")

# =============================================================================
# ------------------------------ Dimer Params ---------------------------------
# =============================================================================

@dataclass
class DimerParams:
    # Rotation / curvature estimation
    use_hvp: bool = False               # if True, call hvp_fn(atoms, n_flat) for Hn
    delta: float = 0.005                 # Angstrom; used only if not use_hvp
    rot_max_iter: int = 5               # rotation inner iterations per outer step
    rot_alpha: float = 0.5              # rotation step factor on F_rot (unitless); small ~ (0.1~1)
    # Rotation-residual thresholds use Ha/Angstrom^2 when unweighted and
    # Ha/(Angstrom^2 amu) when the mass metric is active.
    rot_f_max_th: float = 1.0e-3
    rot_f_rms_th: float = 5.0e-4

    # Translation / trust region
    step0: float = 0.2                  # initial step scaling on search direction
    step_max: float = 0.15              # absolute max Cartesian displacement (Ang)
    trust_radius: float = 0.15          # same role as max_step; kept both for clarity

    # Convergence (translation forces)
    f_max_th: float = 5.0e-3            # max(|F_trans|) Eh/Ang
    f_rms_th: float = 1.0e-3            # RMS(F_trans) Eh/Ang
    kappa_to_flip: float = 0.0          # if kappa < this value, flip parallel component

    # Iterations
    max_iter: int = 200

    # Metric / projections
    use_mass_weight: bool = False       # if True, use M-metric for dot/proj (mass weighted)
    remove_rigid: bool = True           # remove global translation/rotation from n (molecular)

    # Initialization of n
    n_init: str = "random"              # "random" | "force" | "given"
    n_given: Optional[np.ndarray] = None

    # Outputs
    save_traj: bool = True
    save_metrics: bool = True


class DimerStatus(str, Enum):
    """Terminal geometry-search status; neither value certifies a TS."""

    GEOMETRY_CONVERGED = "geometry_converged"
    FAILED_NONNEGATIVE_CURVATURE = "failed_nonnegative_curvature"
    FAILED_MAX_ITER = "failed_max_iter"


@dataclass(frozen=True)
class DimerResult:
    atoms: Atoms
    status: DimerStatus
    iterations: int
    structure_path: str
    energy: float

    @property
    def geometry_converged(self) -> bool:
        return self.status is DimerStatus.GEOMETRY_CONVERGED

    @property
    def structure_file(self) -> str:
        """Backward-compatible descriptive alias for ``structure_path``."""
        return self.structure_path


class DimerConvergenceError(RuntimeError):
    """Raised by :meth:`Dimer.run` when no candidate geometry was produced."""

    def __init__(self, result: DimerResult):
        super().__init__(
            "Dimer did not produce a geometry-converged negative-curvature "
            f"candidate: status={result.status.value}, "
            f"iterations={result.iterations}, diagnostic={result.structure_path}"
        )
        self.result = result

# =============================================================================
# ------------------------------ Metric helpers -------------------------------
# =============================================================================

def _get_metric(atoms: Atoms, use_mass_weight: bool):
    if not use_mass_weight:
        return None
    m = to_numpy_f64(atoms.get_masses()).reshape(-1, 1)  # (N,1)
    if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("Mass-weighted Dimer requires finite positive masses")
    M = np.repeat(m, 3, axis=1).reshape(-1)              # (3N,)
    return M                                             # diagonal metric entries

def _dot(v, w, M=None):
    """Metric dot: if M is None -> Euclidean; else mass-weighted sum(M * v * w)."""
    if M is None:
        return float(np.dot(v, w))
    return float(np.dot(M * v, w))

def _norm(v, M=None):
    val = _dot(v, v, M=M)
    return math.sqrt(max(val, 0.0))

def _proj_parallel(F, n, M=None):
    """Return parallel component of F along n (unit under metric)."""
    # assume n is normalized in metric (n^T M n = 1) if M given
    c = _dot(F, n, M=M)
    return c * n

def _proj_perp(F, n, M=None):
    return F - _proj_parallel(F, n, M=M)

def _normalize(n, M=None, eps=1e-20):
    dn = _norm(n, M=M)
    if dn < eps:
        raise ValueError("Zero-length direction encountered during normalization.")
    return n / dn

def _remove_rigid_body_components(n, atoms: Atoms, M=None):
    """
    Remove global translation & rotation components from direction n.
    Weighted directions are projected in q=sqrt(M)x coordinates through the
    shared orthonormal rigid basis. The unweighted path retains its historical
    approximate Cartesian projector unchanged.
    """
    if M is not None:
        sqrt_M = np.sqrt(M)
        weighted = sqrt_M * n
        rigid = mass_weighted_rigid_basis(atoms)
        weighted -= rigid @ (rigid.T @ weighted)
        return weighted / sqrt_M

    # Preserve the historical unweighted projector exactly.
    X = to_numpy_f64(atoms.get_positions()).reshape(-1, 3)
    v = n.reshape(-1, 3).copy()
    t = v.mean(axis=0, keepdims=True)
    v -= t

    # rotation (approx): project out components proportional to r x omega, omega = basis unit vectors
    # center positions
    rc = X.mean(axis=0)
    r = X - rc
    # three axes basis
    axes = np.eye(3)
    for k in range(3):
        rot_mode = np.cross(r, axes[k])  # (N,3)
        # project each component of v onto rot_mode
        a = (v * rot_mode).sum() / ((rot_mode * rot_mode).sum() + 1e-20)
        v -= a * rot_mode
    return v.reshape(-1)

# =============================================================================
# ------------------------------ Dimer class ----------------------------------
# =============================================================================

class Dimer(JobABC):
    """
    Dimer saddle search with optional HVP (autograd) backend.

    hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]]
        If provided and DimerParams.use_hvp=True, returns H @ n (shape (3N,))
        Else: we use finite-difference via two force calls at R ± Δ n.
    """
    def __init__(self,
                 output: str,
                 atoms_init: Atoms,
                 paras: Optional[dict] = None,
                 hvp_fn: Optional[Callable[[Atoms, np.ndarray], np.ndarray]] = None):
        super().__init__(output)

        self.atoms = atoms_init
        if self.atoms.calc is None:
            raise ValueError("atoms_init must have a working calculator set (atoms.calc).")

        if getattr(self.atoms, "constraints", None):
            raise NotImplementedError(
                "constrained Dimer search is unsupported until a validated "
                "constraint-tangent projection is implemented"
            )

        # Initialize params from paras dict
        self.params = self._init_params(
            DimerParams,
            paras,
            ("dimer", "DIMER", "ts"),
            strict=True,
            context="Dimer",
            allowed_keys=self.TASK_ROUTING_PARAM_KEYS,
        )

        self.hvp_fn = hvp_fn if self.params.use_hvp else None
        if self.params.use_hvp and self.hvp_fn is None:
            has_analytic_hvp = bool(
                getattr(self.atoms.calc, "supports_hvp", False)
                and callable(getattr(self.atoms.calc, "get_hvp", None))
            )
            if not has_analytic_hvp:
                raise NotImplementedError(
                    "Dimer(use_hvp=True) requires a declared analytic HVP "
                    "calculator or an explicit hvp_fn"
                )
        # Validate the mass metric before constructing/evaluating a backend.
        self.M = _get_metric(self.atoms, self.params.use_mass_weight)
        use_calculator_analytic = self.params.use_hvp and self.hvp_fn is None
        self._evaluator = HVPEvaluator(
            self.atoms.calc,
            use_analytic=use_calculator_analytic,
            require_analytic=use_calculator_analytic,
        )
        self.status = None
        self.result = None

        # init direction n
        self.n = self._init_direction()

        # trust region bookkeeping
        self.alpha = float(self.params.step0)

    # ------------------------------ init n -----------------------------------

    def _init_direction(self) -> np.ndarray:
        N = len(self.atoms)
        D = 3 * N
        p = self.params

        if p.n_init.lower() == "given" and (p.n_given is not None):
            n = vec1d(p.n_given, D)
        elif p.n_init.lower() == "force":
            gradient = -vec1d(self.atoms.get_forces(), D)
            n = gradient if self.M is None else gradient / self.M
        else:  # random
            rng = np.random.default_rng()
            n = rng.normal(size=D)

        if p.remove_rigid:
            n = _remove_rigid_body_components(n, self.atoms, M=self.M)
        n = _normalize(n, M=self.M)
        return n

    # ----------------------- curvature & rotation helpers ---------------------

    def _evaluate_current(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(Hn, forces, energy)`` at the atoms' current coordinates."""
        n_flat = vec1d(n, 3 * len(self.atoms))
        if not np.all(np.isfinite(n_flat)):
            raise ValueError("Dimer direction must contain only finite values")

        if self.hvp_fn is None:
            Hn, forces, energy = self._evaluator.hn(
                self.atoms, n_flat, delta=float(self.params.delta)
            )
        else:
            Hn = vec1d(self.hvp_fn(self.atoms, n_flat), n_flat.size)
            energy, force_array = energy_forces_one(
                self.atoms.calc, self.atoms, force_consistent=True
            )
            forces = force_array.reshape(-1)

        Hn = vec1d(Hn, n_flat.size)
        forces = vec1d(forces, n_flat.size)
        energy = float(energy)
        if (
            not np.all(np.isfinite(Hn))
            or not np.all(np.isfinite(forces))
            or not np.isfinite(energy)
        ):
            raise FloatingPointError(
                "Dimer evaluator returned non-finite HVP, force, or energy values"
            )
        return Hn, forces, energy

    def _rotational_metrics(
        self, Hn: np.ndarray, n: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        action = Hn if self.M is None else Hn / self.M
        F_rot = action - _proj_parallel(action, n, M=self.M)
        residual = F_rot if self.M is None else np.sqrt(self.M) * F_rot
        max_frot = float(np.max(np.abs(residual)))
        rms_frot = float(math.sqrt(np.mean(residual * residual)))
        return F_rot, max_frot, rms_frot

    @staticmethod
    def _curvature(Hn: np.ndarray, n: np.ndarray) -> float:
        """Cartesian Rayleigh quotient for an M-normalized direction."""
        return float(np.dot(n, Hn))

    def _translation_force(
        self, forces: np.ndarray, n: np.ndarray, mode_flip: bool
    ) -> np.ndarray:
        """Return the Cartesian update vector from preconditioned forces."""
        preconditioned = forces if self.M is None else forces / self.M
        parallel = _proj_parallel(preconditioned, n, M=self.M)
        perpendicular = preconditioned - parallel
        return perpendicular - parallel if mode_flip else perpendicular

    def _rotate_minimize_kappa(self, n: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Do up to rot_max_iter steps of rotation to minimize kappa = n^T H n.
        Returns (n_new, max|F_rot|, rms(F_rot)).
        """
        p = self.params
        n_cur = n.copy()
        Hn, _, _ = self._evaluate_current(n_cur)
        F_rot, max_frot, rms_frot = self._rotational_metrics(Hn, n_cur)

        for _ in range(p.rot_max_iter):
            # convergence of rotation
            if (max_frot <= p.rot_f_max_th) and (rms_frot <= p.rot_f_rms_th):
                return n_cur, max_frot, rms_frot

            # gradient descent on kappa: n <- n - α * F_rot (and renormalize)
            n_next = n_cur - p.rot_alpha * F_rot
            if self.params.remove_rigid:
                n_next = _remove_rigid_body_components(n_next, self.atoms, M=self.M)
            n_next = _normalize(n_next, M=self.M)
            n_cur = n_next
            Hn, _, _ = self._evaluate_current(n_cur)
            F_rot, max_frot, rms_frot = self._rotational_metrics(Hn, n_cur)

        # after max rot steps return last
        return n_cur, max_frot, rms_frot

    # ------------------------------- main flow --------------------------------

    def atoms_to_xyz_block(self, atoms: Atoms) -> str:
        lines = []
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for idx, (s, (x, y, z)) in enumerate(zip(syms, pos)):
            lines.append(f"{idx:<4d}{s:>2s}{x:18.4f}{y:18.4f}{z:18.4f}")
        return "\n".join(lines) + "\n"

    def run(self):
        """Return converged atoms or raise with the non-candidate result."""
        result = self.run_result()
        if not result.geometry_converged:
            raise DimerConvergenceError(result)
        return result.atoms

    def run_result(self) -> DimerResult:
        """
        Main Dimer optimization loop with an explicitly selected HVP evaluator.
        - ``use_hvp=False`` always uses central force differences plus center E/F
        - ``use_hvp=True`` requires a declared analytic HVP or explicit callback
        - Post-step E/F/Hn, metrics, log coordinates, and output share one point
        - Logs curvature, rotational force, mode flip, 4 PRFO-style criteria, and trust region
        """
        p = self.params
        base, _ = os.path.splitext(self.output)
        ext = ".pdb" if self.atoms.info.get("pdb_template") else ".xyz"
        traj_file = base + "_dimer_traj" + ext
        candidate_file = base + "_dimer_ts_candidate" + ext
        curvature_file = base + "_dimer_failed_nonnegative_curvature" + ext
        failed_file = base + "_dimer_failed_max_iter" + ext
        artifact_suffixes = (
            "_dimer_traj",
            "_dimer_ts",
            "_dimer_ts_candidate",
            "_dimer_failed_nonnegative_curvature",
            "_dimer_failed_max_iter",
        )
        archive_artifacts(
            (
                base + suffix + artifact_ext
                for suffix in artifact_suffixes
                for artifact_ext in (".xyz", ".pdb")
            ),
            base + "_dimer_history",
        )

        # ------------------ init direction & step size ------------------
        n = self.n.copy()        # initial dimer orientation (assumed normalized & rigid-body removed if requested)
        alpha = float(self.alpha)

        # ------------------ initial same-point evaluation ------------------
        _, forces_np, E0 = self._evaluate_current(n)
        maxF0 = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
        rmsF0  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

        log_info([
            "\n---------------------------------------------------------------\n",
            "                       DIMER INITIAL STATE\n",
            "---------------------------------------------------------------\n",
            f"Energy (initial)                        ....  {E0: .8f} Eh\n",
            f"RMS(|F|)                                ....  {rmsF0: .6f} Eh/Angstrom\n",
            f"MAX(|F|)                                ....  {maxF0: .6f} Eh/Angstrom\n",
            "\nINITIAL COORDINATES (ANGSTROEM):\n",
            self.atoms_to_xyz_block(self.atoms),
        ], self.output)

        # ------------------ ensure PRFO-style thresholds exist ------------------
        if not hasattr(self.atoms, "f_max_th"):  self.atoms.f_max_th  = p.f_max_th
        if not hasattr(self.atoms, "f_rms_th"):  self.atoms.f_rms_th  = p.f_rms_th
        if not hasattr(self.atoms, "dp_max_th"): self.atoms.dp_max_th = 1.8e-3
        if not hasattr(self.atoms, "dp_rms_th"): self.atoms.dp_rms_th = 1.2e-3

        log_info([
            "\n----------------------------------------------------------------------\n",
            "                           Dimer Iterations                           \n",
            "----------------------------------------------------------------------\n"
        ], self.output)

        # ======================= main iteration loop =======================
        converged = False
        nonnegative_curvature = False
        completed_iterations = 0
        E = E0
        for it in range(1, p.max_iter + 1):

            # (1) rotation at the current point
            n, _, _ = self._rotate_minimize_kappa(n)

            # (2) translation-side evaluation at the pre-step point
            Hn, forces_np, _ = self._evaluate_current(n)

            # curvature kappa = n^T H n
            kappa = self._curvature(Hn, n)

            mode_flip = (kappa < p.kappa_to_flip)
            Ftrans = self._translation_force(forces_np, n, mode_flip)

            # (3) trust-region step
            step_vec = alpha * Ftrans
            step_norm_inf = float(np.max(np.abs(step_vec)))
            on_boundary = False
            max_allow = min(p.trust_radius, p.step_max)
            if step_norm_inf > max_allow:
                step_vec *= (max_allow / (step_norm_inf + 1e-20))
                on_boundary = True

            new_positions = self.atoms.get_positions() + step_vec.reshape(-1, 3)
            self.atoms.set_positions(new_positions)
            alpha = (max(0.5 * alpha, 0.1 * p.step0) if on_boundary
                    else min(1.2 * alpha, p.step_max))

            # (4) Re-evaluate the new state before any metric/log/write.
            Hn, forces_np, E = self._evaluate_current(n)
            kappa = self._curvature(Hn, n)
            _, max_frot, rms_frot = self._rotational_metrics(Hn, n)
            mode_flip = (kappa < p.kappa_to_flip)

            # (5) PRFO-style metrics
            max_dp = float(np.max(np.linalg.norm(step_vec.reshape(-1, 3), axis=1)))
            rms_dp = float(np.sqrt(np.mean(np.linalg.norm(step_vec.reshape(-1, 3), axis=1) ** 2)))
            max_f  = float(np.max(np.linalg.norm(forces_np.reshape(-1, 3), axis=1)))
            rms_f  = float(np.sqrt(np.mean(np.linalg.norm(forces_np.reshape(-1, 3), axis=1) ** 2)))

            self.atoms.max_f  = max_f
            self.atoms.rms_f  = rms_f
            self.atoms.max_dp = max_dp
            self.atoms.rms_dp = rms_dp

            # (6) report one iteration block
            coords_block = self.atoms_to_xyz_block(self.atoms)
            curvature_unit = (
                "Ha/(Angstrom^2 amu)"
                if self.M is not None else "Ha/Angstrom^2"
            )
            rotation_unit = curvature_unit
            rotation_label = (
                "mass-weighted rotation residual"
                if self.M is not None else "rotation residual"
            )
            info = [
                "\n----------------------------------------------------------------------\n",
                f"                             Iteration: {it:<3d}                              \n\n",
                "                             Coordinates                               \n",
                "----------------------------------------------------------------------\n",
                coords_block,
                "\n",
                f"Energy:                  {E: .6f} Convergence criteria  Is converged \n",
                f"Maximum Force:         {max_f:>12.6f} {self.atoms.f_max_th:>12.6f}                {'Yes' if max_f <= self.atoms.f_max_th else 'No'}\n",
                f"RMS Force:             {rms_f:>12.6f} {self.atoms.f_rms_th:>12.6f}                {'Yes' if rms_f <= self.atoms.f_rms_th else 'No'}\n",
                f"Maximum Displacement:  {max_dp:>12.6f} {self.atoms.dp_max_th:>12.6f}                {'Yes' if max_dp <= self.atoms.dp_max_th else 'No'}\n",
                f"RMS Displacement:      {rms_dp:>12.6f} {self.atoms.dp_rms_th:>12.6f}                {'Yes' if rms_dp <= self.atoms.dp_rms_th else 'No'}\n",
                f"\nCartesian trust radius (Angstrom): {max_allow: .6f}  "
                f"Cartesian step max component (Angstrom): {step_norm_inf: .6f}  "
                f"On boundary: {on_boundary}\n"
                f"Curvature (kappa):      {kappa:>12.6f} {curvature_unit}\n",
                f"Max {rotation_label}: {max_frot:>12.6f} {rotation_unit}\n",
                f"RMS {rotation_label}: {rms_frot:>12.6f} {rotation_unit}\n",
                f"Mode Flip:              {'Yes' if mode_flip else 'No'}\n\n",
            ]
            log_info(info, self.output)

            if p.save_traj:
                write_all_images_xyz(traj_file, self.atoms, energy=E, iteration=it)

            completed_iterations = it

            # (7) convergence: all 4 PRFO-style criteria
            geometry_converged = (
                max_f <= self.atoms.f_max_th
                and rms_f <= self.atoms.f_rms_th
                and max_dp <= self.atoms.dp_max_th
                and rms_dp <= self.atoms.dp_rms_th
            )
            if geometry_converged:
                if kappa < 0.0:
                    log_info(
                        [f"\nDimer geometry converged at iteration {it}.\n"],
                        self.output,
                    )
                    converged = True
                else:
                    log_info(
                        [
                            "\nDimer geometry thresholds were reached without "
                            "negative curvature; no TS candidate will be emitted.\n"
                        ],
                        self.output,
                    )
                    nonnegative_curvature = True
                break

        # ------------------ terminal diagnostic/candidate ------------------
        _, _, E_final = self._evaluate_current(n)
        if converged:
            self.status = DimerStatus.GEOMETRY_CONVERGED
            structure_file = candidate_file
        elif nonnegative_curvature:
            self.status = DimerStatus.FAILED_NONNEGATIVE_CURVATURE
            structure_file = curvature_file
        else:
            self.status = DimerStatus.FAILED_MAX_ITER
            structure_file = failed_file
        write_xyz(structure_file, [self.atoms], energies=[E_final])
        self.result = DimerResult(
            atoms=self.atoms,
            status=self.status,
            iterations=completed_iterations,
            structure_path=structure_file,
            energy=E_final,
        )

        terminal_label = (
            "GEOMETRY-CONVERGED DIMER CANDIDATE"
            if converged
            else "DIMER DIAGNOSTIC (NOT A TS CANDIDATE)"
        )
        terminal_info = [
            "\n---------------------------------------------------------------\n",
            f"{terminal_label:^63s}\n",
            "---------------------------------------------------------------\n",
            f"Energy (final)                           ....  {E_final: .8f} Eh\n",
            "\n-----------------------------------------\n",
            "  FINAL GEOMETRY (ANGSTROEM)\n",
            "-----------------------------------------\n",
            self.atoms_to_xyz_block(self.atoms),
        ]
        if p.save_traj:
            terminal_info.append(f"\nWrote Dimer trajectory to: {traj_file}\n")
        terminal_info.extend([
            f"Terminal status:           {self.status.value}\n",
            f"Wrote terminal structure:  {structure_file}\n"
        ])
        log_info(terminal_info, self.output)
        return self.result
