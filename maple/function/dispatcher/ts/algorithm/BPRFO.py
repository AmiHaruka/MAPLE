# -*- coding: utf-8 -*-
"""Runtime-disabled, device-native experimental batched RS-P-RFO."""

from __future__ import annotations

import math
import os
import numpy as np
import torch
from ase import Atoms

from .PRFO import (
    PRFOResult, PRFOStatus, _resolve_prfo_coordinate_space,
    _rigid_internal_complement, _finite_positive_real,
    append_xyz_trajectory, write_xyz,
)
from ._prfo_batch_math import prfo_step_batched

DTYPE = torch.float64
BIG = 1e8
INERTIA_THRESHOLD = 1e-6


class _StepSolverFailure(RuntimeError):
    """A physical P-RFO row could not produce a bounded finite step."""


class _ModeTrackingFailure(RuntimeError):
    """The current physical eigensystem could not be matched to a mode."""


def _regularize_signed(values: torch.Tensor, eps: float) -> torch.Tensor:
    eps_tensor = torch.full_like(values, float(eps))
    replacement = torch.where(values < 0.0, -eps_tensor, eps_tensor)
    return torch.where(values.abs() < float(eps), replacement, values)


def _concrete_device(device) -> torch.device:
    """Resolve CUDA's implicit current index before comparing tensor devices."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _ptr_from_atoms(at_list, device):
    ptr = [0]
    for atoms in at_list:
        ptr.append(ptr[-1] + len(atoms))
    return torch.tensor(ptr, dtype=torch.long, device=device)


def _masses_flat(at_list, nmax, device):
    mass = torch.ones((len(at_list), nmax), dtype=DTYPE, device=device)
    for index, atoms in enumerate(at_list):
        values = np.asarray(atoms.get_masses(), dtype=np.float64)
        mass[index, : 3 * len(atoms)] = torch.as_tensor(
            np.repeat(values, 3), dtype=DTYPE, device=device
        )
    return mass


def _get_coord_gpu(calc) -> torch.Tensor:
    if hasattr(calc, "coord"):
        return calc.coord
    if hasattr(calc, "coord32"):
        return calc.coord32
    raise AttributeError("calculator has no coord buffer")


def _atoms_snapshot(atoms: Atoms) -> Atoms:
    """Copy mutable geometry metadata without copying calculator/model weights."""
    def copy_metadata(value):
        if isinstance(value, dict):
            return {key: copy_metadata(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_metadata(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_metadata(item) for item in value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    calculator = atoms.calc
    snapshot = atoms.copy()
    snapshot.info = copy_metadata(atoms.info)
    snapshot.calc = calculator
    return snapshot


class BatchPRFO:
    """Batched RS-P-RFO prototype.  The public entry point stays disabled."""

    _STATUS_SUFFIX = {
        PRFOStatus.GEOMETRY_CONVERGED: "_prfo_ts_candidate",
        PRFOStatus.FAILED_NONFINITE: "_prfo_nonfinite",
        PRFOStatus.FAILED_BACKEND: "_prfo_backend_failure",
        PRFOStatus.FAILED_HESSIAN: "_prfo_hessian_failure",
        PRFOStatus.FAILED_WRONG_INERTIA: "_prfo_wrong_inertia",
        PRFOStatus.FAILED_STAGNATION: "_prfo_stagnation",
        PRFOStatus.FAILED_STEP_SOLVER: "_prfo_step_failure",
        PRFOStatus.FAILED_MODE_TRACKING: "_prfo_mode_failure",
        PRFOStatus.FAILED_MAXITER: "_prfo_unconverged",
    }

    def __init__(
        self,
        output: str,
        trust_init: float = 0.20,
        trust_min: float = 1e-3,
        trust_max: float = 1.00,
        eta_reject: float = 0.0,
        eta_shrink: float = 0.5,
        eta_expand: float = 0.75,
        max_inner_attempts: int = 10,
        max_outer_iter: int = 256,
        device: str = "cuda",
        recalc: int = 4,
        hessian_update: str = "bofill",
        rigid_symmetry: str = "auto",
    ):
        numeric = {
            "trust_min": trust_min, "trust_init": trust_init,
            "trust_max": trust_max, "eta_reject": eta_reject,
            "eta_shrink": eta_shrink, "eta_expand": eta_expand,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not (0.0 < trust_min <= trust_init <= trust_max):
            raise ValueError("trust radii must satisfy 0 < trust_min <= trust_init <= trust_max")
        if not 0.0 <= eta_expand <= 1.0:
            raise ValueError("eta_expand must be within [0, 1]")
        if not (0.0 <= eta_reject <= eta_shrink <= eta_expand):
            raise ValueError("eta values must satisfy 0 <= reject <= shrink <= expand <= 1")
        for name, value, allow_zero in (
            ("max_inner_attempts", max_inner_attempts, False),
            ("max_outer_iter", max_outer_iter, True),
            ("recalc", recalc, False),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if allow_zero else 1):
                raise ValueError(f"{name} is outside its allowed range")
        update = str(hessian_update).lower()
        if update not in {"bofill", "bfgs"}:
            raise ValueError("hessian_update must be 'bofill' or 'bfgs'")
        symmetry = str(rigid_symmetry).lower()
        if symmetry not in {"auto", "free_molecule", "cartesian_external"}:
            raise ValueError(
                "rigid_symmetry must be 'auto', 'free_molecule', or 'cartesian_external'"
            )

        self.trust_init, self.trust_min, self.trust_max = map(
            float, (trust_init, trust_min, trust_max)
        )
        self.eta_reject, self.eta_shrink, self.eta_expand = map(
            float, (eta_reject, eta_shrink, eta_expand)
        )
        self.max_inner_attempts = int(max_inner_attempts)
        self.max_outer_iter = int(max_outer_iter)
        self.device = _concrete_device(device)
        self.recalc = int(recalc)
        self.hessian_update = update
        self.rigid_symmetry = symmetry
        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        self.log_fp = None
        self.xyz_paths: list[str] = []
        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None
        self._ptr = self._L_vec = self._arange_n = self._real_mask = self._D = None
        self._P_vec = self._physical_bases = self._spaces = None
        self._nmax = self._B = 0
        self._orig_index = None
        self._H_work = None
        self._trial_energy = self._trial_forces = self._trial_hessian = None

    def run(self, mols) -> None:
        raise NotImplementedError(
            "BatchPRFO is experimental and runtime-disabled. Use the single-"
            "structure PRFO candidate workflow until batched Hessian/mode and "
            "TS-verification tests are available."
        )

    def _run_experimental(self, mols) -> tuple[PRFOResult, ...]:
        original_atoms = list(mols.multiatoms)
        results: list[PRFOResult | None] = [None] * len(original_atoms)
        valid_atoms, valid_orig, valid_spaces = [], [], []
        calc = mols.calc
        for index, atoms in enumerate(original_atoms):
            status, detail = self._validate_member(atoms)
            if status is None:
                try:
                    source_capability = getattr(atoms.calc, "rigid_body_invariant", None)
                    if (source_capability is not None
                            and not isinstance(source_capability, (bool, np.bool_))):
                        raise TypeError("Member calculator rigid_body_invariant must be boolean")
                    space = _resolve_prfo_coordinate_space(
                        atoms, self.rigid_symmetry, calculator=calc
                    )
                    if (source_capability is not None
                            and bool(source_capability) != (space == "free_molecule")):
                        raise ValueError(
                            "Member calculator and selected batch symmetry make "
                            "conflicting rigid-body assertions"
                        )
                    if space == "free_molecule" and len(atoms) < 2:
                        raise NotImplementedError(
                            "Free-molecule PRFO requires an internal degree of freedom"
                        )
                except (TypeError, ValueError, NotImplementedError) as exc:
                    status = PRFOStatus.FAILED_HESSIAN
                    detail = f"Physical symmetry contract failed: {exc}"
                else:
                    valid_atoms.append(atoms)
                    valid_orig.append(index)
                    valid_spaces.append(space)
            if status is not None:
                results[index] = PRFOResult(
                    _atoms_snapshot(atoms), status, 0, "", None, detail
                )
        if not valid_atoms:
            return tuple(results)  # type: ignore[arg-type]

        atoms_list = valid_atoms
        self._spaces = valid_spaces
        self._H_work = None
        self.tracked_mode_vec_mw = self.tracked_mode_idx = None
        self._orig_index = torch.tensor(valid_orig, dtype=torch.long, device=self.device)
        confirmed = [_atoms_snapshot(atoms) for atoms in atoms_list]
        iterations = torch.zeros(len(atoms_list), dtype=torch.long, device=self.device)
        trust_r = torch.full(
            (len(atoms_list),), self.trust_init, dtype=DTYPE, device=self.device
        )
        try:
            calc.prepare(atoms_list)
            self._nmax = int(getattr(calc, "nmax_dof", 0))
            self._validate_prepared(calc, atoms_list, self._nmax)
            self._arange_n = torch.arange(self._nmax, device=self.device)
            self._rebuild_topology(atoms_list)
            self._init_xyz_paths(len(original_atoms))
            self._open_log()
            self._w("# RS-PRFO batched TS search start (experimental)\n")
            self._append_current_trajectories(atoms_list, None, iterations)

            while atoms_list:
                try:
                    E, F = self._validate_ef(*calc.get_ef_gpu(), len(atoms_list))
                except Exception as exc:
                    status = self._exception_status(exc)
                    self._fill_active_results(
                        results, confirmed, iterations, status,
                        f"Current E/F evaluation failed: {type(exc).__name__}: {exc}",
                    )
                    break
                g_cart = -F * self._real_mask.to(DTYPE)
                force_done = self._force_converged(g_cart, atoms_list)
                need_exact = (
                    self._H_work is None or bool(force_done.any())
                    or bool((iterations >= self.max_outer_iter).any())
                    or bool(((iterations % self.recalc) == 0).any())
                )
                if need_exact:
                    try:
                        E, F, H, _ = calc.get_efh_gpu()
                        E, F, H = self._validate_efh(E, F, H, len(atoms_list))
                    except Exception as exc:
                        self._fill_active_results(
                            results, confirmed, iterations,
                            self._exception_status(exc, hessian=True),
                            f"Current E/F/H evaluation failed: {type(exc).__name__}: {exc}",
                        )
                        break
                    H_cart, g_cart = self._build_cartesian_hg(F, H, self._real_mask)
                    self._H_work = H_cart.clone()
                    force_done = self._force_converged(g_cart, atoms_list)
                else:
                    H_cart = self._H_work
                try:
                    self._set_physical_bases(atoms_list)
                    H_mw, g_mw = self._mass_weight_hg(H_cart, g_cart)
                    w, V, gp = self._eigh_and_track_modes(H_mw, g_mw)
                except Exception as exc:
                    tracking_failed = isinstance(exc, _ModeTrackingFailure)
                    status = (PRFOStatus.FAILED_MODE_TRACKING if tracking_failed
                              else PRFOStatus.FAILED_HESSIAN)
                    operation = "Mode tracking" if tracking_failed else "Physical Hessian eigensystem"
                    self._fill_active_results(
                        results, confirmed, iterations,
                        status,
                        f"{operation} failed: {type(exc).__name__}: {exc}",
                    )
                    break
                physical_mask = self._arange_n[None, :] < self._P_vec[:, None]
                negative = ((w < -INERTIA_THRESHOLD) & physical_mask).sum(1)
                significant = ((w.abs() > INERTIA_THRESHOLD) & physical_mask).sum(1)
                no_mode = significant == 0
                exhausted = iterations >= self.max_outer_iter
                terminal = force_done | exhausted | no_mode
                for local in terminal.nonzero(as_tuple=False).flatten().tolist():
                    if bool(force_done[local]):
                        status = (PRFOStatus.GEOMETRY_CONVERGED
                                  if int(negative[local]) == 1
                                  else PRFOStatus.FAILED_WRONG_INERTIA)
                        detail = ("Forces and first-order inertia converged."
                                  if status is PRFOStatus.GEOMETRY_CONVERGED else
                                  f"Geometry converged with {int(negative[local])} significant negative modes; expected 1.")
                    elif bool(exhausted[local]):
                        status = PRFOStatus.FAILED_MAXITER
                        detail = "BatchPRFO exhausted its accepted-step budget."
                    else:
                        status = PRFOStatus.FAILED_MODE_TRACKING
                        detail = "No significant physical Hessian mode is available for tracking."
                    orig = int(self._orig_index[local])
                    results[orig] = self._terminal_result(
                        original_index=orig, atoms=confirmed[local], status=status,
                        iteration=int(iterations[local]), energy=float(E[local]),
                        negative_modes=int(negative[local]), detail=detail,
                    )

                if bool(terminal.any()):
                    keep = (~terminal).nonzero(as_tuple=False).flatten()
                    if keep.numel() == 0:
                        break
                    atoms_list, confirmed, iterations, trust_r = self._shrink(
                        calc, atoms_list, confirmed, iterations, trust_r, keep
                    )
                    continue

                try:
                    trust_r, _, _, accepted = self._inner_rs_prfo_loop(
                        calc=calc, w=w, V=V,
                        gp=gp, H=H_cart, g_cart=g_cart, trust_r=trust_r,
                        last_step=torch.zeros_like(g_cart),
                        real_mask=self._real_mask, E_old=E,
                    )
                except Exception as exc:
                    if not bool(getattr(calc, "_prepared", True)):
                        detail = f"Trial failed and coordinate restore failed: {exc}"
                    else:
                        detail = f"Trial/commit backend failed: {type(exc).__name__}: {exc}"
                    status = (PRFOStatus.FAILED_STEP_SOLVER
                              if isinstance(exc, _StepSolverFailure)
                              else self._exception_status(exc))
                    self._fill_active_results(
                        results, confirmed, iterations, status, detail
                    )
                    break

                if bool(accepted.any()):
                    self._sync_atoms_from_calc(calc, atoms_list, accepted)
                    for local in accepted.nonzero(as_tuple=False).flatten().tolist():
                        confirmed[local] = _atoms_snapshot(atoms_list[local])
                    iterations = iterations + accepted.to(iterations.dtype)
                if not bool(accepted.all()):
                    failed = ~accepted
                    for local in failed.nonzero(as_tuple=False).flatten().tolist():
                        orig = int(self._orig_index[local])
                        results[orig] = self._terminal_result(
                            original_index=orig, atoms=confirmed[local],
                            status=PRFOStatus.FAILED_STAGNATION,
                            iteration=int(iterations[local]), energy=float(E[local]),
                            negative_modes=int(negative[local]),
                            detail=f"No acceptable step after {self.max_inner_attempts} trust-radius attempts.",
                        )
                    keep = accepted.nonzero(as_tuple=False).flatten()
                    if keep.numel() == 0:
                        break
                    atoms_list, confirmed, iterations, trust_r = self._shrink(
                        calc, atoms_list, confirmed, iterations, trust_r, keep
                    )
                    self._trial_energy = self._trial_energy[keep]
                    self._trial_forces = self._trial_forces[keep]
                    self._trial_hessian = self._trial_hessian[keep]

                trial_forces = self._trial_forces
                if trial_forces is None:
                    raise RuntimeError("accepted trial did not retain validated forces")
                self._H_work = self._trial_hessian.clone()
                self._append_current_trajectories(atoms_list, self._trial_energy, iterations)
        except Exception as exc:
            self._fill_active_results(
                results, confirmed, iterations, PRFOStatus.FAILED_BACKEND,
                f"Batch preparation/protocol failed: {type(exc).__name__}: {exc}",
            )
        finally:
            self._close_log()

        if any(result is None for result in results):
            self._fill_active_results(
                results, confirmed, iterations, PRFOStatus.FAILED_BACKEND,
                "BatchPRFO terminated without a complete member record.",
            )
        return tuple(results)  # type: ignore[arg-type]

    @staticmethod
    def _validate_member(atoms):
        if len(atoms) == 0:
            return PRFOStatus.FAILED_HESSIAN, "Empty structures are unsupported."
        if not np.all(np.isfinite(atoms.get_positions())):
            return PRFOStatus.FAILED_NONFINITE, "Coordinates must be finite."
        if atoms.constraints:
            return PRFOStatus.FAILED_HESSIAN, "constrained structures are unsupported."
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            return PRFOStatus.FAILED_HESSIAN, "Atomic masses must be finite and positive."
        for name, default in (("f_max_th", 2e-3), ("f_rms_th", 1e-3)):
            try:
                _finite_positive_real(getattr(atoms, name, default))
            except ValueError:
                return (
                    PRFOStatus.FAILED_HESSIAN,
                    f"{name} threshold must be a finite positive real scalar.",
                )
        return None, ""

    def _validate_prepared(self, calc, atoms_list, width):
        if not bool(getattr(calc, "_prepared", False)):
            raise RuntimeError("prepared flag is false")
        if _concrete_device(getattr(calc, "device", self.device)) != self.device:
            raise RuntimeError("prepared device does not match optimizer device")
        expected_ptr = _ptr_from_atoms(atoms_list, self.device)
        ptr = getattr(calc, "_ptr", None)
        if (not isinstance(ptr, torch.Tensor) or ptr.device != self.device
                or ptr.dtype != torch.long or not torch.equal(ptr, expected_ptr)):
            raise RuntimeError("prepared pointer is invalid")
        numbers = getattr(calc, "numbers", None)
        expected_numbers = torch.as_tensor(
            np.concatenate([at.numbers for at in atoms_list]),
            dtype=torch.int32, device=self.device,
        )
        if (not isinstance(numbers, torch.Tensor) or numbers.device != self.device
                or numbers.dtype not in (torch.int32, torch.int64)
                or not torch.equal(numbers.to(torch.int32), expected_numbers)):
            raise RuntimeError("prepared numbers/order are invalid")
        coord = _get_coord_gpu(calc)
        expected_coord = torch.as_tensor(
            np.concatenate([at.positions for at in atoms_list]),
            dtype=coord.dtype, device=coord.device,
        )
        if coord.device != self.device:
            raise RuntimeError("prepared coordinate device is invalid")
        if coord.shape != expected_coord.shape or not torch.equal(coord, expected_coord):
            raise RuntimeError("prepared coordinates do not match inputs")
        required = max(3 * len(at) for at in atoms_list)
        if width < required or width % 3:
            raise RuntimeError("prepared fixed width is invalid")

    def _validate_ef(self, E, F, batch):
        if not isinstance(E, torch.Tensor) or E.shape != (batch,):
            raise RuntimeError("energy shape is invalid")
        expected_width = self._nmax or int(F.shape[1])
        if not isinstance(F, torch.Tensor) or F.ndim != 2 or F.shape != (batch, expected_width):
            raise RuntimeError("forces shape is invalid")
        if E.device != self.device or F.device != self.device:
            raise RuntimeError("E/F device is invalid")
        E, F = E.to(DTYPE), F.to(DTYPE)
        if not bool(torch.isfinite(E).all()) or not bool(torch.isfinite(F).all()):
            raise FloatingPointError("energy or forces are non-finite")
        return E, F

    def _validate_efh(self, E, F, H, batch):
        E, F = self._validate_ef(E, F, batch)
        if not isinstance(H, torch.Tensor) or H.shape != (batch, self._nmax, self._nmax):
            raise ValueError("hessian shape is invalid")
        if H.device != self.device:
            raise ValueError("hessian device is invalid")
        H = H.to(DTYPE)
        if not bool(torch.isfinite(H).all()):
            raise FloatingPointError("hessian finite check failed")
        return E, F, 0.5 * (H + H.transpose(-1, -2))

    @staticmethod
    def _exception_status(exc, hessian=False):
        if isinstance(exc, FloatingPointError):
            return PRFOStatus.FAILED_NONFINITE
        return PRFOStatus.FAILED_HESSIAN if hessian and isinstance(exc, ValueError) else PRFOStatus.FAILED_BACKEND

    def _rebuild_topology(self, atoms_list):
        self._B = len(atoms_list)
        self._ptr = _ptr_from_atoms(atoms_list, self.device)
        self._L_vec = torch.tensor(
            [3 * len(at) for at in atoms_list], dtype=torch.long, device=self.device
        )
        self._real_mask = self._arange_n[None, :] < self._L_vec[:, None]
        masses = _masses_flat(atoms_list, self._nmax, self.device)
        self._D = 1.0 / torch.sqrt(masses)
        self._P_vec = self._physical_bases = None

    def _set_physical_bases(self, atoms_list):
        """Map current physical coordinates into the full mass-weighted DOFs.

        The free-molecule complement changes with geometry; rebuilding it at
        each confirmed point keeps step, mode tracking, and inertia coherent.
        External-field members retain their full Cartesian mass-weighted space.
        """
        lengths = []
        bases = []
        for atoms, space in zip(atoms_list, self._spaces, strict=True):
            if space == "free_molecule":
                basis = torch.as_tensor(
                    _rigid_internal_complement(atoms), dtype=DTYPE, device=self.device
                )
                lengths.append(basis.shape[1])
                bases.append(basis)
            else:
                lengths.append(3 * len(atoms))
                bases.append(None)
        self._P_vec = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self._physical_bases = bases

    def _physical_groups(self):
        """Group by both Cartesian and internal widths for native Torch kernels."""
        physical = self._P_vec if self._P_vec is not None else self._L_vec
        dimensions = torch.stack((self._L_vec, physical), dim=1)
        for pair in torch.unique(dimensions, dim=0):
            length, width = map(int, pair)
            members = (dimensions == pair).all(dim=1).nonzero(as_tuple=False).flatten()
            yield length, width, members

    def _group_basis(self, length, width, members):
        if self._physical_bases is None or width == length:
            return None
        return torch.stack([self._physical_bases[int(member)] for member in members])

    def _sync_atoms_from_calc(self, calc, atoms_list, accepted):
        positions = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for index, atoms in enumerate(atoms_list):
            if bool(accepted[index]):
                atoms.positions[:] = positions[ptr[index]:ptr[index + 1]]

    def _build_cartesian_hg(self, F_raw, H_raw, real_mask):
        mask = real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)
        return H_raw * mask.to(DTYPE), -F_raw * real_mask.to(DTYPE)

    def _force_converged(self, g_cart, atoms_list):
        fmax = g_cart.abs().amax(dim=1)
        frms = torch.sqrt(g_cart.square().sum(dim=1) / self._L_vec.to(DTYPE))
        return torch.tensor(
            [
                fmax[i] <= _finite_positive_real(getattr(atoms, "f_max_th", 2e-3))
                and frms[i] <= _finite_positive_real(getattr(atoms, "f_rms_th", 1e-3))
                for i, atoms in enumerate(atoms_list)
            ], dtype=torch.bool, device=self.device,
        )

    def _mass_weight_hg(self, H, g_cart):
        g_mw = self._D * g_cart
        H_mw = self._D.unsqueeze(-1) * H * self._D.unsqueeze(-2)
        return H_mw, g_mw

    def _eigh_and_track_modes(self, H_mw, g_mw):
        batch, width = g_mw.shape
        w = torch.full((batch, width), BIG, dtype=DTYPE, device=self.device)
        V = torch.eye(width, dtype=DTYPE, device=self.device).expand(batch, -1, -1).clone()
        gp = torch.zeros_like(g_mw)
        old_modes = self.tracked_mode_vec_mw
        tracked = torch.empty(batch, dtype=torch.long, device=self.device)
        modes = torch.zeros((batch, width), dtype=DTYPE, device=self.device)
        for length, width_physical, members in self._physical_groups():
            blocks = H_mw[members, :length, :length]
            basis = self._group_basis(length, width_physical, members)
            gradients = g_mw[members, :length]
            if basis is not None:
                blocks = torch.bmm(basis.transpose(1, 2), torch.bmm(blocks, basis))
                gradients = torch.bmm(
                    basis.transpose(1, 2), gradients.unsqueeze(-1)
                ).squeeze(-1)
            blocks = 0.5 * (blocks + blocks.transpose(1, 2))
            wb, Vb = torch.linalg.eigh(blocks)
            gpb = torch.bmm(
                Vb.transpose(1, 2), gradients.unsqueeze(-1)
            ).squeeze(-1)
            w[members, :width_physical] = wb
            V[members, :width_physical, :width_physical] = Vb
            gp[members, :width_physical] = gpb
            eligible = wb.abs() > INERTIA_THRESHOLD
            mode_vectors = Vb if basis is None else torch.bmm(basis, Vb)
            try:
                if old_modes is None:
                    ranked = torch.where(eligible, wb, torch.full_like(wb, float("inf")))
                    chosen = torch.argmin(ranked, dim=1)
                    signs = torch.ones_like(chosen, dtype=DTYPE)
                else:
                    overlap = torch.bmm(
                        mode_vectors.transpose(1, 2),
                        old_modes[members, :length].unsqueeze(-1),
                    ).squeeze(-1)
                    ranked = torch.where(
                        eligible, overlap.abs(), torch.full_like(overlap, -1.0)
                    )
                    chosen = torch.argmax(ranked, dim=1)
                    signs = torch.sign(overlap.gather(1, chosen[:, None]).squeeze(1))
                    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
                local_rows = torch.arange(members.numel(), device=self.device)
                tracked[members] = chosen
                modes[members, :length] = (
                    mode_vectors[local_rows, :, chosen] * signs[:, None]
                )
            except Exception as exc:
                raise _ModeTrackingFailure(str(exc)) from exc
        self.tracked_mode_idx = tracked
        self.tracked_mode_vec_mw = modes
        return w, V, gp

    def _inner_rs_prfo_loop(self, calc, w, V, gp, H, g_cart,
                            trust_r, last_step, real_mask, E_old):
        batch = gp.shape[0]
        accepted = torch.zeros(batch, dtype=torch.bool, device=self.device)
        last_step = torch.zeros_like(last_step)
        last_rho = torch.full((batch,), float("nan"), dtype=DTYPE, device=self.device)
        cached_E = torch.full_like(E_old, float("nan"))
        cached_F = torch.zeros_like(g_cart)
        cached_H = H.clone()
        for _ in range(self.max_inner_attempts):
            pending = ~accepted
            if not bool(pending.any()):
                break
            step_mw = torch.zeros_like(gp)
            solved = torch.zeros(batch, dtype=torch.bool, device=self.device)
            if self._L_vec is None:
                self._L_vec = torch.full(
                    (batch,), gp.shape[1], dtype=torch.long, device=self.device
                )
            for length, width_physical, members in self._physical_groups():
                group_step, group_ok = prfo_step_batched(
                    w[members, :width_physical],
                    V[members, :width_physical, :width_physical],
                    gp[members, :width_physical], self.tracked_mode_idx[members],
                    trust_r[members],
                )
                basis = self._group_basis(length, width_physical, members)
                if basis is not None:
                    group_step = torch.bmm(
                        basis, group_step.unsqueeze(-1)
                    ).squeeze(-1)
                step_mw[members, :length] = group_step
                solved[members] = group_ok
            if not bool((solved | ~pending).all()):
                raise _StepSolverFailure("P-RFO step solver failed for an active member")
            norm_mw = torch.linalg.norm(step_mw, dim=1)
            s_cart = self._D * step_mw * real_mask.to(DTYPE)
            s_try = torch.zeros_like(s_cart)
            s_try[pending] = s_cart[pending]

            trial_error = None
            restore_error = None
            E_new = F_new = None
            trial_base = _get_coord_gpu(calc).detach().clone()
            calc.backup_coords()
            try:
                calc.step_cart_(s_try)
                E_new, F_new = calc.get_ef_gpu()
                E_new, F_new = self._validate_ef(E_new, F_new, batch)
            except Exception as exc:
                trial_error = exc
            finally:
                try:
                    calc.restore_coords()
                    if not torch.equal(_get_coord_gpu(calc), trial_base):
                        calc._prepared = False
                        raise RuntimeError("trial coordinate restore did not recover the base point")
                except Exception as exc:
                    restore_error = exc
                    calc._prepared = False
            if trial_error is not None or restore_error is not None:
                pieces = []
                if trial_error is not None:
                    pieces.append(f"{type(trial_error).__name__}: {trial_error}")
                if restore_error is not None:
                    pieces.append(f"restore {type(restore_error).__name__}: {restore_error}")
                raise RuntimeError("; ".join(pieces))

            Hs = torch.einsum("bij,bj->bi", H, s_try)
            predicted = (g_cart * s_try).sum(1) + 0.5 * (s_try * Hs).sum(1)
            actual = E_new - E_old
            rho = torch.full_like(predicted, float("nan"))
            valid = pending & torch.isfinite(predicted) & torch.isfinite(actual) & (predicted.abs() > 1e-16)
            rho[valid] = actual[valid] / predicted[valid]
            quality = 1.0 - (rho - 1.0).abs()
            boundary = (norm_mw - trust_r).abs() <= 1e-6 * torch.clamp(trust_r, min=1.0)
            last_rho = torch.where(pending, rho, last_rho)
            accept = pending & torch.isfinite(quality) & (quality > self.eta_reject)
            reject = pending & ~accept
            shrink = pending & (quality < self.eta_shrink)
            shrink_radius = torch.maximum(
                torch.full_like(trust_r, self.trust_min),
                0.5 * torch.minimum(trust_r, norm_mw),
            )
            trust_r = torch.where(shrink | reject, shrink_radius, trust_r)
            grow = accept & (quality >= self.eta_expand) & boundary
            trust_r = torch.where(
                grow, torch.minimum(torch.full_like(trust_r, self.trust_max), math.sqrt(2.0) * trust_r), trust_r
            )
            if bool(accept.any()):
                accepted |= accept
                last_step[accept] = s_cart[accept]
                cached_E[accept] = E_new[accept]
                cached_F[accept] = F_new[accept]
        if bool(accepted.any()):
            update_fn = (self._bfgs_update_batched if self.hessian_update == "bfgs"
                         else self._bofill_update_batched)
            candidate_H = update_fn(
                H, last_step, g_cart, -cached_F * real_mask.to(DTYPE),
                real_mask, accepted,
            )
            if candidate_H.shape != H.shape or not bool(torch.isfinite(candidate_H).all()):
                raise FloatingPointError("Hessian update is invalid")
            cached_H[accepted] = candidate_H[accepted]
            base_coord = _get_coord_gpu(calc).detach().clone()
            expected_coord = base_coord.clone()
            ptr = self._ptr if self._ptr is not None else getattr(calc, "_ptr")
            for member in accepted.nonzero(as_tuple=False).flatten().tolist():
                start, stop = int(ptr[member]), int(ptr[member + 1])
                expected_coord[start:stop] += last_step[
                    member, : 3 * (stop - start)
                ].reshape(-1, 3).to(expected_coord)
            calc.backup_coords()
            try:
                calc.step_cart_(last_step)
                if not torch.equal(_get_coord_gpu(calc), expected_coord):
                    raise RuntimeError("accepted commit coordinates do not match staged trial")
                calc._discard_coord_backup()
            except Exception as exc:
                try:
                    calc.restore_coords()
                    if not torch.equal(_get_coord_gpu(calc), base_coord):
                        calc._prepared = False
                        raise RuntimeError("accepted commit restore did not recover the base point")
                except Exception as restore_exc:
                    calc._prepared = False
                    raise RuntimeError(f"{exc}; restore {restore_exc}") from exc
                raise
        self._trial_energy, self._trial_forces, self._trial_hessian = cached_E, cached_F, cached_H
        return trust_r, last_step, last_rho, accepted

    def _shrink(self, calc, atoms_list, confirmed, iterations, trust_r, keep):
        indices = keep.cpu().tolist()
        survivors = [atoms_list[i] for i in indices]
        survivor_snapshots = [confirmed[i] for i in indices]
        survivor_iterations = iterations[keep]
        survivor_trust = trust_r[keep]
        survivor_orig_index = self._orig_index[keep]
        survivor_modes = (
            None if self.tracked_mode_idx is None else self.tracked_mode_idx[keep]
        )
        survivor_vectors = (
            None if self.tracked_mode_vec_mw is None else self.tracked_mode_vec_mw[keep]
        )
        survivor_hessian = None if self._H_work is None else self._H_work[keep]
        survivor_spaces = [self._spaces[i] for i in indices]

        try:
            calc.prepare(survivors, fixed_nmax=self._nmax)
            self._validate_prepared(calc, survivors, self._nmax)
            self._rebuild_topology(survivors)
        except Exception:
            calc._prepared = False
            raise

        self.tracked_mode_idx = survivor_modes
        self.tracked_mode_vec_mw = survivor_vectors
        self._H_work = survivor_hessian
        self._orig_index = survivor_orig_index
        self._spaces = survivor_spaces
        return survivors, survivor_snapshots, survivor_iterations, survivor_trust

    def _fill_active_results(self, results, confirmed, iterations, status, detail):
        if self._orig_index is None:
            return
        if not (len(confirmed) == len(iterations) == len(self._orig_index)):
            raise RuntimeError("active result state is misaligned")
        for local, original in enumerate(self._orig_index.detach().cpu().tolist()):
            if results[original] is None:
                results[original] = self._terminal_result(
                    original_index=original, atoms=confirmed[local], status=status,
                    iteration=int(iterations[local]),
                    energy=None, negative_modes=None, detail=detail,
                )

    def _terminal_result(self, original_index, atoms, status, iteration,
                         energy, negative_modes, detail):
        base, _ = os.path.splitext(self.output)
        ext = ".pdb" if atoms.info.get("pdb_template") else ".xyz"
        path = f"{base}_batch{original_index + 1:04d}{self._STATUS_SUFFIX[status]}{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_xyz(path, atoms, energy=energy, iteration=iteration)
        return PRFOResult(_atoms_snapshot(atoms), status, iteration, path, negative_modes, detail)

    def _init_xyz_paths(self, count):
        base, _ = os.path.splitext(self.output)
        self.xyz_paths = [f"{base}_batch{i + 1:04d}_prfo_traj.xyz" for i in range(count)]

    def _append_current_trajectories(self, atoms_list, energies, iterations):
        for local, atoms in enumerate(atoms_list):
            original = int(self._orig_index[local])
            base, _ = os.path.splitext(self.output)
            ext = ".pdb" if atoms.info.get("pdb_template") else ".xyz"
            path = f"{base}_batch{original + 1:04d}_prfo_traj{ext}"
            self.xyz_paths[original] = path
            append_xyz_trajectory(
                path, atoms,
                energy=None if energies is None else float(energies[local]),
                iteration=int(iterations[local]),
            )

    @staticmethod
    @torch.no_grad()
    def _bfgs_update_batched(H, s_cart, g_prev, g_new, real_mask,
                             step_accepted=None, step_tol=1e-8,
                             grad_tol=1e-8, curvature_tol=1e-12):
        result = H.clone()
        for i in range(H.shape[0]):
            if step_accepted is not None and not bool(step_accepted[i]):
                continue
            mask = real_mask[i]
            s, y = s_cart[i] * mask, (g_new[i] - g_prev[i]) * mask
            if float(torch.linalg.norm(s)) <= step_tol or float(torch.linalg.norm(y)) <= grad_tol:
                continue
            ys, Hs = torch.dot(y, s), H[i] @ s
            sHs = torch.dot(s, Hs)
            if float(ys) <= curvature_tol or float(sHs) <= curvature_tol:
                continue
            updated = H[i] + torch.outer(y, y) / ys - torch.outer(Hs, Hs) / sHs
            result[i] = 0.5 * (updated + updated.T)
        return result

    @staticmethod
    @torch.no_grad()
    def _bofill_update_batched(H, s_cart, g_prev, g_new, real_mask,
                               step_accepted=None, step_tol=1e-8,
                               grad_tol=1e-8, sr1_tol=1e-8):
        result = H.clone()
        for i in range(H.shape[0]):
            if step_accepted is not None and not bool(step_accepted[i]):
                continue
            mask = real_mask[i]
            s, y = s_cart[i] * mask, (g_new[i] - g_prev[i]) * mask
            s2, y2 = torch.dot(s, s), torch.dot(y, y)
            if float(s2) <= step_tol**2 or float(y2) <= grad_tol**2:
                continue
            z = y - H[i] @ s
            z2, sz = torch.dot(z, z), torch.dot(s, z)
            psb = (torch.outer(z, s) + torch.outer(s, z)) / s2 - (sz / s2.square()) * torch.outer(s, s)
            if abs(float(sz)) > sr1_tol and float(z2) > sr1_tol**2:
                ms = torch.outer(z, z) / sz
                phi = torch.clamp(1.0 - sz.square() / (s2 * z2), 0.0, 1.0)
            else:
                ms, phi = torch.zeros_like(H[i]), torch.ones((), dtype=H.dtype, device=H.device)
            updated = H[i] + (1.0 - phi) * ms + phi * psb
            result[i] = 0.5 * (updated + updated.T)
        return result

    def _open_log(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_fp = open(self.output, "w", encoding="utf-8")

    def _close_log(self):
        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, text):
        if self.log_fp is not None:
            self.log_fp.write(text)
            self.log_fp.flush()
