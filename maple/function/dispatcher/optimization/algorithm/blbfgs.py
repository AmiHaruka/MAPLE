# -*- coding: utf-8 -*-
"""
Batch L-BFGS optimizer with fixed padded dimension nmax.
Highly parallelized optimization using GPU tensors.
Individual batches exit when converged (dynamic batch shrinking).
"""

import os
import operator
import numpy as np
import torch

from ._common import write_xyz

DTYPE = torch.float64
ARMIJO_C1 = 1e-4
BACKTRACK_FACTOR = 0.5
MAX_BACKTRACK_STEPS = 12
ENERGY_ACCEPTANCE_ATOL = 1e-12


class _BatchNonFiniteError(FloatingPointError):
    """Internal marker for non-finite values already assigned by structure."""


class _BatchProtocolError(ValueError):
    """Calculator adapter violated the BatchLBFGS shape/topology contract."""


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_finite_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and positive, got {value!r}"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive, got {value!r}"
        )
    return value


def _ptr_from_atoms(at_list, device):
    ptr = [0]
    for at in at_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


def _masses_flat(at_list, nmax, device):
    B = len(at_list)
    mass = torch.ones((B, nmax), dtype=DTYPE, device=device)
    for b, at in enumerate(at_list):
        m = np.asarray(at.get_masses(), dtype=np.float64)
        mass[b, :3 * len(at)] = torch.from_numpy(np.repeat(m, 3)).to(device=device, dtype=DTYPE)
    return mass


def _symbols_flat(at_list):
    return [at.get_chemical_symbols() for at in at_list]


def _get_coord_gpu(calc) -> torch.Tensor:
    if hasattr(calc, "coord"):
        return calc.coord
    elif hasattr(calc, "coord32"):
        return calc.coord32
    raise AttributeError("calculator has no coord buffer")


class BatchLBFGS:
    """
    Batched L-BFGS optimizer with dynamic batch shrinking.
    Each batch maintains its own L-BFGS history (S, Y, rho).
    """

    def __init__(self,
                 output: str,
                 memory: int = 5,
                 curvature: float = 70.0,
                 maxstep: float = 0.2,
                 maxiter: int = 256,
                 device: str = "cuda",
                 write_traj: bool = False,
                 traj_every: int = 1,
                 verbose: int = 1):
        
        self.memory = _positive_int(memory, "memory")
        self.curvature = _positive_finite_float(curvature, "curvature")
        self.maxstep = _positive_finite_float(maxstep, "maxstep")
        self.maxiter = _positive_int(maxiter, "maxiter")
        self.device = torch.device(device)
        self.write_traj = write_traj
        self.traj_every = _positive_int(traj_every, "traj_every")
        self.verbose = verbose
        
        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_fp = None
        self.xyz_paths = []
        self.frame_counts = []

        self._ptr = None
        self._L_vec = None
        self._nmax = 0
        self._B = 0
        self._symbols_per_batch = None
        self._arange_n = None
        self._real_mask = None
        self._orig_index = None

        # Per-batch L-BFGS history
        self.S_history = []  # List of (B, nmax) tensors per history step
        self.Y_history = []  # List of (B, nmax) tensors per history step
        self.rho_history = []  # List of (B,) tensors per history step
        self.history_valid = None  # (B, memory) mask indicating valid history slots
        self.statuses = []
        self.failure_details = []

    # ===================================================
    # PUBLIC RUN
    # ===================================================
    def run(self, mols) -> tuple[str, ...]:
        device = self.device
        atoms_list = list(mols.multiatoms)
        # Original structures in input order; positions are updated in place
        # by _sync_atoms_from_calc, so these carry the final geometries.
        atoms_all = list(atoms_list)
        output_extension = (
            ".pdb"
            if atoms_all and atoms_all[0].info.get("pdb_template")
            else ".xyz"
        )
        calc = mols.calc

        B0 = len(atoms_list)
        if B0 == 0:
            self.statuses = []
            return ()
        if any(len(at) == 0 for at in atoms_list):
            raise ValueError("BatchLBFGS does not support empty structures.")
        self._validate_convergence_thresholds(atoms_list)
        self.statuses = ["running"] * B0
        self.failure_details = []

        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)

        self._open_log()
        self._w("# Batch L-BFGS optimization start\n")
        self._w(f"# Memory: {self.memory}, Curvature: {self.curvature}, Maxstep: {self.maxstep}\n")
        self._w(f"# Maxiter: {self.maxiter}, Device: {device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # === First prepare to fix nmax ===
        try:
            calc.prepare(atoms_list)
            E0, F0 = calc.get_ef_gpu()
            self._validate_evaluation_shapes(E0, F0, B0, stage="initial evaluation")
            self._validate_prepared_protocol(
                calc,
                atoms_list,
                force_width=int(F0.shape[1]),
                stage="initial evaluation",
            )
            self._require_finite("initial evaluation", energies=E0, forces=F0)
        except _BatchProtocolError as exc:
            self._mark_active_failure(
                "failed_protocol",
                "initial evaluation",
                detail=f"{type(exc).__name__}: {exc}",
            )
            self._close_log()
            raise
        except _BatchNonFiniteError:
            self._close_log()
            raise
        except FloatingPointError as exc:
            self._mark_active_failure(
                "failed_nonfinite",
                "initial evaluation",
                detail=f"{type(exc).__name__}: {exc}",
                fields=("backend",),
            )
            self._close_log()
            raise
        except Exception as exc:
            self._mark_active_failure(
                "failed_backend",
                "initial evaluation",
                detail=f"{type(exc).__name__}: {exc}",
            )
            self._close_log()
            raise
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax fixed)
        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        # Initialize history tracking
        B = self._B
        self.history_valid = torch.zeros((B, self.memory), dtype=torch.bool, device=device)

        E0 = E0.to(dtype=DTYPE)

        # Final energy per original structure, refreshed as batches converge
        # out; used for the closing optimized-structure dump.
        final_E = E0.clone()

        iteration = 0

        # Get initial energy and forces
        E_old = E0
        F_old = F0.to(dtype=DTYPE)
        
        while iteration < self.maxiter and len(atoms_list) > 0:
            iteration += 1
            real_mask = self._real_mask

            # Current gradient (negative force)
            g_cart = -F_old * real_mask.to(DTYPE)

            # Compute L-BFGS search direction
            search_dir = self._two_loop_batched(g_cart)
            self._require_finite(
                "raw L-BFGS search direction",
                search_direction=search_dir,
            )
            search_dir = self._safeguard_search_direction(
                search_dir,
                g_cart,
            )
            self._require_finite("search direction", search_direction=search_dir)

            # Clip step size
            step_cart = self._clip_step_batched(search_dir)
            self._require_finite("Cartesian step", step=step_cart)

            # Accept a finite, energy-decreasing step independently for every
            # active structure. Rejected rows are backtracked from the same
            # coordinate snapshot; no trial geometry is accepted implicitly.
            E_new, F_new, step_cart = self._evaluate_accepted_step(
                calc,
                step_cart,
                E_old,
                g_cart,
                iteration,
            )
            g_new = -F_new * real_mask.to(DTYPE)

            # Update L-BFGS history
            s_vec = step_cart
            y_vec = g_new - g_cart
            self._update_history_batched(s_vec, y_vec)

            # Log iteration info
            self._log_iteration(iteration, E_new, step_cart, F_new, real_mask)

            # Write trajectory for accepted steps
            if self.write_traj and iteration % self.traj_every == 0:
                self._dump_xyz_all(calc, atoms_list, tag=f"iter={iteration}")

            # Check convergence
            done = self._check_convergence(
                it=iteration,
                E=E_new,
                step_cart=step_cart,
                F=F_new,
            )

            final_E[self._orig_index] = E_new
            for original_index in self._orig_index[done].detach().cpu().tolist():
                self.statuses[int(original_index)] = "converged"

            # Dynamic batch shrinking
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                self._sync_atoms_from_calc(calc, atoms_list)

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                # Shrink history
                self._shrink_history(survive_local)

                if atoms_list:
                    try:
                        calc.prepare(atoms_list, fixed_nmax=self._nmax)
                        self._validate_prepared_protocol(
                            calc,
                            atoms_list,
                            force_width=self._nmax,
                            stage=f"iteration {iteration} batch shrink",
                        )
                    except _BatchProtocolError as exc:
                        self._mark_active_failure(
                            "failed_protocol",
                            f"iteration {iteration} batch shrink",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        self._close_log()
                        raise
                    except Exception as exc:
                        self._mark_active_failure(
                            "failed_backend",
                            f"iteration {iteration} batch shrink",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        self._close_log()
                        raise
                    self._rebuild_topology(atoms_list)

                # Update old values
                E_old = E_new[survive_local]
                F_old = F_new[survive_local]
            else:
                E_old = E_new
                F_old = F_new

            if len(atoms_list) == 0:
                self._w("\n# All batches converged!\n")
                break

        else:
            self._w("\n# Maximum iterations reached.\n")

        # Converged batches were synced when they left the batch; push the
        # final coordinates of any still-unconverged structures back too.
        if len(atoms_list) > 0:
            self._sync_atoms_from_calc(calc, atoms_list)
            for original_index in self._orig_index.detach().cpu().tolist():
                self.statuses[int(original_index)] = "maxiter"

            failed_file = (
                os.path.splitext(self.output)[0]
                + "_opt_unconverged"
                + output_extension
            )
            write_xyz(
                failed_file,
                atoms_all,
                energies=final_E.detach().cpu().tolist(),
            )
            self._w(
                "\n# Unconverged diagnostic frames written to "
                f"{failed_file}\n"
            )
            self._w(f"# Per-structure status: {self.statuses}\n")
            self._close_log()
            raise RuntimeError(
                "BatchLBFGS reached maxiter with unconverged structures; "
                f"no production _opt{output_extension} was written. "
                f"Diagnostic: {failed_file}"
            )

        # Final geometries, matching the single-structure output convention.
        opt_file = (
            os.path.splitext(self.output)[0] + "_opt" + output_extension
        )
        write_xyz(opt_file, atoms_all, energies=final_E.detach().cpu().tolist())
        self._w(f"\n# Final frames written to {opt_file}\n")
        self._w(f"# Per-structure status: {self.statuses}\n")

        self._close_log()
        return tuple(self.statuses)

    def _require_finite(self, stage: str, **named_tensors) -> None:
        """Fail before non-finite optimizer state can be propagated or written."""
        invalid_by_index = {}
        active_count = (
            int(self._orig_index.numel())
            if self._orig_index is not None
            else 0
        )
        for name, tensor in named_tensors.items():
            tensor = torch.as_tensor(tensor)
            finite = torch.isfinite(tensor)
            if bool(finite.all().item()):
                continue
            if tensor.ndim >= 1 and tensor.shape[0] == active_count:
                invalid_rows = ~finite.reshape(active_count, -1).all(dim=1)
            else:
                invalid_rows = torch.ones(
                    active_count,
                    dtype=torch.bool,
                    device=self.device,
                )
            for local_index in invalid_rows.nonzero(as_tuple=False).flatten().tolist():
                original_index = int(self._orig_index[local_index].item())
                invalid_by_index.setdefault(original_index, []).append(name)

        if not invalid_by_index:
            return

        details = "; ".join(
            f"structure {index}: {','.join(fields)}"
            for index, fields in sorted(invalid_by_index.items())
        )
        message = f"BatchLBFGS {stage} produced non-finite values ({details})"
        for original_index, fields in invalid_by_index.items():
            self.statuses[original_index] = "failed_nonfinite"
            self.failure_details.append(
                {
                    "index": original_index,
                    "status": "failed_nonfinite",
                    "stage": stage,
                    "fields": tuple(fields),
                }
            )
        self._terminalize_running_after_abort(
            stage,
            cause="another active structure produced non-finite values",
        )
        self._w(f"\n# ERROR: {message}\n")
        self._w(f"# Per-structure status: {self.statuses}\n")
        self._close_log()
        raise _BatchNonFiniteError(message)

    def _mark_active_failure(
        self,
        status: str,
        stage: str,
        *,
        detail: str,
        fields: tuple[str, ...] | None = None,
    ) -> None:
        if self._orig_index is None:
            return
        for original_index in self._orig_index.detach().cpu().tolist():
            index = int(original_index)
            if self.statuses[index] == "running":
                self.statuses[index] = status
                failure_detail = {
                    "index": index,
                    "status": status,
                    "stage": stage,
                    "detail": detail,
                }
                if fields is not None:
                    failure_detail["fields"] = fields
                self.failure_details.append(failure_detail)
        self._w(f"\n# ERROR: BatchLBFGS {stage}: {detail}\n")
        self._w(f"# Per-structure status: {self.statuses}\n")

    def _terminalize_running_after_abort(
        self,
        stage: str,
        *,
        cause: str,
    ) -> None:
        """Ensure a raised batch-wide abort leaves no structure ``running``."""
        for index, status in enumerate(self.statuses):
            if status != "running":
                continue
            self.statuses[index] = "aborted_peer_failure"
            self.failure_details.append(
                {
                    "index": index,
                    "status": "aborted_peer_failure",
                    "stage": stage,
                    "detail": cause,
                }
            )

    def _validate_evaluation_shapes(
        self,
        energies: torch.Tensor,
        forces: torch.Tensor,
        batch_size: int,
        *,
        stage: str,
    ) -> None:
        energy_shape = tuple(torch.as_tensor(energies).shape)
        force_shape = tuple(torch.as_tensor(forces).shape)
        if energy_shape != (batch_size,):
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} energy shape is {energy_shape}, "
                f"expected {(batch_size,)}"
            )
        if len(force_shape) != 2 or force_shape[0] != batch_size:
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} force shape is {force_shape}, "
                f"expected ({batch_size}, nmax)"
            )
        if self._nmax:
            expected_force_shape = (batch_size, self._nmax)
        else:
            expected_force_shape = force_shape
        if self._nmax and force_shape != expected_force_shape:
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} force shape is {force_shape}, "
                f"expected {expected_force_shape}"
            )

    def _validate_prepared_protocol(
        self,
        calc,
        atoms_list,
        *,
        force_width: int,
        stage: str,
    ) -> None:
        """Validate the public adapter topology before any optimizer step."""
        force_width = int(force_width)
        required_width = 3 * max(len(atoms) for atoms in atoms_list)
        if force_width < required_width or force_width % 3 != 0:
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} force width {force_width} is invalid; "
                f"need a multiple of 3 covering at least {required_width} "
                "Cartesian DOFs."
            )

        advertised_width = getattr(calc, "nmax_dof", None)
        if advertised_width is not None:
            try:
                advertised_width = operator.index(advertised_width)
            except TypeError as exc:
                raise _BatchProtocolError(
                    f"BatchLBFGS {stage} calculator nmax_dof must be an "
                    f"integer, got {advertised_width!r}."
                ) from exc
            if advertised_width != force_width:
                raise _BatchProtocolError(
                    f"BatchLBFGS {stage} force width {force_width} does not "
                    f"match calculator nmax_dof={advertised_width}."
                )

        total_atoms = sum(len(atoms) for atoms in atoms_list)
        coord = torch.as_tensor(_get_coord_gpu(calc))
        coord_shape = tuple(coord.shape)
        if coord_shape != (total_atoms, 3):
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} coordinate shape is {coord_shape}, "
                f"expected {(total_atoms, 3)}."
            )
        if (
            not torch.is_floating_point(coord)
            or not bool(torch.isfinite(coord).all().item())
        ):
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} coordinate buffer must be finite "
                "floating-point Cartesian data."
            )
        expected_device = self.device
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
        if coord.device != expected_device:
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} coordinate buffer is on {coord.device}, "
                f"expected optimizer device {expected_device}."
            )
        expected_coord = torch.as_tensor(
            np.concatenate(
                [atoms.get_positions() for atoms in atoms_list],
                axis=0,
            ),
            dtype=coord.dtype,
            device=coord.device,
        )
        eps = float(torch.finfo(coord.dtype).eps)
        if not bool(
            torch.allclose(
                coord,
                expected_coord,
                rtol=8.0 * eps,
                atol=8.0 * eps,
            )
        ):
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} coordinate contents/order do not match "
                "the supplied structures."
            )

        ptr = getattr(calc, "_ptr", None)
        if ptr is None:
            ptr = getattr(calc, "ptr", None)
        if ptr is None:
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} calculator must expose atom pointer "
                "topology as _ptr or ptr."
            )
        ptr = torch.as_tensor(ptr).detach().cpu()
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        expected_ptr = [0]
        for atoms in atoms_list:
            expected_ptr.append(expected_ptr[-1] + len(atoms))
        if (
            ptr.dtype not in integer_dtypes
            or ptr.ndim != 1
            or ptr.numel() != len(expected_ptr)
            or ptr.tolist() != expected_ptr
        ):
            raise _BatchProtocolError(
                f"BatchLBFGS {stage} atom pointer topology is {ptr.tolist()}, "
                f"expected {expected_ptr}."
            )

    def _evaluate_accepted_step(
        self,
        calc,
        proposed_step: torch.Tensor,
        old_energy: torch.Tensor,
        gradient: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backtrack rejected structures until every active row is accepted."""
        base_coord = _get_coord_gpu(calc).detach().clone()
        batch_size = proposed_step.shape[0]
        scales = torch.ones(batch_size, dtype=DTYPE, device=self.device)
        rejected = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        nonfinite_rows = torch.zeros_like(rejected)
        nonfinite_energy_rows = torch.zeros_like(rejected)
        nonfinite_force_rows = torch.zeros_like(rejected)

        for attempt in range(MAX_BACKTRACK_STEPS + 1):
            with torch.no_grad():
                _get_coord_gpu(calc).copy_(base_coord)
            trial_step = proposed_step * scales.unsqueeze(-1)
            try:
                calc.step_cart_(trial_step)
                trial_energy, trial_forces = calc.get_ef_gpu()
                self._validate_evaluation_shapes(
                    trial_energy,
                    trial_forces,
                    batch_size,
                    stage=f"iteration {iteration} trial {attempt}",
                )
                trial_energy = trial_energy.to(dtype=DTYPE)
                trial_forces = trial_forces.to(dtype=DTYPE)
            except _BatchProtocolError as exc:
                with torch.no_grad():
                    _get_coord_gpu(calc).copy_(base_coord)
                self._mark_active_failure(
                    "failed_protocol",
                    f"iteration {iteration} trial {attempt}",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                self._close_log()
                raise
            except FloatingPointError as exc:
                with torch.no_grad():
                    _get_coord_gpu(calc).copy_(base_coord)
                self._mark_active_failure(
                    "failed_nonfinite",
                    f"iteration {iteration} trial {attempt}",
                    detail=f"{type(exc).__name__}: {exc}",
                    fields=("backend",),
                )
                self._close_log()
                raise
            except Exception as exc:
                with torch.no_grad():
                    _get_coord_gpu(calc).copy_(base_coord)
                self._mark_active_failure(
                    "failed_backend",
                    f"iteration {iteration} trial {attempt}",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                self._close_log()
                raise

            nonfinite_energy_rows = ~torch.isfinite(trial_energy)
            nonfinite_force_rows = ~torch.isfinite(trial_forces).all(dim=1)
            nonfinite_rows = nonfinite_energy_rows | nonfinite_force_rows
            directional = (gradient * trial_step).sum(dim=-1)
            armijo_bound = (
                old_energy
                + ARMIJO_C1 * directional
                + ENERGY_ACCEPTANCE_ATOL
            )
            rejected = nonfinite_rows | (trial_energy > armijo_bound)
            if not bool(rejected.any().item()):
                return trial_energy, trial_forces, trial_step

            if attempt < MAX_BACKTRACK_STEPS:
                scales = torch.where(
                    rejected,
                    scales * BACKTRACK_FACTOR,
                    scales,
                )

        with torch.no_grad():
            _get_coord_gpu(calc).copy_(base_coord)

        failed = []
        for local_index in rejected.nonzero(as_tuple=False).flatten().tolist():
            original_index = int(self._orig_index[local_index].item())
            status = (
                "failed_nonfinite"
                if bool(nonfinite_rows[local_index].item())
                else "failed_line_search"
            )
            self.statuses[original_index] = status
            detail = {
                "index": original_index,
                "status": status,
                "stage": f"iteration {iteration} line search",
                "attempts": MAX_BACKTRACK_STEPS + 1,
            }
            if status == "failed_nonfinite":
                fields = []
                if bool(nonfinite_energy_rows[local_index].item()):
                    fields.append("energies")
                if bool(nonfinite_force_rows[local_index].item()):
                    fields.append("forces")
                detail["fields"] = tuple(fields)
            self.failure_details.append(detail)
            failed.append(f"{original_index}:{status}")

        self._terminalize_running_after_abort(
            f"iteration {iteration} line search",
            cause="the active batch aborted after another structure failed line search",
        )
        message = (
            "BatchLBFGS could not accept a finite energy-decreasing step for "
            f"structures {', '.join(failed)} after "
            f"{MAX_BACKTRACK_STEPS + 1} attempts"
        )
        self._w(f"\n# ERROR: {message}\n")
        self._w(f"# Per-structure status: {self.statuses}\n")
        self._close_log()
        raise RuntimeError(message)

    @staticmethod
    def _validate_convergence_thresholds(atoms_list) -> None:
        defaults = {
            "f_max_th": 2e-3,
            "f_rms_th": 1e-3,
            "dp_max_th": 1e-3,
            "dp_rms_th": 5e-4,
        }
        for index, atoms in enumerate(atoms_list):
            for name, default in defaults.items():
                value = getattr(atoms, name, default)
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Structure {index} {name} must be finite and positive, "
                        f"got {value!r}"
                    ) from exc
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"Structure {index} {name} must be finite and positive, "
                        f"got {value!r}"
                    )

    # ===================================================
    # L-BFGS TWO-LOOP RECURSION (BATCHED)
    # ===================================================
    def _two_loop_batched(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Batched L-BFGS two-loop recursion.
        
        Args:
            grad: (B, nmax) gradient tensor
            
        Returns:
            search_dir: (B, nmax) search direction tensor
        """
        device = self.device
        B, nmax = grad.shape
        
        q = grad.clone()
        alpha_list = []

        # Backward pass through history (most recent first)
        num_history = len(self.S_history)
        for t in range(num_history - 1, -1, -1):
            s_t = self.S_history[t]  # (B, nmax)
            y_t = self.Y_history[t]  # (B, nmax)
            rho_t = self.rho_history[t]  # (B,)
            valid_t = self.history_valid[:, t]  # (B,)

            # alpha = rho * (s^T q)
            alpha = rho_t * (s_t * q).sum(dim=-1)  # (B,)
            alpha = torch.where(valid_t, alpha, torch.zeros_like(alpha))
            alpha_list.append(alpha)

            # q = q - alpha * y
            q = q - alpha.unsqueeze(-1) * y_t

        # Reverse alpha_list for forward pass
        alpha_list = list(reversed(alpha_list))

        # Initial Hessian approximation
        if num_history > 0:
            # gamma = (y^T s) / (y^T y)
            # The newest history entry lives at column len(S_history)-1 until
            # the buffer is full; column -1 is still all-False before that.
            s_last = self.S_history[-1]
            y_last = self.Y_history[-1]
            valid_last = self.history_valid[:, num_history - 1]
            
            ys = (y_last * s_last).sum(dim=-1)
            yy = (y_last * y_last).sum(dim=-1)
            gamma = ys / (yy + 1e-20)
            gamma = torch.where(valid_last, gamma, torch.ones_like(gamma) / self.curvature)
        else:
            gamma = torch.full((B,), 1.0 / self.curvature, dtype=DTYPE, device=device)

        z = gamma.unsqueeze(-1) * q

        # Forward pass through history (oldest first)
        for t in range(num_history):
            s_t = self.S_history[t]
            y_t = self.Y_history[t]
            rho_t = self.rho_history[t]
            valid_t = self.history_valid[:, t]
            alpha_t = alpha_list[t]

            # beta = rho * (y^T z)
            beta = rho_t * (y_t * z).sum(dim=-1)
            beta = torch.where(valid_t, beta, torch.zeros_like(beta))

            # z = z + s * (alpha - beta)
            z = z + s_t * (alpha_t - beta).unsqueeze(-1)

        return -z

    # ===================================================
    # STEP CLIPPING
    # ===================================================
    def _safeguard_search_direction(
        self,
        direction: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Use steepest descent only for finite but non-downhill directions."""
        directional_derivative = (direction * gradient).sum(dim=-1)
        invalid = directional_derivative >= 0.0
        fallback = -(1.0 / self.curvature) * gradient
        return torch.where(invalid.unsqueeze(-1), fallback, direction)

    def _clip_step_batched(self, step: torch.Tensor) -> torch.Tensor:
        """
        Clip step size per batch to maxstep.
        
        Args:
            step: (B, nmax) step tensor
            
        Returns:
            clipped_step: (B, nmax) clipped step tensor
        """
        max_disp = step.abs().amax(dim=-1)  # (B,)
        scale = torch.clamp(self.maxstep / (max_disp + 1e-20), max=1.0)
        return step * scale.unsqueeze(-1)

    # ===================================================
    # HISTORY UPDATE
    # ===================================================
    def _update_history_batched(self, s_vec: torch.Tensor, y_vec: torch.Tensor):
        """
        Update L-BFGS history for all batches.
        
        Args:
            s_vec: (B, nmax) position change
            y_vec: (B, nmax) gradient change
        """
        device = self.device
        B = s_vec.shape[0]

        # Compute rho = 1 / (y^T s)
        ys = (y_vec * s_vec).sum(dim=-1)  # (B,)
        rho = 1.0 / (ys + 1e-20)
        
        # Check validity (finite rho and positive curvature)
        valid = torch.isfinite(rho) & (ys > 1e-12)

        # Add to history
        self.S_history.append(s_vec.clone())
        self.Y_history.append(y_vec.clone())
        self.rho_history.append(rho)

        # Update validity mask
        if len(self.S_history) > self.memory:
            # Remove oldest
            self.S_history.pop(0)
            self.Y_history.pop(0)
            self.rho_history.pop(0)
            # Shift validity
            self.history_valid = torch.cat([
                self.history_valid[:, 1:],
                valid.unsqueeze(-1)
            ], dim=-1)
        else:
            # Append new validity column
            if self.history_valid.shape[1] < self.memory:
                new_col = torch.zeros((B, self.memory - self.history_valid.shape[1]), 
                                     dtype=torch.bool, device=device)
                self.history_valid = torch.cat([self.history_valid, new_col], dim=-1)
            self.history_valid[:, len(self.S_history) - 1] = valid

    # ===================================================
    # HISTORY SHRINKING
    # ===================================================
    def _shrink_history(self, survive_idx: torch.Tensor):
        """Shrink history when batches are removed."""
        for i in range(len(self.S_history)):
            self.S_history[i] = self.S_history[i][survive_idx]
            self.Y_history[i] = self.Y_history[i][survive_idx]
            self.rho_history[i] = self.rho_history[i][survive_idx]
        self.history_valid = self.history_valid[survive_idx]

    # ===================================================
    # CONVERGENCE CHECK
    # ===================================================
    def _check_convergence(self, it, E, step_cart, F):
        """
        Check convergence criteria for each batch.

        E, F and step_cart are the values already evaluated for this
        iteration; thresholds come from the topology rebuild.

        Returns:
            done: (B,) boolean tensor indicating converged batches
        """
        f_max_th = self._f_max_th
        f_rms_th = self._f_rms_th
        dp_max_th = self._dp_max_th
        dp_rms_th = self._dp_rms_th

        L_eff = self._L_vec.clamp(min=1).to(DTYPE)

        # Compute metrics
        max_f = F.abs().amax(dim=-1)
        rms_f = torch.sqrt((F ** 2).sum(-1) / L_eff)
        max_dp = step_cart.abs().amax(dim=-1)
        rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

        # Check convergence
        done = (
            (max_f <= f_max_th)
            & (rms_f <= f_rms_th)
            & (max_dp <= dp_max_th)
            & (rms_dp <= dp_rms_th)
        )

        # Log convergence table
        self._w(self._fmt_convergence_table(
            it=it,
            E=E.to(dtype=DTYPE),
            max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp,
            f_max_th=f_max_th, f_rms_th=f_rms_th,
            dp_max_th=dp_max_th, dp_rms_th=dp_rms_th,
            done=done
        ))

        return done

    # ===================================================
    # TOPOLOGY
    # ===================================================
    def _rebuild_topology(self, atoms_list):
        device = self.device
        B = len(atoms_list)
        self._B = B

        self._ptr = _ptr_from_atoms(atoms_list, device)
        L_list = [3 * len(at) for at in atoms_list]
        self._L_vec = torch.tensor(L_list, dtype=torch.int64, device=device)

        # real_mask padded to fixed nmax
        self._real_mask = (self._arange_n[None, :] < self._L_vec[:, None])

        # Per-structure convergence thresholds, fixed for the batch lifetime
        self._f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        self._dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list],
            dtype=DTYPE, device=device)

    def _sync_atoms_from_calc(self, calc, atoms_list):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i + 1]
            at.positions[:] = pos[s:t]

    # ===================================================
    # LOGGING
    # ===================================================
    def _open_log(self):
        # Append: self.output is the shared job output file, already holding
        # the input-reading and calculator-setup sections.
        self.log_fp = open(self.output, "a", encoding="utf-8")

    def _close_log(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, s):
        if self.log_fp:
            self.log_fp.write(s)
            self.log_fp.flush()

    def _log_iteration(self, it, E, step_cart, F, real_mask):
        """Log iteration information."""
        if self.verbose == 0:
            return

        L_eff = self._L_vec.clamp(min=1).to(DTYPE)
        max_f = F.abs().amax(dim=-1)
        rms_f = torch.sqrt((F ** 2).sum(-1) / L_eff)
        max_dp = step_cart.abs().amax(dim=-1)
        rms_dp = torch.sqrt((step_cart ** 2).sum(-1) / L_eff)

        msg = f"\nIter {it}: "
        msg += f"E_mean={E.mean().item():.6f} "
        msg += f"max|F|_mean={max_f.mean().item():.6f} "
        msg += f"rms|F|_mean={rms_f.mean().item():.6f} "
        msg += f"max|dX|_mean={max_dp.mean().item():.6f} "
        msg += f"rms|dX|_mean={rms_dp.mean().item():.6f}\n"
        
        self._w(msg)

    def _fmt_convergence_table(self, it, E, max_f, rms_f, max_dp, rms_dp,
                               f_max_th, f_rms_th, dp_max_th, dp_rms_th, done):
        """Format convergence table."""
        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"{('Batch L-BFGS Iteration ' + str(it)).center(70)}\n")
        lines.append("-" * 70 + "\n")
        lines.append(
            "Batch   Energy(Ha)   Max|F|(H/A)  RMS|F|  Max|dX|(A)  RMS|dX|  Conv\n"
        )
        lines.append("-" * 70 + "\n")

        B = E.shape[0]
        for b in range(B):
            idx_orig = int(self._orig_index[b].item())
            lines.append(
                f"[{idx_orig:2d}] "
                f"{E[b].item():14.6f} "
                f"{max_f[b].item():12.6f} "
                f"{rms_f[b].item():10.6f} "
                f"{max_dp[b].item():12.6f} "
                f"{rms_dp[b].item():10.6f} "
                f"{('YES' if bool(done[b]) else 'NO')}\n"
            )

        lines.append("\n")
        return "".join(lines)

    # ===================================================
    # XYZ OUTPUT
    # ===================================================
    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"lbfgs_batch{i + 1}.xyz")
            for i in range(B_all)
        ]
        self.frame_counts = [0 for _ in range(B_all)]
        for p in self.xyz_paths:
            open(p, "w").close()

    def _dump_xyz_all(self, calc, atoms_list, tag="init"):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local, at in enumerate(atoms_list):
            s, t = ptr[i_local], ptr[i_local + 1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], tag)

    def _append_xyz(self, idx_orig, symbols, pos_np, comment=""):
        path = self.xyz_paths[idx_orig]
        n = pos_np.shape[0]
        with open(path, "a") as f:
            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for k in range(n):
                x, y, z = pos_np[k]
                f.write(f"{symbols[k]:<2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
        self.frame_counts[idx_orig] += 1
