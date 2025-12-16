# -*- coding: utf-8 -*-
"""
Batch L-BFGS optimizer with fixed padded dimension nmax.
Highly parallelized optimization using GPU tensors.
Individual batches exit when converged (dynamic batch shrinking).
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64


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
        
        self.memory = memory
        self.curvature = curvature
        self.maxstep = maxstep
        self.maxiter = maxiter
        self.device = torch.device(device)
        self.write_traj = write_traj
        self.traj_every = traj_every
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

    # ===================================================
    # PUBLIC RUN
    # ===================================================
    def run(self, mols) -> None:
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc

        B0 = len(atoms_list)
        if B0 == 0:
            return

        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)

        self._open_log()
        self._w("# Batch L-BFGS optimization start\n")
        self._w(f"# Memory: {self.memory}, Curvature: {self.curvature}, Maxstep: {self.maxstep}\n")
        self._w(f"# Maxiter: {self.maxiter}, Device: {device}\n\n")

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # === First prepare to fix nmax ===
        calc.prepare(atoms_list)
        _, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax fixed)
        self._rebuild_topology(atoms_list)
        self._dump_xyz_all(calc, atoms_list, tag="init")

        # Initialize history tracking
        B = self._B
        self.history_valid = torch.zeros((B, self.memory), dtype=torch.bool, device=device)

        # Collect initial trajectory
        E0, F0 = calc.get_ef_gpu()
        E0 = E0.to(dtype=DTYPE)
        
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

            # Clip step size
            step_cart = self._clip_step_batched(search_dir)

            # Backup coordinates
            calc.backup_coords()

            # Take step
            calc.step_cart_(step_cart)

            # Evaluate new point
            E_new, F_new = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            F_new = F_new.to(dtype=DTYPE)
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
                calc=calc,
                atoms_list=atoms_list,
                step_cart=step_cart,
                F=F_new,
                real_mask=real_mask,
            )

            # Dynamic batch shrinking
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                self._sync_atoms_from_calc(calc, atoms_list)

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                # Shrink history
                self._shrink_history(survive_local)

                calc.prepare(atoms_list, fixed_nmax=self._nmax)
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

        self._close_log()

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
            s_last = self.S_history[-1]
            y_last = self.Y_history[-1]
            valid_last = self.history_valid[:, -1]
            
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
    def _check_convergence(self, it, calc, atoms_list, step_cart, F, real_mask):
        """
        Check convergence criteria for each batch.
        
        Returns:
            done: (B,) boolean tensor indicating converged batches
        """
        device = self.device
        B = len(atoms_list)

        # Get convergence thresholds
        f_max_th = torch.tensor(
            [getattr(at, "f_max_th", 2e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        f_rms_th = torch.tensor(
            [getattr(at, "f_rms_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        dp_max_th = torch.tensor(
            [getattr(at, "dp_max_th", 1e-3) for at in atoms_list],
            dtype=DTYPE, device=device)
        dp_rms_th = torch.tensor(
            [getattr(at, "dp_rms_th", 5e-4) for at in atoms_list],
            dtype=DTYPE, device=device)

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
        E_final, _ = calc.get_ef_gpu()
        self._w(self._fmt_convergence_table(
            it=it,
            E=E_final.to(dtype=DTYPE),
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
        self.log_fp = open(self.output, "w", encoding="utf-8")

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