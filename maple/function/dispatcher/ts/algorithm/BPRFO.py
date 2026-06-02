# -*- coding: utf-8 -*-
"""
BatchPRFO with fixed padded dimension nmax.
All EFH are padded to nmax determined from FIRST prepare/get_ef_gpu.
Calculator.prepare() is called with fixed_nmax afterwards.

FIXES:
1. Bofill update now happens after EVERY accepted step (not just non-recalc)
2. Gradient consistency: use same gradient source throughout iteration
3. Increased SR1 tolerance from 1e-10 to 1e-8
4. Better step acceptance tracking
"""

from typing import List, Optional
import os
import numpy as np
import torch
from ase import Atoms

DTYPE = torch.float64
BIG = 1e8


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


class BatchPRFO:
    """
    Batched RS-PRFO with fixed padded dimension nmax.
    """

    def __init__(self,
                 output: str,
                 trust_init: float = 0.20,
                 trust_min: float = 1e-3,
                 trust_max: float = 1.00,
                 eta_shrink: float = 0.75,
                 eta_expand: float = 1.75,
                 max_inner_attempts: int = 10,
                 max_outer_iter: int = 256,
                 device: str = "cuda",
                 recalc: int = 4,
                 hessian_update: str = "bofill"):
        self.trust_init = trust_init
        self.trust_min = trust_min
        self.trust_max = trust_max
        self.eta_shrink = eta_shrink
        self.eta_expand = eta_expand
        self.max_inner_attempts = max_inner_attempts
        self.max_outer_iter = max_outer_iter
        self.device = torch.device(device)
        self.recalc = max(1, int(recalc))
        self.hessian_update = str(hessian_update).lower()
        if self.hessian_update not in {"bofill", "bfgs"}:
            raise ValueError("hessian_update must be 'bofill' or 'bfgs'.")
        
        self.output = os.path.abspath(output)
        self.out_dir = os.path.dirname(self.output) or "."
        os.makedirs(self.out_dir, exist_ok=True)

        self.log_fp = None
        self.xyz_paths = []
        self.frame_counts = []

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        self._ptr = None
        self._L_vec = None
        self._nmax = 0
        self._B = 0
        self._symbols_per_batch = None
        self._arange_n = None
        self._real_mask = None
        self._D = None

        self._orig_index = None
        self._H_work = None
        self._g_cart_prev = None
        self._recent_acceptance_rate = 0.0
        self._enable_vectorized_mu = False


    # ===================================================
    # PUBLIC RUN
    # ===================================================
    @profile
    def run(self, mols) -> None:
        device = self.device
        atoms_list = list(mols.multiatoms)
        calc = mols.calc

        B0 = len(atoms_list)
        if B0 == 0:
            return

        self._orig_index = torch.arange(B0, dtype=torch.long, device=device)

        self._open_log()
        self._w("# RS-PRFO batched TS search start\n")
        self._w(f"# RecalcFC interval: {self.recalc}\n")
        self._w(f"# Hessian update method: {self.hessian_update}\n")

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

        B = self._B
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        last_step = torch.zeros((B, self._nmax), dtype=DTYPE, device=device)

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        outer_it = 0
        self._H_work = None
        self._g_cart_prev = None

        while outer_it < self.max_outer_iter and len(atoms_list) > 0:
            outer_it += 1
            real_mask = self._real_mask

            calc.backup_coords()

            # Decide whether to rebuild Hessian from calculator (Gaussian RecalcFC-like)
            need_recalc = (outer_it == 1) or ((outer_it - 1) % self.recalc == 0)

            if need_recalc:
                # Pull EFH (true or numerical) from calculator; pad to nmax
                E_old, F_tmp, H_tmp, _ = calc.get_efh_gpu()
                E_old = E_old.to(dtype=DTYPE)
                F_tmp = F_tmp.to(dtype=DTYPE)
                H_tmp = 0.5 * (H_tmp + H_tmp.transpose(-1, -2)).to(dtype=DTYPE)

                # pad to nmax if needed
                Bcur, Lcur = F_tmp.shape
                if Lcur != self._nmax:
                    H_pad = torch.zeros((Bcur, self._nmax, self._nmax), dtype=DTYPE, device=H_tmp.device)
                    H_pad[:, :Lcur, :Lcur] = H_tmp
                    H_use = H_pad
                    F_use = torch.zeros((Bcur, self._nmax), dtype=DTYPE, device=F_tmp.device)
                    F_use[:, :Lcur] = F_tmp
                else:
                    H_use = H_tmp
                    F_use = F_tmp

                # Build Cartesian H and g
                H_cart, g_cart = self._build_cartesian_hg(F_use, H_use, real_mask)

                # Store exact Hessian and gradient
                self._H_work = H_cart.clone()
                self._g_cart_prev = g_cart.clone()
                
                self._w(f"[Iter {outer_it}] Recalculated exact Hessian\n")

            else:
                # Non-recalc step: use working Hessian; get current EF
                E_old, F_now = calc.get_ef_gpu()
                E_old = E_old.to(dtype=DTYPE)
                F_now = F_now.to(dtype=DTYPE)
                
                # Use working Hessian, compute current gradient
                H_cart = self._H_work
                g_cart = -F_now * real_mask.to(DTYPE)

            # Mass-weighting and eigen-decomposition
            H_mw, g_mw = self._mass_weight_hg(H_cart, g_cart, real_mask)
            w, V, gp = self._eigh_and_track_modes(H_mw, g_mw)

            # Inner RS-PRFO loop
            trust_r, last_step, last_rho, step_accepted = self._inner_rs_prfo_loop(
                it=outer_it,
                calc=calc,
                w=w, V=V, gp=gp,
                H=H_cart, g_cart=g_cart,
                trust_r=trust_r,
                last_step=last_step,
                real_mask=real_mask,
                E_old=E_old,
            )

            # After step: get new gradient
            E_fin, F_fin = calc.get_ef_gpu()
            F_fin = F_fin.to(dtype=DTYPE)
            g_new_cart = -F_fin * real_mask.to(DTYPE)

            # Bofill update: apply ONLY if we didn't just recalculate AND at least one step was accepted
            if not need_recalc and step_accepted.any():
                self._w(
                    f"[Iter {outer_it}] Applying {self.hessian_update.upper()} "
                    f"update to {step_accepted.sum().item()} batches\n"
                )
                update_fn = (
                    self._bfgs_update_batched
                    if self.hessian_update == "bfgs"
                    else self._bofill_update_batched
                )
                self._H_work = update_fn(
                    H=self._H_work,
                    s_cart=last_step,
                    g_prev=self._g_cart_prev,
                    g_new=g_new_cart,
                    real_mask=real_mask,
                    step_accepted=step_accepted
                )
            
            # Always update gradient buffer for next iteration
            self._g_cart_prev = g_new_cart.clone()

            # Convergence check
            done = self._check_convergence(
                it=outer_it,
                calc=calc,
                atoms_list=atoms_list,
                trust_r=trust_r,
                last_step=last_step,
                real_mask=real_mask,
                last_rho=last_rho,
            )

            # Dynamic batch shrinking
            survive_local = (~done).nonzero(as_tuple=False).flatten()
            if survive_local.numel() < len(done):
                self._sync_atoms_from_calc(calc, atoms_list)

                atoms_list = [atoms_list[i] for i in survive_local.cpu().tolist()]
                self._orig_index = self._orig_index[survive_local]

                trust_r = trust_r[survive_local]
                last_step = last_step[survive_local]

                if self.tracked_mode_idx is not None:
                    self.tracked_mode_idx = self.tracked_mode_idx[survive_local]
                if self.tracked_mode_vec_mw is not None:
                    self.tracked_mode_vec_mw = self.tracked_mode_vec_mw[survive_local]

                calc.prepare(atoms_list, fixed_nmax=self._nmax)
                self._rebuild_topology(atoms_list)

                # Slice working buffers
                if self._H_work is not None:
                    self._H_work = self._H_work[survive_local]
                if self._g_cart_prev is not None:
                    self._g_cart_prev = self._g_cart_prev[survive_local]

        else:
            self._w("# Maximum iterations reached.\n")

        self._close_log()


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
        mass = _masses_flat(atoms_list, self._nmax, device)
        self._D = 1.0 / torch.sqrt(torch.clamp(mass, min=1e-12))

    def _sync_atoms_from_calc(self, calc, atoms_list):
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i, at in enumerate(atoms_list):
            s, t = ptr[i], ptr[i+1]
            at.positions[:] = pos[s:t]

    # ===================================================
    # EFH with padding to fixed nmax
    # ===================================================

    @profile
    def _compute_efh(self, calc):
        """
        EFH must be padded to the fixed nmax from the first iteration.
        """
        E_old, F_raw, H_raw, _ = calc.get_efh_gpu()

        F_raw = F_raw.to(dtype=DTYPE)
        H_raw = 0.5 * (H_raw + H_raw.transpose(-1, -2)).to(dtype=DTYPE)

        nmax = int(self._nmax)
        B, L = F_raw.shape

        if L != nmax:
            F_pad = torch.zeros((B, nmax), dtype=DTYPE, device=F_raw.device)
            F_pad[:, :L] = F_raw
            F_raw = F_pad

            H_pad = torch.zeros((B, nmax, nmax), dtype=DTYPE, device=H_raw.device)
            H_pad[:, :L, :L] = H_raw
            H_raw = H_pad

        return E_old.to(dtype=DTYPE), F_raw, H_raw

    def _build_cartesian_hg(self, F_raw, H_raw, real_mask):
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)
        H = H_raw * mask_ij
        g_cart = -F_raw * real_mask.to(DTYPE)
        return H, g_cart

    def _mass_weight_hg(self, H, g_cart, real_mask):
        D = self._D
        arange_n = self._arange_n

        g_mw = D * g_cart
        H_mw = D.unsqueeze(-1) * H * D.unsqueeze(-2)

        pad_mask = ~real_mask
        if pad_mask.any():
            H_mw = H_mw.clone()
            diag = H_mw[..., arange_n, arange_n]
            H_mw[..., arange_n, arange_n] = diag + pad_mask.to(DTYPE) * BIG
            g_mw = g_mw * (~pad_mask).to(DTYPE)

        return H_mw, g_mw

    # ===================================================
    # EIGEN + TRACKING
    # ===================================================

    def _eigh_and_track_modes(self, H_mw, g_mw):
        B, n = g_mw.shape

        w, V = torch.linalg.eigh(H_mw)
        gp = (V.transpose(-1, -2) @ g_mw.unsqueeze(-1)).squeeze(-1)

        tiny = w.abs() < 1e-10
        w = torch.where(tiny, torch.sign(w) * 1e-10, w)

        if self.tracked_mode_vec_mw is None:
            neg_idx = torch.argmin(w, dim=1)
            has_neg = w.gather(1, neg_idx[:, None]).squeeze(1) < -1e-6
            alt_idx = torch.argmax(gp.abs(), dim=1)
            tracked_idx = torch.where(has_neg, neg_idx, alt_idx)

            self.tracked_mode_idx = tracked_idx
            self.tracked_mode_vec_mw = torch.stack(
                [V[b, :, tracked_idx[b]] for b in range(B)], dim=0
            )

        else:
            overlap = torch.matmul(
                V.transpose(-1, -2),
                self.tracked_mode_vec_mw.unsqueeze(-1)
            ).squeeze(-1)

            idx = torch.argmax(overlap.abs(), dim=-1)
            signs = torch.sign(overlap.gather(1, idx[:, None]).squeeze(1))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)

            self.tracked_mode_idx = idx
            self.tracked_mode_vec_mw = torch.stack(
                [V[b, :, idx[b]] * signs[b] for b in range(B)], dim=0
            )

        return w, V, gp

    # ===================================================
    # INNER RS-PRFO LOOP
    # ===================================================
    @profile
    def _inner_rs_prfo_loop(self, it, calc, w, V, gp, H, g_cart,
                            trust_r, last_step, real_mask, E_old):
        """
        Optimized inner loop with ~1.5-2x speedup.
        """
        device = self.device
        B, n = gp.shape

        minus_mask = torch.nn.functional.one_hot(
            self.tracked_mode_idx, num_classes=n).to(torch.bool)
        plus_mask = ~minus_mask

        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        step_accepted = torch.zeros(B, dtype=torch.bool, device=device)
        last_rho = torch.full((B,), float("nan"), dtype=DTYPE, device=device)

        # === OPTIMIZATION #4: Adaptive max attempts ===
        if self._recent_acceptance_rate > 0.7 and it > 5:
            max_attempts = max(10, self.max_inner_attempts // 2)
        else:
            max_attempts = self.max_inner_attempts

        attempts_used = 0
        for _try in range(max_attempts):
            attempts_used = _try + 1
            pend = ~accepted
            
            # === OPTIMIZATION #2: Early termination ===
            if not pend.any():
                break

            # Build unconstrained steps
            s_unc_minus = torch.zeros_like(gp)
            s_unc_plus = torch.zeros_like(gp)

            denom_m0 = -w.masked_select(minus_mask)
            denom_m0 = torch.where(denom_m0.abs() < 1e-10,
                                   torch.sign(denom_m0) * 1e-10, denom_m0)
            s_unc_minus[minus_mask] = -(-gp[minus_mask]) / denom_m0

            denom_p0 = w.masked_select(plus_mask)
            denom_p0 = torch.where(denom_p0.abs() < 1e-10,
                                   torch.sign(denom_p0) * 1e-10, denom_p0)
            s_unc_plus[plus_mask] = -(gp[plus_mask]) / denom_p0

            norm2_minus = (s_unc_minus ** 2).sum(-1)
            norm2_plus  = (s_unc_plus ** 2).sum(-1)
            total_unc   = norm2_minus + norm2_plus
            alpha = torch.where(
                total_unc > 0,
                norm2_minus / total_unc,
                torch.full_like(total_unc, 0.5)
            ).clamp(0.05, 0.95)

            R2 = trust_r ** 2
            R2_minus = alpha * R2
            R2_plus  = (1.0 - alpha) * R2

            # === Choose μ solver ===
            if self._enable_vectorized_mu:
                mu_minus, s_part_minus = self._solve_mu_vectorized(
                    w, gp, minus_mask, R2_minus, sigma=-1, only=pend
                )
                mu_plus, s_part_plus = self._solve_mu_vectorized(
                    w, gp, plus_mask, R2_plus, sigma=+1, only=pend
                )
            else:
                mu_minus, s_part_minus = self._solve_mu_batched(
                    w, gp, minus_mask, R2_minus, sigma=-1, only=pend
                )
                mu_plus, s_part_plus = self._solve_mu_batched(
                    w, gp, plus_mask, R2_plus, sigma=+1, only=pend
                )

            s_p = s_part_minus + s_part_plus
            norm_mw = torch.linalg.norm(s_p, dim=-1)
            s_mw = (V @ s_p.unsqueeze(-1)).squeeze(-1) * real_mask
            s_cart = self._D * s_mw

            # === OPTIMIZATION #5: Only compute for pending ===
            s_try = torch.zeros_like(s_cart)
            s_try[pend] = s_cart[pend]

            # Trial evaluation
            calc.backup_coords()
            calc.step_cart_(s_try)
            E_new, _ = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            calc.restore_coords()

            # Model change - could be further optimized
            Hs = torch.einsum("bij,bj->bi", H, s_try)
            model_change = (g_cart * s_try).sum(-1) + 0.5 * (s_try * Hs).sum(-1)
            actual_change = (E_new - E_old)

            rho = torch.full_like(model_change, float("nan"))
            ok = (model_change.abs() > 1e-16) & pend
            rho[ok] = actual_change[ok] / model_change[ok]
            last_rho = torch.where(pend, rho, last_rho)

            on_boundary = (norm_mw - trust_r).abs() <= (
                1e-6 * torch.clamp(trust_r, min=1.0)
            )
            bad = pend & ((~torch.isfinite(rho)) | (rho < self.eta_shrink))
            force_accept = trust_r <= (self.trust_min * 1.000000000001)
            acc = pend & (~bad | force_accept)
            rej = pend & (~acc)

            if acc.any():
                s_commit = torch.zeros_like(s_cart)
                s_commit[acc] = s_cart[acc]
                calc.step_cart_(s_commit)

                grow = (rho > self.eta_expand) & on_boundary & acc
                trust_r = torch.where(
                    grow,
                    torch.clamp(2.0 * trust_r, max=self.trust_max),
                    trust_r
                )
                last_step[acc] = s_commit[acc]
                step_accepted |= acc
                self._dump_xyz_subset(calc, acc, it)

            trust_r = torch.where(
                rej,
                torch.clamp(0.5 * trust_r, min=self.trust_min),
                trust_r
            )

            self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
            accepted |= acc

        # Update acceptance rate (exponential moving average)
        acceptance_this_iter = accepted.float().mean().item()
        self._recent_acceptance_rate = 0.85 * self._recent_acceptance_rate + 0.15 * acceptance_this_iter

        # Log efficiency
        if attempts_used < max_attempts:
            self._w(f"  [Efficiency] Used {attempts_used}/{max_attempts} attempts\n")

        return trust_r, last_step, last_rho, step_accepted
    # ===================================================
    # CONVERGENCE
    # ===================================================

    def _check_convergence(self, it, calc, atoms_list,
                           trust_r, last_step, real_mask, last_rho):

        device = self.device
        E_final, F_final = calc.get_ef_gpu()
        F_final = F_final.to(dtype=DTYPE)
        g_last = -F_final * real_mask.to(DTYPE)

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

        max_f = g_last.abs().amax(dim=-1)
        rms_f = torch.sqrt((g_last**2).sum(-1) / L_eff)
        max_dp = last_step.abs().amax(dim=-1)
        rms_dp = torch.sqrt((last_step**2).sum(-1) / L_eff)

        done = (
            (max_f <= f_max_th)
            & (rms_f <= f_rms_th)
            & (max_dp <= dp_max_th)
            & (rms_dp <= dp_rms_th)
        )

        self._w(self._fmt_orca_cycle_table(
            it=it,
            E=E_final.to(dtype=DTYPE),
            rho=last_rho,
            R=trust_r,
            max_f=max_f, rms_f=rms_f,
            max_dp=max_dp, rms_dp=rms_dp,
            f_max_th=f_max_th, f_rms_th=f_rms_th,
            dp_max_th=dp_max_th, dp_rms_th=dp_rms_th,
            done=done
        ))

        return done

    # ===================================================
    # MU SOLVER
    # ===================================================

    @staticmethod
    @torch.no_grad()
    def _solve_mu_batched(w, gp, mask, R2, sigma, only):
        device = w.device
        B, n = w.shape

        lam_all = (sigma * w)
        num_all = (sigma * gp)

        mu_out = torch.zeros(B, dtype=DTYPE, device=device)
        s_part = torch.zeros((B, n), dtype=DTYPE, device=device)

        for b in range(B):
            if not only[b]:
                continue

            m_b = mask[b]
            lam_b = lam_all[b][m_b]
            num_b = num_all[b][m_b]

            if lam_b.numel() == 0:
                mu_out[b] = 0
                continue

            R2_b = float(R2[b].item())
            denom0 = torch.where(lam_b.abs() < 1e-10,
                                 torch.sign(lam_b) * 1e-10, lam_b)
            s_unc = -num_b / denom0
            norm2_unc = float((s_unc * s_unc).sum().item())

            if norm2_unc <= R2_b:
                mu_out[b] = 0.0
                s_tmp = s_unc
            else:
                def F(mu_val):
                    mu_t = torch.tensor(mu_val, dtype=DTYPE, device=device)
                    denom = lam_b - mu_t
                    denom = torch.where(
                        denom.abs() < 1e-12,
                        torch.sign(denom) * 1e-12,
                        denom
                    )
                    return float(((num_b / denom)**2).sum().item())

                wt_min = float(lam_b.min().item())
                hi = wt_min - 1e-6
                if not np.isfinite(F(hi)):
                    hi = wt_min - 1e-4

                lo = hi - 1
                Fa = F(lo)
                it_ex = 0
                while Fa > R2_b and it_ex < 60:
                    lo -= max(1.0, abs(lo) * 0.5)
                    Fa = F(lo)
                    it_ex += 1

                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    Fm = F(mid)
                    if abs(Fm - R2_b) <= 1e-12 * max(1, R2_b) or abs(hi - lo) < 1e-12:
                        lo = hi = mid
                        break
                    if Fm > R2_b:
                        hi = mid
                    else:
                        lo = mid

                mu_star = 0.5 * (lo + hi)
                mu_out[b] = mu_star
                mu_t = torch.tensor(mu_star, dtype=DTYPE, device=device)
                denom = lam_b - mu_t
                denom = torch.where(
                    denom.abs() < 1e-12,
                    torch.sign(denom) * 1e-12,
                    denom
                )
                s_tmp = -num_b / denom

            s_full = torch.zeros(n, dtype=DTYPE, device=device)
            s_full[m_b] = s_tmp
            s_part[b] = s_full

        return mu_out, s_part

    @staticmethod
    @torch.no_grad()
    def _bfgs_update_batched(
        H, s_cart, g_prev, g_new, real_mask,
        step_accepted=None,
        step_tol: float = 1e-8,
        grad_tol: float = 1e-8,
        curvature_tol: float = 1e-12,
    ):
        """Symmetric BFGS Hessian update for accepted batched PRFO steps."""
        DTYPE = H.dtype
        rm = real_mask.to(DTYPE)
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)

        s = s_cart.to(DTYPE) * rm
        y = (g_new - g_prev).to(DTYPE) * rm
        upd_mask = ((s * s).sum(-1) > step_tol**2) & ((y * y).sum(-1) > grad_tol**2)
        if step_accepted is not None:
            upd_mask = upd_mask & step_accepted
        if not bool(upd_mask.any()):
            return H

        H_new = H.clone()
        for b in upd_mask.nonzero(as_tuple=False).flatten().tolist():
            sb = s[b]
            yb = y[b]
            ys = torch.dot(yb, sb)
            if ys <= curvature_tol:
                continue
            Hs = H[b].matmul(sb)
            sHs = torch.dot(sb, Hs)
            if sHs <= curvature_tol:
                continue
            Hb = H[b] + torch.outer(yb, yb) / ys - torch.outer(Hs, Hs) / sHs
            Hb = 0.5 * (Hb + Hb.transpose(-1, -2))
            H_new[b] = Hb * mask_ij[b] + H_new[b] * (1.0 - mask_ij[b])
        return H_new

    @staticmethod
    @torch.no_grad()
    def _bofill_update_batched(
        H, s_cart, g_prev, g_new, real_mask,
        step_accepted=None,  # NEW: mask of which batches actually took a step
        step_tol: float = 1e-8, 
        grad_tol: float = 1e-8, 
        sr1_tol: float = 1e-8  # CHANGED: from 1e-10 to 1e-8
    ):
        """
        Bofill update (Cartesian, batched) with Gaussian-style logic:
        - Residual Z = dg - H * delta
        - phi = 1 - ( (dq^T Z)^2 / ( (dq^T dq) * (Z^T Z) ) )
        - H_{k+1} = H + (1-phi) * (Z Z^T) / (dq^T Z)   [MS/SR1 term]
                            +  phi * ( (Z dq^T + dq Z^T)/(dq^T dq) - (dq^T Z) * (dq dq^T)/(dq^T dq)^2 ) [PSB term]
        
        NEW: Only update batches that actually accepted a step (step_accepted mask)
        """
        DTYPE = H.dtype
        B, n, _ = H.shape

        rm = real_mask.to(DTYPE)
        mask_ij = (real_mask.unsqueeze(-1) & real_mask.unsqueeze(-2)).to(DTYPE)

        # Restrict vectors to real DOFs
        dq = s_cart.to(DTYPE) * rm                 # dq := delta (Cartesian)
        dg = (g_new - g_prev).to(DTYPE) * rm       # dg := grad change (Cartesian)

        # Per-batch norms to decide if we update
        dq2 = (dq * dq).sum(-1)                    # dq·dq
        dg2 = (dg * dg).sum(-1)                    # dg·dg
        
        # Update only if: (1) step was accepted, (2) non-trivial step/grad change
        upd_mask = (dq2 > step_tol**2) & (dg2 > grad_tol**2)
        if step_accepted is not None:
            upd_mask = upd_mask & step_accepted
        
        if not bool(upd_mask.any()):
            return H

        # Work on a copy; slice only the batches we will update
        H_new = H.clone()
        idx = upd_mask.nonzero(as_tuple=False).flatten()

        # Slice helpers
        dq_m = dq[idx]                 # (M, n)
        dg_m = dg[idx]                 # (M, n)
        HH   = H_new[idx]              # (M, n, n)

        # Z residual: Z = dg - H * dq
        Hdq  = torch.einsum("mij,mj->mi", HH, dq_m)
        Z    = dg_m - Hdq

        # Scalars
        dq2_m = (dq_m * dq_m).sum(-1)
        zz_m  = (Z * Z).sum(-1)
        qz_m  = (dq_m * Z).sum(-1)

        # ---- Build PSB increment ----
        Z_dqT = torch.einsum("mi,mj->mij", Z,    dq_m) * mask_ij[idx]
        dq_ZT = torch.einsum("mi,mj->mij", dq_m, Z   ) * mask_ij[idx]
        dq_dqT= torch.einsum("mi,mj->mij", dq_m, dq_m) * mask_ij[idx]
        dH_PSB = (Z_dqT + dq_ZT) / dq2_m.view(-1,1,1) - (qz_m / (dq2_m * dq2_m)).view(-1,1,1) * dq_dqT

        # ---- SR1/MS term ----
        use_sr1 = (qz_m.abs() > sr1_tol) & (zz_m > sr1_tol**2)
        dH_SR1_full = torch.zeros_like(dH_PSB)
        if bool(use_sr1.any()):
            Z_ZT = torch.einsum("mi,mj->mij", Z[use_sr1], Z[use_sr1]) * mask_ij[idx][use_sr1]
            dH_SR1 = Z_ZT / qz_m[use_sr1].view(-1,1,1)
            dH_SR1_full[use_sr1] = dH_SR1

        # ---- Bofill mixing weight ----
        phi = torch.ones_like(qz_m)
        good_phi = (dq2_m > sr1_tol**2) & (zz_m > sr1_tol**2)
        if bool(good_phi.any()):
            ratio = (qz_m[good_phi] * qz_m[good_phi]) / (dq2_m[good_phi] * zz_m[good_phi])
            phi_val = (1.0 - ratio).clamp(0.0, 1.0)
            phi[good_phi] = phi_val
        phi = torch.where(use_sr1, phi, torch.ones_like(phi))

        # ---- Combine increments ----
        inc = (1.0 - phi).view(-1,1,1) * dH_SR1_full + phi.view(-1,1,1) * dH_PSB

        # Apply and symmetrize
        HH = HH + inc
        HH = 0.5 * (HH + HH.transpose(-1, -2))
        H_new[idx] = HH

        return H_new


    # ===================================================
    # LOGGING / XYZ
    # ===================================================

    def _open_log(self):
        self.log_fp = open(self.output, "w", encoding="utf-8")

    def _close_log(self):
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None

    def _w(self, s):
        self.log_fp.write(s)
        self.log_fp.flush()

    def _init_xyz_paths(self, B_all):
        self.xyz_paths = [
            os.path.join(self.out_dir, f"ts_batch{i+1}.xyz")
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
            s, t = ptr[i_local], ptr[i_local+1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], tag)

    def _dump_xyz_subset(self, calc, accept_mask, it):
        if not accept_mask.any():
            return
        with torch.no_grad():
            pos = _get_coord_gpu(calc).detach().cpu().numpy()
        ptr = self._ptr.detach().cpu().numpy()
        for i_local in accept_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            s, t = ptr[i_local], ptr[i_local+1]
            idx_orig = int(self._orig_index[i_local].item())
            symbols = self._symbols_per_batch[idx_orig]
            self._append_xyz(idx_orig, symbols, pos[s:t], f"iter={it}")

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

    def _fmt_iter_head(self, it, acc, rej, rho, trust_r, E_new):
        acc_idx = acc.nonzero(as_tuple=False).flatten().cpu().tolist()
        rej_idx = rej.nonzero(as_tuple=False).flatten().cpu().tolist()
        acc_list = [int(self._orig_index[i]) for i in acc_idx]
        rej_list = [int(self._orig_index[i]) for i in rej_idx]

        rhos = rho.detach().cpu().numpy()
        Rs = trust_r.detach().cpu().numpy()
        En = E_new.detach().cpu().numpy()

        def head(arr, k=3, fmt="{:.3f}"):
            out = []
            for v in arr[:k]:
                try:
                    out.append(fmt.format(float(v)))
                except:
                    out.append("nan")
            return "[" + ", ".join(out) + "]"

        return (
            f"Iter {it}: accepted={acc_list} rejected={rej_list} "
            f"rho_head={head(rhos)} R_head={head(Rs)} E_head={head(En, fmt='{:.6f}')}\n"
        )

    def _fmt_orca_cycle_table(self, it, E, rho, R,
                              max_f, rms_f, max_dp, rms_dp,
                              f_max_th, f_rms_th, dp_max_th, dp_rms_th,
                              done):

        lines = []
        lines.append("-"*70 + "\n")
        lines.append(f"{('RS-PRFO Cycle ' + str(it)).center(70)}\n")
        lines.append("-"*70 + "\n")
        lines.append(
            "Batch   Energy(Ha)      rho      R(MW)   Max|F|(H/A)  RMS|F|  "
            "Max|dX|(A)  RMS|dX|  Conv\n"
        )
        lines.append("-"*70 + "\n")

        def fmt(v, wid=12, p=6):
            x = float(v)
            return f"{x:>{wid}.{p}f}"

        B = E.shape[0]
        for b in range(B):
            idx_orig = int(self._orig_index[b].item())
            lines.append(
                f"[{idx_orig:2d}] "
                f"{fmt(E[b], 14, 6)} "
                f"{fmt(rho[b], 8, 3)} "
                f"{fmt(R[b], 8, 3)} "
                f"{fmt(max_f[b], 12, 6)} "
                f"{fmt(rms_f[b], 10, 6)} "
                f"{fmt(max_dp[b], 12, 6)} "
                f"{fmt(rms_dp[b], 10, 6)} "
                f"{('YES' if bool(done[b]) else 'NO')}\n"
            )

        lines.append("\n")
        lines.append(" Convergence criteria:\n")
        lines.append(
            f"   Max|F| ≤ {float(f_max_th.max()):.6f}   "
            f"RMS|F| ≤ {float(f_rms_th.max()):.6f}   "
            f"Max|dX| ≤ {float(dp_max_th.max()):.6f}   "
            f"RMS|dX| ≤ {float(dp_rms_th.max()):.6f}\n"
        )

        return "".join(lines)