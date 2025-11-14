# -*- coding: utf-8 -*-
"""
BatchPRFO with fixed padded dimension nmax.
All EFH are padded to nmax determined from FIRST prepare/get_ef_gpu.
Calculator.prepare() is called with fixed_nmax afterwards.
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
                 max_inner_attempts: int = 8,
                 max_outer_iter: int = 256,
                 device: str = "cuda"):
        self.trust_init = trust_init
        self.trust_min = trust_min
        self.trust_max = trust_max
        self.eta_shrink = eta_shrink
        self.eta_expand = eta_expand
        self.max_inner_attempts = max_inner_attempts
        self.max_outer_iter = max_outer_iter
        self.device = torch.device(device)

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

    # ===================================================
    # PUBLIC RUN
    # ===================================================

    #@profile
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

        self._init_xyz_paths(B0)
        self._symbols_per_batch = _symbols_flat(atoms_list)

        # =====================================================
        # FIRST PREPARE → GET FIXED nmax
        # =====================================================
        calc.prepare(atoms_list)         # ← FIRST (NO fixed_nmax)
        _, F0 = calc.get_ef_gpu()
        self._nmax = int(F0.shape[1])    # fixed padded DOF
        self._arange_n = torch.arange(self._nmax, device=device)

        # Build topology (nmax is now fixed)
        self._rebuild_topology(atoms_list)

        self._dump_xyz_all(calc, atoms_list, tag="init")

        B = self._B
        trust_r = torch.full((B,), self.trust_init, dtype=DTYPE, device=device)
        last_step = torch.zeros((B, self._nmax), dtype=DTYPE, device=device)

        self.tracked_mode_vec_mw = None
        self.tracked_mode_idx = None

        outer_it = 0

        # =====================================================
        # OUTER LOOP
        # =====================================================
        while outer_it < self.max_outer_iter and len(atoms_list) > 0:
            outer_it += 1
            real_mask = self._real_mask

            calc.backup_coords()

            # EFH padded to fixed self._nmax
            E_old, F_raw, H_raw = self._compute_efh(calc)

            H, g_cart = self._build_cartesian_hg(F_raw, H_raw, real_mask)
            H_mw, g_mw = self._mass_weight_hg(H, g_cart, real_mask)

            w, V, gp = self._eigh_and_track_modes(H_mw, g_mw)

            trust_r, last_step, last_rho = self._inner_rs_prfo_loop(
                it=outer_it,
                calc=calc,
                w=w, V=V, gp=gp,
                H=H, g_cart=g_cart,
                trust_r=trust_r,
                last_step=last_step,
                real_mask=real_mask,
                E_old=E_old,
            )

            done = self._check_convergence(
                it=outer_it,
                calc=calc,
                atoms_list=atoms_list,
                trust_r=trust_r,
                last_step=last_step,
                real_mask=real_mask,
                last_rho=last_rho,
            )

            for i_local in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                idx_orig = int(self._orig_index[i_local].item())
                self._w(f">>> Batch {idx_orig} converged at cycle {outer_it}\n")

            if bool(done.all()):
                self._w("# Normal termination (all converged)\n")
                break

            # =====================================================
            # Dynamic batch shrinking (BUT nmax remains fixed)
            # =====================================================
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

                # ------------------------------
                # CRITICAL CHANGE:
                # re-prepare MUST use fixed_nmax
                # ------------------------------
                calc.prepare(atoms_list, fixed_nmax=self._nmax)

                # rebuild masks/topology (nmax DO NOT change)
                self._rebuild_topology(atoms_list)

            B = self._B

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

    def _inner_rs_prfo_loop(self, it, calc, w, V, gp, H, g_cart,
                            trust_r, last_step, real_mask, E_old):

        device = self.device
        B, n = gp.shape

        minus_mask = torch.nn.functional.one_hot(
            self.tracked_mode_idx, num_classes=n).to(torch.bool)
        plus_mask = ~minus_mask

        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        last_rho = torch.full((B,), float("nan"), dtype=DTYPE, device=device)

        for _try in range(self.max_inner_attempts):
            pend = ~accepted
            if not pend.any():
                break

            # ——— Build unconstrained steps in eigenbasis (RS-PRFO signed)
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

            # ——— Solve μ for RS-PRFO subspace updates
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

            s_try = torch.zeros_like(s_cart)
            s_try[pend] = s_cart[pend]

            # trial
            calc.backup_coords()
            calc.step_cart_(s_try)
            E_new, _ = calc.get_ef_gpu()
            E_new = E_new.to(dtype=DTYPE)
            calc.restore_coords()

            Hs = torch.einsum("bij,bj->bi", H, s_try)
            model_change = (g_cart * s_try).sum(-1) + 0.5 * (s_try * Hs).sum(-1)
            actual_change = (E_new - E_old)

            rho = torch.full_like(model_change, float("nan"))
            ok = model_change.abs() > 1e-16
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
                self._dump_xyz_subset(calc, acc, it)

            trust_r = torch.where(
                rej,
                torch.clamp(0.5 * trust_r, min=self.trust_min),
                trust_r
            )

            self._w(self._fmt_iter_head(it, acc, rej, rho, trust_r, E_new))
            accepted |= acc

        return trust_r, last_step, last_rho

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
