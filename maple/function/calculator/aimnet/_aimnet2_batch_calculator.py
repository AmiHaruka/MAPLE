# -*- coding: utf-8 -*-
"""Out-of-scope for the unified MAPLE calculator protocol.

Consumed by BatchLBFGS only; does not implement the CalcABC protocol
(`_finalize_results`, `_analytic_hessian`, `MODEL_*` class attrs). Keep
self-contained until a future commit retrofits the batch path.
"""
import torch
from typing import List
from ase import Atoms
import numpy as np

EH2EV = 27.211386245988


def pad_dim0(a: torch.Tensor, value=0) -> torch.Tensor:
    pad_shape = list(a.shape); pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)


def nblist_dense_padded_multi(coord: torch.Tensor, mol_idx: torch.Tensor, cutoff: float) -> torch.Tensor:
    device = coord.device
    dtype  = coord.dtype
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int64, device=device)

    diff  = coord[:, None, :] - coord[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    same  = (mol_idx[:, None] == mol_idx[None, :])
    eye   = torch.eye(N, dtype=torch.bool, device=device)
    mask  = (dist2 <= cutoff * cutoff) & same & (~eye)

    deg = mask.sum(dim=1)
    M   = int(max(int(deg.max().item()), 1))

    big = torch.finfo(dtype).max / 4.0
    sort_key = torch.where(mask, dist2, dist2.new_full(dist2.shape, big))
    order = torch.argsort(sort_key, dim=1, stable=True)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int64, device=device)
    for i in range(N):
        ki = int(deg[i].item())
        if ki > 0:
            k = min(ki, M)
            nbmat[i, :k] = order[i, :k]
    return nbmat


def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


class AIMNet2BatchCalc:
    """
    AIMNet2 batch calculator.
    """

    def __init__(self, model_path: str, device: str = "cuda", cutoff: float = 5.0, dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype  = dtype
        self.model  = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.cutoff = float(cutoff)

        # prepare-related internal buffers
        self._prepared    = False
        self._atoms_B     = 0
        self._ptr         = None
        self.numbers      = None
        self.mol_idx      = None
        self.coord        = None
        self.N_atoms      = 0
        self.Nmax_atoms   = 0
        self.nmax_dof     = 0   # <<< will be overridden if fixed_nmax is provided
        self.sentinel_mol = 0
        self.charge       = None

        self._coord_backup = None

    # -------------------------------------------------------------------------
    # prepare() modified to accept fixed_nmax
    # -------------------------------------------------------------------------
    def prepare(self, atoms_list: List[Atoms], fixed_nmax: int = None):
        """
        Prepare topology & initial coordinates.
        If fixed_nmax is provided, we override the internal nmax_dof so that
        PRFO and calculator share the same padded DOF size.

        This is crucial for compatibility with batch-PRFO, where PRFO wants
        a fixed padded size (self._nmax) across all iterations.
        """
        device, dtype = self.device, self.dtype
        self._atoms_B = len(atoms_list)
        self._ptr     = _ptr_from_atoms(atoms_list, device)

        nums, mids = [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int64, device=device)
            n = Z.shape[0]
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int64, device=device))

        self.numbers = torch.cat(nums, dim=0) if nums else torch.zeros((0,), dtype=torch.int64, device=device)
        self.mol_idx = torch.cat(mids, dim=0) if mids else torch.zeros((0,), dtype=torch.int64, device=device)

        self.N_atoms     = int(self.numbers.numel())
        self.Nmax_atoms  = int(max((len(at) for at in atoms_list), default=0))

        # ---------------------------- MODIFICATION ----------------------------
        # If PRFO supplies a fixed_nmax (padded 3*Nmax from first iteration),
        # we MUST adopt that dimension for the calculator too.
        #
        # Otherwise step_cart_() will fail with shape mismatch: PRFO passes
        # (B, fixed_nmax) but calculator expects (B, 3*Nmax_atoms_current).
        # ----------------------------------------------------------------------
        if fixed_nmax is None:
            # normal behavior (first prepare call)
            self.nmax_dof = 3 * self.Nmax_atoms
        else:
            # PRFO-defined padded dimension
            self.nmax_dof = int(fixed_nmax)
        # ----------------------------------------------------------------------

        if self.N_atoms > 0:
            pos_list = [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list]
            coord0   = torch.cat(pos_list, dim=0)
        else:
            coord0   = torch.zeros((0, 3), dtype=dtype)

        self.coord = coord0.to(device, non_blocking=True).contiguous()

        self.sentinel_mol = (int(self.mol_idx.max().item()) + 1) if self.N_atoms > 0 else 0
        self.charge       = torch.zeros(self._atoms_B + 1, dtype=dtype, device=device)

        self._coord_backup = None
        self._prepared     = True

    # -------------------------------------------------------------------------
    # coordinate update
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def step_cart_(self, s_cart: torch.Tensor):
        """
        s_cart: (B, nmax_dof). MUST match self.nmax_dof (PRFO padded).
        """
        assert self._prepared, "call prepare() first"

        B = self._atoms_B

        # ---------------------------- MODIFICATION ----------------------------
        # PRFO enforces a fixed padded DOF; here we enforce the same.
        # ----------------------------------------------------------------------
        assert s_cart.shape == (B, self.nmax_dof), \
            f"step_cart_ expects (B,{self.nmax_dof}), got {tuple(s_cart.shape)}"
        # ----------------------------------------------------------------------

        s_cart = s_cart.to(self.device, dtype=self.dtype)
        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                self.coord[s[i]:t[i], :].add_(s_cart[i, :3*ni].reshape(ni, 3))

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        assert self._prepared, "call prepare() first"
        assert coord.shape == (self.N_atoms, 3)
        self.coord.copy_(coord.to(self.device, dtype=self.dtype))

    @torch.no_grad()
    def backup_coords(self):
        if self._prepared:
            self._coord_backup = self.coord.clone()

    @torch.no_grad()
    def restore_coords(self):
        if self._coord_backup is not None:
            self.coord.copy_(self._coord_backup)
            self._coord_backup = None

    # -------------------------------------------------------------------------
    # forward (unchanged)
    # -------------------------------------------------------------------------
    def _forward_energy_forces_(self, c: torch.Tensor, need_graph: bool):
        assert self._prepared, "call prepare() first"
        device, dtype = self.device, self.dtype
        B = self._atoms_B
        N = self.N_atoms

        coord_leaf = c.detach().to(device=device, dtype=dtype).requires_grad_(True)
        nbmat = nblist_dense_padded_multi(coord_leaf, self.mol_idx, self.cutoff)

        data = {
            "coord":    pad_dim0(coord_leaf, 0.0),
            "numbers":  pad_dim0(self.numbers, 0).to(torch.int64),
            "charge":   self.charge,
            "mol_idx":  pad_dim0(self.mol_idx, self.sentinel_mol).to(torch.int64),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat,
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)

        e_vec = out["energy"].to(dtype).reshape(-1)
        if e_vec.numel() == N + 1 or e_vec.numel() == B + 1:
            e_vec = e_vec[:-1]

        if e_vec.numel() == B:
            E_eV = e_vec
        elif e_vec.numel() == N:
            E_eV = torch.bincount(self.mol_idx, weights=e_vec, minlength=B).to(dtype)
        else:
            raise RuntimeError(f"Unexpected energy shape {tuple(e_vec.shape)}")

        grad = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        F_all_eV = -grad

        return E_eV, F_all_eV, coord_leaf

    # -------------------------------------------------------------------------
    # get_ef_gpu() and get_efh_gpu(): no structural change needed.
    # They already use self.nmax_dof for padding, which now matches PRFO.
    # -------------------------------------------------------------------------
    def get_ef_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device))

        E_eV, F_all_eV, _ = self._forward_energy_forces_(self.coord, need_graph=False)

        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        s = self._ptr[:-1]; t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                F_eV[i, :3*ni] = F_all_eV[s[i]:t[i], :].reshape(-1)

        return E_eV / EH2EV, F_eV / EH2EV

    def get_efh_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        if B == 0:
            return (torch.zeros((0,), dtype=dtype, device=device),
                    torch.zeros((0, 0), dtype=dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.int64, device=device))

        E_eV, F_all_eV, coord_leaf = self._forward_energy_forces_(self.coord, need_graph=True)

        f_flat = F_all_eV.reshape(-1)
        cols = []
        for k in range(f_flat.numel()):
            g2 = torch.autograd.grad(f_flat[k], coord_leaf,
                                     retain_graph=True, create_graph=False)[0]
            cols.append(g2.reshape(-1))
        H_global_eV = -torch.stack(cols, dim=1)
        H_global_eV = 0.5 * (H_global_eV + H_global_eV.transpose(0, 1))

        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=dtype, device=device)
        H_eV = torch.zeros((B, nmax, nmax), dtype=dtype, device=device)
        P    = torch.empty((B,), dtype=torch.int64, device=device)

        s = self._ptr[:-1]; t = self._ptr[1:]
        for i in range(B):
            ni  = int((t[i] - s[i]).item())
            dof = 3 * ni
            P[i] = self.Nmax_atoms - ni
            if dof > 0:
                F_eV[i, :dof]       = F_all_eV[s[i]:t[i], :].reshape(-1)
                H_eV[i, :dof, :dof] = H_global_eV[3*s[i]:3*t[i], 3*s[i]:3*t[i]]

        return (E_eV / EH2EV,
                F_eV / EH2EV,
                H_eV / EH2EV,
                P)
