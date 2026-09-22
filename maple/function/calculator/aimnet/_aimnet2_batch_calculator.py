# -*- coding: utf-8 -*-
"""Out-of-scope for the unified MAPLE calculator protocol.

Consumed by the batch optimizers (BatchLBFGS/BatchPRFO); does not implement
the CalcABC protocol (`_finalize_results`, `_analytic_hessian`, `MODEL_*`
class attrs). Build it from an already-initialized single-molecule
AIMNet2Calculator via `from_ase_calculator` so the jit model is loaded once.
"""
import torch
from typing import List
from ase import Atoms
import numpy as np

from .._batch_eval import (
    ALL_BATCH_SIZE,
    AUTO_BATCH_SIZE,
    _AutoBatchSizer,
    _calculator_batch_size,
    _is_cuda_oom,
)
from ..electronic_state import requested_electronic_state
from ._aimnet2_calculator import (
    AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
    build_aimnet2_neighbor_matrices,
    identify_aimnet2_checkpoint_capabilities,
    pad_dim0,
)

EH2EV = 27.211386245988

def _ptr_from_atoms(atoms_list: List[Atoms], device) -> torch.Tensor:
    ptr = [0]
    for at in atoms_list:
        ptr.append(ptr[-1] + len(at))
    return torch.tensor(ptr, dtype=torch.long, device=device)


class AIMNet2BatchCalc:
    """
    AIMNet2 batch calculator.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        cutoff: float = 5.0,
        cutoff_lr: float = float("inf"),
        dtype: torch.dtype | None = None,
        num_charge_channels: int | None = None,
        batch_size=None,
        auto_batch_hard_cap: int = 8,
        model=None,
        batch_energy_layout: str | None = None,
    ):
        self.device = torch.device(device)
        capabilities = None
        if model_path is not None:
            capabilities = identify_aimnet2_checkpoint_capabilities(model_path)
            identified_layout = (
                None if capabilities is None else capabilities.batch_energy_layout
            )
            if batch_energy_layout is None:
                batch_energy_layout = identified_layout
            elif batch_energy_layout != identified_layout:
                raise ValueError(
                    "Explicit AIMNet2 batch schema does not match the "
                    "checkpoint SHA256 identity"
                )
        if batch_energy_layout != AIMNET2_PADDED_PER_MOLECULE_LAYOUT:
            raise NotImplementedError(
                "AIMNet2BatchCalc requires a checkpoint with the validated "
                f"{AIMNET2_PADDED_PER_MOLECULE_LAYOUT!r} output schema"
            )
        self.batch_energy_layout = batch_energy_layout
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.jit.load(model_path, map_location=self.device).eval()
        else:
            raise ValueError("AIMNet2BatchCalc requires either model_path or a preloaded model.")
        for p in self.model.parameters():
            p.requires_grad_(False)

        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        inherited_dtype = (
            model_dtype if capabilities is None else capabilities.input_dtype
        )
        if dtype is None:
            dtype = inherited_dtype
        elif dtype != inherited_dtype:
            raise ValueError(
                "AIMNet2BatchCalc input dtype must match the checkpoint; "
                f"expected {inherited_dtype}, got {dtype}"
            )
        if model_dtype != dtype:
            raise RuntimeError(
                "AIMNet2 model parameter dtype does not match batch input dtype"
            )
        self.dtype = dtype

        model_charge_channels = int(
            getattr(self.model, "num_charge_channels", 1)
        )
        if num_charge_channels is None:
            num_charge_channels = (
                model_charge_channels
                if capabilities is None
                else capabilities.num_charge_channels
            )
        if int(num_charge_channels) != model_charge_channels:
            raise RuntimeError(
                "AIMNet2 checkpoint charge-channel count does not match the "
                "batch capability contract"
            )
        self.num_charge_channels = int(num_charge_channels)
        self.supports_multiplicity = self.num_charge_channels == 2
        self.cutoff = float(cutoff)
        self.cutoff_lr = float(cutoff_lr)
        self.batch_size = batch_size
        self.auto_batch_hard_cap = auto_batch_hard_cap
        self.batch_memory_model = "concat_dense_neighbor"

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
        self.mult         = None
        self._atoms_list  = []

        self._coord_backup = None

    @classmethod
    def from_ase_calculator(cls, calc, dtype: torch.dtype | None = None):
        """Share the jit model already loaded by a single-molecule AIMNet2Calculator.

        Implicit solvation corrections are per-molecule post-processing in the
        ASE wrapper and are not applied on the batch path, so refuse them here
        rather than silently dropping the correction.
        """
        if getattr(calc, "solvent_correction", None) is not None:
            raise NotImplementedError(
                "AIMNet2BatchCalc does not support implicit solvation; "
                "remove #solv(...) or run structures one at a time."
            )
        if not getattr(calc, "supports_batch_energy_forces", False):
            raise NotImplementedError(
                "AIMNet2 native optimization batching is disabled for an "
                "unrecognized checkpoint; run structures one at a time."
            )
        input_dtype = getattr(calc, "input_dtype", torch.float32)
        if dtype is not None and dtype != input_dtype:
            raise ValueError(
                "AIMNet2BatchCalc must inherit calc.input_dtype exactly; "
                f"expected {input_dtype}, got {dtype}"
            )
        return cls(
            model=calc.model,
            device=calc.device,
            cutoff=calc.cutoff,
            cutoff_lr=calc.cutoff_lr,
            dtype=input_dtype,
            num_charge_channels=getattr(calc, "num_charge_channels", 1),
            batch_size=getattr(calc, "batch_size", None),
            auto_batch_hard_cap=getattr(calc, "auto_batch_hard_cap", 8),
            batch_energy_layout=getattr(calc, "_batch_energy_layout", None),
        )

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
        self._atoms_list = list(atoms_list)
        if any(bool(np.any(getattr(at, "pbc", False))) for at in atoms_list):
            raise NotImplementedError(
                "AIMNet2BatchCalc is a no-PBC batch wrapper; use a validated "
                "periodic backend for periodic systems."
            )

        if any(len(at) == 0 for at in atoms_list):
            raise ValueError("AIMNet2BatchCalc does not support empty structures.")
        self._atoms_B = len(atoms_list)
        self._ptr     = _ptr_from_atoms(atoms_list, device)

        nums, mids = [], []
        for i, at in enumerate(atoms_list):
            Z = torch.tensor(at.get_atomic_numbers(), dtype=torch.int32, device=device)
            n = Z.shape[0]
            nums.append(Z)
            mids.append(torch.full((n,), i, dtype=torch.int32, device=device))

        self.numbers = torch.cat(nums, dim=0) if nums else torch.zeros((0,), dtype=torch.int32, device=device)
        self.mol_idx = torch.cat(mids, dim=0) if mids else torch.zeros((0,), dtype=torch.int32, device=device)

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
            required_dof = 3 * self.Nmax_atoms
            if self.nmax_dof < required_dof:
                raise ValueError(
                    f"fixed_nmax={self.nmax_dof} is too small for the current batch; "
                    f"need at least {required_dof} Cartesian DOFs."
                )
            if self.nmax_dof % 3 != 0:
                raise ValueError(
                    f"fixed_nmax={self.nmax_dof} is not a multiple of 3 Cartesian DOFs."
                )
        # ----------------------------------------------------------------------

        if self.N_atoms > 0:
            pos_list = [torch.tensor(at.get_positions(), dtype=dtype) for at in atoms_list]
            coord0   = torch.cat(pos_list, dim=0)
            if not bool(torch.isfinite(coord0).all().item()):
                raise ValueError(
                    "AIMNet2BatchCalc input coordinates must be finite."
                )
        else:
            coord0   = torch.zeros((0, 3), dtype=dtype)

        self.coord = coord0.to(device, non_blocking=True).contiguous()

        self.sentinel_mol = (int(self.mol_idx.max().item()) + 1) if self.N_atoms > 0 else 0

        # Per-molecule charge/multiplicity; trailing entries belong to the
        # sentinel pad molecule.
        states = [requested_electronic_state(at) for at in atoms_list]
        charges = [charge for charge, _ in states]
        mults = [mult for _, mult in states]
        if not self.supports_multiplicity and any(m != 1.0 for m in mults):
            raise NotImplementedError(
                "The closed-shell AIMNet2 checkpoint only supports mult=1; "
                "use AIMNet2-NSE for open-shell structures."
            )
        self.charge = torch.tensor(charges + [0.0], dtype=dtype, device=device)
        self.mult = torch.tensor(mults + [1.0], dtype=dtype, device=device)

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
        if not self._prepared:
            raise RuntimeError("call prepare() before step_cart_()")

        B = self._atoms_B

        # ---------------------------- MODIFICATION ----------------------------
        # PRFO enforces a fixed padded DOF; here we enforce the same.
        # ----------------------------------------------------------------------
        expected = (B, self.nmax_dof)
        if tuple(s_cart.shape) != expected:
            raise ValueError(
                f"step_cart_ expects {expected}, got {tuple(s_cart.shape)}"
            )
        # ----------------------------------------------------------------------

        s_cart = s_cart.to(self.device, dtype=self.dtype)
        if not bool(torch.isfinite(s_cart).all().item()):
            raise FloatingPointError(
                "AIMNet2BatchCalc received a non-finite Cartesian step."
            )
        s = self._ptr[:-1]
        t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                self.coord[s[i]:t[i], :].add_(s_cart[i, :3*ni].reshape(ni, 3))

    @torch.no_grad()
    def set_coords_(self, coord: torch.Tensor):
        if not self._prepared:
            raise RuntimeError("call prepare() before set_coords_()")
        expected = (self.N_atoms, 3)
        if tuple(coord.shape) != expected:
            raise ValueError(f"set_coords_ expects {expected}, got {tuple(coord.shape)}")
        coord = coord.to(self.device, dtype=self.dtype)
        if not bool(torch.isfinite(coord).all().item()):
            raise FloatingPointError(
                "AIMNet2BatchCalc received non-finite coordinates."
            )
        self.coord.copy_(coord)

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
        return self._forward_energy_forces_data(
            c,
            self.numbers,
            self.mol_idx,
            self.charge,
            self.mult,
            batch_size=self._atoms_B,
            need_graph=need_graph,
        )

    def _forward_energy_forces_data(
        self,
        c: torch.Tensor,
        numbers: torch.Tensor,
        mol_idx: torch.Tensor,
        charge: torch.Tensor,
        mult: torch.Tensor,
        *,
        batch_size: int,
        need_graph: bool,
    ):
        if not self._prepared:
            raise RuntimeError("call prepare() before evaluating energy/forces")
        device, dtype = self.device, self.dtype

        coord_leaf = c.detach().to(device=device, dtype=dtype).requires_grad_(True)
        nbmat, nbmat_lr = build_aimnet2_neighbor_matrices(
            coord_leaf,
            mol_idx,
            cutoff=self.cutoff,
            cutoff_lr=self.cutoff_lr,
        )

        data = {
            "coord":    pad_dim0(coord_leaf, 0.0),
            "numbers":  pad_dim0(numbers, 0).to(torch.int32),
            "charge":   charge,
            "mult":     mult,
            "mol_idx":  pad_dim0(mol_idx, batch_size).to(torch.int32),
            "nbmat":    nbmat,
            "nbmat_lr": nbmat_lr,
            "cutoff_lr": torch.tensor(
                self.cutoff_lr,
                dtype=dtype,
                device=device,
            ),
        }

        with torch.jit.optimized_execution(False):
            out = self.model(data)

        # Packaged checkpoints accumulate molecular energies in float64 even
        # though coordinates/weights are float32. Preserve that output dtype;
        # casting it back to the input dtype loses several micro-Hartree.
        e_vec = out["energy"].reshape(-1)
        if self.batch_energy_layout != AIMNET2_PADDED_PER_MOLECULE_LAYOUT:
            raise RuntimeError(
                "AIMNet2 batch output schema is not validated for this checkpoint"
            )
        expected = batch_size + 1
        if e_vec.numel() != expected:
            raise RuntimeError(
                "AIMNet2 checkpoint must return the padded per-molecule energy "
                f"layout with {expected} entries for batch size {batch_size}; "
                f"got {e_vec.numel()} entries"
            )
        if not bool(torch.isfinite(e_vec).all().item()):
            raise FloatingPointError(
                "AIMNet2 returned non-finite padded per-molecule energies"
            )
        E_eV = e_vec[:-1]

        grad = torch.autograd.grad(E_eV.sum(), coord_leaf,
                                   create_graph=need_graph, retain_graph=need_graph)[0]
        F_all_eV = -grad
        if not bool(torch.isfinite(F_all_eV).all().item()):
            raise FloatingPointError("AIMNet2 returned non-finite forces")

        return E_eV, F_all_eV, coord_leaf

    # -------------------------------------------------------------------------
    # get_ef_gpu() and get_efh_gpu(): no structural change needed.
    # They already use self.nmax_dof for padding, which now matches PRFO.
    # -------------------------------------------------------------------------
    def get_ef_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        result_dtype = torch.float64
        if B == 0:
            return (torch.zeros((0,), dtype=result_dtype, device=device),
                    torch.zeros((0, 0), dtype=result_dtype, device=device))

        E_eV, F_all_eV = self._forward_energy_forces_chunked()

        nmax = self.nmax_dof
        F_eV = torch.zeros((B, nmax), dtype=result_dtype, device=device)
        s = self._ptr[:-1]; t = self._ptr[1:]
        for i in range(B):
            ni = int((t[i] - s[i]).item())
            if ni > 0:
                F_eV[i, :3*ni] = F_all_eV[s[i]:t[i], :].reshape(-1)

        energies = E_eV.to(result_dtype) / EH2EV
        forces = F_eV / EH2EV
        if (
            not bool(torch.isfinite(energies).all().item())
            or not bool(torch.isfinite(forces).all().item())
        ):
            raise FloatingPointError(
                "AIMNet2BatchCalc produced non-finite energies or forces."
            )
        return energies, forces

    def _forward_energy_forces_chunked(self):
        """Run optimizer E/F in ordered molecule chunks with CUDA-OOM backoff."""
        B = self._atoms_B
        setting = _calculator_batch_size(self, "optimization_batch_size")
        sizer = _AutoBatchSizer(
            self,
            self._atoms_list,
            ("energy", "forces"),
            kind="optimization",
        )
        if setting == AUTO_BATCH_SIZE:
            chunk = sizer.chunk
        elif setting == ALL_BATCH_SIZE:
            chunk = B
        else:
            chunk = int(setting)
        sizer.chunk = max(1, min(chunk, B))
        self._auto_batch_size_last = sizer.chunk

        energies = []
        forces = []
        start = 0
        while start < B:
            stop = min(B, start + sizer.chunk)
            atom_start = int(self._ptr[start].item())
            atom_stop = int(self._ptr[stop].item())
            local_mol_idx = self.mol_idx[atom_start:atom_stop] - int(start)
            local_charge = torch.cat(
                [
                    self.charge[start:stop],
                    self.charge.new_zeros(1),
                ]
            )
            local_mult = torch.cat(
                [
                    self.mult[start:stop],
                    self.mult.new_ones(1),
                ]
            )
            try:
                energy, force, _ = self._forward_energy_forces_data(
                    self.coord[atom_start:atom_stop],
                    self.numbers[atom_start:atom_stop],
                    local_mol_idx,
                    local_charge,
                    local_mult,
                    batch_size=stop - start,
                    need_graph=False,
                )
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or not sizer.backoff_after_oom():
                    raise
                continue
            energies.append(energy.detach())
            forces.append(force.detach())
            start = stop
        return torch.cat(energies, dim=0), torch.cat(forces, dim=0)

    def get_efh_gpu(self):
        B = self._atoms_B
        device, dtype = self.device, self.dtype
        result_dtype = torch.float64
        if B == 0:
            return (torch.zeros((0,), dtype=result_dtype, device=device),
                    torch.zeros((0, 0), dtype=result_dtype, device=device),
                    torch.zeros((0, 0, 0), dtype=result_dtype, device=device),
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
        F_eV = torch.zeros((B, nmax), dtype=result_dtype, device=device)
        H_eV = torch.zeros(
            (B, nmax, nmax),
            dtype=result_dtype,
            device=device,
        )
        P    = torch.empty((B,), dtype=torch.int64, device=device)

        s = self._ptr[:-1]; t = self._ptr[1:]
        for i in range(B):
            ni  = int((t[i] - s[i]).item())
            dof = 3 * ni
            P[i] = self.Nmax_atoms - ni
            if dof > 0:
                F_eV[i, :dof]       = F_all_eV[s[i]:t[i], :].reshape(-1)
                H_eV[i, :dof, :dof] = H_global_eV[3*s[i]:3*t[i], 3*s[i]:3*t[i]]

        energies = E_eV.to(result_dtype) / EH2EV
        forces = F_eV / EH2EV
        hessians = H_eV / EH2EV
        if (
            not bool(torch.isfinite(energies).all().item())
            or not bool(torch.isfinite(forces).all().item())
            or not bool(torch.isfinite(hessians).all().item())
        ):
            raise FloatingPointError(
                "AIMNet2BatchCalc produced non-finite energies, forces, or Hessians."
            )
        return energies, forces, hessians, P
