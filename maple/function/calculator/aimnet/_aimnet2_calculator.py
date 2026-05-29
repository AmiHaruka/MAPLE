import os
import torch
import numpy as np
from typing import Dict, Literal
from ase.calculators.calculator import Calculator, all_changes
from ..calculator_base import CalcABC, IMPLICIT_SOLVENT_FORCE_ERROR


EV2HARTREE = 1.0 / 27.211386245988

# --------------------------------------------
# Build dense neighbor list (N+1, M) sentinel padded
# --------------------------------------------
def nblist_dense_padded(coord: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Brute-force dense neighbor list for single molecule.
    Returns: (N+1, M) int32 tensor, sentinel row = N
    """
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff ** 2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float('inf')
    mask = dist2 <= cutoff ** 2
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, :min(nb_i.numel(), M)] = nb_i[:min(nb_i.numel(), M)]
    return nbmat

# --------------------------------------------
# Pad helpers
# --------------------------------------------
def pad_dim0(a: torch.Tensor, value=0.0) -> torch.Tensor:
    """
    Pad one row along dim0.
    For (N, C) -> (N+1, C), (N,) -> (N+1,)
    """
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)

def maybe_pad_dim0(a: torch.Tensor, N: int, value=0.0) -> torch.Tensor:
    """
    If a.shape[0] == N, return as is.
    If a.shape[0] == N-1, pad one row to length N.
    """
    diff = N - a.shape[0]
    assert diff in (0, 1), f"Invalid pad: {a.shape[0]} vs target {N}"
    if diff == 1:
        a = pad_dim0(a, value=value)
    return a

# ==========================================================
# AIMNet2 Calculator (single-molecule minimal version)
# ==========================================================
class AIMNet2Calculator(CalcABC):
    implemented_properties = ["energy", "forces", "hessian", "free_energy"]
    supported_hessian_modes = ("analytic", "numerical")

    def __init__(self, device: torch.device, 
                model: str = "aimnet2", 
                coulomb_method: str = "simple",
                implicit: Literal["gbsa", "none"] = "gbsa",
                solvent: str = 'none',
                ):
        super().__init__()
        self.device = device

        # Load model
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, "model", f"{model}.pt")
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, "cutoff"))
        # CHANGED: do not rely on hasattr(model, 'cutoff_lr') to decide LR; model may still need nbmat_lr
        self.cutoff_lr = float(getattr(self.model, "cutoff_lr", float("inf")))
        # keep a flag (optional); but we will always provide nbmat_lr anyway
        self.lr = True  # CHANGED: force-true to avoid conditional omission
        self.hessian: str = 'analytic'  # 'analytic' or 'numerical'

        # Coulomb settings (keep original behavior)
        self._set_lrcoulomb_method(coulomb_method)

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2):
            """
            Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
            method: 'simple', 'dsf', or 'ewald'
            cutoff: cutoff distance for long-range interactions
            dsf_alpha: DSF damping parameter (if used)
            """
            assert method in ("simple", "dsf", "ewald"), f"Invalid method: {method}"

            # recursively look for 'lrcoulomb' submodules
            def _iter_lrcoulomb_mods(model):
                for name, mod in model.named_modules():
                    if name == "lrcoulomb":
                        yield mod

            for mod in _iter_lrcoulomb_mods(self.model):
                mod.method = method
                if method == "dsf" and hasattr(mod, "dsf_alpha"):
                    mod.dsf_alpha = dsf_alpha

            # update cutoff_lr based on the chosen method
            self.cutoff_lr = float("inf") if method == "simple" else float(cutoff)
            self._coulomb_method = method

    # ------------------------ calculate ------------------------
    def calculate(self, atoms=None, properties=["energy", "forces", "free_energy", "hessian"], system_changes=all_changes):
        self._reject_implicit_solvent_derivatives(properties)
        super().calculate(atoms, properties, system_changes)

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=("forces" in properties or "hessian" in properties)
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]

        charge_val = float(self.atoms.info.get("charge", 0.0))
        mult_val = float(self.atoms.info.get("mult", 1.0))

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),         # (N+1, 3)
            "numbers": pad_dim0(Z, value=0),             # (N+1,)
            "charge": torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            "mult": torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }

        # CHANGED: ALWAYS provide nbmat_lr + cutoff_lr to avoid KeyError inside TorchScript
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE  # to Hartree

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results["energy"] = float(energy.item())
        self.results["free_energy"] = float(energy.item())

        if "forces" in properties:
            grad_full = torch.autograd.grad(
                energy, data["coord"], create_graph=("hessian" in properties)
            )[0]                      # (N+1, 3)
            forces = -grad_full[:N]   # (N, 3)
            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force

            self.results["forces"] = forces.detach().cpu().numpy()
            
        if "hessian" in properties:
            self.results["hessian"] = self.get_hessian(atoms)

    # ------------------------ get_energy ------------------------
    def get_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out["energy"].sum()

    # ------------------------ get_hessian ------------------------
    def get_hessian(
        self, 
        atoms,
        delta: float = 0.002,
    ) -> np.ndarray:
        """
        Compute the Hessian matrix using either analytic or numerical method.
        
        Method is determined by self.hessian:
        - 'analytic': Use automatic differentiation (faster, exact)
        - 'numerical': Use finite-difference forces (slower, approximate)
        
        Returns a (3N, 3N) numpy array.
        
        Args:
            atoms: ASE Atoms object
            delta: Step size for numerical differentiation (only used if method='numerical')
        """
        if self.solvent_correction:
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

        if self.hessian == 'analytic':
            return self._get_hessian_analytic(atoms)
        elif self.hessian == 'numerical':
            return self._get_hessian_numerical(atoms, delta)
        else:
            raise ValueError(f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'")


    def _get_hessian_analytic(self, atoms) -> np.ndarray:
        """
        Compute Hessian using automatic differentiation.
        Fast and exact, but requires energy to be differentiable w.r.t. coordinates.
        """
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]
        nbmat = nblist_dense_padded(coord, self.cutoff)
        charge_val = float(atoms.info.get("charge", 0.0))
        mult_val = float(atoms.info.get("mult", 1.0))

        data = {
            "coord": pad_dim0(coord, value=0.0),
            "numbers": pad_dim0(Z, value=0),
            "charge": torch.tensor([charge_val], dtype=torch.float32, device=self.device),
            "mult": torch.tensor([mult_val], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }
        
        # ALWAYS provide nbmat_lr + cutoff_lr
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE

        forces_full = torch.autograd.grad(energy, data["coord"], create_graph=True)[0]  # (N+1, 3)
        forces = -forces_full[:N]  # (N, 3)

        # Use original-style assembly to stay consistent with AIMNet2 padding
        hessian = - torch.stack([
            torch.autograd.grad(f, data["coord"], retain_graph=True)[0]
            for f in forces.flatten().unbind()
        ]).view(-1, 3, N + 1, 3)[:, :, :N, :]  # slice out the padded row on atom-axis

        return hessian.detach().cpu().numpy().reshape(3 * N, 3 * N)


    def _get_hessian_numerical(
        self, 
        atoms, 
        delta: float = 0.002
    ) -> np.ndarray:
        """
        Compute Hessian using finite-difference forces.
        Hessian is defined as: H = d²E/dx_i dx_j = -∂F_i/∂x_j
        
        Args:
            atoms: ASE Atoms object
            delta: Step size for finite difference
        """
        import numpy as np
        from ase.constraints import FixAtoms
        from ase.calculators.calculator import all_changes

        # Basic geometry setup
        N = len(atoms)
        pos0 = atoms.get_positions().copy()  # (N, 3) numpy array

        # Identify frozen atoms from FixAtoms constraint
        fixed = {
            i for c in getattr(atoms, "constraints", [])
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]

        # Allocate Hessian as numpy array
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)

        # If everything is frozen, return zero Hessian
        if len(movable) == 0:
            return H

        # Helper function: evaluate AIMNet forces at a displaced geometry
        def aimnet_force_at(pos_numpy: np.ndarray) -> np.ndarray:
            """
            Evaluate AIMNet forces at the given coordinates.
            Returns a numpy array of shape (N, 3).
            """
            at = atoms.copy()
            at.set_positions(pos_numpy)

            # Preserve constraints if present
            if getattr(atoms, "constraints", None):
                at.set_constraint(atoms.constraints)

            # Compute forces with the internal AIMNet calculator
            self.calculate(at, properties=["forces"], system_changes=all_changes)
            F_np = self.results["forces"]  # numpy (N, 3)
            return F_np

        # Finite-difference second derivatives:
        # H_ij = -∂F_i/∂x_j ≈ -(F(+δ) - F(-δ)) / (2δ)
        for a in movable:      # iterate over movable atoms
            for k in range(3):  # iterate over x, y, z directions
                row = 3 * a + k

                # +delta displacement
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                Fp = aimnet_force_at(pos_p)

                # -delta displacement
                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                Fm = aimnet_force_at(pos_m)

                # Central difference derivative of force
                # ∂F/∂x ≈ (F(+δ) - F(-δ)) / (2δ)
                dF = (Fp - Fm) / (2.0 * delta)

                # Hessian uses: H = -∂F/∂x
                H[row, :] = (-dF).reshape(-1)

        # Optional: symmetrize to reduce numerical noise
        # H = 0.5 * (H + H.T)

        return H
