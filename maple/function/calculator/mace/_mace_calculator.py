from __future__ import annotations

import os
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30
}

def _symbols_to_Z(symbols: Sequence[Union[str,int]]) -> list:
    """Convert element symbols to atomic numbers."""
    out = []
    for s in symbols:
        if isinstance(s, int):
            out.append(int(s))
        else:
            z = _SYMBOL2Z.get(str(s))
            if z is None:
                raise ValueError(f"Unknown element symbol: {s}")
            out.append(z)
    return out

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with atomic_number_table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct a simple O(N^2) radius graph without periodic boundaries."""
    N = positions.size(0)
    rij = positions[:, None, :] - positions[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=positions.device)
    mask.fill_diagonal_(False)
    mask &= (d2 <= (r_max + 1e-12) ** 2)
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    src = torch.cat([iu, ju], dim=0)
    dst = torch.cat([ju, iu], dim=0)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    shifts = torch.zeros((edge_index.size(1), 3), dtype=positions.dtype, device=positions.device)
    return edge_index, shifts

def build_data_from_atoms(atoms, model, device='cpu', positions: Optional[torch.Tensor] = None):
    """Build a data_dict for Wrapper.forward() from an ASE Atoms object."""
    device = torch.device(device)
    if positions is None:
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    else:
        pos = positions
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    r_max = float(model.r_max)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
    edge_index, shifts = _radius_graph_no_pbc(pos, r_max)

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    cell = torch.zeros(3, 3, dtype=torch.float64, device=device)
    charge = torch.zeros(N, dtype=torch.float64, device=device)
    dipole = torch.zeros(1, 3, dtype=torch.float64, device=device)
    energy = torch.tensor([0.0], dtype=torch.float64, device=device)
    energy_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    force = torch.zeros(N, 3, dtype=torch.float64, device=device)
    forces_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
    stress = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
    stress_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=torch.float64, device=device)
    virials = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
    virials_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    weight = torch.tensor([1.0], dtype=torch.float64, device=device)

    data_dict = {
        'batch': batch,
        'cell': cell,
        'charges': charge,
        'dipole': dipole,
        'edge_index': edge_index,
        'energy': energy,
        'energy_weight': energy_weight,
        'forces': force,
        'forces_weight': forces_weight,
        'node_attrs': node_attrs,
        'positions': pos,
        'ptr': ptr,
        'shifts': shifts,
        'stress': stress,
        'stress_weight': stress_weight,
        'unit_shifts': unit_shifts,
        'virials': virials,
        'virials_weight': virials_weight,
        'weight': weight
    }

    local_or_ghost = torch.ones(N, dtype=torch.float64, device=device)
    return data_dict, local_or_ghost


# ------------------------ Calculator ------------------------

@register_calculator
class MACECalculator(CalcABC):
    """ASE-style calculator wrapping a scripted Wrapper MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy']

    MODEL_NAMES = ('maceoff23s', 'maceoff23m', 'maceoff23l', 'egret')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    # Only the auto-downloaded variants. maceoff23s and maceoff23l are
    # local-only (REQUIRES_LOCAL_MODEL_FILE) — the factory falls back to
    # _require_local_model_file when CHECKPOINT_FILENAME has no entry.
    CHECKPOINT_FILENAME = {'maceoff23m': 'maceoff23m.pt', 'egret': 'egret1s.pt'}
    REQUIRES_LOCAL_MODEL_FILE = True

    supported_hessian_modes = SUPPORTED_HESSIAN_MODES

    def __init__(self,
        device: torch.device,
        model: str = 'maceoff23s',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'gbsa',
        solvent: str = 'none',
        ):
        """
        Args:
            device (torch.device): Torch device.
            model (str): Name of the model (expects `<model>.pt` under `model/`).
            model_path (str, optional): Explicit path to the scripted model file.
            overwrite (bool): Whether to overwrite existing models (unused).
        """
        super().__init__()
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float64
        self.overwrite = overwrite

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        super().calculate(atoms, properties, system_changes)

        # Energy-only forward (no autograd) — cheap path when forces not requested.
        data_dict, local_or_ghost = build_data_from_atoms(atoms, self.model, device=self.device)
        total_energy_local = self.model.forward(
            data=data_dict, local_or_ghost=local_or_ghost, compute_virials=False
        )
        energy_eV = total_energy_local.sum()

        forces_np = None
        if 'forces' in properties:
            data_dict['positions'].requires_grad_(True)
            total_energy_local = self.model.forward(
                data=data_dict, local_or_ghost=local_or_ghost, compute_virials=False
            )
            forces = -torch.autograd.grad(
                total_energy_local.sum(), data_dict['positions'],
                create_graph=False, retain_graph=False,
            )[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _build_graph_inputs(self, atoms, positions: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.tensor(atoms.get_positions(), dtype=self.dtype, device=self.device)
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        node_attrs = _one_hot_node_attrs(Z, self.atomic_numbers, dtype=self.dtype)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)
        N = positions.size(0)
        batch = torch.zeros(N, dtype=torch.int64, device=self.device)
        cell = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        charge = torch.zeros(N, dtype=self.dtype, device=self.device)
        dipole = torch.zeros(1, 3, dtype=self.dtype, device=self.device)
        energy = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        energy_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        force = torch.zeros(N, 3, dtype=self.dtype, device=self.device)
        forces_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=self.device)
        stress = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        stress_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=self.dtype, device=self.device)
        virials = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        virials_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        weight = torch.tensor([1.0], dtype=self.dtype, device=self.device)

        data_dict = {
            'batch': batch,
            'cell': cell,
            'charges': charge,
            'dipole': dipole,
            'edge_index': edge_index,
            'energy': energy,
            'energy_weight': energy_weight,
            'forces': force,
            'forces_weight': forces_weight,
            'node_attrs': node_attrs,
            'positions': positions,
            'ptr': ptr,
            'shifts': shifts,
            'stress': stress,
            'stress_weight': stress_weight,
            'unit_shifts': unit_shifts,
            'virials': virials,
            'virials_weight': virials_weight,
            'weight': weight
        }
        local_or_ghost = torch.ones(N, dtype=self.dtype, device=self.device)
        return data_dict, local_or_ghost

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True
        )
        data_dict, local_or_ghost = self._build_graph_inputs(atoms, positions=positions)
        total_energy_local = self.model.forward(
            data=data_dict, local_or_ghost=local_or_ghost, compute_virials=False
        )
        energy = total_energy_local.sum() * EV2HARTREE

        num_atoms = positions.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=positions.dtype, device=positions.device)
        grad = torch.autograd.grad(energy, positions, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], positions, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2

        return hessian.detach().cpu().numpy()
