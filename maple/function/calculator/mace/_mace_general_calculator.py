from __future__ import annotations

import os
from typing import Literal

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

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    """Convert atomic numbers to one-hot vectors aligned with atomic_number_table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct O(N^2) radius graph without periodic boundaries."""
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

def build_inputs_from_atoms(atoms, model, device='cpu', positions=None):
    """Build model inputs from an ASE Atoms object.

    Returns (positions, node_attrs, edge_index, shifts, batch, ptr).
    The newer MACE wrappers may need total_charge / total_spin, but those are
    handled inside the wrapper.
    """
    device = torch.device(device)
    if positions is None:
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    else:
        pos = positions
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
    edge_index, shifts = _radius_graph_no_pbc(pos, float(model.r_max))

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)

    return pos, node_attrs, edge_index, shifts, batch, ptr


# ------------------------ Calculator ------------------------

@register_calculator
class MACEModelCalculator(CalcABC):
    """ASE calculator wrapping a traced MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('maceomol',)
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = True

    def __init__(self,
        device: torch.device,
        model: str = 'maceomol',
        overwrite: bool = False,
        implicit: Literal['gbsa', 'none'] = 'gbsa',
        solvent: str = 'none',
        ):
        """
        Args:
            device (torch.device): compute device
            model (str): model name; expects model/<model>.pt
            overwrite (bool): whether to overwrite existing models
            implicit (str): implicit-solvent model type
            solvent (str): solvent name
        """
        super().__init__()
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
        """Main ASE entry point."""
        super().calculate(atoms, properties, system_changes)

        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device)
        with torch.no_grad():
            total_energy = self.model(*inputs)

        energy_eV = total_energy.sum()

        forces_np = None
        if 'forces' in properties:
            positions_grad = torch.tensor(
                atoms.get_positions(), dtype=torch.float64, device=self.device, requires_grad=True,
            )
            inputs_grad = build_inputs_from_atoms(atoms, self.model, device=self.device, positions=positions_grad)
            total_energy = self.model(*inputs_grad)
            forces = -torch.autograd.grad(
                total_energy.sum(), positions_grad, create_graph=False, retain_graph=False,
            )[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True,
        )
        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device, positions=positions)
        total_energy = self.model(*inputs)
        energy = total_energy.sum() * EV2HARTREE

        num_atoms = positions.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=positions.dtype, device=positions.device)
        grad = torch.autograd.grad(energy, positions, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], positions, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2

        return hessian.detach().cpu().numpy()
