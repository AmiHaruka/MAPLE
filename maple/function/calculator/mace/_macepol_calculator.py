from __future__ import annotations

import os
from typing import Literal, Sequence, Union

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from ..calculator_base import CalcABC, register_calculator


# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26,
    "Co":27, "Ni":28, "Cu":29, "Zn":30, "Br":35, "I":53,
}

# Model name → filename mapping
_MACEPOL_MODEL_FILES = {
    'macepols': 'macepols.pt',
    'macepolm': 'macepolm.pt',
    'macepoll': 'macepoll.pt',
}


def _integer_info(atoms, key: str, default: int) -> int:
    value = atoms.info.get(key, default)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}.") from exc
    if not numeric_value.is_integer():
        raise ValueError(f"MACE-POLAR requires integer atoms.info['{key}']; got {value!r}.")
    return int(numeric_value)


def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float32) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with atomic_number_table."""
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


# ------------------------ Calculator ------------------------

@register_calculator
class MACEPolCalculator(CalcABC):
    """ASE calculator for MACE-POLAR models (pure MLIP, f32).

    The traced model accepts flat tensor inputs and returns:
        (total_energy, node_energy, density_coefficients)

    Supports total_charge and total_spin via atoms.info['charge'] and atoms.info['mult'].
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']

    MODEL_NAMES = ('macepols', 'macepolm', 'macepoll')
    MODEL_ENERGY_UNIT = 'eV'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = True
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ()
    MODEL_PATH_OPTION = 'model_path'

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device: torch.device,
        model: str = 'macepols',
        model_path: str = None,
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device: Torch device.
            model: Model name ('macepols', 'macepolm', 'macepoll').
            model_path: Optional explicit path to .pt file (overrides model name lookup).
            implicit: Implicit solvent model type.
            solvent: Solvent type.
        """
        super().__init__()

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            filename = _MACEPOL_MODEL_FILES.get(model, f'{model}.pt')
            model_path = os.path.join(model_dir, 'model', filename)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32  # MACE-POLAR traced models are f32
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _build_inputs(self, atoms, requires_grad=False):
        """Build the 12 flat tensor inputs for MACE-POLAR forward pass."""
        device = self.device
        dtype = self.dtype

        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=device,
            requires_grad=requires_grad,
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
        node_attrs = _one_hot_node_attrs(Z, self.atomic_numbers, dtype=dtype)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)

        N = positions.size(0)
        unit_shifts = torch.zeros_like(shifts)
        batch = torch.zeros(N, dtype=torch.int64, device=device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
        cell = torch.zeros(3, 3, dtype=dtype, device=device)

        # Charge and spin from atoms.info (default: 0, singlet)
        charge = float(atoms.info.get('charge', 0))
        mult = _integer_info(atoms, 'mult', 1)
        spin = float(mult - 1)
        total_charge = torch.tensor([charge], dtype=dtype, device=device)
        total_spin = torch.tensor([spin], dtype=dtype, device=device)

        # No external field for pure MLIP
        external_field = torch.zeros(N, 3, dtype=dtype, device=device)
        local_or_ghost = torch.ones(N, dtype=dtype, device=device)

        return (positions, node_attrs, edge_index, shifts, unit_shifts,
                batch, ptr, cell, total_charge, total_spin,
                external_field, local_or_ghost)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        # Energy (no grad)
        inputs = self._build_inputs(atoms, requires_grad=False)
        with torch.no_grad():
            total_energy, _, _ = self.model(*inputs)

        energy_eV = total_energy.sum().double()

        forces_np = None
        if 'forces' in properties:
            inputs_grad = self._build_inputs(atoms, requires_grad=True)
            total_energy_grad, _, _ = self.model(*inputs_grad)

            forces = -torch.autograd.grad(
                total_energy_grad.sum(), inputs_grad[0],
                create_graph=False, retain_graph=False,
            )[0]
            forces_np = forces.double().detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        from ..calculator_base import EV2HARTREE

        inputs = self._build_inputs(atoms, requires_grad=True)
        total_energy, _, _ = self.model(*inputs)
        energy = total_energy.sum() * EV2HARTREE

        positions = inputs[0]
        num_atoms = positions.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=positions.dtype, device=positions.device)
        grad = torch.autograd.grad(energy, positions, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], positions, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2

        return hessian.detach().cpu().numpy()
