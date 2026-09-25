from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .._batch_types import BatchResult
from .._batch_utils import (
    atoms_list_has_pbc,
    empty_batch_result,
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from ..calculator_base import (
    EV2HARTREE,
    CalcABC,
    hessian_via_double_autograd,
    register_calculator,
)
from ._batch_graph import build_mace_tuple_batch, energy_vector_from_output
from ..electronic_state import (
    attach_calculator_identity,
    solvation_identity_settings,
    validate_electronic_state,
)
from ._common import model_float_dtype, one_hot_node_attrs, radius_graph_no_pbc


# ------------------------ Input builder ------------------------

def build_inputs_from_atoms(atoms, model, device='cpu', positions=None, dtype=None):
    """Build model inputs from an ASE Atoms object.

    Returns (positions, node_attrs, edge_index, shifts, batch, ptr).
    The newer MACE wrappers may need total_charge / total_spin, but those are
    handled inside the wrapper.
    """
    device = torch.device(device)
    dtype = dtype or model_float_dtype(model)
    if positions is None:
        pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    else:
        pos = positions
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = one_hot_node_attrs(Z, atomic_number_table, dtype=pos.dtype)
    edge_index, shifts = radius_graph_no_pbc(pos, float(model.r_max))

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
    SUPPORTS_PBC = False
    CHECKPOINT_FILENAME = None
    REQUIRES_LOCAL_MODEL_FILE = True
    OPTION_KEYS = ('batch_size', 'path_batch_size')
    MODEL_PATH_OPTION = 'model_path'
    supports_batch_energy_forces = True
    supports_analytic_hessian = True
    batch_memory_model = 'disconnected_graph'
    auto_path_batch_cap = 8

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        kwargs = {}
        if options.get('batch_size') is not None:
            kwargs['batch_size'] = options.get('batch_size')
        if options.get('path_batch_size') is not None:
            kwargs['path_batch_size'] = options.get('path_batch_size')
        if resolved_model_path is not None:
            kwargs['model_path'] = resolved_model_path
        return kwargs

    def __init__(self,
        device: torch.device,
        model: str = 'maceomol',
        model_path: Optional[str] = None,
        overwrite: bool = False,
        batch_size=None,
        path_batch_size='auto',
        implicit: Literal['gbsa', 'none'] = 'none',
        solvent: str = 'none',
        ):
        """
        Args:
            device (torch.device): compute device
            model (str): model name; expects model/<model>.pt
            model_path (str, optional): explicit path to the scripted model file;
                overrides the model-name lookup when provided.
            overwrite (bool): whether to overwrite existing models
            implicit (str): implicit-solvent model type
            solvent (str): solvent name
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
        self.dtype = model_float_dtype(self.model)
        self.overwrite = overwrite
        self.batch_size = batch_size
        self.path_batch_size = path_batch_size

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = 'analytic'

        attach_calculator_identity(
            self,
            backend=str(model).lower(),
            checkpoint_path=model_path,
            relevant_settings=solvation_identity_settings(implicit, solvent),
        )

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Main ASE entry point."""
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)

        # Single forward; positions carry grad only when forces are requested.
        needs_forces = 'forces' in properties
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device,
            requires_grad=needs_forces,
        )
        inputs = build_inputs_from_atoms(
            atoms, self.model, device=self.device, positions=positions, dtype=self.dtype
        )
        total_energy = self.model(*inputs)
        energy_eV = total_energy.sum()

        forces_np = None
        if needs_forces:
            forces = -torch.autograd.grad(energy_eV, positions)[0]
            forces_np = forces.detach().cpu().numpy()

        hessian = None
        if 'hessian' in properties:
            if self.solvent_correction is not None:
                raise NotImplementedError('Hessian calculation with implicit solvent is not implemented yet.')
            hessian = self.get_hessian(atoms)

        self._finalize_results(atoms, energy=energy_eV.item(), forces=forces_np, hessian=hessian)

    def calculate_many(self, atoms_list, properties=("energy", "forces")) -> BatchResult:
        """Evaluate independent MACE-OMol structures as one graph batch."""
        _, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        if not request:
            return BatchResult()

        atoms_list = list(atoms_list)
        if not atoms_list:
            return empty_batch_result(want_energy, want_forces)

        if atoms_list_has_pbc(atoms_list):
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )

        if getattr(self, 'solvent_correction', None) is not None:
            return sequential_calculate_many(
                self, atoms_list, request, want_energy, want_forces
            )

        for atoms in atoms_list:
            validate_electronic_state(atoms, self)

        inputs, counts = build_mace_tuple_batch(
            atoms_list,
            atomic_numbers=self.atomic_numbers,
            r_max=self.r_max,
            device=self.device,
            dtype=self.dtype,
            requires_grad=want_forces,
        )
        positions, _, _, _, batch, _ = inputs

        if want_forces:
            total_energy = self.model(*inputs)
        else:
            with torch.no_grad():
                total_energy = self.model(*inputs)

        energy_vec_eV = energy_vector_from_output(
            total_energy,
            batch_size=len(atoms_list),
            n_atoms_total=positions.shape[0],
            batch=batch,
        )

        # Match calculate()/_finalize_results(): promote model energies before
        # conversion so FP32 models do not incur an extra Hartree-rounding step.
        energies = (
            energy_vec_eV.detach().cpu().numpy().astype(np.float64) * EV2HARTREE
            if want_energy else None
        )

        forces_list = None
        if want_forces:
            forces_eV = -torch.autograd.grad(
                energy_vec_eV.sum(),
                positions,
                create_graph=False,
                retain_graph=False,
            )[0]
            forces_ha = forces_eV.detach().cpu().numpy() * EV2HARTREE
            forces_list = split_atomwise_array(
                forces_ha.astype(np.float64),
                counts,
            )

        return BatchResult(
            energies=energies,
            forces=forces_list,
        ).validate_against(atoms_list, request)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        """Analytic Hessian via autograd. Returns (3N, 3N) np.ndarray in Hartree/Å²."""
        positions = torch.tensor(
            atoms.get_positions(), dtype=self.dtype, device=self.device, requires_grad=True,
        )
        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device, positions=positions)

        def energy_fn():
            return self.model(*inputs).sum() * EV2HARTREE

        return hessian_via_double_autograd(energy_fn, positions)
