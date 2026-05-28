from __future__ import annotations

import os

import ase
import numpy as np

from ..calculator_base import CalcABC, register_calculator


@register_calculator
class ANICalculator(CalcABC):
    implemented_properties = ['energy', 'forces', 'free_energy']

    MODEL_NAMES = ('ani2x', 'ani1x', 'ani1ccx', 'ani1xnr')
    # ANI's TorchScript checkpoints already return Hartree; no eV→Ha conversion.
    MODEL_ENERGY_UNIT = 'hartree'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    CHECKPOINT_FILENAME = {
        'ani2x': 'ani2x.pt',
        'ani1x': 'ani1x.pt',
        'ani1ccx': 'ani1ccx.pt',
        'ani1xnr': 'ani1xnr.pt',
    }
    REQUIRES_LOCAL_MODEL_FILE = False

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'d4': bool(options.get('d4', False))}

    def __init__(self, device,
        model: str = 'ani2x',
        overwrite=False,
        d4=False,
        implicit: str = 'none',
        solvent: str = 'none',
        ):
        import torch

        super().__init__()

        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32
        self.overwrite = overwrite
        self.d4 = d4
        self.hessian: str = 'analytic'

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        import torch

        super().calculate(atoms, properties, system_changes)

        needs_forces = 'forces' in properties
        coordinates = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=needs_forces,
        ).unsqueeze(0)

        energy = self._forward_energy(atoms, coordinates)

        if needs_forces:
            forces = -torch.autograd.grad(energy, coordinates)[0]
            forces_np = forces.squeeze(0).cpu().numpy()
        else:
            forces_np = None

        self._finalize_results(atoms, energy=energy.item(), forces=forces_np)

    def _forward_energy(self, atoms, coordinates):
        import torch

        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        energy = self.model(species, coordinates)[0]
        if self.d4:
            energy = energy + self.dftd4(species, coordinates)

        return energy

    def _analytic_hessian(self, atoms) -> np.ndarray:
        import torch

        coordinates = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        ).unsqueeze(0)

        energy = self._forward_energy(atoms, coordinates)

        num_atoms = coordinates.shape[1]
        hessian = torch.zeros(
            (3 * num_atoms, 3 * num_atoms),
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coordinates, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2

        return hessian.detach().cpu().numpy()

    def dftd4(self, species, coordinates):
        import torch
        import tad_dftd4 as d4

        charge = torch.tensor(0.0, device=self.device)
        param = {
            's6': coordinates.new_tensor(1.0),
            's8': coordinates.new_tensor(0.34783580),
            's9': coordinates.new_tensor(1.0),
            'a1': coordinates.new_tensor(0.57488291),
            'a2': coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))
