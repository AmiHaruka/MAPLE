from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC, register_calculator, parse_bool_option
from .. import _inputs, _torch


@register_calculator
class ANICalculator(CalcABC):
    """ASE calculator for standalone ANI TorchScript models and optional D4."""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian', 'charges', 'dipole', 'stress', 'virial']
    MODEL_NAMES = (
        'ani1x', 'ani1ccx', 'ani2x', 'ani1xnr', 'ani2xr', 'ani2dr', 'snn-ani2xr',
        'ani-mbis',
    )
    MODEL_ENERGY_UNIT = 'hartree'
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_PBC = True
    SUPPORTS_CHARGE_MULT = False
    REQUIRES_LOCAL_MODEL_FILE = True
    MODEL_PATH_OPTION = 'model_path'
    OPTION_KEYS = ('d4', 'dtype')

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        return {'model_path': resolved_model_path, 'dtype': options.get('dtype'),
                'd4': parse_bool_option(options.get('d4', False), name='d4')}

    def __init__(self, device,
                 model: str = 'ani2x',
                 model_path: Optional[str] = None,
                 overwrite=False,
                 d4=False,
                 implicit: str = 'none',
                 solvent: str = 'none',
                 dtype=None):
        super().__init__()
        path = model_path or Path(__file__).resolve().parents[1] / 'model' / (model + '.pt')
        _torch.load_model(self, path, device, dtype)
        self.model_name, self.overwrite, self.d4 = model, overwrite, d4
        if d4 and getattr(self.model, 'model_name', model) == 'ani2dr':
            raise ValueError('ani2dr already contains native dispersion; extra D4 is unsupported')
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)
        requested = ['energy' if name == 'free_energy' else name for name in properties]
        requested = list(dict.fromkeys(['energy', *requested]))
        numerical = 'hessian' in requested and self.hessian == 'numerical'
        if numerical:
            requested.remove('hessian')
        result = self._calculate([atoms], requested)
        if numerical:
            result['hessian'] = [self.get_hessian(atoms)]
        self._store_properties(atoms, result)

    def _forward(self, data):
        output = self.model((data['numbers'], data['positions']),
                            cell=data['cell'][0] if 'cell' in data else None,
                            pbc=data['pbc'][0] if 'pbc' in data else None)
        result = {'energy': output.energies}
        charges = getattr(output, 'atomic_charges', None)
        if charges is not None:
            result['charges'] = charges
        if self.d4:
            if len(data['positions']) != 1:
                raise NotImplementedError('External D4 uses single-structure composition')
            result['energy'] = result['energy'] + self.dftd4(data['numbers'], data['positions'])
        return result

    def _forward_energy(self, atoms, coordinates):
        self._validate_dispersion([atoms])
        data = _inputs.prepare_ani([atoms], device=self.device, dtype=self.dtype,
                                  supported_elements=self.model.atomic_numbers.tolist())
        data['positions'] = coordinates
        return self._forward(data)['energy']

    def _calculate(self, atoms_list, properties, vector=None):
        self._validate_dispersion(atoms_list)
        data = _inputs.prepare_ani(atoms_list, device=self.device, dtype=self.dtype,
                                  supported_elements=self.model.atomic_numbers.tolist())
        if 'cell' in data and any(
                not np.array_equal(a.cell.array, atoms_list[0].cell.array)
                or not np.array_equal(a.pbc, atoms_list[0].pbc) for a in atoms_list[1:]):
            raise NotImplementedError('ANI periodic native batches require the same cell and PBC mask')
        order = _torch.derivative_order(properties)
        # CUDA fusion can emit unsupported allocations in repeated force/Hessian backward graphs.
        context = (torch.jit.optimized_execution(False)
                   if order and self.device.type == 'cuda' else nullcontext())
        with context, torch.set_grad_enabled(order > 0):
            positions, live, cells, strain, volume = _torch.prepare_derivatives(
                data['positions'], atoms_list, properties)
            data['positions'] = live
            if cells is not None:
                data['cell'] = cells
            output = self._forward(data)
            return _torch.tensor_properties(output['energy'], positions, atoms_list, properties,
                                            outputs=output, source_unit=self.MODEL_ENERGY_UNIT,
                                            strain=strain, volume=volume,
                                            single=vector is not None, vector=vector)

    def _batch_key(self, atoms):
        if atoms.pbc.any():
            return (tuple(atoms.pbc), tuple(atoms.cell.array.flatten()))
        return ()

    def _calculate_batch(self, atoms_list, properties, vector=None):
        if self.d4:
            return NotImplemented
        if not self.model.supports_native_batch:
            return NotImplemented
        if (any(atoms.pbc.any() for atoms in atoms_list)
                and not self.model.supports_native_pbc_batch):
            return NotImplemented
        return self._calculate(atoms_list, properties, vector=vector)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        return self._calculate([atoms], ['hessian'])['hessian'][0]

    def get_hvp(self, atoms, n: np.ndarray):
        self._reject_unsupported_pbc(atoms)
        if getattr(self, 'solvent_correction', None) is not None:
            return super().get_hvp(atoms, n)
        return self._calculate([atoms], ['hvp'], vector=n)['hvp']

    def _validate_dispersion(self, atoms_list):
        if self.d4 and any(atoms.pbc.any() for atoms in atoms_list):
            raise ValueError('External D4 has no periodic cell/PBC implementation')

    def dftd4(self, species, coordinates):
        import tad_dftd4 as d4
        charge = coordinates.new_tensor(0.)
        param = {
            's6': coordinates.new_tensor(1.0),
            's8': coordinates.new_tensor(0.34783580),
            's9': coordinates.new_tensor(1.0),
            'a1': coordinates.new_tensor(0.57488291),
            'a2': coordinates.new_tensor(6.41921802),
        }
        bohr_coords = coordinates[0] * 1.8897261245864
        return torch.sum(d4.dftd4(species[0], bohr_coords, charge, param))
