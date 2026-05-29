from __future__ import annotations

import inspect
import types

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import CalcABC, numerical_hessian_from_atoms


class StateTrackingForceCalc:
    def __init__(self):
        self.results = {"sentinel": 1}
        self.atoms = Atoms("H", positions=[[9.0, 9.0, 9.0]])

    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()
        self.results = {"forces": -atoms.get_positions()}


def test_numerical_hessian_restores_results_and_atoms_state():
    atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]])
    calc = StateTrackingForceCalc()
    old_atoms = calc.atoms
    old_results = dict(calc.results)

    hessian = numerical_hessian_from_atoms(calc, atoms, delta=1e-5)

    np.testing.assert_allclose(hessian, np.eye(3), atol=1e-10)
    assert calc.results == old_results
    assert calc.atoms is old_atoms


def test_uma_explicit_omol_rejects_periodic_atoms():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    calc = UMACalculator.__new__(UMACalculator)
    calc._auto_task = False
    calc._task_name = "omol"

    calc._set_task_from_atoms(atoms)
    with pytest.raises(ValueError, match="task='omol'.*periodic"):
        calc._validate_task_atoms_compatibility(atoms)


def test_uma_explicit_periodic_task_accepts_periodic_atoms():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    calc = UMACalculator.__new__(UMACalculator)
    calc._auto_task = False
    calc._task_name = "oc20"

    calc._set_task_from_atoms(atoms)
    calc._validate_task_atoms_compatibility(atoms)


def test_ani_calculate_hessian_property_writes_result():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = ANICalculator.__new__(ANICalculator)
    CalcABC.__init__(calc)
    calc.device = torch.device("cpu")
    calc.dtype = torch.float32
    calc.d4 = False
    calc.solvent_correction = None

    def fake_forward(self, atoms, coordinates):
        return coordinates.new_tensor(1.25)

    calc._forward_energy = types.MethodType(fake_forward, calc)
    calc.get_hessian = types.MethodType(lambda self, atoms: np.eye(3), calc)

    calc.calculate(atoms, properties=["hessian"])

    assert calc.results["energy"] == pytest.approx(1.25)
    np.testing.assert_allclose(calc.results["hessian"], np.eye(3))


def test_ani_calculate_properties_none_is_energy_only():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = ANICalculator.__new__(ANICalculator)
    CalcABC.__init__(calc)
    calc.device = torch.device("cpu")
    calc.dtype = torch.float32
    calc.d4 = False
    calc.solvent_correction = None
    calc._forward_energy = types.MethodType(lambda self, atoms, coordinates: coordinates.new_tensor(2.0), calc)

    calc.calculate(atoms, properties=None)

    assert calc.results == {"energy": pytest.approx(2.0), "free_energy": pytest.approx(2.0)}


def test_all_calcabc_backends_normalize_properties_before_membership_checks():
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
    from maple.function.calculator.ani._ani_calculator import ANICalculator
    from maple.function.calculator.mace._mace_calculator import MACECalculator
    from maple.function.calculator.mace._mace_general_calculator import MACEModelCalculator
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    for cls in (ANICalculator, AIMNet2Calculator, MACECalculator, MACEModelCalculator, MACEPolCalculator):
        source = inspect.getsource(cls.calculate)
        assert "properties = self._normalize_properties(properties)" in source, cls.__name__


def test_ani_hvp_rejects_implicit_solvent():
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    calc = ANICalculator.__new__(ANICalculator)
    calc.solvent_correction = object()

    with pytest.raises(NotImplementedError, match="implicit solvent"):
        calc.get_hvp(Atoms("H", positions=[[0.0, 0.0, 0.0]]), np.zeros(3))


def test_macepolar_rejects_noninteger_multiplicity():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["mult"] = 1.5

    calc = MACEPolCalculator.__new__(MACEPolCalculator)
    calc.device = torch.device("cpu")
    calc.dtype = torch.float32
    calc.r_max = 5.0
    calc.atomic_numbers = [1]

    with pytest.raises(ValueError, match=r"integer atoms.info\['mult'\]"):
        calc._build_inputs(atoms)
