from __future__ import annotations

import inspect
import types
import warnings

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


def test_uma_non_omol_rejects_nondefault_charge_or_mult():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    calc._task_name = "oc20"

    charged = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    charged.info["charge"] = -1
    with pytest.raises(ValueError, match="does not use charge/spin"):
        calc._validate_charge_spin_task_compatibility(charged)

    open_shell = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    open_shell.info["mult"] = 3
    with pytest.raises(ValueError, match="does not use charge/spin"):
        calc._validate_charge_spin_task_compatibility(open_shell)


@pytest.mark.parametrize("info", [{"charge": -1}, {"mult": 3}])
def test_uma_calculate_rejects_non_omol_charge_spin_before_backend(monkeypatch, info):
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma import _uma_calculator as uma_module

    def fail_if_called(self, atoms, properties, system_changes):
        raise AssertionError("FAIRChemCalculator.calculate should not be called")

    monkeypatch.setattr(uma_module.FAIRChemCalculator, "calculate", fail_if_called)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info.update(info)
    calc = uma_module.UMACalculator.__new__(uma_module.UMACalculator)
    calc._auto_task = False
    calc._task_name = "oc20"
    calc.solvent_correction = None
    calc.results = {}

    with pytest.raises(ValueError, match="does not use charge/spin"):
        calc.calculate(atoms, properties=["energy"], system_changes=[])


def test_uma_non_omol_accepts_default_charge_spin():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    calc._task_name = "oc20"

    assert calc._validate_charge_spin_task_compatibility(
        Atoms("H", positions=[[0.0, 0.0, 0.0]])
    ) == (0, 1)


def test_uma_omol_charge_spin_warns_without_blocking():
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    calc = UMACalculator.__new__(UMACalculator)
    calc._task_name = "omol"

    with pytest.warns(RuntimeWarning, match="charged/open-shell"):
        assert calc._validate_charge_spin_task_compatibility(atoms) == (1, 1)


def test_uma_calculate_omol_charge_spin_warns_and_reaches_backend(monkeypatch):
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma import _uma_calculator as uma_module

    calls = {}

    def fake_fairchem_calculate(self, atoms, properties, system_changes):
        calls["charge"] = atoms.info["charge"]
        calls["spin"] = atoms.info["spin"]
        self.results = {"energy": 0.0}

    monkeypatch.setattr(uma_module.FAIRChemCalculator, "calculate", fake_fairchem_calculate)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    atoms.info["mult"] = 3
    calc = uma_module.UMACalculator.__new__(uma_module.UMACalculator)
    calc._auto_task = False
    calc._task_name = "omol"
    calc.solvent_correction = None
    calc.results = {}

    with pytest.warns(RuntimeWarning, match="charged/open-shell"):
        calc.calculate(atoms, properties=["energy"], system_changes=[])

    assert calls == {"charge": 1, "spin": 3}


def test_uma_calculate_normalizes_properties_none(monkeypatch):
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma import _uma_calculator as uma_module

    calls = {}

    def fake_fairchem_calculate(self, atoms, properties, system_changes):
        calls["properties"] = properties
        calls["system_changes"] = system_changes
        self.results = {"energy": 0.0}

    monkeypatch.setattr(uma_module.FAIRChemCalculator, "calculate", fake_fairchem_calculate)

    calc = uma_module.UMACalculator.__new__(uma_module.UMACalculator)
    calc._auto_task = False
    calc._task_name = "omol"
    calc.solvent_correction = None
    calc.results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calc.calculate(Atoms("H", positions=[[0.0, 0.0, 0.0]]), properties=None, system_changes=None)

    assert calls["properties"] == ["energy"]
    assert calls["system_changes"] == uma_module.all_changes


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


def test_mace_graph_builder_matches_model_float_dtype():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import build_data_from_atoms
    from maple.function.calculator.mace._mace_general_calculator import build_inputs_from_atoms

    class Float32Model:
        r_max = 5.0
        atomic_numbers = torch.tensor([1, 8])

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

        def buffers(self):
            return iter(())

    data, local_or_ghost = build_data_from_atoms(
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        Float32Model(),
        device="cpu",
    )

    assert data["positions"].dtype is torch.float32
    assert data["node_attrs"].dtype is torch.float32
    assert local_or_ghost.dtype is torch.float32

    positions, node_attrs, edge_index, shifts, batch, ptr = build_inputs_from_atoms(
        Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        Float32Model(),
        device="cpu",
    )
    assert positions.dtype is torch.float32
    assert node_attrs.dtype is torch.float32
    assert shifts.dtype is torch.float32
