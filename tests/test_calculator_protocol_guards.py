import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.calculator.calculator_base import (
    CalcABC,
    EV2HARTREE,
    _convert_energy_force_units,
    atoms_has_pbc,
    normalize_none_option,
    numerical_hessian_from_atoms,
    validate_implicit_solvent_choice,
)
from maple.function.calculator.set_calculator import (
    HF_MODEL_REVISION,
    SetCalculator,
    _model_download_url,
)


class DummyNoPBC:
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False
    SUPPORTED_COULOMB_METHODS = ("simple", "dsf")
    OPTION_KEYS = ("coulomb_method",)
    MODEL_PATH_OPTION = None

    def __init__(self):
        pass


class DummyCalcABC(CalcABC):
    implemented_properties = ["energy"]


class DummyEVFinalize(CalcABC):
    implemented_properties = ["energy", "forces", "free_energy", "hessian"]
    MODEL_ENERGY_UNIT = "eV"


class LinearForceCalc:
    def __init__(self):
        self.results = {
            "energy": 123.0,
            "forces": np.array([[9.0, 8.0, 7.0]]),
            "sentinel": "keep-me",
        }

    def calculate(self, atoms, properties, system_changes):
        # F = -x, so H = -dF/dx = I.
        self.results = {"forces": -atoms.get_positions()}


class CoupledForceCalc:
    def __init__(self, hessian):
        self.hessian = np.asarray(hessian, dtype=float)
        self.results = {}

    def calculate(self, atoms, properties, system_changes):
        x = atoms.get_positions().reshape(-1)
        self.results = {"forces": -(self.hessian @ x).reshape((-1, 3))}


def test_none_like_options_are_normalized():
    assert normalize_none_option(None) == "none"
    assert normalize_none_option("None") == "none"
    assert normalize_none_option("  NULL ") == "none"
    assert normalize_none_option("water") == "water"


def test_gbsa_requires_real_solvent_name():
    assert validate_implicit_solvent_choice("None", "None") == ("none", "none")
    assert validate_implicit_solvent_choice("gbsa", "water") == ("gbsa", "water")
    with pytest.raises(ValueError, match="requires an explicit solvent"):
        validate_implicit_solvent_choice("gbsa", "None")


def test_setcalculator_normalizes_options_and_validates_solvent(tmp_path):
    out = tmp_path / "maple.out"
    calc = SetCalculator(
        "cpu",
        "DUMMY",
        str(out),
        implicit=None,
        solvent="None",
        model_options={"Hessian": "ANALYTIC", "MODEL_PATH": "/tmp/M.pt", "coulomb_method": "DSF"},
    )
    assert calc.model == "dummy"
    assert calc.implicit == "none"
    assert calc.solvent == "none"
    assert calc.model_options == {
        "hessian": "analytic",
        "model_path": "/tmp/M.pt",
        "coulomb_method": "dsf",
    }
    calc._validate_solvent_config()

    bad = SetCalculator("cpu", "dummy", str(out), implicit="gbsa", solvent="None")
    with pytest.raises(ValueError, match="requires an explicit solvent"):
        bad._validate_solvent_config()


def test_strict_model_options_reject_unknown_keys(tmp_path):
    calc = SetCalculator(
        "cpu",
        "dummy",
        str(tmp_path / "maple.out"),
        model_options={"coulomb_method": "simple", "typoo": "1"},
    )
    with pytest.raises(ValueError, match="Unsupported model option"):
        calc._validate_model_options(DummyNoPBC)


def test_explicit_model_path_must_be_supported_and_exist(tmp_path):
    out = str(tmp_path / "maple.out")
    existing_model = tmp_path / "custom.pt"
    existing_model.write_text("not-a-real-checkpoint")

    unsupported = SetCalculator(
        "cpu",
        "dummy",
        out,
        model_options={"model_path": str(existing_model)},
    )
    with pytest.raises(ValueError, match="does not support model_path"):
        unsupported._resolve_explicit_model_path(DummyNoPBC, unsupported.model_options)

    from maple.function.calculator.ani._ani_calculator import ANICalculator

    supported = SetCalculator(
        "cpu",
        "ani2x",
        out,
        model_options={"model_path": str(existing_model)},
    )
    assert supported._resolve_explicit_model_path(ANICalculator, supported.model_options) == existing_model

    missing = SetCalculator(
        "cpu",
        "ani2x",
        out,
        model_options={"model_path": str(tmp_path / "missing.pt")},
    )
    with pytest.raises(FileNotFoundError, match="Explicit model_path"):
        missing._resolve_explicit_model_path(ANICalculator, missing.model_options)


def test_explicit_uma_model_path_checkpoint_path_compare_resolved_paths(tmp_path):
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    target = tmp_path / "uma.pt"
    target.write_text("not-a-real-checkpoint")
    symlink = tmp_path / "uma-link.pt"
    try:
        symlink.symlink_to(target)
    except (OSError, NotImplementedError):
        symlink = target

    same = SetCalculator(
        "cpu",
        "uma",
        str(tmp_path / "maple.out"),
        model_options={"model_path": str(symlink), "checkpoint_path": str(target)},
    )
    assert same._resolve_explicit_model_path(UMACalculator, same.model_options) == symlink

    other = tmp_path / "other.pt"
    other.write_text("not-a-real-checkpoint")
    different = SetCalculator(
        "cpu",
        "uma",
        str(tmp_path / "maple.out"),
        model_options={"model_path": str(symlink), "checkpoint_path": str(other)},
    )
    with pytest.raises(ValueError, match="only one of model_path or checkpoint_path"):
        different._resolve_explicit_model_path(UMACalculator, different.model_options)


def test_uma_checkpoint_path_must_exist_before_model_construction(tmp_path):
    calc = SetCalculator(
        "cpu",
        "uma",
        str(tmp_path / "maple.out"),
        model_options={"checkpoint_path": str(tmp_path / "missing.pt")},
    )
    with pytest.raises(FileNotFoundError, match="Explicit checkpoint_path"):
        calc._build_calculator()


def test_backend_build_kwargs_receive_explicit_model_path():
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    assert ANICalculator.build_kwargs_from_options(
        "ani2x", {}, resolved_model_path="/tmp/ani.pt"
    )["model_path"] == "/tmp/ani.pt"
    kwargs = AIMNet2Calculator.build_kwargs_from_options(
        "aimnet2", {"coulomb_method": "dsf"}, resolved_model_path="/tmp/aim.pt"
    )
    assert kwargs == {"coulomb_method": "dsf", "model_path": "/tmp/aim.pt"}


def test_no_pbc_model_rejected_by_factory_before_construction(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    calc = SetCalculator("cpu", "dummy", str(tmp_path / "maple.out"), atoms=atoms)
    with pytest.raises(NotImplementedError, match="no-PBC molecular wrapper"):
        calc._validate_against_class(DummyNoPBC)


def test_real_no_pbc_calculator_classes_reject_periodic_factory_use(tmp_path):
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
    from maple.function.calculator.ani._ani_calculator import ANICalculator
    from maple.function.calculator.mace._mace_calculator import MACECalculator
    from maple.function.calculator.mace._mace_general_calculator import MACEModelCalculator
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    for cls in (ANICalculator, AIMNet2Calculator, MACECalculator, MACEModelCalculator, MACEPolCalculator):
        calc = SetCalculator("cpu", cls.MODEL_NAMES[0], str(tmp_path / f"{cls.__name__}.out"), atoms=atoms)
        with pytest.raises(NotImplementedError, match="no-PBC molecular wrapper"):
            calc._validate_against_class(cls)


def test_no_pbc_calcabc_rejects_direct_periodic_calculate():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    assert atoms_has_pbc(atoms)
    with pytest.raises(NotImplementedError, match="no-PBC molecular wrapper"):
        DummyCalcABC().calculate(atoms, properties=["energy"])


def test_calcabc_direct_no_arg_without_attached_atoms_is_safe():
    assert DummyCalcABC().calculate(properties=["energy"]) is None


def test_calcabc_calculate_returns_normalized_atoms_for_direct_no_arg_use():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = DummyCalcABC()
    calc.atoms = atoms
    assert calc.calculate(properties=["energy"]) is atoms


def test_aimnet_ewald_option_rejected_by_capability_gate(tmp_path):
    calc = SetCalculator(
        "cpu",
        "aimnet2",
        str(tmp_path / "maple.out"),
        model_options={"coulomb_method": "ewald"},
    )
    with pytest.raises(NotImplementedError, match="ewald"):
        calc._validate_against_class(DummyNoPBC)


def test_aimnet_direct_ewald_selector_rejected_without_loading_model():
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    calc = AIMNet2Calculator.__new__(AIMNet2Calculator)
    calc.model = type("EmptyModel", (), {"named_modules": lambda self: []})()
    with pytest.raises(NotImplementedError, match="ewald"):
        calc._set_lrcoulomb_method("ewald")


def test_pinned_hf_revision_is_used_in_model_download_url():
    url = _model_download_url("ani2x.pt")
    assert f"/resolve/{HF_MODEL_REVISION}/" in url
    assert url.endswith("/ani2x.pt")


def test_numerical_hessian_restores_results_and_zeroes_fixed_dofs():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    atoms.set_constraint(FixAtoms(indices=[1]))
    calc = LinearForceCalc()
    before = dict(calc.results)

    hessian = numerical_hessian_from_atoms(calc, atoms, delta=1e-4)

    expected = np.zeros((6, 6))
    expected[:3, :3] = np.eye(3)
    np.testing.assert_allclose(hessian, expected, atol=1e-10)
    assert calc.results.keys() == before.keys()
    assert calc.results["sentinel"] == "keep-me"
    np.testing.assert_allclose(calc.results["forces"], before["forces"])


def test_numerical_hessian_handles_coupled_quadratic_forces():
    hessian_ref = np.eye(6)
    hessian_ref[0, 4] = hessian_ref[4, 0] = 0.37
    hessian_ref[2, 3] = hessian_ref[3, 2] = -0.21
    atoms = Atoms("H2", positions=[[0.3, -0.2, 0.1], [1.1, 0.4, -0.7]])
    calc = CoupledForceCalc(hessian_ref)

    hessian = numerical_hessian_from_atoms(calc, atoms, delta=1e-5)

    np.testing.assert_allclose(hessian, hessian_ref, atol=1e-10)


def test_finalize_results_converts_units_and_clears_stale_properties():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = DummyEVFinalize()
    calc.results = {"forces": np.ones((1, 3)), "hessian": np.ones((3, 3))}

    calc._finalize_results(
        atoms,
        energy=27.211386245988,
        forces=np.array([[1.0, 2.0, 3.0]]),
        hessian=np.eye(3),
    )

    assert calc.results["energy"] == pytest.approx(1.0)
    assert calc.results["free_energy"] == pytest.approx(1.0)
    np.testing.assert_allclose(calc.results["forces"], np.array([[1.0, 2.0, 3.0]]) * EV2HARTREE)
    np.testing.assert_allclose(calc.results["hessian"], np.eye(3))

    calc._finalize_results(atoms, energy=54.422772491976)
    assert calc.results == {
        "energy": pytest.approx(2.0),
        "free_energy": pytest.approx(2.0),
    }


def test_hartree_backend_unit_conversion_is_noop():
    energy, forces = _convert_energy_force_units(
        1.25,
        np.array([[0.1, 0.2, 0.3]]),
        source_unit="hartree",
    )
    assert energy == pytest.approx(1.25)
    np.testing.assert_allclose(forces, [[0.1, 0.2, 0.3]])


def test_uma_auto_task_requires_explicit_task_for_pbc():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    calc._auto_task = True
    calc._task_name = "omol"
    pbc_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)

    with pytest.raises(ValueError, match="explicit FAIR-Chem task"):
        calc._set_task_from_atoms(pbc_atoms)


def test_uma_explicit_task_is_not_overridden_for_pbc():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    calc._auto_task = False
    calc._task_name = "oc20"
    pbc_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)

    calc._set_task_from_atoms(pbc_atoms)
    assert calc.task_name == "oc20"


def test_uma_auto_task_keeps_nonperiodic_omol_without_predictor():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    calc._auto_task = True
    calc._task_name = "omol"
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    calc._set_task_from_atoms(atoms)
    assert calc.task_name == "omol"


def test_uma_charged_open_shell_omol_warns_without_blocking():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    atoms.info["mult"] = 2

    calc = UMACalculator.__new__(UMACalculator)
    calc._task_name = "omol"
    with pytest.warns(RuntimeWarning, match="charged/open-shell"):
        calc._warn_unvalidated_charge_spin(atoms)


def test_uma_rejects_noninteger_charge_before_truncation():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0.7

    calc = UMACalculator.__new__(UMACalculator)
    calc._task_name = "omol"
    with pytest.raises(ValueError, match="integer atoms.info\\['charge'\\]"):
        calc._warn_unvalidated_charge_spin(atoms)


def test_uma_first_omol_calculate_passes_charge_spin_to_fairchem(monkeypatch):
    from maple.function.calculator.uma import _uma_calculator as uma_module

    calls = {}

    def fake_fairchem_calculate(self, atoms, properties=None, system_changes=None):
        calls["info"] = dict(atoms.info)
        self.results = {"energy": 0.0}

    monkeypatch.setattr(uma_module.FAIRChemCalculator, "calculate", fake_fairchem_calculate)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 1
    atoms.info["mult"] = 2

    calc = uma_module.UMACalculator.__new__(uma_module.UMACalculator)
    calc._auto_task = True
    calc._task_name = "omol"
    calc.solvent_correction = None
    calc.results = {}

    with pytest.warns(RuntimeWarning, match="charged/open-shell"):
        calc.calculate(atoms, properties=["energy"], system_changes=[])

    assert calls["info"]["charge"] == 1
    assert calls["info"]["spin"] == 2


def test_uma_stress_and_virial_requests_fail_before_backend_call():
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    calc = UMACalculator.__new__(UMACalculator)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(NotImplementedError, match="stress/virial"):
        calc.calculate(atoms, properties=["stress"], system_changes=[])


def test_aimnet_batch_prepare_validates_fixed_padding_and_pbc():
    import torch

    from maple.function.calculator.aimnet._aimnet2_batch_calculator import AIMNet2BatchCalc

    calc = AIMNet2BatchCalc.__new__(AIMNet2BatchCalc)
    calc.device = torch.device("cpu")
    calc.dtype = torch.float64
    atoms = [Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])]

    with pytest.raises(ValueError, match="too small"):
        calc.prepare(atoms, fixed_nmax=3)
    with pytest.raises(ValueError, match="multiple of 3"):
        calc.prepare(atoms, fixed_nmax=7)

    periodic = [Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)]
    with pytest.raises(NotImplementedError, match="no-PBC batch"):
        calc.prepare(periodic)
