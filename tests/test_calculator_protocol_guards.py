import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.calculator.calculator_base import (
    CalcABC,
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

    def __init__(self):
        pass


class DummyCalcABC(CalcABC):
    implemented_properties = ["energy"]


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


def test_no_pbc_model_rejected_by_factory_before_construction(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    calc = SetCalculator("cpu", "dummy", str(tmp_path / "maple.out"), atoms=atoms)
    with pytest.raises(NotImplementedError, match="no-PBC molecular wrapper"):
        calc._validate_against_class(DummyNoPBC)


def test_no_pbc_calcabc_rejects_direct_periodic_calculate():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    assert atoms_has_pbc(atoms)
    with pytest.raises(NotImplementedError, match="no-PBC molecular wrapper"):
        DummyCalcABC().calculate(atoms, properties=["energy"])


def test_aimnet_ewald_option_rejected_by_capability_gate(tmp_path):
    calc = SetCalculator(
        "cpu",
        "aimnet2",
        str(tmp_path / "maple.out"),
        model_options={"coulomb_method": "ewald"},
    )
    with pytest.raises(NotImplementedError, match="ewald"):
        calc._validate_against_class(DummyNoPBC)


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
