"""Nonlinear scan constraints must not be treated as Cartesian RFO degrees of freedom."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixInternals
from ase.io import read

from maple.function.calculator.calculator_base import CalcABC
from maple.function.dispatcher.optimization.algorithm.RFO import RFO
from maple.function.dispatcher.scan.scan import Scan


class _QuadraticCalculator(CalcABC):
    """Deterministic E/F/H probe with no model weights or optional backend."""

    implemented_properties = ["energy", "free_energy", "forces"]
    MODEL_NAMES = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False

    def __init__(self):
        super().__init__()
        self.hessian = "numerical"
        self.calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        x = atoms.get_positions().reshape(-1)
        energy = float(0.5 * x @ x)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -x.reshape(-1, 3),
        }


def _bent_triatomic():
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.9, 0.0]])
    atoms.calc = _QuadraticCalculator()
    return atoms


@pytest.mark.parametrize("max_iter", [0, 1])
def test_rfo_rejects_fixinternals_before_evaluation_or_output(tmp_path, max_iter):
    atoms = _bent_triatomic()
    atoms.set_constraint(FixInternals(bonds=[[1.0, [0, 1]]]))
    before = atoms.get_positions().copy()
    output = tmp_path / "direct.out"

    optimizer = RFO(atoms, str(output), {"max_iter": max_iter})
    with pytest.raises(NotImplementedError, match=r"FixInternals.*(?:LBFGS|lbfgs)"):
        optimizer.run()

    assert atoms.calc.calls == 0
    np.testing.assert_array_equal(atoms.get_positions(), before)
    assert not list(tmp_path.iterdir())


def test_relaxed_rfo_scan_rejects_before_evaluation_or_trajectory(tmp_path):
    atoms = _bent_triatomic()
    before = atoms.get_positions().copy()
    output = tmp_path / "scan.out"
    scan = Scan(
        str(output), atoms, method="rfo", constraints=[[1, 2, 0.0, 0]],
        params={"mode": "relaxed", "max_iter": 1},
    )

    with pytest.raises(NotImplementedError, match=r"FixInternals.*(?:LBFGS|lbfgs)"):
        scan.run_scan()

    assert atoms.calc.calls == 0
    np.testing.assert_array_equal(atoms.get_positions(), before)
    assert not (tmp_path / "scan_opt_traj.xyz").exists()
    assert not (tmp_path / "scan_opt.xyz").exists()
    assert not (tmp_path / "scan_scan_final.xyz").exists()


@pytest.mark.parametrize(
    "constraint", [FixAtoms(indices=[0]), FixCartesian(0, mask=(True, False, False))]
)
def test_fixed_cartesian_constraints_remain_supported(tmp_path, constraint):
    atoms = _bent_triatomic()
    atoms.set_constraint(constraint)
    optimizer = RFO(atoms, str(tmp_path / "fixed.out"), {"max_iter": 0})

    assert optimizer.run() is atoms
    assert atoms.calc.calls == 1
    assert optimizer._calculate_hessian(atoms).shape == (9, 9)


def test_explicit_lbfgs_relaxed_scan_remains_available(tmp_path):
    atoms = _bent_triatomic()
    scan = Scan(
        str(tmp_path / "scan.out"), atoms, method="lbfgs",
        constraints=[[1, 2, 0.0, 0]],
        params={"mode": "relaxed", "max_iter": 1},
    )

    assert scan.run_scan() is None

    assert atoms.calc.calls > 0
    assert "Scanning combination 1/1" in (tmp_path / "scan_scan_final.xyz").read_text()
    final = read(tmp_path / "scan_scan_final.xyz")
    assert final.get_distance(0, 1) == pytest.approx(1.0, abs=1e-8)
    assert not np.array_equal(final.get_positions(), atoms.get_positions())
