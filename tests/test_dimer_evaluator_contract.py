import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from scipy.linalg import eigh

from maple.function.dispatcher.ts.algorithm.dimer import (
    Dimer,
    DimerConvergenceError,
    DimerParams,
    DimerStatus,
)
from maple.function.utility.rigid_body import mass_weighted_rigid_basis


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]
    hvp_energy_kind = "free_energy"

    def __init__(self, hessian, *, analytic=False, bad_hvp=None):
        super().__init__()
        self.hessian = np.asarray(hessian, dtype=np.float64)
        self.supports_hvp = analytic
        self.bad_hvp = bad_hvp
        self.hvp_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
        energy = 0.5 * float(x @ self.hessian @ x)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": (-(self.hessian @ x)).reshape(-1, 3),
        }

    def get_hvp(self, atoms, direction):
        self.hvp_calls += 1
        if not self.supports_hvp:
            raise AssertionError("undeclared get_hvp must not be called")
        x = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
        force = -(self.hessian @ x)
        energy = 0.5 * float(x @ self.hessian @ x)
        hn = self.hessian @ np.asarray(direction, dtype=np.float64).reshape(-1)
        if self.bad_hvp == "shape":
            hn = hn[:-1]
        elif self.bad_hvp == "nonfinite":
            hn[0] = np.nan
        return hn, force, energy


class EnergyOnlyQuadratic(QuadraticCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, hessian, *, declares_equal):
        super().__init__(hessian, analytic=False)
        if declares_equal:
            self.energy_free_energy_equal = True

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results.pop("free_energy")


class DistinctEnergyQuadratic(QuadraticCalculator):
    def __init__(self, hessian):
        super().__init__(hessian, analytic=False)
        self.requests = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.requests.append(tuple(properties))
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] += 10.0
        self.results["free_energy"] += 20.0


class SuccessfulRefinementDimer:
    method = "dimer"
    extension = ".xyz"

    def __init__(self, output, atoms_init=None, atoms=None, paras=None):
        self.output = output
        self.atoms = atoms_init if atoms_init is not None else atoms
        self.result = None

    def run(self):
        base = str(Path(self.output).with_suffix(""))
        structure_path = (
            base + f"_{self.method}_ts_candidate" + self.extension
        )
        Path(structure_path).write_text("child-managed candidate\n")
        self.result = SimpleNamespace(structure_path=structure_path)
        return self.atoms


def make_dimer(tmp_path, calc, **params):
    atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
    defaults = {
        "remove_rigid": False,
        "n_init": "given",
        "n_given": np.asarray([1.0, 0.0, 0.0]),
    }
    defaults.update(params)
    return Dimer(
        output=str(tmp_path / "dimer.out"),
        atoms_init=atoms,
        paras={"dimer": defaults},
    )


def make_mass_dimer(tmp_path, hessian, *, analytic, **params):
    calc = QuadraticCalculator(hessian, analytic=analytic)
    atoms = Atoms("HO", positions=np.zeros((2, 3)), calculator=calc)
    defaults = {
        "use_hvp": analytic,
        "use_mass_weight": True,
        "remove_rigid": False,
        "n_init": "given",
        "n_given": [-4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
    defaults.update(params)
    return Dimer(
        output=str(tmp_path / ("mass-analytic.out" if analytic else "mass-fd.out")),
        atoms_init=atoms,
        paras={"dimer": defaults},
    )


@pytest.mark.parametrize("analytic", [False, True])
def test_mass_weighted_spd_counterexample_is_not_a_candidate(tmp_path, analytic):
    hessian = np.eye(6)
    hessian[0, 3] = hessian[3, 0] = 0.9
    dimer = make_mass_dimer(
        tmp_path,
        hessian,
        analytic=analytic,
        rot_alpha=0.001,
        max_iter=1,
    )

    result = dimer.run_result()

    assert result.status is DimerStatus.FAILED_NONNEGATIVE_CURVATURE
    assert not Path(str(Path(dimer.output).with_suffix("")) + "_dimer_ts_candidate.xyz").exists()
    log = Path(dimer.output).read_text()
    assert "Cartesian trust radius" in log
    assert "Ha/(Angstrom^2 amu)" in log


def test_mass_weighted_generalized_eigenmode_has_zero_rotation_residual(tmp_path):
    hessian = np.diag([-0.4, 0.7, 1.1, 1.8, 2.2, 2.9])
    hessian[0, 3] = hessian[3, 0] = 0.2
    dimer = make_mass_dimer(tmp_path, hessian, analytic=True)
    metric = np.diag(dimer.M)
    eigenvalues, eigenvectors = eigh(hessian, metric)
    mode = eigenvectors[:, 0]

    hn = hessian @ mode
    residual, max_residual, rms_residual = dimer._rotational_metrics(hn, mode)

    assert np.dot(mode, hn) == pytest.approx(eigenvalues[0], abs=1.0e-13)
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-13)
    assert max_residual == pytest.approx(0.0, abs=1.0e-13)
    assert rms_residual == pytest.approx(0.0, abs=1.0e-13)

    dimer.params.rot_alpha = 0.05
    dimer.params.rot_max_iter = 2000
    dimer.params.rot_f_max_th = 1.0e-11
    dimer.params.rot_f_rms_th = 1.0e-12
    rotated, _, _ = dimer._rotate_minimize_kappa(dimer.n)
    overlap = abs(np.dot(dimer.M * rotated, mode))
    assert overlap == pytest.approx(1.0, abs=1.0e-11)
    assert dimer._curvature(hessian @ rotated, rotated) == pytest.approx(
        eigenvalues[0], abs=1.0e-12
    )


def test_mass_weighted_translation_matches_explicit_q_space_reference(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dimer = make_mass_dimer(tmp_path, hessian, analytic=True)
    force = np.asarray([0.3, -0.2, 0.5, -0.7, 0.11, -0.13])
    q_direction = np.sqrt(dimer.M) * dimer.n
    q_force = force / np.sqrt(dimer.M)
    q_parallel = np.dot(q_force, q_direction) * q_direction

    expected_no_flip = (q_force - q_parallel) / np.sqrt(dimer.M)
    expected_flip = (q_force - 2.0 * q_parallel) / np.sqrt(dimer.M)

    np.testing.assert_allclose(
        dimer._translation_force(force, dimer.n, mode_flip=False),
        expected_no_flip,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        dimer._translation_force(force, dimer.n, mode_flip=True),
        expected_flip,
        atol=1.0e-14,
    )


def test_unit_masses_match_unweighted_math_when_rigid_projection_is_off(tmp_path):
    hessian = np.asarray(
        [[-1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 3.0]]
    )
    calc_a = QuadraticCalculator(hessian, analytic=True)
    calc_b = QuadraticCalculator(hessian, analytic=True)
    atoms_a = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc_a)
    atoms_b = atoms_a.copy()
    atoms_b.calc = calc_b
    atoms_a.set_masses([1.0])
    atoms_b.set_masses([1.0])
    common = {
        "use_hvp": True,
        "remove_rigid": False,
        "n_init": "given",
        "n_given": [1.0, 2.0, -1.0],
    }
    weighted = Dimer(
        str(tmp_path / "weighted.out"), atoms_a,
        {"dimer": {**common, "use_mass_weight": True}},
    )
    unweighted = Dimer(
        str(tmp_path / "unweighted.out"), atoms_b,
        {"dimer": {**common, "use_mass_weight": False}},
    )
    force = np.asarray([0.3, -0.2, 0.5])
    hn = hessian @ weighted.n

    np.testing.assert_allclose(weighted.n, unweighted.n, atol=0.0)
    np.testing.assert_allclose(
        weighted._rotational_metrics(hn, weighted.n)[0],
        unweighted._rotational_metrics(hn, unweighted.n)[0],
        atol=0.0,
    )
    np.testing.assert_allclose(
        weighted._translation_force(force, weighted.n, True),
        unweighted._translation_force(force, unweighted.n, True),
        atol=0.0,
    )


def test_mass_weighted_force_initialization_uses_inverse_mass(tmp_path):
    hessian = np.eye(6)
    calc = QuadraticCalculator(hessian, analytic=False)
    atoms = Atoms("HO", positions=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], calculator=calc)
    dimer = Dimer(
        str(tmp_path / "force-init.out"),
        atoms,
        {"dimer": {
            "use_mass_weight": True,
            "remove_rigid": False,
            "n_init": "force",
        }},
    )
    expected = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / dimer.M
    expected /= np.sqrt(np.dot(dimer.M * expected, expected))
    np.testing.assert_allclose(dimer.n, expected, atol=1.0e-14)


def test_weighted_rigid_projection_uses_mass_orthogonal_basis(tmp_path):
    atoms = Atoms(
        "HOH",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.3]],
        calculator=QuadraticCalculator(np.eye(9), analytic=False),
    )
    dimer = Dimer(
        str(tmp_path / "rigid.out"),
        atoms,
        {"dimer": {
            "use_mass_weight": True,
            "remove_rigid": True,
            "n_init": "given",
            "n_given": np.arange(1.0, 10.0),
        }},
    )
    weighted_direction = np.sqrt(dimer.M) * dimer.n
    rigid = mass_weighted_rigid_basis(atoms)
    np.testing.assert_allclose(rigid.T @ weighted_direction, 0.0, atol=1.0e-13)


@pytest.mark.parametrize("bad_mass", [0.0, -1.0, np.nan, np.inf])
def test_invalid_weighted_mass_fails_before_evaluation_or_output(tmp_path, bad_mass):
    calc = QuadraticCalculator(np.eye(6), analytic=False)
    atoms = Atoms("HO", positions=np.zeros((2, 3)), calculator=calc)
    atoms.set_masses([bad_mass, 16.0])
    output = tmp_path / "bad-mass.out"

    with pytest.raises(ValueError, match="finite positive masses"):
        Dimer(
            str(output),
            atoms,
            {"dimer": {
                "use_mass_weight": True,
                "remove_rigid": False,
                "n_init": "given",
                "n_given": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }},
        )

    assert calc.hvp_calls == 0
    assert calc.results == {}
    assert not output.exists()


def test_mass_weighting_remains_opt_in():
    assert DimerParams().use_mass_weight is False


def test_fd_branch_uses_force_pairs_and_center_without_calling_get_hvp(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0])
    calc = QuadraticCalculator(hessian, analytic=False)
    dimer = make_dimer(tmp_path, calc, use_hvp=False, delta=1.0e-4)

    hn, forces, energy = dimer._evaluate_current(dimer.n)

    x = dimer.atoms.positions.reshape(-1)
    np.testing.assert_allclose(hn, hessian @ dimer.n, atol=1.0e-11)
    np.testing.assert_allclose(forces, -(hessian @ x), atol=1.0e-13)
    assert energy == pytest.approx(0.5 * x @ hessian @ x)
    assert calc.hvp_calls == 0


def test_fd_generic_energy_only_calculator_requires_explicit_equality(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0])
    undeclared = make_dimer(
        tmp_path,
        EnergyOnlyQuadratic(hessian, declares_equal=False),
        use_hvp=False,
    )
    with pytest.raises(RuntimeError, match="energy_free_energy_equal=True"):
        undeclared._evaluate_current(undeclared.n)

    declared = make_dimer(
        tmp_path,
        EnergyOnlyQuadratic(hessian, declares_equal=True),
        use_hvp=False,
    )
    hn, forces, energy = declared._evaluate_current(declared.n)
    np.testing.assert_allclose(hn, hessian @ declared.n, atol=1.0e-11)
    assert np.all(np.isfinite(forces))
    assert np.isfinite(energy)


def test_analytic_branch_uses_declared_hvp(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0])
    calc = QuadraticCalculator(hessian, analytic=True)
    dimer = make_dimer(tmp_path, calc, use_hvp=True)

    hn, forces, energy = dimer._evaluate_current(dimer.n)

    np.testing.assert_allclose(hn, hessian @ dimer.n)
    assert np.all(np.isfinite(forces))
    assert np.isfinite(energy)
    assert calc.hvp_calls == 1


def test_explicit_hvp_callable_preserves_hn_only_contract(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0])
    calc = QuadraticCalculator(hessian, analytic=False)
    calls = []

    def hvp_fn(atoms, direction):
        calls.append(atoms.positions.copy())
        return hessian @ direction

    atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
    dimer = Dimer(
        output=str(tmp_path / "callable.out"),
        atoms_init=atoms,
        paras={"dimer": {
            "use_hvp": True,
            "remove_rigid": False,
            "n_init": "given",
            "n_given": [1.0, 0.0, 0.0],
        }},
        hvp_fn=hvp_fn,
    )

    hn, forces, energy = dimer._evaluate_current(dimer.n)

    np.testing.assert_allclose(hn, hessian @ dimer.n)
    np.testing.assert_allclose(forces, -(hessian @ atoms.positions.reshape(-1)))
    assert energy == pytest.approx(0.5 * atoms.positions.reshape(-1) @ hessian @ atoms.positions.reshape(-1))
    assert len(calls) == 1
    assert calc.hvp_calls == 0


def test_missing_analytic_hvp_fails_before_output_or_calculation(tmp_path):
    calc = QuadraticCalculator(np.eye(3), analytic=False)
    output = tmp_path / "missing.out"
    atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)

    with pytest.raises(NotImplementedError, match="analytic HVP"):
        Dimer(
            output=str(output),
            atoms_init=atoms,
            paras={"dimer": {
                "use_hvp": True,
                "remove_rigid": False,
                "n_init": "given",
                "n_given": [1.0, 0.0, 0.0],
            }},
        )

    assert not output.exists()
    assert calc.hvp_calls == 0
    assert calc.results == {}


@pytest.mark.parametrize("bad_hvp, match", [("shape", "components"), ("nonfinite", "non-finite")])
def test_analytic_hvp_rejects_wrong_shape_and_nonfinite(tmp_path, bad_hvp, match):
    calc = QuadraticCalculator(np.eye(3), analytic=True, bad_hvp=bad_hvp)
    dimer = make_dimer(tmp_path, calc, use_hvp=True)
    with pytest.raises((ValueError, FloatingPointError), match=match):
        dimer._evaluate_current(dimer.n)


def test_written_energy_and_forces_are_for_written_post_step_point(tmp_path):
    hessian = np.diag([-1.0, 2.0, 3.0])
    calc = QuadraticCalculator(hessian, analytic=False)
    dimer = make_dimer(
        tmp_path,
        calc,
        use_hvp=False,
        max_iter=1,
        rot_max_iter=1,
        step0=0.05,
        step_max=0.05,
        trust_radius=0.05,
        f_max_th=10.0,
        f_rms_th=10.0,
    )
    dimer.atoms.dp_max_th = 10.0
    dimer.atoms.dp_rms_th = 10.0

    returned = dimer.run()
    result = dimer.result

    assert returned is dimer.atoms
    assert result.status is DimerStatus.GEOMETRY_CONVERGED
    written = Path(result.structure_file)
    assert written.name.endswith("_dimer_ts_candidate.xyz")
    lines = written.read_text().splitlines()
    written_energy = float(lines[1].split("=")[1])
    written_xyz = np.asarray([float(v) for v in lines[2].split()[1:]])
    expected_energy = 0.5 * float(written_xyz @ hessian @ written_xyz)
    expected_forces = -(hessian @ written_xyz)
    assert written_energy == pytest.approx(expected_energy, abs=1.0e-9)
    assert dimer.atoms.max_f == pytest.approx(np.linalg.norm(expected_forces))


def test_max_iterations_writes_diagnostic_not_ts_candidate(tmp_path):
    calc = QuadraticCalculator(np.diag([-1.0, 2.0, 3.0]), analytic=False)
    dimer = make_dimer(tmp_path, calc, use_hvp=False, max_iter=0)
    stale_candidate = tmp_path / "dimer_dimer_ts_candidate.xyz"
    stale_candidate.write_text("stale candidate\n")

    result = dimer.run_result()

    assert result.status is DimerStatus.FAILED_MAX_ITER
    assert Path(result.structure_file).name.endswith("_dimer_failed_max_iter.xyz")
    assert Path(result.structure_file).exists()
    assert not (tmp_path / "dimer_dimer_ts.xyz").exists()
    assert not stale_candidate.exists()


def test_force_converged_minimum_is_diagnostic_not_ts_candidate(tmp_path):
    calc = QuadraticCalculator(np.eye(3), analytic=False)
    dimer = make_dimer(
        tmp_path,
        calc,
        use_hvp=False,
        max_iter=1,
        rot_max_iter=1,
        f_max_th=10.0,
        f_rms_th=10.0,
    )
    dimer.atoms.dp_max_th = 10.0
    dimer.atoms.dp_rms_th = 10.0

    dimer.run_result()

    assert dimer.result.status is DimerStatus.FAILED_NONNEGATIVE_CURVATURE
    assert Path(dimer.result.structure_file).name.endswith(
        "_dimer_failed_nonnegative_curvature.xyz"
    )
    assert not (tmp_path / "dimer_dimer_ts_candidate.xyz").exists()


@pytest.mark.parametrize(
    "hessian, expected_status",
    [
        (np.diag([-1.0, 2.0, 3.0]), DimerStatus.FAILED_MAX_ITER),
        (np.eye(3), DimerStatus.FAILED_NONNEGATIVE_CURVATURE),
    ],
)
def test_run_raises_with_noncandidate_result(tmp_path, hessian, expected_status):
    params = {"use_hvp": False, "max_iter": 0}
    if expected_status is DimerStatus.FAILED_NONNEGATIVE_CURVATURE:
        params.update(max_iter=1, f_max_th=10.0, f_rms_th=10.0)
    dimer = make_dimer(tmp_path, QuadraticCalculator(hessian), **params)
    if expected_status is DimerStatus.FAILED_NONNEGATIVE_CURVATURE:
        dimer.atoms.dp_max_th = 10.0
        dimer.atoms.dp_rms_th = 10.0

    with pytest.raises(DimerConvergenceError) as exc_info:
        dimer.run()

    assert exc_info.value.result.status is expected_status
    assert exc_info.value.result is dimer.result


def test_dmf_does_not_promote_failed_dimer_refinement(tmp_path, monkeypatch):
    class FakeIpoptProblem:
        pass

    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=FakeIpoptProblem),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    try:
        dmf_module = importlib.import_module(module_name)
        calc = QuadraticCalculator(np.diag([-1.0, 2.0, 3.0]))
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine="dimer")
        dmf._paras = {
            "dimer": {
                "use_hvp": False,
                "remove_rigid": False,
                "n_init": "given",
                "n_given": [1.0, 0.0, 0.0],
                "max_iter": 0,
            }
        }
        dmf._refinement_paras = dmf._paras

        result = dmf._refine_ts(atoms, calc, str(tmp_path / "path"))

        assert result is None
        assert not (tmp_path / "path_dmf_refine_dimer_ts_candidate.xyz").exists()
        assert "failed_max_iter" in Path(dmf.output).read_text()
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


@pytest.mark.parametrize("method", ["prfo", "dimer"])
@pytest.mark.parametrize("extension", [".xyz", ".pdb"])
def test_dmf_refinement_reuses_child_managed_candidate_and_paired_values(
    tmp_path, monkeypatch, extension, method
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    child_module = importlib.import_module(
        "maple.function.dispatcher.ts.algorithm."
        + ("PRFO" if method == "prfo" else "dimer")
    )
    child_name = "PRFO" if method == "prfo" else "Dimer"
    monkeypatch.setattr(child_module, child_name, SuccessfulRefinementDimer)
    monkeypatch.setattr(SuccessfulRefinementDimer, "method", method)
    monkeypatch.setattr(SuccessfulRefinementDimer, "extension", extension)
    try:
        dmf_module = importlib.import_module(module_name)
        calc = DistinctEnergyQuadratic(np.diag([-1.0, 2.0, 3.0]))
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine=method)
        dmf._paras = {method: {}}
        dmf._refinement_paras = dmf._paras

        result = dmf._refine_ts(atoms, calc, str(tmp_path / "path"))

        expected_free = 20.0 + 0.5 * float(
            atoms.positions.reshape(-1)
            @ np.diag([-1.0, 2.0, 3.0])
            @ atoms.positions.reshape(-1)
        )
        assert result is not None
        assert result["energy"] == pytest.approx(expected_free)
        assert calc.requests[-1] == ("free_energy", "forces")
        child_candidate = tmp_path / f"dmf_{method}_ts_candidate{extension}"
        assert result["ts_file"] == str(child_candidate)
        assert child_candidate.exists()
        assert not (
            tmp_path / f"path_dmf_refine_{method}_ts_candidate.xyz"
        ).exists()
        assert not (
            tmp_path / f"path_dmf_refine_{method}_ts_candidate.pdb"
        ).exists()
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_dmf_refinement_rejects_missing_free_energy_without_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    dimer_module = importlib.import_module(
        "maple.function.dispatcher.ts.algorithm.dimer"
    )
    monkeypatch.setattr(dimer_module, "Dimer", SuccessfulRefinementDimer)
    try:
        dmf_module = importlib.import_module(module_name)
        calc = EnergyOnlyQuadratic(
            np.diag([-1.0, 2.0, 3.0]),
            declares_equal=False,
        )
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine="dimer")
        dmf._paras = {"dimer": {}}
        dmf._refinement_paras = dmf._paras

        result = dmf._refine_ts(atoms, calc, str(tmp_path / "path"))

        assert result is None
        assert not (tmp_path / "path_dmf_refine_dimer_ts_candidate.xyz").exists()
        assert "did not return the requested 'free_energy' scalar" in Path(
            dmf.output
        ).read_text()
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_dmf_failed_refinement_archives_all_owned_legacy_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    dimer_module = importlib.import_module(
        "maple.function.dispatcher.ts.algorithm.dimer"
    )

    class FailedRefinementDimer:
        def __init__(self, output, atoms_init, paras=None):
            pass

        def run(self):
            raise RuntimeError("synthetic refinement failure")

    monkeypatch.setattr(dimer_module, "Dimer", FailedRefinementDimer)
    base = tmp_path / "path"
    stale = {}
    for method in ("prfo", "dimer"):
        for suffix in ("_ts", "_ts_candidate"):
            for extension in (".xyz", ".pdb"):
                path = Path(f"{base}_dmf_refine_{method}{suffix}{extension}")
                payload = f"{method}{suffix}{extension}".encode()
                path.write_bytes(payload)
                stale[path] = payload
    try:
        dmf_module = importlib.import_module(module_name)
        calc = DistinctEnergyQuadratic(np.diag([-1.0, 2.0, 3.0]))
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine="dimer")
        dmf._paras = {"dimer": {}}
        dmf._refinement_paras = dmf._paras

        assert dmf._refine_ts(atoms, calc, str(base)) is None

        archive = base.parent / "path_dmf_refine_history" / "run-0001"
        assert archive.is_dir()
        for path, payload in stale.items():
            assert not path.exists()
            assert (archive / path.name).read_bytes() == payload
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_dmf_run_archives_refinement_artifacts_when_refine_is_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    old_candidate = tmp_path / "dmf_prfo_ts_candidate.xyz"
    old_candidate.write_bytes(b"old child candidate")
    try:
        dmf_module = importlib.import_module(module_name)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine=None)
        dmf.input_images = []

        with pytest.raises(ValueError, match="at least two structures"):
            dmf.run()

        archived = (
            tmp_path
            / "dmf_dmf_refine_history"
            / "run-0001"
            / old_candidate.name
        )
        assert not old_candidate.exists()
        assert archived.read_bytes() == b"old child candidate"
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_dmf_run_archives_refinement_artifacts_before_early_path_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    old_candidate = tmp_path / "dmf_dmf_refine_dimer_ts_candidate.pdb"
    old_candidate.write_bytes(b"old DMF alias")
    try:
        dmf_module = importlib.import_module(module_name)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine="dimer")
        dmf.input_images = [Atoms("H"), Atoms("H")]

        with pytest.raises(ValueError, match="requires a calculator"):
            dmf.run()

        archived = (
            tmp_path
            / "dmf_dmf_refine_history"
            / "run-0001"
            / old_candidate.name
        )
        assert not old_candidate.exists()
        assert archived.read_bytes() == b"old DMF alias"
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_dmf_switching_refiner_archives_old_child_candidate(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "cyipopt",
        SimpleNamespace(Problem=type("FakeIpoptProblem", (), {})),
    )
    module_name = "maple.function.dispatcher.ts.algorithm.dmf"
    package = importlib.import_module("maple.function.dispatcher.ts.algorithm")
    dimer_module = importlib.import_module(
        "maple.function.dispatcher.ts.algorithm.dimer"
    )
    monkeypatch.setattr(dimer_module, "Dimer", SuccessfulRefinementDimer)
    old_candidate = tmp_path / "dmf_prfo_ts_candidate.pdb"
    old_candidate.write_bytes(b"old PRFO child candidate")
    try:
        dmf_module = importlib.import_module(module_name)
        calc = DistinctEnergyQuadratic(np.diag([-1.0, 2.0, 3.0]))
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dmf = dmf_module.DMF.__new__(dmf_module.DMF)
        dmf.output = str(tmp_path / "dmf.out")
        dmf.params = SimpleNamespace(refine="dimer")
        dmf._paras = {"dimer": {}}
        dmf._refinement_paras = dmf._paras

        result = dmf._refine_ts(atoms, calc, str(tmp_path / "dmf"))

        assert result is not None
        archived = (
            tmp_path
            / "dmf_dmf_refine_history"
            / "run-0001"
            / old_candidate.name
        )
        assert not old_candidate.exists()
        assert archived.read_bytes() == b"old PRFO child candidate"
        assert Path(result["ts_file"]).name == "dmf_dimer_ts_candidate.xyz"
    finally:
        sys.modules.pop(module_name, None)
        if getattr(package, "dmf", None) is locals().get("dmf_module"):
            delattr(package, "dmf")


def test_prior_dimer_artifacts_are_archived_with_bytes_preserved(tmp_path):
    calc = QuadraticCalculator(np.diag([-1.0, 2.0, 3.0]))
    dimer = make_dimer(
        tmp_path,
        calc,
        use_hvp=False,
        max_iter=0,
        save_traj=False,
    )
    stale = {
        tmp_path / "dimer_dimer_ts.xyz": b"legacy xyz\x00\xff",
        tmp_path / "dimer_dimer_ts_candidate.pdb": b"candidate pdb\n",
        tmp_path / "dimer_dimer_traj.xyz": b"trajectory xyz\n",
        tmp_path / "dimer_dimer_failed_max_iter.pdb": b"failure pdb\n",
    }
    for path, payload in stale.items():
        path.write_bytes(payload)

    dimer.run_result()

    run_dirs = list((tmp_path / "dimer_dimer_history").glob("run-*"))
    assert len(run_dirs) == 1
    archived = run_dirs[0]
    for path, payload in stale.items():
        assert not path.exists()
        assert (archived / path.name).read_bytes() == payload

    active_failure = tmp_path / "dimer_dimer_failed_max_iter.xyz"
    first_failure_bytes = active_failure.read_bytes()
    dimer.run_result()
    second_archive = tmp_path / "dimer_dimer_history" / "run-0002"
    assert (second_archive / active_failure.name).read_bytes() == first_failure_bytes


def test_constraints_fail_closed_before_output(tmp_path):
    calc = QuadraticCalculator(np.eye(6), analytic=False)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], calculator=calc)
    atoms.set_constraint(FixAtoms(indices=[0]))
    output = tmp_path / "constrained.out"

    with pytest.raises(NotImplementedError, match="constrained Dimer"):
        Dimer(output=str(output), atoms_init=atoms)

    assert not output.exists()


def test_unknown_parameter_fails_before_output_or_calculation(tmp_path):
    calc = QuadraticCalculator(np.eye(3), analytic=False)
    output = tmp_path / "typo.out"
    atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)

    with pytest.raises(ValueError, match="Unknown Dimer parameter.*use_hvp"):
        Dimer(
            output=str(output),
            atoms_init=atoms,
            paras={"dimer": {"use_hvpp": False}},
        )

    assert not output.exists()
    assert calc.results == {}


def test_fd_dimer_module_import_does_not_require_torch(monkeypatch):
    module_name = "maple.function.dispatcher.ts.algorithm.dimer"
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    old_module = sys.modules.pop(module_name)
    try:
        reloaded = importlib.import_module(module_name)
        calc = QuadraticCalculator(np.diag([-1.0, 2.0, 3.0]), analytic=False)
        atoms = Atoms("H", positions=[[0.2, 0.1, -0.1]], calculator=calc)
        dimer = reloaded.Dimer(
            output="unused.out",
            atoms_init=atoms,
            paras={"dimer": {
                "use_hvp": False,
                "remove_rigid": False,
                "n_init": "given",
                "n_given": [1.0, 0.0, 0.0],
            }},
        )
        hn, _, _ = dimer._evaluate_current(dimer.n)
        np.testing.assert_allclose(hn, [-1.0, 0.0, 0.0], atol=1.0e-11)
    finally:
        sys.modules[module_name] = old_module
