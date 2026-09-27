"""Analytic regression locks for the PRFO acceptance and output contracts."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.ts.algorithm.PRFO import PRFO, prfo_step


class Quadratic(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces']
    rigid_body_invariant = False

    def __init__(self, diagonal=(-1.0, 1.0, 1.0), *, bad_trial=False):
        super().__init__()
        self.H = np.diag(diagonal)
        self.bad_trial = bad_trial

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions.ravel()
        energy = float(0.5 * x @ self.H @ x)
        forces = (-self.H @ x).reshape(-1, 3)
        if self.bad_trial and x[0] < 0.1 - 1e-12:
            forces = np.ones_like(forces)
        self.results = dict(energy=energy, free_energy=energy, forces=forces)

    def get_hessian(self, atoms):
        return self.H.copy()


def run_at(directory, x, *, name='job', calc=None, max_iter=20, **params):
    atoms = Atoms('H', positions=[x], masses=[1.0], calculator=calc or Quadratic())
    result = PRFO(atoms=atoms, output=str(Path(directory) / f'{name}.out'),
                  paras=dict(max_iter=max_iter, **params)).run_result()
    return atoms, result


def test_exact_cancelling_saddle_step_is_not_false_stagnation(tmp_path):
    atoms, result = run_at(tmp_path, [0.1, 0.1, 0.0], f_max_th=1e-6, f_rms_th=1e-6)
    assert result.geometry_converged
    assert result.iterations > 0
    assert np.max(np.abs(atoms.get_forces())) < 1e-6


def test_cancellation_does_not_accept_inconsistent_trial_gradient(tmp_path):
    atoms, result = run_at(tmp_path, [0.1, 0.1, 0.0], calc=Quadratic(bad_trial=True))
    assert not result.geometry_converged
    np.testing.assert_array_equal(atoms.positions, [[0.1, 0.1, 0.0]])


@pytest.mark.parametrize('diagonal', [(-1, 1e-8, 1), (-1e-8, 1, 1), (1e-8, 1, 1)])
def test_stationary_near_zero_mode_has_explicit_status(tmp_path, diagonal):
    _, result = run_at(tmp_path, [0, 0, 0], calc=Quadratic(diagonal))
    assert result.status.value == 'failed_near_zero_curvature'
    assert result.iterations == 0
    assert 'near_zero_curvature' in result.structure_path


def test_previous_candidate_is_archived_before_failed_rerun(tmp_path):
    _, first = run_at(tmp_path, [0, 0, 0])
    candidate = Path(first.structure_path)
    previous_bytes = candidate.read_bytes()
    _, second = run_at(tmp_path, [0.1, 0.1, 0], max_iter=0)
    assert not second.geometry_converged
    assert not candidate.exists()
    archived = list(tmp_path.glob('job_prfo_history/*/job_prfo_ts_candidate.xyz'))
    assert len(archived) == 1
    assert archived[0].read_bytes() == previous_bytes


def test_repeated_run_does_not_append_to_previous_trajectory(tmp_path):
    for _ in range(2):
        _, result = run_at(tmp_path, [0.1, 0, 0])
        assert result.geometry_converged
    lines = (tmp_path / 'job_prfo_traj.xyz').read_text().splitlines()
    assert sum(line.startswith('Iteration') for line in lines) == result.iterations


def test_alpha_loop_does_not_construct_dense_augmented_eigensystems():
    w = np.linspace(0.05, 2.0, 90)
    w[0] = -0.5
    V = np.eye(len(w))
    g = np.sin(np.arange(len(w)) + 1) * 0.05
    with patch('numpy.linalg.eigvalsh', side_effect=AssertionError('dense root solve')):
        step = prfo_step(np.diag(w), g, is_ts=True, trust_radius=0.2, pre_eig=(w, V, g))
    assert np.linalg.norm(step) <= 0.2 + 1e-10


@pytest.mark.parametrize('seed', range(10))
def test_arrowhead_extreme_roots_match_dense_oracle(seed):
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import extremal_root
    rng = np.random.default_rng(seed)
    diagonal = rng.normal(size=12)
    coupling = rng.normal(size=12)
    if seed % 2:
        coupling[::3] = 0.0
        diagonal[1:3] = diagonal[0]
    A = np.zeros((13, 13)); A[1:, 1:] = np.diag(diagonal)
    A[0, 1:] = coupling; A[1:, 0] = coupling
    roots = np.linalg.eigvalsh(A)
    for maximum, expected in [(False, roots[0]), (True, roots[-1])]:
        assert extremal_root(diagonal, coupling, maximum=maximum) == pytest.approx(expected, abs=1e-12, rel=1e-10)


@pytest.mark.parametrize('quality', [-0.001, 0.001, 0.499, 0.501, 0.749, 0.751])
def test_cancelling_energy_quality_has_existing_eta_boundaries(quality):
    from maple.function.dispatcher.ts.algorithm.PRFO import _ts_step_quality
    _, actual = _ts_step_quality(
        2.0 * (1.0 - quality), 0.0, model_scale=2.0,
        gradient=np.array([1.0]), trial_gradient=np.array([0.0]),
        hessian_step=np.array([-1.0]),
    )
    assert actual == pytest.approx(quality, abs=1e-15)


@pytest.mark.parametrize('quality', [0.001, 0.499, 0.501, 0.749, 0.751])
def test_cancelling_gradient_quality_has_existing_eta_boundaries(quality):
    from maple.function.dispatcher.ts.algorithm.PRFO import _ts_step_quality
    _, actual = _ts_step_quality(
        0.0, 0.0, model_scale=2.0,
        gradient=np.array([1.0]), trial_gradient=np.array([1.0 - quality]),
        hessian_step=np.array([-1.0]),
    )
    assert actual == pytest.approx(quality, abs=1e-15)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1.0, 2.0])
def test_cancellation_requires_finite_gradient_progress(bad):
    from maple.function.dispatcher.ts.algorithm.PRFO import _ts_step_quality
    _, quality = _ts_step_quality(
        0.0, 0.0, model_scale=2.0,
        gradient=np.array([1.0]), trial_gradient=np.array([bad]),
        hessian_step=np.array([-1.0]),
    )
    assert quality == -np.inf


def test_scalar_and_experimental_quality_share_the_same_policy():
    torch = pytest.importorskip('torch')
    from maple.function.dispatcher.ts.algorithm.PRFO import _ts_step_quality
    from maple.function.dispatcher.ts.experimental.bprfo import BatchPRFO
    rng = np.random.default_rng(741)
    u, d = rng.normal(size=(2, 60))
    d[::2] = -u[::2]
    actual = u + d + rng.normal(size=60) * 0.01
    g = rng.normal(size=(60, 9))
    hs = -0.9 * g
    gt = g + hs + rng.normal(size=g.shape) * 0.01
    rho, quality = BatchPRFO._model_quality_batched(*[
        torch.tensor(v, dtype=torch.float64) for v in (actual, u, d, g, hs, gt)
    ])
    for i in range(len(u)):
        scalar_rho, scalar_quality = _ts_step_quality(
            actual[i], u[i] + d[i], model_scale=abs(u[i]) + abs(d[i]),
            gradient=g[i], trial_gradient=gt[i], hessian_step=hs[i],
        )
        assert float(quality[i]) == pytest.approx(scalar_quality, abs=1e-14)
        if scalar_rho is None:
            assert torch.isnan(rho[i])
        else:
            assert float(rho[i]) == pytest.approx(scalar_rho, abs=1e-14)


@pytest.mark.parametrize('field', ['max_iter', 'recalc', 'max_bisect_it'])
@pytest.mark.parametrize('value', [True, 1.5, float('nan')])
def test_integer_controls_do_not_silently_truncate(field, value, tmp_path):
    atoms = Atoms('H', positions=[[0, 0, 0]], calculator=Quadratic())
    with pytest.raises(ValueError, match=field):
        PRFO(atoms=atoms, output=str(tmp_path / 'invalid.out'), paras={field: value})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('coupling', [1.0, 1e-100])
def test_deflated_large_pole_does_not_destroy_small_active_root(coupling):
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import extremal_root
    assert extremal_root([0.0, 1e308], [coupling, 0.0]) == pytest.approx(-coupling, rel=1e-14, abs=0)


def test_arrowhead_rescaling_overflow_is_a_solver_failure():
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import extremal_root, ArrowheadSolverError
    with pytest.raises(ArrowheadSolverError):
        extremal_root([0.0, 0.0], [1.7e308, 1.7e308])


def test_positive_target_near_a_pole_keeps_uphill_direction():
    step = prfo_step(np.array([[1.0]]), np.array([1e-12]), is_ts=True, trust_radius=0.2)
    assert step[0] == pytest.approx(0.2, abs=1e-10)


def test_unrepresentable_alpha_cannot_pretend_to_solve_with_zero_step():
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import ArrowheadSolverError
    with pytest.raises(ArrowheadSolverError):
        prfo_step(np.array([[1.0]]), np.array([1.0]), is_ts=True,
                  trust_radius=1e-300, max_bisect_it=2000)


def test_scaled_secular_underflow_cannot_report_a_successful_zero_root():
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import extremal_root, ArrowheadSolverError
    with pytest.raises(ArrowheadSolverError):
        extremal_root([1e308, 1e308], [1.0, 1.0])


def test_representable_single_active_root_avoids_intermediate_overflow():
    from maple.function.dispatcher.ts.algorithm._rfo_arrowhead import extremal_root
    expected = 0.5 * (1.0 - np.sqrt(5.0)) * 1e308
    assert extremal_root([1e308], [1e308]) == pytest.approx(expected, rel=2e-15)
