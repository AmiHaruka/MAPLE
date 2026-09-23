"""Numerical Hessian object and streaming contracts (no model weights needed)."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixInternals

from maple.function.calculator._batch_eval import (
    FDHessianEvaluator,
    _copy_with_positions,
)
from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.uma._uma_calculator import UMACalculator


class _PolynomialCalculator(CalcABC):
    """E = x.K.x/2 + sum(x**4)/4 in Ha; forces are raw -dE/dx."""

    MODEL_NAMES = ()
    SUPPORTS_CHARGE_MULT = False
    SUPPORTS_PBC = False

    def __init__(self, K, *, fail_at=None, require_raw=False):
        super().__init__()
        self.K = np.asarray(K, dtype=np.float64)
        self.hessian = "numerical"
        self.fail_at = fail_at
        self.require_raw = require_raw
        self.calls = 0
        self.batches = []

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        if self.require_raw:
            assert not atoms.constraints, "raw Cartesian FD retained a constraint"
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic force failure")
        x = atoms.get_positions().reshape(-1)
        self.results = {
            "energy": float(0.5 * x @ self.K @ x + np.sum(x**4) / 4.0),
            "forces": (-(self.K @ x + x**3)).reshape(-1, 3),
        }

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        self.batches.append([at.get_positions().copy() for at in atoms_list])
        return super().calculate_many(atoms_list, properties)


def _pair_with_bond_constraint():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_constraint(FixInternals(bonds=[[1.0, [0, 1]]]))
    return atoms


def test_public_numerical_hessian_requires_explicit_raw_request_for_nonlinear_constraint():
    atoms = _pair_with_bond_constraint()
    calc = _PolynomialCalculator(np.eye(6), require_raw=True)
    before = atoms.get_positions().copy()

    with pytest.raises(NotImplementedError, match="FixAtoms/FixCartesian"):
        calc.get_hessian(atoms)
    assert calc.calls == 0

    raw = calc.get_hessian(atoms, delta=0.01, constraint_mode="raw_cartesian")
    expected = np.eye(6) + np.diag(3 * before.reshape(-1) ** 2 + 0.01**2)
    np.testing.assert_allclose(raw, expected, atol=3e-13, rtol=0)
    np.testing.assert_array_equal(atoms.get_positions(), before)
    assert len(atoms.constraints) == 1
    # The bond constraint would move both atoms if ASE adjusted this step.
    np.testing.assert_allclose(calc.batches[0][0][0, 0], 0.01, atol=0, rtol=0)
    np.testing.assert_allclose(calc.batches[0][0][1, 0], 1.0, atol=0, rtol=0)


def test_fixed_cartesian_mode_embeds_only_the_active_hessian_block():
    atoms = Atoms("H2", positions=[[0.2, -0.3, 0.1], [1.0, 0.4, -0.2]])
    atoms.set_constraint(FixCartesian(0, mask=(True, False, False)))
    K = np.eye(6) * 2.0
    K[0, 3] = K[3, 0] = 0.2
    K[1, 4] = K[4, 1] = -0.1
    calc = _PolynomialCalculator(K, require_raw=True)
    delta = 0.02

    fixed = calc.get_hessian(atoms, delta=delta)
    expected = K + np.diag(3 * atoms.get_positions().reshape(-1) ** 2 + delta**2)
    expected[0, :] = expected[:, 0] = 0.0
    np.testing.assert_allclose(fixed, expected, atol=3e-13, rtol=0)
    assert sum(map(len, calc.batches)) == 10

    raw = calc.get_hessian(atoms, delta=delta, constraint_mode="raw_cartesian")
    assert raw.shape == (6, 6)
    np.testing.assert_allclose(raw[0, 3], 0.2, atol=3e-13, rtol=0)
    assert sum(map(len, calc.batches)) == 22


def test_invalid_constraint_mode_rejected_before_calculator_invocation():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = _PolynomialCalculator(np.eye(3))
    with pytest.raises(ValueError, match="constraint_mode"):
        calc.get_hessian(atoms, constraint_mode="constrained")
    assert calc.calls == 0


def test_analytic_hessian_has_the_same_explicit_constraint_object_contract():
    class _AnalyticCalculator(_PolynomialCalculator):
        def __init__(self, K):
            super().__init__(K)
            self.hessian = "analytic"

        def _analytic_hessian(self, atoms):
            assert not atoms.constraints
            x = atoms.get_positions().reshape(-1)
            return self.K + np.diag(3 * x**2)

    atoms = Atoms("H2", positions=[[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_constraint(FixCartesian(0, mask=(True, False, False)))
    calc = _AnalyticCalculator(np.eye(6))
    fixed = calc.get_hessian(atoms)
    assert np.array_equal(fixed[0, :], np.zeros(6))
    assert np.array_equal(fixed[:, 0], np.zeros(6))
    raw = calc.get_hessian(atoms, constraint_mode="raw_cartesian")
    assert raw[0, 0] > 1.0

    nonlinear = _pair_with_bond_constraint()
    with pytest.raises(NotImplementedError, match="FixAtoms/FixCartesian"):
        calc.get_hessian(nonlinear)
    assert calc.get_hessian(nonlinear, constraint_mode="raw_cartesian").shape == (6, 6)


@pytest.mark.parametrize("mode", ["analytic", "numerical"])
@pytest.mark.parametrize("constraint", ["atoms", "cartesian"])
def test_negative_ase_atom_index_matches_positive_fixed_cartesian_projection(
    mode, constraint
):
    class _IdentityAnalytic(_PolynomialCalculator):
        def __init__(self):
            super().__init__(np.eye(6))
            self.hessian = "analytic"

        def _analytic_hessian(self, atoms):
            return np.eye(6)

    hessians = []
    for atom_index in (1, -1):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        if constraint == "atoms":
            atoms.set_constraint(FixAtoms(indices=[atom_index]))
        else:
            atoms.set_constraint(
                FixCartesian(atom_index, mask=(True, False, True))
            )
        calc = _IdentityAnalytic() if mode == "analytic" else _PolynomialCalculator(np.eye(6))
        hessians.append(calc.get_hessian(atoms, delta=0.01))

    np.testing.assert_array_equal(hessians[0], hessians[1])
    active = 1.0 + (0.01**2 if mode == "numerical" else 0.0)
    if constraint == "atoms":
        np.testing.assert_allclose(
            np.diag(hessians[1]), [active, active, active, 0, 0, 0], atol=1e-12
        )
    else:
        np.testing.assert_allclose(
            np.diag(hessians[1]), [active, active, active, 0, active, 0],
            atol=1e-12,
        )


@pytest.mark.parametrize("index", [-3, 2])
def test_out_of_range_fixed_atom_index_fails_before_hessian_work(index):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_constraint(FixAtoms(indices=[index]))
    calc = _PolynomialCalculator(np.eye(6))

    with pytest.raises(ValueError, match="index"):
        calc.get_hessian(atoms)
    assert calc.calls == 0


def test_uma_public_numerical_entry_forwards_explicit_raw_mode():
    class _UMAProtocol(_PolynomialCalculator):
        def _set_task_from_atoms(self, atoms):
            return None

        def _validate_task_atoms_compatibility(self, atoms):
            return None

        def _validate_charge_spin_task_compatibility(self, atoms):
            return 0, 1

    calc = _UMAProtocol(np.eye(6), require_raw=True)
    atoms = _pair_with_bond_constraint()
    with pytest.raises(NotImplementedError, match="FixAtoms/FixCartesian"):
        UMACalculator.get_hessian(calc, atoms)
    assert UMACalculator.get_hessian(
        calc, atoms, constraint_mode="raw_cartesian"
    ).shape == (6, 6)


def test_fd_streams_complete_pairs_in_dof_order_without_eager_atom_copies():
    atoms = Atoms("H4", positions=np.arange(12, dtype=float).reshape(4, 3) * 0.13)
    calc = _PolynomialCalculator(np.eye(12))
    weak_copies = []
    resident_at_calls = []
    original_copy = _copy_with_positions
    original_batch = calc.calculate_many

    def tracked_copy(*args, **kwargs):
        clone = original_copy(*args, **kwargs)
        weak_copies.append(weakref.ref(clone))
        return clone

    def tracked_batch(atoms_list, properties=("energy", "forces")):
        gc.collect()
        resident_at_calls.append(sum(ref() is not None for ref in weak_copies))
        return original_batch(atoms_list, properties)

    with patch(
        "maple.function.calculator._batch_eval._copy_with_positions",
        side_effect=tracked_copy,
    ):
        calc.calculate_many = tracked_batch
        H = FDHessianEvaluator(calc, fd_batch_size=4).hessian(atoms, delta=0.01)

    expected = np.eye(12) + np.diag(3 * atoms.get_positions().reshape(-1) ** 2 + 0.01**2)
    np.testing.assert_allclose(H, expected, atol=3e-13, rtol=0)
    assert len(calc.batches) == 6
    assert all(len(batch) == 4 for batch in calc.batches)
    assert max(resident_at_calls) <= 4
    ordered = [x for batch in calc.batches for x in batch]
    original = atoms.get_positions()
    for dof in range(12):
        plus = original.copy().reshape(-1)
        minus = original.copy().reshape(-1)
        plus[dof] += 0.01
        minus[dof] -= 0.01
        np.testing.assert_array_equal(ordered[2 * dof], plus.reshape(-1, 3))
        np.testing.assert_array_equal(ordered[2 * dof + 1], minus.reshape(-1, 3))


def test_default_auto_fd_stream_has_a_bounded_host_resident_chunk():
    atoms = Atoms("H20", positions=np.arange(60, dtype=float).reshape(20, 3) * 0.011)
    calc = _PolynomialCalculator(np.eye(60))
    H = FDHessianEvaluator(calc).hessian(atoms, delta=0.01)
    assert H.shape == (60, 60)
    assert len(calc.batches) > 1
    assert max(map(len, calc.batches)) <= 64


def test_explicit_all_is_an_intentional_unbounded_host_batch_opt_out():
    atoms = Atoms("H20", positions=np.arange(60, dtype=float).reshape(20, 3) * 0.011)
    calc = _PolynomialCalculator(np.eye(60))
    FDHessianEvaluator(calc, fd_batch_size="all").hessian(atoms, delta=0.01)
    assert [len(batch) for batch in calc.batches] == [120]


def test_fd_chunk_size_changes_dispatch_but_not_analytic_central_difference():
    atoms = Atoms("H2", positions=[[0.2, -0.3, 0.4], [0.7, 0.1, -0.2]])
    K = np.diag([1.0, 1.4, 2.0, 0.8, 1.2, 1.8])
    K[0, 4] = K[4, 0] = 0.15
    expected = K + np.diag(3 * atoms.get_positions().reshape(-1) ** 2 + 0.01**2)
    for size in (1, 2, 3, 4, 8, "auto", "all"):
        calc = _PolynomialCalculator(K)
        actual = FDHessianEvaluator(calc, fd_batch_size=size).hessian(
            atoms, delta=0.01
        )
        np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=0)
        assert sum(map(len, calc.batches)) == 12
        assert max(map(len, calc.batches)) <= (12 if size in ("auto", "all") else size)


def test_fixed_topology_context_receives_raw_reference_and_closes_on_failure():
    class _ContextCalculator(_PolynomialCalculator):
        def __init__(self, K):
            super().__init__(K)
            self.context_closed = False
            self.context_calls = 0

        def make_fd_context(self, atoms, *, delta=None, fd_context_mode=None):
            assert not atoms.constraints
            parent = self

            class _Context:
                def force_at(self, positions):
                    parent.context_calls += 1
                    if parent.context_calls == 3:
                        raise RuntimeError("synthetic context failure")
                    x = positions.reshape(-1)
                    return (-(parent.K @ x + x**3)).reshape(-1, 3)

                def close(self):
                    parent.context_closed = True

            return _Context()

    atoms = _pair_with_bond_constraint()
    calc = _ContextCalculator(np.eye(6))
    before = atoms.get_positions().copy()
    with pytest.raises(RuntimeError, match="synthetic context failure"):
        calc.get_hessian(atoms, constraint_mode="raw_cartesian")
    assert calc.context_closed
    assert calc.context_calls == 3
    assert calc.calls == 0
    np.testing.assert_array_equal(atoms.get_positions(), before)


def test_single_displacement_cap_preserves_plus_minus_order():
    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    calc = _PolynomialCalculator(np.eye(3))
    H = FDHessianEvaluator(calc, fd_batch_size=1).hessian(atoms, delta=0.01)
    assert H.shape == (3, 3)
    assert [len(batch) for batch in calc.batches] == [1] * 6
    np.testing.assert_allclose(
        calc.batches[0][0] - calc.batches[1][0], [[0.02, 0.0, 0.0]],
        atol=3e-17, rtol=0,
    )


def test_auto_fd_oom_backoff_retries_same_pair_without_cpu_fallback():
    class _OOMWhenBatched(_PolynomialCalculator):
        def calculate_many(self, atoms_list, properties=("energy", "forces")):
            if len(atoms_list) > 1:
                raise RuntimeError("CUDA out of memory (synthetic)")
            return super().calculate_many(atoms_list, properties)

    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    calc = _OOMWhenBatched(np.eye(3))
    H = FDHessianEvaluator(calc).hessian(atoms, delta=0.01)
    np.testing.assert_allclose(
        H, np.eye(3) + np.diag(3 * atoms.get_positions().reshape(-1) ** 2 + 0.01**2)
    )
    assert [len(batch) for batch in calc.batches] == [1] * 6


def test_fd_failure_restores_input_coordinates_constraints_and_calculator_cache():
    atoms = _pair_with_bond_constraint()
    before = atoms.get_positions().copy()
    calc = _PolynomialCalculator(np.eye(6), fail_at=5, require_raw=True)
    old_atoms = Atoms("He", positions=[[8.0, 0.0, 0.0]])
    old_results = {"energy": 7.0, "forces": np.full((2, 3), 9.0)}
    calc.atoms = old_atoms
    calc.results = old_results

    with pytest.raises(RuntimeError, match="synthetic force failure"):
        calc.get_hessian(
            atoms, delta=0.01, constraint_mode="raw_cartesian"
        )
    np.testing.assert_array_equal(atoms.get_positions(), before)
    assert len(atoms.constraints) == 1
    assert calc.atoms is old_atoms
    assert calc.results is old_results
    np.testing.assert_array_equal(calc.results["forces"], np.full((2, 3), 9.0))


def test_fd_delta_sweep_matches_analytic_quartic_error_and_unit_metadata():
    atoms = Atoms("H", positions=[[0.25, -0.36, 0.42]])
    calc = _PolynomialCalculator(np.diag([2.0, 3.0, 4.0]))
    analytic = calc.K + np.diag(3 * atoms.get_positions().reshape(-1) ** 2)
    errors = []
    for delta in (0.1, 0.05, 0.025):
        actual = calc.get_hessian(atoms, delta=delta)
        errors.append(np.max(np.abs(actual - analytic)))
    np.testing.assert_allclose(errors, [0.01, 0.0025, 0.000625], atol=3e-14, rtol=0)
    diag = calc._fd_hessian_last_antisymmetry
    assert diag["absolute_unit"] == "Ha/Angstrom^2"
    assert diag["scale_unit"] == "Ha/Angstrom^2"
    assert diag["scale_floor"] == 1.0
