"""RFO job-local finite-difference settings must not leak through shared calculators."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from maple.function.dispatcher.optimization.algorithm.RFO import RFO


@pytest.mark.parametrize("initial", [None, 2])
@pytest.mark.parametrize("backend_fails", [False, True])
def test_fd_batch_size_override_is_scoped_to_one_rfo_run(initial, backend_fails):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = SinglePointCalculator(
        atoms,
        energy=0.0,
        free_energy=0.0,
        forces=np.zeros((1, 3), dtype=np.float64),
    )
    atoms.calc = calc
    if initial is not None:
        calc.fd_batch_size = initial

    with tempfile.TemporaryDirectory() as directory:
        optimizer = RFO(
            atoms=atoms,
            output=str(Path(directory) / "rfo.out"),
            paras={"max_iter": 0, "fd_batch_size": 4},
        )
        if backend_fails:
            with patch(
                "maple.function.dispatcher.optimization.algorithm.RFO.energy_forces_one",
                side_effect=RuntimeError("synthetic backend failure"),
            ):
                with pytest.raises(RuntimeError, match="synthetic backend failure"):
                    optimizer.run()
        else:
            assert optimizer.run() is atoms

    if initial is None:
        assert not hasattr(calc, "fd_batch_size")
    else:
        assert calc.fd_batch_size == initial


def test_rfo_forwards_fd_batch_size_without_shared_calculator_mutation():
    class _HessianSpy:
        def __init__(self):
            self.fd_batch_size = 2
            self.calls = []

        def get_hessian(self, atoms, **kwargs):
            assert self.fd_batch_size == 2
            self.calls.append(kwargs)
            return np.eye(3)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calc = _HessianSpy()
    atoms.calc = calc
    optimizer = RFO(
        atoms=atoms,
        output="unused.out",
        paras={
            "fd_batch_size": 4,
            "fd_hessian_antisymmetry_action": "warn",
            "fd_hessian_antisymmetry_threshold": 2e-4,
        },
    )

    np.testing.assert_array_equal(optimizer._calculate_hessian(atoms), np.eye(3))
    assert calc.calls == [{
        "fd_batch_size": 4,
        "fd_hessian_antisymmetry_action": "warn",
        "fd_hessian_antisymmetry_threshold": 2e-4,
    }]
    assert calc.fd_batch_size == 2
