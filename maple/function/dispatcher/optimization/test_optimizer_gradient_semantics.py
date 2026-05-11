import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS
from maple.function.dispatcher.optimization.algorithm.SDCG import SDCG


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        energy = 0.5 * float(np.sum(positions**2))
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = -positions


def test_lbfgs_first_step_follows_negative_gradient(tmp_path):
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = HarmonicCalculator()
    atoms.f_max_th = 1e-12
    atoms.f_rms_th = 1e-12
    atoms.dp_max_th = 1e-12
    atoms.dp_rms_th = 1e-12

    initial_energy = atoms.get_potential_energy(force_consistent=True)

    optimizer = LBFGS(
        atoms,
        output=str(tmp_path / "lbfgs.out"),
        paras={
            "method": "lbfgs",
            "curvature": 1.0,
            "max_step": 0.2,
            "max_iter": 1,
            "verbose": 0,
        },
    )
    result = optimizer.run()

    final_energy = atoms.get_potential_energy(force_consistent=True)

    assert result is atoms
    assert final_energy < initial_energy
    assert 0.0 <= atoms.positions[0, 0] < 1.0
    assert len(optimizer.S) == 1
    assert np.dot(optimizer.Y[0], optimizer.S[0]) > 0.0


def test_sdcg_bb_history_pairs_previous_position_with_previous_force(tmp_path):
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = HarmonicCalculator()
    atoms.f_max_th = 1e-12
    atoms.f_rms_th = 1e-12
    atoms.dp_max_th = 1e-12
    atoms.dp_rms_th = 1e-12

    initial_forces = atoms.get_forces().copy()

    optimizer = SDCG(
        atoms,
        output=str(tmp_path / "sdcg.out"),
        paras={
            "method": "sd",
            "max_step": 0.2,
            "max_iter": 1,
            "verbose": 0,
            "diis_enabled": False,
        },
    )
    optimizer.run()

    np.testing.assert_allclose(optimizer._prev_forces, initial_forces)
    assert np.isclose(optimizer._estimate_step_scale(atoms.get_forces()), 1.0)
