"""Analytic rigid-symmetry contracts for scalar PRFO candidate selection.

The distance-only potential is exactly translation/rotation invariant.  At a
nonstationary geometry its *full* Cartesian Hessian can nevertheless have
negative rotational curvatures; they are not internal reaction modes.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from importlib import import_module

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.ts.algorithm.PRFO import PRFO, PRFOStatus
from maple.function.utility.rigid_body import mass_weighted_rigid_basis


_REFERENCE = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],
    dtype=np.float64,
)
_PAIRS = ((0, 1), (1, 2), (0, 2))
_DISTANCES = np.array(
    [np.linalg.norm(_REFERENCE[i] - _REFERENCE[j]) for i, j in _PAIRS]
)
_NEAR_SADDLE = np.array(
    [
        [-2.5922858640189335e-05, -1.3648435118199722e-06, 9.5241066477645269e-06],
        [1.0000146247156523, 5.5614566356841447e-06, -1.4587427886997985e-05],
        [0.19999512460295313, 0.7999998127579802, -1.8440494683402567e-06],
    ],
    dtype=np.float64,
)
_NEAR_MINIMUM = np.array(
    [
        [7.4308489176372538e-06, -1.8022359396905907e-05, -2.4598459874286933e-06],
        [1.0000034787253382, -3.7556863795258513e-06, -4.4319037428695455e-06],
        [0.19999859246354054, 0.7999899695862879, -1.3566759480264276e-05],
    ],
    dtype=np.float64,
)


class DistanceOnlyTriatomic(Calculator):
    """E=0.5*sum(k_a*(r_a-r0_a)^2), with analytic force and Hessian."""

    implemented_properties = ["energy", "free_energy", "forces"]
    rigid_body_invariant = True

    def __init__(self, curvatures):
        super().__init__()
        self.curvatures = np.asarray(curvatures, dtype=np.float64)

    def _derivatives(self, atoms):
        positions = np.asarray(atoms.positions, dtype=np.float64)
        gradient = np.zeros(9, dtype=np.float64)
        hessian = np.zeros((9, 9), dtype=np.float64)
        energy = 0.0
        for pair_index, (i, j) in enumerate(_PAIRS):
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            direction = delta / distance
            residual = distance - _DISTANCES[pair_index]
            curvature = self.curvatures[pair_index]
            energy += 0.5 * curvature * residual * residual
            pair_gradient = curvature * residual * direction
            gradient[3 * i : 3 * i + 3] += pair_gradient
            gradient[3 * j : 3 * j + 3] -= pair_gradient
            pair_hessian = curvature * (
                np.outer(direction, direction)
                + residual / distance
                * (np.eye(3) - np.outer(direction, direction))
            )
            for atom_a, sign_a in ((i, 1), (j, -1)):
                for atom_b, sign_b in ((i, 1), (j, -1)):
                    hessian[
                        3 * atom_a : 3 * atom_a + 3,
                        3 * atom_b : 3 * atom_b + 3,
                    ] += sign_a * sign_b * pair_hessian
        return energy, gradient, hessian

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, gradient, _ = self._derivatives(atoms)
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": -gradient.reshape(3, 3),
        }

    def get_hessian(self, atoms):
        return self._derivatives(atoms)[2]


class ExternalOneAtomSaddle(Calculator):
    """Laboratory-frame potential, intentionally not rigid-body invariant."""

    implemented_properties = ["energy", "free_energy", "forces"]
    rigid_body_invariant = False

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, z = atoms.positions[0]
        energy = 0.5 * (-x * x + y * y + z * z)
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.array([[x, -y, -z]], dtype=np.float64),
        }

    def get_hessian(self, atoms):
        return np.diag([-1.0, 1.0, 1.0])


class UnknownSymmetrySaddle(ExternalOneAtomSaddle):
    rigid_body_invariant = None


def _atoms(positions, curvatures):
    atoms = Atoms(
        "H3", positions=positions.copy(), calculator=DistanceOnlyTriatomic(curvatures)
    )
    atoms.set_masses([1.0, 1.0, 1.0])
    return atoms


def _internal_negative_modes(atoms):
    hessian = atoms.calc.get_hessian(atoms)
    rigid = mass_weighted_rigid_basis(atoms)
    complement = np.linalg.svd(rigid.T, full_matrices=True)[2].T[:, rigid.shape[1] :]
    return int(np.count_nonzero(np.linalg.eigvalsh(complement.T @ hessian @ complement) < -1.0e-6))


class PRFORigidInternalTests(unittest.TestCase):
    def test_near_minimum_is_not_false_saddle_candidate(self):
        atoms = _atoms(_NEAR_MINIMUM, [1.0, 1.0, 1.0])
        self.assertLess(float(np.max(np.abs(atoms.get_forces()))), 9.5e-3)
        full_modes = np.count_nonzero(
            np.linalg.eigvalsh(atoms.calc.get_hessian(atoms)) < -1e-6
        )
        self.assertEqual(full_modes, 1)
        self.assertEqual(_internal_negative_modes(atoms), 0)
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "minimum.out"),
                atoms=atoms,
                paras={"max_iter": 0, "rigid_symmetry": "free_molecule"},
            ).run_result()
            self.assertEqual(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
            self.assertEqual(result.negative_modes, 0)
            self.assertFalse(result.geometry_converged)

    def test_near_first_order_saddle_is_not_rejected_by_rigid_curvature(self):
        atoms = _atoms(_NEAR_SADDLE, [-1.0, 1.0, 1.0])
        self.assertLess(float(np.max(np.abs(atoms.get_forces()))), 9.5e-3)
        full_modes = np.count_nonzero(
            np.linalg.eigvalsh(atoms.calc.get_hessian(atoms)) < -1e-6
        )
        self.assertEqual(full_modes, 3)
        self.assertEqual(_internal_negative_modes(atoms), 1)
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "saddle.out"),
                atoms=atoms,
                paras={"max_iter": 0, "rigid_symmetry": "free_molecule"},
            ).run_result()
            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertEqual(result.negative_modes, 1)

    def test_internal_inertia_is_stable_under_small_rigid_invariant_perturbations(self):
        generator = np.random.default_rng(20260923)
        cases = (
            ([1.0, 1.0, 1.0], PRFOStatus.FAILED_WRONG_INERTIA, 0),
            ([-1.0, 1.0, 1.0], PRFOStatus.GEOMETRY_CONVERGED, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            for case_index, (curvatures, expected_status, expected_modes) in enumerate(cases):
                for sample in range(10):
                    with self.subTest(case=case_index, sample=sample):
                        positions = _REFERENCE + 1.0e-4 * generator.normal(size=(3, 3))
                        atoms = _atoms(positions, curvatures)
                        result = PRFO(
                            output=str(Path(directory) / f"case{case_index}-{sample}.out"),
                            atoms=atoms,
                            paras={"max_iter": 0, "rigid_symmetry": "free_molecule"},
                        ).run_result()
                        self.assertEqual(result.status, expected_status)
                        self.assertEqual(result.negative_modes, expected_modes)

    def test_analytic_hessian_matches_force_difference(self):
        atoms = _atoms(_NEAR_SADDLE, [-1.0, 1.0, 1.0])
        positions = atoms.positions.copy()
        analytical = atoms.calc.get_hessian(atoms)
        numerical = np.zeros_like(analytical)
        delta = 1.0e-5
        for column in range(9):
            plus = positions.copy().reshape(-1)
            minus = positions.copy().reshape(-1)
            plus[column] += delta
            minus[column] -= delta
            atoms.set_positions(plus.reshape(3, 3))
            gradient_plus = -atoms.get_forces().reshape(-1)
            atoms.set_positions(minus.reshape(3, 3))
            gradient_minus = -atoms.get_forces().reshape(-1)
            numerical[:, column] = (gradient_plus - gradient_minus) / (2 * delta)
        atoms.set_positions(positions)
        np.testing.assert_allclose(analytical, numerical, rtol=0, atol=2.0e-10)

    def test_distance_potential_is_rigid_motion_invariant(self):
        atoms = _atoms(_NEAR_SADDLE, [-1.0, 1.0, 1.0])
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        moved = _NEAR_SADDLE @ rotation.T + np.array([1.3, -2.1, 0.7])
        atoms.set_positions(moved)
        self.assertAlmostEqual(atoms.get_potential_energy(), energy, places=15)
        np.testing.assert_allclose(atoms.get_forces(), forces @ rotation.T, atol=1e-14)

    def test_linear_and_near_linear_rigid_rank_is_geometry_adaptive(self):
        from maple.function.dispatcher.ts.algorithm.PRFO import (
            _rigid_internal_complement,
        )

        linear = _atoms([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 1, 1])
        near_linear = _atoms(
            [[0, 0, 0], [1, 1.0e-4, 0], [2, 0, 0]], [1, 1, 1]
        )
        for atoms, expected_internal_dof in ((linear, 4), (near_linear, 3)):
            with self.subTest(internal_dof=expected_internal_dof):
                complement = _rigid_internal_complement(atoms)
                self.assertEqual(complement.shape, (9, expected_internal_dof))
                rigid = mass_weighted_rigid_basis(atoms)
                np.testing.assert_allclose(rigid.T @ complement, 0, atol=1e-12)
                np.testing.assert_allclose(
                    complement.T @ complement,
                    np.eye(expected_internal_dof), atol=1e-12,
                )

    def test_step_and_mode_tracking_both_use_internal_space(self):
        module = import_module("maple.function.dispatcher.ts.algorithm.PRFO")
        atoms = _atoms(_NEAR_SADDLE, [-1.0, 1.0, 1.0])
        seen = []

        def record_step(*, H, g, pre_eig, **unused):
            seen.append((H.shape, g.shape, pre_eig[1].shape))
            raise RuntimeError("stopped after internal-space inspection")

        with tempfile.TemporaryDirectory() as directory:
            optimizer = PRFO(
                output=str(Path(directory) / "step.out"),
                atoms=atoms,
                paras={
                    "max_iter": 1,
                    "f_max_th": 1.0e-8,
                    "f_rms_th": 1.0e-8,
                    "rigid_symmetry": "free_molecule",
                },
            )
            with patch.object(module, "prfo_step", side_effect=record_step):
                result = optimizer.run_result()

        self.assertEqual(result.status, PRFOStatus.FAILED_STEP_SOLVER)
        self.assertEqual(seen, [((3, 3), (3,), (3, 3))])
        self.assertEqual(optimizer.tracked_mode_vec_mw.shape, (9,))
        rigid = mass_weighted_rigid_basis(atoms)
        np.testing.assert_allclose(
            rigid.T @ optimizer.tracked_mode_vec_mw, 0, atol=1e-12
        )

    def test_accepted_step_does_not_spend_trust_radius_on_rigid_motion(self):
        initial = _REFERENCE.copy()
        initial[1, 0] += 1.0e-4
        atoms = _atoms(initial, [-1.0, 1.0, 1.0])
        rigid_at_start = mass_weighted_rigid_basis(atoms)
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "accepted-step.out"),
                atoms=atoms,
                paras={
                    "rigid_symmetry": "free_molecule", "max_iter": 1,
                    "f_max_th": 1.0e-8, "f_rms_th": 1.0e-8,
                },
            ).run_result()
        self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.iterations, 1)
        step_mw = (atoms.positions - initial).reshape(-1)  # all masses = 1 amu
        self.assertGreater(float(np.linalg.norm(step_mw)), 0.0)
        np.testing.assert_allclose(rigid_at_start.T @ step_mw, 0, atol=1e-12)

    def test_periodic_geometry_does_not_assume_free_molecule_symmetry(self):
        atoms = _atoms(_REFERENCE, [-1.0, 1.0, 1.0])
        atoms.set_cell([10.0, 10.0, 10.0])
        atoms.set_pbc(True)
        with self.assertRaisesRegex(NotImplementedError, "periodic"):
            PRFO(
                output="unused.out", atoms=atoms,
                paras={"rigid_symmetry": "free_molecule"},
            )

    def test_declared_external_field_cannot_be_treated_as_rigid_invariant(self):
        atoms = _atoms(_REFERENCE, [-1.0, 1.0, 1.0])
        atoms.calc.rigid_body_invariant = False
        with self.assertRaisesRegex(ValueError, "rigid-body invariant"):
            PRFO(
                output="unused.out", atoms=atoms,
                paras={"rigid_symmetry": "free_molecule"},
            )

    def test_explicit_external_mode_preserves_lab_frame_saddle(self):
        atoms = Atoms(
            "H", positions=[[0.0, 0.0, 0.0]],
            calculator=ExternalOneAtomSaddle(),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "external.out"),
                atoms=atoms,
                paras={"max_iter": 0, "rigid_symmetry": "cartesian_external"},
            ).run_result()
        self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.negative_modes, 1)

    def test_declared_backend_symmetry_selects_the_same_space_automatically(self):
        cases = (
            (_atoms(_NEAR_MINIMUM, [1.0, 1.0, 1.0]), "free_molecule", 0),
            (
                Atoms(
                    "H", positions=[[0.0, 0.0, 0.0]],
                    calculator=ExternalOneAtomSaddle(),
                ),
                "cartesian_external", 1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (atoms, expected_space, expected_modes) in enumerate(cases):
                with self.subTest(space=expected_space):
                    optimizer = PRFO(
                        output=str(Path(directory) / f"auto{index}.out"),
                        atoms=atoms,
                        paras={"max_iter": 0},
                    )
                    self.assertEqual(optimizer._coordinate_space, expected_space)
                    self.assertEqual(optimizer.run_result().negative_modes, expected_modes)

    def test_unknown_backend_requires_an_explicit_physical_assertion(self):
        atoms = Atoms(
            "H", positions=[[0.0, 0.0, 0.0]],
            calculator=UnknownSymmetrySaddle(),
        )
        with self.assertRaisesRegex(NotImplementedError, "unknown symmetry fails closed"):
            PRFO(output="unused.out", atoms=atoms)
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "asserted-external.out"),
                atoms=atoms,
                paras={"max_iter": 0, "rigid_symmetry": "cartesian_external"},
            ).run_result()
        self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)

    def test_near_zero_internal_negative_mode_is_not_promoted(self):
        atoms = _atoms(_REFERENCE, [-2.0e-7, 1.0, 1.0])
        self.assertEqual(_internal_negative_modes(atoms), 0)
        with tempfile.TemporaryDirectory() as directory:
            result = PRFO(
                output=str(Path(directory) / "soft.out"),
                atoms=atoms,
                paras={"max_iter": 0, "rigid_symmetry": "free_molecule"},
            ).run_result()
        self.assertEqual(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.negative_modes, 0)

    def test_free_atom_has_no_internal_ts_coordinate(self):
        atoms = Atoms(
            "H", positions=[[0.0, 0.0, 0.0]],
            calculator=ExternalOneAtomSaddle(),
        )
        atoms.calc.rigid_body_invariant = True
        with self.assertRaisesRegex(NotImplementedError, "internal degree"):
            PRFO(
                output="unused.out", atoms=atoms,
                paras={"rigid_symmetry": "free_molecule"},
            )


if __name__ == "__main__":
    unittest.main()
