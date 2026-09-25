"""Local behavior locks for the runtime-disabled BatchPRFO prototype.

These tests intentionally exercise existing private experimental interfaces.
They do not enable ``BatchPRFO.run`` or admit batched TS optimization.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.dispatcher.ts.algorithm.BPRFO import (
    BIG, BatchPRFO, prfo_step_batched,
)
from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFOResult,
    PRFOStatus,
    prfo_step,
)
from maple.function.utility.rigid_body import mass_weighted_rigid_basis


_TRIATOMIC_REFERENCE = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]], dtype=np.float64
)
_TRIATOMIC_PAIRS = ((0, 1), (1, 2), (0, 2))
_TRIATOMIC_DISTANCES = np.array([
    np.linalg.norm(_TRIATOMIC_REFERENCE[i] - _TRIATOMIC_REFERENCE[j])
    for i, j in _TRIATOMIC_PAIRS
])
_TRIATOMIC_NEAR_SADDLE = np.array([
    [-2.5922858640189335e-05, -1.3648435118199722e-06, 9.5241066477645269e-06],
    [1.0000146247156523, 5.5614566356841447e-06, -1.4587427886997985e-05],
    [0.19999512460295313, 0.7999998127579802, -1.8440494683402567e-06],
], dtype=np.float64)
_TRIATOMIC_NEAR_MINIMUM = np.array([
    [7.4308489176372538e-06, -1.8022359396905907e-05, -2.4598459874286933e-06],
    [1.0000034787253382, -3.7556863795258513e-06, -4.4319037428695455e-06],
    [0.19999859246354054, 0.7999899695862879, -1.3566759480264276e-05],
], dtype=np.float64)


def _soft_saddle_bond_displacement(shift):
    """Move one pair while preserving the other two reference distances."""
    bond = _TRIATOMIC_DISTANCES[0] + shift
    other = _TRIATOMIC_DISTANCES[2]
    opposite = _TRIATOMIC_DISTANCES[1]
    x = (other**2 + bond**2 - opposite**2) / (2.0 * bond)
    y = np.sqrt(other**2 - x**2)
    return Atoms("H3", positions=[[0, 0, 0], [bond, 0, 0], [x, y, 0]])


class _SyntheticFP64Adapter:
    """Minimal existing BatchPRFO calculator protocol for CPU state tests."""

    def __init__(
        self, *, trial_energy=0.0, fail_trial=False, actual_hessian=None
    ):
        self.device = torch.device("cpu")
        self.trial_energy = float(trial_energy)
        self.fail_trial = fail_trial
        self.actual_hessian = actual_hessian
        self.coord = torch.zeros((0, 3), dtype=torch.float64)
        self._prepared = False
        self._backup = None
        self._atoms = []
        self.nmax = 0
        self.nmax_dof = 0
        self._ptr = torch.zeros(1, dtype=torch.long)
        self.numbers = torch.zeros(0, dtype=torch.int32)
        self.prepare_calls = 0
        self.backup_calls = 0
        self.restore_calls = 0
        self.discard_calls = 0
        self.step_calls = 0
        self.ef_calls = 0
        self.efh_calls = 0

    def prepare(self, atoms_list, fixed_nmax=None):
        self._prepared = False
        self._backup = None
        self.prepare_calls += 1
        self._atoms = list(atoms_list)
        lengths = [len(atoms) for atoms in self._atoms]
        self._ptr = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.long)
        self.numbers = (
            torch.tensor(
                np.concatenate([at.numbers for at in self._atoms]),
                dtype=torch.int32,
            ) if self._atoms else torch.zeros(0, dtype=torch.int32)
        )
        positions = [atoms.get_positions() for atoms in self._atoms]
        self.coord = (
            torch.tensor(np.concatenate(positions, axis=0), dtype=torch.float64)
            if positions
            else torch.zeros((0, 3), dtype=torch.float64)
        )
        self.nmax = fixed_nmax or max(
            (3 * len(atoms) for atoms in self._atoms), default=0
        )
        self.nmax_dof = self.nmax
        self._prepared = True

    def _require_prepared(self):
        if not self._prepared:
            raise RuntimeError("call prepare before evaluation")

    def backup_coords(self):
        self.backup_calls += 1
        self._backup = self.coord.clone()

    def restore_coords(self):
        self.restore_calls += 1
        if self._backup is not None:
            self.coord.copy_(self._backup)
            self._backup = None

    def _discard_coord_backup(self):
        self.discard_calls += 1
        self._backup = None

    def step_cart_(self, step):
        self.step_calls += 1
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            self.coord[start:stop].add_(
                step[index, : 3 * len(atoms)].reshape(-1, 3)
            )
            start = stop

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        if self.fail_trial:
            raise RuntimeError("synthetic trial failure")
        batch = len(self._atoms)
        return (
            torch.full((batch,), self.trial_energy, dtype=torch.float64),
            torch.zeros((batch, self.nmax), dtype=torch.float64),
        )

    def get_efh_gpu(self):
        self._require_prepared()
        self.efh_calls += 1
        energy, forces = self.get_ef_gpu()
        base_hessian = (
            torch.eye(self.nmax, dtype=torch.float64)
            if self.actual_hessian is None
            else torch.as_tensor(self.actual_hessian, dtype=torch.float64)
        )
        hessian = base_hessian.expand(len(self._atoms), -1, -1).clone()
        padding = torch.zeros(len(self._atoms), dtype=torch.long)
        return energy, forces, hessian, padding


class _DistanceTriatomicAdapter(_SyntheticFP64Adapter):
    """Exact E/F/H for an invariant three-distance PES, not a fitted model."""

    rigid_body_invariant = True

    def __init__(self, curvatures):
        super().__init__()
        self.curvatures = np.asarray(curvatures, dtype=np.float64)

    def _values(self):
        energies = torch.zeros(len(self._atoms), dtype=torch.float64)
        forces = torch.zeros((len(self._atoms), self.nmax), dtype=torch.float64)
        hessians = torch.zeros((len(self._atoms), self.nmax, self.nmax), dtype=torch.float64)
        for member, atoms in enumerate(self._atoms):
            positions = self.coord[self._ptr[member]:self._ptr[member + 1]].numpy()
            curvatures = np.asarray(
                atoms.info.get("curvatures", self.curvatures), dtype=np.float64
            )
            gradient = np.zeros(9, dtype=np.float64)
            hessian = np.zeros((9, 9), dtype=np.float64)
            energy = 0.0
            for pair_index, (i, j) in enumerate(_TRIATOMIC_PAIRS):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                direction = delta / distance
                residual = distance - _TRIATOMIC_DISTANCES[pair_index]
                curvature = curvatures[pair_index]
                energy += 0.5 * curvature * residual**2
                pair_gradient = curvature * residual * direction
                gradient[3 * i:3 * i + 3] += pair_gradient
                gradient[3 * j:3 * j + 3] -= pair_gradient
                pair_hessian = curvature * (
                    np.outer(direction, direction)
                    + residual / distance * (np.eye(3) - np.outer(direction, direction))
                )
                for atom_a, sign_a in ((i, 1), (j, -1)):
                    for atom_b, sign_b in ((i, 1), (j, -1)):
                        hessian[
                            3 * atom_a:3 * atom_a + 3,
                            3 * atom_b:3 * atom_b + 3,
                        ] += sign_a * sign_b * pair_hessian
            energies[member] = energy
            forces[member, :9] = -torch.from_numpy(gradient)
            hessians[member, :9, :9] = torch.from_numpy(hessian)
        return energies, forces, hessians

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        return self._values()[:2]

    def get_efh_gpu(self):
        self._require_prepared()
        self.efh_calls += 1
        return (*self._values(), torch.zeros(len(self._atoms), dtype=torch.long))


class _ForbiddenPrepareAdapter:
    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, atoms_list, fixed_nmax=None):
        self.prepare_calls += 1
        raise AssertionError("calculator preparation is a forbidden side effect")


class _ScaledQuadraticAdapter(_SyntheticFP64Adapter):
    def __init__(self, *, scale):
        super().__init__()
        self.scale = float(scale)
        self.gradient = torch.tensor([0.10, 0.05, 0.05], dtype=torch.float64)
        self.hessian = torch.diag(
            torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
        )
        self.actual_hessian = self.hessian

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        displacement = self.coord.reshape(-1)
        model_change = torch.dot(self.gradient, displacement) + 0.5 * torch.dot(
            displacement, self.hessian @ displacement
        )
        return (
            (self.scale * model_change).reshape(1),
            (-self.scale * (self.gradient + self.hessian @ displacement)).reshape(
                1, -1
            ),
        )


class _MisleadingCubicAdapter(_SyntheticFP64Adapter):
    """Smooth E/F/H with a prescribed finite-trial rho and exact local Hessian."""

    def __init__(self, *, rho, mass=1.0):
        super().__init__()
        self.gradient = torch.tensor([0.10, 0.05, 0.05], dtype=torch.float64)
        self.hessian = torch.diag(torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64))
        self.direction, self.cubic_coefficient = self._cubic_terms(rho, mass)

    def _cubic_terms(self, rho, mass):
        diagonal = self.hessian.numpy().diagonal() / mass
        projected_gradient = self.gradient.numpy() / np.sqrt(mass)
        mw_step = prfo_step(
            np.diag(diagonal), projected_gradient, is_ts=True,
            target_mode=0, trust_radius=0.2,
            pre_eig=(diagonal, np.eye(3), projected_gradient),
        )
        step = mw_step / np.sqrt(mass)
        norm = np.linalg.norm(step)
        predicted = (
            float(self.gradient.numpy() @ step)
            + 0.5 * float(step @ self.hessian.numpy() @ step)
        )
        assert norm > 0.0 and abs(predicted) > 1.0e-16
        return (
            torch.tensor(step / norm, dtype=torch.float64),
            float((rho - 1.0) * predicted / norm**3),
        )

    def _values(self):
        position = self.coord.reshape(-1)
        projection = torch.dot(self.direction, position)
        quadratic = (
            torch.dot(self.gradient, position)
            + 0.5 * torch.dot(position, self.hessian @ position)
        )
        energy = quadratic + self.cubic_coefficient * projection**3
        gradient = (
            self.gradient + self.hessian @ position
            + 3.0 * self.cubic_coefficient * projection**2 * self.direction
        )
        hessian = (
            self.hessian
            + 6.0 * self.cubic_coefficient * projection
            * torch.outer(self.direction, self.direction)
        )
        return energy.reshape(1), -gradient.reshape(1, 3), hessian.unsqueeze(0)

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        energy, forces, _ = self._values()
        return energy, forces

    def get_efh_gpu(self):
        self._require_prepared()
        self.efh_calls += 1
        energy, forces, hessian = self._values()
        return energy, forces, hessian, torch.zeros(1, dtype=torch.long)


class _MixedQualityAdapter(_SyntheticFP64Adapter):
    """One conjugate quadratic member and one high-rho cubic member."""

    def __init__(self):
        self.cubic = _MisleadingCubicAdapter(rho=10.0, mass=1.0)
        super().__init__(actual_hessian=self.cubic.hessian)

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        energies, forces = [], []
        for index in range(len(self._atoms)):
            position = self.coord[index]
            energy = torch.dot(self.cubic.gradient, position) + 0.5 * torch.dot(
                position, self.cubic.hessian @ position
            )
            gradient = self.cubic.gradient + self.cubic.hessian @ position
            if index == 1:
                projection = torch.dot(self.cubic.direction, position)
                energy = energy + self.cubic.cubic_coefficient * projection**3
                gradient = gradient + (
                    3.0 * self.cubic.cubic_coefficient * projection**2
                    * self.cubic.direction
                )
            energies.append(energy)
            forces.append(-gradient)
        return torch.stack(energies), torch.stack(forces)


class _TrialAndRestoreFailureAdapter(_SyntheticFP64Adapter):
    def __init__(self):
        super().__init__(
            actual_hessian=torch.diag(
                torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
            )
        )

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        if self.step_calls:
            raise RuntimeError("synthetic trial failure")
        return (
            torch.zeros(len(self._atoms), dtype=torch.float64),
            torch.tensor([[-0.1, -0.05, -0.05]], dtype=torch.float64),
        )

    def restore_coords(self):
        self.restore_calls += 1
        raise RuntimeError("synthetic restore failure")


class _BatchStationaryAdapter(_SyntheticFP64Adapter):
    def get_efh_gpu(self):
        self._require_prepared()
        self.efh_calls += 1
        batch = len(self._atoms)
        forces = torch.zeros((batch, self.nmax), dtype=torch.float64)
        hessians = torch.zeros(
            (batch, self.nmax, self.nmax), dtype=torch.float64
        )
        for index, atoms in enumerate(self._atoms):
            curvatures = torch.as_tensor(
                atoms.info["curvatures"], dtype=torch.float64
            )
            length = curvatures.numel()
            hessians[index, :length, :length] = torch.diag(curvatures)
        return (
            torch.zeros(batch, dtype=torch.float64),
            forces,
            hessians,
            torch.zeros(batch, dtype=torch.long),
        )


class _RaggedShrinkingAdapter(_SyntheticFP64Adapter):
    def __init__(self):
        super().__init__()
        self.prepare_history = []

    def prepare(self, atoms_list, fixed_nmax=None):
        atoms_list = list(atoms_list)
        self.prepare_history.append(
            {
                "labels": [atoms.info["label"] for atoms in atoms_list],
                "positions": [atoms.get_positions().copy() for atoms in atoms_list],
                "fixed_nmax": fixed_nmax,
            }
        )
        super().prepare(atoms_list, fixed_nmax=fixed_nmax)

    def _values(self):
        energies = torch.zeros(len(self._atoms), dtype=torch.float64)
        forces = torch.zeros((len(self._atoms), self.nmax), dtype=torch.float64)
        hessians = torch.zeros(
            (len(self._atoms), self.nmax, self.nmax), dtype=torch.float64
        )
        start = 0
        for index, atoms in enumerate(self._atoms):
            length = 3 * len(atoms)
            curvatures = torch.ones(length, dtype=torch.float64)
            curvatures[0] = -1.0
            hessians[index, :length, :length] = torch.diag(curvatures)
            if atoms.info["label"] == "mover":
                x = self.coord[start, 0]
                energies[index] = 0.1 * x - 0.5 * x * x
                forces[index, 0] = -(0.1 - x)
            start += len(atoms)
        return energies, forces, hessians

    def get_ef_gpu(self):
        self._require_prepared()
        self.ef_calls += 1
        energy, forces, _ = self._values()
        return energy, forces

    def get_efh_gpu(self):
        self._require_prepared()
        self.efh_calls += 1
        energy, forces, hessians = self._values()
        return (
            energy,
            forces,
            hessians,
            torch.zeros(len(self._atoms), dtype=torch.long),
        )


class BatchPRFOExperimentalTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _optimizer(self, **kwargs):
        kwargs.setdefault("rigid_symmetry", "cartesian_external")
        optimizer = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "batch-prfo.out"),
            device="cpu",
            **kwargs,
        )
        optimizer._w = Mock()
        optimizer._orig_index = torch.tensor([0], dtype=torch.long)
        return optimizer

    @staticmethod
    def _one_atom():
        return Atoms("H", positions=[[0.0, 0.0, 0.0]])

    def _prepare_inner_loop(self, optimizer, adapter):
        atoms = self._one_atom()
        adapter.prepare([atoms])
        optimizer._D = torch.ones((1, 3), dtype=torch.float64)
        optimizer.tracked_mode_idx = torch.tensor([0], dtype=torch.long)
        return {
            "calc": adapter,
            "w": torch.tensor([[-1.0, 1.0, 2.0]], dtype=torch.float64),
            "V": torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "gp": torch.tensor([[0.10, 0.05, 0.05]], dtype=torch.float64),
            "H": torch.diag(
                torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
            ).unsqueeze(0),
            "g_cart": torch.tensor([[0.10, 0.05, 0.05]], dtype=torch.float64),
            "trust_r": torch.tensor([0.20], dtype=torch.float64),
            "last_step": torch.zeros((1, 3), dtype=torch.float64),
            "real_mask": torch.ones((1, 3), dtype=torch.bool),
            "E_old": torch.zeros(1, dtype=torch.float64),
        }

    def _assert_distance_only_inertia(
        self, positions, curvatures, expected_full, expected_internal
    ):
        atoms = Atoms("H3", positions=positions.copy())
        atoms.set_masses([1.0, 1.0, 1.0])
        adapter = _DistanceTriatomicAdapter(curvatures)
        adapter.prepare([atoms])
        energy, forces, hessians, _ = adapter.get_efh_gpu()
        self.assertLess(float(forces.abs().max()), 2.0e-3)
        self.assertTrue(torch.isfinite(energy).all())
        hessian = hessians[0].numpy()
        rigid = mass_weighted_rigid_basis(atoms)
        internal = np.linalg.qr(rigid, mode="complete")[0][:, rigid.shape[1]:]
        self.assertEqual(int(np.count_nonzero(np.linalg.eigvalsh(hessian) < -1e-6)), expected_full)
        internal_modes = np.linalg.eigvalsh(internal.T @ hessian @ internal)
        self.assertEqual(int(np.count_nonzero(internal_modes < -1e-6)), expected_internal)
        return atoms, adapter

    def test_analytic_distance_minimum_is_not_false_ts_candidate(self):
        atoms, adapter = self._assert_distance_only_inertia(
            _TRIATOMIC_NEAR_MINIMUM, [1.0, 1.0, 1.0], 1, 0
        )
        (result,) = self._optimizer(
            max_outer_iter=0, rigid_symmetry="free_molecule"
        )._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )
        self.assertIs(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.negative_modes, 0)

    def test_analytic_distance_saddle_is_not_falsely_rejected(self):
        atoms, adapter = self._assert_distance_only_inertia(
            _TRIATOMIC_NEAR_SADDLE, [-1.0, 1.0, 1.0], 3, 1
        )
        (result,) = self._optimizer(
            max_outer_iter=0, rigid_symmetry="free_molecule"
        )._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )
        self.assertIs(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.negative_modes, 1)

    def test_soft_internal_saddle_far_from_stationary_is_not_batch_candidate(self):
        atoms = _soft_saddle_bond_displacement(0.2)
        atoms.set_masses([1.0, 1.0, 1.0])
        adapter = _DistanceTriatomicAdapter([-1.0e-5, 1.0, 1.0])
        optimizer = self._optimizer(max_outer_iter=0, rigid_symmetry="free_molecule")

        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_MAXITER)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.negative_modes, 1)
        self.assertIn("unrestricted correction", result.detail)

    def test_soft_internal_saddle_at_or_near_stationary_is_batch_candidate(self):
        for shift in (0.0, 1.0e-4):
            with self.subTest(shift=shift):
                atoms = _soft_saddle_bond_displacement(shift)
                atoms.set_masses([1.0, 1.0, 1.0])
                adapter = _DistanceTriatomicAdapter([-1.0e-5, 1.0, 1.0])

                (result,) = self._optimizer(
                    max_outer_iter=0, rigid_symmetry="free_molecule"
                )._run_experimental(
                    SimpleNamespace(multiatoms=[atoms], calc=adapter)
                )

                self.assertIs(result.status, PRFOStatus.GEOMETRY_CONVERGED)
                self.assertEqual(result.negative_modes, 1)

    def test_unresolved_internal_mode_prevents_batch_candidate(self):
        atoms = Atoms("H3", positions=_TRIATOMIC_REFERENCE.copy())
        adapter = _DistanceTriatomicAdapter([-1.0e-5, 1.0e-8, 1.0])

        (result,) = self._optimizer(
            max_outer_iter=0, rigid_symmetry="free_molecule"
        )._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_MAXITER)
        self.assertEqual(result.negative_modes, 1)
        self.assertIn("unresolved physical curvature", result.detail)

    def test_ragged_soft_member_does_not_inherit_stationary_peers_candidate(self):
        class RaggedSoftAdapter(_RaggedShrinkingAdapter):
            def _values(self):
                energies, forces, hessians = super()._values()
                start = 0
                for index, atoms in enumerate(self._atoms):
                    if atoms.info["label"] == "soft":
                        displacement = self.coord[start, 0] - 0.2
                        energies[index] = -0.5e-5 * displacement**2
                        forces[index, 0] = 1.0e-5 * displacement
                        hessians[index, 0, 0] = -1.0e-5
                    start += len(atoms)
                return energies, forces, hessians

        stationary = self._one_atom()
        stationary.info["label"] = "stationary"
        soft = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        soft.info["label"] = "soft"
        for members in ([stationary, soft], [soft, stationary]):
            with self.subTest(first=members[0].info["label"]):
                results = self._optimizer(max_outer_iter=0)._run_experimental(
                    SimpleNamespace(multiatoms=members, calc=RaggedSoftAdapter())
                )
                by_label = {
                    atoms.info["label"]: result
                    for atoms, result in zip(members, results)
                }
                self.assertIs(
                    by_label["stationary"].status, PRFOStatus.GEOMETRY_CONVERGED
                )
                self.assertIs(by_label["soft"].status, PRFOStatus.FAILED_MAXITER)
                self.assertEqual(by_label["soft"].negative_modes, 1)

    def test_unknown_batch_backend_symmetry_fails_before_prepare(self):
        adapter = _SyntheticFP64Adapter()
        (result,) = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "unknown.out"), device="cpu",
        )._run_experimental(
            SimpleNamespace(multiatoms=[self._one_atom()], calc=adapter)
        )
        self.assertIs(result.status, PRFOStatus.FAILED_HESSIAN)
        self.assertIn("unknown symmetry fails closed", result.detail)
        self.assertEqual(adapter.prepare_calls, 0)

        source_declared = self._one_atom()
        source_declared.calc = SimpleNamespace(rigid_body_invariant=True)
        second_adapter = _SyntheticFP64Adapter()
        (second_result,) = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "unknown-with-source.out"),
            device="cpu",
        )._run_experimental(
            SimpleNamespace(multiatoms=[source_declared], calc=second_adapter)
        )
        self.assertIs(second_result.status, PRFOStatus.FAILED_HESSIAN)
        self.assertIn("unknown symmetry fails closed", second_result.detail)
        self.assertEqual(second_adapter.prepare_calls, 0)

    def test_declared_external_batch_backend_uses_cartesian_auto(self):
        adapter = _SyntheticFP64Adapter(
            actual_hessian=torch.diag(
                torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
            )
        )
        adapter.rigid_body_invariant = False
        (result,) = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "external-auto.out"),
            device="cpu", max_outer_iter=0,
        )._run_experimental(
            SimpleNamespace(multiatoms=[self._one_atom()], calc=adapter)
        )
        self.assertIs(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.negative_modes, 1)

    def test_periodic_and_free_atom_unsupported_spaces_fail_before_prepare(self):
        periodic = self._one_atom()
        periodic.set_cell([10.0, 10.0, 10.0])
        periodic.set_pbc(True)
        free_atom = self._one_atom()
        for atoms, declared, symmetry, diagnostic in (
            (periodic, False, "cartesian_external", "periodic"),
            (free_atom, True, "free_molecule", "internal degree"),
        ):
            with self.subTest(diagnostic=diagnostic):
                adapter = _SyntheticFP64Adapter()
                adapter.rigid_body_invariant = declared
                (result,) = self._optimizer(
                    rigid_symmetry=symmetry
                )._run_experimental(
                    SimpleNamespace(multiatoms=[atoms], calc=adapter)
                )
                self.assertIs(result.status, PRFOStatus.FAILED_HESSIAN)
                self.assertIn(diagnostic, result.detail)
                self.assertEqual(adapter.prepare_calls, 0)

    def test_auto_uses_the_actual_batch_backend_symmetry(self):
        atoms = Atoms("H3", positions=_TRIATOMIC_NEAR_MINIMUM.copy())
        adapter = _DistanceTriatomicAdapter([1.0, 1.0, 1.0])
        (result,) = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "declared.out"),
            device="cpu", max_outer_iter=0,
        )._run_experimental(SimpleNamespace(multiatoms=[atoms], calc=adapter))
        self.assertIs(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.negative_modes, 0)

    def test_near_zero_internal_negative_mode_is_not_promoted(self):
        atoms = Atoms("H3", positions=_TRIATOMIC_REFERENCE.copy())
        adapter = _DistanceTriatomicAdapter([-2.0e-7, 1.0, 1.0])
        (result,) = self._optimizer(
            max_outer_iter=0, rigid_symmetry="free_molecule"
        )._run_experimental(SimpleNamespace(multiatoms=[atoms], calc=adapter))
        self.assertIs(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.negative_modes, 0)

    def test_mixed_analytic_internal_members_keep_independent_inertia_and_order(self):
        minimum = Atoms("H3", positions=_TRIATOMIC_NEAR_MINIMUM.copy())
        saddle = Atoms("H3", positions=_TRIATOMIC_NEAR_SADDLE.copy())
        minimum.info["curvatures"] = [1.0, 1.0, 1.0]
        saddle.info["curvatures"] = [-1.0, 1.0, 1.0]
        for atoms_list in ([minimum, saddle], [saddle, minimum]):
            with self.subTest(order=atoms_list[0].info["curvatures"][0]):
                adapter = _DistanceTriatomicAdapter([1.0, 1.0, 1.0])
                results = self._optimizer(
                    max_outer_iter=0, rigid_symmetry="free_molecule"
                )._run_experimental(
                    SimpleNamespace(multiatoms=atoms_list, calc=adapter)
                )
                by_curvature = {
                    atoms.info["curvatures"][0]: result
                    for atoms, result in zip(atoms_list, results)
                }
                self.assertIs(by_curvature[1.0].status, PRFOStatus.FAILED_WRONG_INERTIA)
                self.assertEqual(by_curvature[1.0].negative_modes, 0)
                self.assertIs(by_curvature[-1.0].status, PRFOStatus.GEOMETRY_CONVERGED)
                self.assertEqual(by_curvature[-1.0].negative_modes, 1)

    def test_conflicting_member_and_batch_symmetry_fails_before_prepare(self):
        atoms = self._one_atom()
        atoms.calc = SimpleNamespace(rigid_body_invariant=False)
        adapter = _SyntheticFP64Adapter()
        adapter.rigid_body_invariant = True
        (result,) = self._optimizer(rigid_symmetry="free_molecule")._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )
        self.assertIs(result.status, PRFOStatus.FAILED_HESSIAN)
        self.assertIn("conflicting", result.detail)
        self.assertEqual(adapter.prepare_calls, 0)

    def test_linear_and_near_linear_batch_internal_rank_is_adaptive(self):
        atoms_list = [
            Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            Atoms("H3", positions=[[0, 0, 0], [1, 1e-4, 0], [2, 0, 0]]),
        ]
        optimizer = self._optimizer(rigid_symmetry="free_molecule")
        optimizer._spaces = ["free_molecule", "free_molecule"]
        optimizer._nmax = 9
        optimizer._arange_n = torch.arange(9)
        optimizer._rebuild_topology(atoms_list)
        optimizer._set_physical_bases(atoms_list)
        self.assertEqual(optimizer._P_vec.tolist(), [4, 3])
        for atoms, basis in zip(atoms_list, optimizer._physical_bases):
            rigid = mass_weighted_rigid_basis(atoms)
            np.testing.assert_allclose(rigid.T @ basis.cpu().numpy(), 0, atol=1e-12)

    def test_mode_tracking_and_step_use_same_internal_space(self):
        atoms = Atoms("H3", positions=_TRIATOMIC_NEAR_SADDLE.copy())
        atoms.set_masses([1.0, 2.0, 3.0])
        adapter = _DistanceTriatomicAdapter([-1.0, 1.0, 1.0])
        adapter.prepare([atoms])
        optimizer = self._optimizer(
            rigid_symmetry="free_molecule", max_inner_attempts=1,
        )
        optimizer._spaces = ["free_molecule"]
        optimizer._nmax = adapter.nmax_dof
        optimizer._arange_n = torch.arange(optimizer._nmax)
        optimizer._rebuild_topology([atoms])
        optimizer._set_physical_bases([atoms])
        energy, forces, hessian, _ = adapter.get_efh_gpu()
        hessian, gradient = optimizer._build_cartesian_hg(
            forces, hessian, optimizer._real_mask
        )
        h_mw, g_mw = optimizer._mass_weight_hg(hessian, gradient)
        eigenvalues, eigenvectors, eigen_gradient = optimizer._eigh_and_track_modes(
            h_mw, g_mw
        )
        rigid = mass_weighted_rigid_basis(atoms)
        np.testing.assert_allclose(
            rigid.T @ optimizer.tracked_mode_vec_mw[0].numpy(), 0, atol=1e-12
        )
        seen_steps = []
        original_step = adapter.step_cart_

        def record_step(step):
            seen_steps.append(step.clone())
            original_step(step)

        adapter.step_cart_ = record_step
        with patch(
            "maple.function.dispatcher.ts.algorithm.BPRFO.prfo_step_batched",
            wraps=prfo_step_batched,
        ) as kernel:
            optimizer._inner_rs_prfo_loop(
                calc=adapter, w=eigenvalues, V=eigenvectors,
                gp=eigen_gradient, H=hessian, g_cart=gradient,
                trust_r=torch.tensor([0.2], dtype=torch.float64),
                last_step=torch.zeros_like(gradient),
                real_mask=optimizer._real_mask, E_old=energy,
            )
        self.assertEqual(kernel.call_args.args[0].shape, (1, 3))
        self.assertTrue(seen_steps)
        basis = optimizer._physical_bases[0].numpy()
        expected_physical_step = prfo_step(
            basis.T @ h_mw[0].numpy() @ basis,
            basis.T @ g_mw[0].numpy(),
            is_ts=True,
            target_mode=int(optimizer.tracked_mode_idx[0]),
            trust_radius=0.2,
            pre_eig=(
                eigenvalues[0, :3].numpy(),
                eigenvectors[0, :3, :3].numpy(),
                eigen_gradient[0, :3].numpy(),
            ),
        )
        np.testing.assert_allclose(
            seen_steps[0][0].numpy(),
            optimizer._D[0, :9].numpy() * (basis @ expected_physical_step),
            rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            rigid.T @ (seen_steps[0][0].numpy() / optimizer._D[0, :9].numpy()),
            0, atol=1e-12,
        )

    def test_initial_mode_selection_excludes_zero_and_padded_modes(self):
        optimizer = self._optimizer()
        optimizer._L_vec = torch.tensor([3], dtype=torch.long)
        optimizer._real_mask = torch.tensor([[True, True, True, False]])
        optimizer.tracked_mode_vec_mw = None
        optimizer.tracked_mode_idx = None
        hessian = torch.diag(
            torch.tensor([0.0, 1.0, 2.0, BIG], dtype=torch.float64)
        ).unsqueeze(0)
        gradient = torch.tensor([[10.0, 0.1, 0.2, 0.0]], dtype=torch.float64)

        optimizer._eigh_and_track_modes(hessian, gradient)

        self.assertEqual(optimizer.tracked_mode_idx.tolist(), [1])

    def test_physical_curvature_above_padding_sentinel_remains_eligible(self):
        optimizer = self._optimizer()
        optimizer._L_vec = torch.tensor([3], dtype=torch.long)
        optimizer._real_mask = torch.tensor([[True, True, True, False]])
        optimizer.tracked_mode_vec_mw = None
        optimizer.tracked_mode_idx = None
        hessian = torch.diag(torch.tensor(
            [1.5 * BIG, 2.0 * BIG, 3.0 * BIG, BIG], dtype=torch.float64
        )).unsqueeze(0)
        gradient = torch.tensor([[0.1, 0.0, 0.0, 0.0]], dtype=torch.float64)

        optimizer._eigh_and_track_modes(hessian, gradient)

        self.assertEqual(optimizer.tracked_mode_idx.tolist(), [0])
        self.assertTrue(torch.count_nonzero(optimizer.tracked_mode_vec_mw[:, 3:]) == 0)

    def test_tracked_mode_overlap_excludes_zero_and_padded_modes(self):
        optimizer = self._optimizer()
        optimizer._L_vec = torch.tensor([3], dtype=torch.long)
        optimizer._real_mask = torch.tensor([[True, True, True, False]])
        optimizer.tracked_mode_idx = torch.tensor([3], dtype=torch.long)
        optimizer.tracked_mode_vec_mw = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
        )
        hessian = torch.diag(
            torch.tensor([0.0, 1.0, 2.0, BIG], dtype=torch.float64)
        ).unsqueeze(0)
        gradient = torch.zeros((1, 4), dtype=torch.float64)

        optimizer._eigh_and_track_modes(hessian, gradient)

        self.assertEqual(optimizer.tracked_mode_idx.tolist(), [1])

    def test_stationary_first_order_saddle_converges(self):
        optimizer = self._optimizer()
        atoms = self._one_atom()
        saddle_hessian = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
        )
        adapter = _SyntheticFP64Adapter(actual_hessian=saddle_hessian)
        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.negative_modes, 1)
        self.assertEqual(adapter.efh_calls, 1)

    def test_stationary_inertia_uses_current_backend_hessian_not_stale_working_hessian(self):
        optimizer = self._optimizer()
        atoms = self._one_atom()
        adapter = _SyntheticFP64Adapter(
            actual_hessian=torch.eye(3, dtype=torch.float64)
        )
        optimizer._H_work = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
        ).unsqueeze(0)

        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.negative_modes, 0)
        self.assertEqual(adapter.efh_calls, 1)

    def test_stationary_minimum_does_not_converge_as_saddle(self):
        optimizer = self._optimizer()
        atoms = self._one_atom()
        adapter = _SyntheticFP64Adapter()
        optimizer._H_work = torch.eye(3, dtype=torch.float64).unsqueeze(0)

        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
        self.assertEqual(result.negative_modes, 0)

    def test_exact_efh_forces_override_stale_preliminary_convergence_mask(self):
        class InconsistentEvaluation(_SyntheticFP64Adapter):
            def get_efh_gpu(self):
                energy, forces, hessian, padding = super().get_efh_gpu()
                forces[0, 0] = 1.0
                return energy, forces, hessian, padding

        saddle = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
        optimizer = self._optimizer(max_outer_iter=0)
        atoms = self._one_atom()
        (result,) = optimizer._run_experimental(
            SimpleNamespace(
                multiatoms=[atoms], calc=InconsistentEvaluation(actual_hessian=saddle)
            )
        )

        self.assertIs(result.status, PRFOStatus.FAILED_MAXITER)
        self.assertEqual(result.iterations, 0)
        self.assertNotIn("ts_candidate", result.structure_path)

    def test_trial_exception_restores_coordinates(self):
        optimizer = self._optimizer(max_inner_attempts=1)
        adapter = _SyntheticFP64Adapter(fail_trial=True)
        kwargs = self._prepare_inner_loop(optimizer, adapter)
        original = adapter.coord.clone()

        with self.assertRaisesRegex(RuntimeError, "synthetic trial failure"):
            optimizer._inner_rs_prfo_loop(**kwargs)

        torch.testing.assert_close(adapter.coord, original, rtol=0.0, atol=0.0)
        self.assertEqual(adapter.restore_calls, 1)

    def test_unsolved_physical_step_has_step_solver_status(self):
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=1)
        adapter = _ScaledQuadraticAdapter(scale=1.0)
        atoms = self._one_atom()
        with patch(
            "maple.function.dispatcher.ts.algorithm.BPRFO.prfo_step_batched",
            return_value=(
                torch.zeros((1, 3), dtype=torch.float64),
                torch.zeros(1, dtype=torch.bool),
            ),
        ):
            (result,) = optimizer._run_experimental(
                SimpleNamespace(multiatoms=[atoms], calc=adapter)
            )

        self.assertIs(result.status, PRFOStatus.FAILED_STEP_SOLVER)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(adapter.step_calls, 0)
        np.testing.assert_array_equal(result.atoms.positions, atoms.positions)

    def test_silent_trial_restore_failure_is_detected(self):
        class SilentRestore(_ScaledQuadraticAdapter):
            def restore_coords(self):
                self.restore_calls += 1
                self._backup = None

        atoms = self._one_atom()
        adapter = SilentRestore(scale=1.0)
        (result,) = self._optimizer(
            max_inner_attempts=1, max_outer_iter=1
        )._run_experimental(SimpleNamespace(multiatoms=[atoms], calc=adapter))

        self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("restore did not recover", result.detail)
        self.assertFalse(adapter._prepared)
        np.testing.assert_array_equal(result.atoms.positions, atoms.positions)

    def test_rejected_iteration_clears_previous_step(self):
        optimizer = self._optimizer(max_inner_attempts=1)
        adapter = _SyntheticFP64Adapter(trial_energy=0.0)
        kwargs = self._prepare_inner_loop(optimizer, adapter)
        kwargs["last_step"] = torch.ones((1, 3), dtype=torch.float64)
        original = adapter.coord.clone()

        _, last_step, _, accepted = optimizer._inner_rs_prfo_loop(**kwargs)

        self.assertEqual(accepted.tolist(), [False])
        torch.testing.assert_close(last_step, torch.zeros_like(last_step))
        torch.testing.assert_close(adapter.coord, original, rtol=0.0, atol=0.0)

    def test_configured_inner_attempt_budget_is_never_exceeded(self):
        optimizer = self._optimizer(max_inner_attempts=2)
        adapter = _SyntheticFP64Adapter(trial_energy=0.0)
        kwargs = self._prepare_inner_loop(optimizer, adapter)

        optimizer._inner_rs_prfo_loop(**kwargs)

        self.assertEqual(adapter.ef_calls, 2)

    def test_symmetric_quality_rejects_large_overprediction_and_uses_step_norm(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _MisleadingCubicAdapter(rho=10.0)
        kwargs = self._prepare_inner_loop(optimizer, adapter)
        original = adapter.coord.clone()
        expected_step = prfo_step(
            np.diag([-1.0, 1.0, 2.0]),
            np.array([0.10, 0.05, 0.05]),
            is_ts=True,
            target_mode=0,
            trust_radius=0.2,
            pre_eig=(
                np.array([-1.0, 1.0, 2.0]),
                np.eye(3),
                np.array([0.10, 0.05, 0.05]),
            ),
        )
        expected_radius = max(
            optimizer.trust_min,
            0.5 * min(0.2, float(np.linalg.norm(expected_step))),
        )

        trust, _, rho, accepted = optimizer._inner_rs_prfo_loop(**kwargs)

        self.assertAlmostEqual(float(rho[0]), 10.0, places=10)
        self.assertEqual(accepted.tolist(), [False])
        self.assertAlmostEqual(float(trust[0]), expected_radius, places=10)
        torch.testing.assert_close(adapter.coord, original, rtol=0.0, atol=0.0)

    def test_symmetric_quality_accepts_but_shrinks_marginal_trial(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _MisleadingCubicAdapter(rho=0.25)
        kwargs = self._prepare_inner_loop(optimizer, adapter)

        trust, last_step, rho, accepted = optimizer._inner_rs_prfo_loop(**kwargs)
        expected_radius = max(
            optimizer.trust_min,
            0.5 * min(0.2, float(torch.linalg.norm(last_step[0]))),
        )

        self.assertAlmostEqual(float(rho[0]), 0.25, places=10)
        self.assertEqual(accepted.tolist(), [True])
        self.assertAlmostEqual(float(trust[0]), expected_radius, places=10)

    def test_good_boundary_trial_expands_by_sqrt_two_and_caps(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            trust_init=0.02,
            trust_max=1.0,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _ScaledQuadraticAdapter(scale=1.0)
        kwargs = self._prepare_inner_loop(optimizer, adapter)
        kwargs["trust_r"] = torch.tensor([0.02], dtype=torch.float64)

        trust, _, rho, accepted = optimizer._inner_rs_prfo_loop(**kwargs)

        self.assertAlmostEqual(float(rho[0]), 1.0, places=10)
        self.assertEqual(accepted.tolist(), [True])
        self.assertAlmostEqual(float(trust[0]), np.sqrt(2.0) * 0.02, places=12)

    def test_constructor_defaults_match_scalar_quality_policy(self):
        optimizer = self._optimizer()
        self.assertEqual(optimizer.eta_reject, 0.0)
        self.assertEqual(optimizer.eta_shrink, 0.5)
        self.assertEqual(optimizer.eta_expand, 0.75)

    def test_accepted_trial_commits_once_without_duplicate_energy_evaluation(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _ScaledQuadraticAdapter(scale=1.0)
        kwargs = self._prepare_inner_loop(optimizer, adapter)

        _, last_step, _, accepted = optimizer._inner_rs_prfo_loop(**kwargs)

        self.assertEqual(accepted.tolist(), [True])
        self.assertEqual(adapter.ef_calls, 1)
        self.assertEqual(adapter.backup_calls, 2)
        self.assertEqual(adapter.restore_calls, 1)
        self.assertEqual(adapter.discard_calls, 1)
        self.assertIsNone(adapter._backup)
        self.assertTrue(adapter._prepared)
        self.assertEqual(adapter.step_calls, 2)
        torch.testing.assert_close(
            adapter.coord.reshape(1, -1), last_step, rtol=0.0, atol=0.0
        )

    def test_first_recalculated_iteration_applies_secant_update_after_acceptance(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            max_outer_iter=1,
            recalc=4,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        update = Mock(side_effect=BatchPRFO._bofill_update_batched)
        optimizer._bofill_update_batched = update
        adapter = _ScaledQuadraticAdapter(scale=1.0)
        atoms = self._one_atom()

        optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        update.assert_called_once()

    def test_trial_and_restore_failures_return_confirmed_geometry_snapshot(self):
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=1)
        adapter = _TrialAndRestoreFailureAdapter()
        atoms = self._one_atom()
        original = atoms.get_positions().copy()

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("synthetic trial failure", results[0].detail)
        self.assertIn("synthetic restore failure", results[0].detail)
        np.testing.assert_array_equal(results[0].atoms.get_positions(), original)
        self.assertEqual(adapter.restore_calls, 1)
        self.assertFalse(adapter._prepared)
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            adapter.get_ef_gpu()

    def test_partial_trial_step_failure_restores_confirmed_point(self):
        class PartialTrial(_ScaledQuadraticAdapter):
            def step_cart_(self, step):
                super().step_cart_(step)
                if self.step_calls == 1:
                    raise RuntimeError("partial trial step")

        adapter = PartialTrial(scale=1.0)
        atoms = self._one_atom()
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=1)
        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("partial trial step", result.detail)
        np.testing.assert_array_equal(result.atoms.positions, [[0.0, 0.0, 0.0]])
        torch.testing.assert_close(adapter.coord, torch.zeros_like(adapter.coord))
        self.assertGreaterEqual(adapter.restore_calls, 1)

    def test_trial_energy_force_failure_restores_confirmed_point(self):
        class FailedTrialEF(_ScaledQuadraticAdapter):
            def get_ef_gpu(self):
                self._require_prepared()
                if self.step_calls:
                    raise RuntimeError("trial E/F backend failure")
                return super().get_ef_gpu()

        adapter = FailedTrialEF(scale=1.0)
        atoms = self._one_atom()
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=1)
        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("trial E/F backend failure", result.detail)
        np.testing.assert_array_equal(result.atoms.positions, [[0.0, 0.0, 0.0]])
        torch.testing.assert_close(adapter.coord, torch.zeros_like(adapter.coord))
        self.assertGreaterEqual(adapter.restore_calls, 1)

    def test_partial_accepted_commit_failure_restores_confirmed_point(self):
        class PartialCommit(_ScaledQuadraticAdapter):
            def step_cart_(self, step):
                super().step_cart_(step)
                if self.step_calls == 2:
                    raise RuntimeError("partial accepted commit")

        adapter = PartialCommit(scale=1.0)
        atoms = self._one_atom()
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=1)
        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("partial accepted commit", result.detail)
        self.assertEqual(result.iterations, 0)
        np.testing.assert_array_equal(result.atoms.positions, [[0.0, 0.0, 0.0]])
        torch.testing.assert_close(adapter.coord, torch.zeros_like(adapter.coord))
        self.assertEqual(adapter.step_calls, 2)
        self.assertEqual(adapter.restore_calls, 2)

    def test_silent_accepted_commit_restore_failure_invalidates_calculator(self):
        class SilentCommitRestore(_ScaledQuadraticAdapter):
            def step_cart_(self, step):
                super().step_cart_(step)
                if self.step_calls == 2:
                    raise RuntimeError("partial accepted commit")

            def restore_coords(self):
                if self.restore_calls == 0:
                    return super().restore_coords()
                self.restore_calls += 1
                self._backup = None

        atoms = self._one_atom()
        adapter = SilentCommitRestore(scale=1.0)
        (result,) = self._optimizer(
            max_inner_attempts=1, max_outer_iter=1
        )._run_experimental(SimpleNamespace(multiatoms=[atoms], calc=adapter))

        self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("restore", result.detail)
        self.assertFalse(adapter._prepared)
        np.testing.assert_array_equal(result.atoms.positions, [[0.0, 0.0, 0.0]])

    def test_later_pending_failure_cannot_leave_earlier_member_partly_committed(self):
        class MixedTrialFailure(_MixedQualityAdapter):
            def get_ef_gpu(self):
                if self.ef_calls == 1:
                    self.ef_calls += 1
                    raise RuntimeError("later pending trial failed")
                return super().get_ef_gpu()

        atoms = [self._one_atom(), self._one_atom()]
        adapter = MixedTrialFailure()
        gradient = adapter.cubic.gradient
        hessian = adapter.cubic.hessian
        adapter.prepare(atoms)
        optimizer = self._optimizer(max_inner_attempts=2)
        optimizer._D = torch.ones((2, 3), dtype=torch.float64)
        optimizer.tracked_mode_idx = torch.zeros(2, dtype=torch.long)
        eigenvalues = torch.tensor([[-1.0, 1.0, 2.0]] * 2, dtype=torch.float64)
        basis = torch.eye(3, dtype=torch.float64).expand(2, -1, -1).clone()
        gradients = gradient.expand(2, -1).clone()
        hessians = hessian.expand(2, -1, -1).clone()
        original = adapter.coord.clone()

        with self.assertRaisesRegex(RuntimeError, "later pending trial failed"):
            optimizer._inner_rs_prfo_loop(
                calc=adapter, w=eigenvalues, V=basis, gp=gradients,
                H=hessians, g_cart=gradients,
                trust_r=torch.full((2,), 0.2, dtype=torch.float64),
                last_step=torch.zeros((2, 3), dtype=torch.float64),
                real_mask=torch.ones((2, 3), dtype=torch.bool),
                E_old=torch.zeros(2, dtype=torch.float64),
            )

        torch.testing.assert_close(adapter.coord, original, rtol=0, atol=0)
        self.assertEqual(adapter.step_calls, 2)  # trial A/B, then pending B
        self.assertEqual(adapter.backup_calls, 2)
        self.assertEqual(adapter.restore_calls, 2)
        self.assertEqual(adapter.discard_calls, 0)

    def test_mixed_acceptance_shrink_keeps_survivor_step_storage_aligned(self):
        atoms = [self._one_atom(), self._one_atom()]
        for item in atoms:
            item.set_masses([1.0])
            item.f_max_th = 1.0e-12
            item.f_rms_th = 1.0e-12
        adapter = _MixedQualityAdapter()
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=3)

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=atoms, calc=adapter)
        )

        self.assertEqual(len(results), 2)
        self.assertIs(results[1].status, PRFOStatus.FAILED_STAGNATION)
        self.assertEqual(results[1].iterations, 0)
        self.assertIsNot(results[0].status, PRFOStatus.FAILED_BACKEND)
        self.assertGreaterEqual(results[0].iterations, 2)

    def test_post_commit_reprepare_failure_counts_confirmed_accepted_step(self):
        class FailSecondPrepare(_MixedQualityAdapter):
            def prepare(self, atoms_list, fixed_nmax=None):
                if self.prepare_calls:
                    self._prepared = False
                    raise RuntimeError("post-commit reprepare failed")
                return super().prepare(atoms_list, fixed_nmax=fixed_nmax)

        atoms = [self._one_atom(), self._one_atom()]
        for item in atoms:
            item.set_masses([1.0])
            item.f_max_th = 1.0e-12
            item.f_rms_th = 1.0e-12
        optimizer = self._optimizer(max_inner_attempts=1, max_outer_iter=3)
        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=atoms, calc=FailSecondPrepare())
        )

        self.assertEqual(len(results), 2)
        self.assertIs(results[0].status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("post-commit reprepare failed", results[0].detail)
        self.assertEqual(results[0].iterations, 1)
        self.assertNotEqual(results[0].atoms.positions[0, 0], 0.0)
        self.assertIs(results[1].status, PRFOStatus.FAILED_STAGNATION)
        self.assertEqual(results[1].iterations, 0)

    def test_rejected_peer_is_not_round_tripped_through_calculator_dtype(self):
        accepted = self._one_atom()
        rejected = Atoms("H", positions=[[0.123456789, 0.0, 0.0]])
        original = rejected.positions.copy()
        optimizer = self._optimizer()
        optimizer._ptr = torch.tensor([0, 1, 2], dtype=torch.long)
        calculator = SimpleNamespace(coord=torch.tensor(
            [[0.1, 0.0, 0.0], [0.12345679, 0.0, 0.0]],
            dtype=torch.float32,
        ))

        optimizer._sync_atoms_from_calc(
            calculator, [accepted, rejected],
            torch.tensor([True, False], dtype=torch.bool),
        )

        self.assertNotEqual(accepted.positions[0, 0], 0.0)
        np.testing.assert_array_equal(rejected.positions, original)

    def test_rejected_trial_budget_returns_stagnation_at_confirmed_current_point(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            max_outer_iter=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _MisleadingCubicAdapter(rho=10.0, mass=self._one_atom().get_masses()[0])
        atoms = self._one_atom()
        original = atoms.get_positions().copy()

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, PRFOStatus.FAILED_STAGNATION)
        self.assertEqual(results[0].iterations, 0)
        np.testing.assert_array_equal(results[0].atoms.get_positions(), original)

    def test_accepted_step_count_and_terminal_geometry_match_confirmed_point(self):
        optimizer = self._optimizer(
            max_inner_attempts=1,
            max_outer_iter=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _ScaledQuadraticAdapter(scale=1.0)
        atoms = self._one_atom()

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(results[0].iterations, 1)
        np.testing.assert_allclose(
            results[0].atoms.get_positions(),
            adapter.coord.detach().cpu().numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_nonpositive_or_nonfinite_mass_returns_failed_hessian_before_prepare(self):
        for mass in (0.0, -1.0, float("nan")):
            with self.subTest(mass=mass):
                atoms = self._one_atom()
                atoms.set_masses([mass])
                original_positions = atoms.get_positions().copy()
                adapter = _ForbiddenPrepareAdapter()
                mols = SimpleNamespace(multiatoms=[atoms], calc=adapter)
                optimizer = self._optimizer(max_outer_iter=1)

                results = optimizer._run_experimental(mols)

                self.assertEqual(adapter.prepare_calls, 0)
                self.assertIsInstance(results, tuple)
                self.assertEqual(len(results), 1)
                self.assertIsInstance(results[0], PRFOResult)
                self.assertIs(results[0].status, PRFOStatus.FAILED_HESSIAN)
                self.assertEqual(results[0].iterations, 0)
                self.assertEqual(results[0].structure_path, "")
                self.assertRegex(results[0].detail, "mass|masses|positive|finite")
                np.testing.assert_array_equal(
                    results[0].atoms.get_positions(), original_positions
                )

    def test_invalid_member_convergence_thresholds_fail_before_prepare(self):
        for attribute in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
            for value in (
                0.0, -1.0, float("nan"), float("inf"),
                "loose", "0.002", True, np.bool_(True), np.bool_(False),
            ):
                with self.subTest(attribute=attribute, value=value):
                    atoms = self._one_atom()
                    setattr(atoms, attribute, value)
                    adapter = _ForbiddenPrepareAdapter()
                    results = self._optimizer(max_outer_iter=0)._run_experimental(
                        SimpleNamespace(multiatoms=[atoms], calc=adapter)
                    )

                    self.assertEqual(adapter.prepare_calls, 0)
                    self.assertIs(results[0].status, PRFOStatus.FAILED_HESSIAN)
                    self.assertIn("threshold", results[0].detail.lower())
                    self.assertEqual(results[0].structure_path, "")

    def test_nonfinite_coordinates_fail_only_that_member_before_prepare(self):
        invalid = self._one_atom()
        invalid.positions[0, 0] = np.nan
        valid = self._one_atom()
        saddle = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
        adapter = _SyntheticFP64Adapter(actual_hessian=saddle)
        optimizer = self._optimizer(max_outer_iter=0)

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[invalid, valid], calc=adapter)
        )

        self.assertEqual([item.status for item in results], [
            PRFOStatus.FAILED_NONFINITE, PRFOStatus.GEOMETRY_CONVERGED,
        ])
        self.assertEqual(results[0].structure_path, "")
        self.assertEqual(adapter.prepare_calls, 1)
        self.assertEqual(adapter._atoms, [valid])

    def test_invalid_member_snapshot_preserves_calculator_identity_without_copying_model(self):
        class UncopyableCalculator:
            def __deepcopy__(self, _):
                raise AssertionError("model calculator must not be deep-copied")

        atoms = self._one_atom()
        calc = UncopyableCalculator()
        atoms.calc = calc
        atoms.set_constraint(FixAtoms(indices=[0]))
        optimizer = self._optimizer(max_outer_iter=0)

        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=_ForbiddenPrepareAdapter())
        )

        self.assertIs(result.status, PRFOStatus.FAILED_HESSIAN)
        self.assertIs(result.atoms.calc, calc)
        self.assertIsNot(result.atoms, atoms)

    def test_constraints_return_failed_hessian_before_prepare(self):
        atoms = self._one_atom()
        atoms.set_constraint(FixAtoms(indices=[0]))
        original_positions = atoms.get_positions().copy()
        adapter = _ForbiddenPrepareAdapter()
        mols = SimpleNamespace(multiatoms=[atoms], calc=adapter)
        optimizer = self._optimizer(max_outer_iter=1)

        results = optimizer._run_experimental(mols)

        self.assertEqual(adapter.prepare_calls, 0)
        self.assertIsInstance(results, tuple)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, PRFOStatus.FAILED_HESSIAN)
        self.assertEqual(results[0].iterations, 0)
        self.assertEqual(results[0].structure_path, "")
        self.assertRegex(results[0].detail, "constraint|constrained")
        np.testing.assert_array_equal(
            results[0].atoms.get_positions(), original_positions
        )

    def test_invalid_member_snapshots_do_not_alias_mutable_atoms_state(self):
        first = self._one_atom()
        first.info["label"] = {"value": 1}
        first.set_constraint(FixAtoms(indices=[0]))
        second = self._one_atom()
        second.positions[0, 0] = 1.0
        second.set_masses([0.0])
        adapter = _ForbiddenPrepareAdapter()
        optimizer = self._optimizer(max_outer_iter=1)

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[first, second], calc=adapter)
        )

        first.positions[0, 0] = 9.0
        first.info["label"]["value"] = 2
        first.constraints[0].index[0] = 99
        second.positions[0, 0] = 8.0
        self.assertEqual(adapter.prepare_calls, 0)
        self.assertEqual([result.status for result in results], [
            PRFOStatus.FAILED_HESSIAN,
            PRFOStatus.FAILED_HESSIAN,
        ])
        self.assertEqual(results[0].atoms.positions[0, 0], 0.0)
        self.assertEqual(results[0].atoms.info["label"]["value"], 1)
        self.assertNotEqual(results[0].atoms.constraints[0].index[0], 99)
        self.assertEqual(results[1].atoms.positions[0, 0], 1.0)
        self.assertIsNot(results[0].atoms.positions, first.positions)
        self.assertIsNot(results[0].atoms.info, first.info)
        self.assertIsNot(results[0].atoms.constraints, first.constraints)

    def test_mixed_invalid_and_valid_members_preserve_original_result_order(self):
        invalid = self._one_atom()
        invalid.set_constraint(FixAtoms(indices=[0]))
        valid = self._one_atom()
        saddle_hessian = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
        )
        adapter = _SyntheticFP64Adapter(actual_hessian=saddle_hessian)
        optimizer = self._optimizer(max_outer_iter=1)

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[invalid, valid], calc=adapter)
        )

        self.assertEqual(len(results), 2)
        self.assertIs(results[0].status, PRFOStatus.FAILED_HESSIAN)
        self.assertIs(results[1].status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(results[0].iterations, 0)
        self.assertEqual(results[1].iterations, 0)
        self.assertEqual(adapter.prepare_calls, 1)
        self.assertEqual(len(adapter._atoms), 1)
        self.assertIs(adapter._atoms[0], valid)

    def test_invalid_global_parameters_raise_before_creating_artifacts(self):
        directory = Path(self.tmpdir.name) / "no-created-directory"
        output = directory / "invalid.out"
        with self.assertRaisesRegex(ValueError, "max_inner_attempts"):
            BatchPRFO(
                output=str(output),
                device="cpu",
                max_inner_attempts=0,
            )
        self.assertFalse(output.exists())
        self.assertFalse(directory.exists())

    def test_global_configuration_rejects_invalid_trust_quality_and_budgets_before_io(self):
        invalid = (
            ({"trust_min": 0.0}, "trust_min"),
            ({"trust_init": 0.0}, "trust_init"),
            ({"trust_min": 0.3, "trust_init": 0.2}, "trust"),
            ({"trust_max": 0.1, "trust_init": 0.2}, "trust"),
            ({"eta_shrink": 0.8, "eta_expand": 0.7}, "eta"),
            ({"eta_expand": 1.75}, "eta_expand"),
            ({"max_inner_attempts": True}, "max_inner_attempts"),
            ({"max_outer_iter": -1}, "max_outer_iter"),
            ({"recalc": 0}, "recalc"),
        )
        for index, (options, message) in enumerate(invalid):
            with self.subTest(options=options):
                directory = Path(self.tmpdir.name) / f"invalid-{index}"
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    BatchPRFO(
                        output=str(directory / "job.out"),
                        device="cpu",
                        **options,
                    )
                self.assertFalse(directory.exists())

    def test_output_paths_are_namespaced_by_output_basename(self):
        first = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "first.out"), device="cpu"
        )
        second = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "second.out"), device="cpu"
        )

        first._init_xyz_paths(2)
        second._init_xyz_paths(2)

        self.assertTrue(set(first.xyz_paths).isdisjoint(second.xyz_paths))
        self.assertEqual(
            [Path(path).name for path in first.xyz_paths],
            ["first_batch0001_prfo_traj.xyz", "first_batch0002_prfo_traj.xyz"],
        )

    def test_every_terminal_status_uses_exact_scalar_compatible_suffix(self):
        suffixes = {
            PRFOStatus.GEOMETRY_CONVERGED: "_prfo_ts_candidate",
            PRFOStatus.FAILED_NONFINITE: "_prfo_nonfinite",
            PRFOStatus.FAILED_BACKEND: "_prfo_backend_failure",
            PRFOStatus.FAILED_HESSIAN: "_prfo_hessian_failure",
            PRFOStatus.FAILED_WRONG_INERTIA: "_prfo_wrong_inertia",
            PRFOStatus.FAILED_STAGNATION: "_prfo_stagnation",
            PRFOStatus.FAILED_STEP_SOLVER: "_prfo_step_failure",
            PRFOStatus.FAILED_MODE_TRACKING: "_prfo_mode_failure",
            PRFOStatus.FAILED_MAXITER: "_prfo_unconverged",
        }
        for status, suffix in suffixes.items():
            with self.subTest(status=status):
                optimizer = self._optimizer()
                result = optimizer._terminal_result(
                    original_index=0,
                    atoms=self._one_atom(),
                    status=status,
                    iteration=0,
                    energy=None,
                    negative_modes=None,
                    detail="synthetic terminal test",
                )
                expected = Path(self.tmpdir.name) / (
                    f"batch-prfo_batch0001{suffix}.xyz"
                )
                self.assertEqual(result.structure_path, str(expected))
                self.assertTrue(expected.is_file())

    def test_corrupt_prepared_pointer_order_coordinates_device_or_width_fails_closed(self):
        def bad_pointer(calc):
            calc._ptr[-1] = 0

        def bad_numbers(calc):
            calc.numbers[0] = 8

        def bad_coordinates(calc):
            calc.coord[0, 0] = 0.5

        def bad_device(calc):
            calc.device = torch.device("cuda:0")

        def bad_width(calc):
            calc.nmax_dof = 2

        for name, damage in (
            ("pointer", bad_pointer),
            ("numbers", bad_numbers),
            ("coordinates", bad_coordinates),
            ("device", bad_device),
            ("width", bad_width),
        ):
            with self.subTest(case=name):
                class CorruptPrepared(_SyntheticFP64Adapter):
                    def prepare(self, atoms_list, fixed_nmax=None):
                        super().prepare(atoms_list, fixed_nmax=fixed_nmax)
                        damage(self)

                optimizer = self._optimizer(max_outer_iter=0)
                atoms = self._one_atom()
                adapter = CorruptPrepared()
                (result,) = optimizer._run_experimental(
                    SimpleNamespace(multiatoms=[atoms], calc=adapter)
                )
                self.assertIs(result.status, PRFOStatus.FAILED_BACKEND)
                self.assertEqual(result.iterations, 0)
                np.testing.assert_array_equal(result.atoms.positions, atoms.positions)
                self.assertIn(name, result.detail.lower())

    def test_malformed_efh_backend_arrays_fail_without_candidate(self):
        def bad_energy(calc):
            return torch.zeros(2, dtype=torch.float64), torch.zeros((1, 3), dtype=torch.float64)

        def bad_forces(calc):
            return torch.zeros(1, dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64)

        def bad_h_shape(calc):
            return (torch.zeros(1, dtype=torch.float64),
                torch.zeros((1, 3), dtype=torch.float64),
                torch.zeros((1, 2, 2), dtype=torch.float64),
                torch.zeros(1, dtype=torch.long))

        def bad_h_finite(calc):
            hessian = torch.eye(3, dtype=torch.float64).unsqueeze(0)
            hessian[0, 0, 0] = float("nan")
            return (torch.zeros(1, dtype=torch.float64),
                torch.zeros((1, 3), dtype=torch.float64), hessian,
                torch.zeros(1, dtype=torch.long))

        for name, ef_damage, h_damage, status in (
            ("energy", bad_energy, None, PRFOStatus.FAILED_BACKEND),
            ("forces", bad_forces, None, PRFOStatus.FAILED_BACKEND),
            ("hessian shape", None, bad_h_shape, PRFOStatus.FAILED_HESSIAN),
            ("hessian finite", None, bad_h_finite, PRFOStatus.FAILED_NONFINITE),
        ):
            with self.subTest(case=name):
                class Malformed(_SyntheticFP64Adapter):
                    def get_ef_gpu(self):
                        return ef_damage(self) if ef_damage else super().get_ef_gpu()

                    def get_efh_gpu(self):
                        return h_damage(self) if h_damage else super().get_efh_gpu()

                atoms = self._one_atom()
                (result,) = self._optimizer(max_outer_iter=0)._run_experimental(
                    SimpleNamespace(multiatoms=[atoms], calc=Malformed())
                )
                self.assertIs(result.status, status)
                self.assertNotIn("ts_candidate", result.structure_path)
                self.assertIn(name.split()[0], result.detail.lower())

    def test_ragged_stationary_results_are_invariant_to_input_permutation(self):
        one = self._one_atom()
        one.info.update(label="one", curvatures=[-1.0, 1.0, 1.0])
        two = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]
        )
        two.info.update(label="two", curvatures=[1.0] * 6)
        expected = {
            "one": PRFOStatus.GEOMETRY_CONVERGED,
            "two": PRFOStatus.FAILED_WRONG_INERTIA,
        }

        for order in ([one, two], [two, one]):
            with self.subTest(order=[atoms.info["label"] for atoms in order]):
                optimizer = self._optimizer(max_outer_iter=1)
                adapter = _BatchStationaryAdapter()
                results = optimizer._run_experimental(
                    SimpleNamespace(multiatoms=order, calc=adapter)
                )

                self.assertIsInstance(results, tuple)
                self.assertEqual(len(results), 2)
                for atoms, result in zip(order, results):
                    self.assertIs(result.status, expected[atoms.info["label"]])
                    self.assertEqual(result.iterations, 0)
                    np.testing.assert_array_equal(
                        result.atoms.get_positions(), atoms.get_positions()
                    )

    def test_ragged_batch_shrinks_to_survivor_at_confirmed_coordinates(self):
        stationary = self._one_atom()
        stationary.info["label"] = "stationary"
        mover = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]
        )
        mover.info["label"] = "mover"
        optimizer = self._optimizer(
            max_outer_iter=1,
            max_inner_attempts=1,
            eta_shrink=0.5,
            eta_expand=0.75,
        )
        adapter = _RaggedShrinkingAdapter()

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[stationary, mover], calc=adapter)
        )

        self.assertEqual([result.status for result in results], [
            PRFOStatus.GEOMETRY_CONVERGED,
            PRFOStatus.GEOMETRY_CONVERGED,
        ])
        self.assertEqual([result.iterations for result in results], [0, 1])
        self.assertEqual(adapter.prepare_history[0]["labels"], [
            "stationary", "mover"
        ])
        self.assertEqual(adapter.prepare_history[1]["labels"], ["mover"])
        self.assertEqual(adapter.prepare_history[1]["fixed_nmax"], 6)
        # The stationary peer is removed before any trial; reprepare must still
        # see the mover's confirmed starting point, not a speculative step.
        self.assertEqual(adapter.prepare_history[1]["positions"][0][0, 0], 0.0)
        self.assertNotEqual(results[1].atoms.positions[0, 0], 0.0)

    def test_reprepare_failure_preserves_original_member_result_alignment(self):
        class FailSecondPrepare(_RaggedShrinkingAdapter):
            def prepare(self, atoms_list, fixed_nmax=None):
                if self.prepare_calls:
                    self._prepared = False
                    raise RuntimeError("second prepare failed")
                return super().prepare(atoms_list, fixed_nmax=fixed_nmax)

        stationary = self._one_atom()
        stationary.info["label"] = "stationary"
        mover = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
        mover.info["label"] = "mover"
        initial_mover = mover.positions.copy()
        optimizer = self._optimizer(max_outer_iter=1)

        results = optimizer._run_experimental(
            SimpleNamespace(
                multiatoms=[stationary, mover], calc=FailSecondPrepare()
            )
        )

        self.assertEqual(len(results), 2)
        self.assertIs(results[0].status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertIs(results[1].status, PRFOStatus.FAILED_BACKEND)
        self.assertIn("second prepare failed", results[1].detail)
        self.assertEqual(results[1].iterations, 0)
        np.testing.assert_array_equal(results[1].atoms.positions, initial_mover)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_implicit_cuda_device_uses_concrete_prepared_tensor_device(self):
        class CUDASynthetic(_SyntheticFP64Adapter):
            def __init__(self):
                super().__init__(actual_hessian=torch.diag(torch.tensor(
                    [-1.0, 1.0, 1.0], dtype=torch.float64
                )))
                self.device = torch.device("cuda")

            def prepare(self, atoms_list, fixed_nmax=None):
                super().prepare(atoms_list, fixed_nmax=fixed_nmax)
                self.coord = self.coord.to(self.device)
                self._ptr = self._ptr.to(self.device)
                self.numbers = self.numbers.to(self.device)

            def get_ef_gpu(self):
                energy, forces = super().get_ef_gpu()
                return energy.to(self.device), forces.to(self.device)

            def get_efh_gpu(self):
                energy, forces, hessian, padding = super().get_efh_gpu()
                return (energy.to(self.device), forces.to(self.device),
                        hessian.to(self.device), padding.to(self.device))

        optimizer = BatchPRFO(
            output=str(Path(self.tmpdir.name) / "implicit-cuda.out"),
            max_outer_iter=0,
            rigid_symmetry="cartesian_external",
        )
        (result,) = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[self._one_atom()], calc=CUDASynthetic())
        )

        self.assertIs(result.status, PRFOStatus.GEOMETRY_CONVERGED)
        self.assertEqual(result.negative_modes, 1)

    def test_pdb_candidate_output_uses_job_basename_and_pdb_format(self):
        atoms = self._one_atom()
        atoms.info["curvatures"] = [-1.0, 1.0, 1.0]
        atoms.info["pdb_template"] = [
            "ATOM      1  H   MOL A   1       0.000   0.000   0.000  "
            "1.00  0.00           H  \n",
            "END\n",
        ]
        output = Path(self.tmpdir.name) / "pdb-job.out"
        optimizer = BatchPRFO(
            output=str(output), device="cpu", max_outer_iter=1,
            rigid_symmetry="cartesian_external",
        )
        adapter = _BatchStationaryAdapter()

        results = optimizer._run_experimental(
            SimpleNamespace(multiatoms=[atoms], calc=adapter)
        )

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, PRFOStatus.GEOMETRY_CONVERGED)
        path = Path(results[0].structure_path)
        self.assertEqual(path.name, "pdb-job_batch0001_prfo_ts_candidate.pdb")
        self.assertEqual(path.suffix, ".pdb")
        self.assertTrue(path.is_file())

    def test_public_run_remains_runtime_disabled(self):
        optimizer = self._optimizer()
        with self.assertRaisesRegex(
            NotImplementedError, "experimental and runtime-disabled"
        ):
            optimizer.run(object())


class BatchPRFOBatchMathTests(unittest.TestCase):
    @staticmethod
    def _kernel():
        from maple.function.dispatcher.ts.algorithm._prfo_batch_math import (
            prfo_step_batched,
        )

        return prfo_step_batched

    @staticmethod
    def _scalar(w, V, gp, target, radius, *, max_bisect_it=60):
        hessian = V @ np.diag(w) @ V.T
        gradient = V @ gp
        return prfo_step(
            hessian,
            gradient,
            is_ts=True,
            target_mode=target,
            trust_radius=radius,
            max_bisect_it=max_bisect_it,
            pre_eig=(w, V, gp),
        )

    def test_positive_curvature_target_matches_scalar_uphill_step(self):
        kernel = self._kernel()
        w = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        V = torch.eye(2, dtype=torch.float64).unsqueeze(0)
        gp = torch.tensor([[0.05, 0.10]], dtype=torch.float64)
        target = torch.tensor([0], dtype=torch.long)
        radius = torch.tensor([0.2], dtype=torch.float64)

        steps, success = kernel(w, V, gp, target, radius)
        expected = self._scalar(
            w[0].numpy(), V[0].numpy(), gp[0].numpy(), 0, 0.2
        )

        self.assertEqual(success.tolist(), [True])
        torch.testing.assert_close(
            steps[0], torch.from_numpy(expected), rtol=1e-10, atol=1e-12
        )
        self.assertGreater(float(gp[0, 0] * steps[0, 0]), 0.0)
        self.assertLess(float(gp[0, 1] * steps[0, 1]), 0.0)
        self.assertLessEqual(float(torch.linalg.norm(steps[0])), 0.2 + 1e-12)

    def test_per_member_shared_alpha_matches_stacked_scalar_oracles(self):
        kernel = self._kernel()
        w = torch.tensor(
            [[-1.2, 0.8, 2.0], [0.5, 1.5, 3.0]], dtype=torch.float64
        )
        V = torch.eye(3, dtype=torch.float64).expand(2, -1, -1).clone()
        gp = torch.tensor(
            [[0.4, -0.2, 0.1], [0.05, 0.10, -0.03]], dtype=torch.float64
        )
        target = torch.tensor([0, 1], dtype=torch.long)
        radius = torch.tensor([0.12, 0.8], dtype=torch.float64)

        steps, success = kernel(w, V, gp, target, radius)
        expected = torch.stack(
            [
                torch.from_numpy(
                    self._scalar(
                        w[index].numpy(),
                        V[index].numpy(),
                        gp[index].numpy(),
                        int(target[index]),
                        float(radius[index]),
                    )
                )
                for index in range(2)
            ]
        )

        self.assertEqual(success.tolist(), [True, True])
        torch.testing.assert_close(steps, expected, rtol=1e-10, atol=1e-12)
        self.assertLessEqual(
            float(torch.linalg.norm(steps[0]) - radius[0]), 1e-10
        )

    def test_nonfinite_member_fails_without_corrupting_finite_member(self):
        kernel = self._kernel()
        w = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
        V = torch.eye(2, dtype=torch.float64).expand(2, -1, -1).clone()
        gp = torch.tensor([[float("nan"), 0.1], [0.05, 0.1]], dtype=torch.float64)
        target = torch.tensor([0, 0], dtype=torch.long)
        radius = torch.tensor([0.2, 0.2], dtype=torch.float64)

        steps, success = kernel(w, V, gp, target, radius)
        expected = self._scalar(
            w[1].numpy(), V[1].numpy(), gp[1].numpy(), 0, 0.2
        )

        self.assertEqual(success.tolist(), [False, True])
        torch.testing.assert_close(
            steps[1], torch.from_numpy(expected), rtol=1e-10, atol=1e-12
        )

    def test_bounded_bracket_failure_does_not_fail_unrestricted_member(self):
        kernel = self._kernel()
        w = torch.tensor([[0.1, 0.2], [-1.0, 2.0]], dtype=torch.float64)
        V = torch.eye(2, dtype=torch.float64).expand(2, -1, -1).clone()
        gp = torch.tensor([[10.0, 10.0], [1e-4, 1e-4]], dtype=torch.float64)
        target = torch.tensor([0, 0], dtype=torch.long)
        radius = torch.tensor([1e-8, 1.0], dtype=torch.float64)

        steps, success = kernel(
            w, V, gp, target, radius, max_bisect_it=1
        )
        expected = self._scalar(
            w[1].numpy(), V[1].numpy(), gp[1].numpy(), 0, 1.0
        )

        self.assertEqual(success.tolist(), [False, True])
        torch.testing.assert_close(
            steps[1], torch.from_numpy(expected), rtol=1e-10, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
