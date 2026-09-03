import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from maple.function.calculator._batch_types import BatchResult
from maple.function.dispatcher.optimization.algorithm.blbfgs import BatchLBFGS
from maple.function.dispatcher.optimization.algorithm.calculate_many_batch import (
    CalculateManyBatchCalc,
)


class _Molecules:
    def __init__(self, atoms, calc):
        self.multiatoms = atoms
        self.calc = calc


class _SyntheticBatchCalculator:
    def __init__(self, *, nonfinite=False, stationary=False):
        self.device = torch.device("cpu")
        self.nonfinite = nonfinite
        self.stationary = stationary
        self.coord = torch.zeros((0, 3), dtype=torch.float64)
        self._backup = None
        self._atoms = []

    def prepare(self, atoms_list, fixed_nmax=None):
        self._atoms = list(atoms_list)
        ptr = [0]
        for atoms in self._atoms:
            ptr.append(ptr[-1] + len(atoms))
        self._ptr = torch.tensor(ptr, dtype=torch.long)
        self.coord = torch.from_numpy(
            np.concatenate(
                [atoms.get_positions() for atoms in atoms_list],
                axis=0,
            )
        ).to(
            dtype=torch.float64,
        )
        self.nmax = fixed_nmax or max(3 * len(atoms) for atoms in atoms_list)
        self.nmax_dof = self.nmax

    def get_ef_gpu(self):
        batch_size = len(self._atoms)
        energy = torch.zeros(batch_size, dtype=torch.float64)
        force = torch.zeros((batch_size, self.nmax), dtype=torch.float64)
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            if not self.stationary:
                energy[index] = -self.coord[start:stop].sum()
                force[index, : 3 * len(atoms)] = 1.0
            start = stop
        if self.nonfinite:
            energy[0] = torch.nan
        return energy, force

    def backup_coords(self):
        self._backup = self.coord.clone()

    def restore_coords(self):
        if self._backup is not None:
            self.coord.copy_(self._backup)
            self._backup = None

    def step_cart_(self, step):
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            self.coord[start:stop].add_(
                step[index, : 3 * len(atoms)].reshape(-1, 3)
            )
            start = stop


class _QuadraticBatchCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        batch_size = len(self._atoms)
        energy = torch.zeros(batch_size, dtype=torch.float64)
        force = torch.zeros((batch_size, self.nmax), dtype=torch.float64)
        start = 0
        for index, atoms in enumerate(self._atoms):
            stop = start + len(atoms)
            positions = self.coord[start:stop]
            energy[index] = 500.0 * (positions * positions).sum()
            force[index, : 3 * len(atoms)] = (
                -1000.0 * positions
            ).reshape(-1)
            start = stop
        return energy, force


class _RaisingBatchCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        raise RuntimeError("synthetic backend failure")


class _RaisingNonfiniteCalculator(_SyntheticBatchCalculator):
    def get_ef_gpu(self):
        raise FloatingPointError("synthetic backend non-finite state")


class _TrialNonfiniteCalculator(_SyntheticBatchCalculator):
    def __init__(self):
        super().__init__()
        self.evaluations = 0

    def get_ef_gpu(self):
        self.evaluations += 1
        if self.evaluations > 1:
            raise FloatingPointError("synthetic trial non-finite state")
        return super().get_ef_gpu()


class _MalformedForceWidthCalculator(_SyntheticBatchCalculator):
    def prepare(self, atoms_list, fixed_nmax=None):
        super().prepare(atoms_list, fixed_nmax=fixed_nmax)
        self.nmax = 1
        self.nmax_dof = 1


class _MalformedPointerCalculator(_SyntheticBatchCalculator):
    def prepare(self, atoms_list, fixed_nmax=None):
        super().prepare(atoms_list, fixed_nmax=fixed_nmax)
        self._ptr[-1] -= 1


class _ReorderedCoordinateCalculator(_SyntheticBatchCalculator):
    def prepare(self, atoms_list, fixed_nmax=None):
        super().prepare(atoms_list, fixed_nmax=fixed_nmax)
        self.coord = torch.flip(self.coord, dims=(0,))


class _OOMCalculateMany:
    supports_batch_energy_forces = True
    device = torch.device("cpu")
    optimization_batch_size = "all"

    def __init__(self):
        self.batch_lengths = []

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        atoms_list = list(atoms_list)
        self.batch_lengths.append(len(atoms_list))
        if len(atoms_list) > 2:
            raise RuntimeError("CUDA out of memory")
        return BatchResult(
            energies=np.asarray(
                [atoms.positions[0, 0] for atoms in atoms_list],
                dtype=np.float64,
            ),
            forces=[
                np.full((len(atoms), 3), atoms.positions[0, 0])
                for atoms in atoms_list
            ],
        ).validate_against(atoms_list, properties)


class BatchOptimizerSafetyTests(unittest.TestCase):
    @staticmethod
    def _pdb_atom():
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.info["pdb_template"] = [
            "ATOM      1  H   MOL A   1       0.000   0.000   0.000  "
            "1.00  0.00           H  \n",
            "END\n",
        ]
        return atoms

    def test_constructor_rejects_invalid_parameters(self):
        invalid = (
            {"memory": 0},
            {"memory": True},
            {"curvature": float("nan")},
            {"curvature": 0.0},
            {"maxstep": float("inf")},
            {"maxiter": 0},
            {"traj_every": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BatchLBFGS(output="unused.out", device="cpu", **kwargs)

    def test_run_rejects_invalid_structure_thresholds_before_output(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.f_max_th = float("nan")
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        with self.assertRaisesRegex(ValueError, "f_max_th"):
            optimizer.run(
                _Molecules([atoms], _SyntheticBatchCalculator())
            )
        self.assertIsNone(optimizer.log_fp)

    def test_nonfinite_initial_state_never_writes_production_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(nonfinite=True),
            )

            with self.assertRaisesRegex(
                FloatingPointError, "initial evaluation"
            ):
                optimizer.run(molecules)

            self.assertFalse((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertIsNone(optimizer.log_fp)

    def test_initial_force_width_must_cover_every_cartesian_dof(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            atoms = Atoms(
                "H10",
                positions=np.zeros((10, 3), dtype=np.float64),
            )
            with self.assertRaisesRegex(ValueError, "force width"):
                optimizer.run(
                    _Molecules(
                        [atoms],
                        _MalformedForceWidthCalculator(stationary=True),
                    )
                )
            self.assertEqual(optimizer.statuses, ["failed_protocol"])

    def test_initial_atom_pointer_must_match_packed_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            atoms = Atoms("H2", positions=np.zeros((2, 3), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "pointer topology"):
                optimizer.run(
                    _Molecules(
                        [atoms],
                        _MalformedPointerCalculator(stationary=True),
                    )
                )
            self.assertEqual(optimizer.statuses, ["failed_protocol"])

    def test_initial_coordinate_order_must_match_supplied_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            atoms = Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            )
            with self.assertRaisesRegex(ValueError, "contents/order"):
                optimizer.run(
                    _Molecules(
                        [atoms],
                        _ReorderedCoordinateCalculator(stationary=True),
                    )
                )
            self.assertEqual(optimizer.statuses, ["failed_protocol"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_default_cuda_alias_accepts_current_indexed_device(self):
        optimizer = BatchLBFGS(
            output="unused.out",
            device="cuda",
            maxiter=1,
        )
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        calc = _SyntheticBatchCalculator(stationary=True)
        calc.prepare([atoms])
        calc.coord = calc.coord.to(
            torch.device("cuda", torch.cuda.current_device())
        )

        optimizer._validate_prepared_protocol(
            calc,
            [atoms],
            force_width=calc.nmax_dof,
            stage="CUDA alias regression",
        )

    def test_nonfinite_status_is_limited_to_the_bad_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [
                    Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                    Atoms("H", positions=[[1.0, 0.0, 0.0]]),
                ],
                _SyntheticBatchCalculator(nonfinite=True),
            )
            with self.assertRaises(FloatingPointError):
                optimizer.run(molecules)
            self.assertEqual(
                optimizer.statuses,
                ["failed_nonfinite", "aborted_peer_failure"],
            )
            self.assertEqual(optimizer.failure_details[0]["index"], 0)
            self.assertIn("energies", optimizer.failure_details[0]["fields"])
            self.assertEqual(
                optimizer.failure_details[1]["status"],
                "aborted_peer_failure",
            )

    def test_backend_exception_sets_explicit_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _RaisingBatchCalculator(),
            )
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                optimizer.run(molecules)
            self.assertEqual(optimizer.statuses, ["failed_backend"])
            self.assertEqual(
                optimizer.failure_details[0]["stage"],
                "initial evaluation",
            )

    def test_backend_nonfinite_exception_sets_terminal_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _RaisingNonfiniteCalculator(),
            )
            with self.assertRaisesRegex(
                FloatingPointError, "backend non-finite"
            ):
                optimizer.run(molecules)
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertEqual(
                optimizer.failure_details[0]["fields"],
                ("backend",),
            )

    def test_trial_nonfinite_exception_restores_coordinates_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchLBFGS(
                output=str(Path(tmpdir) / "job.out"),
                device="cpu",
                maxiter=1,
            )
            atoms = Atoms("H", positions=[[0.25, 0.0, 0.0]])
            calc = _TrialNonfiniteCalculator()
            with self.assertRaisesRegex(
                FloatingPointError, "trial non-finite"
            ):
                optimizer.run(_Molecules([atoms], calc))
            np.testing.assert_allclose(
                calc.coord,
                [[0.25, 0.0, 0.0]],
            )
            self.assertEqual(optimizer.statuses, ["failed_nonfinite"])
            self.assertIn(
                "trial 0",
                optimizer.failure_details[0]["stage"],
            )

    def test_non_descent_direction_falls_back_per_structure(self):
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        gradient = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
        proposed = torch.tensor([[2.0, -1.0], [-1.0, -3.0]])
        safeguarded = optimizer._safeguard_search_direction(
            proposed,
            gradient,
        )
        torch.testing.assert_close(
            safeguarded[0],
            -gradient[0] / optimizer.curvature,
        )
        torch.testing.assert_close(safeguarded[1], proposed[1])

    def test_nonfinite_search_direction_fails_instead_of_falling_back(self):
        optimizer = BatchLBFGS(output="unused.out", device="cpu")
        optimizer._orig_index = torch.tensor([0])
        optimizer.statuses = ["running"]
        with self.assertRaisesRegex(
            FloatingPointError, "raw L-BFGS search direction"
        ):
            optimizer._require_finite(
                "raw L-BFGS search direction",
                search_direction=torch.tensor([[float("nan"), 1.0]]),
            )
        self.assertEqual(optimizer.statuses, ["failed_nonfinite"])

    def test_armijo_backtracking_rejects_uphill_clipped_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms("H", positions=[[0.05, 0.0, 0.0]])
            initial_energy = 500.0 * 0.05**2
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                curvature=1.0,
                maxstep=0.2,
                maxiter=1,
            )
            with self.assertRaisesRegex(RuntimeError, "unconverged"):
                optimizer.run(
                    _Molecules([atoms], _QuadraticBatchCalculator())
                )
            final_energy = 500.0 * float(np.sum(atoms.positions**2))
            self.assertLessEqual(final_energy, initial_energy)
            self.assertAlmostEqual(atoms.positions[0, 0], 0.0, places=12)

    def test_maxiter_writes_diagnostic_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(),
            )

            with self.assertRaisesRegex(RuntimeError, "unconverged"):
                optimizer.run(molecules)

            self.assertFalse((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertTrue(
                (Path(tmpdir) / "job_opt_unconverged.xyz").exists()
            )
            self.assertEqual(optimizer.statuses, ["maxiter"])
            self.assertIsNone(optimizer.log_fp)

    def test_converged_run_writes_production_output_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=2,
            )
            molecules = _Molecules(
                [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
                _SyntheticBatchCalculator(stationary=True),
            )

            statuses = optimizer.run(molecules)

            self.assertEqual(statuses, ("converged",))
            self.assertTrue((Path(tmpdir) / "job_opt.xyz").exists())
            self.assertFalse(
                (Path(tmpdir) / "job_opt_unconverged.xyz").exists()
            )

    def test_converged_pdb_run_uses_pdb_production_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=2,
            )
            molecules = _Molecules(
                [self._pdb_atom()],
                _SyntheticBatchCalculator(stationary=True),
            )

            statuses = optimizer.run(molecules)

            self.assertEqual(statuses, ("converged",))
            self.assertTrue((Path(tmpdir) / "job_opt.pdb").exists())
            self.assertFalse((Path(tmpdir) / "job_opt.xyz").exists())

    def test_unconverged_pdb_run_uses_pdb_diagnostic_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            optimizer = BatchLBFGS(
                output=str(output),
                device="cpu",
                maxiter=1,
            )
            molecules = _Molecules(
                [self._pdb_atom()],
                _SyntheticBatchCalculator(),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"no production _opt\.pdb",
            ):
                optimizer.run(molecules)

            self.assertTrue(
                (Path(tmpdir) / "job_opt_unconverged.pdb").exists()
            )
            self.assertFalse(
                (Path(tmpdir) / "job_opt_unconverged.xyz").exists()
            )

    def test_calculate_many_adapter_chunks_and_retries_oom_in_order(self):
        calc = _OOMCalculateMany()
        atoms = [
            Atoms("H", positions=[[float(index), 0.0, 0.0]])
            for index in range(5)
        ]
        adapter = CalculateManyBatchCalc(calc, device="cpu")
        adapter.prepare(atoms)

        energies, forces = adapter.get_ef_gpu()

        self.assertEqual(calc.batch_lengths, [5, 2, 2, 1])
        torch.testing.assert_close(
            energies,
            torch.arange(5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            forces[:, 0],
            torch.arange(5, dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
