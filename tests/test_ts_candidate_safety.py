import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.dispatcher.ts.algorithm.BPRFO import (
    BatchPRFO,
    _regularize_signed,
)
from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFO,
    PRFOConvergenceError,
    PRFOResult,
    PRFOStatus,
    _ts_step_quality,
    prfo_step,
)
from maple.function.dispatcher.ts.algorithm.neb import NEB
from maple.function.dispatcher.ts.algorithm.string import GSM
from maple.function.utility import Molecules


class _AnalyticSaddleCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]
    # This one-atom laboratory-frame quadratic is not a free-molecule PES.
    rigid_body_invariant = False

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        x, y, z = atoms.positions[0]
        energy = -0.5 * x * x + 0.5 * y * y + 0.5 * z * z
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.asarray([[x, -y, -z]], dtype=np.float64),
        }

    def get_hessian(self, atoms):
        return np.diag([-1.0, 1.0, 1.0])


class _AnalyticMinimumCalculator(_AnalyticSaddleCalculator):
    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        self.results = {
            "energy": float(0.5 * np.sum(positions * positions)),
            "free_energy": float(0.5 * np.sum(positions * positions)),
            "forces": -positions,
        }

    def get_hessian(self, atoms):
        return np.eye(3, dtype=np.float64)


class _NonfiniteForceCalculator(_AnalyticSaddleCalculator):
    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.full((len(atoms), 3), np.nan),
        }


class _InconsistentFlatEnergyCalculator(_AnalyticSaddleCalculator):
    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float64),
        }

    def get_hessian(self, atoms):
        return np.eye(3, dtype=np.float64)


class PRFOCandidateSafetyTests(unittest.TestCase):
    def test_scalar_prfo_rejects_non_real_force_thresholds_before_execution(self):
        atoms = Atoms(
            "H", positions=[[0.0, 0.0, 0.0]],
            calculator=_AnalyticSaddleCalculator(),
        )
        for value in (True, np.bool_(True), "0.002"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "f_max_th"):
                    PRFO(
                        output="unused.out", atoms=atoms,
                        paras={"f_max_th": value},
                    )
        PRFO(output="unused.out", atoms=atoms,
             paras={"f_max_th": np.float32(2e-3)})

    def test_constrained_prfo_fails_closed_without_hessian_projection(self):
        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
            calculator=_AnalyticSaddleCalculator(),
        )
        atoms.set_constraint(FixAtoms(indices=[0]))
        with self.assertRaisesRegex(
            NotImplementedError,
            "constrained TS search fails closed",
        ):
            PRFO(output="unused.out", atoms=atoms)

    def test_prfo_rejects_unsafe_negative_quality_threshold(self):
        atoms = Atoms(
            "H",
            positions=[[0.0, 0.0, 0.0]],
            calculator=_AnalyticSaddleCalculator(),
        )
        with self.assertRaisesRegex(ValueError, "0 <= eta_reject"):
            PRFO(
                output="unused.out",
                atoms=atoms,
                paras={"eta_reject": -1.0},
            )

    def test_prfo_ascends_positive_curvature_target_mode(self):
        gradient = np.asarray([0.05], dtype=np.float64)
        step = prfo_step(
            np.asarray([[1.0]], dtype=np.float64),
            gradient,
            is_ts=True,
            target_mode=0,
            trust_radius=0.2,
        )

        self.assertGreater(float(gradient @ step), 0.0)
        self.assertLessEqual(float(np.linalg.norm(step)), 0.2 + 1.0e-10)

    def test_prfo_maximizes_target_and_minimizes_complement(self):
        gradient = np.asarray([0.05, 0.10], dtype=np.float64)
        step = prfo_step(
            np.diag([1.0, 2.0]),
            gradient,
            is_ts=True,
            target_mode=0,
            trust_radius=0.2,
        )

        self.assertGreater(float(gradient[0] * step[0]), 0.0)
        self.assertLess(float(gradient[1] * step[1]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(step)), 0.2 + 1.0e-10)

    def test_mode_tracking_starts_from_lowest_curvature_not_largest_gradient(self):
        optimizer = PRFO.__new__(PRFO)
        optimizer.tracked_mode_vec_mw = None
        optimizer.tracked_mode_idx = None
        optimizer.params = SimpleNamespace(inertia_threshold=1.0e-6)

        index = optimizer.update_mode_tracking(
            np.asarray([0.0, 1.0, 2.0]),
            np.eye(3),
            np.asarray([100.0, 0.01, 10.0]),
        )

        self.assertEqual(index, 1)

    def test_mode_tracking_never_switches_to_a_near_zero_mode(self):
        optimizer = PRFO.__new__(PRFO)
        optimizer.tracked_mode_vec_mw = np.asarray([1.0, 0.0])
        optimizer.tracked_mode_idx = 0
        optimizer.params = SimpleNamespace(inertia_threshold=1.0e-6)

        index = optimizer.update_mode_tracking(
            np.asarray([0.0, 1.0]),
            np.eye(2),
            np.zeros(2),
        )

        self.assertEqual(index, 1)

    def test_ts_trust_quality_penalizes_large_overprediction_symmetrically(self):
        rho, quality = _ts_step_quality(
            actual_change=10.0,
            predicted_change=1.0,
        )
        self.assertEqual(rho, 10.0)
        self.assertEqual(quality, -8.0)

    def test_exact_saddle_converges_before_zero_step_trial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "exact-saddle.out"
            atoms = Atoms(
                "H",
                positions=[[0.0, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            result = PRFO(
                output=str(output),
                atoms=atoms,
                paras={"max_iter": 0},
            ).run_result()

            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertEqual(result.iterations, 0)
            self.assertEqual(result.negative_modes, 1)
            self.assertTrue(
                (Path(tmpdir) / "exact-saddle_prfo_ts_candidate.xyz").exists()
            )

    def test_last_allowed_step_is_re_evaluated_for_convergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            result = PRFO(
                output=str(Path(tmpdir) / "last-step.out"),
                atoms=atoms,
                paras={"max_iter": 1},
            ).run_result()

            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertEqual(result.iterations, 1)
            self.assertEqual(result.negative_modes, 1)

    def test_exact_minimum_fails_wrong_inertia(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "exact-minimum.out"
            atoms = Atoms(
                "H",
                positions=[[0.0, 0.0, 0.0]],
                calculator=_AnalyticMinimumCalculator(),
            )
            result = PRFO(
                output=str(output),
                atoms=atoms,
                paras={"max_iter": 2},
            ).run_result()

            self.assertEqual(result.status, PRFOStatus.FAILED_WRONG_INERTIA)
            self.assertEqual(result.iterations, 0)
            self.assertEqual(result.negative_modes, 0)
            self.assertFalse(
                (Path(tmpdir) / "exact-minimum_prfo_ts_candidate.xyz").exists()
            )

    def test_first_accepted_step_updates_hessian_when_recalc_is_delayed(self):
        module = import_module(
            "maple.function.dispatcher.ts.algorithm.PRFO"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            with patch.object(
                module,
                "_bofill_update",
                wraps=module._bofill_update,
            ) as update:
                result = PRFO(
                    output=str(Path(tmpdir) / "secant.out"),
                    atoms=atoms,
                    paras={"max_iter": 4, "recalc": 2},
                ).run_result()

            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertEqual(result.iterations, 1)
            self.assertGreaterEqual(update.call_count, 1)

    def test_pdb_template_fails_closed_without_shared_writer(self):
        atoms = Atoms(
            "H",
            positions=[[0.0, 0.0, 0.0]],
            calculator=_AnalyticSaddleCalculator(),
        )
        atoms.info["pdb_template"] = [
            "ATOM      1  H   MOL A   1       0.000   0.000   0.000"
            "  1.00  0.00           H"
        ]
        try:
            import_module("maple.function.read.filereader.pdb_reader")
        except ImportError:
            with self.assertRaisesRegex(
                NotImplementedError,
                "shared format-aware writer",
            ):
                PRFO(output="unused.out", atoms=atoms, paras={"max_iter": 0})
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = PRFO(
                    output=str(Path(tmpdir) / "pdb.out"),
                    atoms=atoms,
                    paras={"max_iter": 1},
                ).run_result()
                self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
                self.assertTrue(result.structure_path.endswith(".pdb"))
                content = Path(result.structure_path).read_text()
                self.assertTrue(content.startswith("ATOM"))
                self.assertTrue(content.endswith("END\n"))

    def test_nonfinite_force_has_explicit_terminal_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nonfinite.out"
            atoms = Atoms(
                "H",
                positions=[[0.0, 0.0, 0.0]],
                calculator=_NonfiniteForceCalculator(),
            )
            result = PRFO(
                output=str(output),
                atoms=atoms,
                paras={"max_iter": 1},
            ).run_result()

            self.assertEqual(result.status, PRFOStatus.FAILED_NONFINITE)
            self.assertTrue(
                (Path(tmpdir) / "nonfinite_prfo_nonfinite.xyz").exists()
            )
            self.assertFalse(
                (Path(tmpdir) / "nonfinite_prfo_ts_candidate.xyz").exists()
            )

    def test_zero_quality_trial_is_rejected_and_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial = np.asarray([[0.0, 0.0, 0.0]])
            atoms = Atoms(
                "H",
                positions=initial.copy(),
                calculator=_InconsistentFlatEnergyCalculator(),
            )
            result = PRFO(
                output=str(Path(tmpdir) / "zero-quality.out"),
                atoms=atoms,
                paras={"max_iter": 1},
            ).run_result()

            self.assertEqual(result.status, PRFOStatus.FAILED_STAGNATION)
            np.testing.assert_allclose(atoms.positions, initial)

    def test_geometry_convergence_writes_candidate_not_certified_ts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            prfo = PRFO(
                output=str(output),
                atoms=atoms,
                paras={
                    "max_iter": 2,
                    "f_max_th": 0.1,
                    "f_rms_th": 0.1,
                    "dp_max_th": 0.2,
                    "dp_rms_th": 0.2,
                },
            )

            result = prfo.run_result()

            self.assertEqual(result.status, PRFOStatus.GEOMETRY_CONVERGED)
            self.assertTrue(result.geometry_converged)
            self.assertTrue(
                (Path(tmpdir) / "job_prfo_ts_candidate.xyz").exists()
            )
            self.assertFalse((Path(tmpdir) / "job_prfo_ts.xyz").exists())
            self.assertIn(
                "not a frequency/IRC-verified transition state",
                output.read_text(),
            )

    def test_maxiter_is_structured_failure_and_run_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "job.out"
            atoms = Atoms(
                "H",
                positions=[[0.05, 0.0, 0.0]],
                calculator=_AnalyticSaddleCalculator(),
            )
            prfo = PRFO(
                output=str(output),
                atoms=atoms,
                paras={"max_iter": 0},
            )

            result = prfo.run_result()

            self.assertEqual(result.status, PRFOStatus.FAILED_MAXITER)
            self.assertFalse(result.geometry_converged)
            self.assertTrue(
                (Path(tmpdir) / "job_prfo_unconverged.xyz").exists()
            )
            self.assertFalse((Path(tmpdir) / "job_prfo_ts.xyz").exists())
            self.assertFalse(
                (Path(tmpdir) / "job_prfo_ts_candidate.xyz").exists()
            )

            retry = PRFO(
                output=str(Path(tmpdir) / "retry.out"),
                atoms=atoms,
                paras={"max_iter": 0},
            )
            with self.assertRaises(PRFOConvergenceError):
                retry.run()


class TSCallerSafetyTests(unittest.TestCase):
    @staticmethod
    def _failed_prfo_result(atoms, path):
        return PRFOResult(
            atoms=atoms,
            status=PRFOStatus.FAILED_MAXITER,
            iterations=0,
            structure_path=str(path),
        )

    def test_string_rejects_unconverged_prfo_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "string.out"
            images = [
                Atoms(
                    "H", positions=[[float(i), 0.0, 0.0]],
                    calculator=_AnalyticMinimumCalculator(),
                )
                for i in range(3)
            ]
            gsm = GSM.__new__(GSM)
            gsm.output = str(output)
            gsm.raw_paras = {}
            failed = self._failed_prfo_result(
                images[1],
                Path(tmpdir) / "string_prfo_unconverged.xyz",
            )

            with patch.object(PRFO, "run_result", return_value=failed):
                with self.assertRaises(PRFOConvergenceError):
                    gsm.restart_run(images, 1, str(Path(tmpdir) / "string"))

            self.assertFalse(
                (Path(tmpdir) / "string_stringts_ts_candidate.xyz").exists()
            )

    def test_string_reports_real_candidate_global_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "string.out"
            images = [
                Atoms(
                    "H",
                    positions=[[float(i), 0.0, 0.0]],
                    calculator=_AnalyticMinimumCalculator(),
                )
                for i in range(3)
            ]
            gsm = GSM.__new__(GSM)
            gsm.output = str(output)
            gsm.raw_paras = {}
            converged = PRFOResult(
                atoms=images[1],
                status=PRFOStatus.GEOMETRY_CONVERGED,
                iterations=1,
                structure_path=str(
                    Path(tmpdir) / "string_prfo_ts_candidate.xyz"
                ),
                negative_modes=1,
            )
            energies = [0.0, 0.5, 0.5, 0.0]
            forces = [
                np.zeros((1, 3)),
                np.zeros((1, 3)),
                np.asarray([[-2.0, 0.0, 0.0]]),
                np.zeros((1, 3)),
            ]

            with (
                patch.object(PRFO, "run_result", return_value=converged),
                patch(
                    "maple.function.dispatcher.ts.algorithm.string.get_energy_forces",
                    return_value=(energies, forces),
                ),
                patch(
                    "maple.function.dispatcher.ts.algorithm.string.write_xyz"
                ),
            ):
                gsm.restart_run(
                    images,
                    1,
                    str(Path(tmpdir) / "string"),
                )

            report = output.read_text()
            self.assertIn("candidate-global", report)
            self.assertIn("<= HEI", report)
            self.assertNotIn("<= CI", report)
            candidate_line = next(
                line for line in report.splitlines()
                if line.startswith("CAND")
            )
            self.assertIn("2.00000", candidate_line)

    def test_neb_rejects_unconverged_prfo_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "neb.out"
            images = [
                Atoms(
                    "H", positions=[[float(i), 0.0, 0.0]],
                    calculator=_AnalyticMinimumCalculator(),
                )
                for i in range(3)
            ]
            for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
                setattr(images[0], name, 1.0)

            neb = NEB.__new__(NEB)
            neb.output = str(output)
            neb.atoms_R = images[0]
            neb.raw_paras = {"prfo": {"rigid_symmetry": "cartesian_external"}}
            neb.params = SimpleNamespace(
                lbfgs_m=2,
                cistep0=0.01,
                neb_f_max_th=1.0,
                neb_f_rms_th=1.0,
                cineb_f_max_th=1.0,
                cineb_f_rms_th=1.0,
                k_max=0.3,
                max_iter=1,
                refine="nebts",
            )
            neb._pack_internal = lambda current: np.zeros(3)
            neb._unpack_internal = lambda values, current: None
            neb._path_energy_forces = lambda current: (
                [0.0, 1.0, 0.0],
                [np.zeros((1, 3)) for _ in current],
            )
            neb.cineb_forces = lambda current, energies, spring, raw_forces: (
                [np.zeros((1, 3)) for _ in current],
                0.0,
                1,
            )
            neb.atoms_to_xyz = lambda atoms: ""
            failed = self._failed_prfo_result(
                images[1],
                Path(tmpdir) / "neb_prfo_unconverged.xyz",
            )

            with (
                patch.object(PRFO, "__init__", return_value=None) as construct,
                patch.object(PRFO, "run_result", return_value=failed),
            ):
                with self.assertRaises(PRFOConvergenceError):
                    neb.restart_run(images, energies=[0.0, 1.0, 0.0])
            self.assertIs(construct.call_args.kwargs["paras"], neb.raw_paras)

            self.assertFalse(
                (Path(tmpdir) / "neb_nebts_ts_candidate.xyz").exists()
            )

    def test_neb_labels_projected_ci_and_candidate_forces_separately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "neb.out"
            images = [
                Atoms(
                    "H",
                    positions=[[float(i), 0.0, 0.0]],
                    calculator=_AnalyticMinimumCalculator(),
                )
                for i in range(3)
            ]
            for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"):
                setattr(images[0], name, 1.0)

            candidate = Atoms(
                "H",
                positions=[[2.0, 0.0, 0.0]],
                calculator=_AnalyticMinimumCalculator(),
            )
            converged = PRFOResult(
                atoms=candidate,
                status=PRFOStatus.GEOMETRY_CONVERGED,
                iterations=1,
                structure_path=str(
                    Path(tmpdir) / "neb_prfo_ts_candidate.xyz"
                ),
                negative_modes=1,
            )
            raw_forces = [
                np.zeros((1, 3)),
                np.asarray([[3.0, 4.0, 0.0]]),
                np.zeros((1, 3)),
            ]
            projected = [
                np.zeros((1, 3)),
                np.asarray([[1.0, 0.0, 0.0]]),
                np.zeros((1, 3)),
            ]

            neb = NEB.__new__(NEB)
            neb.output = str(output)
            neb.atoms_R = images[0]
            neb.raw_paras = {"prfo": {"rigid_symmetry": "cartesian_external"}}
            neb.params = SimpleNamespace(
                lbfgs_m=2,
                cistep0=0.01,
                neb_f_max_th=10.0,
                neb_f_rms_th=10.0,
                cineb_f_max_th=10.0,
                cineb_f_rms_th=10.0,
                k_max=0.3,
                max_iter=1,
                refine="nebts",
            )
            neb._pack_internal = lambda current: np.zeros(3)
            neb._unpack_internal = lambda values, current: None
            neb._path_energy_forces = lambda current: (
                [0.0, 1.0, 0.0],
                raw_forces,
            )
            neb.cineb_forces = lambda current, energies, spring, raw_forces: (
                projected,
                1.0,
                1,
            )
            neb.atoms_to_xyz = lambda atoms: ""

            with (
                patch.object(PRFO, "run_result", return_value=converged),
                patch(
                    "maple.function.dispatcher.ts.algorithm.neb.write_xyz"
                ),
            ):
                neb.restart_run(images, energies=[0.0, 1.0, 0.0])

            report = output.read_text()
            self.assertIn("projected", report)
            self.assertIn("CI-global", report)
            self.assertIn("candidate-global", report)
            ci_line = next(
                line for line in report.splitlines()
                if "CI-global" in line
            )
            candidate_line = next(
                line for line in report.splitlines()
                if "candidate-global" in line
            )
            self.assertIn("5.00000", ci_line)
            self.assertIn("2.00000", candidate_line)

    def test_neb_and_string_fail_closed_for_pdb_without_shared_writer(self):
        reactant = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        product = Atoms("H", positions=[[1.0, 0.0, 0.0]])
        reactant.info["pdb_template"] = object()
        constructors = (
            lambda: NEB(
                output="unused-neb.out",
                atoms_or_molecules=Molecules([reactant, product]),
            ),
            lambda: GSM(
                output="unused-string.out",
                atoms_R=reactant,
                atoms_P=product,
            ),
        )
        try:
            import_module("maple.function.read.filereader.pdb_reader")
        except ImportError:
            for construct in constructors:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "shared format-aware writer",
                ):
                    construct()
        else:
            for construct in constructors:
                construct()


class BatchPRFOIsolationTests(unittest.TestCase):
    def test_exact_zero_regularization_is_nonzero(self):
        values = torch.tensor(
            [-1.0e-20, 0.0, 1.0e-20, 2.0],
            dtype=torch.float64,
        )
        actual = _regularize_signed(values, 1.0e-10)
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [-1.0e-10, 1.0e-10, 1.0e-10, 2.0],
                dtype=torch.float64,
            ),
        )
        self.assertTrue(torch.all(actual != 0.0))

    def test_batch_prfo_is_runtime_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = BatchPRFO(
                output=str(Path(tmpdir) / "batch-prfo.out"),
                device="cpu",
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "experimental and runtime-disabled",
            ):
                optimizer.run(object())


if __name__ == "__main__":
    unittest.main()
