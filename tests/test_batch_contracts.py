import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from maple.function.calculator._batch_eval import (
    ALL_BATCH_SIZE,
    AUTO_BATCH_SIZE,
    EnergyEvaluator,
    FDHessianEvaluator,
    HVPEvaluator,
    PathEvaluator,
    _copy_with_positions,
    _estimate_auto_batch_size_from_item_bytes,
)
from maple.function.calculator._batch_types import BatchResult
from maple.function.calculator._batch_utils import (
    normalize_energy_forces_request,
    sequential_calculate_many,
    split_atomwise_array,
)
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_BATCH_LAYOUT_BY_SHA256,
    AIMNET2_CHECKPOINT_CAPABILITIES_BY_SHA256,
    AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
    AIMNet2Calculator,
    build_aimnet2_neighbor_matrices,
    identify_aimnet2_checkpoint_capabilities,
    identify_aimnet2_batch_layout,
    nblist_all_pairs_padded_multi,
)
from maple.function.calculator.aimnet._aimnet2_batch_calculator import (
    AIMNet2BatchCalc,
)
from maple.function.calculator.ani._ani_calculator import ANICalculator


def _require_or_skip_real_checkpoints(test_case, *paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if not missing:
        return
    message = "required real checkpoints are unavailable: " + ", ".join(missing)
    required = os.environ.get(
        "MAPLE_REQUIRE_REAL_CHECKPOINTS",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if required:
        test_case.fail(message)
    test_case.skipTest(message)


class _CacheCalculator:
    def __init__(self, fail_at=None):
        self.results = {
            "energy": 7.0,
            "free_energy": 7.0,
            "forces": np.full((1, 3), 4.0),
        }
        self.atoms = Atoms("He", positions=[[9.0, 8.0, 7.0]])
        self.fail_at = fail_at
        self.calls = 0

    def calculate(self, atoms, properties, system_changes):
        self.calls += 1
        self.atoms = atoms.copy()
        self.results = {
            "energy": float(self.calls),
            "free_energy": float(self.calls),
            "forces": np.full((len(atoms), 3), float(self.calls)),
        }
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic backend failure")


class _QuadraticCalculator:
    supports_hvp = False

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        energies = []
        forces = []
        for atoms in atoms_list:
            positions = atoms.get_positions()
            energies.append(0.5 * float(np.sum(positions**2)))
            forces.append(-positions)
        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64),
            forces=forces,
        )


class _BadAnalyticHVP:
    supports_hvp = True

    def get_hvp(self, atoms, n):
        size = 3 * len(atoms)
        return (
            torch.full((size,), torch.nan),
            torch.zeros(size),
            torch.tensor(0.0),
        )


class _ShortANIModel:
    def __call__(self, species, coordinates):
        return (torch.zeros(1, dtype=coordinates.dtype, device=coordinates.device),)


class _NaNANIModel:
    def __call__(self, species, coordinates):
        energy = coordinates.sum(dim=(1, 2)) * float("nan")
        return (energy,)


class _RecordingAIMNetModel:
    def __init__(self):
        self.coord_dtype = None

    def __call__(self, data):
        self.coord_dtype = data["coord"].dtype
        return {
            "energy": torch.zeros(
                data["charge"].shape[0],
                dtype=data["coord"].dtype,
                device=data["coord"].device,
            )
        }


class BatchResultContractTests(unittest.TestCase):
    def test_required_real_checkpoint_gate_fails_instead_of_skipping(self):
        missing = Path("/definitely/missing/maple-checkpoint.pt")
        with (
            patch.dict(
                os.environ,
                {"MAPLE_REQUIRE_REAL_CHECKPOINTS": "1"},
            ),
            self.assertRaisesRegex(AssertionError, "required real checkpoints"),
        ):
            _require_or_skip_real_checkpoints(self, missing)

    def test_rejects_nonfinite_numeric_fields(self):
        with self.assertRaisesRegex(ValueError, "energies.*finite"):
            BatchResult(energies=np.array([np.nan]))
        with self.assertRaisesRegex(ValueError, r"forces\[0\].*finite"):
            BatchResult(forces=[np.array([[0.0, np.inf, 0.0]])])
        with self.assertRaisesRegex(ValueError, r"hessians\[0\].*finite"):
            BatchResult(hessians=[np.array([[np.nan]])])

    def test_rejects_fractional_or_negative_padding(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            BatchResult(padding_counts=np.array([1.5]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BatchResult(padding_counts=np.array([-1]))

    def test_validate_against_checks_requested_fields_and_atom_counts(self):
        atoms = [Atoms("H"), Atoms("H2")]
        result = BatchResult(
            energies=np.array([1.0, 2.0]),
            forces=[np.zeros((1, 3)), np.zeros((1, 3))],
        )
        with self.assertRaisesRegex(ValueError, r"forces\[1\].*expected"):
            result.validate_against(atoms, ("energy", "forces"))

        with self.assertRaisesRegex(ValueError, "requested.*forces"):
            BatchResult(energies=np.array([1.0, 2.0])).validate_against(
                atoms, ("energy", "forces")
            )

    def test_nested_arrays_are_read_only_and_revalidated(self):
        atoms = [Atoms("H")]
        result = BatchResult(
            energies=np.array([1.0]),
            forces=[np.zeros((1, 3))],
        )
        self.assertIsInstance(result.forces, tuple)
        with self.assertRaises(ValueError):
            result.energies[0] = np.nan
        with self.assertRaises(ValueError):
            result.forces[0][0, 0] = np.inf

        result.energies.setflags(write=True)
        result.energies[0] = np.nan
        with self.assertRaisesRegex(ValueError, "energies.*finite"):
            result.validate_against(atoms, ("energy", "forces"))


class BatchUtilityContractTests(unittest.TestCase):
    def test_unknown_properties_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported calculate_many"):
            normalize_energy_forces_request(("energy", "stress"))

    def test_split_atomwise_array_rejects_wrong_shape_or_incomplete_counts(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            split_atomwise_array(np.zeros((2, 2)), [1, 1])
        with self.assertRaisesRegex(ValueError, "sum"):
            split_atomwise_array(np.zeros((3, 3)), [1, 1])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            split_atomwise_array(np.zeros((2, 3)), [1.5, 0.5])

    def test_sequential_fallback_restores_cache_after_success(self):
        calc = _CacheCalculator()
        old_results = calc.results
        old_atoms = calc.atoms
        old_results_copy = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in old_results.items()
        }

        result = sequential_calculate_many(
            calc,
            [Atoms("H"), Atoms("H")],
            ("energy", "forces"),
            True,
            True,
        )

        np.testing.assert_allclose(result.energies, [1.0, 2.0])
        self.assertIs(calc.results, old_results)
        self.assertIs(calc.atoms, old_atoms)
        for key, value in old_results_copy.items():
            np.testing.assert_allclose(calc.results[key], value)

    def test_sequential_fallback_restores_cache_after_exception(self):
        calc = _CacheCalculator(fail_at=2)
        old_results = calc.results
        old_atoms = calc.atoms

        with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
            sequential_calculate_many(
                calc,
                [Atoms("H"), Atoms("H")],
                ("energy", "forces"),
                True,
                True,
            )

        self.assertIs(calc.results, old_results)
        self.assertIs(calc.atoms, old_atoms)
        self.assertEqual(calc.results["energy"], 7.0)

    def test_displaced_atoms_do_not_share_constraint_objects(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
        atoms.set_constraint(FixAtoms(indices=[0]))
        displaced = _copy_with_positions(
            atoms,
            atoms.get_positions() + 0.01,
            apply_constraints=False,
        )
        self.assertIsNot(displaced.constraints[0], atoms.constraints[0])


class BackendContractTests(unittest.TestCase):
    def test_aimnet_batch_schema_is_bound_to_checkpoint_identity(self):
        self.assertEqual(
            set(AIMNET2_BATCH_LAYOUT_BY_SHA256.values()),
            {AIMNET2_PADDED_PER_MOLECULE_LAYOUT},
        )
        self.assertEqual(len(AIMNET2_BATCH_LAYOUT_BY_SHA256), 2)
        with tempfile.NamedTemporaryFile() as custom:
            custom.write(b"unrecognized checkpoint")
            custom.flush()
            self.assertIsNone(
                identify_aimnet2_batch_layout(custom.name)
            )

    def test_aimnet_checkpoint_manifest_distinguishes_nse(self):
        capabilities = list(
            AIMNET2_CHECKPOINT_CAPABILITIES_BY_SHA256.values()
        )
        self.assertEqual(
            sorted(item.num_charge_channels for item in capabilities),
            [1, 2],
        )
        self.assertTrue(all(item.input_dtype == torch.float32 for item in capabilities))
        self.assertEqual(
            sum(item.supports_multiplicity for item in capabilities),
            1,
        )

    def test_aimnet_direct_batch_uses_manifest_input_dtype(self):
        calc = AIMNet2Calculator.__new__(AIMNet2Calculator)
        calc.device = torch.device("cpu")
        calc.input_dtype = torch.float64
        calc.cutoff = 5.0
        calc.cutoff_lr = 5.0
        calc.solvent_correction = None
        calc.supports_batch_energy_forces = True
        calc.supports_multiplicity = False
        calc._batch_energy_layout = AIMNET2_PADDED_PER_MOLECULE_LAYOUT
        calc.model = _RecordingAIMNetModel()

        result = calc.calculate_many(
            [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
            properties=("energy",),
        )

        self.assertEqual(calc.model.coord_dtype, torch.float64)
        np.testing.assert_allclose(result.energies, [0.0])

    def test_aimnet_simple_neighbor_list_is_all_pairs_per_molecule(self):
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 16.0, 0.0],
            ],
            dtype=torch.float32,
        )
        mol_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
        short, long_range = build_aimnet2_neighbor_matrices(
            coord,
            mol_idx,
            cutoff=5.0,
            cutoff_lr=float("inf"),
        )
        sentinel = len(coord)

        self.assertTrue(torch.all(short[:2] == sentinel))
        self.assertEqual(
            {
                int(value)
                for value in long_range[0].tolist()
                if value != sentinel
            },
            {1},
        )
        self.assertEqual(
            {
                int(value)
                for value in long_range[2].tolist()
                if value != sentinel
            },
            {3, 4},
        )
        for atom_index, row in enumerate(long_range[:-1]):
            self.assertNotIn(atom_index, row.tolist())
        self.assertTrue(torch.all(long_range[-1] == sentinel))
        torch.testing.assert_close(
            long_range,
            nblist_all_pairs_padded_multi(mol_idx),
        )

    @staticmethod
    def _packaged_aimnet_path(model_name):
        release_dir = os.environ.get("MAPLE_RELEASE_CHECKPOINT_DIR")
        if release_dir:
            return Path(release_dir) / f"{model_name}.pt"
        return (
            Path(__file__).parents[1]
            / "maple"
            / "function"
            / "calculator"
            / "model"
            / f"{model_name}.pt"
        )

    def test_aimnet_real_checkpoint_long_range_and_batch_parity(self):
        model_path = self._packaged_aimnet_path("aimnet2")
        _require_or_skip_real_checkpoints(self, model_path)
        calc = AIMNet2Calculator(
            device=torch.device("cpu"),
            model="aimnet2",
            model_path=str(model_path),
            coulomb_method="simple",
        )
        atoms = Atoms(
            "OH",
            positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=calc.input_dtype,
            requires_grad=True,
        )
        data = calc._build_data(coord, atoms)
        sentinel = len(atoms)
        self.assertTrue(torch.all(data["nbmat"][:-1] == sentinel))
        self.assertEqual(data["nbmat_lr"][0, 0].item(), 1)
        self.assertEqual(data["nbmat_lr"][1, 0].item(), 0)
        self.assertTrue(torch.isinf(data["cutoff_lr"]))

        with torch.jit.optimized_execution(False):
            all_pair_energy = calc.model(data)["energy"].sum()

        truncated_data = calc._build_data(
            torch.tensor(
                atoms.get_positions(),
                dtype=calc.input_dtype,
                requires_grad=True,
            ),
            atoms,
        )
        mol_idx = torch.zeros(len(atoms), dtype=torch.int32)
        _, truncated_lr = build_aimnet2_neighbor_matrices(
            truncated_data["coord"][:-1],
            mol_idx,
            cutoff=calc.cutoff,
            cutoff_lr=calc.cutoff,
        )
        truncated_data["nbmat_lr"] = truncated_lr
        truncated_data["cutoff_lr"] = torch.tensor(
            calc.cutoff,
            dtype=calc.input_dtype,
        )
        with torch.jit.optimized_execution(False):
            truncated_energy = calc.model(truncated_data)["energy"].sum()
        self.assertGreater(
            abs(float((all_pair_energy - truncated_energy).detach())),
            1.0e-4,
        )

        calc.calculate(atoms, properties=("energy", "forces"))
        sequential_energy = calc.results["energy"]
        sequential_forces = calc.results["forces"].copy()
        batched = calc.calculate_many([atoms], properties=("energy", "forces"))
        np.testing.assert_allclose(
            batched.energies,
            [sequential_energy],
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            batched.forces[0],
            sequential_forces,
            rtol=0.0,
            atol=2.0e-7,
        )

    def test_aimnet_batch_adapter_inherits_dtype_and_passes_nse_mult(self):
        standard_path = self._packaged_aimnet_path("aimnet2")
        nse_path = self._packaged_aimnet_path("aimnet2nse")
        _require_or_skip_real_checkpoints(self, standard_path, nse_path)

        standard = AIMNet2Calculator(
            device=torch.device("cpu"),
            model="aimnet2",
            model_path=str(standard_path),
        )
        open_shell = Atoms(
            "O",
            positions=[[0.0, 0.0, 0.0]],
            info={"charge": 0, "mult": 3},
        )
        with self.assertRaisesRegex(NotImplementedError, "closed-shell"):
            standard.calculate(open_shell, properties=("energy",))
        with self.assertRaisesRegex(NotImplementedError, "closed-shell"):
            AIMNet2BatchCalc.from_ase_calculator(standard).prepare([open_shell])

        nse = AIMNet2Calculator(
            device=torch.device("cpu"),
            model="aimnet2nse",
            model_path=str(nse_path),
        )
        capabilities = identify_aimnet2_checkpoint_capabilities(str(nse_path))
        self.assertIsNotNone(capabilities)
        self.assertTrue(capabilities.supports_multiplicity)
        adapter = AIMNet2BatchCalc.from_ase_calculator(nse)
        adapter.prepare([open_shell])
        self.assertEqual(adapter.dtype, torch.float32)
        torch.testing.assert_close(
            adapter.mult,
            torch.tensor([3.0, 1.0]),
        )
        energies, forces = adapter.get_ef_gpu()
        self.assertTrue(torch.isfinite(energies).all())
        self.assertTrue(torch.isfinite(forces).all())

    def test_ani_rejects_short_energy_vector(self):
        calc = ANICalculator.__new__(ANICalculator)
        calc.device = torch.device("cpu")
        calc.dtype = torch.float64
        calc.d4 = False
        calc.solvent_correction = None
        calc.model = _ShortANIModel()
        atoms = [
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        ]
        with self.assertRaisesRegex(RuntimeError, "ANI energy output"):
            calc.calculate_many(atoms, properties=("energy",))

    def test_aimnet_requires_padded_per_molecule_energy_layout(self):
        output = torch.tensor([1.0, 2.0, 0.0])
        actual = AIMNet2Calculator._energy_vector_from_output(
            output,
            batch_size=2,
            layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
        )
        torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]))

        with self.assertRaisesRegex(RuntimeError, "padded per-molecule"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.tensor([1.0, 2.0]),
                batch_size=2,
                layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
            )
        with self.assertRaisesRegex(RuntimeError, "padded per-molecule"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.arange(5.0),
                batch_size=2,
                layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
            )
        with self.assertRaisesRegex(RuntimeError, "validated"):
            AIMNet2Calculator._energy_vector_from_output(
                torch.tensor([1.0, 2.0, 3.0]),
                batch_size=2,
                layout=None,
            )

    def test_ani_direct_hvp_rejects_nonfinite_outputs(self):
        calc = ANICalculator.__new__(ANICalculator)
        calc.device = torch.device("cpu")
        calc.dtype = torch.float64
        calc.d4 = False
        calc.solvent_correction = None
        calc.model = _NaNANIModel()
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

        with self.assertRaisesRegex(FloatingPointError, "finite scalar energy"):
            calc.get_hvp(atoms, np.ones(3))
        with self.assertRaisesRegex(ValueError, "non-zero"):
            calc.get_hvp(atoms, np.zeros(3))


class EvaluatorSafetyTests(unittest.TestCase):
    def test_unset_evaluator_batch_sizes_default_to_auto(self):
        calc = _QuadraticCalculator()
        self.assertEqual(
            FDHessianEvaluator(calc).fd_batch_size,
            AUTO_BATCH_SIZE,
        )
        self.assertEqual(
            PathEvaluator(calc).batch_size,
            AUTO_BATCH_SIZE,
        )
        self.assertEqual(
            EnergyEvaluator(calc).batch_size,
            AUTO_BATCH_SIZE,
        )
        self.assertEqual(
            HVPEvaluator(calc).batch_size,
            AUTO_BATCH_SIZE,
        )
        self.assertEqual(
            PathEvaluator(calc, batch_size="all").batch_size,
            ALL_BATCH_SIZE,
        )

    def test_auto_batch_sizing_never_rounds_above_memory_budget(self):
        self.assertEqual(
            _estimate_auto_batch_size_from_item_bytes(
                n_total=10,
                item_bytes=1,
                free_bytes=12,
                target_fraction=0.75,
            ),
            9,
        )
        self.assertEqual(
            _estimate_auto_batch_size_from_item_bytes(
                n_total=10,
                item_bytes=0,
                free_bytes=12,
                target_fraction=0.75,
            ),
            1,
        )

    def test_fd_hessian_rejects_nonfinite_delta_and_forces(self):
        calc = _QuadraticCalculator()
        atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            FDHessianEvaluator(calc).hessian(atoms, delta=np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            FDHessianEvaluator._validated_force(
                np.array([[np.nan, 0.0, 0.0]]), 1, 0
            )

    def test_fd_hessian_large_antisymmetry_fails_closed(self):
        evaluator = FDHessianEvaluator(_QuadraticCalculator())
        hessian = np.array([[0.0, 1.0], [0.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "antisymmetric residual"):
            evaluator._symmetrize(hessian)

    def test_hvp_validates_direction_and_backend_outputs(self):
        atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.zeros(2), delta=0.005
            )
        with self.assertRaisesRegex(ValueError, "non-zero"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.zeros(3), delta=0.005
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            HVPEvaluator(_QuadraticCalculator()).hn(
                atoms, np.ones(3), delta=np.nan
            )
        with self.assertRaisesRegex(FloatingPointError, "HVP"):
            HVPEvaluator(_BadAnalyticHVP()).hn(
                atoms, np.ones(3), delta=0.005
            )


if __name__ == "__main__":
    unittest.main()
