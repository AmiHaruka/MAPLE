import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

from maple.function.calculator._batch_eval import (
    ALL_BATCH_SIZE,
    AUTO_BATCH_SIZE,
    EnergyEvaluator,
    FDHessianEvaluator,
    HVPEvaluator,
    PathEvaluator,
    energy_forces_one,
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
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.dispatcher.ts.algorithm.dimer import Dimer


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


class _DistinctEnergyCalculator(Calculator):
    """Synthetic ASE calculator whose potential and free energies differ."""

    energy_free_energy_equal = False
    implemented_properties = ("energy", "free_energy", "forces")
    supports_batch_energy_forces = True

    def __init__(self, *, native=False, pbc=False):
        super().__init__()
        self.native = native
        self.pbc = pbc
        self.requests = []

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.requests.append(tuple(properties))
        x = float(np.sum(atoms.get_positions()))
        self.results = {"forces": np.full((len(atoms), 3), -x)}
        if "energy" in properties:
            self.results["energy"] = 10.0 + x
        if "free_energy" in properties:
            self.results["free_energy"] = 20.0 + x

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        if not self.native:
            raise AssertionError("native path was not expected")
        props, want_energy, want_forces, request = normalize_energy_forces_request(
            properties
        )
        energy_kind = next(
            (prop for prop in props if prop in {"energy", "free_energy"}), None
        )
        energies = [] if want_energy else None
        forces = [] if want_forces else None
        for atoms in atoms_list:
            self.calculate(atoms, request, None)
            if energies is not None:
                energies.append(self.results[energy_kind])
            if forces is not None:
                forces.append(self.results["forces"])
        return BatchResult(
            energies=energies,
            energy_kind=energy_kind,
            forces=forces,
        )


class _QuadraticCalculator:
    supports_hvp = False
    energy_free_energy_equal = True

    def calculate_many(self, atoms_list, properties=("energy", "forces")):
        energies = []
        forces = []
        for atoms in atoms_list:
            positions = atoms.get_positions()
            energies.append(0.5 * float(np.sum(positions**2)))
            forces.append(-positions)
        return BatchResult(
            energies=np.asarray(energies, dtype=np.float64),
            energy_kind=next(
                (
                    prop
                    for prop in properties
                    if prop in ("energy", "free_energy")
                ),
                None,
            ),
            forces=forces,
        )


class _BadAnalyticHVP:
    supports_hvp = True
    hvp_energy_kind = "free_energy"

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
    def test_energy_kind_is_required_and_must_match_the_request(self):
        atoms = [Atoms("H")]
        unlabeled = BatchResult(energies=np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "energy_kind"):
            unlabeled.validate_against(atoms, ("energy",))

        mislabeled = BatchResult(
            energies=np.array([1.0]), energy_kind="free_energy"
        )
        with self.assertRaisesRegex(ValueError, "energy_kind.*requested"):
            mislabeled.validate_against(atoms, ("energy",))

        with self.assertRaisesRegex(ValueError, "energy_kind"):
            BatchResult(forces=[np.zeros((1, 3))], energy_kind="energy")

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
            energy_kind="energy",
            forces=[np.zeros((1, 3)), np.zeros((1, 3))],
        )
        with self.assertRaisesRegex(ValueError, r"forces\[1\].*expected"):
            result.validate_against(atoms, ("energy", "forces"))

        with self.assertRaisesRegex(ValueError, "requested.*forces"):
            BatchResult(
                energies=np.array([1.0, 2.0]), energy_kind="energy"
            ).validate_against(
                atoms, ("energy", "forces")
            )

    def test_nested_arrays_are_read_only_and_revalidated(self):
        atoms = [Atoms("H")]
        result = BatchResult(
            energies=np.array([1.0]),
            energy_kind="energy",
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
    def test_energy_request_is_exactly_one_ase_scalar(self):
        self.assertEqual(
            normalize_energy_forces_request(("free_energy", "forces"))[0],
            ("free_energy", "forces"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            normalize_energy_forces_request(("energy", "free_energy", "forces"))

    def test_sequential_fallback_selects_only_the_requested_energy_kind(self):
        calc = _DistinctEnergyCalculator()
        atoms = [Atoms("H", positions=[[0.25, 0.0, 0.0]])]
        potential = sequential_calculate_many(
            calc, atoms, ("energy", "forces"), True, True
        )
        free = sequential_calculate_many(
            calc, atoms, ("free_energy", "forces"), True, True
        )
        self.assertEqual(potential.energy_kind, "energy")
        self.assertEqual(free.energy_kind, "free_energy")
        np.testing.assert_allclose(potential.energies, [10.25])
        np.testing.assert_allclose(free.energies, [20.25])
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


class EnergyEvaluatorIdentityTests(unittest.TestCase):
    def test_scalar_helper_requests_the_exact_selected_ase_energy(self):
        calc = _DistinctEnergyCalculator()
        atoms = Atoms("H", positions=[[0.5, 0.0, 0.0]])

        potential, _ = energy_forces_one(calc, atoms, force_consistent=False)
        free, _ = energy_forces_one(calc, atoms, force_consistent=True)

        self.assertEqual(calc.requests, [
            ("energy", "forces"),
            ("free_energy", "forces"),
        ])
        self.assertEqual(potential, 10.5)
        self.assertEqual(free, 20.5)

    def test_scalar_free_energy_fallback_requires_explicit_equality(self):
        class _EnergyOnly(Calculator):
            implemented_properties = ("energy", "forces")

            def __init__(self, declares_equal):
                super().__init__()
                if declares_equal:
                    self.energy_free_energy_equal = True

            def calculate(self, atoms, properties, system_changes):
                super().calculate(atoms, properties, system_changes)
                self.results = {
                    "energy": 10.5,
                    "forces": np.zeros((len(atoms), 3)),
                }

        atoms = Atoms("H", positions=[[0.5, 0.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "energy_free_energy_equal=True"):
            energy_forces_one(_EnergyOnly(False), atoms, force_consistent=True)
        malformed = _EnergyOnly(False)
        malformed.energy_free_energy_equal = "false"
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            energy_forces_one(malformed, atoms, force_consistent=True)
        energy, _ = energy_forces_one(
            _EnergyOnly(True), atoms, force_consistent=True
        )
        self.assertEqual(energy, 10.5)

    def test_dimer_callback_center_evaluation_requests_free_energy(self):
        calc = _DistinctEnergyCalculator()
        atoms = Atoms("H", positions=[[0.5, 0.0, 0.0]], calculator=calc)
        with tempfile.TemporaryDirectory() as directory:
            dimer = Dimer(
                output=str(Path(directory) / "dimer.out"),
                atoms_init=atoms,
                paras={"dimer": {
                    "use_hvp": True,
                    "remove_rigid": False,
                    "n_init": "given",
                    "n_given": [1.0, 0.0, 0.0],
                }},
                hvp_fn=lambda _atoms, direction: np.asarray(direction),
            )
            _, _, energy = dimer._evaluate_current(dimer.n)

        self.assertEqual(energy, 20.5)
        self.assertIn(("free_energy", "forces"), calc.requests)

    def test_native_backend_cannot_relabel_potential_energy_as_free_energy(self):
        calc = _DistinctEnergyCalculator(native=True)
        calc.calculate_many = lambda atoms_list, properties: BatchResult(
            energies=np.array([10.5]),
            energy_kind="energy",
            forces=[np.zeros((1, 3))],
        )
        with self.assertRaisesRegex(ValueError, "energy_kind.*requested"):
            PathEvaluator(calc, batch_size="all").energy_forces(
                [Atoms("H", positions=[[0.5, 0.0, 0.0]])]
            )

    def test_native_path_and_energy_evaluators_keep_distinct_scalars(self):
        calc = _DistinctEnergyCalculator(native=True)
        images = [Atoms("H", positions=[[0.5, 0.0, 0.0]])]
        path_e, _ = PathEvaluator(calc, batch_size="all").energy_forces(images)
        sp_e = EnergyEvaluator(calc, batch_size="all").energies(images)
        path_summary_e = EnergyEvaluator(
            calc,
            batch_size="all",
            force_consistent=True,
        ).energies(images)
        np.testing.assert_allclose(path_e, [20.5])
        np.testing.assert_allclose(sp_e, [10.5])
        np.testing.assert_allclose(path_summary_e, [20.5])

    def test_periodic_fallback_preserves_requested_energy_kind(self):
        calc = _DistinctEnergyCalculator(native=True)
        atoms = Atoms("H", positions=[[0.75, 0.0, 0.0]], cell=[4, 4, 4], pbc=True)
        path_e, _ = PathEvaluator(calc, batch_size="all").energy_forces([atoms])
        sp_e = EnergyEvaluator(calc, batch_size="all").energies([atoms])
        np.testing.assert_allclose(path_e, [20.75])
        np.testing.assert_allclose(sp_e, [10.75])

    def test_sp_always_uses_potential_energy_for_verbose_and_sequential_routes(self):
        cases = []
        for native, constrained in ((True, False), (False, True)):
            calc = _DistinctEnergyCalculator(native=native)
            frames = [Atoms("H", positions=[[0.5, 0.0, 0.0]])]
            if constrained:
                frames[0].set_constraint(FixAtoms(indices=[0]))
            frames[0].calc = calc
            for verbose in (0, 1):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "sp.out"
                    SinglePoint(
                        str(output), frames, paras={"verbose": verbose}
                    ).run()
                    cases.append(output.read_text())
        for text in cases:
            self.assertIn("Energy: 10.5000000000 Hartree", text)
            self.assertNotIn("Energy: 20.5000000000 Hartree", text)


class BackendContractTests(unittest.TestCase):
    def test_aimnet_rigid_invariance_is_exact_hash_and_configuration_bound(self):
        calc = AIMNet2Calculator.__new__(AIMNet2Calculator)
        calc.external_field = None
        for known in AIMNET2_CHECKPOINT_CAPABILITIES_BY_SHA256:
            for method in ("simple", "dsf"):
                calc.maple_pes_identity = {
                    "model_fingerprint": {"digest": known},
                    "relevant_settings": {
                        "implicit": "none",
                        "coulomb": {"method": method},
                    },
                }
                self.assertIs(calc.rigid_body_invariant, True)

        calc.maple_pes_identity["model_fingerprint"]["digest"] = "0" * 64
        self.assertIsNone(calc.rigid_body_invariant)
        calc.maple_pes_identity["model_fingerprint"]["digest"] = known
        calc.maple_pes_identity["relevant_settings"]["implicit"] = "gbsa"
        self.assertIsNone(calc.rigid_body_invariant)
        calc.maple_pes_identity["relevant_settings"]["implicit"] = "none"
        calc.external_field = np.ones(3)
        self.assertIsNone(calc.rigid_body_invariant)

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
    def test_analytic_hvp_requires_explicit_energy_identity_before_call(self):
        class _AnalyticEnergyIdentity:
            supports_hvp = True

            def __init__(self, *, kind=None, equal=False):
                if kind is not None:
                    self.hvp_energy_kind = kind
                if equal:
                    self.energy_free_energy_equal = True
                self.calls = 0

            def get_hvp(self, atoms, direction):
                self.calls += 1
                energy = 20.0 if getattr(
                    self, "hvp_energy_kind", None
                ) == "free_energy" else 10.0
                return np.asarray(direction), np.zeros(3), energy

        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        direction = np.array([1.0, 0.0, 0.0])

        free = _AnalyticEnergyIdentity(kind="free_energy")
        self.assertEqual(HVPEvaluator(free).hn(atoms, direction)[2], 20.0)
        self.assertEqual(free.calls, 1)

        potential = _AnalyticEnergyIdentity(kind="energy")
        with self.assertRaisesRegex(RuntimeError, "energy_free_energy_equal"):
            HVPEvaluator(potential).hn(atoms, direction)
        self.assertEqual(potential.calls, 0)

        unknown = _AnalyticEnergyIdentity()
        with self.assertRaisesRegex(RuntimeError, "hvp_energy_kind"):
            HVPEvaluator(unknown).hn(atoms, direction)
        self.assertEqual(unknown.calls, 0)

        malformed = _AnalyticEnergyIdentity()
        malformed.energy_free_energy_equal = "false"
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            HVPEvaluator(malformed).hn(atoms, direction)
        self.assertEqual(malformed.calls, 0)

        equal = _AnalyticEnergyIdentity(equal=True)
        self.assertEqual(HVPEvaluator(equal).hn(atoms, direction)[2], 10.0)
        self.assertEqual(equal.calls, 1)

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

    def test_hvp_fd_selection_never_touches_analytic_backend(self):
        class _AnalyticSpy(_QuadraticCalculator):
            supports_hvp = True

            def get_hvp(self, atoms, n):
                raise AssertionError("analytic HVP must not be called")

        atoms = Atoms("H", positions=[[0.1, 0.0, 0.0]])
        hn, force, energy = HVPEvaluator(
            _AnalyticSpy(), use_analytic=False
        ).hn(atoms, np.array([1.0, 0.0, 0.0]), delta=0.005)
        np.testing.assert_allclose(hn, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(force, [-0.1, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(energy, 0.005)


if __name__ == "__main__":
    unittest.main()
