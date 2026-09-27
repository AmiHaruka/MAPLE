"""Asset-free behavior locks for the real-backend verification script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from maple.function.calculator._batch_types import BatchResult


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_batch_backends.py"
SPEC = importlib.util.spec_from_file_location("verify_batch_backends", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify)


class _FakeCalculator:
    supports_batch_energy_forces = True

    def __init__(self):
        self.results = {}
        self.maple_pes_identity = {"backend": "fake", "checkpoint": "fixed"}

    def get_pes_identity(self):
        return dict(self.maple_pes_identity)

    @staticmethod
    def _values(atoms):
        energy = float(np.sum(atoms.numbers) + np.sum(atoms.positions) * 0.01)
        forces = np.full((len(atoms), 3), -0.01, dtype=np.float64)
        return energy, forces

    def calculate(self, atoms, properties):
        energy, forces = self._values(atoms)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}

    def calculate_many(self, atoms_list, properties):
        values = [self._values(atoms) for atoms in atoms_list]
        return BatchResult(
            energies=np.asarray([value[0] for value in values]),
            energy_kind=properties[0],
            forces=tuple(value[1] for value in values),
        ).validate_against(atoms_list, properties)


class _WrongChunkLabelCalculator(_FakeCalculator):
    def calculate_many(self, atoms_list, properties):
        result = super().calculate_many(atoms_list, properties)
        if len(atoms_list) == 2:
            return BatchResult(
                energies=result.energies,
                energy_kind="free_energy" if properties[0] == "energy" else "energy",
                forces=result.forces,
            )
        return result


class _SequentialOnlyCalculator(_FakeCalculator):
    supports_batch_energy_forces = False


class _HiddenScalarFallbackCalculator(_FakeCalculator):
    def calculate_many(self, atoms_list, properties):
        energies, forces = [], []
        for atoms in atoms_list:
            self.calculate(atoms, properties)
            energies.append(self.results[properties[0]])
            forces.append(self.results["forces"])
        return BatchResult(
            energies=np.asarray(energies),
            energy_kind=properties[0],
            forces=tuple(forces),
        ).validate_against(atoms_list, properties)


class VerifyBatchBackendsTests(unittest.TestCase):
    def test_neutral_fixture_contains_eight_structures_and_a_60_atom_case(self):
        structures = verify.neutral_structures()
        self.assertEqual(len(structures), 8)
        self.assertEqual(len(structures[-1]), 60)

    def test_panel_records_all_fixed_cases_for_both_energy_kinds(self):
        structures = verify.neutral_structures()
        report = verify.run_parity_panel(
            _FakeCalculator(), "ani2x", structures, verify.FROZEN_BOUNDS["ani2x"]
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["measurements"]), 16)
        self.assertEqual({item["energy_kind"] for item in report["measurements"]}, {"energy", "free_energy"})
        self.assertTrue(all(item["status"] == "PASS" for item in report["measurements"]))

    def test_missing_bound_is_report_only_unqualified(self):
        report = verify.run_parity_panel(
            _FakeCalculator(), "uma", verify.neutral_structures(), None
        )
        self.assertEqual(report["status"], "UNQUALIFIED")
        self.assertTrue(all(item["status"] == "UNQUALIFIED" for item in report["measurements"]))

    def test_unavailable_record_never_reports_pass(self):
        record = verify.unavailable("uma", "resource opt-in absent")
        self.assertEqual(record["status"], "UNAVAILABLE")
        self.assertIn("resource", record["reason"])

    def test_chunked_validates_each_native_result_before_aggregation(self):
        with self.assertRaisesRegex(ValueError, "does not match requested"):
            verify._chunked(
                _WrongChunkLabelCalculator(),
                verify.neutral_structures(),
                ("energy", "forces"),
                2,
            )

    def test_sequential_only_calculator_cannot_pass_native_panel(self):
        with self.assertRaisesRegex(
            verify.NativeBatchUnavailableError,
            "does not explicitly declare",
        ):
            verify.run_parity_panel(
                _SequentialOnlyCalculator(),
                "ani2x",
                verify.neutral_structures(),
                verify.FROZEN_BOUNDS["ani2x"],
            )

    def test_hidden_scalar_fallback_fails_multi_structure_native_panel(self):
        report = verify.run_parity_panel(
            _HiddenScalarFallbackCalculator(),
            "ani2x",
            verify.neutral_structures(),
            verify.FROZEN_BOUNDS["ani2x"],
        )
        self.assertEqual(report["status"], "FAIL")
        b2 = [
            item
            for item in report["measurements"]
            if item["case"] == "B2" and item["energy_kind"] == "energy"
        ][0]
        self.assertEqual(b2["status"], "FAIL")
        self.assertFalse(b2["native_no_serial_fallback"])
        self.assertEqual(
            b2["native_fallback_observations"][0]["scalar_calculate_calls"],
            2,
        )

    def test_require_complete_rejects_an_unavailable_selected_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.json"
            exit_code = verify.main(
                [
                    "--models",
                    "uma",
                    "--checkpoint-dir",
                    directory,
                    "--artifact",
                    str(artifact),
                    "--require-complete",
                ]
            )
            self.assertEqual(exit_code, 1)
            self.assertEqual(verify.json.loads(artifact.read_text())["status"], "INCOMPLETE")

    def test_source_manifest_hashes_relevant_backend_and_shared_sources(self):
        manifest = verify.source_manifest(["ani2x"])
        self.assertIn("maple/function/calculator/_batch_eval.py", manifest)
        self.assertIn("maple/function/calculator/ani/_ani_calculator.py", manifest)
        self.assertTrue(all(len(digest) == 64 for digest in manifest.values()))


if __name__ == "__main__":
    unittest.main()
