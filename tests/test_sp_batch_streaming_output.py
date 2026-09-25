"""Single-point trajectory output is a committed prefix of evaluated frames."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._batch_types import BatchResult
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.read.filereader.pdb_reader import PDBReader
from maple.function.read.filereader.xyz_traj_reader import XYZTrajReader


class _RecordingBatchCalculator:
    supports_batch_energy_forces = True

    def __init__(self, *, batch_size=2, fail_call=None, malformed_call=None):
        self.path_batch_size = batch_size
        self.fail_call = fail_call
        self.malformed_call = malformed_call
        self.calls = []

    def calculate_many(self, atoms_list, properties=("energy",)):
        atoms_list = list(atoms_list)
        self.calls.append(tuple(float(at.positions[0, 0]) for at in atoms_list))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("deliberate later batch failure")
        energies = [float(np.sum(at.positions**2)) for at in atoms_list]
        forces = [-2.0 * at.positions for at in atoms_list]
        if len(self.calls) == self.malformed_call:
            energies.pop()
        return BatchResult(
            energies=np.asarray(energies),
            forces=forces if "forces" in properties else None,
        )


class _StatefulFailingCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def __init__(self):
        super().__init__()
        self.atoms = Atoms("He", positions=[[9.0, 8.0, 7.0]])
        self.results = {"energy": 7.0, "forces": np.full((1, 3), 4.0)}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms.positions[0, 0] >= 2.0:
            raise RuntimeError("deliberate later scalar failure")
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(np.sum(atoms.positions**2)),
            "forces": -2.0 * atoms.positions,
        }


def _frames(calc, count):
    frames = []
    for index in range(count):
        atoms = Atoms("H", positions=[[float(index), 0.0, 0.0]])
        atoms.info.update(charge=index % 2, mult=1)
        atoms.calc = calc
        frames.append(atoms)
    return frames


class SinglePointStreamingTests(unittest.TestCase):
    def test_success_keeps_frame_order_energy_force_and_coordinate_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _RecordingBatchCalculator()
            output = Path(tmpdir) / "sp.out"
            SinglePoint(str(output), _frames(calc, 3), paras={"verbose": 1}).run()

            text = output.read_text()
            self.assertEqual(calc.calls, [(0.0, 1.0), (2.0,)])
            self.assertEqual(text.count("Gradients (Hartree/Angstrom):"), 3)
            self.assertLess(text.index("Frame 1"), text.index("Frame 2"))
            self.assertLess(text.index("Frame 2"), text.index("Frame 3"))
            self.assertIn("Energy: 4.0000000000 Hartree", text)
            self.assertIn("  1    H       2.00000000      0.00000000      0.00000000", text)
            self.assertIn("  1    H       4.00000000      0.00000000      0.00000000", text)
            self.assertIn("Total frames processed: 3", text)
            self.assertIn("Energy range: 0.0000000000 to 4.0000000000 Hartree", text)
            self.assertIn("Energy span: 4.0000000000 Hartree", text)
            self.assertIn("Charge: 1, Multiplicity: 1", text)

    def test_later_batch_failure_keeps_completed_prefix_without_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _RecordingBatchCalculator(fail_call=2)
            output = Path(tmpdir) / "sp.out"
            with self.assertRaisesRegex(RuntimeError, "later batch failure"):
                SinglePoint(str(output), _frames(calc, 5), paras={"verbose": 1}).run()

            self.assertEqual(calc.calls, [(0.0, 1.0), (2.0, 3.0)])
            text = output.read_text()
            self.assertIn("Frame 1", text)
            self.assertIn("Frame 2", text)
            self.assertNotIn("Frame 3", text)
            self.assertNotIn("SUMMARY", text)

    def test_invalid_later_batch_does_not_write_part_of_failed_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _RecordingBatchCalculator(malformed_call=2)
            output = Path(tmpdir) / "sp.out"
            with self.assertRaisesRegex(ValueError, "length"):
                SinglePoint(str(output), _frames(calc, 5)).run()

            text = output.read_text()
            self.assertIn("Frame 2", text)
            self.assertNotIn("Frame 3", text)
            self.assertNotIn("SUMMARY", text)

    def test_sequential_failure_keeps_prefix_and_restores_shared_ase_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _StatefulFailingCalculator()
            original_atoms = calc.atoms
            original_results = calc.results
            output = Path(tmpdir) / "sp.out"
            with self.assertRaisesRegex(RuntimeError, "later scalar failure"):
                SinglePoint(str(output), _frames(calc, 4), paras={"verbose": 1}).run()

            text = output.read_text()
            self.assertIn("Frame 1", text)
            self.assertIn("Frame 2", text)
            self.assertNotIn("Frame 3", text)
            self.assertNotIn("SUMMARY", text)
            self.assertIs(calc.atoms, original_atoms)
            self.assertIs(calc.results, original_results)
            np.testing.assert_array_equal(calc.results["forces"], 4.0)

    def test_auto_sized_workload_flushes_bounded_multi_frame_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _RecordingBatchCalculator(batch_size="auto")
            output = Path(tmpdir) / "sp.out"
            SinglePoint(str(output), _frames(calc, 40)).run()

            self.assertEqual(list(map(len, calc.calls)), [32, 8])
            self.assertIn("Frame 40", output.read_text())
            self.assertIn("Total frames processed: 40", output.read_text())

    def test_auto_path_cap_also_bounds_the_last_committed_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _RecordingBatchCalculator(batch_size="auto", fail_call=2)
            calc.auto_path_batch_cap = 2
            output = Path(tmpdir) / "sp.out"
            with self.assertRaisesRegex(RuntimeError, "later batch failure"):
                SinglePoint(str(output), _frames(calc, 5)).run()

            self.assertEqual(calc.calls, [(0.0, 1.0), (2.0, 3.0)])
            text = output.read_text()
            self.assertIn("Frame 2", text)
            self.assertNotIn("Frame 3", text)

    def test_xyz_and_pdb_trajectory_inputs_retain_frame_order_in_sp_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xyz = root / "source.xyz"
            xyz.write_text(
                "1\nfirst\nH 0.000 0.000 0.000\n"
                "1\nsecond\nH 1.000 0.000 0.000\n"
            )
            pdb = root / "source.pdb"
            pdb.write_text(
                "MODEL        1\n"
                "HETATM    1  H   MOL A   1       0.000   0.000   0.000  1.00  0.00           H\n"
                "ENDMDL\n"
                "MODEL        2\n"
                "HETATM    1  H   MOL A   1       1.000   0.000   0.000  1.00  0.00           H\n"
                "ENDMDL\nEND\n"
            )

            for name, frames in (
                ("xyz", XYZTrajReader(str(xyz)).multiatoms),
                ("pdb", PDBReader(str(pdb)).multiatoms),
            ):
                with self.subTest(source=name):
                    calc = _RecordingBatchCalculator()
                    for frame in frames:
                        frame.calc = calc
                    output = root / f"{name}.out"
                    SinglePoint(str(output), frames).run()
                    text = output.read_text()
                    self.assertEqual(calc.calls, [(0.0, 1.0)])
                    self.assertLess(text.index("Frame 1"), text.index("Frame 2"))
                    self.assertIn("Energy: 0.0000000000 Hartree", text)
                    self.assertIn("Energy: 1.0000000000 Hartree", text)
                    self.assertIn("  1    H       1.00000000      0.00000000      0.00000000", text)


if __name__ == "__main__":
    unittest.main()
