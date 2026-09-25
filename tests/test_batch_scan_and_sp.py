import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._batch_types import BatchResult
from maple.function.dispatcher.scan.scan import Scan
from maple.function.dispatcher.sp.sp import SinglePoint


class _BatchEnergyCalculator:
    supports_batch_energy_forces = True

    def __init__(self):
        self.batch_lengths = []

    def calculate_many(self, atoms_list, properties=("energy",)):
        atoms_list = list(atoms_list)
        self.batch_lengths.append(len(atoms_list))
        energies = np.asarray(
            [float(np.sum(atoms.get_positions() ** 2)) for atoms in atoms_list]
        )
        return BatchResult(energies=energies).validate_against(
            atoms_list, properties
        )


class _StatefulCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": float(np.sum(positions**2)),
            "forces": -2.0 * positions,
        }


class BatchScanAndSPTests(unittest.TestCase):
    def test_rigid_scan_flushes_a_bounded_record_buffer_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _BatchEnergyCalculator()
            atoms = Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
            )
            atoms.calc = calc
            scan = Scan(
                output=str(Path(tmpdir) / "scan.out"),
                atoms=atoms,
                constraints=[[1, 2, 0.05, 5]],
                params={"mode": "rigid", "scan_batch_size": 2},
            )
            scan.run_scan()

            self.assertEqual(calc.batch_lengths, [2, 2, 2])
            xyz_text = (Path(tmpdir) / "scan_scan_final.xyz").read_text()
            indices = [
                int(line.split()[2].split("/")[0])
                for line in xyz_text.splitlines()
                if line.startswith("Scanning combination")
            ]
            self.assertEqual(indices, [1, 2, 3, 4, 5, 6])

    def test_empty_single_point_trajectory_has_a_valid_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sp = SinglePoint(
                output=str(Path(tmpdir) / "sp.out"),
                atoms=[],
            )
            sp.run()
            text = (Path(tmpdir) / "sp.out").read_text()
            self.assertIn("Total frames processed: 0", text)
            self.assertIn("No structures to summarize", text)

    def test_single_point_fallback_restores_shared_calculator_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _StatefulCalculator()
            original_atoms = Atoms("He", positions=[[9.0, 8.0, 7.0]])
            original_results = {
                "energy": 7.0,
                "forces": np.full((1, 3), 4.0),
            }
            calc.atoms = original_atoms
            calc.results = original_results

            frames = [
                Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=calc),
                Atoms("H", positions=[[1.0, 0.0, 0.0]], calculator=calc),
            ]
            sp = SinglePoint(
                output=str(Path(tmpdir) / "sp.out"),
                atoms=frames,
                paras={"verbose": 1},
            )
            sp.run()
            text = (Path(tmpdir) / "sp.out").read_text()
            self.assertIn("Energy: 0.0000000000 Hartree", text)
            self.assertIn("Energy: 1.0000000000 Hartree", text)
            self.assertEqual(text.count("Gradients (Hartree/Angstrom):"), 2)
            self.assertIs(calc.atoms, original_atoms)
            self.assertIs(calc.results, original_results)
            np.testing.assert_allclose(calc.results["forces"], 4.0)


if __name__ == "__main__":
    unittest.main()
