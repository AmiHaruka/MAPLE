import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._batch_types import BatchResult
from maple.function.dispatcher.scan.scan import Scan
from maple.function.dispatcher.sp.sp import SinglePoint
from maple.function.dispatcher.ts.algorithm.string import (
    get_energies as get_string_energies,
)


class _BatchEnergyCalculator:
    supports_batch_energy_forces = True

    def __init__(self):
        self.batch_lengths = []

    def calculate_many(self, atoms_list, properties=("energy",)):
        atoms_list = list(atoms_list)
        self.batch_lengths.append(len(atoms_list))
        energy_kind = next(
            (name for name in properties if name in {"energy", "free_energy"}),
            None,
        )
        energies = np.asarray(
            [float(np.sum(atoms.get_positions() ** 2)) for atoms in atoms_list]
        )
        return BatchResult(
            energies=energies,
            energy_kind=energy_kind,
        ).validate_against(atoms_list, properties)


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


class _DistinctEnergyCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, *, native):
        super().__init__()
        self.supports_batch_energy_forces = native
        self.requests = []

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.requests.append(tuple(properties))
        x = float(np.sum(atoms.get_positions()))
        self.results = {
            "energy": 10.0 + x,
            "free_energy": 20.0 + x,
            "forces": np.full((len(atoms), 3), -x),
        }

    def calculate_many(self, atoms_list, properties=("energy",)):
        if not self.supports_batch_energy_forces:
            raise AssertionError("native batch route was not expected")
        energy_kind = next(
            (name for name in properties if name in {"energy", "free_energy"}),
            None,
        )
        energies = [] if energy_kind is not None else None
        forces = [] if "forces" in properties else None
        for atoms in atoms_list:
            self.calculate(atoms, properties, all_changes)
            if energies is not None:
                energies.append(self.results[energy_kind])
            if forces is not None:
                forces.append(self.results["forces"])
        return BatchResult(
            energies=energies,
            energy_kind=energy_kind,
            forces=forces,
        )


class BatchScanAndSPTests(unittest.TestCase):
    @staticmethod
    def _run_distinct_energy_rigid_scan(*, native):
        with tempfile.TemporaryDirectory() as tmpdir:
            calc = _DistinctEnergyCalculator(native=native)
            atoms = Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
                calculator=calc,
            )
            scan = Scan(
                output=str(Path(tmpdir) / "scan.out"),
                atoms=atoms,
                constraints=[[1, 2, 0.1, 1]],
                params={"mode": "rigid", "scan_batch_size": "all"},
            )
            scan._total_combinations = 2
            scan._current_index = 0
            scan.xyz_file = io.StringIO()
            _, energies = scan._scan_1d([[0.7, 0.8]])
            return energies, calc.requests

    def test_rigid_scan_native_and_sequential_use_force_consistent_energy(self):
        native_energies, native_requests = self._run_distinct_energy_rigid_scan(
            native=True
        )
        sequential_energies, sequential_requests = (
            self._run_distinct_energy_rigid_scan(native=False)
        )
        np.testing.assert_allclose(native_energies, sequential_energies)
        self.assertTrue(all(energy > 20.0 for energy in native_energies))
        self.assertTrue(
            any("free_energy" in request for request in native_requests)
        )
        self.assertTrue(
            any("free_energy" in request for request in sequential_requests)
        )

    def test_string_path_summary_native_and_sequential_use_free_energy(self):
        results = []
        for native in (True, False):
            calc = _DistinctEnergyCalculator(native=native)
            images = [
                Atoms("H", positions=[[0.5, 0.0, 0.0]], calculator=calc),
                Atoms("H", positions=[[1.0, 0.0, 0.0]], calculator=calc),
            ]
            results.append(get_string_energies(images))
        np.testing.assert_allclose(results[0], [20.5, 21.0])
        np.testing.assert_allclose(results[1], results[0])

    def test_three_dimensional_scan_keeps_initial_plane_and_full_order(self):
        expected = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
        ]
        for mode in ("rigid", "relaxed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmpdir:
                calc = _StatefulCalculator()
                atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=calc)
                scan = Scan(
                    output=str(Path(tmpdir) / "scan.out"),
                    atoms=atoms,
                    constraints=[[1, 1, 1.0, 1]],
                    params={"mode": mode},
                )
                scan._total_combinations = 8
                scan._current_index = 0
                recorded = []

                def apply_point(current, coord):
                    current.positions[0] = coord
                    return current

                scan._apply_rigid_geometry = apply_point
                scan._apply_constraints = apply_point
                scan._run_optimizer = lambda current: current
                scan._record_result = (
                    lambda current, coord, coords, energies: (
                        recorded.append((coord[:], current.calc)),
                        coords.append(coord[:]),
                        energies.append(float(sum(coord))),
                    )
                )

                coords, energies = scan._scan_3d(
                    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                )

                self.assertEqual(coords, expected)
                self.assertEqual(len(energies), 8)
                self.assertEqual([coord for coord, _ in recorded], expected)
                self.assertTrue(all(point_calc is calc for _, point_calc in recorded))

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
