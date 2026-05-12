import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.scan.scan import Scan
from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        energy = 0.5 * float(np.sum(positions**2))
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = -positions


def _atoms():
    atoms = Atoms("HH", positions=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    atoms.calc = HarmonicCalculator()
    atoms.f_max_th = 1e-12
    atoms.f_rms_th = 1e-12
    atoms.dp_max_th = 1e-12
    atoms.dp_rms_th = 1e-12
    return atoms


def test_scan_validation_allows_opt_methods(tmp_path):
    for method in ("lbfgs", "rfo", "sd", "cg", "sdcg"):
        cc = CommandControl.from_settings(
            [f"#scan(method={method})", "#model=uma"],
            output_path=str(tmp_path / f"{method}.out"),
        )
        assert cc.task == "scan"
        assert cc.params["method"] == method


def test_scan_validation_rejects_empty_method(tmp_path):
    with pytest.raises(ValueError, match="Method '' not implemented for task 'scan'"):
        CommandControl.from_settings(
            ["#scan(method=)", "#model=uma"],
            output_path=str(tmp_path / "scan.out"),
        )


def test_scan_defaults_to_lbfgs_when_method_omitted(tmp_path):
    atoms = _atoms()
    scan = Scan(
        output=str(tmp_path / "scan.out"),
        atoms=atoms,
        method=None,
        constraints=[[1, 2, 0.0, 0]],
        params={},
    )

    assert scan.method == "lbfgs"


def test_scan_input_reader_treats_s_line_as_scan_constraint(tmp_path):
    input_path = tmp_path / "scan.inp"
    input_path.write_text(
        "\n".join(
            [
                "#model=uma",
                "#scan(method=lbfgs)",
                "",
                "O 0.000000 0.000000 0.000000",
                "H 0.000000 0.000000 0.960000",
                "",
                "S 1 2 0.02 1",
                "",
            ]
        )
    )
    reader = InputReader()

    atoms = reader(str(input_path), str(tmp_path / "scan.out"))

    assert len(atoms) == 2
    assert reader.scan_constraints == [[1, 2, 0.02, 1]]


def test_scan_input_reader_keeps_sulfur_coordinate_lines(tmp_path):
    input_path = tmp_path / "scan_sulfur.inp"
    input_path.write_text(
        "\n".join(
            [
                "#model=uma",
                "#scan(method=lbfgs)",
                "",
                "H 0 0 0",
                "S 1 2 3",
                "S 1 2 3 4 5 6",
                "",
            ]
        )
    )
    reader = InputReader()

    atoms = reader(str(input_path), str(tmp_path / "scan_sulfur.out"))

    assert atoms.get_chemical_symbols() == ["H", "S", "S"]
    assert not hasattr(reader, "scan_constraints")


def test_scan_uses_opt_dispatch_for_cg(tmp_path):
    atoms = _atoms()
    scan = Scan(
        output=str(tmp_path / "scan.out"),
        atoms=atoms,
        method="cg",
        constraints=[[1, 2, 0.0, 0]],
        params={"method": "cg", "max_iter": 1, "max_step": 0.2},
    )
    initial_energy = atoms.get_potential_energy(force_consistent=True)

    result = scan._run_optimizer(atoms)

    assert result is atoms
    assert atoms.get_potential_energy(force_consistent=True) < initial_energy


def test_scan_cleanup_removes_opt_temp_outputs(tmp_path):
    output = tmp_path / "scan.out"
    for suffix in ("_opt.xyz", "_opt_traj.xyz"):
        output.with_name(output.stem + suffix).write_text("temporary\n")

    Scan._cleanup_opt_files(output)

    for suffix in ("_opt.xyz", "_opt_traj.xyz"):
        assert not output.with_name(output.stem + suffix).exists()
