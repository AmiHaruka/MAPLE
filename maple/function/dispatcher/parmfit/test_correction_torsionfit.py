from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

try:
    from ase import Atoms
except ModuleNotFoundError:
    ase_stub = types.ModuleType("ase")
    constraints_stub = types.ModuleType("ase.constraints")
    neighborlist_stub = types.ModuleType("ase.neighborlist")

    class Atoms:  # type: ignore[override]
        def __init__(self, symbols, positions):
            self.symbols = list(symbols)
            self.positions = np.asarray(positions, dtype=float)
            self.info = {}
            self.calc = None

        def __len__(self):
            return len(self.symbols)

        def copy(self):
            atoms = Atoms(self.symbols[:], self.positions.copy())
            atoms.info = dict(self.info)
            atoms.calc = self.calc
            return atoms

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def set_positions(self, positions):
            self.positions = np.asarray(positions, dtype=float)

        def get_forces(self):
            return self.calc.get_forces(self)

        def get_potential_energy(self, force_consistent=True):
            return self.calc.get_potential_energy(self, force_consistent=force_consistent)

        def get_chemical_symbols(self):
            return self.symbols[:]

    class FixInternals:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    class NeighborList:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

        def update(self, atoms):
            return None

        def get_neighbors(self, index):
            return np.asarray([], dtype=int), np.asarray([], dtype=int)

    def natural_cutoffs(atoms):
        return [1.0] * len(atoms)

    ase_stub.Atoms = Atoms
    constraints_stub.FixInternals = FixInternals
    neighborlist_stub.NeighborList = NeighborList
    neighborlist_stub.natural_cutoffs = natural_cutoffs
    sys.modules["ase"] = ase_stub
    sys.modules["ase.constraints"] = constraints_stub
    sys.modules["ase.neighborlist"] = neighborlist_stub

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.correction import correction as correction_module
from maple.function.dispatcher.parmfit.correction.correction import Correction
from maple.function.dispatcher.parmfit.utils.mmfunc import angle_radians, distance_angstrom


def _four_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(180.0 + phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
    ]
    return Atoms(symbols=["C", "C", "C", "C"], positions=positions)


def _set_thresholds(atoms: Atoms) -> None:
    atoms.f_max_th = 0.00285
    atoms.f_rms_th = 0.00190
    atoms.dp_max_th = 0.00315
    atoms.dp_rms_th = 0.00210


class FakeCalculator:
    def get_forces(self, atoms):
        return np.zeros_like(atoms.get_positions(), dtype=float)

    def get_potential_energy(self, atoms, force_consistent=True):
        return 0.0

    def get_hessian(self, atoms):
        size = 3 * len(atoms)
        return np.eye(size, dtype=float)


class FakeScan:
    def __init__(self, output, atoms, method="lbfgs", constraints=None, params=None):
        self.output = output
        self.atoms = atoms
        self.constraints = constraints or []

    def run(self):
        base, _ = os.path.splitext(self.output)
        xyz_path = base + "_scan_final.xyz"
        phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
        k_true = 1.75
        with open(xyz_path, "w", encoding="utf-8") as handle:
            for index, phi in enumerate(phis, start=1):
                atoms = _four_atom_frame(phi)
                energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
                energy_hartree = energy_kcal / 627.509474
                handle.write(f"{len(atoms)}\n")
                handle.write(
                    f"Scanning combination {index}/{len(phis)}: [{phi:.4f}]  Energy = {energy_hartree:.10f}\n"
                )
                for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                    handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _fake_apply_mseminario(atoms, hessian, bonds, angles, vibrational_scaling=1.0):
    for bond in bonds:
        bond.kBond = 111.0
        bond.rEq = distance_angstrom(atoms.get_positions(), *bond.atoms)
    for angle in angles:
        angle.kTheta = 3.21
        angle.thetaEq = angle_radians(atoms.get_positions(), *angle.atoms)


def _write_chain_mol2(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "TEST",
                "4 3 0 0 0",
                "SMALL",
                "NO_CHARGES",
                "@<TRIPOS>ATOM",
                "1 C1 1.0000 0.0000 0.0000 c 1 RES 0.0000",
                "2 C2 0.0000 0.0000 0.0000 c 1 RES 0.0000",
                "3 C3 0.0000 0.0000 1.0000 c 1 RES 0.0000",
                "4 C4 -1.0000 0.0000 1.0000 c 1 RES 0.0000",
                "@<TRIPOS>BOND",
                "1 1 2 1",
                "2 2 3 1",
                "3 3 4 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_chain_frcmod(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "BOND",
                "c-c   100.0   1.5000",
                "ANGLE",
                "c-c-c   2.0000   120.0000",
                "DIHE",
                "c-c-c-c   1.0   0.5000   0.0000   1.0",
                "NONBON",
                "c   1.5000   0.0000",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_correction_integrates_torsion_stage1_initializer_into_main_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(correction_module, "Scan", FakeScan)
    monkeypatch.setattr(correction_module, "apply_mseminario", _fake_apply_mseminario)

    mol2_path = tmp_path / "chain_ff.mol2"
    frcmod_path = tmp_path / "chain_ff.frcmod"
    output_path = tmp_path / "corr.out"
    _write_chain_mol2(mol2_path)
    _write_chain_frcmod(frcmod_path)
    output_path.write_text("", encoding="utf-8")

    atoms = _four_atom_frame(0.0)
    atoms.calc = FakeCalculator()
    _set_thresholds(atoms)

    run = Correction(
        output=str(output_path),
        atoms=atoms,
        params={
            "mol2": str(mol2_path),
            "frcmod": str(frcmod_path),
            "torsionfit": True,
            "torsion_scan_step": 60.0,
            "torsion_scan_steps": 5,
            "torsion_refine_rounds": 0,
        },
    )
    run.run()

    assert run.result is not None
    assert run.stage0_result is not None
    assert run.stage1_result is not None
    assert run.result.bonds[0].kBond == pytest.approx(111.0)
    assert run.result.angles[0].kTheta == pytest.approx(3.21)
    assert len(run.torsion_fit_reports) == 1
    assert run.torsion_fit_reports[0]["center_bond"] == (2, 3)

    stage1_k = run.torsion_fit_reports[0]["fitted_terms"][0][0].kPhi
    assert run.torsion_fit_reports[0]["original_terms"][0][0].kPhi == pytest.approx(0.5)
    assert 0.5 < stage1_k < 1.75
    assert run.result.dihedrals[0].terms[0].kPhi == pytest.approx(stage1_k)

    text = output_path.read_text(encoding="utf-8")
    assert "Parmfit Correction Summary" in text
    assert "torsionfit:        enabled" in text
    assert "scan grid:         60.0000 deg x 5 steps" in text
    assert "refinement:        0 rounds, max_iter=100, tol=0.0001" in text
    assert "Center bond: (2, 3)" in text
    assert "Stage-1 delta initializer" in text
    assert "Local initializer RMSE" in text
    assert "refine state:     not requested" in text
    assert "Parmfit Correction Parameter Report" in text
    assert "DIHEDRALS" in text
    assert "orig_MM" in text
    assert "k=100.000000  r=1.500000" in text
    assert "k=111.000000" in text
    assert "theta=120.0000" in text
    assert "theta=90.0000" in text
    assert output_path.with_suffix(".top").is_file()
    assert output_path.with_suffix(".gro").is_file()

    assert "parmfit" not in atoms.info


def test_correction_stage2_global_refine_cycles_are_reported(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(correction_module, "Scan", FakeScan)
    monkeypatch.setattr(correction_module, "apply_mseminario", _fake_apply_mseminario)

    mol2_path = tmp_path / "chain_ff.mol2"
    frcmod_path = tmp_path / "chain_ff.frcmod"
    output_path = tmp_path / "corr_refine.out"
    _write_chain_mol2(mol2_path)
    _write_chain_frcmod(frcmod_path)
    output_path.write_text("", encoding="utf-8")

    atoms = _four_atom_frame(0.0)
    atoms.calc = FakeCalculator()
    _set_thresholds(atoms)

    run = Correction(
        output=str(output_path),
        atoms=atoms,
        params={
            "mol2": str(mol2_path),
            "frcmod": str(frcmod_path),
            "torsionfit": True,
            "torsion_scan_step": 60.0,
            "torsion_scan_steps": 5,
            "torsion_refine_rounds": 2,
            "torsion_refine_max_iter": 5,
            "torsion_refine_tol": 1.0e-8,
        },
    )
    run.run()

    assert run.result is not None
    assert len(run.torsion_fit_reports) == 1
    assert len(run.torsion_refine_reports) >= 1
    first_cycle = run.torsion_refine_reports[0]
    assert len(first_cycle["block_reports"]) == 1
    assert first_cycle["total_loss_after"] <= first_cycle["total_loss_before"] + 1.0e-12
    assert first_cycle["accepted_blocks"] + first_cycle["rejected_blocks"] == 1
    assert (2, 3) in first_cycle["per_scan_rmse_before"]
    assert (2, 3) in first_cycle["per_scan_rmse_after"]
    assert run.result.dihedrals[0].terms[0].kPhi != pytest.approx(0.5)

    text = output_path.read_text(encoding="utf-8")
    assert "cycle 1: accepted=" in text
    assert "objective" in text
    assert "global RMSE" in text
    assert "per-scan RMSE" in text
    assert "NONBONDS" not in text

    assert "parmfit" not in atoms.info


def test_correction_requires_runtime_thresholds(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(correction_module, "Scan", FakeScan)
    monkeypatch.setattr(correction_module, "apply_mseminario", _fake_apply_mseminario)

    mol2_path = tmp_path / "chain_ff.mol2"
    frcmod_path = tmp_path / "chain_ff.frcmod"
    output_path = tmp_path / "corr_missing_thresholds.out"
    _write_chain_mol2(mol2_path)
    _write_chain_frcmod(frcmod_path)
    output_path.write_text("", encoding="utf-8")

    atoms = _four_atom_frame(0.0)
    atoms.calc = FakeCalculator()

    run = Correction(
        output=str(output_path),
        atoms=atoms,
        params={
            "mol2": str(mol2_path),
            "frcmod": str(frcmod_path),
            "torsionfit": True,
            "torsion_scan_step": 60.0,
            "torsion_scan_steps": 5,
            "torsion_refine_rounds": 0,
        },
    )

    with pytest.raises(ValueError, match="LBFGS thresholds"):
        run.run()


def test_correction_requires_calculator_with_hessian(tmp_path: Path):
    mol2_path = tmp_path / "chain_ff.mol2"
    frcmod_path = tmp_path / "chain_ff.frcmod"
    output_path = tmp_path / "corr_missing_calc.out"
    _write_chain_mol2(mol2_path)
    _write_chain_frcmod(frcmod_path)
    output_path.write_text("", encoding="utf-8")

    atoms = _four_atom_frame(0.0)
    _set_thresholds(atoms)

    run = Correction(
        output=str(output_path),
        atoms=atoms,
        params={"mol2": str(mol2_path), "frcmod": str(frcmod_path)},
    )

    with pytest.raises(ValueError, match="atoms.calc"):
        run.run()




if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
