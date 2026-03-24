from pathlib import Path
import sys
import types

import numpy as np
import pytest

try:
    from ase import Atoms
except ModuleNotFoundError:
    ase_stub = types.ModuleType("ase")
    masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

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

        def get_masses(self):
            return np.asarray([masses[symbol] for symbol in self.symbols], dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

        def get_atomic_numbers(self):
            return np.asarray([atomic_numbers[symbol] for symbol in self.symbols], dtype=int)

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.mSeminario import _is_qube_linear_angle, apply_mseminario
from maple.function.dispatcher.parmfit.utils.nma import (
    AMU_TO_AU,
    BOHR_TO_ANGSTROM,
    KELVIN_TO_HARTREE,
    compute_normal_modes,
    sample_normal_modes,
)
from maple.function.dispatcher.parmfit.utils.parm import Angle, Bond


HARTREE_TO_KCAL_MOL = 627.509474


def _build_atoms() -> Atoms:
    return Atoms(
        symbols=["H", "H", "H"],
        positions=[
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
    )


def _build_mseminario_hessian() -> np.ndarray:
    hessian = np.zeros((9, 9), dtype=float)
    block_12 = np.diag([-2.0, -4.0, -6.0]) / HARTREE_TO_KCAL_MOL
    block_32 = np.diag([-5.0, -3.0, -1.0]) / HARTREE_TO_KCAL_MOL
    hessian[0:3, 3:6] = block_12
    hessian[3:6, 0:3] = block_12.T
    hessian[6:9, 3:6] = block_32
    hessian[3:6, 6:9] = block_32.T
    return hessian


def test_apply_mseminario_fills_bonds_and_angles():
    atoms = _build_atoms()
    hessian = _build_mseminario_hessian()
    bonds = [
        Bond(atoms=(1, 2), atom_types=("h", "h")),
        Bond(atoms=(2, 3), atom_types=("h", "h")),
    ]
    angles = [Angle(atoms=(1, 2, 3), atom_types=("h", "h", "h"))]

    out_bonds, out_angles = apply_mseminario(atoms, hessian, bonds, angles)

    assert out_bonds is bonds
    assert out_angles is angles
    assert bonds[0].rEq == pytest.approx(1.0)
    assert bonds[1].rEq == pytest.approx(1.0)
    assert angles[0].thetaEq == pytest.approx(np.pi / 2.0)
    assert np.isfinite(bonds[0].kBond) and bonds[0].kBond >= 0.0
    assert np.isfinite(bonds[1].kBond) and bonds[1].kBond >= 0.0
    assert np.isfinite(angles[0].kTheta) and angles[0].kTheta >= 0.0


def test_apply_mseminario_scaling_and_order_independence():
    atoms = _build_atoms()
    hessian = _build_mseminario_hessian()
    bond = Bond(atoms=(1, 2), atom_types=("h", "h"))
    angle = Angle(atoms=(1, 2, 3), atom_types=("h", "h", "h"))
    rev_bond = Bond(atoms=(2, 1), atom_types=("h", "h"))
    rev_angle = Angle(atoms=(3, 2, 1), atom_types=("h", "h", "h"))

    apply_mseminario(atoms, hessian, [bond], [angle], vibrational_scaling=1.0)
    apply_mseminario(atoms, hessian, [rev_bond], [rev_angle], vibrational_scaling=2.0)

    assert rev_bond.rEq == pytest.approx(bond.rEq)
    assert rev_angle.thetaEq == pytest.approx(angle.thetaEq)
    assert rev_bond.kBond == pytest.approx(bond.kBond * 4.0)
    assert rev_angle.kTheta == pytest.approx(angle.kTheta * 4.0)


def test_apply_mseminario_rejects_bad_hessian_shape():
    atoms = _build_atoms()
    with pytest.raises(ValueError, match="Hessian shape"):
        apply_mseminario(atoms, np.zeros((3, 3)), [], [])


def test_qube_linear_angle_detection_matches_reference_threshold():
    assert _is_qube_linear_angle(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert _is_qube_linear_angle(np.array([1.0, 0.0, 0.0]), np.array([-0.99999, 0.0, 0.0]))
    assert not _is_qube_linear_angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def test_compute_normal_modes_projects_and_shapes():
    atoms = _build_atoms()
    normal_modes = compute_normal_modes(atoms, np.eye(9))

    assert normal_modes.eigenvalues.shape == (9,)
    assert normal_modes.angular_frequencies_au.shape == (9,)
    assert normal_modes.frequencies_cm1.shape == (9,)
    assert normal_modes.modes_mw.shape == (9, 9)
    assert normal_modes.mass_weights.shape == (9,)
    assert normal_modes.hessian_projected.shape == (9, 9)
    assert np.all(np.abs(normal_modes.eigenvalues[:6]) < 1.0e-10)
    assert np.all(normal_modes.eigenvalues[6:] > 1.0e-8)


def test_sample_normal_modes_is_reproducible_and_uses_mode7_default():
    atoms = _build_atoms()
    normal_modes = compute_normal_modes(atoms, np.eye(9))

    result_a = sample_normal_modes(atoms, normal_modes, n_samples=3, random_seed=7)
    result_b = sample_normal_modes(atoms, normal_modes, n_samples=3, random_seed=7)

    assert result_a.mode_indices == (7, 8, 9)
    assert len(result_a.samples) == 3
    assert result_a.amplitudes_mw.shape == (3, 3)
    np.testing.assert_allclose(result_a.amplitudes_mw, result_b.amplitudes_mw)
    for sample_a, sample_b in zip(result_a.samples, result_b.samples):
        np.testing.assert_allclose(sample_a.get_positions(), sample_b.get_positions())


def test_sample_normal_modes_rejects_nonpositive_modes():
    atoms = _build_atoms()
    normal_modes = compute_normal_modes(atoms, np.eye(9))

    with pytest.raises(ValueError, match="positive eigenvalues"):
        sample_normal_modes(atoms, normal_modes, n_samples=1, mode_indices=[1], random_seed=1)


def test_sample_normal_modes_classical_sigma_matches_formula():
    atoms = _build_atoms()
    normal_modes = compute_normal_modes(atoms, np.eye(9))
    eigenvalue = normal_modes.eigenvalues[6]
    sigma = np.sqrt(350.0 * KELVIN_TO_HARTREE / eigenvalue)
    z = np.random.default_rng(11).normal(loc=0.0, scale=1.0, size=(2, 1))

    result = sample_normal_modes(
        atoms,
        normal_modes,
        n_samples=2,
        temperature=350.0,
        mode_indices=[7],
        random_seed=11,
        distribution="classical",
    )

    np.testing.assert_allclose(result.amplitudes_mw, z * sigma)


def test_sample_normal_modes_wigner_sigma_matches_formula():
    atoms = _build_atoms()
    normal_modes = compute_normal_modes(atoms, np.eye(9))
    omega = normal_modes.angular_frequencies_au[6]
    sigma = np.sqrt((BOHR_TO_ANGSTROM**2) / (2.0 * AMU_TO_AU * omega))
    z = np.random.default_rng(13).normal(loc=0.0, scale=1.0, size=(2, 1))

    result = sample_normal_modes(
        atoms,
        normal_modes,
        n_samples=2,
        temperature=0.0,
        mode_indices=[7],
        random_seed=13,
        distribution="wigner",
    )

    np.testing.assert_allclose(result.amplitudes_mw, z * sigma)
