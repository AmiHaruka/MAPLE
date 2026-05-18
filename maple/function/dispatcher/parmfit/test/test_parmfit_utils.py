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

from maple.function.dispatcher.parmfit.utils.mSeminario import apply_mseminario
from maple.function.dispatcher.parmfit.utils.readparm import Angle, Bond


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
    assert bonds[0].kBond == pytest.approx(1.0)
    assert bonds[1].kBond == pytest.approx(1.5)
    assert angles[0].kTheta == pytest.approx(10.0 / 9.0)


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
