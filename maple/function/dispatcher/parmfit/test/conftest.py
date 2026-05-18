from __future__ import annotations

import sys
import types

import numpy as np


try:
    import ase  # noqa: F401
except ModuleNotFoundError:
    ase_stub = types.ModuleType("ase")
    ase_stub.__path__ = []  # type: ignore[attr-defined]
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


class FixInternals:  # pragma: no cover - import stub only
    def __init__(self, bonds=None, angles_deg=None, dihedrals_deg=None):
        self.bonds = bonds
        self.angles_deg = angles_deg
        self.dihedrals_deg = dihedrals_deg


class NeighborList:  # pragma: no cover - import stub only
    def __init__(self, cutoffs, self_interaction=False, bothways=True):
        del cutoffs, self_interaction, bothways
        self.atoms = None

    def update(self, atoms):
        self.atoms = atoms

    def get_neighbors(self, index):
        del index
        return np.asarray([], dtype=int), np.asarray([], dtype=int)


def natural_cutoffs(atoms):
    return [1.0] * len(atoms)


ase_module = sys.modules.get("ase")
if ase_module is not None and not hasattr(ase_module, "__path__"):
    ase_module.__path__ = []  # type: ignore[attr-defined]

if "ase.constraints" not in sys.modules:
    constraints_stub = types.ModuleType("ase.constraints")
    constraints_stub.FixInternals = FixInternals
    sys.modules["ase.constraints"] = constraints_stub
    if ase_module is not None:
        ase_module.constraints = constraints_stub

if "ase.neighborlist" not in sys.modules:
    neighborlist_stub = types.ModuleType("ase.neighborlist")
    neighborlist_stub.NeighborList = NeighborList
    neighborlist_stub.natural_cutoffs = natural_cutoffs
    sys.modules["ase.neighborlist"] = neighborlist_stub
    if ase_module is not None:
        ase_module.neighborlist = neighborlist_stub
