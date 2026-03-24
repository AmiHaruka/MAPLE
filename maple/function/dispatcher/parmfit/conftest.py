from __future__ import annotations

import sys
import types

import numpy as np


try:
    import ase  # noqa: F401
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
