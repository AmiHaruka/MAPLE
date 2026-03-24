from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Optional


@dataclass(frozen=True)
class FourierTerm:
    """Single Fourier term for proper or improper torsions."""

    kPhi: float
    period: float
    phase: float

    def __str__(self) -> str:
        return f"<k={self.kPhi:.6f}, n={self.period:.3f}, phase={self.phase * 180.0 / pi:.2f}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Bond:
    """
    Molecular bond instance keyed by MAPLE inp atom indices.
    kBond is in kcal/mol/A^2 and rEq in Angstrom.
    """

    atoms: tuple[int, int]
    atom_types: tuple[str, str]
    kBond: Optional[float] = None
    rEq: Optional[float] = None

    def __str__(self) -> str:
        return f"<{self.atoms}, r={self.rEq}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Angle:
    """
    Molecular angle instance keyed by MAPLE inp atom indices.
    kTheta is in kcal/mol/rad^2 and thetaEq is stored in radians.
    """

    atoms: tuple[int, int, int]
    atom_types: tuple[str, str, str]
    kTheta: Optional[float] = None
    thetaEq: Optional[float] = None

    def __str__(self) -> str:
        angle_deg = None if self.thetaEq is None else self.thetaEq * 180.0 / pi
        return f"<{self.atoms}, ang={angle_deg}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Dihedral:
    """
    Molecular proper torsion instance keyed by MAPLE inp atom indices.
    Each topology dihedral may carry multiple Fourier terms.
    """

    atoms: tuple[int, int, int, int]
    atom_types: tuple[str, str, str, str]
    terms: list[FourierTerm] = field(default_factory=list)

    def __str__(self) -> str:
        return f"<{self.atoms}, n_terms={len(self.terms)}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Improper:
    """
    Molecular improper torsion instance keyed by MAPLE inp atom indices.
    Amber-style matching keeps the center atom in position 3.
    """

    atoms: tuple[int, int, int, int]
    atom_types: tuple[str, str, str, str]
    terms: list[FourierTerm] = field(default_factory=list)

    def __str__(self) -> str:
        return f"<{self.atoms}, n_terms={len(self.terms)}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Nonbond:
    """Per-atom nonbonded term keyed by the MAPLE inp atom index."""

    atom: int
    atom_type: str
    charge: float
    rmin_half: Optional[float] = None
    epsilon: Optional[float] = None

    def __str__(self) -> str:
        return f"<{self.atom}:{self.atom_type}, q={self.charge}>"

    def __repr__(self) -> str:
        return self.__str__()
