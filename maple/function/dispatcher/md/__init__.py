"""
Molecular Dynamics (MD) module for MAPLE.

This module provides classical molecular dynamics simulation capabilities
using machine learning potentials.

Supported ensembles:
    - NVE (microcanonical)
    - NVT (canonical) - future
    - NPT (isothermal-isobaric) - future

Main components:
    - Integrators: Velocity Verlet (symplectic)
    - Thermostats: Langevin, Berendsen - future
    - Barostats: Berendsen, Parrinello-Rahman - future

Author: Claude
Date: 2026-01-30
"""

__version__ = '0.1.0'
__author__ = 'MAPLE Development Team'

# Main MD dispatcher will be imported here when implemented
# from .md import MDDispatcher

# Utilities
from .utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    KELVIN_TO_HARTREE,
    AMU_TO_AU,
    FS_TO_AU
)

__all__ = [
    'calculate_temperature',
    'calculate_kinetic_energy',
    'initialize_velocities',
    'KELVIN_TO_HARTREE',
    'AMU_TO_AU',
    'FS_TO_AU',
]
