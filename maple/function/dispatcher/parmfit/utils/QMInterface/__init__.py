"""External QM reference jobs for ParmFit."""

from .calculator import (
    QMReferenceConfig,
    QMReferenceResult,
)

from .gaussian import (
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR2_TO_HARTREE_PER_ANG2,
    GaussianReferenceRunner,
    build_qm_reference_config,
    build_qm_reference_runner,
    parse_gaussian_fchk_hessian,
    parse_gaussian_force_log,
    parse_gaussian_log,
    write_gaussian_input,
)
from .orca import (
    ORCAReferenceRunner,
    parse_orca_engrad,
    parse_orca_hessian,
    parse_orca_output,
    write_orca_input,
)

__all__ = [
    "BOHR_TO_ANGSTROM",
    "HARTREE_PER_BOHR2_TO_HARTREE_PER_ANG2",
    "GaussianReferenceRunner",
    "ORCAReferenceRunner",
    "QMReferenceConfig",
    "QMReferenceResult",
    "build_qm_reference_config",
    "build_qm_reference_runner",
    "parse_gaussian_fchk_hessian",
    "parse_gaussian_force_log",
    "parse_gaussian_log",
    "parse_orca_engrad",
    "parse_orca_hessian",
    "parse_orca_output",
    "write_orca_input",
    "write_gaussian_input",
]
