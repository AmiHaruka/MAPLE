"""
Vibrational frequency and gas-phase thermochemistry (RRHO + GERM–ME).

This module computes molecular vibrational frequencies, normal modes, and
thermochemical properties in the gas phase. It supports both mass-weighted
and non-mass-weighted treatments, fixes common Hessian shape issues, and
produces clean, diagnostic-rich outputs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Tuple

import numpy as np
from ase import Atoms

from maple.function.calculator.electronic_state import validate_electronic_state
from maple.function.timer import timer
from maple.function.utility.active_dof import active_dof_mask
from maple.function.utility.rigid_body import (
    is_linear_geometry,
    mass_weighted_rigid_basis,
    rotational_dof,
)

from ..jobABC import JobABC

# ------------------ optional torch (GPU) ------------------
try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False

# ======================================================================
# Physical constants (SI units)
# ======================================================================
H = 6.62607015e-34                 # Planck constant (J·s)
K_B = 1.380649e-23                 # Boltzmann constant (J/K)
C = 2.99792458e8                   # Speed of light (m/s)
NA = 6.02214076e23                 # Avogadro constant (1/mol)
CM_TO_M = 1e-2                     # cm⁻¹ -> m⁻¹ conversion factor
R_GAS = K_B * NA                   # Ideal gas constant (J/mol/K)
AMU = 1.66053906660e-27            # Atomic mass unit (kg)
ANG2_TO_M2 = 1e-20                 # Å² -> m² conversion factor

# ======================================================================
# Unit conversions (kJ <-> kcal / Hartree)
# ======================================================================
HARTREE_TO_KJ_PER_MOL   = 2625.499638
HARTREE_TO_KCAL_PER_MOL = 627.509474
KJ_PER_MOL_TO_HARTREE   = 1.0 / HARTREE_TO_KJ_PER_MOL
KJ_PER_MOL_TO_KCAL      = 0.23900573614  # 1 kJ/mol = 0.23900573614 kcal/mol

class Units:
    """Small helpers for common unit conversions used in reporting."""

    @staticmethod
    def kj_to_ha(x: float) -> float:
        """Convert kJ/mol to Hartree. Keep this one tight and trustworthy."""
        return x * KJ_PER_MOL_TO_HARTREE

    @staticmethod
    def kj_to_kcal(x: float) -> float:
        """Convert kJ/mol to kcal/mol. Never skimp on clarity in outputs."""
        return x * KJ_PER_MOL_TO_KCAL

    @staticmethod
    def j_to_kcal_per_mol(x_j_per_mol: float) -> float:
        """Convert J/mol to kcal/mol. Simple, explicit, no surprises."""
        return (x_j_per_mol * 1e-3) * KJ_PER_MOL_TO_KCAL  # J -> kJ -> kcal

# ======================================================================
# User parameter surfaces 
# ======================================================================

def _lower_keys(d: Dict) -> Dict:
    """Lowercase dict keys defensively; return empty dict on non-dict input."""
    if not isinstance(d, dict):
        return {}
    return {(k.lower() if isinstance(k, str) else k): v for k, v in d.items()}

def _select_subdict(paras: Dict, name_aliases: tuple[str, ...]) -> Dict:
    """
    Allow {"freq": {...}} / {"frequency": {...}} / {"frequency_analysis": {...}}.
    If no matching sub-dict, fall back to the top-level (case-insensitive).
    """
    if not isinstance(paras, dict):
        return {}
    low = _lower_keys(paras)
    for alias in name_aliases:
        key = alias.lower()
        if key in low and isinstance(low[key], dict):
            return low[key]
    return low

def _update_dataclass_from_dict(dc_obj, d: Dict):
    """Patch a dataclass instance from dict, ignoring unknown keys gracefully."""
    if not isinstance(d, dict):
        return dc_obj
    low = _lower_keys(d)
    fld_names = {f.name.lower(): f.name for f in fields(dc_obj)}
    for k_low, v in low.items():
        if k_low in fld_names:
            setattr(dc_obj, fld_names[k_low], v)
    return dc_obj

@dataclass
class FrequencyParams:
    """User-facing frequency-analysis parameters."""
    method: str = "mw"
    temperature: float = 298.15
    ilowfreq: int = 2
    verbose: int = 1
    treat_imag_as_real: bool = False
    pressure_kpa: float = 101.325
    symmetry_number: int = 1
    thermochemistry: Literal["auto", "gas", "none"] = "auto"
    device: str = "cpu"

@dataclass
class PrintParams:
    """
    Printing/display layer (advanced). These are not required for normal users,
    but help produce concise or verbose reports as needed.
    """
    n_freqs_to_print: int = 10           # Print the first N frequencies (sorted)
    sort_ascending: bool = True          # True: sort ascending; False: keep order
    imag_tol_cm1: float = 10.0           # threshold for treating tiny imag. freqs

# ======================================================================
# Thermochemistry result container 
# ======================================================================
@dataclass
class ThermoResults:
    """
    Container for gas-phase thermochemistry results.

    Attributes:
        zpe_kjmol: Zero-point vibrational energy (kJ/mol).
        h_trans_kjmol: Translational enthalpy contribution (kJ/mol).
        h_rot_kjmol: Rotational enthalpy contribution (kJ/mol).
        h_vib_thermal_kjmol: Thermal vibrational enthalpy (kJ/mol).
        s_trans_jmolK: Translational entropy (J/mol/K).
        s_rot_jmolK: Rotational entropy (J/mol/K).
        s_vib_jmolK: Vibrational entropy (J/mol/K).
        s_elec_jmolK: Electronic degeneracy entropy (J/mol/K).
        g_correction_kjmol: Gibbs free-energy correction (kJ/mol).
    """

    zpe_kjmol: float
    h_trans_kjmol: float
    h_rot_kjmol: float
    h_vib_thermal_kjmol: float
    s_trans_jmolK: float = 0.0
    s_rot_jmolK: float = 0.0
    s_vib_jmolK: float = 0.0
    s_elec_jmolK: float = 0.0
    g_correction_kjmol: float = 0.0

    @property
    def h_total_kjmol(self) -> float:
        """
        Total enthalpy correction at the working temperature.

        Returns:
            float: h_trans + h_rot + (ZPE + h_vib_thermal) in kJ/mol.
        """
        return self.h_trans_kjmol + self.h_rot_kjmol + (self.zpe_kjmol + self.h_vib_thermal_kjmol)

    @property
    def s_total_jmolK(self) -> float:
        """
        Total entropy at the working temperature.

        Returns:
            float: s_trans + s_rot + s_vib + s_elec in J/mol/K.
        """
        return (
            self.s_trans_jmolK
            + self.s_rot_jmolK
            + self.s_vib_jmolK
            + self.s_elec_jmolK
        )

# ======================================================================
# Frequency-analysis base
# ======================================================================
class FrequencyBase(JobABC):
    """
    Base class for frequency analysis with Hessian repair and thermochemistry.

    It provides a uniform workflow to:
      1) obtain and fix the Cartesian Hessian,
      2) compute vibrational frequencies and modes,
      3) evaluate RRHO and low-frequency-corrected thermochemistry.
    """

    def __init__(
        self,
        output: str,
        atoms: Atoms,
        *,
        temperature: float = 298.15,
        pressure_kpa: float = 101.325,   # *** kPa ***
        symmetry_number: int = 1,
        ilowfreq: int = 2,
        omega0_cm1: float = 100.0,
        nu_floor_cm1: float = 1.0,
        thermochemistry: Literal["auto", "gas", "none"] = "auto",
        device: str = "cpu",            
    ):
        """
        Initialize the frequency analysis job.

        Args:
            output: Path to the output text file.
            atoms: ASE Atoms object holding geometry and masses.
            temperature: Working temperature (K).
            pressure_kpa: Working pressure (kPa).
            symmetry_number: Rotational symmetry number σ.
            ilowfreq: Low-frequency treatment selector:
                0=RRHO (harmonic), 1=Truhlar (S-only cap),
                2=Grimme (S interpolation), 3=Minenkov/MRRHO (S+U interpolation).
            omega0_cm1: Characteristic frequency (cm⁻¹) for low-frequency treatments.
            nu_floor_cm1: Minimal frequency magnitude (cm⁻¹) to avoid singularities.
            thermochemistry: ``auto`` for applicable molecular MW analyses,
                ``gas`` to require it, or ``none`` to disable it.

        Notes:
            The attributes `alpha` (default 4) and `Bav_amuA2` (optional) may be
            referenced by some low-frequency models if present on `self`.
        """
        super().__init__(output)
        self.atoms = atoms
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(pressure_kpa) or pressure_kpa <= 0:
            raise ValueError("pressure_kpa must be finite and positive")
        self.temperature = temperature
        self.pressure_kpa = pressure_kpa  # store in kPa
        self.symmetry_number = symmetry_number
        if isinstance(ilowfreq, bool) or ilowfreq not in (0, 1, 2, 3):
            raise ValueError("ilowfreq must be one of: 0, 1, 2, 3")
        self.ilowfreq = ilowfreq
        self.omega0_cm1 = omega0_cm1
        self.nu_floor_cm1 = max(float(nu_floor_cm1), 1e-6)
        self.thermochemistry = str(thermochemistry).lower()
        if self.thermochemistry not in {"auto", "gas", "none"}:
            raise ValueError("thermochemistry must be one of: auto, gas, none")
        if (
            isinstance(symmetry_number, bool)
            or int(symmetry_number) != symmetry_number
            or symmetry_number < 1
        ):
            raise ValueError("symmetry_number must be a positive integer")
        self.symmetry_number = int(symmetry_number)

       # ---- Parse device string: support cpu / auto / gpu / gpu0 / gpu1 / cuda:0, etc. ----
        self.device = self._parse_device(device)

        # User/print-surface injection points (set by Frequency)
        self.verbosity: int = 1
        self.treat_imag_as_real: bool = False
        self._print = PrintParams()

    # ---------------------- main workflow ----------------------
    def run(self) -> None:
        """
        Execute the full frequency analysis workflow.

        Steps:
            1) Fetch Hessian and fix shape.
            2) Diagonalize to obtain frequencies and modes.
            3) Compute thermochemistry.
            4) Write all results to the output file.
            5) If verbosity=10, also write a summary file with XYZ trajectory.
        """
        self.log_info([f"Starting frequency analysis calculation, Number of atoms: {len(self.atoms)}"])

        try:
            hessian = self.get_hessian()
            freqs_cm1, modes_cart, thermo = self._analyze_hessian(hessian)

            self._write_output(freqs_cm1, modes_cart, thermo)
            self.log_info(["Frequency analysis completed", f"Output file: {self.output}"])

            # verbosity=10: additional summary output with XYZ trajectory
            if self.verbosity == 10:
                self._write_summary(freqs_cm1, modes_cart, thermo)

        except Exception as e:
            self.log_error(f"Frequency analysis failed: {str(e)}")
            raise

    # ---------------------- helpers ----------------------
    def _analyze_hessian(
        self,
        hessian: np.ndarray,
        solver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        *,
        mass_weighted: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, ThermoResults | None]:
        solver = solver or self.compute_frequencies
        freqs, modes = solver(hessian)
        if self.treat_imag_as_real:
            tol = float(getattr(self._print, "imag_tol_cm1", 10.0))
            freqs = np.where(freqs < -tol, freqs, np.abs(freqs))

        is_mw = getattr(self, "mass_weighted", True) if mass_weighted is None else mass_weighted
        if not is_mw:
            reason = "non-mass-weighted frequencies are diagnostic only"
            if self.thermochemistry == "gas" and not getattr(self, "mass_weighted", True):
                raise ValueError(f"gas-phase thermochemistry is unavailable: {reason}")
            self._thermochemistry_skip_reason = reason
            return freqs, modes, None
        thermo = self.compute_thermo(freqs) if self._thermochemistry_enabled() else None
        return freqs, modes, thermo

    def _thermochemistry_enabled(self) -> bool:
        if self.thermochemistry == "none":
            return False

        reason = None
        if not getattr(self, "mass_weighted", True):
            reason = "non-mass-weighted frequencies are diagnostic only"
        elif np.any(self.atoms.get_pbc()):
            reason = "periodic frequencies do not define ideal-gas molecular thermochemistry"
        elif self.atoms.constraints:
            reason = "partial-Hessian frequencies do not define whole-molecule gas thermochemistry"

        if reason is None:
            return True
        if self.thermochemistry == "gas":
            raise ValueError(f"gas-phase thermochemistry is unavailable: {reason}")
        self._thermochemistry_skip_reason = reason
        return False

    def _parse_device(self, s: str) -> str:
        """
        - 'cpu' → Use CPU
        - 'auto' / 'gpu' → Use cuda:0 if CUDA is available, otherwise use CPU
        - 'gpu0' / 'gpu1' ... → Map to cuda:<index>
        - 'cuda:0' / 'cuda:1' ... → Direct pass-through
    """
        s = (s or "cpu").strip().lower()
        if s in ("cpu",):
            return "cpu"
        if s in ("gpu", "cuda", "auto"):
            if _TORCH_OK and torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        if s.startswith("gpu") and s[3:].isdigit():
            idx = s[3:]
            if _TORCH_OK and torch.cuda.is_available():
                return f"cuda:{idx}"
            return "cpu"
        if s.startswith("cuda:"):
            return s if (_TORCH_OK and torch.cuda.is_available()) else "cpu"
        return "cpu"

    def _eigh(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified eigenvalue decomposition interface:
        Uses torch.linalg.eigh when running on GPU, and numpy.linalg.eigh on CPU.
        All outputs are returned as NumPy arrays.      
        """
        if self.device.startswith("cuda") and _TORCH_OK and torch.cuda.is_available():
            t = torch.tensor(mat, dtype=torch.float64, device=self.device)
            evals, evecs = torch.linalg.eigh(t)
            return evals.detach().cpu().numpy(), evecs.detach().cpu().numpy()
        else:
            return np.linalg.eigh(np.asarray(mat))

    def get_hessian(self) -> np.ndarray:
        """
        Retrieve the Cartesian Hessian and fix common shape issues.

        Returns:
            np.ndarray: Square Hessian with shape (3N, 3N).

        Raises:
            RuntimeError: If the calculator lacks `get_hessian`.
            ValueError: If the Hessian shape does not match (3N, 3N).
        """
        calc = self.atoms.calc
        if calc is None or not hasattr(calc, "get_hessian"):
            raise RuntimeError("Atom calculator must implement get_hessian method")

        hessian = calc.get_hessian(self.atoms)

        # Convert to numpy array defensively.
        if _TORCH_OK and isinstance(hessian, torch.Tensor):
            hessian = hessian.detach().cpu().numpy()
        else:
            try:
                hessian = hessian.numpy()
            except Exception:
                hessian = np.asarray(hessian)

        # Collapse leading singleton batch: (1, 3N, 3N) -> (3N, 3N).
        if hessian.ndim == 3 and hessian.shape[0] == 1:
            hessian = hessian[0]

        # Validate final shape.
        n_atoms = len(self.atoms)
        expected_shape = (3 * n_atoms, 3 * n_atoms)

        if hessian.shape != expected_shape:
            raise ValueError(f"Hessian shape{hessian.shape}does not match expected{expected_shape}")

        return hessian

    def compute_frequencies(self, hessian_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vibrational frequencies and normal modes.

        Args:
            hessian_matrix: Cartesian Hessian (3N, 3N).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies_cm1, normal_modes_cart).

        Notes:
            Subclasses must implement the diagonalization strategy.
        """
     
        raise NotImplementedError("Subclasses must implement compute_frequencies().")

    def _project_hessian(self, hessian: np.ndarray) -> np.ndarray:
        """
        Project rigid translations and rotations from a mass-weighted Hessian.
        
        Principle:
            H_proj = P^T @ H @ P
            where P = I - Q @ Q^T, Q is orthonormal translation-rotation basis
        
        Args:
            hessian: Mass-weighted Hessian (3N, 3N) in Hartree/(Angstrom² amu)
        
        Returns:
            Projected Hessian (3N, 3N)
        """
        Q = self._rigid_basis()

        n_dof = len(hessian)
        P = np.eye(n_dof) - Q @ Q.T
        H_projected = P @ hessian @ P
        H_projected = 0.5 * (H_projected + H_projected.T)
        
        return H_projected

    def _rigid_basis(self) -> np.ndarray:
        return mass_weighted_rigid_basis(self.atoms)

    @staticmethod
    def _sort_modes(
        frequencies: np.ndarray,
        modes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        zero = np.where(frequencies == 0.0)[0]
        imaginary = np.where(frequencies < 0.0)[0]
        real = np.where(frequencies > 0.0)[0]
        order = np.concatenate(
            [
                zero[np.argsort(np.abs(frequencies[zero]))],
                imaginary[np.argsort(frequencies[imaginary])],
                real[np.argsort(frequencies[real])],
            ]
        )
        return frequencies[order], modes[order]

    def _compute_mass_weighted_frequencies(
        self, hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        masses = np.asarray(self.atoms.get_masses())
        coordinate_masses = np.repeat(masses, 3)
        active = active_dof_mask(self.atoms)
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            return np.empty(0), np.empty((0, 3 * len(self.atoms)))

        h_active = hessian[np.ix_(active_indices, active_indices)]
        inv_sqrt_m = 1.0 / np.sqrt(coordinate_masses[active_indices])
        h_mw = h_active * inv_sqrt_m[None, :] * inv_sqrt_m[:, None]
        if np.all(active):
            h_mw = self._project_hessian(h_mw)
            rigid_mode_count = self._rigid_basis().shape[1]
        else:
            rigid_mode_count = 0

        eigenvalues, eigenvectors = self._eigh(h_mw)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * 2721.1383
        if rigid_mode_count:
            rigid_indices = np.argsort(np.abs(eigenvalues))[:rigid_mode_count]
            frequencies[rigid_indices] = 0.0

        modes = np.zeros((len(eigenvalues), 3 * len(self.atoms)))
        modes[:, active_indices] = eigenvectors.T * inv_sqrt_m[None, :]
        for mode in modes:
            norm = np.sqrt(np.sum(mode**2 * coordinate_masses))
            if norm > 1e-10:
                mode /= norm

        frequencies, modes = self._sort_modes(frequencies, modes)
        if self.verbosity >= 1:
            self.log_info([
                "\nFrequency analysis summary:\n",
                f"  Projected/zero modes: {np.count_nonzero(frequencies == 0.0)}\n",
                f"  Imaginary modes:      {np.count_nonzero(frequencies < 0.0)}\n",
                f"  Real modes:           {np.count_nonzero(frequencies > 0.0)}\n\n",
            ])
        return frequencies, modes

    def _compute_non_mass_weighted_frequencies(
        self, hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        active_indices = np.flatnonzero(active_dof_mask(self.atoms))
        if not len(active_indices):
            return np.empty(0), np.empty((0, 3 * len(self.atoms)))

        eigenvalues, eigenvectors = self._eigh(
            hessian[np.ix_(active_indices, active_indices)]
        )
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * 2721.1383
        modes = np.zeros((len(eigenvalues), 3 * len(self.atoms)))
        modes[:, active_indices] = eigenvectors.T
        return self._sort_modes(frequencies, modes)

    def compute_thermo(self, frequencies_cm1: np.ndarray) -> ThermoResults:
        """
        Compute gas-phase thermochemical properties.
        Only real vibrational modes are used (skip zeros and imaginary).
        """
        if self.atoms.calc is not None:
            validate_electronic_state(self.atoms, self.atoms.calc)
        T = self.temperature
        multiplicity = self.atoms.info.get("mult", 1)
        if (
            isinstance(multiplicity, bool)
            or int(multiplicity) != multiplicity
            or multiplicity < 1
        ):
            raise ValueError("atoms.info['mult'] must be a positive integer")
        multiplicity = int(multiplicity)

        # Projected rigid modes are represented by exact zeros.
        vib_mask = frequencies_cm1 > 0.0
        vib_freqs = frequencies_cm1[vib_mask]
        
        # Zero-point energy from all real modes
        zpe_j_per_mol = 0.5 * H * C * NA * np.sum(vib_freqs) / CM_TO_M
        zpe_kjmol = zpe_j_per_mol * 1e-3
        
        # Apply frequency floor
        nu_eff = np.maximum(vib_freqs, self.nu_floor_cm1)
        theta_v = (H * C * (nu_eff / CM_TO_M)) / K_B
        
        s_vib_total, u_vib_total_J = 0.0, 0.0
        
        for nu_e, theta_i in zip(nu_eff, theta_v):
            x = theta_i / max(T, 1e-12)
            
            # Harmonic entropy and internal energy
            s_harmonic = self._harmonic_entropy(nu_e, T)
            u_harmonic_J = R_GAS * T * (x / (np.exp(x) - 1 + 1e-300))
            
            if self.ilowfreq == 0:
                s_mode = s_harmonic
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 1:
                s_mode = self._truhlar_entropy(nu_e, T, s_harmonic)
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 2:
                s_mode = self._grimme_entropy(nu_e, T, s_harmonic)
                u_mode_J = u_harmonic_J
            elif self.ilowfreq == 3:
                w = self._headgordon_weight(nu_e, getattr(self, "omega0_cm1", 100.0), getattr(self, "alpha", 4))
                s_rotor = self._free_rotor_entropy(nu_e, T, getattr(self, "Bav_amuA2", None))
                s_mode = w * s_harmonic + (1.0 - w) * s_rotor
                R_local = 8.314462618
                u_rotor_J = 0.5 * R_local * T
                u_mode_J = w * u_harmonic_J + (1.0 - w) * u_rotor_J
            else:
                raise ValueError("ilowfreq must be one of: 0, 1, 2, 3")
            
            s_vib_total += s_mode
            u_vib_total_J += u_mode_J
        
        # Thermal contributions
        h_vib_thermal_kjmol = u_vib_total_J * 1e-3
        h_trans_kjmol = 2.5 * R_GAS * T * 1e-3
        
        rot_dof = rotational_dof(self.atoms)
        h_rot_kjmol = (rot_dof / 2) * R_GAS * T * 1e-3
        
        # Entropies
        s_trans, s_rot = self._trans_rot_entropy()
        s_vib = s_vib_total
        s_elec = R_GAS * np.log(multiplicity)
        
        thermo = ThermoResults(
            zpe_kjmol=zpe_kjmol,
            h_trans_kjmol=h_trans_kjmol,
            h_rot_kjmol=h_rot_kjmol,
            h_vib_thermal_kjmol=h_vib_thermal_kjmol,
            s_trans_jmolK=s_trans,
            s_rot_jmolK=s_rot,
            s_vib_jmolK=s_vib,
            s_elec_jmolK=s_elec,
        )
        
        thermo.g_correction_kjmol = thermo.h_total_kjmol - T * thermo.s_total_jmolK * 1e-3
        
        return thermo

    @staticmethod
    def _harmonic_entropy(nu_cm1: float, T: float) -> float:
        theta = (H * C * (float(nu_cm1) / CM_TO_M)) / K_B
        x = theta / T
        return R_GAS * (
            x / np.expm1(x) - np.log1p(-np.exp(-x))
        )

    def _truhlar_entropy(self, nu_cm1: float, T: float, s_harmonic: float) -> float:
        """
        Entropy with low-frequency capping in the harmonic formula.

        Args:
            nu_cm1: Mode frequency (cm⁻¹).
            T: Temperature (K).
            s_harmonic: Harmonic entropy of the mode (J/mol/K).

        Returns:
            float: Entropy of the mode after low-frequency capping (J/mol/K).
        """
        threshold = self.omega0_cm1
        nu_effective = max(nu_cm1, threshold)
        theta_eff = (H * C * (nu_effective / CM_TO_M)) / K_B
        x_eff = theta_eff / max(T, 1e-12)
        return R_GAS * (x_eff / (np.exp(x_eff) - 1 + 1e-300) - np.log(1 - np.exp(-x_eff) + 1e-300))

    def _grimme_entropy(self, nu_cm1: float, T: float, s_harmonic: float) -> float:
        """
        Entropy interpolation between harmonic oscillator and free-rotor.

        Args:
            nu_cm1: Mode frequency (cm⁻¹).
            T: Temperature (K).
            s_harmonic: Harmonic entropy of the mode (J/mol/K).

        Returns:
            float: Interpolated entropy (J/mol/K).
        """
        w = self._weight_w(nu_cm1)
        s_rotor = self._free_rotor_entropy(nu_cm1, T, Bav_amuA2=self._grimme_bav_amuA2())
        return w * s_harmonic + (1.0 - w) * s_rotor

    def _weight_w(self, nu_cm1: float) -> float:
        """
        Damping weight for entropy interpolation (alpha=4 by construction here).

        Args:
            nu_cm1: Mode frequency (cm⁻¹).

        Returns:
            float: Weight w in [0, 1].
        """
        nu = max(float(nu_cm1), 1e-12)
        w0 = max(self.omega0_cm1, 1e-12)
        return 1.0 / (1.0 + (w0 / nu) ** 4)

    @staticmethod
    def _grimme_bav_amuA2() -> float:
        """Grimme's 1e-44 kg m2 average inertia in amu Angstrom2."""
        return 1e-44 / (AMU * ANG2_TO_M2)

    # Low-frequency helpers
    def _headgordon_weight(self, nu_cm1, omega0_cm1=100.0, alpha=4):
        """
        Smooth damping weight: w = 1 / (1 + (omega0/nu)^alpha).

        Args:
            nu_cm1: Scalar or array-like of frequencies (cm⁻¹).
            omega0_cm1: Characteristic frequency (cm⁻¹).
            alpha: Damping exponent (dimensionless).

        Returns:
            np.ndarray or float: Weight(s) w in [0, 1] matching the input shape.
        """
        nu = np.asarray(nu_cm1, dtype=float)
        nu = np.where(nu <= 0.0, 1e-12, nu)
        return 1.0 / (1.0 + (omega0_cm1 / nu) ** float(alpha))

    def _free_rotor_entropy(self, nu_cm1, T, Bav_amuA2=None):
        """
        Free-rotor entropy per mode for interpolation (J/mol/K).

        Args:
            nu_cm1: Scalar or array-like frequencies (cm⁻¹).
            T: Temperature (K).
            Bav_amuA2: Optional average moment of inertia in amu·Å² for an
                effective reduced inertia; if None, uses a direct mapping.

        Returns:
            np.ndarray or float: Free-rotor entropy value(s) (J/mol/K).
        """
        nu_hz = np.asarray(nu_cm1, dtype=float) * C / CM_TO_M
        pi = np.pi
        mu = H / (8.0 * pi * pi * np.where(nu_hz <= 0.0, 1e-12, nu_hz))

        if Bav_amuA2 is not None:
            Bav = Bav_amuA2 * AMU * ANG2_TO_M2
            mu_eff = (mu * Bav) / (mu + Bav)
        else:
            mu_eff = mu

        term = ((8.0 * (pi**3) * mu_eff * K_B * T) / (H * H)) ** 0.5
        return R_GAS * (0.5 + np.log(term))

    def _trans_rot_entropy(self) -> Tuple[float, float]:
        """
        Translational and rotational entropy for a nonlinear/linear molecule.

        Returns:
            Tuple[float, float]: (S_trans, S_rot) in J/mol/K.
        """
        T, P_kpa = self.temperature, self.pressure_kpa
        P = P_kpa * 1000.0  # convert kPa -> Pa for ideal-gas formulas

        I_SI, m_tot = self._principal_moments()
        sigma = max(self.symmetry_number, 1)

        # Translation (ideal gas).
        q_trans = ((2 * np.pi * m_tot * K_B * T) / (H**2)) ** 1.5 * (K_B * T / P)
        s_trans = R_GAS * (np.log(q_trans) + 2.5)

        rot_dof = rotational_dof(self.atoms)
        if rot_dof == 0:
            return s_trans, 0.0

        # Rotation (linear vs nonlinear).
        if rot_dof == 2:
            I = np.max(I_SI)
            q_rot = (8 * np.pi**2 * I * K_B * T) / (sigma * H**2)
            s_rot = R_GAS * (np.log(q_rot) + 1)
        else:
            q_rot = (np.sqrt(np.pi) / sigma) * ((8 * np.pi**2 * K_B * T) / (H**2)) ** 1.5 * np.sqrt(np.prod(I_SI))
            s_rot = R_GAS * (np.log(q_rot) + 1.5)

        return s_trans, s_rot

    def _principal_moments(self) -> Tuple[np.ndarray, float]:
        """
        Principal moments of inertia (SI) and total mass (kg).

        Returns:
            Tuple[np.ndarray, float]: (I_x,I_y,I_z) in kg·m² and total mass (kg).
        """
        I_amuA2 = np.asarray(self.atoms.get_moments_of_inertia(vectors=False))
        I_SI = I_amuA2 * AMU * ANG2_TO_M2
        m_tot = np.sum(self.atoms.get_masses()) * AMU
        return I_SI, m_tot

    def _is_linear_molecule(self) -> bool:
        return is_linear_geometry(self.atoms)

    # ---------------------- writers ----------------------
    def _write_output(
        self,
        freqs: np.ndarray,
        modes: np.ndarray,
        thermo: ThermoResults | None,
    ) -> None:
        """
        Write full analysis report in ORCA-style order.
        Order: Header → Frequencies → Thermodynamics → Normal Modes
        Uses log_info to append to existing output file.
        """
        
        # 1. Frequencies
        self._write_frequencies(freqs)
        
        # 2. Thermodynamics
        if thermo is None:
            reason = getattr(self, "_thermochemistry_skip_reason", "disabled by request")
            self.log_info([f"Thermochemistry omitted: {reason}.\n\n"])
        else:
            self._write_thermochemistry(thermo)
        
        # 3. Normal modes (with zeros masked)
        self._write_normal_modes(freqs, modes)

    def _write_thermochemistry(self, thermo: ThermoResults) -> None:
        """
        Print thermochemistry summary in ORCA-inspired format.
        """
        T = self.temperature
        
        # Convert units
        zpe_kcal = Units.kj_to_kcal(thermo.zpe_kjmol)
        h_trans_kcal = Units.kj_to_kcal(thermo.h_trans_kjmol)
        h_rot_kcal = Units.kj_to_kcal(thermo.h_rot_kjmol)
        h_vib_kcal = Units.kj_to_kcal(thermo.h_vib_thermal_kjmol)
        h_total_kcal = Units.kj_to_kcal(thermo.h_total_kjmol)
        g_corr_kcal = Units.kj_to_kcal(thermo.g_correction_kjmol)
        
        s_trans_kcalK = thermo.s_trans_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_rot_kcalK = thermo.s_rot_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_vib_kcalK = thermo.s_vib_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_elec_kcalK = thermo.s_elec_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        s_total_kcalK = thermo.s_total_jmolK * 1e-3 * KJ_PER_MOL_TO_KCAL
        
        self.log_info(["\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info([f"THERMOCHEMISTRY AT {T:.2f}K\n"])
        self.log_info(["-" * 60 + "\n\n"])
        
        self.log_info([f"Temperature         ...   {T:.2f} K\n"])
        self.log_info([f"Pressure            ...   {self.pressure_kpa / 101.325:.2f} atm\n"])
        self.log_info([f"Total Mass          ...   {np.sum(self.atoms.get_masses()):.2f} AMU\n\n"])
        
        # Zero point energy and corrections
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["INNER ENERGY\n"])
        self.log_info(["-" * 60 + "\n\n"])
        
        self.log_info([f"Zero point energy                ...   {zpe_kcal:10.2f} kcal/mol\n"])
        self.log_info([f"Thermal vibrational correction   ...   {h_vib_kcal:10.2f} kcal/mol\n"])
        self.log_info([f"Thermal rotational correction    ...   {h_rot_kcal:10.2f} kcal/mol\n"])
        self.log_info([f"Thermal translational correction ...   {h_trans_kcal:10.2f} kcal/mol\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info([f"Total thermal correction         ...   {h_total_kcal:10.2f} kcal/mol\n\n"])
        
        # Entropy
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["ENTROPY\n"])
        self.log_info(["-" * 60 + "\n\n"])
        
        self.log_info([f"Translational entropy            ...   {s_trans_kcalK:10.6f} kcal/(mol*K)\n"])
        self.log_info([f"Rotational entropy               ...   {s_rot_kcalK:10.6f} kcal/(mol*K)\n"])
        self.log_info([f"Vibrational entropy              ...   {s_vib_kcalK:10.6f} kcal/(mol*K)\n"])
        self.log_info([f"Electronic entropy               ...   {s_elec_kcalK:10.6f} kcal/(mol*K)\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info([f"Total entropy                    ...   {s_total_kcalK:10.6f} kcal/(mol*K)\n\n"])
        
        # Gibbs free energy
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["GIBBS FREE ENERGY\n"])
        self.log_info(["-" * 60 + "\n\n"])
        self.log_info([f"Total enthalpy correction        ...   {h_total_kcal:10.2f} kcal/mol\n"])
        self.log_info([f"Total entropy correction         ...   {-T * s_total_kcalK:10.2f} kcal/mol\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info([f"Final Gibbs free energy corr.    ...   {g_corr_kcal:10.2f} kcal/mol\n\n"])

    def _write_frequencies(self, freqs: np.ndarray) -> None:
        """
        Print vibrational frequencies in ORCA style.
        Output control:
            - Always show all zero frequencies (typically 6)
            - Then show N non-zero frequencies based on verbosity:
            verbosity=0: show 0 non-zero
            verbosity=1: show 10 non-zero (default)
            verbosity>=2: show all non-zero
        """
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["VIBRATIONAL FREQUENCIES\n"])
        self.log_info(["-" * 60 + "\n\n"])
        
        # Count frequency types
        n_zero = int(np.sum(freqs == 0.0))
        n_imag = int(np.sum(freqs < 0.0))
        n_real = len(freqs) - n_zero - n_imag
        
        self.log_info(["Scaling factor for frequencies = 1.000000000 (already applied!)\n\n"])
        
        # Determine how many non-zero frequencies to print
        v = int(getattr(self, "verbosity", 1))
        if v == 0:
            n_nonzero_to_print = 0
        elif v == 1:
            n_nonzero_to_print = 10  # default: 10 non-zero modes
        else:  # v >= 2 (including v == 10)
            n_nonzero_to_print = len(freqs) - n_zero  # all non-zero modes
        
        # Total to print: all zeros + limited non-zeros
        n_to_print = min(n_zero + n_nonzero_to_print, len(freqs))
        
        # Print frequencies in ORCA format
        for i in range(n_to_print):
            freq_val = freqs[i]
            
            # Determine display value and tag
            if freq_val == 0.0:
                display_val = 0.00
                tag = ""
            elif freq_val < 0.0:
                display_val = freq_val
                tag = "  ***imaginary mode***"
            else:
                display_val = freq_val
                tag = ""
            
            self.log_info([f"{i:6d}:  {display_val:10.2f} cm**-1{tag}\n"])
        
        self.log_info(["\n\n"])
        
    def _write_normal_modes(self, freqs: np.ndarray, modes: np.ndarray) -> None:
        """
        Print normal modes in ORCA column format.
        Output control matches frequency output:
            - Always show all zero-frequency modes
            - Then show N non-zero modes based on verbosity
        Zero-frequency modes are displayed as all zeros (ORCA convention).
        """
        v = int(getattr(self, "verbosity", 1))
        if v <= 0:
            return
        
        # Count zero frequencies
        n_zero = int(np.sum(freqs == 0.0))
        
        # Determine how many non-zero modes to print
        if v == 1:
            n_nonzero_to_print = 10  # default: 10 non-zero modes
        else:  # v >= 2 (including v == 10)
            n_nonzero_to_print = len(freqs) - n_zero  # all non-zero modes
        
        # Total to print: all zeros + limited non-zeros
        n_to_print = min(n_zero + n_nonzero_to_print, len(freqs))
        
        if n_to_print <= 0:
            return
        
        self.log_info(["\n"])
        self.log_info(["-" * 60 + "\n"])
        self.log_info(["NORMAL MODES\n"])
        self.log_info(["-" * 60 + "\n\n"])
        self.log_info(["These modes are the Cartesian displacements weighted by the diagonal matrix\n"])
        self.log_info(["M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom\n"])
        self.log_info(["Thus, these vectors are normalized but *not* orthogonal\n\n"])
        
        n_atoms = len(self.atoms)
        n_coords = 3 * n_atoms
        
        # Mask zero-frequency modes (ORCA convention: display as zeros)
        modes_display = modes.copy()
        for i in range(n_to_print):
            if freqs[i] == 0.0:
                modes_display[i, :] = 0.0
        
        # Print in blocks of 6 columns (ORCA style)
        block_size = 6
        for block_start in range(0, n_to_print, block_size):
            block_end = min(block_start + block_size, n_to_print)
            
            # Header line with mode indices
            header = "       "
            for col in range(block_start, block_end):
                header += f"{col:11d}    "
            self.log_info([header + "\n"])
            
            # Print each Cartesian coordinate (3N rows)
            for coord_idx in range(n_coords):
                line = f"{coord_idx:6d}  "
                for mode_idx in range(block_start, block_end):
                    value = modes_display[mode_idx].flatten()[coord_idx]
                    line += f"{value:13.6f}  "
                self.log_info([line + "\n"])
            
            self.log_info(["\n"])

    # ---------------------- verbosity=10 summary writer ----------------------
    def _write_summary(
        self,
        freqs: np.ndarray,
        modes: np.ndarray,
        thermo: ThermoResults | None,
    ) -> None:
        """
        Write a concise summary file (*_summary.out) for verbosity=10.
        
        Contents:
            1. Thermodynamic properties summary (concise)
            2. All non-zero vibrational frequencies
            3. Normal modes in XYZ trajectory format
        
        Projected rigid zero modes are excluded.
        """
        # Determine summary file path
        base, ext = os.path.splitext(self.output)
        summary_path = f"{base}.sum"
        
        nonzero_mask = freqs != 0.0
        nonzero_freqs = freqs[nonzero_mask]
        nonzero_modes = modes[nonzero_mask]
        
        # Get atomic information
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        n_atoms = len(self.atoms)
        T = self.temperature
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            # ==================== Section 1: Thermodynamic Summary ====================
            f.write("=" * 70 + "\n")
            f.write("THERMODYNAMIC SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Temperature:            {T:.2f} K\n")
            f.write(f"Pressure:               {self.pressure_kpa:.3f} kPa ({self.pressure_kpa / 101.325:.3f} atm)\n")
            f.write(f"Total Mass:             {np.sum(self.atoms.get_masses()):.4f} amu\n")
            f.write(f"Number of Atoms:        {n_atoms}\n")
            f.write(f"Number of Vib. Modes:   {len(nonzero_freqs)}\n\n")
            
            if thermo is None:
                reason = getattr(self, "_thermochemistry_skip_reason", "disabled by request")
                f.write(f"Thermochemistry omitted: {reason}.\n\n")
            else:
                zpe_kcal = Units.kj_to_kcal(thermo.zpe_kjmol)
                h_total_kcal = Units.kj_to_kcal(thermo.h_total_kjmol)
                g_corr_kcal = Units.kj_to_kcal(thermo.g_correction_kjmol)
                s_total_calK = thermo.s_total_jmolK / 4.184

                f.write("-" * 40 + "\n")
                f.write("Key Thermodynamic Values\n")
                f.write("-" * 40 + "\n")
                f.write(f"ZPE:                    {thermo.zpe_kjmol:12.4f} kJ/mol  ({zpe_kcal:10.4f} kcal/mol)\n")
                f.write(f"H_corr (total):         {thermo.h_total_kjmol:12.4f} kJ/mol  ({h_total_kcal:10.4f} kcal/mol)\n")
                f.write(f"G_corr (total):         {thermo.g_correction_kjmol:12.4f} kJ/mol  ({g_corr_kcal:10.4f} kcal/mol)\n")
                f.write(f"S_total:                {thermo.s_total_jmolK:12.4f} J/mol/K ({s_total_calK:10.4f} cal/mol/K)\n\n")

                f.write("-" * 40 + "\n")
                f.write("Enthalpy Contributions (kJ/mol)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  H_trans:              {thermo.h_trans_kjmol:12.4f}\n")
                f.write(f"  H_rot:                {thermo.h_rot_kjmol:12.4f}\n")
                f.write(f"  H_vib (thermal):      {thermo.h_vib_thermal_kjmol:12.4f}\n")
                f.write(f"  ZPE:                  {thermo.zpe_kjmol:12.4f}\n\n")

                f.write("-" * 40 + "\n")
                f.write("Entropy Contributions (J/mol/K)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  S_trans:              {thermo.s_trans_jmolK:12.4f}\n")
                f.write(f"  S_rot:                {thermo.s_rot_jmolK:12.4f}\n")
                f.write(f"  S_vib:                {thermo.s_vib_jmolK:12.4f}\n")
                f.write(f"  S_elec:               {thermo.s_elec_jmolK:12.4f}\n\n")
            
            # ==================== Section 2: All Frequencies ====================
            f.write("=" * 70 + "\n")
            f.write("VIBRATIONAL FREQUENCIES (cm^-1)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"{'Mode':>6}  {'Frequency':>12}  {'Type':<15}\n")
            f.write("-" * 40 + "\n")
            
            for i, freq in enumerate(nonzero_freqs):
                mode_type = "imaginary" if freq < 0.0 else "real"
                f.write(f"{i+1:>6}  {freq:>12.2f}  {mode_type:<15}\n")
            
            f.write("\n")
            
            # ==================== Section 3: Normal Modes as XYZ Trajectory ====================
            f.write("=" * 70 + "\n")
            f.write("NORMAL MODES (XYZ Trajectory Format)\n")
            f.write("=" * 70 + "\n")
            f.write("# Each frame represents one vibrational mode\n")
            f.write("# Coordinates show equilibrium position + displacement (scaled for visualization)\n\n")
            
            # Scale factor for visualization (adjustable)
            disp_scale = 1.0  # Can be adjusted for visualization purposes
            
            for mode_idx, (freq, mode) in enumerate(zip(nonzero_freqs, nonzero_modes)):
                # Reshape mode to (N, 3)
                mode_3d = mode.reshape(n_atoms, 3)
                
                # XYZ frame header
                f.write(f"{n_atoms}\n")
                mode_type = "imag" if freq < 0 else "real"
                f.write(f"Mode {mode_idx + 1}: {freq:.2f} cm^-1 ({mode_type})\n")
                
                # Write atomic positions with displacement
                for atom_idx in range(n_atoms):
                    symbol = symbols[atom_idx]
                    # Equilibrium position
                    x0, y0, z0 = positions[atom_idx]
                    # Displacement (eigenvector component)
                    dx, dy, dz = mode_3d[atom_idx] * disp_scale
                    
                    # Write: symbol, equilibrium coords, displacement vector
                    f.write(f"{symbol:2s}  {x0:12.6f}  {y0:12.6f}  {z0:12.6f}  "
                            f"{dx:12.6f}  {dy:12.6f}  {dz:12.6f}\n")
            
            f.write("\n# End of normal modes trajectory\n")
        
        self.log_info([f"Summary file written: {summary_path}\n"])


# ======================================================================
# Concrete implementations
# ======================================================================
class MWFrequency(FrequencyBase):
    """Mass-weighted molecular, periodic, or partial-Hessian analysis."""

    def compute_frequencies(
        self, hessian_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._compute_mass_weighted_frequencies(hessian_matrix)


class NonMWFrequency(FrequencyBase):
    """Diagnostic Cartesian-Hessian eigenanalysis without atomic masses."""

    mass_weighted = False

    def compute_frequencies(
        self, hessian_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._compute_non_mass_weighted_frequencies(hessian_matrix)


class BothFrequency(FrequencyBase):
    """
    Dual-mode frequency analysis.
    
    Produces both mass-weighted and non-mass-weighted results in one run.
    """

    def run(self) -> None:
        self.log_info([f"Starting frequency analysis calculation, Number of atoms: {len(self.atoms)}"])
        try:
            hessian = self.get_hessian()
            mw_freqs, mw_modes, mw_thermo = self._analyze_hessian(
                hessian, self._compute_mw, mass_weighted=True
            )
            nonmw_freqs, nonmw_modes, _ = self._analyze_hessian(
                hessian, self._compute_nonmw, mass_weighted=False
            )
            self._write_combined_output(
                mw_freqs, mw_modes, mw_thermo, nonmw_freqs, nonmw_modes
            )
            if self.verbosity == 10:
                self._write_summary(mw_freqs, mw_modes, mw_thermo)
            self.log_info(["Frequency analysis completed", f"Output file: {self.output}"])
        except Exception as exc:
            self.log_error(f"Frequency analysis failed: {exc}")
            raise

    def _compute_mw(self, hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._compute_mass_weighted_frequencies(hessian)

    def _compute_nonmw(self, hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._compute_non_mass_weighted_frequencies(hessian)

    def _write_combined_output(self, mw_freqs: np.ndarray, mw_modes: np.ndarray,
                               mw_thermo: ThermoResults | None, nonmw_freqs: np.ndarray,
                               nonmw_modes: np.ndarray) -> None:
        """
        Write a combined report for MW and non-MW analyses.

        Args:
            mw_freqs: Mass-weighted frequencies (cm⁻¹).
            mw_modes: Mass-weighted normal modes (Cartesian).
            mw_thermo: Thermochemistry from MW frequencies.
            nonmw_freqs: Non-mass-weighted frequencies (cm⁻¹).
            nonmw_modes: Non-mass-weighted normal modes (Cartesian).
        """
        self.log_info(["\n" + "=" * 60 + "\nMASS-WEIGHTED ANALYSIS\n" + "=" * 60 + "\n"])
        self._write_output(mw_freqs, mw_modes, mw_thermo)
        self.log_info([
            "\n" + "=" * 60 + "\nNON-MASS-WEIGHTED DIAGNOSTIC ANALYSIS\n" + "=" * 60 + "\n",
            "Thermochemistry omitted: non-mass-weighted frequencies are diagnostic only.\n\n",
        ])
        self._write_frequencies(nonmw_freqs)
        self._write_normal_modes(nonmw_freqs, nonmw_modes)

# ======================================================================
# Front driver 
# ======================================================================
class Frequency:
    def __init__(
        self,
        output: str,
        atoms: Atoms,
        params: FrequencyParams | None = None,
        paras: dict | None = None,
    ):
        self.output = output
        self.atoms = atoms
        self.params = params if params is not None else FrequencyParams()
        if isinstance(paras, dict):
            user = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
            _update_dataclass_from_dict(self.params, user)
            if "verbosity" in _lower_keys(user):
                self.params.verbose = _lower_keys(user)["verbosity"]

        # ---- Back-compat: map legacy 'mode' -> 'method' if present ----
        if hasattr(self.params, "method"):
            if isinstance(paras, dict):
                _user_chk = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
                if isinstance(_user_chk, dict) and ("mode" in _user_chk) and ("method" not in _user_chk):
                    self.params.method = str(_user_chk["mode"]).lower()

        # ---- Back-compat: pressure_pa -> pressure_kpa ----
        if isinstance(paras, dict):
            _low = _lower_keys(_select_subdict(paras, ("freq", "frequency", "frequency_analysis")))
            if "pressure_pa" in _low and "pressure_kpa" not in _low:
                try:
                    _pa_val = float(_low["pressure_pa"])
                    self.params.pressure_kpa = _pa_val / 1000.0
                except Exception:
                    pass  # leave for later validation

        # ---- Merge print-layer parameters (advanced) ----
        self.print_params = PrintParams()
        if isinstance(paras, dict):
            _user_print = _select_subdict(paras, ("freq", "frequency", "frequency_analysis"))
            _update_dataclass_from_dict(self.print_params, _user_print)

        # ---- Basic validation ----
        from ...calculator.calculator_base import parse_bool_option

        self.params.treat_imag_as_real = parse_bool_option(
            self.params.treat_imag_as_real, name="freq.treat_imag_as_real"
        )
        self.print_params.sort_ascending = parse_bool_option(
            self.print_params.sort_ascending, name="freq.sort_ascending"
        )
        if not np.isfinite(float(self.params.temperature)) or float(self.params.temperature) <= 0:
            raise ValueError("temperature must be > 0 K")
        if not np.isfinite(float(self.params.pressure_kpa)) or float(self.params.pressure_kpa) <= 0:
            raise ValueError("pressure_kpa must be > 0 kPa")
        if str(self.params.method).lower() not in ("mw", "nonmw", "both"):
            raise ValueError('method must be one of: "mw", "nonmw", "both"')
        if str(self.params.thermochemistry).lower() not in ("auto", "gas", "none"):
            raise ValueError('thermochemistry must be one of: "auto", "gas", "none"')
        sigma = self.params.symmetry_number
        if isinstance(sigma, bool) or int(sigma) != sigma or sigma < 1:
            raise ValueError("symmetry_number must be a positive integer")

     
  

    def run(self) -> None:
        with timer("Frequency Calculation"):
            method = self.params.method.lower()

            common_kwargs: dict[str, Any] = {
                "output": self.output,
                "atoms": self.atoms,
                "temperature": self.params.temperature,
                "pressure_kpa": self.params.pressure_kpa,
                "symmetry_number": self.params.symmetry_number,
                "ilowfreq": self.params.ilowfreq,
                "thermochemistry": self.params.thermochemistry,
                "device": self.params.device,
            }

            if method == "both":
                job = BothFrequency(**common_kwargs)
            elif method == "nonmw":
                job = NonMWFrequency(**common_kwargs)
            else:  # "mw"
                job = MWFrequency(**common_kwargs)
            job.verbosity = int(self.params.verbose)
            job.treat_imag_as_real = self.params.treat_imag_as_real
            # pass print-layer params if available
            if hasattr(self, "print_params"):
                job._print = self.print_params
            job.run()

# Public API
__all__ = [
    'FrequencyBase', 'MWFrequency', 'NonMWFrequency', 'BothFrequency', 'ThermoResults',
    'FrequencyParams', 'PrintParams', 'Frequency'
]
