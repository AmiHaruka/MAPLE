"""Usage: select spectral torsion candidates for shared-group Stage1 refits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TORSIONFIT_CANONICAL_PERIODS

DEFAULT_SPECTRAL_PERIODS = TORSIONFIT_CANONICAL_PERIODS
_AMPLITUDE_FLOOR = 1.0e-8


def _wrap_signed_pi(angle: float) -> float:
    value = ((float(angle) + np.pi) % (2.0 * np.pi)) - np.pi
    return float(np.pi) if np.isclose(abs(value), np.pi, atol=1.0e-12) else float(value)


@dataclass(frozen=True)
class SpectralPeak:
    period: int
    amplitude: float
    phase: float


@dataclass(frozen=True)
class SharedGroupResponse:
    period: int
    coherence: float
    phase_shift: float


@dataclass(frozen=True)
class SharedGroupSpectralSlot:
    label: str
    period: int
    phase_seed: float
    coherence: float
    amplitude: float
    score: float


def dominant_spectral_peaks(
    phi_values: np.ndarray,
    target_rel: np.ndarray,
    *,
    ref_idx: int,
    allowed_periods: tuple[int, ...] = DEFAULT_SPECTRAL_PERIODS,
    amplitude_floor: float = _AMPLITUDE_FLOOR,
) -> tuple[SpectralPeak, ...]:
    """Fit Fourier columns for allowed AMBER periods and rank by amplitude."""
    phi = np.asarray(phi_values, dtype=float).reshape(-1)
    target = np.asarray(target_rel, dtype=float).reshape(-1)
    if phi.shape != target.shape:
        raise ValueError("phi_values and target_rel must have the same shape.")
    if phi.size == 0:
        return ()
    ref = int(ref_idx)
    if ref < 0 or ref >= phi.size:
        raise ValueError("ref_idx is outside the profile.")

    peaks: list[SpectralPeak] = []
    for period in allowed_periods:
        n = int(period)
        if n <= 0:
            continue
        cos_col = np.cos(float(n) * phi) - np.cos(float(n) * phi[ref])
        sin_col = np.sin(float(n) * phi) - np.sin(float(n) * phi[ref])
        matrix = np.column_stack((cos_col, sin_col))
        coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        cos_coeff, sin_coeff = (float(coefficients[0]), float(coefficients[1]))
        amplitude = float(np.hypot(cos_coeff, sin_coeff))
        if amplitude <= float(amplitude_floor):
            continue
        peaks.append(
            SpectralPeak(
                period=n,
                amplitude=amplitude,
                phase=_wrap_signed_pi(float(np.arctan2(sin_coeff, cos_coeff))),
            )
        )
    return tuple(sorted(peaks, key=lambda peak: peak.amplitude, reverse=True))


def shared_group_response(
    path_phi_values: np.ndarray,
    representative_phi_values: np.ndarray,
    *,
    period: int,
) -> SharedGroupResponse:
    path_phi = np.asarray(path_phi_values, dtype=float)
    rep_phi = np.asarray(representative_phi_values, dtype=float).reshape(-1)
    if path_phi.ndim == 1:
        path_phi = path_phi.reshape((-1, 1))
    if path_phi.shape[0] != rep_phi.size:
        raise ValueError("path_phi_values must have one row per representative phi value.")
    if path_phi.size == 0:
        return SharedGroupResponse(period=int(period), coherence=0.0, phase_shift=0.0)

    n = int(period)
    values = np.exp(1j * float(n) * (path_phi - rep_phi[:, np.newaxis]))
    response = np.mean(values)
    return SharedGroupResponse(
        period=n,
        coherence=float(abs(response)),
        phase_shift=_wrap_signed_pi(float(np.angle(response))),
    )


def rank_shared_group_spectral_slots(
    *,
    label: str,
    representative_phi_values: np.ndarray,
    path_phi_values: np.ndarray,
    peaks: tuple[SpectralPeak, ...],
    min_coherence: float = 0.15,
) -> tuple[SharedGroupSpectralSlot, ...]:
    slots: list[SharedGroupSpectralSlot] = []
    for peak in peaks:
        response = shared_group_response(path_phi_values, representative_phi_values, period=peak.period)
        if response.coherence < float(min_coherence):
            continue
        phase_seed = _wrap_signed_pi(float(peak.phase) + float(response.phase_shift))
        slots.append(
            SharedGroupSpectralSlot(
                label=str(label),
                period=int(peak.period),
                phase_seed=phase_seed,
                coherence=float(response.coherence),
                amplitude=float(peak.amplitude),
                score=float(peak.amplitude) * float(response.coherence),
            )
        )
    return tuple(sorted(slots, key=lambda slot: slot.score, reverse=True))
