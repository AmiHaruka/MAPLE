"""Usage: score torsion scan profiles and energy-weighted residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PROFILE_WEIGHT_FLOOR = 0.20
_PROFILE_WEIGHT_E0 = 2.0
_PROFILE_SCALE_FLOOR = 0.50
_STAGE1_DISCONTINUITY_MIN_JUMP = 3.0
_STAGE1_DISCONTINUITY_PROFILE_FRACTION = 0.30
_STAGE1_DISCONTINUITY_STEP_MULTIPLIER = 3.0


@dataclass(frozen=True)
class ScanProfileQuality:
    quality: str
    max_adjacent_qm_jump: float
    median_adjacent_qm_jump: float
    profile_range: float
    jump_threshold: float
    max_jump: float
    transition_rows: tuple[int, ...] = ()

    @property
    def has_geometry_jump(self) -> bool:
        return self.quality in {"discontinuous", "geometry_jump"}


def _rmse_from_residual(residual: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(residual, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))

def _robust_profile_range(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return 0.0
    if data.size < 8:
        return float(np.max(data) - np.min(data))
    return float(np.percentile(data, 95.0) - np.percentile(data, 5.0))

def _profile_fit_scale(qm_rel: np.ndarray, target_like: np.ndarray | None = None) -> float:
    qm_scale = _robust_profile_range(np.asarray(qm_rel, dtype=float))
    target_scale = _robust_profile_range(np.asarray(target_like, dtype=float)) if target_like is not None else 0.0
    return max(qm_scale, target_scale, _PROFILE_SCALE_FLOOR)

def _profile_loss_metrics(
    qm_rel: np.ndarray,
    mm_rel: np.ndarray,
    *,
    profile_scale: float | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    qm_values = np.asarray(qm_rel, dtype=float)
    mm_values = np.asarray(mm_rel, dtype=float)
    residual = mm_values - qm_values
    scale = float(profile_scale) if profile_scale is not None else _profile_fit_scale(qm_values)
    weight_values = np.asarray(weights, dtype=float) if weights is not None else _scan_energy_weights(qm_values)
    if weight_values.shape != qm_values.shape:
        weight_values = _scan_energy_weights(qm_values)
    weight_sum = float(np.sum(weight_values))
    normalized_residual = residual / max(scale, 1.0e-12)
    data_loss = float(np.sum(weight_values * (normalized_residual**2)) / weight_sum) if weight_sum > 0.0 else 0.0
    weighted_rmse = float(np.sqrt(data_loss)) if data_loss > 0.0 else 0.0
    return {
        "data_loss": float(data_loss),
        "weighted_rmse": float(weighted_rmse),
        "scale": float(scale),
    }

def _profile_scale_from_arrays(qm_rel: np.ndarray, correction_like: np.ndarray | None = None) -> tuple[float, str]:
    qm_scale = _robust_profile_range(np.asarray(qm_rel, dtype=float))
    correction_scale = _robust_profile_range(np.asarray(correction_like, dtype=float)) if correction_like is not None else 0.0
    profile_scale = max(qm_scale, correction_scale, _PROFILE_SCALE_FLOOR)
    if qm_scale < 0.50 and correction_scale < 0.50:
        return profile_scale, "low"
    if profile_scale >= 5.0:
        return profile_scale, "high"
    return profile_scale, "normal"

def _scan_profile_quality(qm_rel: np.ndarray) -> ScanProfileQuality:
    values = np.asarray(qm_rel, dtype=float).reshape(-1)
    if values.size < 2:
        return ScanProfileQuality(
            quality="smooth",
            max_adjacent_qm_jump=0.0,
            median_adjacent_qm_jump=0.0,
            profile_range=0.0,
            jump_threshold=_STAGE1_DISCONTINUITY_MIN_JUMP,
            max_jump=0.0,
        )
    diffs = np.abs(np.diff(values))
    max_index = int(np.argmax(diffs))
    max_jump = float(diffs[max_index])
    median_jump = float(np.median(diffs)) if diffs.size else 0.0
    profile_range = float(np.max(values) - np.min(values))
    fraction_threshold = _STAGE1_DISCONTINUITY_PROFILE_FRACTION * max(profile_range, _PROFILE_SCALE_FLOOR)
    outlier_threshold = _STAGE1_DISCONTINUITY_STEP_MULTIPLIER * max(median_jump, 1.0e-12)
    threshold = max(_STAGE1_DISCONTINUITY_MIN_JUMP, fraction_threshold, outlier_threshold)
    discontinuous = (
        max_jump >= _STAGE1_DISCONTINUITY_MIN_JUMP
        and max_jump >= fraction_threshold
        and max_jump >= outlier_threshold
    )
    transition_rows = ()
    if discontinuous:
        row_start = max(0, max_index - 1)
        row_end = min(values.size - 1, max_index + 2)
        transition_rows = tuple(range(row_start, row_end + 1))
    quality = "discontinuous" if discontinuous else "smooth"
    return ScanProfileQuality(
        quality=quality,
        max_adjacent_qm_jump=float(max_jump),
        median_adjacent_qm_jump=float(median_jump),
        profile_range=float(profile_range),
        jump_threshold=float(threshold),
        max_jump=float(max_jump),
        transition_rows=transition_rows,
    )

def _stable_rows_from_scan_quality(length: int, scan_quality: ScanProfileQuality) -> np.ndarray:
    mask = np.ones(int(length), dtype=bool)
    for row in scan_quality.transition_rows:
        index = int(row)
        if 0 <= index < mask.size:
            mask[index] = False
    if np.count_nonzero(mask) < 2:
        mask[:] = True
    return mask

def _scan_energy_weights(qm_rel: np.ndarray) -> np.ndarray:
    qm_values = np.maximum(np.asarray(qm_rel, dtype=float), 0.0)
    return _PROFILE_WEIGHT_FLOOR + ((1.0 - _PROFILE_WEIGHT_FLOOR) / (1.0 + qm_values / _PROFILE_WEIGHT_E0))
