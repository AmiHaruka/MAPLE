from typing import Dict, List

import numpy as np


def energy_stats(primary_energy: float, observer_energies: List[float]) -> Dict[str, object]:
    """
    Compute ensemble energy statistics in Hartree.
    """
    all_energies = [float(primary_energy)] + [float(e) for e in observer_energies]
    arr = np.asarray(all_energies, dtype=np.float64)

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    rms = float(np.sqrt(np.mean((arr - mean) ** 2)))
    deltas = [float(e - primary_energy) for e in observer_energies]

    return {
        "primary": float(primary_energy),
        "mean": mean,
        "std": std,
        "rms": rms,
        "deltas": deltas,
    }


def force_stats(primary_forces: np.ndarray, observer_forces: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute force uncertainty stats in Hartree/Angstrom.
    """
    all_forces = [np.asarray(primary_forces, dtype=np.float64)]
    for f in observer_forces:
        all_forces.append(np.asarray(f, dtype=np.float64))

    stack = np.stack(all_forces, axis=0)  # (M, N, 3)
    sigma = np.std(stack, axis=0)         # (N, 3)

    return {
        "sigma_rms": float(np.sqrt(np.mean(sigma ** 2))),
        "sigma_max": float(np.max(sigma)),
    }

