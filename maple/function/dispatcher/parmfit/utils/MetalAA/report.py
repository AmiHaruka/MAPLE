"""Usage: format MetalAA workflow status lines."""

from __future__ import annotations

from typing import Optional

from .config import MetalAbinitioConfig
from .artifacts import MetalArtifacts, MetalAtomTypeRow


def format_metal_start_lines(
    config: MetalAbinitioConfig,
    *,
    large_charge: int,
    large_mult: int,
) -> list[str]:
    return [
        f"  Target metal selector: {config.target}\n",
        f"  add_resid: {config.add_resid}\n",
        f"  cluster_cutoff: {config.cluster_cutoff:.2f} A\n",
        f"  donor_cutoff: {config.donor_cutoff:.2f} A\n",
        f"  water model: {config.resp.watm}\n",
        f"  ion parameter set: {config.ionm}\n",
        f"  metal site charge/mult: {config.charge} {config.mult}\n",
        f"  metal oxidation: {config.oxy if config.oxy is not None else config.charge}\n",
        f"  large model charge/mult: {large_charge} {large_mult}\n",
        f"  chgmod: {config.resp.chgmod}\n",
        f"  fixchg_resids: {config.resp.fixchg_resids}\n",
        f"  QM method: {config.resp.qm.theory}/{config.resp.qm.basis}\n",
    ]


def format_metal_final_lines(
    *,
    artifacts: MetalArtifacts,
    atom_type_rows: list[MetalAtomTypeRow] | None = None,
    ion_frcmods: list[str] | None = None,
    metal_formal_charge: int | None = None,
    metal_fitted_charge: float | None = None,
    mseminario_warning: Optional[str] = None,
    external_residues: list[str] | None = None,
    stage_timings: list[tuple[str, float]] | None = None,
) -> list[str]:
    del artifacts, atom_type_rows, ion_frcmods, metal_formal_charge
    del metal_fitted_charge, mseminario_warning, external_residues, stage_timings
    return ["  [MetalAA] route completed; final summary follows.\n"]
