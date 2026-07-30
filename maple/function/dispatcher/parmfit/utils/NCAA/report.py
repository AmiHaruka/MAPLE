"""Usage: format NCAA workflow status lines."""

from __future__ import annotations

from .artifacts import AtomTypeRow, NCAAArtifacts
from .config import NCAAAbinitioConfig
from .models import NCAAIdentity


def format_ncaa_start_lines(
    *,
    config: NCAAAbinitioConfig,
    identity: NCAAIdentity,
) -> list[str]:
    lines = [
        f"  NCAA target selector: {config.target}\n",
        f"  residue name: {config.rn}\n",
        f"  chirality: {identity.chirality}\n",
        f"  charge/mult: {config.charge} {config.mult}\n",
        f"  protein model: {config.prom}\n",
        f"  bonded refinement: {config.bonded}\n",
        f"  charge fitting: {config.charge_fit.method}\n",
    ]
    if config.charge_fit.method == "resp":
        lines.append(f"  charge level: {config.charge_fit.level}\n")
    return lines


def format_ncaa_final_lines(
    *,
    artifacts: NCAAArtifacts,
    atom_type_rows: list[AtomTypeRow] | None = None,
    stage_timings: list[tuple[str, float]] | None = None,
) -> list[str]:
    del artifacts, atom_type_rows, stage_timings
    return ["  [NCAA] route completed; final summary follows.\n"]
