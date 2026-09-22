"""Two-phase preparation and validation of molecular-dynamics restart state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.cell import Cell

from ...calculator.electronic_state import (
    canonical_identity_json,
    electronic_state_identity,
)
from ...utility.active_dof import active_atom_mask
from .rst_io import constraint_identity, read_rst_bytes
from .utils import normalize_velocity_representation, set_atoms_velocity_representation


@dataclass(frozen=True)
class PreparedMDState:
    atoms: Atoms
    velocities: np.ndarray
    checkpoint: dict
    source: Path
    source_sha256: str
    source_bytes: bytes
    load_state: bool

    @property
    def step_offset(self) -> int:
        return 0 if self.load_state else int(self.checkpoint["step"])

    def with_velocities(
        self, velocities: np.ndarray, representation: str
    ) -> PreparedMDState:
        checkpoint = {**self.checkpoint, "velocity_representation": representation}
        return replace(self, velocities=velocities, checkpoint=checkpoint)


def prepare_restart_candidate(
    atoms: Atoms,
    candidates: list[Path],
    *,
    load_state: bool,
) -> PreparedMDState:
    """Read and restore a checkpoint onto a copy without mutating live state."""
    errors = []
    state = None
    source = None
    source_bytes = None
    for path in candidates:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        try:
            source_bytes = path.read_bytes()
            state = read_rst_bytes(source_bytes, path)
            source = path
            break
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
    if state is None or source is None or source_bytes is None:
        raise RuntimeError("MD restart failed: " + "; ".join(errors))

    if state["natoms"] != len(atoms):
        raise RuntimeError(
            f"Atom count mismatch: rst has {state['natoms']}, input has {len(atoms)}"
        )
    if state["symbols"] != atoms.get_chemical_symbols():
        raise RuntimeError("Element mismatch between RST and input.")
    representation = state["velocity_representation"]
    if representation not in {"standard", "lfmiddle_carried"}:
        raise RuntimeError("Unsupported velocity representation in restart checkpoint.")
    saved_pbc = state["pbc"] if state["pbc"] is not None else [False, False, False]
    if not np.array_equal(np.asarray(saved_pbc), np.asarray(atoms.pbc)):
        raise RuntimeError("PBC mismatch between RST and input.")

    candidate = atoms.copy()
    candidate.calc = atoms.calc
    candidate.set_positions(state["positions"], apply_constraint=False)
    if state["cell"] is None:
        candidate.set_cell(np.zeros((3, 3)), apply_constraint=False)
    elif state["version"] == 2:
        candidate.set_cell(state["cell"], apply_constraint=False)
    else:
        candidate.set_cell(Cell.fromcellpar(state["cell"]), apply_constraint=False)
    candidate.set_pbc(saved_pbc)
    set_atoms_velocity_representation(
        candidate, normalize_velocity_representation(representation)
    )
    return PreparedMDState(
        atoms=candidate,
        velocities=np.asarray(state["velocities"], dtype=float).copy(),
        checkpoint=state,
        source=source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_bytes=source_bytes,
        load_state=load_state,
    )


def validate_prepared_restart(
    prepared: PreparedMDState,
    *,
    ensemble: str,
    timestep: float,
    n_steps: int,
    dynamics_parameters: dict,
) -> bool:
    """Validate the full restart contract; return whether the run is complete."""
    state = prepared.checkpoint
    carried_load = prepared.load_state and state["velocity_representation"] == "lfmiddle_carried"
    if not prepared.load_state or carried_load:
        if state["version"] < 2:
            if prepared.load_state:
                raise RuntimeError(
                    "Legacy LF-Middle carried velocities cannot be rebound safely "
                    "because MAPLE_RST_V1 has no PES identity."
                )
            raise RuntimeError(
                "MAPLE_RST_V1 cannot prove an exact continuation because it lacks "
                "the full cell and state identity. Use load_state=yes to start a new run."
            )
        if canonical_identity_json(state["pes_identity"]) != canonical_identity_json(
            electronic_state_identity(prepared.atoms)
        ):
            message = (
                "LF-Middle carried velocities cannot be loaded onto a different PES."
                if carried_load
                else "Electronic state or PES identity mismatch between RST and input."
            )
            raise RuntimeError(message)
        if not np.array_equal(state["masses"], prepared.atoms.get_masses()):
            raise RuntimeError("Atomic-mass mismatch between RST and input.")
        saved_constraints = json.dumps(
            state["constraints"], sort_keys=True, separators=(",", ":")
        )
        current_constraints = json.dumps(
            constraint_identity(prepared.atoms), sort_keys=True, separators=(",", ":")
        )
        if saved_constraints != current_constraints:
            raise RuntimeError("Constraint mismatch between RST and input.")
        if not prepared.load_state:
            active = active_atom_mask(prepared.atoms)
            if np.any(prepared.velocities[~active] != 0.0):
                raise RuntimeError(
                    "Exact restart checkpoint contains nonzero frozen-atom velocity."
                )

    if prepared.load_state:
        return False
    saved_dynamics = json.dumps(
        state["dynamics_parameters"], sort_keys=True, separators=(",", ":")
    )
    current_dynamics = json.dumps(
        dynamics_parameters, sort_keys=True, separators=(",", ":")
    )
    if saved_dynamics != current_dynamics:
        raise RuntimeError("Dynamics-parameter mismatch between RST and input.")
    thermostat = dynamics_parameters.get("thermostat")
    expected_representation = (
        "lfmiddle_carried"
        if ensemble in {"nvt", "npt"} and thermostat == "langevin"
        else "standard"
    )
    if state["velocity_representation"] != expected_representation:
        raise RuntimeError(
            "Velocity-representation mismatch: exact "
            f"{ensemble.upper()} restart requires {expected_representation}."
        )
    if state["ensemble"] != ensemble:
        raise RuntimeError(
            f"Ensemble mismatch: rst has '{state['ensemble']}', input specifies '{ensemble}'"
        )
    if abs(state["timestep"] - timestep) > 1e-12:
        raise RuntimeError(
            f"Timestep mismatch: rst has {state['timestep']}, input specifies {timestep}"
        )
    if ensemble in {"nvt", "npt"} and state["rng_state"] is None:
        raise RuntimeError(f"Exact {ensemble.upper()} restart requires the saved RNG state.")
    return state["step"] >= n_steps
