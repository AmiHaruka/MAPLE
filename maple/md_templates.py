"""
MDP template generator for MAPLE MD ensembles.

Provides GROMACS-style MDP template files for NVE, NVT, and NPT ensembles.
"""

import os
from pathlib import Path


# MDP template definitions
_MDP_TEMPLATES = {
    'nve': """\
; NVE (microcanonical) ensemble - Constant N, V, E
; MAPLE MD template - Velocity Verlet integrator

integrator  = md          ; Velocity Verlet

timestep    = 0.1        ; fs
nsteps      = 400000     ; steps (= 40 ps)

ref-t       = 300.0      ; K (initial temperature for velocity init)

nstxout     = 100        ; trajectory output every N steps
nstenergy   = 100        ; log energy every N steps
traj-format = xyz        ; xyz (text) or dcd (binary)

nstcomm     = 100        ; remove COM motion every N steps

gen-vel     = yes
gen-temp    = 300.0      ; K

restart     = no
nst-rst      = 1000       ; checkpoint frequency
; random-seed = 12345   ; uncomment for reproducibility
""",

    'nvt': """\
; NVT (canonical) ensemble - Constant N, V, T
; MAPLE MD template - Langevin or V-rescale thermostat

integrator  = md

timestep    = 0.1        ; fs
nsteps      = 100000     ; steps (= 10 ps)

ref-t       = 300.0      ; K
tcoupl      = langevin   ; langevin or v-rescale

friction    = 0.001      ; 1/fs (Langevin only)
; tau-t      = 0.1       ; ps (V-rescale only)

nstxout     = 100
nstenergy   = 100
traj-format = dcd

gen-vel     = yes
gen-temp    = 300.0      ; K

restart     = no
nst-rst      = 1000
""",

    'npt': """\
; NPT (isothermal-isobaric) ensemble - Constant N, P, T
; MAPLE MD template - V-rescale thermostat + C-rescale barostat

integrator  = md

timestep    = 0.1        ; fs
nsteps      = 20000      ; steps (= 2 ps)

ref-t       = 300.0      ; K
tcoupl      = v-rescale  ; langevin or v-rescale
; tau-t      = 0.2       ; ps (V-rescale only)

ref-p       = 1.0        ; bar
pcoupl      = c-rescale  ; berendsen or c-rescale
tau-p       = 2.0        ; ps
compressibility = 4.5e-5 ; 1/bar

nstxout     = 100
nstenergy   = 100
traj-format = dcd

gen-vel     = yes
gen-temp    = 300.0      ; K

restart     = no
nst-rst      = 1000
""",
}


def generate_mdp_template(ensemble: str, output: str = None, force: bool = False) -> None:
    """
    Generate MDP template file for the specified ensemble.

    Parameters
    ----------
    ensemble : str
        Ensemble type ('nve', 'nvt', or 'npt')
    output : str, optional
        Output filename. If not provided, defaults to '{ensemble}.mdp'
    force : bool, optional
        If True, overwrite existing file without prompting. Default is False.

    Raises
    ------
    ValueError
        If ensemble is not one of 'nve', 'nvt', or 'npt'
    FileExistsError
        If output file already exists and force=False

    Examples
    --------
    >>> generate_mdp_template('nve')  # Creates nve.mdp
    >>> generate_mdp_template('nvt', 'my_template.mdp')
    >>> generate_mdp_template('npt', force=True)  # Overwrites npt.mdp
    """
    ensemble = ensemble.lower()

    if ensemble not in _MDP_TEMPLATES:
        raise ValueError(
            f"Unknown ensemble: {ensemble}. "
            f"Valid options are: {', '.join(sorted(_MDP_TEMPLATES.keys()))}"
        )

    if output is None:
        output = f"{ensemble}.mdp"

    output_path = Path(output)

    # Check if file exists
    if output_path.exists() and not force:
        raise FileExistsError(
            f"File '{output}' already exists. Use -f/--force to overwrite."
        )

    # Write template
    output_path.write_text(_MDP_TEMPLATES[ensemble])

    print(f"Generated: {output}")
    print(f"  Ensemble: {ensemble.upper()}")
    print(f"  Edit the file to adjust parameters, then use:")
    print(f"  #md(mdp={output})")


def list_templates() -> None:
    """List available MDP templates."""
    print("Available MDP templates:")
    for name in sorted(_MDP_TEMPLATES.keys()):
        print(f"  maple md {name}")
