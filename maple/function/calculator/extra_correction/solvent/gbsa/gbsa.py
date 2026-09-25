"""Source-typed GBSA parameter data; the former MLIP/QEq GB model is disabled."""

from pathlib import Path


GBSA_UNAVAILABLE = (
    "GBSA is disabled: the previous implementation misinterpreted its source "
    "parameter types and units. These xTB parameter tables do not define a "
    "validated MLIP/QEq solvation model. Use implicit='none'; a replacement "
    "requires a separately specified and validated solvation model."
)


def load_gbsa_params(solvent="water"):
    """Read the two retained element columns using the Grimme source schema.

    These legacy Grimme-style extracts mix parameter families, and some files
    have unresolved provenance. The retained columns are surface tension and
    descreening, not atomic radii; the H-bond column is absent. This is audit
    metadata, not a verified solvent parameterization or an energy model.
    """
    path = Path(__file__).parent / "data" / f"{solvent}.dat"
    with path.open() as stream:
        lines = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
    header = [float(value) for value in lines[0].split(",")]
    values = [float(value) for line in lines[1:] for value in line.split(",") if value.strip()]
    if len(header) != 8 or len(values) != 188:
        raise ValueError(f"Invalid GBSA parameter schema in {path.name}")
    eps, molar_mass, density, born_scale, probe_radius, energy_shift, born_offset, _ = header
    return {
        "provenance_status": "unverified_legacy_extract",
        "eps": eps,
        "molar_mass": molar_mass,
        "density": density,
        "born_scale": born_scale,
        "probe_radius_angstrom": probe_radius,
        "energy_shift_kcal_mol": energy_shift,
        "born_offset_angstrom": born_offset * 0.1,
        "surface_tension": values[:94],
        "descreening": values[94:],
    }


class GBSA:
    """Rejected legacy entry point; no parameter reinterpretation or fake energy."""

    def __init__(self, solvent="water", device="cpu"):
        raise NotImplementedError(GBSA_UNAVAILABLE)
