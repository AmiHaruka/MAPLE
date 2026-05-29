"""Usage: parse user-facing MetalAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interface import RespConfig, set_method


SUPPORTED_PROM = ("ff14SB", "ff19SB")
SUPPORTED_WATER_MODELS = ("tip3p", "spce", "tip4pew", "opc3", "opc", "fb3", "fb4")
SUPPORTED_ION_PARAMETER_SETS = ("hfe", "cm", "iod", "12_6", "12_6_4")
SUPPORTED_BONDED_METHODS = ("mseminario", "seminario")


@dataclass(frozen=True)
class MetalAbinitioConfig:
    pdb_path: str
    target: str
    charge: int
    mult: int
    target_residue: dict
    resp: RespConfig
    oxy: int | None = None
    add_resid: list[str] = field(default_factory=list)
    set_bonded: list[tuple[int, int]] = field(default_factory=list)
    cfmol2: list[str] = field(default_factory=list)
    cluster_cutoff: float = 3.0
    donor_cutoff: float = 2.7
    opt_max_iter: int = 256
    opt_max_step: float = 0.2
    watm: str = "tip3p"
    ionm: str = "12_6"
    prom: str = "ff14SB"
    bonded: str = "mseminario"


def _parse_set_bonded(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in value.replace(",", " ").split():
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid set_bonded pair {token!r}; expected SERIAL-SERIAL.")
        try:
            left, right = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid set_bonded pair {token!r}; expected integer SERIAL-SERIAL.") from exc
        pairs.append((left, right))
    return pairs


def parse_metal_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
    target: str,
    charge: int,
    mult: int,
    target_residue: dict,
    oxy: int | None = None,
) -> MetalAbinitioConfig:

    raw = dict(raw or {})
    raw_prom = raw.get("prom", "ff14SB").strip()

    prom = next((item for item in SUPPORTED_PROM if item.lower() == raw_prom.lower()), None)
    if prom is None:
        raise ValueError(f"Unsupported protein model {raw_prom!r}; expected one of {', '.join(SUPPORTED_PROM)}.")

    watm = raw.get("watm", "tip3p").strip().lower()
    if watm not in SUPPORTED_WATER_MODELS:
        raise ValueError(f"Unsupported water model {watm!r}; expected one of {', '.join(SUPPORTED_WATER_MODELS)}.")

    ionm = raw.get("ionm", "12_6").strip().lower()
    if ionm not in SUPPORTED_ION_PARAMETER_SETS:
        raise ValueError(
            f"Unsupported ion parameter set {ionm!r}; expected one of {', '.join(SUPPORTED_ION_PARAMETER_SETS)}."
        )

    bonded = raw.get("bonded", "mseminario").strip().lower()
    if bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )

    chgmod = int(raw.get("chgmod", 1))
    if chgmod not in {0, 1, 2, 3}:
        raise ValueError("chgmod must be one of 0, 1, 2, or 3.")

    return MetalAbinitioConfig(
        pdb_path=pdb_path,
        target=target,
        charge=charge,
        mult=mult,
        oxy=oxy,
        target_residue=target_residue,
        resp=RespConfig(
            qm=set_method(
                {
                    "backend": "gaussian",
                    "theory": raw.get("theory", "PBE1PBE").strip(),
                    "basis": raw.get("basis", "def2SVP").strip(),
                    "route": raw.get("route", "").strip(),
                    "nproc": int(raw.get("nproc", 8)),
                    "mem": int(raw.get("mem", 16)),
                }
            ),
            chgmod=chgmod,
            fixchg_resids=raw.get("fixchg_resids", "").split(),
            watm=watm,
            prom=prom,
        ),
        add_resid=raw.get("add_resid", "").split(),
        set_bonded=_parse_set_bonded(raw.get("set_bonded", "")),
        cfmol2=raw.get("cfmol2", "").replace(",", " ").split(),
        cluster_cutoff=float(raw.get("cluster_cutoff", 3.0)),
        donor_cutoff=float(raw.get("donor_cutoff", 2.7)),
        opt_max_iter=256,
        opt_max_step=0.2,
        watm=watm,
        ionm=ionm,
        prom=prom,
        bonded=bonded,
    )
