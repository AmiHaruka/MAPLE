"""Usage: parse user-facing NCAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interface import RespConfig, set_method
from ..TorsionFit import TorsionFitParams, build_torsion_fit_params


@dataclass(frozen=True)
class NCAAAbinitioConfig:
    pdb_path: str
    target: str
    charge: int
    mult: int
    resp: RespConfig
    rn: str = "MOL"
    vib_scale: float = 1.0
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)


def parse_ncaa_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
    target: str,
    charge: int,
    mult: int,
) -> NCAAAbinitioConfig:
    raw = dict(raw or {})
    qm_backend = raw.get("qm_backend", raw.get("resp_backend", "gaussian")).strip().lower()
    return NCAAAbinitioConfig(
        pdb_path=pdb_path,
        target=target,
        charge=charge,
        mult=mult,
        resp=RespConfig(
            qm=set_method(
                {
                    "backend": qm_backend,
                    "theory": raw.get("theory", "HF").strip(),
                    "basis": raw.get("basis", "6-31G(d)").strip(),
                    "route": raw.get("route", "").strip(),
                    "nproc": int(raw.get("nproc", 8)),
                    "mem": int(raw.get("mem", 16)),
                }
            )
        ),
        rn=raw.get("rn", "MOL").strip().upper(),
        vib_scale=float(raw.get("vib_scale", 1.0)),
        torsion=build_torsion_fit_params(raw),
    )
