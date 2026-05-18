"""Usage: parse user-facing correction parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from ...jobABC import JobABC
from ..utils.Scan.optimizer import LBFGSParams
from ..utils.TorsionFit import TorsionFitParams, build_torsion_fit_params


@dataclass
class CorrectionConfig:
    mol2: str = ""
    vib_scale: float = 1.0
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)
    lbfgs: LBFGSParams = field(default_factory=LBFGSParams)
    scan_opt: LBFGSParams = field(default_factory=LBFGSParams)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def build_correction_config(raw_params: Optional[dict]) -> CorrectionConfig:
    params = raw_params if isinstance(raw_params, dict) else {}
    config = CorrectionConfig()
    sub_dict = JobABC._select_subdict(params, ("parmfit", "correction"))
    JobABC._update_dataclass_from_dict(config, sub_dict)
    config.torsion = build_torsion_fit_params(params)

    lbfgs = LBFGSParams()
    JobABC._update_dataclass_from_dict(lbfgs, JobABC._select_subdict(params, ("lbfgs", "LBFGS", "opt")))
    config.lbfgs = lbfgs
    config.scan_opt = deepcopy(lbfgs)
    config.scan_opt.write_traj = False
    config.scan_opt.verbose = 0
    return config
