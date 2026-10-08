"""Usage: expose parmfit refinement as the dispatcher-facing job class."""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from ....jobABC import JobABC
from ...runconfig import as_tracked, write_parmfit_run
from ..readparm import RefinementParameterSet
from ..TorsionFit import run_torsion_workflow
from .artifacts import RefinementWorkflowResult
from .config import build_refinement_config
from .workflow import run_refinement_workflow

from maple.function.timer import timer




class Refinement(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        tracker = as_tracked(params if isinstance(params, dict) else {})
        self.params = build_refinement_config(tracker)
        write_parmfit_run(
            output,
            "refinement",
            tracker,
            warn=lambda message: self.log_info([f"WARNING: {message}\n"]),
        )
        self.workflow_result: RefinementWorkflowResult | None = None

    @property
    def result(self) -> Optional[RefinementParameterSet]:
        if self.workflow_result is None:
            return None
        return self.workflow_result.final_parmset

    def run(self) -> RefinementWorkflowResult:
        with timer("Parmfit refinement"):
            self.workflow_result = run_refinement_workflow(
                output=self.output,
                atoms=self.atoms,
                config=self.params,
                log_info=self.log_info,
                torsion_workflow_fn=run_torsion_workflow,
            )
            return self.workflow_result
