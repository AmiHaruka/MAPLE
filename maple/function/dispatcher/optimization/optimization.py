
from typing import Union

from ase import Atoms

from ..jobABC import JobABC
from ...utility import Molecules

from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Union[Atoms, Molecules]):
        super().__init__(output)
        self.atoms = atoms
        self.commandcontrol = params

    def run(self):
        with timer("Optimization"):
            method = str(self.commandcontrol.get('method') or 'lbfgs').lower()

            if isinstance(self.atoms, Molecules):
                raise NotImplementedError(
                    "Multi-structure optimization is not supported (including "
                    f"method={method!r}). Split the input structures and run "
                    "supported single-structure optimizations."
                )

            if method == 'lbfgs':
                from .algorithm import LBFGS
                return LBFGS(self.atoms, output=self.output,
                             paras=self.commandcontrol).run()
            elif method == 'rfo':
                from .algorithm import RFO
                return RFO(self.atoms, output=self.output,
                           paras=self.commandcontrol).run()
            elif method in ('sd', 'sdcg', 'cg'):
                from .algorithm import SDCG
                return SDCG(self.atoms, output=self.output,
                            paras=self.commandcontrol).run()
            else:
                raise NotImplementedError(
                    f"Unknown opt method: {method!r}. "
                    f"Supported: lbfgs, rfo, sd, sdcg, cg.")

# Backward compatibility for the historical misspelling.
Optmization = Optimization
