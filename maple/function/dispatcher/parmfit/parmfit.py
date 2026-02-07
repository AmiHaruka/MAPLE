from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer

class Parmfit(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms):
        super().__init__(output)
        self.atoms = atoms
        self.output = output
        self.commandcontrol = params
        
    #def run(self):
        #with timer("Parmfit"):
            #from .algorithm import LBFGS
            #prmfit = LBFGS(self.atoms, output=self.output, paras=self.commandcontrol)
            #prmfit.run()