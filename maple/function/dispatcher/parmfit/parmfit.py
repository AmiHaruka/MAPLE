'''
Author:  Kanbe 
Date: 2026-03-06 15:43:25
LastEditors:  Kanbe 
LastEditTime: 2026-03-11 10:20:52
FilePath: /MAPLE/maple/function/dispatcher/parmfit/parmfit.py
Description: 
'''
from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer

class Parmfit(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='abinitio'):
        super().__init__(output)
        self.atoms = atoms
        if method is None:
            self.method = 'abinitio'
        else:
            self.method = method
        self.output = output
        self.commandcontrol = params
        
    def run(self):
        with timer("Parmfit optimization"):
            if self.commandcontrol.get('method', 'abinitio').lower() == 'abinitio':
                from .abinitio import abinitio
                parmfit = abinitio(self.atoms, output=self.output, paras=self.commandcontrol)
                parmfit.run()
            elif self.commandcontrol.get('method').lower() == 'correction': # ffpopt-like
                from .correction import correction
                parmfit = correction(self.atoms, output=self.output, paras=self.commandcontrol)
                parmfit.run()
            else:
                raise NotImplementedError(f'Parmfit Strategy {self.method} not implemented yet.')
            
