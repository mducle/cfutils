"""
McPhase engine class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
from ..engine import EngineBase, BLMS
import numpy as np
import libmcphase

class metaCf1ion(type(EngineBase), type(libmcphase.ic1ion)): ...
class metaIc1ion(type(EngineBase), type(libmcphase.cf1ion)): ...

class McPhaseEngine(EngineBase, libmcphase.cf1ion, metaclass=metaCf1ion):
    def __init__(self, Ion='Ce3+', **kwargs):
        libmcphase.cf1ion.__init__(self, Ion, **kwargs)
    def solve(self):
        V, E = self.eigensystem()
        return E, V, self.hamiltonian()

class McPhaseICEngine(EngineBase, libmcphase.ic1ion, metaclass=metaIc1ion):
    def __init__(self, Ion='Ce3+', **kwargs):
        libmcphase.ic1ion.__init__(self, Ion, **kwargs)
    def solve(self):
        V, E = self.eigensystem()
        return E, V, self.hamiltonian()
