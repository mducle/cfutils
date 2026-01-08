"""
McPhase engine class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
from .engine import EngineBase, BLMS
import numpy as np
import libmcphase

class McPhaseEngine(EngineBase, libmcphase.cf1ion):
    def __init__(self, Ion='Ce3+', **kwargs):
        libmcphase.cf1ion.__init__(self, Ion, **kwargs)

class McPhaseICFEngine(EngineBase, libmcphase.icf1ion):
    def __init__(self, Ion='Ce3+', **kwargs):
        libmcphase.icf1ion.__init__(self, Ion, **kwargs)

class McPhaseICEngine(EngineBase, libmcphase.ic1ion):
    def __init__(self, Ion='Ce3+', **kwargs):
        libmcphase.ic1ion.__init__(self, Ion, **kwargs)
