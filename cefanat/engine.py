"""
Engine base class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
from abc import ABC, abstractmethod
import numpy as np

BLMS = ['B20', 'B21', 'B22', 'B40', 'B41', 'B42', 'B43', 'B44', 'B60', 'B61', 'B62', 'B63', 'B64', 'B65', 'B66',
        'IB21', 'IB22', 'IB41', 'IB42', 'IB43', 'IB44', 'IB61', 'IB62', 'IB63', 'IB64', 'IB65', 'IB66']

class EngineFactory():
    _engines = {}

    @classmethod
    def register(cls, engine_name, engine_cls):
        cls._engines[engine_name] = engine_cls

    @classmethod
    def list(cls):
        return list(cls._engines.keys())
    
    @classmethod
    def get(cls, engine_name):
        if engine_name not in cls._engines:
            raise ValueError(f'Engine {engine_name} not a registered engine')
        return cls._engines[engine_name]


class EngineBase(ABC):

    @abstractmethod
    def hamiltonian(self):
        """Should return the hamiltonian as a numpy array"""
        pass

    @abstractmethod
    def dipolematrices(self):
        """Should return the three dipole matrices Jx, Jy, Jz in that order"""
        pass

    @classmethod
    def __init_subclass__(cls):
        EngineFactory.register(cls.__name__, cls)

    def fitengy(self, energies):
        """If possible with this engine, child should implement Newman-Ng fitengy algorithm here"""
        raise NotImplementedError

    def solve(self):
        """Returns the eigenvectors, eigenvalues and hamiltonian - can be overridden by child"""
        ham = self.hamiltonian()
        en, ev = np.linalg.eig(ham)
        return en, ev, ham

    def transition_matrix(self, wf=None):
        """Returns the dipole transition matrix as an numpy array"""
        if not hasattr(self, 'Jx'):
            self.Jx, self.Jy, self.Jz = self.dipolematrices()
        if wf is None:
            _, wf, _ = self.solve()
        ix = np.dot(np.conj(np.transpose(wf)), np.dot(self.Jx, wf))
        iy = np.dot(np.conj(np.transpose(wf)), np.dot(self.Jy, wf))
        iz = np.dot(np.conj(np.transpose(wf)), np.dot(self.Jz, wf))
        return np.multiply(ix, np.conj(ix)) + np.multiply(iy, np.conj(iy)) + np.multiply(iz, np.conj(iz))

    def peaks(self, temperature=0):
        """Returns a list of peaks at a specific temperature, ordered by dipole transition intensity"""
        en, ev, _ = self.solve()
        trans = self.transition_matrix(ev)

    def magnetisation(self, temperature=1, field=1, hdir='powder', unit='bohr'):
        """Returns the magnetisation at a set of temperatures and fields as a numpy array with T-row-wise and H-col-wise"""
        raise NotImplementedError

    def susceptibility(self, temperature=1, hdir='powder', unit='bohr'):
        """Returns the susceptibility using the Van Vleck formula at a set of fields as a numpy array"""
        raise NotImplementedError

    def heatcapacity(self, temperature=1, field=0, hdir='powder'):
        """Returns the magnetic specific heat at constant volume at a set of temperatures and fields as a numpy array"""
        raise NotImplementedError
