"""
Mantid engine class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
from ..engine import EngineBase, BLMS
import numpy as np

import mantid.simpleapi as s_api
import CrystalField.fitting
CrystalField.fitting.energies = CFEnergy
from CrystalField import CrystalField
from CrystalField.normalisation import split2range, _get_normalisation, ionname2Nre
from CrystalField.fitting import getSymmAllowedParam, makeWorkspace
from CrystalField.function import PhysicalProperties

def CFEnergy(nre, **kwargs):
    from CrystalField.energies import _unpack_complex_matrix
    cfe = s_api.AlgorithmManager.create('CrystalFieldEnergies')
    cfe.initialize()
    cfe.setChild(True)
    cfe.setProperty('nre', nre)
    for k, v in kwargs.items():
        cfe.setProperty(k, v)
    cfe.execute()
    # Unpack the results
    eigenvalues = cfe.getProperty('Energies').value
    dim = len(eigenvalues)
    eigenvectors = _unpack_complex_matrix(cfe.getProperty('Eigenvectors').value, dim, dim)
    hamiltonian = _unpack_complex_matrix(cfe.getProperty('Hamiltonian').value, dim, dim)
    return eigenvalues, eigenvectors, hamiltonian

class metaMantid(type(EngineBase), type(CrystalField)): ...

class MantidEngine(EngineBase, CrystalField, metaclass=metaMantid):

    def __init__(self, Ion='Ce', Symmetry='C1', **kwargs):
        if MANTID_NOT_FOUND:
            raise ModuleNotFoundError('module mantid.simple api not found')
        CrystalField.__init__(Ion, Symmetry, **kwargs)
        
    def hamiltonian(self):
        return self.getHamiltonian()

    def dipolematrices(self):
        coef = 1 / self._calc_gJuB()
        return (CFEnergy(self._nre, **{f'Bext{d}':coef})[2] for d in ['X', 'Y', 'Z'])

    def solve(self):
        if self._dirty_eigensystem:
            self._eigenvalues, self._eigenvectors, self._hamiltonian = CFEnergy(self._nre, **self._getFieldParameters())
        return self._eigenvalues, self._eigenvectors, self._hamiltonian

    def peaks(self):
        return self.getPeakList()

    def magnetisation(self, temperature=1, field=1, hdir='powder', unit='bohr'):
        """Returns the magnetisation at a set of temperatures and fields as a numpy array with T-row-wise and H-col-wise"""
        tt, hh = (np.squeeze(np.array(temperature)), np.squeeze(np.array(field)))
        if len(hh) == 1:
            xdat = makeWorkspace(tt, tt*0)
            ppobj = PhysicalProperties('M(T)', Hmag=hh[0], Hdir=hdir, Unit=unit)
        elif len(tt) == 1:
            xdat = makeWorkspace(hh, hh*0)
            ppobj = PhysicalProperties('mag', Temperature=tt[0], Hdir=hdir, Unit=unit)
        else:
            xdat, rv = (makeWorkspace(tt, tt*0), [])
            for hv in hh:
                rv.append(self._getPhysProp(PhysicalProperties('M(T)', Hmag=hv, Hdir=hdir, Unit=unit), xdat, 0)[1])
            return np.array(rv)
        return self._getPhysProp(ppobj, xdat, 0)[1]

    def susceptibility(self, temperature=1, hdir='powder', unit='bohr'):
        """Returns the susceptibility using the Van Vleck formula at a set of fields as a numpy array"""
        xdat = makeWorkspace(*(np.squeeze(np.array(temperature)),)*2)
        return self._getPhysProp(PhysicalProperties('chi', Hdir=hdir, Unit=unit, Lambda=0, Chi0=0))

    def heatcapacity(self, temperature=1, field=0, hdir='powder'):
        """Returns the magnetic specific heat at constant volume at a set of temperatures and fields as a numpy array"""
        tt, hh = (np.squeeze(np.array(temperature)), np.squeeze(np.array(field)))
        fstr = self.makePhysicalPropertiesFunction(PhysicalProperties('Cp'))
        xdat, rv = (makeWorkspace(tt, tt*0), [])
        def _calcpphext(hmag, hdir):
            return self._calcSpectrum(fstr + ','.join([f'Bext{d}=m' for d,m in zip(['X', 'Y', 'Z'], np.array(hdir)*hmag)]), xdat, 0)[1]
        def _calcpppowder(hmag):
            return np.sum([_calcpphext(hmag, hd) for hd in [[0,0,1], [0,1,0], [1,0,0]]], axis=0)
        if len(hh) > 0:
            for hv in hh:
                rv.append(_calcpppowder(hv) if 'powder' in hdir else _calcpphext(hv, hdir))
        elif abs(hh[0]) < 1e-3:
            return self._calcSpectrum(fstr, xdat, 0)[1]
        else:
            return _calcpppowder(hh[0]) if 'powder' in hdir else _calcpphext(hh[0], hdir)

    def fitengy(self, evec, random_start=False):
        """Uses the Newman-Ng algorithm to fit a set of crystal field parameters to a level scheme."""
        J = [0, 5.0 / 2, 4, 9.0 / 2, 4, 5.0 / 2, 0, 7.0 / 2, 6, 15.0 / 2, 8, 15.0 / 2, 6, 7.0 / 2][self._nre]

        if random_start:
            # Estimate initial parameters using Monte Carlo sampling
            nz_pars = getSymmAllowedParam(self.Symmetry)
            if '2' in self.Symmetry and 'B66' not in nz_pars: # bug in Mantid
                nz_pars += ['B66']
                if any([self.Symmetry == val for val in ["C2", "Cs", "C2h"]]):
                    nz_pars += ['IB66']
            if J < 3:
                nz_pars = [v for v in nz_pars if 'B6' not in v]
            ebw = np.max(E0) - np.min(E0)
            ranges = split2range(Ion=self.Ion, EnergySplitting=ebw, Parameters=nz_pars)
            # Estimate initial parameters using a Monte Carlo method
            initBlm = {p:(np.random.rand()-0.5)*ranges[p.replace('I','')] for p in nz_pars}
        else:
            initBlm = self._getFieldParameters()
            if len(initBlm) == 0:
                raise RuntimeError('You must specify at least one input Blm parameter, or use random_start')

        iscubic = self.Symmetry in ['T', 'Td', 'Th', 'O', 'Oh']
        if iscubic:
            if 'B40' not in kwargs.keys() or 'B60' not in kwargs.keys():
                pass
            else:
                if 'B44' not in kwargs.keys():
                    initBlm['B44'] = 5 * initBlm['B40']
                if 'B64' not in kwargs.keys():
                    initBlm['B44'] = -21 * initBlm['B60']

        # Calculates the matrix elements <n|O_k^q|m>
        Omat = {}
        denom = {}
        for lm in initBlm.keys():
            bdict = {lm: 1}
            ee, vv, ham = CFEnergy(**bdict)
            Omat[lm] = np.asmatrix(ham)
            denom[lm] = np.trace( np.real( (Omat[lm].H) * Omat[lm] ))

        Ecalc, vv, ham = CFEnergy(**initBlm)
        if len(E0) < len(Ecalc):
            #E = list(sorted(kwargs['E'])) + list(Ecalc[-(len(Ecalc)-len(E0)):]*(0.13*((100-num_iter)/100)+1) )
            # For each desired level, find nearest calculated level and substitute it for that
            Eref = Ecalc
            E = copy.deepcopy(Ecalc)
            if all(np.diff(E0) > 0): # If input has no degeneracies, account for degenerate levels
                for en in E0:
                    Edif = np.abs(Eref - en)
                    idx = np.where(Edif == np.min(Edif))[0]
                    E[np.where((Ecalc-Eref[idx[0]])==0)] = en
                    Eref = np.delete(Eref, idx)
            else:
                for en in E0:
                    Idif = np.argmin(np.abs(Eref - en))
                    E[np.argmin(np.abs(Ecalc - Eref[Idif]))] = en
                    Eref = np.delete(Eref, Idif)
            E0 = E - np.mean(E)
        else:
            E0 = E0 - np.mean(E0)

        lsqfit = 0
        Blm = initBlm
        div_count = 0
        for num_iter in range(100):
            if iscubic:
                Blm['B44'] = 5 * Blm['B40']
                Blm['B64'] = -21 * Blm['B60']
            Ecalc, vv, ham = CFEnergy(**Blm)
            V = np.asmatrix(vv)
            Ecalc = Ecalc - np.mean(Ecalc)
            newlsqfit = np.sum(np.power(Ecalc-E0,2))
            if np.fabs(lsqfit - newlsqfit) < 1.e-7:
                break
            if newlsqfit > lsqfit:
                div_count += 1
            if div_count > 10:
                warnings.warn('Fit is diverging')
                break
            lsqfit = newlsqfit
            for lm in initBlm.keys():
                # Calculates the numerator = sum_n En <j|Okq|i>_nn
                numer = np.dot( np.real( np.diag( V.H * Omat[lm] * V ) ), E0 )
                # Calculates the new Blm parameter
                Blm[lm] = numer / denom[lm]

        # Updates self parameters with new fitted values
        for k, v in Blm.items():
            self[k] = v
