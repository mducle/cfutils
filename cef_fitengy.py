# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import warnings

# Symmetry of rare earth ion site in Schoenflies notation
sym = 'C4v'
# The following values can be lists; if they are lists nFWHM x nTemp
# workspaces will be created simulating the fit parameters
FWHM = [0.9]
Temperature = [5]
ion = 'Ce'
# Must specify *all* 2J+1 energy levels (depending on ion type)
Elevels = [0, 0, 2.3, 2.3, 19.2, 19.2]

# ----------------------------------------------------------------

from CrystalField.energies import _unpack_complex_matrix
from CrystalField import CrystalField
from CrystalField.normalisation import split2range
from CrystalField.fitting import getSymmAllowedParam


def CFEnergy(nre, **kwargs):
    cfe = AlgorithmManager.create('CrystalFieldEnergies')
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


def fitengy(**kwargs):
    """ Uses the Newman-Ng algorithm to fit a set of crystal field parameters to a level scheme.

        blm = fitengy(Ion=ionname, sym=point_group, E=evec)
        blm = fitengy(Ion=ionname, E=evec, B=bvec)
        blm = fitengy(Ion=ionname, E=evec, B20=initB20, B40=initB40, ...)
        [B20, B40] = fitengy(IonNum=ionnumber, E=evec, B20=initB20, B40=initB40, OutputTuple=True)
        
        Note: This function only accepts keyword inputs.
        
        Inputs:
            ionname - name of the (tripositive) rare earth ion, e.g. 'Ce', 'Pr'.
            ionnumber - the number index of the rare earth ion: 
                   1=Ce 2=Pr 3=Nd 4=Pm 5=Sm 6=Eu 7=Gd 8=Tb 9=Dy 10=Ho 11=Er 12=Tm 13=Yb
            sym - a string with the Schoenflies symbol of the point group of the rare earth site
                   If bvec and sym are both given, bvec will take precedence (sym will be ignored)
            evec - a vector of the energy values to be fitted. Must equal 2J+1 for the selected ion.
            bvec - a vector of initial CF parameters in order: [B20 B21 B22 B40 B41 ... etc.]
                    zero values will not be fitted.
                    This vector can also a be dictionary instead {'B20':1, 'B40':2}
                    If no parameters are given but the symmetry is given random initial parameters
                    will be used, depending on the symmetry.
            B20 etc. - initial values of the CF parameters to be fitted. 

        Outputs:
            blm - a dictionary of the output crystal field parameters (default)
            [B20, etc] - a tuple of the output crystal field parameters (need to set OutputTuple flag)
    """

    # Some Error checking
    if 'Ion' not in kwargs.keys() and 'IonNum' not in kwargs.keys():
        raise NameError('You must specify the ion using either the ''Ion'' or ''IonNum'' keywords')
    if 'E' not in kwargs.keys():
        raise NameError('Input energy level scheme must be supplied as the ''E'' keyword input')
    E0 = np.array(sorted(kwargs['E']))# - np.mean(kwargs['E']))

    # Some definitions
    Blms = ['B20', 'B21', 'B22', 'B40', 'B41', 'B42', 'B43', 'B44', 'B60', 'B61', 'B62', 'B63', 'B64', 'B65', 'B66',
            'IB20', 'IB21', 'IB22', 'IB40', 'IB41', 'IB42', 'IB43', 'IB44', 'IB60', 'IB61', 'IB62', 'IB63', 'IB64', 'IB65', 'IB66']
    Ions = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb']

    if 'B' in kwargs.keys():
        bvec = kwargs.pop('B')
        if isinstance(bvec, dict):
            kwargs.update(bvec)
        else:
            for ii in range(len(bvec)):
                kwargs[Blms[ii]] = bvec[ii]

    if 'Ion' in kwargs.keys():
        nre = [ id for id,val in enumerate(Ions) if val==kwargs['Ion'] ][0] + 1
    else:
        nre = kwargs['IonNum']

    Jvals = [0, 5.0 / 2, 4, 9.0 / 2, 4, 5.0 / 2, 0, 7.0 / 2, 6, 15.0 / 2, 8, 15.0 / 2, 6, 7.0 / 2]
    J = Jvals[nre]
    #if len(E0) != int(2*J+1):
    #    raise RuntimeError(f'Expected {2*J+1} levels for Ion {Ions[nre]} but only got {len(E0)} elements in E0')

    if 'sym' in kwargs and len(set(kwargs.keys()).intersection(set(Blms))) == 0:
        # No parameters given, estimate using Monte Carlo sampling
        nz_pars = getSymmAllowedParam(kwargs['sym'])
        if J < 3:
            nz_pars = [v for v in nz_pars if 'B6' not in v]
        ebw = np.max(E0) - np.min(E0)
        ranges = split2range(Ion=Ions[nre], EnergySplitting=ebw, Parameters=nz_pars)
        # Estimate initial parameters using a Monte Carlo method
        kwargs.update({p:(np.random.rand()-0.5)*ranges[p] for p in nz_pars})
        kwargs['is_cubic'] = kwargs['sym'] in ['T', 'Td', 'Th', 'O', 'Oh']

    iscubic = kwargs.pop('iscubic', False)
    if iscubic:
        if 'B40' not in kwargs.keys() or 'B60' not in kwargs.keys():
            pass
        else:
            if 'B44' not in kwargs.keys():
                kwargs['B44'] = 5 * kwargs['B40']
            if 'B64' not in kwargs.keys():
                kwargs['B44'] = -21 * kwargs['B60']

    fitparind = []
    initBlm = {}
    for ind in range(len(Blms)):
        if Blms[ind] in kwargs.keys():
            fitparind.append(ind)
            initBlm[Blms[ind]] = kwargs[Blms[ind]]

    if not fitparind:
        raise NameError('You must specify at least one input Blm parameter')

    # Calculates the matrix elements <n|O_k^q|m>
    Omat = {}
    denom = {}
    for ind in fitparind:
        bdict = {Blms[ind]: 1}
        ee, vv, ham = CFEnergy(nre, **bdict)
        Omat[Blms[ind]] = np.mat(ham)
        denom[Blms[ind]] = np.trace( np.real( (Omat[Blms[ind]].H) * Omat[Blms[ind]] ))

    Ecalc, vv, ham = CFEnergy(nre, **initBlm)
    if len(E0) < len(Ecalc):
        #E = list(sorted(kwargs['E'])) + list(Ecalc[-(len(Ecalc)-len(E0)):]*(0.13*((100-num_iter)/100)+1) )
        # For each desired level, find nearest calculated level and substitute it for that
        Eref, Enear = (Ecalc, [])
        E = copy.deepcopy(Ecalc)
        for en in E0:
            Idif = np.argmin(np.abs(Eref - en))
            Enear.append(Eref[Idif])
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
        Ecalc, vv, ham = CFEnergy(nre, **Blm)
        V = np.mat(vv)
        Ecalc = Ecalc - np.mean(Ecalc)
        newlsqfit = np.sum(np.power(Ecalc-E0,2))
        if np.fabs(lsqfit - newlsqfit)<1.e-7:
            break
        if newlsqfit > lsqfit:
            div_count += 1
        if div_count > 10:
            warnings.warn('Fit is diverging')
            break
        lsqfit = newlsqfit
        for ind in fitparind:
            # Calculates the numerator = sum_n En <j|Okq|i>_nn
            numer = np.dot( np.real( np.diag( V.H * Omat[Blms[ind]] * V ) ), E0 )
            # Calculates the new Blm parameter
            Blm[Blms[ind]] = numer / denom[Blms[ind]]

    if 'OutputTuple' in kwargs.keys() and kwargs['OutputTuple']:
        retval = []
        for ind in fitparind:
            retval.append(Blm[Blms[ind]])
        return tuple(retval)
    else:
        return Blm
        
if __name__ == '__main__' or __name__ == 'mantidqt.widgets.codeeditor.execution':
    # Uses fitengy defined above to fit the energy levels
    fitBlm = fitengy(Ion=ion, E=Elevels, sym=sym)
        
    np.set_printoptions(linewidth=200)
    print('Fitted CF parameters:')
    print(fitBlm)
    cf = CrystalField(ion, sym, Temperature=1, **fitBlm)
    print()
    print('Peaks at base temperatures:')
    print(cf.getPeakList())
    print()
    print('Energy Levels:')
    print(cf.getEigenvalues())

    for fw in FWHM:
        for tt in Temperature:
            cf = CrystalField(ion, sym, Temperature=tt, FWHM=fw, **fitBlm)
            cf.PeakShape = 'Gaussian'
            CreateWorkspace(*cf.getSpectrum(), OutputWorkspace=f'fitengy_FWHM{fw}_{tt}K')
