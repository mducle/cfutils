"""
This file tries to fit a point charge model to set of parameters to try to
infer what point charges could lead to such parameters.
The system is SrTb2O4, published in Orlandi et al., Phys. Rev. B 111 054415 (2025)
https://doi.org/10.1103/PhysRevB.111.054415
"""

# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import scipy
from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
from CrystalField.normalisation import split2range
from pychop.Instruments import Instrument

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib
import cef_utils
importlib.reload(cef_utils)

# Some scaling parameters
q_o = -1.15
m1_scale = 0.25
m2_scale = 0.95
p_scale = 0.2
distance = 4.81

np.set_printoptions(linewidth=200, precision=3, suppress=True)

# Reference parameters from fit in scipyfit_example.py (2 sets, set 2 is supposed to be better)
blm_normfac = split2range(Ion='Tb', EnergySplitting=100, Parameters=['B20', 'B22', 'B40', 'B42', 'B44', 'B60', 'B62', 'B64', 'B66', 'IB22', 'IB42', 'IB44', 'IB62', 'IB64', 'IB66'])
ref1 = [{'B20':-0.25203, 'B22':-0.062942, 'B40':0.00041702, 'B42':-0.007776, 'B44':0.00075104, 'B60':-4.3136e-06, 'B62':-0.00014614, 'B64':-0.00011489, 'B66':5.399e-05,
        'IB22':0.18677, 'IB42':0.015887, 'IB44':-0.012857, 'IB62':-5.8902e-05, 'IB64':0.00010916, 'IB66':7.9352e-05},
        {'B20':0.13636, 'B22':0.92383, 'B40':-0.0001386, 'B42':0.016615, 'B44':0.0054329, 'B60':4.7082e-06, 'B62':-8.5791e-06, 'B64':3.9795e-05, 'B66':8.9077e-05,
        'IB22':0.17572, 'IB42':-0.015271, 'IB44':-0.010222, 'IB62':-8.4151e-06, 'IB64':6.075e-05, 'IB66':5.1648e-05}]
ref2 = [{'B20':-0.16912, 'B22':-0.10186, 'B40':0.00014804, 'B42':-0.0078605, 'B44':-0.002352, 'B60':-1.6262e-05, 'B62':1.3479e-05, 'B64':-0.00012705, 'B66':-7.3984e-05,
        'IB22':0.26959, 'IB42':0.014414, 'IB44':-0.012302, 'IB62':2.1008e-06, 'IB64':0.00015075, 'IB66':6.9006e-05},
        {'B20':0.047162, 'B22':0.78707, 'B40':-0.00010101, 'B42':0.01817, 'B44':0.0073358, 'B60':3.2873e-06, 'B62':1.3075e-05, 'B64':1.7755e-05, 'B66':2.3871e-05,
        'IB22':0.21378, 'IB42':-0.019085, 'IB44':-0.012022, 'IB62':1.9808e-05, 'IB64':-5.7097e-05, 'IB66':3.9924e-05}]
ciffile = os.path.join(os.path.dirname(__file__), 'datafiles', 'SrTb2O4_30341-ICSD.cif')
cif_pc_model = PointCharge(ciffile)
chi2v = [10]

cwa = AlgorithmManager.create('CreateWorkspace')
cwa.initialize()
cwa.setChild(True)
cwa.setProperty('OutputWorkspace', 'chi2')
def pc_model(pc, ref=ref2, id0=0, id1=1):
    cif_pc_model.Charges = {'Tb1':pc[0], 'Tb2':pc[1], 'Sr':pc[2], 'O1':pc[3], 'O2':pc[4], 'O3':pc[5], 'O4':pc[6]}
    cif_pc_model.MaxDistance = distance
    cif_pc_model.IonLabel = 'Tb1'
    blm1 = cif_pc_model.calculate()
    cif_pc_model.IonLabel = 'Tb2'
    blm2 = cif_pc_model.calculate()
    chi2 = 0
    for k in ref[0].keys():
        nf = blm_normfac[k.replace('I', '')]
        try:
            chi2 += np.abs(blm1[k] - ref[id0][k]) / nf
            chi2 += np.abs(blm2[k] - ref[id1][k]) / nf
        except:
            return 50.0
    if chi2 < np.min(chi2v):
        print(chi2)
    chi2v.append(chi2)
    cwa.setProperty('DataX', range(len(chi2v)))
    cwa.setProperty('DataY', chi2v)
    cwa.execute()
    outws = cwa.getProperty('OutputWorkspace')
    mtd.addOrReplace('chi2', outws.value)
    #CreateWorkspace(range(len(chi2v)), chi2v, OutputWorkspace='chi2')
    return chi2

p0 = [3*m1_scale*p_scale, 3*m1_scale*p_scale, 2*m2_scale*p_scale, q_o*p_scale, q_o*p_scale, q_o*p_scale, q_o*p_scale]
res = scipy.optimize.minimize(pc_model, p0, method='BFGS', options={'maxiter':1}) # 1 iteration to be quick for tests
#res = scipy.optimize.dual_annealing(pc_model, [[0,4], [0,4], [0,4], [-4,0], [-4,0], [-4,0], [-4,0]])
#res = scipy.optimize.basinhopping(pc_model, p0)
#res = scipy.optimize.differential_evolution(pc_model, [[0,4], [0,4], [0,4], [-4,0], [-4,0], [-4,0], [-4,0]])
print(res)
pc = res.x

# Reference cest fit parameters
#pc = [4.62942,7.45576,9.11684,0.15107,1.31120,1.06926,-2.23211]; distance=4.81;

###########################
# Evaluates the best CEF parameters from the PC model, and calculates the spectrum and physical properties
###########################

ch2 = [pc_model(pc, ref1, 1, 0), pc_model(pc, ref1, 0, 1), pc_model(pc, ref2, 1, 0), pc_model(pc, ref2, 0, 1)]
cif_pc_model.Charges = {'Tb1':pc[0], 'Tb2':pc[1], 'Sr':pc[2], 'O1':pc[3], 'O2':pc[4], 'O3':pc[5], 'O4':pc[6]}
cif_pc_model.MaxDistance = distance
cif_pc_model.IonLabel = 'Tb1'
blm1 = cif_pc_model.calculate()
cif_pc_model.IonLabel = 'Tb2'
blm2 = cif_pc_model.calculate()

merlin = Instrument('MERLIN', 'G', 150.)
merlin.setEi(7.)
resmod1 = ResolutionModel(merlin.getResolution, xstart=-10, xend=6.9, accuracy=0.01)
merlin.setEi(18.)
resmod2 = ResolutionModel(merlin.getResolution, xstart=-10, xend=17.9, accuracy=0.01)
merlin = Instrument('MERLIN', 'G', 300.)
merlin.setEi(30.)
resmod3 = ResolutionModel(merlin.getResolution, xstart=-10, xend=29.9, accuracy=0.01)
merlin.setEi(82.)
resmod4 = ResolutionModel(merlin.getResolution, xstart=-10, xend=82.9, accuracy=0.01)

resmods = [resmod1, resmod2, resmod3, resmod4]

datdir = os.path.join(os.path.dirname(__file__), 'datafiles')
mer46207_ei7_cut = Load(f'{datdir}/mer46207_ei7_cut_paper.nxs')
mer46207_ei18_cut = Load(f'{datdir}/mer46207_ei18_cut_paper.nxs')
mer46210_ei30_cut = Load(f'{datdir}/mer46210_ei30_cut_paper.nxs')
mer46210_ei82_cut = Load(f'{datdir}/mer46210_ei82_cut_paper.nxs')


FWHMs = [np.interp(0, *resmods[irm].model)*1.5 for irm in [0,2,3]]
cf1 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **blm1)
cf2 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **blm2)
cf = cf1 + cf2
fit = CrystalFieldFit(Model=cf, InputWorkspace=[mer46207_ei7_cut, mer46210_ei30_cut, mer46210_ei82_cut],
                      MaxIterations=0, Output='fit')
fit.fit()

cf1 = CrystalField('Tb', 'C2', Temperature=[7], FWHM=0.5, **blm1)
cf2 = CrystalField('Tb', 'C2', Temperature=[7], FWHM=0.5, **blm2)
sp1 = cf1.getSpectrum(x_range=(0.1,100))
sp2 = cf2.getSpectrum(x_range=(0.1,100))
wss = CreateWorkspace(sp1[0], sp1[1]+sp2[1])

xx = np.arange(1,500,1)
chi1_x = CreateWorkspace(*cf1.getSusceptibility(xx, Hdir=[1, 0, 0], Inverse=True, Unit='cgs'))
chi1_y = CreateWorkspace(*cf1.getSusceptibility(xx, Hdir=[0, 1, 0], Inverse=True, Unit='cgs'))
chi1_z = CreateWorkspace(*cf1.getSusceptibility(xx, Hdir=[0, 0, 1], Inverse=True, Unit='cgs'))
chi1_p = (chi1_x + chi1_y + chi1_z) / 3

xx = np.arange(1,500,1)
chi2_x = CreateWorkspace(*cf2.getSusceptibility(xx, Hdir=[1, 0, 0], Inverse=True, Unit='cgs'))
chi2_y = CreateWorkspace(*cf2.getSusceptibility(xx, Hdir=[0, 1, 0], Inverse=True, Unit='cgs'))
chi2_z = CreateWorkspace(*cf2.getSusceptibility(xx, Hdir=[0, 0, 1], Inverse=True, Unit='cgs'))
chi2_p = (chi2_x + chi2_y + chi2_z) / 3

chi_x = (chi1_x + chi2_x) / 2
chi_y = (chi1_y + chi2_y) / 2
chi_z = (chi1_z + chi2_z) / 2
chi_p = (chi1_p + chi2_p) / 2

E1 = cf1.getEigenvalues()
E2 = cf2.getEigenvalues()

print('*** Output ***')
print(f'chi=[{ch2[0]:.3f}, {ch2[1]:.3f}, {ch2[2]:.3f}, {ch2[3]:.3f}]')
print(E1)
print(E2)
print()
print(blm1)
print(blm2)
print('[' + ','.join([f'{v:0.5f}' for v in pc]) + ']')

