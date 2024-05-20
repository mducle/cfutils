# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os

from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
from pychop.Instruments import Instrument

sys.path.insert(0, os.path.dirname(__file__))
#from cef_fitengy import fitengy
import importlib
import cef_fitengy
importlib.reload(cef_fitengy)
import fit_scipy
importlib.reload(fit_scipy)

np.set_printoptions(linewidth=200, precision=3, suppress=True)

# Conversion factor from Wybourne to Stevens normalisation
from scipy import sqrt
lambdakq = {'IB22':sqrt(6.)/2., 'IB21':sqrt(6.), 'B20':1./2., 'B21':sqrt(6.), 'B22':sqrt(6.)/2.,
     'IB44':sqrt(70.)/8., 'IB43':sqrt(35.)/2., 'IB42':sqrt(10.)/4., 'IB41':sqrt(5.)/2., 'B40':1./8., 'B41':sqrt(5.)/2., 'B42':sqrt(10.)/4., 'B43':sqrt(35.)/2., 'B44':sqrt(70.)/8., 
     'IB66':sqrt(231.)/16., 'IB65':3*sqrt(77.)/8., 'IB64':3*sqrt(14.)/16., 'IB63':sqrt(105.)/8., 'IB62':sqrt(105.)/16., 'IB61':sqrt(42.)/8.,
     'B60':1./16., 'B61':sqrt(42.)/8., 'B62':sqrt(105.)/16., 'B63':sqrt(105.)/8., 'B64':3*sqrt(14.)/16., 'B65':3*sqrt(77.)/8., 'B66':sqrt(231.)/16.}

# Stevens Operator Equivalent factors
idx = {'IB22':0, 'IB21':0, 'B20':0, 'B21':0, 'B22':0, 'IB44':1, 'IB43':1, 'IB42':1, 'IB41':1, 'B40':1, 'B41':1, 'B42':1, 'B43':1, 'B44':1, 
       'IB66':2, 'IB65':2, 'IB64':2, 'IB63':2, 'IB62':2, 'IB61':2, 'B60':2, 'B61':2, 'B62':2, 'B63':2, 'B64':2, 'B65':2, 'B66':2}
thetakq = {'Tb': [-1.0 * 1/3/3/11, 1.0 * 2/3/3/3/5/11/11, -1.0 * 1/3/3/3/3/7/11/11/13],
           'Dy': [-1.0 * 2/3/3/5/7, -1.0 * 2*2*2/3/3/3/5/7/11/13, 1.0 * 2*2/3/3/3/7/11/11/13/13],
           'Ho': [-1.0 * 1/2/3/3/5/5, -1.0 * 1/2/3/5/7/11/13, -1.0 * 5/3/3/3/7/11/11/13/13],
           'Er': [1.0 * 2*2/3/3/5/5/7, 1.0 * 2/3/3/5/7/11/13, 1.0 * 2*2*2/3/3/3/7/11/11/13/13],
           'Tm': [1.0 * 1/3/3/11, 1.0 * 2*2*2/3/3/3/3/5/11/11, -1.0 * 5/3/3/3/3/7/11/11/13]}

# Expectation values of radial wavefunction from Freeman and Desclaux JMMM 12 (1979) 11
rk = {'Tb': [0.8220, 1.651, 6.852],
      'Dy': [0.7814, 1.505, 6.048],
      'Ho': [0.7446, 1.379, 5.379],
      'Er': [0.7111, 1.270, 4.816],
      'Tm': [0.6804, 1.174, 4.340]}

# CEF parameters from Sala et al., PRB 98 014419 (2018)
pars_Ho2Ge2O7 = {'B20':64.9, 'B40':27.3, 'B43':185, 'B60':1.05, 'B63':-16.9, 'B66':24}
pars_Ho2Ti2O7 = {'B20':50.3, 'B40':26.1, 'B43':185, 'B60':1.05, 'B63':-15.6, 'B66':20}
pars_Ho2Sn2O7 = {'B20':59.7, 'B40':22.7, 'B43':191, 'B60':0.93, 'B63':-14.7, 'B66':19}

# Converts parameters to "Neutron" convention Stevens parameters (without Stevens factor and in meV).
pars_Ge = {k:v*thetakq['Ho'][idx[k]]*rk['Ho'][idx[k]] for k, v in pars_Ho2Ge2O7.items()}
pars_Ti = {k:v*thetakq['Ho'][idx[k]]*rk['Ho'][idx[k]] for k, v in pars_Ho2Ti2O7.items()}
pars_Sn = {k:v*thetakq['Ho'][idx[k]]*rk['Ho'][idx[k]] for k, v in pars_Ho2Sn2O7.items()}

# "de Gennes" scaling of Er/Ho pars to Tb
#tb1 = {k:v*thetakq['Tb'][idx[k]] for k, v in pars_Ho2Ti2O7.items()}

cf = CrystalField('Ho', 'C3', Temperature=5, FWHM=5, **pars_Ti)
print(cf.getPeakList())
calc_spec = CreateWorkspace(*cf.getSpectrum())
print(np.round(cf.getEigenvalues(), 1))
print(np.real(cf.getEigenvectors()))

ciffile = os.path.join(os.path.dirname(__file__), 'ICSD_CollCode112533.cif')
cif_pc_model = PointCharge(ciffile)
cif_pc_model.Charges = {'Tb1':3*m1_scale*p_scale, 'Tb2':3*m1_scale*p_scale, 'Sr':2*m2_scale*p_scale, 'O1':q_o*p_scale, 'O2':q_o*p_scale, 'O3':q_o*p_scale, 'O4':q_o*p_scale}
cif_pc_model.MaxDistance = distance
cif_pc_model.IonLabel = 'Tb1'
blm1 = cif_pc_model.calculate()

blm_normfac = split2range(Ion='Tb', EnergySplitting=100, Parameters=list(blm1.keys()))
ref1 = [{'B20':-0.25203, 'B22':-0.062942, 'B40':0.00041702, 'B42':-0.007776, 'B44':0.00075104, 'B60':-4.3136e-06, 'B62':-0.00014614, 'B64':-0.00011489, 'B66':5.399e-05,
        'IB22':0.18677, 'IB42':0.015887, 'IB44':-0.012857, 'IB62':-5.8902e-05, 'IB64':0.00010916, 'IB66':7.9352e-05},
        {'B20':0.13636, 'B22':0.92383, 'B40':-0.0001386, 'B42':0.016615, 'B44':0.0054329, 'B60':4.7082e-06, 'B62':-8.5791e-06, 'B64':3.9795e-05, 'B66':8.9077e-05,
        'IB22':0.17572, 'IB42':-0.015271, 'IB44':-0.010222, 'IB62':-8.4151e-06, 'IB64':6.075e-05, 'IB66':5.1648e-05}]
ref2 = [{'B20':-0.16912, 'B22':-0.10186, 'B40':0.00014804, 'B42':-0.0078605, 'B44':-0.002352, 'B60':-1.6262e-05, 'B62':1.3479e-05, 'B64':-0.00012705, 'B66':-7.3984e-05,
        'IB22':0.26959, 'IB42':0.014414, 'IB44':-0.012302, 'IB62':2.1008e-06, 'IB64':0.00015075, 'IB66':6.9006e-05},
        {'B20':0.047162, 'B22':0.78707, 'B40':-0.00010101, 'B42':0.01817, 'B44':0.0073358, 'B60':3.2873e-06, 'B62':1.3075e-05, 'B64':1.7755e-05, 'B66':2.3871e-05,
        'IB22':0.21378, 'IB42':-0.019085, 'IB44':-0.012022, 'IB62':1.9808e-05, 'IB64':-5.7097e-05, 'IB66':3.9924e-05}]
cif_pc_model = PointCharge('/home/dl11170/users/olegpetrenko_srtb2o4_pc/SrTb2O4_30341-ICSD.cif')
chi2v = [10]

pc = [4.62942,7.45576,9.11684,0.15107,1.31120,1.06926,-2.23211]; distance=4.81;

cwa = AlgorithmManager.create('CreateWorkspace')
cwa.initialize()
cwa.setChild(True)
cwa.setProperty('OutputWorkspace', 'chi2')
def pc_model(pc, ref, id0=0, id1=1):
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

#res = scipy.optimize.minimize(pc_model, p0, method='BFGS')
#res = scipy.optimize.dual_annealing(pc_model, [[0,4], [0,4], [0,4], [-4,0], [-4,0], [-4,0], [-4,0]])
#res = scipy.optimize.basinhopping(pc_model, p0)
#res = scipy.optimize.differential_evolution(pc_model, [[0,4], [0,4], [0,4], [-4,0], [-4,0], [-4,0], [-4,0]])
#print(res)
#pc = res.x

ch2 = [pc_model(pc, ref1, 1, 0), pc_model(pc, ref1, 0, 1), pc_model(pc, ref2, 1, 0), pc_model(pc, ref2, 0, 1)]
cif_pc_model.Charges = {'Tb1':pc[0], 'Tb2':pc[1], 'Sr':pc[2], 'O1':pc[3], 'O2':pc[4], 'O3':pc[5], 'O4':pc[6]}
cif_pc_model.MaxDistance = distance
cif_pc_model.IonLabel = 'Tb1'
blm1 = cif_pc_model.calculate()
cif_pc_model.IonLabel = 'Tb2'
blm2 = cif_pc_model.calculate()

# from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
# 
# cf = cf1 + cf2
# fit = CrystalFieldFit(Model=cf, InputWorkspace=[mtd['mer46207_ei7_cut'], mtd['mer46210_ei30_cut'], mtd['mer46210_ei82_cut']],
#                       MaxIterations=0, Output='fit')
# import sys
# sys.path.append(os.path.dirname(__file__))
# import cef_fitengy
# import fit_scipy
# 
# fit_scipy.printpars(fit)

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

