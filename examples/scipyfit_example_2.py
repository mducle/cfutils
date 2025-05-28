# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import copy
import scipy

from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
from CrystalField.energies import energies
from pychop.Instruments import Instrument

sys.path.append(os.path.dirname(__file__))
import importlib
import cef_utils
importlib.reload(cef_utils)

np.set_printoptions(linewidth=200, precision=8, floatmode='fixed')

def r5(v):
    return np.round(v*1e5)/1e5
    
## --------------------------------------------------------------------- ##
## PrCuSb2 CEF data analysis
## --------------------------------------------------------------------- ##

# Pr site is in Wyckoff site 2c of P4/nmm (#129) which is 4mm/C4v
# The 9 levels of the J=4 multiplet are split into 2 doublets and 5 singlets
# There could thus in principle be 6 peaks.

# Manually fitting peaks yield the following energy levels:
# 1.3, 2.3, 8.1, 10.2, ~20, ~36

# Loads data
curdir = os.path.dirname(__file__)
if 'Ei100_20K_cut' not in mtd:
    Ei100_7K_cut = Scale(Load(f'{curdir}/cuts/MAR28975_100meV_cut.nxs', OutputWorkspace='tmp'), 1000)
    Ei23_7K_cut = Scale(Load(f'{curdir}/cuts/MAR28975_23meV_cut.nxs', OutputWorkspace='tmp'), 1000)
    Ei10_7K_cut = Scale(Load(f'{curdir}/cuts/MAR28975_10meV_cut.nxs', OutputWorkspace='tmp'), 1000)
    Ei100_20K_cut = Scale(Load(f'{curdir}/cuts/MAR28977_100meV_cut.nxs', OutputWorkspace='tmp'), 1000)
    Ei10_100K_cut = Scale(Load(f'{curdir}/cuts/MAR28979_10meV_cut.nxs', OutputWorkspace='tmp'), 1000)
    DeleteWorkspace('tmp')
else:
    Ei100_7K_cut = mtd['Ei100_7K_cut']
    Ei23_7K_cut = mtd['Ei23_7K_cut']
    Ei10_7K_cut = mtd['Ei10_7K_cut']
    Ei100_20K_cut = mtd['Ei100_20K_cut']
    Ei10_100K_cut = mtd['Ei10_100K_cut']

mari = Instrument('MARI', 'G', 400.)
mari.setEi(7.)
resmod1 = ResolutionModel(mari.getResolution, xstart=-10, xend=6.9, accuracy=0.01)
mari.setEi(23.)
resmod2 = ResolutionModel(mari.getResolution, xstart=-20, xend=22.9, accuracy=0.01)
mari.setEi(100.)
resmod3 = ResolutionModel(mari.getResolution, xstart=-50, xend=99.9, accuracy=0.01)
resmods = [resmod1, resmod2, resmod3]
 
importlib.reload(cef_utils)

#res = cef_utils.fit_en(fit, [1.3, 2.3, 8.1, 10.2, 20, 36], 
#    fit_alg='local', method='Nelder-Mead', jac='3-point', options={'maxiter':500},#, 'samples':10},
#    widths_kwargs={'maxfwhm':[0.5, 3.0, 10.0], 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})

# Refine the energies a bit before fitting spectra
#print('------- Fitting energy guesses ------')
#Blm_en = cef_utils.fitengy(Ion='Pr', E=[0, 1.3, 1.3, 2.3, 8.1, 8.1, 10.2, 20, 26], sym='C4v')
#print(Blm_en)

#cf = CrystalField('Pr', 'C4v', Temperature=0.1, **Blm_en)
#print(r5(cf.getPeakList().T))
#print(r5(cf.getEigenvalues()))

# Best parameters
B0 = {'B20': 0.24118031077452998, 'B40': 0.005740210620195332, 'B44': 0.028535279758660916, 'B60': -4.0752186670939804e-06, 'B64': 0.000688553331809375}

FWHMs = [np.interp(0, *resmods[irm].model)*3 for irm in [1,0,2,2]]
maxFWHMs = [5.0, 1.0, 20.0, 20.0]

cf = CrystalField('Pr', 'C4', Temperature=[7,7,7,20], FWHM=FWHMs, **B0)
cf.IntensityScaling = [0.1]*4
#cf.PeakShape = 'Gaussian'
#cf.PeakShape = 'PseudoVoigt'
cf.ToleranceIntensity = 0.1
cf.background = Background(background=Function('LinearBackground', A0=0.1, A1=0))

fit = CrystalFieldFit(Model=cf, InputWorkspace=[Ei23_7K_cut, Ei10_7K_cut, Ei100_7K_cut, Ei100_20K_cut],
                      MaxIterations=0, Output='fit')
#fit.fit()

e0 = [0, 0, 1.3, 2.3, 8.1, 10.2, 20, 26]

# importlib.reload(cef_utils)
# #c2 = cef_utils.fit_widths(fit, maxfwhm=maxFWHMs, method='L-BFGS-B', jac='3-point', options={'maxiter':200})
# c2 = cef_utils.fit_widths(fit, maxfwhm=maxFWHMs, method='Nelder-Mead', options={'maxiter':200})
# #c2 = cef_utils.fit_widths(fit, maxfwhm=maxFWHMs, method='trust-constr', jac='3-point', options={'maxiter':200})
# fit.fit()

# importlib.reload(cef_utils)
# res = cef_utils.fit_cef(fit, method='SLSQP', jac='3-point', options={'maxiter':600},
#     widths_kwargs={'maxfwhm':maxFWHMs, 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})

# res = cef_utils.fit_en(fit, e0, 
#    fit_alg='local', method='Nelder-Mead', jac='3-point', options={'maxiter':500},
#    widths_kwargs={'maxfwhm':maxFWHMs, 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})

do_global = False

globalg = 'differential_evolution'
#globalg = 'basinhopping'
#globalg = 'shgo'
#globalg = 'dual_annealing'
#globalg = 'direct'

if do_global:
    res = cef_utils.fit_en(fit, e0, is_voigt=True,
        fit_alg='global', algorithm=globalg, #fit_alg='gofit', options={'maxiter':100, 'samples':10},
        widths_kwargs={'maxfwhm':maxFWHMs, 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':10}})
    print(res)
    print(res.x.tolist())
bp = [] if 'bestpars' not in mtd else np.squeeze(mtd['bestpars'].extractY())
print(bp.tolist()) if hasattr(bp, 'tolist') else print(bp)


# Good parameters
bp = [-0.030481321832418973, -0.004304931818583363, 0.06308513873429943, 0.00020277082272632596, -0.00278830529188033]


if 1:#False:
    cf = CrystalField('Pr', 'C4', Temperature=[7,7,7,20], FWHM=FWHMs, **B0)
    cf.IntensityScaling = [1]*4
    #cf.PeakShape = 'Gaussian'
    #cf.PeakShape = 'PseudoVoigt'
    cf.ToleranceIntensity = 10
    cf.background = Background(background=Function('LinearBackground', A0=0.003, A1=0))
    #for jj in range(4): cf.constraints(f'0.8<IntensityScaling{jj}<1.2')
    
    fit = CrystalFieldFit(Model=cf, InputWorkspace=[Ei23_7K_cut, Ei10_7K_cut, Ei100_7K_cut, Ei100_20K_cut],
                          MaxIterations=0, Output='fit')
    chi2bp = cef_utils.fit_en(fit, e0, eval_only=bp, widths_kwargs={'maxfwhm':maxFWHMs, 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}},
                is_voigt=True)
    #try:
    #    fit.fit()
    #except:
    #    pass
    print(chi2bp)
    #cef_utils.genpp(fit)

localg = 'COBYLA'#'trust-constr'#'CG'#'BFGS'#'Powell'#'Nelder-Mead'
localg = 'Nelder-Mead'#'SLSQP'#'TNC'
res = cef_utils.fit_en(fit, e0, is_voigt=True,
   fit_alg='local', method=localg, jac='3-point', options={'maxiter':10},
   widths_kwargs={'maxfwhm':maxFWHMs, 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})
print(res)
print(res.x.tolist())
bp = np.squeeze(mtd['bestpars'].extractY())

print(bp.tolist()) if hasattr(bp, 'tolist') else print(bp)
cef_utils.printpars(fit)
print(r5(fit.model.getEigenvalues()))
print(fit.model.getPeakList())

cef_utils.genpp(fit)
