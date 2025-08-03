"""
Example of using new fitting algorithm that matches the energy levels at each iteration
The system is SrTb2O4, published in Orlandi et al., Phys. Rev. B 111 054415 (2025)
https://doi.org/10.1103/PhysRevB.111.054415
"""

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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import importlib
import cef_utils
importlib.reload(cef_utils)

np.set_printoptions(linewidth=200)

# Fennell PRB 89 224511 (2014) shows data from HET (Fig 2)
#    but no parameters - just a level scheme (Fig 3):
#    SrDy2O4: Dy_4c1: 0, 4, 12, 21, 36, 38; Dy_4c2: 0, 29, 39
#    SrHo2O4: Ho_4c1: 0, 0.8, 2, 3.5, 6, 11, 41, 42; Ho_4c2: 0, 12, 13, 16, 17, 40, 41

# Conversion factor from Wybourne to Stevens normalisation
from math import sqrt
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


# Malkin PRB 92 094415 (2015) - parameters for SrY2O4:Er and SrEr2O4 (in "Standard" Stevens normalisation (with Stevens factor) in cm^-1)
malkin_pars_order = ['B20', 'B22', 'IB22', 'B40', 'B42', 'IB42', 'B44', 'IB44', 'B60', 'B62', 'IB62', 'B64', 'IB64', 'B66', 'IB66']
malkin_pars_er_R1 = [188, 137.5, -171.2, -57.3, -1066.2, 1165.2, -86.9, -972.3, -38, -22.3, 22.8, 30.1, -115.2, -162.2, -84]
malkin_pars_er_R2 = [17, -744, -125, -60.2, 1033.2, -977.8, 430.2, -685.6, -35.2, -68.4, -42.8, -80.2, -191.4, -119.6, 80.5]

# Nikitin Opt. i Spek. 131 441 (2023) - parameters for SrY2O4:Ho  (in "Standard" Stevens normalisation (with Stevens factor) in cm^-1)
nikitin_pars_ho_R1 = [200.3, 143.1, -142.6, -59.45, -1068.3, 1186.6, -62.4, -942, -40.95, -22.1, 23.1, 3.8, -151.1, -155.7, -99.3]
nikitin_pars_ho_R2 = [-8, -748, -133, -63, 1100, -981, 408, -715, -36.9, -70, -37.4, -73, -208, -115, 95]

malkin_pars_dy_R1 = [181.5, 90.2, -113.7, -64.2, -1074, 1180, -78.7,  -972.5,  -41.7, -40, 43.3,  24.7,  -141.9, -170.3, -89.25]
malkin_pars_dy_R2 = [-5, -729, -145, -64.7, 1103.2, -927.8, 380.2, -770.6, -37.2, -71.4, -37.8, -80.2, -211.4, -119.6, 90.5]

# Nikitin new paper
nikitin_pars_Tm1 = [187.3, 212.2, -146.8, -55.4, -1049.1, 953.2, -3.55, -877.4, -36.9, -39.3, 46.2, 9.72, -134.8, -200.0, -62.3]
nikitin_pars_Tm2 = [29.3, -786, -95.8, -57.2, 955.7, -791.6, 475.5, -565.4, -44.7, -5.86, -34.5, -88.9, -155, -122, 82.6]

# Converts parameters to "Neutron" convention Stevens parameters (without Stevens factor and in meV).
er1 = {k:v*thetakq['Er'][idx[k]]/8.066 for k,v in zip (malkin_pars_order, malkin_pars_er_R1)}
er2 = {k:v*thetakq['Er'][idx[k]]/8.066 for k,v in zip (malkin_pars_order, malkin_pars_er_R2)}
ho1 = {k:v*thetakq['Ho'][idx[k]]/8.066 for k,v in zip (malkin_pars_order, nikitin_pars_ho_R1)}
ho2 = {k:v*thetakq['Ho'][idx[k]]/8.066 for k,v in zip (malkin_pars_order, nikitin_pars_ho_R2)}
tm1 = {k:v1*thetakq['Tm'][idx[k]]/8.066 for k,v1 in zip (malkin_pars_order, nikitin_pars_Tm1)}
tm2 = {k:v1*thetakq['Tm'][idx[k]]/8.066 for k,v1 in zip (malkin_pars_order, nikitin_pars_Tm2)}

tb1 = {k:v1*thetakq['Tb'][idx[k]]/8.066 for k,v1 in zip (malkin_pars_order, malkin_pars_dy_R1)}
tb2 = {k:v1*thetakq['Tb'][idx[k]]/8.066 for k,v1 in zip (malkin_pars_order, malkin_pars_dy_R2)}

#tm1 = {k:v1/8.066 for k,v1 in zip (malkin_pars_order, nikitin_pars_Tm1)}
#tm2 = {k:v1/8.066 for k,v1 in zip (malkin_pars_order, nikitin_pars_Tm2)}
#print(tm1)
#print(tm2)

#fp1 = [-0.25203, -0.062942, 0.00041702, -0.007776, 0.00075104, -4.3136e-06, -0.00014614, -0.00011489, 5.399e-05, 0.18677, 0.015887, -0.012857, -5.8902e-05, 0.00010916, 7.9352e-05]
#fp2 = [0.13636, 0.92383, -0.0001386, 0.016615, 0.0054329, 4.7082e-06, -8.5791e-06, 3.9795e-05, 8.9077e-05, 0.17572, -0.015271, -0.010222, -8.4151e-06, 6.075e-05, 5.1648e-05]

fp1 = [-0.16912, -0.10186, 0.00014804, -0.0078605, -0.002352, -1.6262e-05, 1.3479e-05, -0.00012705, -7.3984e-05, 0.26959, 0.014414, -0.012302, 2.1008e-06, 0.00015075, 6.9006e-05]
fp2 = [0.047162, 0.78707, -0.00010101, 0.01817, 0.0073358, 3.2873e-06, 1.3075e-05, 1.7755e-05, 2.3871e-05, 0.21378, -0.019085, -0.012022, 1.9808e-05, -5.7097e-05, 3.9924e-05]
fitpar_order = ['B20', 'B22', 'B40', 'B42', 'B44', 'B60', 'B62', 'B64', 'B66', 'IB22', 'IB42', 'IB44', 'IB62', 'IB64', 'IB66']
tm1 = {k:v1/thetakq['Tb'][idx[k]] for k,v1 in zip (fitpar_order, fp1)}
tm2 = {k:v1/thetakq['Tb'][idx[k]] for k,v1 in zip (fitpar_order, fp2)}
tb1 = {k:v1 for k,v1 in zip (fitpar_order, fp1)}
tb2 = {k:v1 for k,v1 in zip (fitpar_order, fp2)}
tb1['B64' ] = tb1['B64'] / 10
tb1['IB64' ] = tb1['IB64'] / 10
print()
print(tm1)
print(tm2)

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
#mer46210_ei30_cut = Load(f'{curdir}/mer46210_ei30_cut.nxs')
mer46210_ei82_cut = Load(f'{datdir}/mer46210_ei82_cut_paper.nxs')

Blm_en1, Blm_en2 = (tb1, tb2)

FWHMs = [np.interp(0, *resmods[irm].model)*1.5 for irm in [0,1,3]]
 
cf1 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **Blm_en1)
#cf1.ResolutionModel = resmods  # Crashes if using resolution model
cf2 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **Blm_en2)
#cf2.ResolutionModel = resmods  # Crashes if using resolution model
cf = cf1 + cf2
#cf.PeakShape = 'Gaussian'
cf.PeakShape = 'PseudoVoigt'
cf.ToleranceIntensity = 20
#print(dir(cf))

#for ii in range(4):
#    cf.peaks[ii].constrainAll('0.5 < Sigma < 1.4', 6)

# Fix all CEF parameters and just fit the peak widths
tdic = {}
for pn in Blm_en1.keys():
    tdic[f'ion0.{pn}'] = Blm_en1[pn]
    tdic[f'ion1.{pn}'] = Blm_en2[pn]
cf.ties(tdic)
cf.constraints('ion0.B20 < 0')

# Gaussian background only for Ei=7meV
mev7gauss = Function('Gaussian', Height=26869.4, PeakCentre=0.007071, Sigma=0.145)
cf.background = Background(background=Function('LinearBackground', A0=45, A1=0),
                           peak=mev7gauss)

fit = CrystalFieldFit(Model=cf, InputWorkspace=[mer46207_ei7_cut, mer46207_ei18_cut, mer46210_ei82_cut],
                      MaxIterations=0, Output='fit')
fn = fit.model.function
fn.setParameter('sp1.IntensityScaling', 0.8)

# We use the Fit object to extract data and initial parameters for our own fit so don't need to run fit()
# here - if fit() was run here then the optimize parameters from this will be used as initial pars in our fit later.
#fit.fit()

# Some good parameters previously fitted
#c2=18386.86584234639; bp = [-0.22944678112849534, -0.12338457675455622, 0.0005720649564145049, -0.006839654051757746, 0.002971709870025127, -9.357197872608092e-06, -0.00015160393269016674, -0.00013274045367271903, 4.657266269384956e-05, 0.2044687465781279, 0.01620484979835499, -0.011520691549107008, -2.4383678654487005e-05, 0.0001660737874817983, 3.3896004625846956e-05, 0.13446103508129548, 0.9667216670517619, -8.840993527971507e-05, 0.015308, 0.00721241, 6.20744e-06, 1.47268e-05, 8.68551e-06, 3.77251e-06, 0.172559, -0.0148472, -0.010784, 2.62917e-05, 3.12198e-06, 2.45362e-06]
#c2=15453.000886364887; bp = [-0.1553698942008458, -0.11873775005427284, 2.1346833435589113e-05, -0.007017957930301513, -0.0017478002958664115, -1.550978635992538e-05, 1.7313310561109213e-05, -0.0001382400497361041, -7.836633719727565e-05, 0.240405575151383, 0.015419376300137427, -0.011067959956369375, -2.028169869266797e-05, 0.00014626866946865728, 4.679162469875585e-05, 0.04824340462449858, 0.6929608944254517, -0.00029899286772881424, 0.01999464339544023, 0.006441033973347532, 5.534267199921736e-06, 1.576530308175671e-05, 8.604100889925321e-06, 3.613961729871829e-06, 0.17864634300460625, -0.01912900138283579, -0.01203134832626349, 2.5007454989344703e-05, 3.0709109983285683e-06, 2.7215765298750454e-06]
c2=15389.496264834434; bp = [-0.15726033447010263, -0.121794439940964, 2.1797488592494495e-05, -0.007151518683324305, -0.0017831448817177019, -1.6397847135697645e-05, 1.8688256499126993e-05, -0.00013551111493153868, -7.870630154628545e-05, 0.24108533148802103, 0.015366318467031181, -0.011718813988565824, -2.0465801552405912e-05, 0.00014723812025898284, 4.703814737894199e-05, 0.04802911719367105, 0.7443884797493974, -0.00028510553611484817, 0.019835257748601773, 0.006378812619379386, 5.635703920303101e-06, 1.550558284346647e-05, 8.009048093650414e-06, 3.597630660736397e-06, 0.15857614665569283, -0.01888627394015916, -0.012397030987216043, 2.4939600883079418e-05, 2.952750144216143e-06, 2.3157675428880223e-06]
#c2=17419.225233398298; bp = [-0.22948934278967628, -0.12339782854644091, 0.0005720649564145049, -0.006822454167325428, 0.0030043101540715104, -9.357197872608092e-06, -0.00015160393269016674, -0.00013274045367271903, 4.6686905748070085e-05, 0.24227051592741058, 0.015827444467953623, -0.011507619588929053, -2.5436004399678387e-05, 0.0001660737874817983, 3.0327429564407763e-05, 0.13446871182791642, 0.9609928750431485, -9.672383049872361e-05, 0.015308, 0.00721241, 6.20744e-06, 1.47268e-05, 8.68551e-06, 3.77251e-06, 0.172559, -0.0148472, -0.010784, 2.62917e-05, 3.12198e-06, 2.45362e-06]

# Uncomment this to run the fit, otherwise it will evaluate the model with one of the good parameters above only.
# Note that fitting could take a very long time (~hours to days)
bp = None

# Use a local simplex fit by default, but could also use a global algorithm or GOFit (both commented out)
chi2bp = cef_utils.fit_en(fit, [[0, 0.7, 7.5, 30.35, 32.2], [0, 1.3, 12.2, 31.0]], eval_only=bp, 
    fit_alg='local', method='Nelder-Mead', jac='3-point', options={'maxiter':1}, # Use 1 iteration to make it short for tests
    #fit_alg='global', algorithm='differential_evolution', 
    #fit_alg='gofit', options={'maxiter':100, 'samples':10},
    widths_kwargs={'maxfwhm':[0.75, 7.0, 10.0], 'method':'Nelder-Mead', 'jac':'3-point', 'options':{'maxiter':10}})

cfpars, cfobjs, peaks, intscal, origwidths = cef_utils.parse_cef_func(fit.model.function)
cef_utils.genpp(fit)
cef_utils.printpars(fit)
# Adds elastic line to fit workspace for 7meV data
ws_7gauss = EvaluateFunction(mev7gauss.toString(), 'fit_Workspace_0', OutputWorkspace='ws_7gauss')
ws2 = mtd['fit_Workspace_0']
ws2.setY(1, ws2.readY(1) + ws_7gauss.readY(1))
ws2.setY(2, ws2.readY(0) - ws2.readY(1))
fit_Workspace_0 = ws2 + 0.001
for ii, ei in enumerate([7, 30, 82]):
    np.savetxt(f'/tmp/ins_ei{ei}_calc.dat', np.array([mtd[f'fit_Workspace_{ii}'].readX(1), mtd[f'fit_Workspace_{ii}'].readY(1)]).T, header=f'Calculated INS data for SrTb2O4 Ei={ei}meV\nEn Intensity')
icdat = [mtd['invchi_0_x'].readX(0)]
magdat = [mtd['mag_0_x'].readX(0)]
for ii in ['0_x', '0_y', '0_z', '1_x', '1_y', '1_z']:
    icdat.append(mtd[f'invchi_{ii}'].readY(0))
    magdat.append(mtd[f'mag_{ii}'].readY(0))
np.savetxt('/tmp/invchi.dat', np.array(icdat).T, header='Calculated inverse susceptibility for SrTb2O4 in (mole/emu)\nT(K) site_1_x site_1_y site_1_z site_2_x site_2_y site_2_z')
np.savetxt('/tmp/mag.dat', np.array(magdat).T, header='Calculated magnetistion for SrTb2O4 in (uB/Tb-ion)\nH(Tesla) site_1_x site_1_y site_1_z site_2_x site_2_y site_2_z')
print('Site 1')
print(cfobjs[0][0].printWavefunction(range(13)))
print('Site 2')
print(cfobjs[1][0].printWavefunction(range(13)))
print(chi2bp)
print(cfobjs[0][0].getEigenvalues())
print(cfobjs[1][0].getEigenvalues())

## Calculates the local g-factors in the x, y, z directions
# g_{x,y} = 2gJ<gs1|Jx,iJy|gs2>; g_z = 2gJ<gs1|Jz|gs1>
L = 3
S = 3
J = 6
gJ = 1.5 + (S*(S+1) - L*(L+1)) / (2*J*(J+1))

# Gets the ground state wavefunctions for each site
gs1 = (cfobjs[0][0].getEigenvectors()[:,0])
gs2 = (cfobjs[1][0].getEigenvectors()[:,0])

# Gets the magnetic operators for Tb3+ (ion index 8) [actually we're getting the Zeeman operators rather than J]
# the field is 1/(gJ*uB) in Tesla otherwise the matrix elements are in Tesla rather than dimensionless
uB = scipy.constants.physical_constants['Bohr magneton in eV/T'][0] * 1000
_, _, Jx = energies(8, BextX=1./(uB*gJ))
_, _, Jy = energies(8, BextY=1./(uB*gJ))
_, _, Jz = energies(8, BextZ=1./(uB*gJ))
gx1 = 2 * gJ * np.dot(np.conj(gs1), np.dot(Jx, gs1))
gy1 = 2 * gJ * np.dot(np.conj(gs1), np.dot(Jy*1j, gs1))
gz1 = 2 * gJ * np.dot(np.conj(gs1), np.dot(Jz, gs1))
gx2 = 2 * gJ * np.dot(np.conj(gs2), np.dot(Jx, gs2))
gy2 = 2 * gJ * np.dot(np.conj(gs2), np.dot(Jy*1j, gs2))
gz2 = 2 * gJ * np.dot(np.conj(gs2), np.dot(Jz, gs2))
print('gx = {}; gy = {}; gz = {}'.format(np.real(gx1), np.real(gy1), np.real(gz1)))
print('gx = {}; gy = {}; gz = {}'.format(np.real(gx2), np.real(gy2), np.real(gz2)))

# Uncomment out to plot
"""
import matplotlib.pyplot as plt
from mantid.plots.utility import MantidAxType
from mantid.api import AnalysisDataService as ADS
from mantid.simpleapi import mtd

if 'fit_Workspace_0' not in mtd:
    raise RuntimeError('You must run the fit in the cef_fit.py script first')

fit_Workspace_2 = ADS.retrieve('fit_Workspace_2')
fit_Workspace_1 = ADS.retrieve('fit_Workspace_1')
#fit_Workspace_3 = ADS.retrieve('fit_Workspace_3')
fit_Workspace_0 = ADS.retrieve('fit_Workspace_0')

fig, axes = plt.subplots(edgecolor='#ffffff', num='fit_Workspace_0-1', subplot_kw={'projection': 'mantid'})
axes.plot(fit_Workspace_0, color='#1f77b4', label='Ei=7 7K calc', markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e', wkspIndex=1)
axes.plot(fit_Workspace_1, color='#ff7f0e', label='Ei=30 7K calc', markeredgecolor='#d62728', markerfacecolor='#d62728', wkspIndex=1)
axes.plot(fit_Workspace_2, color='#2ca02c', label='Ei=82 7K calc', markeredgecolor='#8c564b', markerfacecolor='#8c564b', wkspIndex=1)
#axes.plot(fit_Workspace_3, color='#e377c2', label='Ei=100 20K calc', markeredgecolor='#7f7f7f', markerfacecolor='#7f7f7f', wkspIndex=1)
axes.errorbar(fit_Workspace_1, color='#2ca02c', ecolor='#ff7f0e', elinewidth=1.0, label='Ei=30 7K', linestyle='None', marker='.', markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e', wkspIndex=0)
axes.errorbar(fit_Workspace_2, color='#2ca02c', elinewidth=1.0, label='Ei=82 7K', linestyle='None', marker='.', wkspIndex=0)
#axes.errorbar(fit_Workspace_3, color='#e377c2', elinewidth=1.0, label='Ei=100 20K', linestyle='None', marker='.', wkspIndex=0)
axes.errorbar(fit_Workspace_0, color='#1f77b4', elinewidth=1.0, label='Ei=7 7K', linestyle='None', marker='.', wkspIndex=0)
axes.tick_params(axis='x', which='major', **{'gridOn': False, 'tick1On': True, 'tick2On': False, 'label1On': True, 'label2On': False, 'size': 6, 'tickdir': 'out', 'width': 1})
axes.tick_params(axis='y', which='major', **{'gridOn': False, 'tick1On': True, 'tick2On': False, 'label1On': True, 'label2On': False, 'size': 6, 'tickdir': 'out', 'width': 1})
axes.set_title('StTb2O4 - CEF Fit')
#axes.set_xlim([-2.4228637113068211, 41.71124999970198])
axes.set_xlim([0.4228637113068211, 41.71124999970198])
axes.set_ylim([29.271179077666755, 3454.0800416098996])
#axes.set_xscale('log')
#axes.set_yscale('log')
legend = axes.legend(fontsize=8.0).set_draggable(True).legend

plt.show()
# Scripting Plots in Mantid:
# https://docs.mantidproject.org/tutorials/python_in_mantid/plotting/02_scripting_plots.html
"""
