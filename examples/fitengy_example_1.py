# import mantid algorithms, numpy and matplotlib
#from mantid.simpleapi import *
import mantid.simpleapi as s_api
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from cef_utils import fitengy
import scipy.io
import scipy.optimize

from CrystalField import CrystalField, CrystalFieldFit, Background, Function
import mslice.cli as mc

# For RKNaNbO5, rare earth is at site 2c which has symmetry 4mm or C4v.
sym = 'C4v'

# Using values from PyChop
FWHM = 0.9

# There should be 2J+1 energy levels (depending on ion type)
ion = 'Nd'
# Assume there are three levels in the blob at 20meV and no higher energies.
# Nd3+ is a Kramers ion (4f^3) so we mush have doublets - need to explicitly state this
Elevels = [0, 0, 2.3, 2.3, 19.2, 19.2, 20.9, 20.9, 22.6, 22.6]

# Some good parameters:
fitBlm = {'B20': 0.10858727514593533, 'B40': -0.0018234869566891522, 'B44': -0.005517385707830169, 'B60': -3.0282931295383034e-05, 'B64': 0.001736039118092679}

# Maximum number of rounds of fitting to use to get best fit
# Set to zero to use good  parameters above
num_iter = 0
fit_cp = False

# ----------------------------------------

# Gets the data using MSlice.
if 'nd_fit_data_6K' not in s_api.mtd:
    MAR28881_40meV = mc.Load(Filename='MAR28881_40meV.nxspe', OutputWorkspace='MAR28881_40meV')
    MAR28891_40meV = mc.Load(Filename='MAR28891_40meV.nxspe', OutputWorkspace='MAR28891_40meV')
    ws_MAR28881_40meV_subtracted = mc.Minus(LHSWorkspace=MAR28881_40meV, RHSWorkspace=MAR28891_40meV, OutputWorkspace='MAR28881_40meV_subtracted')
    cut_ws_0 = mc.Cut(ws_MAR28881_40meV_subtracted, CutAxis="DeltaE,-30.0,30.0,0.25", IntegrationAxis="|Q|,0.0,3.0,0.0")
    s_api.ConvertMDHistoToMatrixWorkspace(InputWorkspace=cut_ws_0.raw_ws, OutputWorkspace='nd_fit_data_6K', Normalization='NumEventsNormalization', FindXAxis=False)
    s_api.Scale(InputWorkspace='nd_fit_data_6K', OutputWorkspace='nd_fit_data_6K', Factor=0.2515723270440251)
    s_api.ConvertToDistribution(Workspace='nd_fit_data_6K')

if 'nd_fit_data_200K' not in s_api.mtd:
    MAR28899_40meV = mc.Load(Filename='MAR28899_40meV.nxspe', OutputWorkspace='MAR28899_40meV')
    MAR28900_40meV = mc.Load(Filename='MAR28900_40meV.nxspe', OutputWorkspace='MAR28900_40meV')
    ws_MAR28899_40meV_subtracted = mc.Minus(LHSWorkspace=MAR28899_40meV, RHSWorkspace=MAR28900_40meV, OutputWorkspace='MAR28899_40meV_subtracted')
    cut_ws_0 = mc.Cut(ws_MAR28899_40meV_subtracted, CutAxis="DeltaE,-30.0,30.0,0.25", IntegrationAxis="|Q|,0.0,3.0,0.0")
    s_api.ConvertMDHistoToMatrixWorkspace(InputWorkspace=cut_ws_0.raw_ws, OutputWorkspace='nd_fit_data_200K', Normalization='NumEventsNormalization', FindXAxis=False)
    s_api.Scale(InputWorkspace='nd_fit_data_200K', OutputWorkspace='nd_fit_data_200K', Factor=0.2515723270440251)
    s_api.ConvertToDistribution(Workspace='nd_fit_data_200K')

# Loads Cp data
if 'cp_data_0T' not in s_api.mtd:
    cp_mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'NdKNaNbO5_cp.mat'))['hcs'][0]
    for ii, hh in enumerate([0, 1, 2, 3, 4.5, 6, 9]):
        s_api.CreateWorkspace(cp_mat[ii][:,0], cp_mat[ii][:,1], OutputWorkspace=f'cp_data_{hh}T')

################## Fits Cp vs field

def cp_min_fun(pp):
    rv = 0;
    fp = {'B20':pp[0], 'B40':pp[1], 'B44':pp[2], 'B60':pp[3], 'B64':pp[4]}
    cf = CrystalField(ion, sym, Temperature=1, **fp)
    cp0 = cf.getHeatCapacity(s_api.mtd['cp_data_0T'])
    rv = np.sum((cp0[1] - s_api.mtd['cp_data_0T'].extractY())**2)
    for h in [0, 1, 2, 3, 4.5, 6, 9]:
        cf = CrystalField(ion, sym, Temperature=1, BextX=h, **fp); cx = cf.getHeatCapacity(s_api.mtd[f'cp_data_{h}T'])
        cf = CrystalField(ion, sym, Temperature=1, BextY=h, **fp); cy = cf.getHeatCapacity(s_api.mtd[f'cp_data_{h}T'])
        cf = CrystalField(ion, sym, Temperature=1, BextZ=h, **fp); cz = cf.getHeatCapacity(s_api.mtd[f'cp_data_{h}T'])
        rv = rv + np.sum(( (cx[1]+cy[1]+cz[1])/3 - s_api.mtd[f'cp_data_{h}T'].extractY())**2)
    return np.sqrt(rv)

if fit_cp and num_iter == 0:
    p0 = [fitBlm['B20'], fitBlm['B40'], fitBlm['B44'], fitBlm['B60'], fitBlm['B64']]
    print(p0)
    res = scipy.optimize.minimize(cp_min_fun, p0, method='Nelder-Mead')
    print(res)
    print
    fitBlm = {'B20':res.x[0], 'B40':res.x[1], 'B44':res.x[2], 'B60':res.x[3], 'B64':res.x[4]}
print('---------')
print(cp_min_fun([fitBlm['B20'], fitBlm['B40'], fitBlm['B44'], fitBlm['B60'], fitBlm['B64']]))

# Fits the elastic line
elastic_pars = {}
for tt in [6, 200]:
    s_api.Fit(Function='name=PseudoVoigt,Mixing=0.5,Intensity=1,PeakCentre=0,FWHM=1.2', InputWorkspace=f'nd_fit_data_{tt}K', 
              Output=f'nd_fit_data_{tt}K', OutputCompositeMembers=True, StartX=-1, EndX=1)
    ws = s_api.mtd[f'nd_fit_data_{tt}K_Parameters']
    elastic_pars[tt] = {v['Name']:v['Value'] for v in [ws.row(i) for i in range(ws.rowCount())] if 'Cost function' not in v['Name']}

cost0 = 9e99
for iiter in range(num_iter):
    # Gets initial parameters by fitting just the energy levels (comment out if want to use good params above)
    fitBlm = fitengy(Ion=ion, E=Elevels, sym=sym)
    
    # Set up the crystal field fit
    cf = CrystalField(ion, sym, Temperature=[6, 200], FWHM=[FWHM, FWHM], **fitBlm)
    cf.background = Background(peak=Function('PseudoVoigt', **elastic_pars[6]), 
                               background=Function('LinearBackground', A0=0.001, A1=0))
    cf.background[1].peak = Function('PseudoVoigt', **elastic_pars[200])
    #cf.background.peak.ties(**elastic_pars)
    cf.IntensityScaling = [0.0005, 0.0005]
    cf.PeakShape = 'Lorentzian'
    cf.peaks[0].constrainAll('0.5 < FWHM < 3', 4)
    cf.peaks[1].constrainAll('0.5 < FWHM < 3', 10)
    fit = CrystalFieldFit(Model=cf, InputWorkspace=['nd_fit_data_6K', 'nd_fit_data_200K'], MaxIterations=10)
    fit.fit()
    fitpars = {pp:fit.model[pp] for pp in fitBlm.keys()}
    ws = s_api.mtd['fit_Parameters']
    cost = ws.row(ws.rowCount()-1)['Value']
    print(cost)
    if cost < cost0:
        bestfitBlm = fitBlm
        #if abs(cost - cost0) < 1e-5:
        #    break
        cost0 = cost
    fitBlm = bestfitBlm

#fitBlm = fitengy(Ion=ion, E=Elevels, **fitBlm)

# Set up the crystal field fit
cf = CrystalField(ion, sym, Temperature=[6, 200], FWHM=[FWHM, FWHM], **fitBlm)
cf.background = Background(peak=Function('PseudoVoigt', **elastic_pars[6]), 
                           background=Function('LinearBackground', A0=0.001, A1=0))
cf.background[1].peak = Function('PseudoVoigt', **elastic_pars[200])
#cf.background.peak.ties(**elastic_pars)
cf.IntensityScaling = [0.0005, 0.0005]
cf.PeakShape = 'Lorentzian'
cf.peaks[0].constrainAll('0.5 < FWHM < 3', 4)
cf.peaks[1].constrainAll('0.5 < FWHM < 3', 10)
fit = CrystalFieldFit(Model=cf, InputWorkspace=['nd_fit_data_6K', 'nd_fit_data_200K'], MaxIterations=10)
fit.fit()
fitpars = {pp:fit.model[pp] for pp in fitBlm.keys()}

np.set_printoptions(linewidth=200)
print('Fitted CF parameters:')
print(fitpars)
cf = CrystalField(ion, sym, Temperature=1, **fitpars)
print()
print('Peaks at base temperatures:')
print(cf.getPeakList())
print()
print('Energy Levels:')
print(cf.getEigenvalues())
ws = s_api.mtd['fit_Parameters']
cost = ws.row(ws.rowCount()-1)['Value']
print(f'Cost function value = {cost}')

cf = CrystalField(ion, sym, Temperature=1, **fitpars)
ws_tt = s_api.CreateWorkspace(np.linspace(0.01, 50, 2000), np.linspace(0.01, 50, 2000))
ws_cp = s_api.CreateWorkspace(*cf.getHeatCapacity(ws_tt))
for h in [0, 1, 2, 3, 4.5, 6, 9]:
    cf = CrystalField(ion, sym, Temperature=1, BextX=h, **fitpars); cx = cf.getHeatCapacity(ws_tt)
    cf = CrystalField(ion, sym, Temperature=1, BextY=h, **fitpars); cy = cf.getHeatCapacity(ws_tt)
    cf = CrystalField(ion, sym, Temperature=1, BextZ=h, **fitpars); cz = cf.getHeatCapacity(ws_tt)
    s_api.CreateWorkspace(cx[0], (cx[1]+cy[1]+cz[1])/3, OutputWorkspace=f'ws_cp{h}')


cf = CrystalField(ion, sym, Temperature=1, **fitpars)
ws_chi = s_api.CreateWorkspace(*cf.getSusceptibility(np.linspace(1,380,1000), Hdir='powder', Inverse=True))
ws_mag = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,90000,30000), Hdir='powder', Temperature=1))

# Loads the data and make a 2D slice
from mantid.simpleapi import mtd, MagFormFactorCorrection
slice_ws = mc.Slice(ws_MAR28881_40meV_subtracted, Axis1="|Q|, 0, 10, 0.05", Axis2="DeltaE,-5, 25, 0.25", NormToOne=False)

# Make 1D slice
cut_ws = mc.Cut(ws_MAR28881_40meV_subtracted, CutAxis='|Q|, 0, 10, 0.01', IntegrationAxis='DeltaE, 19.2, 22.71, 0.05')

# Calculate the form factor for Er3+, 
ion = 'Nd3'
ws_corr = MagFormFactorCorrection(slice_ws.raw_ws, IonName=ion, FormFactorWorkspace='FFCalc')
ws_fsquare = mtd['FFCalc'] * mtd['FFCalc']

# We rescale (to peak intensity of 180) the calculated form factor to match the data and add a background (120)
ws_fsq_scaled = ws_fsquare*0.025

# Now plot everything
fg, axes = plt.subplots(1, 2, subplot_kw={'projection':'mantid'}, figsize=(10,4))
axes[0].plot(cut_ws.raw_ws)
axes[0].set(xlim=(0,7), ylim=(0.0,0.05))
axes[0].plot(ws_fsq_scaled)
axes[1].pcolormesh(mc.Transpose(slice_ws).raw_ws, cmap='viridis', vmin=0, vmax=500)
# Convert Mantid workspace to numpy array


