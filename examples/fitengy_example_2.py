# import mantid algorithms, numpy and matplotlib
#from mantid.simpleapi import *
import mantid.simpleapi as s_api
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from cef_fitengy import fitengy

from CrystalField import CrystalField, CrystalFieldFit, Background, Function
import mslice.cli as mc

# From PRB 105 205112 (2022), Yb is at Wyckoff site 4i with m or Cs symmetry
sym = 'C2v'

# PyChop estimates FWHM is 1.9meV but peaks are significantly broader
FWHM = 2.5
Temperature = 6

# There should be 2J+1 energy levels (depending on ion type)
ion = 'Yb'
# Yb3+ is a Kramers ion (4f^13) so we mush have doublets - need to explicitly state this
Elevels = [0, 0, 10.9, 10.9, 25.2, 25.2, 34.1, 34.1]

# Internal field in Tesla
Bint = 20

# Some good parameters:
fitBlm = {'B20': 0.43673030615762026, 'B22': -0.5540820922709659, 'B40': -0.007098433796422648, 'B42': 0.034251868969617384, 'B44': -0.06272863302853617, 'B60': -0.0006409306494764693, 'B62': 0.0019763015519198342, 'B64': -0.0015487698506927007, 'B66': 0.00351393}

# Maximum number of rounds of fitting to use to get best fit
# Set to zero to use good  parameters above
num_iter = 0#10000

# ----------------------------------------

# Gets the data using MSlice.
if 'yb_fit_data' not in s_api.mtd:
    MAR28873_70meV = mc.Load(Filename='MAR28873_70meV.nxspe', OutputWorkspace='MAR28873_70meV')
    cut_ws_0 = mc.Cut(MAR28873_70meV, CutAxis="DeltaE,-10.0,39.0,0.5", IntegrationAxis="|Q|,0.0,2.0,0.0")
    s_api.ConvertMDHistoToMatrixWorkspace(InputWorkspace=cut_ws_0.raw_ws, OutputWorkspace='yb_fit_data', Normalization='NumEventsNormalization', FindXAxis=False)
    s_api.Scale(InputWorkspace='yb_fit_data', OutputWorkspace='yb_fit_data', Factor=0.50420168067226889)
    s_api.ConvertToDistribution(Workspace='yb_fit_data')

# Fits the elastic line
s_api.Fit(Function='name=PseudoVoigt,Mixing=0.5,Intensity=1,PeakCentre=0,FWHM=2.0', InputWorkspace='yb_fit_data', 
          Output='yb_fit_data', OutputCompositeMembers=True, StartX=-10, EndX=5)
ws = s_api.mtd['yb_fit_data_Parameters']
elastic_pars = {v['Name']:v['Value'] for v in [ws.row(i) for i in range(ws.rowCount())] if 'Cost function' not in v['Name']}

cost0 = 9e99
for iiter in range(num_iter):
    # Gets initial parameters by fitting just the energy levels (comment out if want to use good params above)
    fitBlm = fitengy(Ion=ion, E=Elevels, sym=sym)
    # Set up the crystal field fit
    cf = CrystalField(ion, sym, Temperature=Temperature, FWHM=FWHM, **fitBlm)
    cf.background = Background(peak=Function('PseudoVoigt', **elastic_pars), 
                               background=Function('LinearBackground', A0=0.0005, A1=0))
    #cf.background.peak.ties(**elastic_pars)
    cf.PeakShape = 'Gaussian'
    cf.IntensityScaling = 0.0002
    try:
        cf.peaks.constrainAll('0.5 < Sigma < 1.4', 6)
    except IndexError:
        cf.peaks.constrainAll('0.5 < Sigma < 1.4', 5)
    cf.ties(BextZ=Bint)
    fit = CrystalFieldFit(Model=cf, InputWorkspace='yb_fit_data', MaxIterations=10)
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

#if num_iter == 0:
#    Elevels = [0, 0, 10.5, 11.5, 24.7, 24.7, 32.1, 32.1]
#    fitBlm = fitengy(Ion=ion, E=Elevels, **fitBlm)
cf = CrystalField(ion, sym, Temperature=Temperature, FWHM=FWHM, **fitBlm)
cf.background = Background(peak=Function('PseudoVoigt', **elastic_pars), 
                           background=Function('LinearBackground', A0=0.0005, A1=0))
#cf.background.peak.ties(**elastic_pars)
cf.PeakShape = 'Gaussian'
cf.IntensityScaling = 0.0002
cf.peaks.constrainAll('0.5 < Sigma < 2.5', 6)
cf.ties(BextZ=Bint)
if num_iter == 0:
    cf.ties(**fitBlm)
fit = CrystalFieldFit(Model=cf, InputWorkspace='yb_fit_data', MaxIterations=10)
fit.fit()
fitpars = {pp:fit.model[pp] for pp in fitBlm.keys()}

# Generates physical properties workspaces.
cf = CrystalField(ion, sym, Temperature=1, **fitpars)
susc_p = s_api.CreateWorkspace(*cf.getSusceptibility(np.linspace(1,300,300), Hdir='powder', Inverse=True, Unit='cgs'))
susc_x = s_api.CreateWorkspace(*cf.getSusceptibility(np.linspace(1,300,300), Hdir=[1,0,0], Inverse=True, Unit='cgs'))
susc_y = s_api.CreateWorkspace(*cf.getSusceptibility(np.linspace(1,300,300), Hdir=[0,1,0], Inverse=True, Unit='cgs'))
susc_z = s_api.CreateWorkspace(*cf.getSusceptibility(np.linspace(1,300,300), Hdir=[0,0,1], Inverse=True, Unit='cgs'))
mag_p08 = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,6,60), Hdir='powder', Temperature=8, Unit='bohr'))
mag_p15 = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,6,60), Hdir='powder', Temperature=15, Unit='bohr'))
mag_p25 = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,6,60), Hdir='powder', Temperature=25, Unit='bohr'))
mag_x = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,15,150), Hdir=[1,0,0], Temperature=1, Unit='bohr'))
mag_y = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,15,150), Hdir=[0,1,0], Temperature=1, Unit='bohr'))
mag_z = s_api.CreateWorkspace(*cf.getMagneticMoment(Hmag=np.linspace(0,15,150), Hdir=[0,0,1], Temperature=1, Unit='bohr'))

np.set_printoptions(linewidth=200)
print('Fitted CF parameters:')
#print(fitBlm)
print(fitpars)
cf = CrystalField(ion, sym, Temperature=1, BextZ=Bint, **fitpars)
print()
print('Peaks at base temperatures:')
print(cf.getPeakList())
print()
print('Energy Levels:')
print(cf.getEigenvalues())
print()
ws = s_api.mtd['fit_Parameters']
cost = ws.row(ws.rowCount()-1)['Value']
print(f'Cost function value = {cost}')
