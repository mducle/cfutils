# import mantid algorithms, numpy and matplotlib
import mantid.simpleapi as s_api
import mslice.cli as mc

ws = mc.Load(Filename='MER46207_Ei7.00meV_Rings.nxspe', OutputWorkspace='MER46207_Ei7.00meV_Rings')
ws = mc.Cut(ws, CutAxis=f"DeltaE,-1,3.5,0.05", IntegrationAxis="|Q|,0.5,1.3,0.0", NormToOne=False, IntensityCorrection=False, SampleTemperature=None)
mws = s_api.ConvertMDHistoToMatrixWorkspace(InputWorkspace=ws.raw_ws, OutputWorkspace='MER46207_Ei7.00meV_Rings_scaled_cut(0.500,1.300)', Normalization='NumEventsNormalization', FindXAxis=False)
s_api.Scale(InputWorkspace=mws, OutputWorkspace='MER46207_Ei7.00meV_Rings_scaled_cut(0.500,1.300)', Factor=5*0.050420168067226934)
s_api.ConvertToDistribution(Workspace=mws)
s_api.SaveNexus(InputWorkspace=mws, Filename='mer46207_ei7_cut_paper.nxs', Title='mer46207_ei7_cut')
