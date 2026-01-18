"""
GUI presenter code for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
The presenter class contains all logic for the GUI and interacts with the view and engine(s).
The state of the presenter can be de/serialised to json and represents the GUI state.
"""
import numpy as np
import scipy
import os
import importlib
import traceback
from scipy.optimize._minimize import MINIMIZE_METHODS
import scipy.optimize
from .dataset import Dataset
from .fit import Fit
from .engine import EngineFactory

GLOBAL_METHODS = [mt for mt in ['basinhopping', 'differential_evolution', 'shgo', 'dual_annealing', 'direct'] if hasattr(scipy.optimize, mt)]
CURVEFIT_METHODS = ['lm', 'trf', 'dogbox']

# Finds all engines in subfolder
for eng in [ff for ff in os.listdir(os.path.join(os.path.dirname(__file__), 'engines')) if ff.endswith('.py')]:
    try:
        importlib.import_module(f'.engines.{eng.split(".py")[0]}', '.'.join(__name__.split('.')[:-1]))
    except ModuleNotFoundError:  # Either Mantid or libMcPhase not installed
        pass
if len(EngineFactory.list())==0:
    raise RuntimeError('No calculation engine found! Please install either Mantid or libMcPhase')


def _load_data(filename):
    extras = {}
    name = os.path.splitext(os.path.basename(filename))[0]
    match os.path.splitext(filename)[1]:
        case '.txt' | '.dat' | '.csv' | '.xye':
            with open(filename, 'r') as f:
                raw = f.read()
            return name, Dataset(np.loadtxt(raw.splitlines()), raw, intype='text')
        case '.nxs':
            return name, Dataset(None, None, intype='nxs')
        case '.mat':
            return name, Dataset(None, None, intype='mat')

    
def display_error(func):
    def inner(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
            self.view.display_error(e.message if hasattr(e, 'message') else str(e))
    return inner 


class CEFAnaTPresenter():

    def __init__(self, view):
        self.view = view
        self.fit = Fit()
        self.engines = EngineFactory.list()
        # Make McPhase the default engine if it is found
        self.calc_engine = 'McPhaseEngine' if 'McPhaseEngine' in self.engines else self.engines[0]
        self.view.connect('dataloadbtn', 'clicked', self.on_load_data)
        self.view.connect('datalist', 'currentItemChanged', self.on_change_data)
        self.view.connect('datalist', 'comboChanged', self.on_data_col_changed)
        self.view.connect('datatype', 'changed', self.on_data_type_changed)
        for widg, prop in zip(['instt', 'mhtt', 'mth', 'cph', 'insHdir', 'mthdir', 'cphdir', 'insEi', 'insH'],
            ['Temperature']*2 + ['H', 'H', 'Hdir', 'Hdir', 'Hdir', 'Ei', 'H']):
            self.view.connect(f'datainput_{widg}', 'editingFinished', lambda w=widg, p=prop: self.on_data_edit_finished(w,p))
        for widg in ['ins', 'mh', 'mt', 'chi']:
            self.view.connect(f'datainput_{widg}unit', 'changed', self.on_data_unit_changed)
        self.view.connect('datainput_chiinv', 'clicked', self.on_data_chiinv_changed)
        self.view.connect('datatoolsaddpk', 'clicked', self.on_data_add_peak)
        self.view.connect('datatoolsfitpk', 'clicked', self.on_data_fit_peak)
        self.view.set_fit_local_minimizers(MINIMIZE_METHODS)
        self.view.set_fit_global_minimizers(MINIMIZE_METHODS)
        self.view.set_fit_global(GLOBAL_METHODS)
        self.view.connect('fitlocalalgo', 'changed', self.on_fit_localalgo_changed)
        self.view.set_engines(self.engines)
        self.view.set_calc_engine(self.calc_engine)
        self.view.connect('enginegroup', 'triggered', self.on_engine_change)

    @display_error
    def on_load_data(self, ischecked):
        if (loaded := self.view.get_file('Text (*.txt *.dat *.csv *.xye);; NeXus (*.nxs);; Matlab (*.mat)')):
            for f in loaded:
                name, entry = _load_data(f)
                if (newname := self.view.update_data_list(name)):
                    self.fit.add_data(newname, entry)
            if len(self.fit.data) > 0:
                self.view.set_current_data(len(self.fit.data) - 1)

    @display_error
    def on_change_data(self, current, previous):
        self.fit.set_current_data(current)
        self.view.update_data(self.fit.get_current_data())

    @display_error
    def on_data_col_changed(self, d_ind, value):
        self.fit.update_data_columns(value, d_ind)
        self.view.update_data(self.fit.get_current_data())

    @display_error
    def on_data_type_changed(self, ind):
        if self.view.is_noninteractive or not self.fit.is_current_data_valid():
            return
        self.fit.set_current_data_type_index(ind)
        self.view.update_data(self.fit.get_current_data())

    @display_error
    def on_data_unit_changed(self, ind):
        if self.view.is_noninteractive or not self.fit.is_current_data_valid():
            return
        self.fit.set_current_data_unit_index(ind)
        self.view.plot_data(self.fit.get_current_data())

    @display_error
    def on_data_edit_finished(self, widg, prop):
        if self.view.is_noninteractive or not self.fit.is_current_data_valid():
            return
        self.fit.set_current_data_property(prop, getattr(self.view, f'datainput_{widg}').text())
        
    @display_error
    def on_data_chiinv_changed(self, ischecked):
        if self.view.is_noninteractive or not self.fit.is_current_data_valid():
            return
        self.fit.set_current_data_invchi(ischecked)

    @display_error
    def on_fit_localalgo_changed(self, index):
        if index == 0:
            self.view.set_fit_local_minimizers(MINIMIZE_METHODS)
        else:
            self.view.set_fit_local_minimizers(CURVEFIT_METHODS)

    @display_error
    def on_data_add_peak(self, ischecked):
        self.fit.set_current_data_peaks_guess(np.array(self.view.get_peaks()))
        self.view.plot_data(self.fit.get_current_data())

    @display_error
    def on_data_fit_peak(self, ischecked):
        self.fit.fit_current_data_peaks()
        self.view.plot_data(self.fit.get_current_data())

    @display_error
    def on_engine_change(self, engineobj):
        self.calc_engine = self.view.get_engine_name(engineobj)
