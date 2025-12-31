"""
GUI presenter code for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
The presenter class contains all logic for the GUI and interacts with the view and engine(s).
The state of the presenter can be de/serialised to json and represents the GUI state.
"""
import numpy as np
import scipy
import os
from .dataset import Dataset, DataCollection


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


class CEFAnaTPresenter():

    def __init__(self, view, engine=None):
        self.view = view
        self.engine = engine
        self.data = DataCollection()
        self._current_data = None
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

    def on_load_data(self):
        if (loaded := self.view.get_file('Text (*.txt *.dat *.csv *.xye);; NeXus (*.nxs);; Matlab (*.mat)')):
            for f in loaded:
                name, entry = _load_data(f)
                if (newname := self.view.update_data_list(name)):
                    self.data[newname] = entry
            if len(self.data) > 0:
                self.view.set_current_data(len(self.data) - 1)

    def on_change_data(self, current, previous):
        self._current_data = current
        self.view.update_data(self.data[current])

    def on_data_col_changed(self, d_ind, value):
        for ty, vl in zip(['x_ind', 'y_ind', 'e_ind'], value):
            setattr(self.data[d_ind], ty, vl)
        self.view.update_data(self.data[d_ind])

    def on_data_type_changed(self, ind):
        if self.view.is_noninteractive or self._current_data not in self.data:
            return
        self.data[self._current_data].datatype_index = ind
        self.view.plot_data(self.data[self._current_data])

    def on_data_unit_changed(self, ind):
        if self.view.is_noninteractive or self._current_data not in self.data:
            return
        self.data[self._current_data].dataunit_index = ind
        self.view.plot_data(self.data[self._current_data])

    def on_data_edit_finished(self, widg, prop):
        if self.view.is_noninteractive or self._current_data not in self.data:
            return
        setattr(self.data[self._current_data], prop, getattr(self.view, f'datainput_{widg}').text())
        
    def on_data_chiinv_changed(self, ischecked):
        if self.view.is_noninteractive or self._current_data not in self.data:
            return
        self.data[self._current_data].invchi = ischecked
