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
        self.view.connect('dataloadbtn', 'clicked', self.on_load_data)
        self.view.connect('datalist', 'currentItemChanged', self.on_change_data)
        self.view.connect('datalist', 'comboChanged', self.on_data_col_changed)

    def on_load_data(self):
        if (loaded := self.view.get_file('Text (*.txt *.dat *.csv *.xye);; NeXus (*.nxs);; Matlab (*.mat)')):
            for f in loaded:
                name, entry = _load_data(f)
                if (newname := self.view.update_data_list(name)):
                    self.data[newname] = entry
            if len(self.data) > 0:
                self.view.set_current_data(len(self.data) - 1)

    def on_change_data(self, current, previous):
        self.view.update_data(self.data[current])

    def on_data_col_changed(self, d_ind, value):
        for ty, vl in zip(['x_ind', 'y_ind', 'e_ind'], value):
            setattr(self.data[d_ind], ty, vl)
        self.view.update_data(self.data[d_ind])
