"""
Dataset class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
import numpy as np

class Dataset:
    """Helper data class - all data assumed to be 1D"""

    def __init__(self, dataarray=None, raw=None, intype='text', x_ind=0, y_ind=1, e_ind=2):
        self.x_ind, self.y_ind, self.e_ind = (x_ind, y_ind, e_ind)
        self.array = dataarray
        self.inputtype = intype
        self.raw = raw
        self.datatype = 'INS'

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        if value is None:
            self._array = None
        else:
            self._array = np.array(value)
            assert len(self._array.shape) == 2, 'Input array must be a 2D, n-by-2 or n-by-3 array'
        if self._array.shape[0] < 4 and self.array.shape[1] > self.array.shape[0]:
            self._array = self.array.T
        assert min(self._array.shape) > 1, 'Input must be an n-by-m array, with m >= 2'
        if self._array.shape[1] == 2:
            self.e_ind = None
        if self.x_ind > self.array.shape[1]:
            self.x_ind = 0
        if self.y_ind > self.array.shape[1] or self.y_ind == self.x_ind:
            self.y_ind = 1

    @property
    def xyeind(self):
        return [self.x_ind, self.y_ind, self.e_ind]
    
    @property
    def xye(self):
        return self.array[:, self.xyeind()]

    @property
    def x(self):
        return self.array[:, self.x_ind]

    @property
    def y(self):
        return self.array[:, self.y_ind]

    @property
    def e(self):
        return self.array[:, self.e_ind] if self.e_ind else []


class DataCollection:

    def __init__(self):
        self._keys = {}
        self._datavec = []

    @property
    def nset(self):
        return len(self._datavec)

    def __getitem__(self, ind):
        return self._datavec[ind] if isinstance(ind, int) else self._datavec[self._keys[ind]]

    def __setitem__(self, ind, val):
        self._keys[ind] = len(self._datavec)
        self._datavec.append(val)

    def __len__(self):
        return len(self._datavec)

    def max_x(self):
        return np.max([np.max(self._datavec[ii].x) for ii in range(self.nset)])

"""
    def get_data_from_workspace(self, InputWorkspace):
        ws = s_api.mtd[InputWorkspace] if isinstance(InputWorkspace, str) else InputWorkspace
        x = np.squeeze(ws.extractX())
        assert len(x.shape) == 1, "Error: input workspace must be 1D"
        y = np.squeeze(ws.extractY())
        e = np.squeeze(ws.extractE())
        if len(x) == (len(y) + 1):
            x = (x[:-1] + x[1:]) / 2.0
        assert len(x) == len(y), "Error: x- and y- dimensions are not consistent"
        return x, y, e
"""
