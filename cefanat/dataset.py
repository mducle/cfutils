"""
Dataset class for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
"""
import numpy as np
import scipy
import h5py
import copy
import os

try:
    import mantid.simpleapi as s_api
    import mslice.cli as mc
except ModuleNotFoundError:
    s_api, mc = (None, None)

# The order of these definitions must match the order in GroupBoxes in the View
DATATYPES = ['INS', 'MH', 'MT', 'CHI', 'CP']
MAGUNITS = ['bohr', 'SI', 'cgs']
INSUNITS = ['meV', 'cm', 'THz']

DATATYPE_TO_IND = {k:v for v, k in enumerate(DATATYPES)}
MAGUNIT_TO_IND = {k:v for v, k in enumerate(MAGUNITS)}
INSUNIT_TO_IND = {k:v for v, k in enumerate(INSUNITS)}

def _recursive_parse_nxs(nxfield):
    if hasattr(nxfield, 'keys'):
        return {k:_recursive_parse_nxs(nxfield[k]) for k in nxfield.keys()}
    else:
        return None

class Dataset:
    """Data class - all data assumed to be 1D"""

    def __init__(self, dataarray=None, raw=None, name=None, intype='text', x_ind=0, y_ind=1, e_ind=2):
        self.name = name
        self.x_ind, self.y_ind, self.e_ind = (x_ind, y_ind, e_ind)
        self.array = dataarray
        self.inputtype = intype
        self.raw = raw
        self.datatype = 'INS'
        self.Hdir = 'powder'
        self.peaks = {k:None for k in ['guess', 'widths', 'par', 'trace']}
        self.elastic = {k:None for k in ['guess', 'par', 'trace']}
        self.sub_el = False
        self.mask_el = False

    @classmethod
    def from_file(cls, filename):
        extras = {}
        name = os.path.splitext(os.path.basename(filename))[0]
        match os.path.splitext(filename)[1]:
            case '.txt' | '.dat' | '.csv' | '.xye':
                with open(filename, 'r') as f:
                    raw = f.read()
                return cls(np.loadtxt(raw.splitlines()), raw, name, intype='text')
            case '.nxs':
                rawdic, dat, datargs = ({'filename':filename}, None, {})
                with h5py.File(filename, 'r') as f:
                    rawdic['root'] = _recursive_parse_nxs(f)
                    if 'mantid_workspace_1' in rawdic['root'] and 'workspace' in rawdic['root']['mantid_workspace_1']:
                        dset, axs = (f['mantid_workspace_1']['workspace'], ['axis1', 'values', 'errors'])
                        try:
                            x, y, e = (dset[fl] for fl in axs)
                        except KeyError:
                            pass
                        else:
                            if x.size == y.size + 1:
                                dat = np.array([(dset['axis1'][1:]+dset['axis1'][:-1])/2, np.squeeze(dset['values']), np.squeeze(dset['errors'])])
                            else:
                                dat = np.array([np.squeeze(dset[v]) for v in axs])
                            datargs = {f'{k}_ind':f'mantid_workspace_1/workspace/{v}' for k, v in zip(['x', 'y', 'e'], axs)}
                return cls(dat, rawdic, name, intype='nxs', **datargs)
            case '.mat':
                md, dat, datargs = (scipy.io.loadmat(filename), None, {})
                if all([c in md.keys() for c in ['x', 'y', 'e']]) and (len(md['x']) == len(md['y'])) and (len(md['x']) == len(md['e'])):
                    dat = np.concatenate((md['x'], md['y'], md['e']))  # MSlice saved file
                    datargs = {'x_ind':'x', 'y_ind':'y', 'e_ind':'e'}
                return cls(dat, md, name, intype='mat', **datargs)
            case '.nxspe':
                if mc is None:
                    raise RuntimeError('Mantid or MSlice not installed - cannot load nxspe file')
                rawdic = {'filename':filename, 'workspace':mc.Load(filename, OutputWorkspace=name)}
                sl = mc.Slice(rawdic['workspace'])
                rawdic.update(**{k:sl.get_coordinates()[v] for k, v in zip(['Q', 'E'], ['q', 'Energy transfer'])})
                rawdic['S'] = sl.get_signal()
                erng = 'DeltaE,' + ','.join([f'{c:.3f}' for c in rawdic['workspace'].limits['DeltaE']])
                qrng = '|Q|,' + ','.join([f'{c * f:.3f}' for c, f in zip(rawdic['workspace'].limits['|Q|'], [1, 0.5, 1])])
                cut = mc.Cut(rawdic['workspace'], CutAxis=erng, IntegrationAxis=qrng)
                dat = np.array([list(cut.get_coordinates().items())[0][1], cut.get_signal(), cut.get_error()])
                dat = dat[:, np.where(dat[2,:] != 0)[0]]
                return cls(dat, rawdic, name, intype='nxspe', x_ind=qrng, y_ind=erng, e_ind='e')

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
            if not isinstance(self.x_ind, str):
                if self.x_ind > self.array.shape[1]:
                    self.x_ind = 0
                if self.y_ind > self.array.shape[1] or self.y_ind == self.x_ind:
                    self.y_ind = 1

    def _indsetter(self, col, value):
        setattr(self, f'_{col}ind', value)
        if hasattr(self, 'inputtype') and hasattr(self, f'_{self.inputtype}_array_update'):
            getattr(self, f'_{self.inputtype}_array_update')()

    x_ind = property(lambda self: self._xind, lambda self, val: self._indsetter('x', val))
    y_ind = property(lambda self: self._yind, lambda self, val: self._indsetter('y', val))
    e_ind = property(lambda self: self._eind, lambda self, val: self._indsetter('e', val))

    @property
    def xyeind(self):
        return [self.x_ind, self.y_ind, self.e_ind]
    
    @property
    def xye(self):
        if isinstance(self.x_ind, str):
            y0 = self.array[:, [0, 1] if self.e_ind is None else [0, 1, 2]]
        else:
            y0 = self.array[:, [self.x_ind, self.y_ind] if self.e_ind is None else self.xyeind]
        if self.sub_el:
            y0[:,1] = y0[:,1] - self.elastic['trace']
        elif self.mask_el:
            x0, fwhm = tuple(self.elastic['par'][[0, 2]])
            idx = np.where((y0[:,0] > (x0 - 2*fwhm)) * (y0[:,0] < (x0 + 2*fwhm)))[0]
            y0 = copy.deepcopy(y0)
            y0[:,1][idx] = np.nan
        return y0

    @property
    def x(self):
        return self.array[:, 0 if isinstance(self.x_ind, str) else self.x_ind]

    @property
    def y(self):
        y0 = self.array[:, 1 if isinstance(self.x_ind, str) else self.y_ind]
        if self.mask_el:
            x0, fwhm = tuple(self.elastic['par'][[0, 2]])
            idx = np.where((self.x > (x0 - 2*fwhm)) * (self.x < (x0 + 2*fwhm)))[0]
            y0 = copy.deepcopy(y0)
            y0[idx] = np.nan
        return (y0 - self.elastic['trace']) if self.sub_el else y0

    @property
    def e(self):
        e_ind = 2 if isinstance(self.x_ind, str) else self.e_ind
        return [] if e_ind is None else self.array[:, e_ind]

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        match value.upper():
            case 'INS': self.unit, self.Temperature, self.Ei, self.H = ('meV', 0, 180, 0)
            case 'MH': self.unit, self.Temperature = ('bohr', 1)
            case 'MT': self.unit, self.H = ('bohr', 1)
            case 'CHI': self.unit, self.invchi = ('bohr', False)
            case 'CP': self.H = 0
            case _:
                raise RuntimeError(f'Unknown data type: {value}')
        self._datatype = value.upper()

    @property
    def datatype_index(self):
        return DATATYPE_TO_IND[self.datatype]
 
    @datatype_index.setter
    def datatype_index(self, value):
        self.datatype = DATATYPES[value]

    @property
    def dataunit_index(self):
        return INSUNIT_TO_IND[self.unit] if self.datatype == 'INS' else MAGUNIT_TO_IND[self.unit]

    @dataunit_index.setter
    def dataunit_index(self, value):
        self.unit = INSUNITS[value] if self.datatype == 'INS' else MAGUNITS[value]

    @property
    def h_unit(self):
        return 'T' if 'bohr' in self.unit or 'SI' in self.unit else 'Oe'

    @property
    def mag_unit(self):
        return 'uB/ion' if 'bohr' in self.unit else 'Am' if 'SI' in self.unit else 'emu/mol'

    @property
    def chi_unit(self):
        return 'uB/T/ion' if 'bohr' in self.unit else 'Am/T' if 'SI' in self.unit else 'emu/mol'

    @property
    def xlabel(self):
        match self.datatype:
            case 'INS': return f'Energy Transfer ({self.unit})'
            case 'MH': return f'Applied Field ({self.h_unit})'
            case 'MT': return 'Temperature (K)' 
            case 'CHI': return 'Temperature (K)' 
            case 'CP': return 'Temperature (K)'

    @property
    def ylabel(self):
        match self.datatype:
            case 'INS': return 'Intensity (arb. unit)'
            case 'MH': return f'Magnetic moment ({self.mag_unit})'
            case 'MT': return f'Magnetic moment ({self.mag_unit})'
            case 'CHI': return f'Inverse Susceptibility (1/{self.chi_unit})' if self.invchi else f'Susceptibility ({self.chi_unit})'
            case 'CP': return 'Magnetic Specific Heat (J/mol/K)'

    @property
    def sub_el(self):
        return self._sub_el

    @sub_el.setter
    def sub_el(self, value):
        self._sub_el = value and self.elastic['par'] is not None

    @property
    def mask_el(self):
        return self._mask_el

    @mask_el.setter
    def mask_el(self, value):
        self._mask_el = value and self.elastic['par'] is not None

    def _nxspe_array_update(self):
        cut = mc.Cut(self.raw['workspace'], CutAxis=self.y_ind, IntegrationAxis=self.x_ind)
        self.array = np.array([list(cut.get_coordinates().items())[0][1], cut.get_signal(), cut.get_error()])
        self.array = self.array[:, np.where(self.array[2,:] != 0)[0]]
        self.peaks['trace'], self.elastic['trace'] = (None, None)


class DataCollection:

    def __init__(self):
        self._keys = {}
        self._datavec = []

    @property
    def nset(self):
        return np.sum([1 for d in self._datavec if d.datatype == 'INS'])

    def __getitem__(self, ind):
        return self._datavec[ind] if isinstance(ind, int) else self._datavec[self._keys[ind]]

    def __setitem__(self, ind, val):
        if isinstance(val, str):
            val = Dataset.from_file(val)
        elif isinstance(val, tuple):
            val = Dataset(*val)
        elif not isinstance(val, Dataset):
            raise RuntimeError('DataCollection elements must be a file name, dataset values or Dataset object')
        self._keys[ind] = len(self._datavec)
        self._datavec.append(val)

    def __len__(self):
        return len(self._datavec)

    def __iter__(self):
        return iter(self._keys)

    def max_x(self):
        return np.max([np.max(d.x) for d in self._datavec if d.datatype == 'INS'])

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
