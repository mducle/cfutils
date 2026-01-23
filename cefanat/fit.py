import numpy as np
import scipy.optimize
from .dataset import Dataset, DataCollection

def gauss(x, cen, area, fwhm):
    x = np.array(x)
    flgt = 4 * np.log(2)
    fac = np.sqrt(flgt / np.pi)
    return (area / fwhm * fac) * np.exp(-flgt * ((x - cen) / fwhm)**2)


def lorz(x, cen, area, fwhm):
    x = np.array(x)
    return (area / np.pi * (fwhm / 2)) / ( (x - cen)**2 + (fwhm / 2)**2 )


def voigt(x, cen, area, fwhm, frac=0.5):
    x = np.array(x)
    flgt = 4 * np.log(2)
    frac = max(0, min(frac, 1))
    return (area/fwhm) / (frac*np.pi/2 + (1-frac)*np.sqrt(np.pi / flgt)) \
        * (frac/(1 + 4*((x - cen)/fwhm)**2) + (1-frac)*np.exp(-flgt*((x - cen)/fwhm)**2))


def specfun(xdat, *pp, peakfun):
    np_per_pk = 4 if peakfun == voigt else 3
    npk = int(len(pp) / np_per_pk)
    ycalc = xdat*0
    for pk in range(npk):
        i0 = pk * np_per_pk
        ycalc += peakfun(xdat, *pp[i0:i0+np_per_pk])
    if len(pp) > npk * np_per_pk:
        ycalc += pp[-2] * xdat + pp[-1]
    return ycalc


def get_pk_init(x, y, cc, peakfun):
    hh = (cc[1] - np.nanmin(y)) / 1.5  # Makes peaks narrower otherwise could get flat lines
    x0 = np.argmin(np.abs(x - cc[0]))
    # Heuristic 1: finds width at half height
    xl = np.where(y[:x0] < hh)[0]
    xl = 0 if len(xl) == 0 else xl[-1]
    xr = np.where(y[x0:] < hh)[0]
    xr = len(x) if len(xr) == 0 else xr[0] + x0
    # Heuristic 2: finds average slope over next 3 points
    xl2 = np.polynomial.Polynomial.fit(y[x0-3:x0], x[x0-3:x0], deg=1)(hh) 
    xr2 = np.polynomial.Polynomial.fit(y[x0:x0+3], x[x0:x0+3], deg=1)(hh) 
    fwhm = min(x[xr], xr2) - max(x[xl], xl2)
    if peakfun == voigt:
        return [cc[0], cc[1] * fwhm / 2, fwhm, 0.5]
    return [cc[0], cc[1] * fwhm / 2, fwhm]


class Fit():

    def __init__(self, data=None):
        if data is None:
            self.data, self._current_data = (DataCollection(), None)
        else:
            self.data, self._current_data = (data, data[0])

    def add_data(self, name, value):
        self.data[name] = value

    def set_current_data(self, index):
        self._current_data = index

    def get_current_data(self):
        return self.data[self._current_data]

    def update_data_columns(self, columns, dataindex=None):
        if dataindex is not None:
            self._current_data = dataindex
        for ty, vl in zip(['x_ind', 'y_ind', 'e_ind'], columns):
            setattr(self.data[self._current_data], ty, vl)

    def is_current_data_valid(self):
        return self._current_data in self.data

    def set_current_data_type_index(self, ind):
        self.data[self._current_data].datatype_index = ind

    def set_current_data_unit_index(self, ind):
        self.data[self._current_data].dataunit_index = ind

    def set_current_data_invchi(self, ischecked):
        self.data[self._current_data].invchi = ischecked

    def set_current_data_peaks_guess(self, peaks):
        self.data[self._current_data].peaks = {k:None for k in ['guess', 'widths', 'par', 'trace']}
        if len(peaks) > 0:
            data = self.data[self._current_data]
            self.data[self._current_data].peaks['guess'] = peaks
            self.data[self._current_data].peaks['widths'] = [get_pk_init(data.x, data.y, cc, gauss)[2] for cc in peaks]

    def set_current_data_elastic_guess(self, cc):
        self.data[self._current_data].elastic = {k:None for k in ['guess', 'par', 'trace']}
        if cc.shape[0] > 0:
            x, y = (getattr(self.data[self._current_data], col) for col in ['x', 'y'])
            if cc.shape[0] > 2:
                fwhm = np.abs(cc[1,0] - cc[2,0])
                p0 = [cc[0,0], cc[1,0] * fwhm / 2, fwhm]
            else:
                p0 = get_pk_init(x, y, cc[0,:], voigt)
            self.data[self._current_data].elastic['guess'] = np.array(p0 + [0.01, np.nanmin(x)])

    def set_current_data_property(self, prop, val):
        setattr(self.data[self._current_data], prop, val)

    def fit_current_data_peaks(self, peakfun=voigt):
        x, y, e = (getattr(self.data[self._current_data], col) for col in ['x', 'y', 'e'])
        p0 = np.hstack([get_pk_init(x, y, cc, peakfun) for cc in self.data[self._current_data].peaks['guess']])
        popt, pcov = scipy.optimize.curve_fit(lambda xx, *pp: specfun(xx, *pp, peakfun=peakfun), x, y, p0, e if len(e) > 0 else None)
        self.data[self._current_data].peaks['par'] = popt
        self.data[self._current_data].peaks['trace'] = specfun(x, *popt, peakfun=peakfun)

    def fit_current_data_elastic(self):
        x, y, e = (getattr(self.data[self._current_data], col) for col in ['x', 'y', 'e'])
        y0 = np.nanmax(y[np.where(np.abs(x) < np.nanmax(x)/20)])
        p0 = self.data[self._current_data].peaks['guess']
        if p0 is None:
            p0 = get_pk_init(x, y, [0, y0], voigt) + [0.01, np.nanmin(x)]
        popt, pcov = scipy.optimize.curve_fit(lambda xx, *pp: specfun(xx, *pp, peakfun=voigt), x, y, p0, e if len(e) > 0 else None)
        self.data[self._current_data].elastic['par'] = popt
        self.data[self._current_data].elastic['trace'] = specfun(x, *popt, peakfun=voigt)
