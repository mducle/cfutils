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


def get_pk_init(x, y, cc, peakfun, widths=None):
    if widths is None:
        hh = (cc[1] - np.nanmin(y)) / 1.5  # Makes peaks narrower otherwise could get flat lines
        x0 = np.argmin(np.abs(x - cc[0]))
        # Heuristic 1: finds width at half height
        xl = np.where(y[:x0] < hh)[0]
        xl = 0 if len(xl) == 0 else xl[-1]
        xr = np.where(y[x0:] < hh)[0]
        xr = len(x) if len(xr) == 0 else xr[0] + x0
        # Heuristic 2: finds average slope over next 3 points
        try:
            xl2 = np.polynomial.Polynomial.fit(y[max(x0-3,0):x0], x[max(x0-3,0):x0], deg=1)(hh) 
            xr2 = np.polynomial.Polynomial.fit(y[x0:min(x0+3,len(y))], x[x0:min(x0+3,len(y))], deg=1)(hh) 
        except np.linalg.LinAlgError:
            fwhm = abs(x[xr] - x[xl])
        else:
            fwhm = abs(min(x[xr], xr2) - max(x[xl], xl2))
    else:
        fwhm = widths
    if peakfun == voigt:
        return [cc[0], cc[1] * fwhm / 2, fwhm, 0.5]
    return [cc[0], cc[1] * fwhm / 2, fwhm]


def curvefit(fun, xdat, ydat, p0, edat=None, *args, **kwargs):
    if any(np.isnan(ydat)):
        idx = np.where(~np.isnan(ydat))[0]
        xdat, ydat = (v[idx] for v in [xdat, ydat])
        edat = edat if edat is None else edat[idx]
    return scipy.optimize.curve_fit(fun, xdat, ydat, p0, edat, *args, **kwargs)


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
        if hasattr(columns, '__iter__') and len(columns) == 3:
            for ty, vl in zip(['x_ind', 'y_ind', 'e_ind'], columns):
                setattr(self.data[self._current_data], ty, vl)
        elif hasattr(columns, '__iter__') and len(columns) == 2 and isinstance(columns[0], str):
            setattr(self.data[self._current_data], columns[0], columns[1])

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

    def update_current_data_peaks_guess(self, peaks, widths):
        self.data[self._current_data].peaks['guess'] = peaks
        self.data[self._current_data].peaks['widths'] = widths

    def set_current_data_elastic_guess(self, cc):
        self.data[self._current_data].elastic = {k:None for k in ['guess', 'par', 'trace']}
        if cc.shape[0] > 0:
            x, y = (getattr(self.data[self._current_data], col) for col in ['x', 'y'])
            if cc.shape[0] > 2:
                fwhm = np.abs(cc[1,0] - cc[2,0])
                p0 = [cc[0,0], cc[0,1] * fwhm, fwhm, 0.5]
            else:
                p0 = get_pk_init(x, y, cc[0,:], voigt)
            self.data[self._current_data].elastic['guess'] = np.array(p0 + [1e-12, np.nanmin(y)])

    def set_current_data_property(self, prop, val):
        setattr(self.data[self._current_data], prop, val)

    def fit_current_data_peaks(self, peakfun=voigt):
        data, widths = (self.data[self._current_data], self.data[self._current_data].peaks['widths'])
        x, y, e = (getattr(data, col) for col in ['x', 'y', 'e'])
        p0 = np.hstack([get_pk_init(x, y, cc, peakfun, wd) for cc, wd in zip(data.peaks['guess'], widths)] + [1e-12, np.nanmin(x)])
        popt, pcov = curvefit(lambda xx, *pp: specfun(xx, *pp, peakfun=peakfun), x, y, p0, e if len(e) > 0 else None)
        self.data[self._current_data].peaks['par'] = popt
        self.data[self._current_data].peaks['trace'] = specfun(x, *popt, peakfun=peakfun)

    def fit_current_data_elastic(self):
        x, y, e = (getattr(self.data[self._current_data], col) for col in ['x', 'y', 'e'])
        p0 = self.data[self._current_data].elastic['guess']
        if p0 is None:
            y0 = np.nanmax(y[np.where(np.abs(x) < np.nanmax(x)/10)])
            x0 = x[np.where(y == y0)[0][0]]
            p0 = get_pk_init(x, y, [x0, y0], voigt) + [1e-12, np.nanmin(y)]
        popt, pcov = curvefit(lambda xx, *pp: specfun(xx, *pp, peakfun=voigt), x, y, p0, e if len(e) > 0 else None)
        self.data[self._current_data].elastic['par'] = popt
        self.data[self._current_data].elastic['trace'] = specfun(x, *popt, peakfun=voigt)

    def current_data_toggle_elastic(self):
        self.data[self._current_data].sub_el = not self.data[self._current_data].sub_el

    def current_data_toggle_mask(self):
        self.data[self._current_data].mask_el = not self.data[self._current_data].mask_el
