"""
GUI view code for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
This view contains only visual definitions and no logic so should not be unit-tested.
The CEFAnaTView class is a QWidget which should be set as the centralWidget of a QMainWindow.
In addition it has a method, connect(), which should be used to connect callbacks and mocked for unit tests.
"""

import numpy as np
from qtpy.QtCore import QEventLoop, Qt, QProcess, Signal, QAbstractTableModel  # noqa
from qtpy.QtWidgets import (QAction, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QMenu, QLabel,
                            QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpacerItem, QTabWidget,
                            QGroupBox, QRadioButton, QStackedWidget, QTextEdit, QVBoxLayout, QListWidget, QWidget,
                            QTableWidget, QTableWidgetItem, QActionGroup)  # noqa
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton

try:
    from mantid.plots.utility import legend_set_draggable
    from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas
    from mantidqt.MPLwidgets import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from qtpy import PYQT4, PYQT5, PYSIDE, PYSIDE2  # noqa
    if PYQT4 or PYSIDE:
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
    elif PYQT5 or PYSIDE2:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    else:
        raise RuntimeError("Do not know which matplotlib backend to set")
    from matplotlib.legend import Legend

    if hasattr(Legend, "set_draggable"):
        SET_DRAGGABLE_METHOD = "set_draggable"
    else:
        SET_DRAGGABLE_METHOD = "draggable"

    def legend_set_draggable(legend, state, use_blit=False, update="loc"):
        getattr(legend, SET_DRAGGABLE_METHOD)(state, use_blit, update)


def create_vertical_inputs(parent, spec):
    layout = QVBoxLayout()
    for inp in spec:
        if not isinstance(inp[0], str):
            layout.addWidget(inp[0])
            if len(inp) > 1:
                setattr(parent, inp[1], inp[0])
            continue
        if 'spacer' in inp[0]:
            layout.addItem(QSpacerItem(0, inp[1] if len(inp) > 1 else 35))
            continue
        if 'pair' in inp[0]:
            layout.addWidget(QLabel(inp[1]))
            inpwidget = inp[2](parent)
        elif 'single' in inp[0]:
            if isinstance(inp[1], tuple):
                inpwidget = inp[2](*inp[1])
            else:
                inpwidget = inp[2](inp[1], parent)
        else:
            raise RuntimeError(f'Input item type "{inp[0]}" not recognised')
        setattr(parent, inp[3], inpwidget)
        layout.addWidget(inpwidget)
    widget = QWidget(parent)
    layout.setAlignment(Qt.AlignTop)
    widget.setLayout(layout)
    return widget


def _create_blmwidget(parent, spec):
    layout = QVBoxLayout()
    for inp in spec:
        w1 = QWidget(parent)
        l1 = QHBoxLayout()
        l1.addWidget(QLabel(inp[0]))
        inpwidget = QLineEdit(parent)
        l1.addWidget(inpwidget)
        w1.setLayout(l1)
        layout.addWidget(w1)
        setattr(parent, f'{inp[1]}_holder', w1)
        setattr(parent, inp[1], inpwidget)
    widget = QWidget(parent)
    widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    widget.setLayout(layout)
    return widget


class BlmTable(QWidget):
    def __init__(self, parent=None, symm=0, extras=None):
        """Creates a CEF parameters widget defined by a symmetry
        symm (int): symmetry parameter - m values divisible by symm are allowed
        extras (list): List of extra parameters
        """
        super(BlmTable, self).__init__(parent)
        self.extras = extras
        baselayout = QVBoxLayout()
        ceflayout = QHBoxLayout()
        self.blmwidgets = [_create_blmwidget(self, [
            [f'B<sub>{l}</sub><sup>{m}</sup>', f'b{l}{m}'] for m in range(-l, l+1)
        ]) for l in [2, 4, 6]]
        [ceflayout.addWidget(w) for w in self.blmwidgets]
        self.cefwidget = QWidget(parent=self)
        self.cefwidget.setLayout(ceflayout)
        baselayout.addWidget(self.cefwidget)
        if extras:
            self.extraswidget = _create_blmwidget(self, [[x, f'inp_{x}'] for x in extras])
            baselayout.addWidget(self.extraswidget)
        baselayout.setAlignment(Qt.AlignTop)
        self.setLayout(baselayout)
        self.set_symm(symm)
    def set_symm(self, symm):
        if symm == 0:
            return
        for lm in ([l,m] for l in [2,4,6] for m in range(-l,l+1)):
            if (lm[1] < 0 and symm > 0) or (lm[1] % symm != 0):
                getattr(getattr(self, f'b{lm[0]}{lm[1]}_holder'), 'setHidden')(True)


class RadioGroup(QGroupBox):
    changed = Signal(int)
    def __init__(self, parent, labels, title=None):
        if title:
            super(RadioGroup, self).__init__(title, parent)
        else:
            super(RadioGroup, self).__init__(parent)
        self.layout = QVBoxLayout()
        self.buttons = [QRadioButton(label, self) for label in labels]
        self.buttons[0].setChecked(True)
        for btn in self.buttons:
            self.layout.addWidget(btn)
            btn.toggled.connect(lambda checked: self.changed.emit(self.getSelectedIndex()) if checked else None)
        self.setLayout(self.layout)
    def getSelectedIndex(self):
        return [ii for ii, bt in enumerate(self.buttons) if bt.isChecked()][0]
    def getSelected(self):
        return [bt.text() for bt in self.buttons if bt.isChecked()][0]
    def setSelectedIndex(self, index):
        self.buttons[index].setChecked(True)


class ExclusiveComboHeaders():
    def __init__(self, table, parent, xyeind=[0,1,2]):
        self.table = table
        self.parent = parent
        self.combos = []
        self.xyeind = np.array(xyeind)
        self.can_change = False
        ncol = table.columnCount()
        for ii in range(ncol):
            self.combos.append(QComboBox())
            self.combos[ii].addItems(np.array(['x', 'y', 'e', 'unused'])[:ncol])
            self.combos[ii].setCurrentIndex(3)
            self.table.setCellWidget(0, ii, self.combos[ii])
            self.combos[ii].currentIndexChanged.connect(lambda index, ci=ii: self.on_change(ci, index))
        for ii, jj in enumerate(xyeind):
            if jj is not None and jj < len(self.combos):
                self.combos[jj].setCurrentIndex(ii)
        self.can_change = True
    def on_change(self, col_ind, value):
        # Changes the column type - make sure if column type already specified in another col, to swap them
        if self.can_change:
            old_col = np.where(self.xyeind == value)[0]
            if len(old_col) > 0:
                self.can_change = False
                self.xyeind[old_col[0]] = self.xyeind[col_ind]
                self.combos[old_col[0]].setCurrentIndex(self.xyeind[old_col[0]])
                self.can_change = True
            self.xyeind[col_ind] = value
            self.parent.textdatacombos_changed(self.parent.datalist.currentRow(), self.xyeind)


class CEFAnaTView(QWidget):

    def __init__(self, parent=None):
        super(CEFAnaTView, self).__init__(parent)
        self.is_noninteractive = False
        self.drawlayout()

    def get_file(self, filt=None):
        dlg = QFileDialog(self)
        dlg.setNameFilter(filt)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if dlg.exec():
            return dlg.selectedFiles() 

    def popupdlg(self, message, buttons=None):
        dlg = QMessageBox(self)
        dlg.setText(message)
        dlg.setStandardButtons(buttons if buttons else QMessageBox.Ok)
        return dlg.exec()

    def connect(self, widget, action, target0):
        if not hasattr(self, widget):
            raise RuntimeError(f'Widget "{widget}" not part of this view')
        # Handle special cases
        match f'{widget}.{action}':
            case 'datalist.currentItemChanged':
                target = lambda cur, prv: target0(cur.text() if cur else '', prv.text() if prv else '')
            case 'datalist.comboChanged':
                self.textdatacombos_changed = target0
                return
            case _:
                target = target0
        widgetobj, typestr = (getattr(self, widget), None)
        if '[' in action:
            action, typestr = action.replace(']','').split('[')
        if not hasattr(widgetobj, action):
            raise RuntimeError(f'Widget "{widget}" has no action "{action}"')
        if typestr:
            getattr(widgetobj, action)[typestr].connect(target)
        else:
            getattr(widgetobj, action).connect(target)

    def _setupdata_text_widget(self):
        rv, ly, rw, rwly = (QWidget(self), QHBoxLayout(), QWidget(self), QVBoxLayout())
        self.textdatatable = QTableWidget(self)
        self.textdataraw = QTextEdit(self)
        self.textdataraw.setReadOnly(True)
        self.textdatainput = create_vertical_inputs(self, [
            ['pair', 'Delimiter', QLineEdit, 'textdatadelimiter'],
            ['pair', 'Fixed Width', QLineEdit, 'textdatawidth']])
        rwly.addWidget(self.textdatainput)
        rwly.addWidget(self.textdataraw)
        rw.setLayout(rwly)
        ly.addWidget(self.textdatatable)
        ly.addWidget(rw)
        rv.setLayout(ly)
        return rv

    def set_text_data(self, textdata, textraw, xyeind=[0,1,2]):
        self.textdataraw.setText(textraw)
        self.textdatatable.clear()
        if (textdata is not None and len(tdim := textdata.shape) > 1):
            self.textdatatable.setRowCount(tdim[0])
            self.textdatatable.setColumnCount(tdim[1])
            for i,j in [[i,j] for i in range(tdim[0]) for j in range(tdim[1])]:
                self.textdatatable.setItem(i+1, j, QTableWidgetItem(f'{textdata[i,j]}'))
        self.textdatacombos = ExclusiveComboHeaders(self.textdatatable, self, xyeind)
        
    def _setupdata_nxs_widget(self):
        return QLabel('nxs')

    def _setupdata_mat_widget(self):
        return QLabel('mat')

    def reset_data_meta(self, data=None):
        self.is_noninteractive = True
        for widg in ['datatype', '_insunit', '_mhunit', '_mtunit', '_chiunit']:
            getattr(self, f'datainput{widg}' if widg.startswith('_') else widg).setSelectedIndex(0)
        for widg in ['instt', 'insEi', 'insH', 'mhtt', 'mth', 'cph', 'insHdir', 'mthdir', 'cphdir']:
            getattr(self, f'datainput_{widg}').setText('')
        self.datainput_chiinv.setChecked(False)
        if data is not None:
            idx = data.datatype_index
            self.datatype.setSelectedIndex(idx)
            if data.datatype != 'CP':
                getattr(self, f'datainput_{data.datatype.lower()}unit').setSelectedIndex(data.dataunit_index)
            ew = [{'instt':'Temperature', 'insEi':'Ei', 'insH':'H', 'insHdir':'Hdir'}, {'mhtt':'Temperature'},
                  {'mth':'H', 'mthdir':'Hdir'}, {}, {'cph':'H', 'cphdir':'Hdir'}][idx]
            for wg, prp in ew.items():
                getattr(self, f'datainput_{wg}').setText(str(getattr(data, prp)))
            if data.datatype == 'CHI':
                self.datainput_chiinv.setChecked(data.invchi)
        self.is_noninteractive = False

    def update_data(self, data):
        match data.inputtype:
            case 'text':
                self.datadispstack.setCurrentIndex(0)
                self.set_text_data(data.array, data.raw, data.xyeind)
            case 'nxs':
                self.datadispstack.setCurrentIndex(1)
            case 'mat':
                self.datadispstack.setCurrentIndex(2)
        self.reset_data_meta(data)
        self.datatoolsaddpk.setEnabled(data.datatype == 'INS')
        self.datatoolsfitpk.setEnabled(data.peaks_guess is not None)
        if data.array is not None:
            self.plot_data(data)

    def plot_data(self, data):
        self.datatoolsfitpk.setEnabled(data.peaks_guess is not None)
        self.dataaxes.cla()
        if data.array.shape[1] > 2:
            self.dataaxes.errorbar(data.x, data.y, data.e, fmt='ob')
        else:
            self.dataaxes.plot(data.x, data.y, 'ob')
        if data.peaks_guess is not None:
            self.dataaxes.plot(data.peaks_guess[:,0], data.peaks_guess[:,1], 'sr', markersize=10)
        if data.peaks_trace is not None:
            self.dataaxes.plot(data.x, data.peaks_trace, '-k')
        self.dataaxes.set_xlabel(data.xlabel)
        self.dataaxes.set_ylabel(data.ylabel)
        self.datacanvas.draw()

    def update_data_list(self, name):
        if name in [self.datalist.item(ii).text() for ii in range(self.datalist.count())]:
            txtmsg = 'Name previously used. Click "Yes" to overwrite. Click "No" to rename. Click "Cancel" to not load'
            userinp = self.popupdlg(txtmsg, QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            match userinp:
                case QMessageBox.Yes:
                    return name
                case QMessageBox.No:
                    self.datalist.addItem(f'{name}_new')
                    return f'{name}_new'
                case QMessageBox.Cancel:
                    return None
        else:
            self.datalist.addItem(name)
        return name

    def set_current_data(self, index):
        self.datalist.setCurrentRow(index)

    def get_peaks(self):
        self.datatoolsaddpk.setText('Right click to stop')
        coords = self.datafig.ginput(-1, mouse_pop=None, mouse_stop=MouseButton.RIGHT)
        self.datatoolsaddpk.setText('Define peaks')
        return coords

    def drawdatatab(self):
        self.dataloadbtn = QPushButton("Load Data")
        self.datalist = QListWidget(self.datatab)
        self.datafig = Figure()
        self.datadisplay = QTabWidget(self)
        self.datadispstack = QStackedWidget(self)
        for t in ['text', 'nxs', 'mat']:
            self.datadispstack.addWidget(getattr(self, f'_setupdata_{t}_widget')()) 
        self.datacanvas = FigureCanvas(self.datafig)
        self.datadisplay.addTab(self.datacanvas, 'Plot')
        self.datadisplay.addTab(self.datadispstack, 'Data')
        self.dataaxes = self.datafig.add_subplot(111)
        self.datatools = QWidget()
        datatools1 = QWidget()
        self.datatoolsnav = NavigationToolbar(self.datacanvas, self.datatab)
        self.datatoolsswitch = QPushButton('Switch to tiles', enabled=False)
        toollayout1 = QHBoxLayout()
        toollayout1.addWidget(self.datatoolsnav)
        toollayout1.addWidget(self.datatoolsswitch)
        datatools1.setLayout(toollayout1)
        datatools2 = QWidget()
        self.datatoolsaddpk = QPushButton('Define peaks', enabled=False)
        self.datatoolsfitpk = QPushButton('Fit peaks', enabled=False)
        self.datatoolspk2cf = QPushButton('Peaks to Blm', enabled=False)
        toollayout2 = QHBoxLayout()
        toollayout2.addWidget(self.datatoolsaddpk)
        toollayout2.addWidget(self.datatoolsfitpk)
        toollayout2.addWidget(self.datatoolspk2cf)
        datatools2.setLayout(toollayout2)
        self.dataprops = QWidget()
        self.datatype = RadioGroup(self, ['INS', 'M(H)', 'M(T)', 'chi(T)', 'Cp(T)'], 'Data type')
        self.datapropstack = QStackedWidget(self.datatab)
        for prop in [
              create_vertical_inputs(self, [
                [RadioGroup(self, ['meV', 'cm', 'THz'], 'Unit'), 'datainput_insunit'],
                ['pair', 'Temperature (K)', QLineEdit, 'datainput_instt'],
                ['pair', 'Incident Energy', QLineEdit, 'datainput_insEi'],
                ['pair', 'Applied Field (T)', QLineEdit, 'datainput_insH'],
                ['pair', 'Field Direction', QLineEdit, 'datainput_insHdir'],
                ['spacer', 100]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_mhunit'],
                ['pair', 'Temperature (K)', QLineEdit, 'datainput_mhtt'], ['spacer', 220]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_mtunit'],
                ['pair', 'Magnetic Field (T)', QLineEdit, 'datainput_mth'],
                ['pair', 'Field Direction', QLineEdit, 'datainput_mthdir'], ['spacer', 220]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_chiunit'],
                ['pair', 'Inverse', QCheckBox, 'datainput_chiinv'], ['spacer', 220]]),
              create_vertical_inputs(self, [['pair', 'Magnetic Field (T)', QLineEdit, 'datainput_cph'],
                ['pair', 'Field Direction', QLineEdit, 'datainput_cphdir'], ['spacer', 300]]),
            ]:
            self.datapropstack.addWidget(prop)
        self.datatype.changed.connect(lambda index: self.datapropstack.setCurrentIndex(index))
        self.dataedit = QPushButton('Edit')
        propslayout = QVBoxLayout()
        propslayout.addWidget(self.datatype)
        propslayout.addWidget(self.datapropstack)
        propslayout.addWidget(self.dataedit)
        self.dataprops.setLayout(propslayout)
        datalayout = QHBoxLayout()
        datalayoutc1, datalayoutc2 = (QVBoxLayout() for _ in range(2))
        datac1, datac2 = (QWidget() for _ in range(2))
        datalayoutc1.addWidget(self.dataloadbtn)
        datalayoutc1.addWidget(self.datalist)
        datac1.setLayout(datalayoutc1)
        datalayoutc2.addWidget(datatools1)
        datalayoutc2.addWidget(datatools2)
        datalayoutc2.addWidget(self.datadisplay)
        datac2.setLayout(datalayoutc2)
        datalayout.addWidget(datac1)
        datalayout.addWidget(datac2)
        datalayout.addWidget(self.dataprops)
        self.datatab.setLayout(datalayout)

    def drawmodeltab(self):
        self.modelouterlayout = QHBoxLayout()
        self.modelinner, self.modellayout = zip(*((QWidget(), QVBoxLayout()) for i in range(2)))
        self.modeltype = RadioGroup(self, ['Parameters', 'Point Charge', 'Fit Energy'], 'Model Type')
        self.modeltypestack = QStackedWidget(self.modeltab)
        for prop in [
            create_vertical_inputs(self, [
                ['pair', 'Ion', QComboBox, 'modelinput_ion'],
                ['pair', 'Symmetry', QComboBox, 'modelinput_sym'],
                ['single', 'Add site', QPushButton, 'modelinput_add'], ['spacer', 250]]),
            create_vertical_inputs(self, [
                ['pair', 'CIF File', QLineEdit, 'modelinputciffile'],
                ['single', 'Browse', QPushButton, 'modelinputcifbrowse'],
                ['single', 'Load', QPushButton, 'modelinputcifload'], ['spacer', 250]]),
            create_vertical_inputs(self, [
                ['pair', 'Energy Levels (comma separated list)', QLineEdit, 'modelinputenergies'],
                ['pair', 'Point Symmetry', QLineEdit, 'modelinputensym'],
                ['single', 'Compute', QPushButton, 'modelinputencompute'], ['spacer', 300]])
            ]:
            self.modeltypestack.addWidget(prop)
        self.modeltype.changed.connect(lambda index: self.modeltypestack.setCurrentIndex(index))
        def get_cif():
            if (cifs := self.get_file('*.cif')):
                self.modelinputciffile.setText(cifs[0])
        self.modelinputcifbrowse.clicked.connect(get_cif)
        self.modelparatabs = QTabWidget(self)
        self.modelparams = [BlmTable(parent=self.modelparatabs, symm=0)]
        self.modelparatabs.addTab(self.modelparams[0], 'Site 1')
        self.modellayout[0].addWidget(self.modeltype)
        self.modellayout[0].addWidget(self.modeltypestack)
        self.modellayout[1].addWidget(self.modelparatabs)
        for ii, stretchfac in zip(range(2), [1, 4]):
            self.modelinner[ii].setLayout(self.modellayout[ii])
            self.modelouterlayout.addWidget(self.modelinner[ii])
            self.modelouterlayout.setStretch(ii, stretchfac)
        self.modeltab.setLayout(self.modelouterlayout)

    def set_fit_local_minimizers(self, minimizers):
        [self.fitlocalmin.removeItem(0) for _ in range(self.fitlocalmin.count())]
        [self.fitlocalmin.addItem(mn) for mn in minimizers]

    def set_fit_global_minimizers(self, minimizers):
        [self.fitglobalmin.addItem(mn) for mn in minimizers]

    def set_fit_global(self, algos):
        [self.fitglobalalgo.addItem(alg) for alg in algos]
        
    def drawfittab(self):
        self.fitinner, self.fitlayout = zip(*((QWidget(), QVBoxLayout()) for _ in range(2)))
        self.fittabs = QTabWidget(self)
        for names, prop in zip(['Local', 'Global', 'Search'], [
                [['single', (self, ['minimize', 'curvefit'], 'Fit function'), RadioGroup, 'fitlocalalgo'], 
                 ['pair', 'Minimizer', QComboBox, 'fitlocalmin']], 
                [['pair', 'Algorithm', QComboBox, 'fitglobalalgo'], 
                 ['pair', 'Local Minimizer', QComboBox, 'fitglobalmin']],
                [['pair', 'Max energy splitting', QLineEdit, 'fitsearchmaxE']],
            ]):
            self.fittabs.addTab(create_vertical_inputs(self, prop), names)
        self.fitobjective = RadioGroup(self, ['chi-sq', 'R-sq'], 'Objective')
        self.fittype = RadioGroup(self, ['Fit CEF', 'Fit effective charges'], '')
        self.fitbtn = QPushButton('Fit')
        for widg in [self.fittabs, self.fitobjective, self.fittype, self.fitbtn]:
            self.fitlayout[0].addWidget(widg)
        self.fitlayout[0].addItem(QSpacerItem(0, 300))
        self.fitfig = Figure()
        self.fitcanvas = FigureCanvas(self.fitfig)
        self.fitaxes = self.fitfig.add_subplot(111)
        self.fitnav = NavigationToolbar(self.fitcanvas, self.fittab)
        self.fitlayout[1].addWidget(self.fitnav)
        self.fitlayout[1].addWidget(self.fitcanvas)
        self.fitouterlayout = QHBoxLayout()
        for ii, stretchfac in zip(range(2), [1, 4]):
            self.fitinner[ii].setLayout(self.fitlayout[ii])
            self.fitouterlayout.addWidget(self.fitinner[ii])
            self.fitouterlayout.setStretch(ii, stretchfac)
        self.fittab.setLayout(self.fitouterlayout)

    def drawlayout(self):
        self.mainlayout = QVBoxLayout()
        self.tabs = QTabWidget(self)
        for tabobj, tabname in zip(['datatab', 'modeltab', 'fittab'], ['Data', 'CEF Model', 'Fit']):
            setattr(self, tabobj, QWidget(self.tabs))
            getattr(self, f'draw{tabobj}')()
            self.tabs.addTab(getattr(self, tabobj), tabname)
        self.mainlayout.addWidget(self.tabs)
        self.setLayout(self.mainlayout)

    def set_engines(self, engines):
        self.engines = []
        self.enginegroup = QActionGroup(self.setengine)
        for eng in engines:
            actionitem = QAction(eng, self.setengine, checkable=True)
            self.engines.append([eng, actionitem])
            self.enginegroup.addAction(actionitem)
            self.setengine.addAction(actionitem)

    def set_calc_engine(self, engine):
        for eng in self.engines:
            if eng[0] == engine:
                eng[1].setChecked(True)
                break

    def get_engine_name(self, engineobj):
        for eng in self.engines:
            if eng[1] == engineobj:
                return eng[0]


def setup_menu(mainwindow, mainview):
    for menu in [['File', [["Load data", 'loaddat'], ["Load model", 'loadmodel'], ["Save model", 'savemodel']]],
                 ['Options', [['menu', 'Set calculation engine', 'setengine']]]]:
        menuitem = QMenu(menu[0])
        setattr(mainview, f'menu{menu[0]}', menuitem)
        for act in menu[1]:
            if act[0] == 'menu':
                actionitem = menuitem.addMenu(act[1])
            else:
                actionitem = QAction(act[0], menuitem)
                menuitem.addAction(actionitem)
            setattr(mainview, act[-1], actionitem)
        mainwindow.menuBar().addMenu(menuitem)
