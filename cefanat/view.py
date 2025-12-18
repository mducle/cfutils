"""
GUI view code for the Crystal Electric Field Analysis Toolkit (CEFAnaT)
This view contains only visual definitions and no logic so should not be unit-tested.
The CEFAnaTView class is a QWidget which should be set as the centralWidget of a QMainWindow.
In addition it has a method, connect(), which should be used to connect callbacks and mocked for unit tests.
"""

import numpy as np
from qtpy.QtCore import QEventLoop, Qt, QProcess, Signal, QAbstractTableModel  # noqa
from qtpy.QtWidgets import (QAction, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QMenu, QLabel,
                            QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpacerItem, QTabWidget, QTableView,
                            QGroupBox, QRadioButton, QStackedWidget, QTextEdit, QVBoxLayout, QListWidget, QWidget)  # noqa
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

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
            inpwidget = inp[2](inp[1], parent)
        else:
            raise RuntimeError(f'Input item type "{inp[0]}" not recognised')
        setattr(parent, inp[3], inpwidget)
        layout.addWidget(inpwidget)
    widget = QWidget(parent)
    widget.setLayout(layout)
    return widget


class dummyBlm():
    def __init__(self):
        self.data = {}
    def __getitem__(self, ind): return self.data[ind] if ind in self.data else 0.0
    def __setitem__(self, ind, value): 
        self.data[ind] = value
    def allowed(self, i1, i2): return True


class BlmTableModel(QAbstractTableModel):
    changed = Signal(dict)
    rows = ['0', '1', '-1', '2', '-2', '3', '-3', '4', '-4', '5', '-5', '6', '-6']
    cols = ['2', '4', '6']
    editable = Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
    def __init__(self, Blmobj, parent=None):
        super(BlmTableModel, self).__init__(parent)
        self.Blmobj = Blmobj
    def rowCount(self, parent): return 13
    def columnCount(self, parent): return 3
    def flags(self, index):
        return self.editable if self.Blmobj.allowed(int(self.cols[index.column()]), int(self.rows[index.row()])) else Qt.NoItemFlags
    def data(self, index, role):
        return self.Blmobj[int(self.cols[index.column()]), int(self.rows[index.row()])]
    def setData(self, index, value, role=Qt.EditRole):
        self.Blmobj[int(self.cols[index.column()]), int(self.rows[index.row()])] = value
        return True
    def headerData(self, section, orientation, role=None):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return f'l={self.cols[section]}'
        if role == Qt.DisplayRole and orientation == Qt.Vertical:
            return f'm={self.rows[section]}'


class BlmTable(QTableView):
    def __init__(self, Blmobj=None, parent=None):
        super(BlmTable, self).__init__(parent)
        self.setModel(BlmTableModel(Blmobj if Blmobj is not None else dummyBlm(), parent))


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


class CEFAnaTView(QWidget):

    def __init__(self, parent=None):
        super(CEFAnaTView, self).__init__(parent)
        self.drawlayout()

    def connect(self, widget, action, target):
        if not hasattr(self, widget):
            raise RuntimeError(f'Widget "{widget}" not part of this view')
        widgetobj, typestr = (getattr(self, widget), None)
        if '[' in action:
            action, typestr = action.replace(']','').split('[')
        if not hasattr(widgetobj, action):
            raise RuntimeError(f'Widget "{widget}" has no action "{action}"')
        if typestr:
            getattr(widgetobj, action)[typestr].connect(target)
        else:
            getattr(widgetobj, action).connect(target)

    def drawdatatab(self):
        self.datalayout = QGridLayout()
        self.dataloadbtn = QPushButton("Load Data")
        self.datalist = QListWidget(self.datatab)
        self.datafig = Figure()
        self.datacanvas = FigureCanvas(self.datafig)
        self.dataaxes = self.datafig.add_subplot(111)
        self.datatools = QWidget()
        self.datatoolsnav = NavigationToolbar(self.datacanvas, self.datatab)
        self.datatoolsswitch = QPushButton('Switch to tiles')
        toollayout = QHBoxLayout()
        toollayout.addWidget(self.datatoolsnav)
        toollayout.addWidget(self.datatoolsswitch)
        self.datatools.setLayout(toollayout)
        self.dataprops = QWidget()
        self.datatype = RadioGroup(self, ['INS', 'M(H)', 'M(T)', 'chi(T)', 'Cp(T)'], 'Data type')
        self.datapropstack = QStackedWidget(self.datatab)
        for prop in [
              create_vertical_inputs(self, [
                ['pair', 'Temperature', QLineEdit, 'datainput_instt'],
                ['pair', 'Incident Energy', QLineEdit, 'datainput_insEi'],
                ['pair', 'Applied Field', QLineEdit, 'datainput_insH'],
                ['spacer', 100]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_mhunit'],
                ['pair', 'Temperature', QLineEdit, 'datainput_mhtt'], ['spacer', 85]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_mtunit'],
                ['pair', 'Magnetic Field', QLineEdit, 'datainput_mth'], ['spacer', 85]]),
              create_vertical_inputs(self, [[RadioGroup(self, ['bohr', 'SI', 'cgs'], 'Unit'), 'datainput_chiunit'], ['spacer', 130]]),
              create_vertical_inputs(self, [['pair', 'Magnetic Field', QLineEdit, 'datainput_mth'], ['spacer', 190]])
            ]:
            self.datapropstack.addWidget(prop)
        self.datatype.changed.connect(lambda index: self.datapropstack.setCurrentIndex(index))
        self.dataedit = QPushButton('Edit')
        propslayout = QVBoxLayout()
        propslayout.addWidget(self.datatype)
        propslayout.addWidget(self.datapropstack)
        propslayout.addWidget(self.dataedit)
        self.dataprops.setLayout(propslayout)
        self.datalayout.addWidget(self.dataloadbtn, 0, 0)
        self.datalayout.addWidget(self.datalist, 1, 0)
        self.datalayout.addWidget(self.datatools, 0, 1)
        self.datalayout.addWidget(self.datacanvas, 1, 1)
        self.datalayout.addWidget(self.dataprops, 0, 2, -1, 1)
        self.datatab.setLayout(self.datalayout)

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
                ['pair', 'Energy Levels', QLineEdit, 'modelinputenergies'],
                ['single', 'Compute', QPushButton, 'modelinputencompute'], ['spacer', 300]])
            ]:
            self.modeltypestack.addWidget(prop)
        self.modeltype.changed.connect(lambda index: self.modeltypestack.setCurrentIndex(index))
        def get_cif():
            dlg = QFileDialog(self)
            dlg.setNameFilter('*.cif')
            if dlg.exec():
                cifs = dlg.selectedFiles()
                if cifs:
                    self.modelinputciffile.setText(cifs[0])
        self.modelinputcifbrowse.clicked.connect(get_cif)
        self.modelparatabs = QTabWidget(self)
        self.modelparams = [BlmTable(parent=self.modelparatabs)]
        self.modelparatabs.addTab(self.modelparams[0], 'Site 1')
        self.modellayout[0].addWidget(self.modeltype)
        self.modellayout[0].addWidget(self.modeltypestack)
        self.modellayout[1].addWidget(self.modelparatabs)
        for ii, stretchfac in zip(range(2), [1, 4]):
            self.modelinner[ii].setLayout(self.modellayout[ii])
            self.modelouterlayout.addWidget(self.modelinner[ii])
            self.modelouterlayout.setStretch(ii, stretchfac)
        self.modeltab.setLayout(self.modelouterlayout)
        
    def drawfittab(self):
        pass

    def drawlayout(self):
        self.mainlayout = QVBoxLayout()
        self.tabs = QTabWidget(self)
        for tabobj, tabname in zip(['datatab', 'modeltab', 'fittab'], ['Data', 'CEF Model', 'Fit']):
            setattr(self, tabobj, QWidget(self.tabs))
            getattr(self, f'draw{tabobj}')()
            self.tabs.addTab(getattr(self, tabobj), tabname)
        self.mainlayout.addWidget(self.tabs)
        self.setLayout(self.mainlayout)


def setup_menu(mainwindow, mainview):
    for menu in [['File', [["Load data", 'loaddat'], ["Load model", 'loadmodel'], ["Save model", 'savemodel']]],
                 ['Options', [['Set calculation engine', 'setengine']]]]:
        menuitem = QMenu(menu[0])
        setattr(mainview, f'menu{menu[0]}', menuitem)
        for act in menu[1]:
            actionitem = QAction(act[0], menuitem)
            menuitem.addAction(actionitem)
            setattr(mainview, act[1], actionitem)
        mainwindow.menuBar().addMenu(menuitem)
