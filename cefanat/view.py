"""
General purpose GUI view code, defined by a specification (Python dict / JSON).
Dict keys should be the element name, which will be added to the MainWindow object.
Dict values should be a tuple (type_name, properties, children)
Children should be a list of tuples (name, type_name, properties, children)
"""

from qtpy.QtCore import QEventLoop, Qt, QProcess  # noqa
from qtpy.QtWidgets import (QAction, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QMenu, QLabel,
                            QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpacerItem, QTabWidget,
                            QTextEdit, QVBoxLayout, QWidget)  # noqa
try:
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

class WindowView(QMainWindow):
    # All intrinsic methods are prefixed with a dunder. Non-dunder properties are elements of the Window
    _layouttypes = {'hbox':QHBoxLayout, 'vbox':QVBoxLayout, 'grid':QGridLayout}

    def __init__(self, specfication, parent=None, window_flags=None):
        super(WindowView, self).__init__(parent)
        self._drawLayout(specification)

    def _layout(self, name, typestr='hbox', parent=self, children=None):
        # Note that for grid layouts, children must be a list of list (2D nested)
        qlayout = self._layouttypes[typestr]()
        if typestr == 'grid':
            for ix in range(len(children)):
                for iy in range(len(children[0])):
                    child = children[ix][iy]
                    qlayout.addWidget(getattr(self, '_'+child[1])(child[0], **child[2], children=child[3]), ix, iy)
        else:
            for child in children:
                qlayout.addWidget(getattr(self, '_'+child[1])(child[0], **child[2], children=child[3]))
        parent.setLayout(qlayout)
        setattr(self, name, qlayout)

    def _drawLayout(self, spec):
        # spec should be a dict 
        for name, obj in spec.items():
            setattr(self, name, getattr(self, '_'+obj[0])(name, **obj[1], children=obj[2]))
