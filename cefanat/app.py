import sys
from . import view
from qtpy.QtWidgets import QMainWindow

try:
    from mantidqt.gui_helper import set_matplotlib_backend, get_qapplication
except ImportError:
    within_mantid = False
    from qtpy.QtWidgets import QApplication
    app = QApplication(sys.argv)
else:
    set_matplotlib_backend()  # must be called before anything tries to use matplotlib
    app, within_mantid = get_qapplication()

def main():
    mainwindow = QMainWindow()
    mainview = view.CEFAnaTView()
    view.setup_menu(mainwindow, mainview)
    mainwindow.setCentralWidget(mainview)
    mainwindow.setWindowTitle("CEFAnaT")
    mainwindow.show()
    if not within_mantid:
        sys.exit(app.exec_())
