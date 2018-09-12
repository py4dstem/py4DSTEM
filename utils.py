import os
from PySide2 import QtCore, QtGui, QtUiTools
import pyqtgraph as pg


def sibling_path(fpath, fname):
    """
    Given a file with absolute path fpath, returns the absolute path to another file with name
    fname in the same directory.
    """
    return os.path.join(os.path.dirname(fpath), fname)


def load_qt_ui_file(ui_filename):
    """
    Loads a ui file specifying a user interface configuration
    """
    ui_loader = QtUiTools.QUiLoader()
    ui_file = QtCore.QFile(ui_filename)
    ui_file.open(QtCore.QFile.ReadOnly)
    ui = ui_loader.load(ui_file)
    ui_file.close()
    return ui

def pg_point_roi(view_box):
    """
    Point selection.  Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI( (0,0), (2,2), movable=True, pen=(0,9))
    h = circ_roi.addTranslateHandle((0.5,0.5))
    h.pen = pg.mkPen('r')
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi




