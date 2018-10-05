"""
These utility functions and classes are either modified or unmodified versions of correpsonding
code from the ScopeFoundry project; see http://www.scopefoundry.org/.

In py4DSTEM, all quantities entered into the GUI are stored as LoggedQuantities, which are
collectively contained in an LQCollection object. The key advantages of LoggedQuantities are:
    -signaling: these objects emit Qt signals whenever they change, so that appropriate methods
                can be triggered whenever they change, regardless of where/how they are changed.
    -connection to widgets: they support a single interface for connecting to GUI widgets
"""

import os
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools
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
    circ_roi = pg.CircleROI( (-0.5,-0.5), (2,2), movable=True, pen=(0,9))
    h = circ_roi.addTranslateHandle((0.5,0.5))
    h.pen = pg.mkPen('r')
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi

############### Logged Quantities ###############

class LQCollection_py4DSTEM(object):

    def __init__(self):
        self._logged_quantities = dict()

    def New(self, name, dtype=float, **kwargs):
        if 'array' in kwargs.keys():
            lq = ArrayLQ_py4DSTEM(name=name, dtype=dtype, **kwargs)
        elif dtype == 'file':
            lq = FileLQ_py4DSTEM(name=name, **kwargs)
        else:
            lq = LoggedQuantity_py4DSTEM(name=name, dtype=dtype, **kwargs)
        self._logged_quantities[name] = lq
        self.__dict__[name] = lq
        return lq

    def as_dict(self):
        return self._logged_quantities


class LoggedQuantity_py4DSTEM(QtCore.QObject):

    updated_value = QtCore.Signal((float,),(int,),(bool,),(str,),())  # Emitted on val update

    def __init__(self, name, dtype=float, initial=0,
                 vmin=-1e12, vmax=+1e12,
                 spinbox_decimals=2, spinbox_step=0.1,
                 unit=None):

        QtCore.QObject.__init__(self)

        self.name = name
        self.dtype = dtype
        self.val = dtype(initial)
        self.vmin, self.vmax = vmin, vmax
        self.unit = unit
        if self.dtype==int:
            self.spinbox_step = 1
            self.spinbox_decimals = 0
        else:
            self.spinbox_step = spinbox_step
            self.spinbox_decimals = spinbox_decimals

        self.old_val = None
        self.widget_list = []


    @QtCore.Slot(float)
    @QtCore.Slot(int)
    @QtCore.Slot(bool)
    @QtCore.Slot(str)
    @QtCore.Slot()
    def update_value(self, new_val, send_signal=True):

        # If the value hasn't changed, do nothing
        self.old_val = self.coerce_to_type(self.val)
        new_val = self.coerce_to_type(new_val)
        if self.old_val == new_val:
            return

        # If the value has changed, send signal to any attached widgets
        self.val = new_val
        if send_signal:
            self.send_display_updates()

    def send_display_updates(self):
        self.updated_value[float].emit(self.val)
        self.updated_value[int].emit(self.val)
        self.updated_value[str].emit(str(self.val))
        self.updated_value[()].emit()

    def connect_bidir_to_widget(self, widget):
        """
        Supported widget types:
            Qt: QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QLabel
            pyqtgraph: SpinBox
        """
        if type(widget) == QtWidgets.QDoubleSpinBox:
            widget.setKeyboardTracking(False)
            if self.vmin is not None:
                widget.setMinimum(self.vmin)
            if self.vmax is not None:
                widget.setMaximum(self.vmax)
            if self.unit is not None:
                widget.setSuffix(" "+self.unit)
            widget.setDecimals(self.spinbox_decimals)
            widget.setSingleStep(self.spinbox_step)
            widget.setValue(self.val)

            self.updated_value[float].connect(widget.setValue)
            widget.valueChanged[float].connect(self.update_value)

        elif type(widget) == QtWidgets.QSpinBox:
            widget.setKeyboardTracking(False)
            #if self.vmin is not None:
            #    widget.setMinimum(self.vmin)
            #if self.vmax is not None:
            #    widget.setMaximum(self.vmax)
            #if self.unit is not None:
            #    widget.setSuffix(" "+self.unit)
            widget.setSingleStep(self.spinbox_step)
            widget.setValue(self.val)

            self.updated_value[int].connect(widget.setValue)
            widget.valueChanged[int].connect(self.update_value)

        elif type(widget) == QtWidgets.QCheckBox:
            self.updated_value[bool].connect(widget.setChecked)
            widget.toggled[bool].connect(self.update_value)

        elif type(widget) == QtWidgets.QLineEdit:
            self.updated_value[str].connect(widget.setText)
            widget.editingFinished[str].connect(self.update_value)

        elif type(widget) == QtWidget.QLabel:
            self.updated_value[str].connect(widget.setText)

        elif type(widget) == pg.widgets.SpinBox.SpinBox:
            suffix = self.unit
            if self.unit is None:
                suffix = ""
            if self.dtype == int:
                integer=True
                minStep=1
                step=1
            else:
                integer=False
                minStep=0.1
                step=0.1
            widget.setOpts(suffix=suffix, siPrefix=True, dec=True, int=integer,
                           step=step, minStep=minStep, bounds=[self.vmin, self.vmax])
            widget.setDecimals(self.spinbox_decimals)
            widget.setSingleStep(self.spinbox_step)

            self.updated_value[float].connect(widget.setValue)
            widget.valueChanged.connect(self.update_value)

        else:
            raise ValueError("Unknown widget type: {}".format(type(widget)))

        self.send_display_updates()
        self.widget_list.append(widget)

    def coerce_to_type(self,x):
        return self.dtype(x)

    ########## End of LoggedQuantity object ##########

#class FileLQ(QtCore.QObject):

#    def __init__(name, **kwargs):
#        pass


#class ArrayLQ(QtCore.QObject):

#    def __init__(name, dtype=float, **kwargs):
#        pass









