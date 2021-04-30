"""
These utility functions and classes are either modified or unmodified versions of correpsonding
code from the ScopeFoundry project; see http://www.scopefoundry.org/.

In py4DSTEM, all quantities entered into the GUI are stored as LoggedQuantities, which are
collectively contained in an LQCollection object. The key advantages of LoggedQuantities are:
    -signaling: these objects emit Qt signals whenever they change, so that appropriate methods
                can be triggered whenever they change, regardless of where/how they are changed.
    -connection to widgets: they support a single interface for connecting to GUI widgets
"""

from os.path import join, dirname, expanduser
from PyQt5 import QtCore, QtWidgets
from numpy import nonzero
from ..io.native import get_py4DSTEM_dataobject_info, read_py4DSTEM
import pyqtgraph as pg

def datacube_selector(fp, data_id=0):
    """
    For a py4DSTEM formatted file at fp:
        - if there is a single datacube, return it
        - if there are multiple datacubes, return the one at index = data_id
        - if data_id=-1, return the names and indices of all the datacubes
    """
    info = get_py4DSTEM_dataobject_info(fp)
    inds = nonzero(info['type']=='DataCube')[0]
    N_dc = len(inds)

    if data_id==-1:
        names,indices = [],[]
        for i in inds:
            names.append(info[i]['name'])
            indices.appen(info[i]['index'])
            return names,indices
    if N_dc == 1:
        i = inds[0]
        dc,_ = read_py4DSTEM(fp, data_id=i)
        return dc
    elif N_dc > 1:
        assert(data_id in inds), "No datacube found at index {}.".format(data_id)
        dc,_ = read_py4DSTEM(fp, data_id=data_id)
        return dc
    else:
        print("No datacubes found in this file.")


def datacube_selector_dialog(fpath,window):
    """
    Presents the user with a dialog to choose one of the datacubes inside the file,
    if more than one exists. Returns the selected DataCube object.
    """
    info = get_py4DSTEM_dataobject_info(fpath)
    inds = np.nonzero(info['type']=='DataCube')[0]
    N_dc = len(inds)

    if N_dc == 1:
        ind = inds[0]
        dc,_ = read_py4DSTEM(fpath, data_id=ind)
    elif N_dc > 1:
        # there is more than one object, so we need to present the user with a chooser
        #indices = (self.dataobject_lookup_arr=='DataCube' | 
        #    self.dataobject_lookup_arr=='RawDataCube').nonzero()[0]
        print('More than one datacube detected...')
        print('Pls harass developers to add support feature.')

    return dc




def sibling_path(fpath, fname):
    """
    Given a file with absolute path fpath, returns the absolute path to another file with name
    fname in the same directory.
    """
    return join(dirname(fpath), fname)

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

class LQCollection(object):

    def __init__(self):
        self._logged_quantities = dict()

    def New(self, name, dtype=float, **kwargs):
        if dtype == 'file':
            lq = FileLQ(name=name, **kwargs)
        else:
            lq = LoggedQuantity(name=name, dtype=dtype, **kwargs)
        self._logged_quantities[name] = lq
        self.__dict__[name] = lq
        return lq

    def as_dict(self):
        return self._logged_quantities


class LoggedQuantity(QtCore.QObject):

    updated_value = QtCore.pyqtSignal(object)  # Emitted on val update

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


    @QtCore.pyqtSlot(float)
    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(bool)
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot()
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
        self.updated_value.emit(self.coerce_to_type(self.val))

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

            self.updated_value.connect(widget.setValue)
            widget.valueChanged.connect(self.update_value)

        elif type(widget) == QtWidgets.QSpinBox:
            widget.setKeyboardTracking(False)
            widget.setSingleStep(self.spinbox_step)
            widget.setValue(self.val)

            self.updated_value.connect(widget.setValue)
            widget.valueChanged.connect(self.update_value)

        elif type(widget) == QtWidgets.QButtonGroup:
            checkButtonById = lambda i: widget.button(i).setChecked(True)
            self.updated_value.connect(checkButtonById)
            widget.buttonClicked[int].connect(self.update_value)

        elif type(widget) == QtWidgets.QCheckBox:
            self.updated_value.connect(widget.setChecked)
            widget.toggled.connect(self.update_value)

        elif type(widget) == QtWidgets.QLineEdit:
            self.updated_value.connect(widget.setText)
            def on_edit_finished():
                self.update_value(widget.text())
            widget.editingFinished.connect(on_edit_finished)

        elif type(widget) == QtWidget.QLabel:
            self.updated_value.connect(widget.setText)

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

            self.updated_value.connect(widget.setValue)
            widget.valueChanged.connect(self.update_value)

        else:
            raise ValueError("Unknown widget type: {}".format(type(widget)))

        self.send_display_updates()
        self.widget_list.append(widget)

    def coerce_to_type(self,x):
        return self.dtype(x)

    ########## End of LoggedQuantity object ##########

class FileLQ(LoggedQuantity):

    def __init__(self, name, default_dir=None, **kwargs):
        kwargs.pop('dtype',None)
        LoggedQuantity.__init__(self,name,dtype=str,**kwargs)
        self.default_dir = default_dir

    def connect_to_browse_widgets(self, lineEdit, pushButton):
        assert type(lineEdit) == QtWidgets.QLineEdit
        self.connect_bidir_to_widget(lineEdit)

        assert type(pushButton) == QtWidgets.QPushButton
        pushButton.clicked.connect(self.file_browser)

    def file_browser(self):
        
        # Platform agnotistic method of getting home directory
        home = expanduser("~")

        #Start open filename in home directory
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(None,directory=home)
        print(repr(fname))
        if fname:
            self.update_value(fname)










