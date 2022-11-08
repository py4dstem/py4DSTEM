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
from py4DSTEM.io.native import read_py4DSTEM, get_py4DSTEM_topgroups, get_py4DSTEM_version, get_N_dataobjects
import pyqtgraph as pg

def datacube_selector(fp, data_id=0):
    """
    For a py4DSTEM formatted file at fp:
        - if there is a single datacube, return it
        - if there are multiple datacubes, return the one at index = data_id
        - if data_id=-1, return the names and indices of all the datacubes
    """
    topgroup = get_py4DSTEM_topgroups(fp)[0]
    version_major, version_minor, version_release = get_py4DSTEM_version(fp,topgroup)

    # this is very very bad, but it seems we have to manually
    # go through this in order to specifically get the first
    # datacube in a file...
    if (version_major, version_minor) == (0,12):
        from py4DSTEM.io.native.read.read_utils_v0_12 import get_py4DSTEM_dataobject_info
    elif (version_major, version_minor) == (0,9):
        from py4DSTEM.io.native.read.read_utils_v0_9 import get_py4DSTEM_dataobject_info
    elif (version_major, version_minor) == (0,7):
        from py4DSTEM.io.native.read.read_utils_v0_7 import get_py4DSTEM_dataobject_info
    elif (version_major, version_minor) == (0,6):
        from py4DSTEM.io.native.read.read_utils_v0_6 import get_py4DSTEM_dataobject_info
    elif (version_major, version_minor) == (0,5):
        from py4DSTEM.io.native.read.read_utils_v0_5 import get_py4DSTEM_dataobject_info
    else:
        raise Exception("This EMD file version is not supported by the GUI.")

    print(f"Reading py4DSTEM EMD version {version_major}.{version_minor}.{version_release}")
    info = get_py4DSTEM_dataobject_info(fp,topgroup)
    inds = nonzero(info['type']=='DataCube')[0]
    N_dc = len(inds)

    if data_id==-1:
        names,indices = [],[]
        for i in inds:
            names.append(info[i]['name'])
            indices.appen(info[i]['index'])
            return names,indices
    if N_dc == 1:
        i = int(inds[0])
        dc = read_py4DSTEM(fp, data_id=i)
        return dc
    elif N_dc > 1:
        assert(data_id in inds), "No datacube found at index {}.".format(data_id)
        dc = read_py4DSTEM(fp, data_id=data_id)
        return dc
    else:
        print("No datacubes found in this file.")

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

        elif type(widget) == QtWidgets.QLabel:
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










