######## Viewer for 4D STEM data ########
#
# Defines a class -- DataViewer - enabling a simple GUI for
# interacting with 4D STEM datasets.
#
# Relevant documentation for lower level code:
#
# ScopeFoundry 
# ScopeFoundry is a flexible package for both scientific data visualization and control of labrotory experiments.  See http://www.scopefoundry.org/.  This code uses the ScopeFoundary object
# LQCollection, which enables intelligent interactive storage of logged quantities.
#
# Qt
#  Qt is being run through Pyside/PySide2/PyQt/Qt for Python. See https://www.qt.io/qt-for-python. Presently PySide is being used.  
# TODO: (maybe) use PySide2 (moves some objects from QtGui to the newer QtWidgets. Or (maybe)
# use qtpy, a small wrapper which supports systems with either PySide or PySide2 (basically, for
# python 2 or 3).
#
# pyqtgraph
# pyqtgraph is a library which facilitates fast-running scientific visualization.  See http://pyqtgraph.org/. pyqtgraph is being used for the final data displays.


from __future__ import division, print_function
from PySide2 import QtCore, QtWidgets
import numpy as np
import sys, os
from ScopeFoundry import BaseApp, LQCollection
from utils import load_qt_ui_file, sibling_path, pg_point_roi
import pyqtgraph as pg
import dm3_lib as dm3
from control_panel import ControlPanel
from datacube import DataCube

import IPython
if IPython.version_info[0] < 4:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget as RichJupyterWidget
    from IPython.qt.inprocess import QtInProcessKernelManager
else:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager


class DataViewer(QtCore.QObject):
    """
    DataViewer objects inherit from the ScopeFoundry.BaseApp class.
    ScopeFoundry.BaseApp objects inherit from the QtCore.QObject class.
    Additional functionality is provided by pyqtgraph widgets.

    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = DataViewer(sys.argv)
        app.exec_()
    """
    def __init__(self, argv):
        """
        Initialize class, setting up windows and widgets.
        """
        self.this_dir, self.this_filename = os.path.split(__file__)

        # Set a pointer referring to the application object
        self.qtapp = QtWidgets.QApplication.instance()
        if not self.qtapp:
            self.qtapp = QtWidgets.QApplication(argv)

        # TODO: consider removing dependency on LQCollection object 
        self.settings = LQCollection()

        # Set up temporary datacube
        self.datacube = DataCube("sample_data.dm3")

        # Set up widgets
        self.setup_diffraction_space_control_widget()
        self.setup_real_space_control_widget()
        self.setup_diffraction_space_widget()
        self.setup_real_space_widget()
        self.setup_console_widget()
        self.setup_geometry()

        # Set up initial views in real and diffraction space
        self.update_diffraction_space_view()
        self.update_real_space_view()
        self.diffraction_space_widget.ui.normDivideRadio.setChecked(True)
        self.diffraction_space_widget.normRadioChanged()

        return

    ############ Setup methods #############

    def setup_diffraction_space_control_widget(self):
        """
        Set up the control window for diffraction space.
        """
        #self.diffraction_space_control_widget = load_qt_ui_file(sibling_path(__file__, "diffraction_space_control_widget.ui"))
        self.diffraction_space_control_widget = ControlPanel()
        self.diffraction_space_control_widget.setWindowTitle("Diffraction space")
        self.diffraction_space_control_widget.show()
        self.diffraction_space_control_widget.raise_()

        ########## Controls ##########
        # For each control:
        # -create references in self.settings
        # -connect UI changes to updates in self.settings
        # -call methods
        ##############################

        # File loading
        self.settings.New('data_filename',dtype='file')
        self.settings.data_filename.connect_to_browse_widgets(self.diffraction_space_control_widget.lineEdit_LoadFile, self.diffraction_space_control_widget.pushButton_BrowseFiles)
        self.settings.data_filename.updated_value.connect(self.load_file)

        # Scan shape
        self.settings.New('R_Nx', dtype=int, initial=1)
        self.settings.New('R_Ny', dtype=int, initial=1)

        self.settings.R_Nx.updated_value.connect(self.update_scan_shape_Nx)
        self.settings.R_Ny.updated_value.connect(self.update_scan_shape_Ny)

        self.settings.R_Nx.connect_bidir_to_widget(self.diffraction_space_control_widget.spinBox_Nx)
        self.settings.R_Ny.connect_bidir_to_widget(self.diffraction_space_control_widget.spinBox_Ny)

        return self.diffraction_space_control_widget

    def setup_real_space_control_widget(self):
        """
        Set up the control window.
        """
        self.real_space_control_widget = load_qt_ui_file(sibling_path(__file__, "real_space_control_widget.ui"))
        self.real_space_control_widget.setWindowTitle("Real space")
        self.real_space_control_widget.show()
        self.real_space_control_widget.raise_()
        return self.real_space_control_widget

    def setup_diffraction_space_widget(self):
        """
        Set up the diffraction space window.
        """
        # Create pyqtgraph ImageView object
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.random.random((512,512)))

        # Create virtual detector ROI selector 
        self.virtual_detector_roi = pg.RectROI([256, 256], [50,50], pen=(3,9))
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChanged.connect(self.update_real_space_view)

        # Name, show, return
        self.diffraction_space_widget.setWindowTitle('Diffraction Space')
        self.diffraction_space_widget.show()
        return self.diffraction_space_widget

    def setup_real_space_widget(self):
        """
        Set up the real space window.
        """
        # Create pyqtgraph ImageView object
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.setImage(np.random.random((512,512)))

        # Add point selector connected to displayed diffraction pattern
        self.real_space_point_selector = pg_point_roi(self.real_space_widget.getView())
        self.real_space_point_selector.sigRegionChanged.connect(self.update_diffraction_space_view)

        # Name, show, return
        self.real_space_widget.setWindowTitle('Real Space')
        self.real_space_widget.show()
        return self.real_space_widget

    def setup_console_widget(self):
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push({'np': np, 'app': self})
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.console_widget = RichJupyterWidget()
        self.console_widget.setWindowTitle("4D-STEM IPython Console")
        self.console_widget.kernel_manager = self.kernel_manager
        self.console_widget.kernel_client = self.kernel_client

        self.console_widget.show()
        return self.console_widget


    def setup_geometry(self):
        """
        Arrange windows and their geometries.
        """
        self.diffraction_space_widget.setGeometry(100,0,600,600)
        self.diffraction_space_control_widget.setGeometry(0,0,350,600)
        self.real_space_widget.setGeometry(700,0,600,600)
        self.real_space_control_widget.setGeometry(1150,0,200,600)
        self.console_widget.setGeometry(0,670,1300,170)

        self.console_widget.raise_()
        self.real_space_control_widget.raise_()
        self.real_space_widget.raise_()
        self.diffraction_space_widget.raise_()
        self.diffraction_space_control_widget.raise_()
        return

    ######### Methods controlling responses to user inputs #########

    def load_file(self):
        """
        Loads a file by creating and storing a DataCube object
        """
        fname = self.settings.data_filename.val
        print("Loading file",fname)

        # Instantiate DataCube object
        self.datacube = DataCube(fname)

        # Update scan shape information
        self.R_N = self.datacube.R_N
        self.settings.R_Nx.update_value(1)
        self.settings.R_Ny.update_value(self.R_N)

        # Set the diffraction space image
        self.update_diffraction_space_view()
        self.update_real_space_view()

        # Initial normalization of diffraction space view
        self.diffraction_space_widget.ui.normDivideRadio.setChecked(True)
        self.diffraction_space_widget.normRadioChanged()

        return

    def load_sample_data(self):
        """
        Loads a file by creating and storing a DataCube object
        """
        fname = "sample_data.dm3"
        print("Loading file",fname)

        # Instantiate DataCube object
        self.datacube = DataCube(fname)
        self.datacube.set_scan_shape(10,10)

        # Update scan shape information
        self.R_N = self.datacube.R_N
        self.settings.R_Nx.update_value(10)
        self.settings.R_Ny.update_value(10)

        # Set the diffraction space image
        self.diffraction_space_widget.setImage(self.datacube.data4D[0,0,:,:])
        #self.diffraction_space_widget.setImage(self.datacube.get_summed_diffraction_pattern())
        #self.diffraction_space_widget.setImage(self.data_3Dflattened.swapaxes(1,2))

        return

    def update_diffraction_space_view(self):
        roi_state = self.real_space_point_selector.saveState()
        x0,y0 = roi_state['pos']
        xc,yc = int(x0+1),int(y0+1)

        # Set the diffraction space image
        new_diffraction_space_view, success = self.datacube.get_diffraction_space_view(xc,yc)
        if success:
            self.diffraction_space_view = new_diffraction_space_view
            self.diffraction_space_widget.setImage(self.diffraction_space_view,autoLevels=False)
        else:
            pass
        return

    def update_real_space_view(self):
        # Get slices corresponding to ROI
        slices, transforms = self.virtual_detector_roi.getArraySlice(self.datacube.data4D[0,0,:,:], self.diffraction_space_widget.getImageItem())
        slice_x,slice_y = slices

        # Set the real space view
        new_real_space_view, success = self.datacube.get_real_space_view(slice_y,slice_x)
        if success:
            self.real_space_view = new_real_space_view
            self.real_space_widget.setImage(self.real_space_view,autoLevels=True)
        else:
            pass
        return

    def update_scan_shape_Nx(self):
        R_Nx = self.settings.R_Nx.val
        self.settings.R_Ny.update_value(int(self.datacube.R_N/R_Nx))
        R_Ny = self.settings.R_Ny.val
        try:
            self.datacube.set_scan_shape(R_Ny, R_Nx)
            self.update_real_space_view()
        except ValueError:
            pass
        return

    def update_scan_shape_Ny(self):
        R_Ny = self.settings.R_Ny.val
        self.settings.R_Nx.update_value(int(self.datacube.R_N/R_Ny))
        R_Nx = self.settings.R_Nx.val
        try:
            self.datacube.set_scan_shape(R_Ny, R_Nx)
        except ValueError:
            pass
        #if hasattr(self, "virtual_detector_roi"):
        #    self.update_real_space_view()
        return


    def exec_(self):
        return self.qtapp.exec_()



    ####### DEPRECATED ##########


    #def update_real_space_view(self):
    #    roi_state = self.virtual_detector_roi.saveState()
    #    x0,y0 = roi_state['pos']
    #    slices, transforms = self.virtual_detector_roi.getArraySlice(self.data_3Dflattened, self.diffraction_space_widget.getImageItem())
    #    slice_x, slice_y, slice_z = slices
    #    self.real_space_widget.setImage(self.data4D[:,:,slice_y, slice_x].sum(axis=(2,3)).T)
    #    return

    #def update_diffraction_view(self):
    #    roi_state = self.real_space_point_selector.saveState()
    #    x0,y0 = roi_state['pos']
    #    xc,yc = x0+1,y0+1
    #    stack_num = self.settings.R_Nx.val*int(yc)+int(xc)
    #    self.diffraction_space_widget.setCurrentIndex(stack_num)
    #    return

    #def update_scan_shape_Nx(self):
    #    R_Nx = self.settings.R_Nx.val
    #    self.settings.R_Ny.update_value(int(self.R_N/R_Nx))
    #    R_Ny = self.settings.R_Ny.val
    #    try:
    #        self.data4D = self.data_3Dflattened.reshape(R_Ny,R_Nx,self.Q_Ny,self.Q_Nx)
    #    except ValueError:
    #        pass
    #    if hasattr(self, "virtual_detector_roi"):
    #        self.update_real_space_view()
    #    return

    #def update_scan_shape_Ny(self):
    #    R_Ny = self.settings.R_Ny.val
    #    self.settings.R_Nx.update_value(int(self.R_N/R_Ny))
    #    R_Nx = self.settings.R_Nx.val
    #    try:
    #        self.data4D = self.data_3Dflattened.reshape(R_Ny,R_Nx,self.Q_Ny,self.Q_Nx)
    #    except ValueError:
    #        pass
    #    if hasattr(self, "virtual_detector_roi"):
    #        self.update_real_space_view()
    #    return



    #def on_stem_pt_roi_change(self):
    #    roi_state = self.stem_pt_roi.saveState()
    #    x0,y0 = roi_state['pos']
    #    xc,yc = x0+1, y0+1
    #    stack_num = self.settings.R_Nx.val*int(yc)+int(xc)
    #    self.stack_imv.setCurrentIndex(stack_num)

    #def on_real_space_roi_change(self):
    #    roi_state = self.real_space_roi.saveState()
    #    x0,y0 = roi_state['pos']
    #    slices, transforms = self.virtual_aperture_roi.getArraySlice(self.data_3Dflattened, self.stack_imv.getImageItem())
    #    slice_x, slice_y, slice_z = slices
    #    self.stem_imv.setImage(self.data4D[:,:,slice_y, slice_x].sum(axis=(2,3)).T)

    #def load_file(self):
    #    fname = self.settings.data_filename.val
    #    print("Loading file",fname)
    #
    #    try:
    #        self.dm3f = dm3.DM3(fname, debug=True)
    #        self.data_3Dflattened = self.dm3f.imagedata
    #    except Exception as err:
    #        print("Failed to load", err)
    #        self.data_3Dflattened = np.random.rand(100,512,512)
    #    self.R_N, self.Q_Ny, self.Q_Nx = self.data_3Dflattened.shape
    #
    #    self.diffraction_space_widget.setImage(self.data_3Dflattened.swapaxes(1,2))
    #
    #    self.settings.R_Nx.update_value(1)
    #    self.settings.R_Ny.update_value(self.R_N)
    #    return


############### End of class ###############


if __name__=="__main__":
    app = DataViewer(sys.argv)

    sys.exit(app.exec_())



