######## Viewer for 4D STEM data ########
#
# Defines a class -- Interactive4DSTEMDataViewer - enabling a simple GUI for
# interacting with 4D STEM datasets.
#
# Relevant documentation for lower level code:
# ScopeFoundry is a flexible package for both scientific data visualization and control of labrotory experiments.  See http://www.scopefoundry.org/.
#  Qt is being run through Pyside/PySide2/PyQt/Qt for Python. See https://www.qt.io/qt-for-python. 
# pyqtgraph is a library which facilitates fast-running scientific visualization.  See http://pyqtgraph.org/.


from __future__ import division, print_function
import numpy as np
import sys
from ScopeFoundry import BaseApp
from ScopeFoundry.helper_funcs import load_qt_ui_file, sibling_path
import pyqtgraph as pg
import dm3_lib as dm3

class Interactive4DSTEMDataViewer(BaseApp):
    """
    Interactive4DSTEMDataViewer objects inherit from the ScopeFoundry.BaseApp class.
    ScopeFoundry.BaseApp objects inherit from the QtCore.QObject class.
    Additional functionality is provided by pyqtgraph widgets.

    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = Interactive4DSTEMDataViewer(sys.argv)
        app.exec_()
    """
    def setup(self):

        """
        Sets up the interface.

        Includes three primary windows:
        -Diffraction space view (detector space)
        -Real space view (scan positions + virtual detectors)
        -iPython Console

        Note that the diffraction space window also contains dialogs for basic user inputs.
        (i.e. file loading, etc.)
        """

        # Load the main user interface window
        self.ui = load_qt_ui_file(sibling_path(__file__, "quick_view_gui.ui"))
        self.ui.show()
        self.ui.raise_()

        # Create new self.settings fields
        self.settings.New('data_filename',dtype='file')
        self.settings.New('stem_Nx', dtype=int, initial=1)
        self.settings.New('stem_Ny', dtype=int, initial=1)

        # Methods to be run when UI widgets are changed
        self.settings.data_filename.updated_value.connect(self.on_change_data_filename)
        self.settings.stem_Nx.updated_value.connect(self.on_change_stem_nx)
        self.settings.stem_Ny.updated_value.connect(self.on_change_stem_ny)

         # Connect UI changes to updates in self.settings
        self.settings.data_filename.connect_to_browse_widgets(self.ui.data_filename_lineEdit, self.ui.data_filename_browse_pushButton)
        self.settings.stem_Nx.connect_bidir_to_widget(self.ui.stem_Nx_doubleSpinBox)
        self.settings.stem_Ny.connect_bidir_to_widget(self.ui.stem_Ny_doubleSpinBox)

        # Create and set up display of diffraction patterns
        self.stack_imv = pg.ImageView()
        self.stack_imv.setImage(self.stack_data.swapaxes(1,2))
        self.ui.stack_groupBox.layout().addWidget(self.stack_imv)

        # Create and set up display in real space
        self.stem_imv = pg.ImageView()
        self.stem_imv.setImage(self.data4D.sum(axis=(2,3)).T)
        self.stem_pt_roi = pg_point_roi(self.stem_imv.getView())
        self.stem_pt_roi.sigRegionChanged.connect(self.on_stem_pt_roi_change)
        self.virtual_aperture_roi = pg.RectROI([self.ccd_Nx/2, self.ccd_Ny/2], [50,50], pen=(3,9))
        self.stack_imv.getView().addItem(self.virtual_aperture_roi)
        self.virtual_aperture_roi.sigRegionChanged.connect(self.on_virtual_aperture_roi_change)
        self.stem_imv.setWindowTitle('STEM image')
        self.stem_imv.show()

        # Make a iPython Console widget
        self.console_widget.show()

        # Arrange windows and set their geometries
        px = 600
        self.ui.setGeometry(0,0,px,2*px)
        self.stem_imv.setGeometry(px,0,px,px)
        self.console_widget.setGeometry(px,1.11*px,px,px)
        self.stack_imv.activateWindow()
        self.stack_imv.raise_()
        self.stem_imv.raise_()
        self.console_widget.raise_()


    #### Methods controlling responses to user inputs ####

    def on_change_data_filename(self):
        fname = self.settings.data_filename.val
        print("Loading file",fname)

        try:
            self.dm3f = dm3.DM3(fname, debug=True)
            self.stack_data = self.dm3f.imagedata
        except Exception as err:
            print("Failed to load", err)
            self.stack_data = np.random.rand(100,512,512)
        self.stem_N, self.ccd_Ny, self.ccd_Nx = self.stack_data.shape
        if hasattr(self, 'stem_pt_roi'):
            self.on_stem_pt_roi_change()

        self.stack_imv.setImage(self.stack_data.swapaxes(1,2))

        self.settings.stem_Nx.update_value(1)
        self.settings.stem_Ny.update_value(self.stem_N)

    def on_change_stem_nx(self):
        stem_Nx = self.settings.stem_Nx.val
        self.settings.stem_Ny.update_value(int(self.stem_N/stem_Nx))
        stem_Ny = self. settings.stem_Ny.val
        self.data4D = self.stack_data.reshape(stem_Ny,stem_Nx,self.ccd_Ny,self.ccd_Nx)
        if hasattr(self, "virtual_aperture_roi"):
            self.on_virtual_aperture_roi_change()

    def on_change_stem_ny(self):
        stem_Ny = self.settings.stem_Ny.val
        self.settings.stem_Nx.update_value(int(self.stem_N/stem_Ny))
        stem_Nx = self.settings.stem_Nx.val
        self.data4D = self.stack_data.reshape(stem_Ny,stem_Nx,self.ccd_Ny,self.ccd_Nx)
        if hasattr(self, "virtual_aperture_roi"):
            self.on_virtual_aperture_roi_change()

    def on_stem_pt_roi_change(self):
        roi_state = self.stem_pt_roi.saveState()
        x0,y0 = roi_state['pos']
        xc,yc = x0+1, y0+1
        stack_num = self.settings.stem_Nx.val*int(yc)+int(xc)
        self.stack_imv.setCurrentIndex(stack_num)

    def on_virtual_aperture_roi_change(self):
        roi_state = self.virtual_aperture_roi.saveState()
        x0,y0 = roi_state['pos']
        slices, transforms = self.virtual_aperture_roi.getArraySlice(self.stack_data, self.stack_imv.getImageItem())
        slice_x, slice_y, slice_z = slices
        self.stem_imv.setImage(self.data4D[:,:,slice_y, slice_x].sum(axis=(2,3)).T)

############### End of class ###############


def pg_point_roi(view_box):
    """
    Utility function for point selection.
    Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI( (0,0), (2,2), movable=True, pen=(0,9))
    h = circ_roi.addTranslateHandle((0.5,0.5))
    h.pen = pg.mkPen('r')
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi



if __name__=="__main__":
    app = Interactive4DSTEMDataViewer(sys.argv)

    sys.exit(app.exec_())



