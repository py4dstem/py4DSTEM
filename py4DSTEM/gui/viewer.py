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
#  Qt is being run through PySide2 (i.e. Qt for Python). See https://www.qt.io/qt-for-python. Presently PySide is being used.  Note thay PySide2 is only supported in python 3.
#
# pyqtgraph
# pyqtgraph is a library which facilitates fast-running scientific visualization.  See http://pyqtgraph.org/. pyqtgraph is being used for the final data displays.


from __future__ import division, print_function
from PySide2 import QtCore, QtWidgets
import numpy as np
import sys, os
import pyqtgraph as pg
import gc

from gui.dialogs import ControlPanel, PreprocessingWidget, SaveWidget, EditMetadataWidget
from process.datastructure.datacube import DataCube
from utils.ScopeFoundry import LQCollection
from gui.utils import load_qt_ui_file, sibling_path, pg_point_roi
from readwrite.reader import read_data
from readwrite.writer import save_from_datacube
import utils.dm3_lib as dm3

import IPython
if IPython.version_info[0] < 4:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget as RichJupyterWidget
    from IPython.qt.inprocess import QtInProcessKernelManager
else:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager


class DataViewer(QtCore.QObject):
    """
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
        self.datacube = read_data("sample_data.dm3")

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

    ###############################################
    ############ Widget setup methods #############
    ###############################################

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

        print("\nCheckpoint 1\n")
        # File loading
        self.settings.New('data_filename',dtype='file')
        self.settings.data_filename.connect_to_browse_widgets(self.diffraction_space_control_widget.lineEdit_LoadFile, self.diffraction_space_control_widget.pushButton_BrowseFiles)
        self.settings.data_filename.updated_value.connect(self.load_file)
        print("\nCheckpoint 2\n")

        # Scan shape
        self.settings.New('R_Nx', dtype=int, initial=1)
        self.settings.New('R_Ny', dtype=int, initial=1)
        self.settings.R_Nx.updated_value.connect(self.update_scan_shape_Nx)
        self.settings.R_Ny.updated_value.connect(self.update_scan_shape_Ny)
        self.settings.R_Nx.connect_bidir_to_widget(self.diffraction_space_control_widget.spinBox_Nx)
        self.settings.R_Ny.connect_bidir_to_widget(self.diffraction_space_control_widget.spinBox_Ny)

        # Preprocessing and Saving
        self.diffraction_space_control_widget.pushButton_Preprocess.clicked.connect(self.preprocess)
        self.diffraction_space_control_widget.pushButton_EditMetadata.clicked.connect(self.edit_metadata)
        self.diffraction_space_control_widget.pushButton_SaveAs.clicked.connect(self.save_as)
        print("\nCheckpoint 3\n")

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
        self.datacube = None
        gc.collect()
        self.datacube = read_data(fname)

        # Update scan shape information
        self.R_N = self.datacube.R_N
        self.settings.R_Nx.update_value(self.datacube.R_Nx)
        self.settings.R_Ny.update_value(self.datacube.R_Ny)

        # Set the diffraction space image
        self.update_diffraction_space_view()
        self.update_real_space_view()

        # Initial normalization of diffraction space view
        self.diffraction_space_widget.ui.normDivideRadio.setChecked(True)
        self.diffraction_space_widget.normRadioChanged()

        return

    def update_diffraction_space_view(self):
        roi_state = self.real_space_point_selector.saveState()
        x0,y0 = roi_state['pos']
        xc,yc = int(x0+1),int(y0+1)

        # Set the diffraction space image
        new_diffraction_space_view, success = self.datacube.get_diffraction_space_view(yc,xc)
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
        return

    ############ Preprocessing ###########

    def preprocess(self):
        """
        Binning and cropping.
        This method:
            1) opens a separate dialog for preprocessing parameter control
            2) places crop ROIs in both real and diffraction space
            3) on clicking 'Execute', performs specified preprocessing, altering datacube object,
                 then exits the dialog
            4) on clicking "Cancel', exits without any preprocessing.
        """
        # Make widget
        self.preprocessing_widget = PreprocessingWidget()
        self.preprocessing_widget.setWindowTitle("Preprocessing")
        self.preprocessing_widget.show()
        self.preprocessing_widget.raise_()

        # Create new settings
        self.settings.New('binning_r', dtype=int, initial=1)
        self.settings.New('binning_q', dtype=int, initial=1)
        self.settings.New('cropped_r', dtype=bool)
        self.settings.New('cropped_q', dtype=bool)
        self.settings.New('crop_rx_min', dtype=int)
        self.settings.New('crop_rx_max', dtype=int)
        self.settings.New('crop_ry_min', dtype=int)
        self.settings.New('crop_ry_max', dtype=int)
        self.settings.New('crop_qx_min', dtype=int)
        self.settings.New('crop_qx_max', dtype=int)
        self.settings.New('crop_qy_min', dtype=int)
        self.settings.New('crop_qy_max', dtype=int)

        # Reshaping
        self.settings.R_Nx.connect_bidir_to_widget(self.preprocessing_widget.spinBox_Nx)
        self.settings.R_Ny.connect_bidir_to_widget(self.preprocessing_widget.spinBox_Ny)

        # Binning
        self.settings.binning_r.connect_bidir_to_widget(self.preprocessing_widget.spinBox_Binning_real)
        self.settings.binning_q.connect_bidir_to_widget(self.preprocessing_widget.spinBox_Binning_diffraction)

        # Cropping
        self.preprocessing_widget.checkBox_Crop_Real.stateChanged.connect(self.toggleCropROI_real)
        self.preprocessing_widget.checkBox_Crop_Diffraction.stateChanged.connect(self.toggleCropROI_diffraction)

        # Cancel or execute
        self.preprocessing_widget.pushButton_Cancel.clicked.connect(self.cancel_preprocessing)
        self.preprocessing_widget.pushButton_Execute.clicked.connect(self.execute_preprocessing)

    def toggleCropROI_real(self,on=True):
        """
        Checks if checkbox is True or False.  If True, makes a RIO.  If False, removes the ROI.
        """
        if self.preprocessing_widget.checkBox_Crop_Real.isChecked():
            self.crop_roi_real = pg.RectROI([0,0], [self.datacube.R_Nx, self.datacube.R_Ny], pen=(3,9), removable=True, translateSnap=True, scaleSnap=True)
            self.crop_roi_real.setPen(color='r')
            self.real_space_widget.getView().addItem(self.crop_roi_real)
        else:
            if hasattr(self,'crop_roi_real'):
                self.real_space_widget.getView().removeItem(self.crop_roi_real)
                self.crop_roi_real = None
            else:
                pass

    def toggleCropROI_diffraction(self,on=True):
        """
        Checks if checkbox is True or False.  If True, makes a RIO.  If False, removes the ROI.
        """
        if self.preprocessing_widget.checkBox_Crop_Diffraction.isChecked():
            self.crop_roi_diffraction = pg.RectROI([0,0], [self.datacube.Q_Nx,self.datacube.Q_Ny], pen=(3,9), removable=True, translateSnap=True, scaleSnap=True)
            self.crop_roi_diffraction.setPen(color='r')
            self.diffraction_space_widget.getView().addItem(self.crop_roi_diffraction)
        else:
            if hasattr(self,'crop_roi_diffraction'):
                self.diffraction_space_widget.getView().removeItem(self.crop_roi_diffraction)
                self.crop_roi_diffraction = None
            else:
                pass

    def cancel_preprocessing(self):
        # Update settings to reflect no changes
        self.settings.binning_r.update_value(False)
        self.settings.binning_q.update_value(False)
        self.settings.cropped_r.update_value(False)
        self.settings.cropped_q.update_value(False)
        self.settings.crop_rx_min.update_value(False)
        self.settings.crop_rx_max.update_value(False)
        self.settings.crop_ry_min.update_value(False)
        self.settings.crop_ry_max.update_value(False)
        self.settings.crop_qx_min.update_value(False)
        self.settings.crop_qx_max.update_value(False)
        self.settings.crop_qy_min.update_value(False)
        self.settings.crop_qy_max.update_value(False)

        if hasattr(self,'crop_roi_real'):
            self.real_space_widget.view.scene().removeItem(self.crop_roi_real)
        if hasattr(self,'crop_roi_diffraction'):
            self.diffraction_space_widget.view.scene().removeItem(self.crop_roi_diffraction)

        self.preprocessing_widget.close()

    def execute_preprocessing(self):

        if self.preprocessing_widget.checkBox_Crop_Real.isChecked():
            self.settings.cropped_r.update_value(True)
            slices_r, transforms_r = self.crop_roi_real.getArraySlice(self.datacube.data4D[0,0,:,:], self.diffraction_space_widget.getImageItem())
            slice_rx,slice_ry = slices_r
            self.settings.crop_rx_min.update_value(slice_rx.start)
            self.settings.crop_rx_max.update_value(slice_rx.stop)
            self.settings.crop_ry_min.update_value(slice_ry.start)
            self.settings.crop_ry_max.update_value(slice_ry.stop)
        else:
            self.settings.cropped_r.update_value(False)
            slice_rx, slice_ry = None, None
        if self.preprocessing_widget.checkBox_Crop_Diffraction.isChecked():
            self.settings.cropped_q.update_value(True)
            slices_q, transforms_q = self.crop_roi_diffraction.getArraySlice(self.datacube.data4D[0,0,:,:], self.diffraction_space_widget.getImageItem())
            slice_qx,slice_qy = slices_q
            self.settings.crop_qx_min.update_value(slice_qx.start)
            self.settings.crop_qx_max.update_value(slice_qx.stop)
            self.settings.crop_qy_min.update_value(slice_qy.start)
            self.settings.crop_qy_max.update_value(slice_qy.stop)
        else:
            self.settings.cropped_q.update_value(False)
            slice_qx, slice_qy = None, None

        # Update settings
        # Crop and bin
        self.datacube.cropAndBin(self.settings.binning_r.val, self.settings.binning_q.val, self.settings.cropped_r, self.settings.cropped_q, slice_ry, slice_rx, slice_qy, slice_qx)
        self.virtual_detector_roi.setPos(self.virtual_detector_roi.pos()/self.settings.binning_q.val)
        self.virtual_detector_roi.scale(1/self.settings.binning_q.val)
        self.real_space_point_selector.setPos(self.real_space_point_selector.pos()/self.settings.binning_r.val)
        self.update_diffraction_space_view()
        self.update_real_space_view()

        if hasattr(self,'crop_roi_real'):
            self.real_space_widget.view.scene().removeItem(self.crop_roi_real)
        if hasattr(self,'crop_roi_diffraction'):
            self.diffraction_space_widget.view.scene().removeItem(self.crop_roi_diffraction)

        self.preprocessing_widget.close()

    def edit_metadata(self):
        """
        Creates a popup dialog with tabs for different metadata groups, and fields in each
        group with current, editable metadata values.
        """
        # Make widget
        self.EditMetadataWidget = EditMetadataWidget(self.datacube)
        self.EditMetadataWidget.setWindowTitle("Metadata Editor")
        self.EditMetadataWidget.show()
        self.EditMetadataWidget.raise_()

        # Cancel or save
        self.EditMetadataWidget.pushButton_Cancel.clicked.connect(self.cancel_editMetadata)
        self.EditMetadataWidget.pushButton_Save.clicked.connect(self.save_editMetadata)

    def cancel_editMetadata(self):
        self.EditMetadataWidget.close()

    def save_editMetadata(self):
        print("Updating metadata...")
        for i in range(self.EditMetadataWidget.tabs.count()):
            tab = self.EditMetadataWidget.tabs.widget(i)
            # Get appropriate metadata dict
            tabname = self.EditMetadataWidget.tabs.tabText(i)
            metadata_dict_name = [name for name in self.datacube.metadata.__dict__.keys() if tabname[1:] in name][0]
            metadata_dict = getattr(self.datacube.metadata, metadata_dict_name)
            for row in tab.layout().children():
                key=row.itemAt(0).widget().text()
                try:
                    value=row.itemAt(1).widget().text()
                except AttributeError:
                    # Catches alternate widget (QPlainTextEdit) in comments tab
                    value=row.itemAt(1).widget().toPlainText()
                try:
                    value=float(value)
                except ValueError:
                    pass
                metadata_dict[key]=value
        self.EditMetadataWidget.close()
        print("Done.")



    ############ Saving ###########

    def save_as(self):
        """
        Saving files to the .h5 format.
        This method:
            1) opens a separate dialog
            2) puts a name in the "Save as:" field according to the original filename and any
               preprocessing that's been done
            2) Exits with or without saving when 'Save' or 'Cancel' buttons are pressed.
        """
        # Make widget
        save_path = os.path.splitext(self.settings.data_filename.val)[0]+'.h5'
        self.save_widget = SaveWidget(save_path)
        self.save_widget.setWindowTitle("Save as...")
        self.save_widget.show()
        self.save_widget.raise_()

        # Cancel or save
        self.save_widget.pushButton_Cancel.clicked.connect(self.cancel_saveas)
        self.save_widget.pushButton_Execute.clicked.connect(self.execute_saveas)

    def cancel_saveas(self):
        self.save_widget.close()

    def execute_saveas(self):
        f = self.save_widget.lineEdit_SavePath.text()
        print("Saving file to {}.".format(f))
        save_from_datacube(self.datacube,f)
        self.save_widget.close()




    def exec_(self):
        return self.qtapp.exec_()




############### End of class ###############


if __name__=="__main__":
    app = DataViewer(sys.argv)

    sys.exit(app.exec_())



