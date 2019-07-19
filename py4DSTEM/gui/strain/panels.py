import sys
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from ..dialogs import SectionLabel
import numpy as np
from ..utils import pg_point_roi


class ProbeKernelTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window

		# make the DC load selector
		self.dc_loader = QtWidgets.QTabWidget()
		self.load_vac_DC_tab = VacuumDCTab(main_window=self.main_window)
		self.use_main_DC_tab = UseMainDCTab(main_window=self.main_window)
		self.dc_loader.addTab(self.load_vac_DC_tab,"Load Vacuum Datacube")
		self.dc_loader.addTab(self.use_main_DC_tab,"Use Vacuum ROI")

		# make the layout for the load selector
		layout_load = QtWidgets.QHBoxLayout()
		layout_load.addWidget(self.dc_loader)


		# make the top right views and ROIs (DP and RS)
		self.diffraction_widget = pg.ImageView()
		self.diffraction_widget.setImage(np.zeros((512,512)))
		self.diffraction_ROI = pg.RectROI([256,256],[512,512], pen=(3,9))
		self.diffraction_widget.getView().addItem(self.diffraction_ROI)
		self.diffraction_ROI.sigRegionChangeFinished.connect(self.update_RS)

		self.realspace_widget = pg.ImageView()
		self.realspace_widget.setImage(np.zeros((25,25)))
		self.realspace_ROI = pg.RectROI([256,256],[5,5],pen=(3,9))
		self.realspace_widget.getView().addItem(self.realspace_ROI)
		self.realspace_ROI.sigRegionChangeFinished.connect(self.update_DP)

		# make the layout for the RS and DP
		layout_DPRS = QtWidgets.QHBoxLayout()
		layout_DPRS.addWidget(self.diffraction_widget,1)
		layout_DPRS.addWidget(self.realspace_widget,1)

		# make the layout for the top half
		top_half_layout = QtWidgets.QHBoxLayout()
		top_half_layout.addLayout(layout_load)
		top_half_layout.addLayout(layout_DPRS)

		## make the settings box
		self.probe_kernel_settings = ProkeKernelSettings(main_window=self.main_window)
		self.probe_kernel_display = ProbeKernelDisplay(main_window=self.main_window)

		# make the layout for the bottom half
		bottom_half_layout = QtWidgets.QHBoxLayout()
		bottom_half_layout.addWidget(self.probe_kernel_settings)
		bottom_half_layout.addWidget(self.probe_kernel_display)

		# add the layouts and apply
		main_layout = QtWidgets.QVBoxLayout()
		main_layout.addLayout(top_half_layout)
		main_layout.addLayout(bottom_half_layout)

		self.setLayout(main_layout)


	def update_RS(self):
		return 0

	def update_DP(self):
		return 0



class VacuumDCTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		# Load
		load_widget = QtWidgets.QWidget()
		load_widget_layout = QtWidgets.QVBoxLayout()

		self.label_Filename = QtWidgets.QLabel("Filename")
		self.lineEdit_LoadFile = QtWidgets.QLineEdit("")
		self.pushButton_BrowseFiles = QtWidgets.QPushButton("Browse")

		self.loadRadioAuto = QtWidgets.QRadioButton("Automatic")
		self.loadRadioAuto.setChecked(True)
		self.loadRadioMMAP = QtWidgets.QRadioButton("DM Memory Map")
		self.loadRadioGatan = QtWidgets.QRadioButton("Gatan K2 Binary")

		line1 = QtWidgets.QHBoxLayout()
		line1.addWidget(self.label_Filename,stretch=0)
		line1.addWidget(self.lineEdit_LoadFile,stretch=1)
		optionLine = QtWidgets.QHBoxLayout()
		optionLine.addWidget(self.loadRadioAuto)
		optionLine.addWidget(self.loadRadioMMAP)
		optionLine.addWidget(self.loadRadioGatan)

		line2 = QtWidgets.QHBoxLayout()
		line2.addWidget(self.pushButton_BrowseFiles,0,QtCore.Qt.AlignRight)

		load_widget_layout.addLayout(line1)
		load_widget_layout.addLayout(optionLine)
		load_widget_layout.addLayout(line2)
		load_widget_layout.setSpacing(0)
		load_widget_layout.setContentsMargins(0,0,0,0)
		load_widget.setLayout(load_widget_layout)

		# Layout
		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(SectionLabel('Load'))
		layout.addWidget(load_widget)

		self.setLayout(layout)


class UseMainDCTab(QtWidgets.QWidget):
	def __init__(self,main_window):
		QtWidgets.QWidget.__init__(self)
		self.button_copy_DC = QtWidgets.QPushButton("Copy from Browser")
		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.button_copy_DC)
		self.setLayout(layout)


class ProkeKernelSettings(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self,"Probe Kernel Settings")

		#groupbox = QtWidgets.QGroupBox("Probe Kernel Settings")
		
		settingsGroup = QtWidgets.QFormLayout()
		
		self.mask_threshold_spinBox = QtWidgets.QDoubleSpinBox()
		self.mask_threshold_spinBox.setMinimum(0.0)
		self.mask_threshold_spinBox.setMaximum(1.0)
		self.mask_threshold_spinBox.setSingleStep(0.05)
		self.mask_threshold_spinBox.setValue(0.2)
		self.mask_threshold_spinBox.setDecimals(2)
		settingsGroup.addRow("Mask Threshold", self.mask_threshold_spinBox)

		self.mask_expansion_spinBox = QtWidgets.QSpinBox()
		self.mask_expansion_spinBox.setMinimum(0.0)
		self.mask_expansion_spinBox.setMaximum(500)
		self.mask_expansion_spinBox.setSingleStep(1)
		self.mask_expansion_spinBox.setValue(12)
		settingsGroup.addRow("Mask Expansion", self.mask_expansion_spinBox)

		self.mask_opening_spinBox = QtWidgets.QSpinBox()
		self.mask_opening_spinBox.setMinimum(0.0)
		self.mask_opening_spinBox.setMaximum(500)
		self.mask_opening_spinBox.setSingleStep(1)
		self.mask_opening_spinBox.setValue(3)
		settingsGroup.addRow("Mask Threshold", self.mask_opening_spinBox)

		self.button_generate_probe = QtWidgets.QPushButton("Generate Probe")
		self.button_accept_probe = QtWidgets.QPushButton("Accept")

		button_layout = QtWidgets.QHBoxLayout()
		button_layout.addWidget(self.button_generate_probe)
		button_layout.addWidget(self.button_accept_probe)

		boxlayout = QtWidgets.QVBoxLayout()
		boxlayout.addLayout(settingsGroup)
		boxlayout.addLayout(button_layout)

		self.setLayout(boxlayout)


class ProbeKernelDisplay(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)
		layout = QtWidgets.QHBoxLayout()

		self.probe_kernel_view = pg.ImageView()
		self.probe_kernel_view.setImage(np.zeros((100,100)))

		self.probe_kernel_linetrace = pg.PlotWidget()

		layout.addWidget(self.probe_kernel_view)
		layout.addWidget(self.probe_kernel_linetrace)

		self.setLayout(layout)


################################################
########### BRAGG DISK TAB #####################
################################################
class BraggDiskTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window

		# make the settings pane
		layout = QtWidgets.QHBoxLayout()

		leftpane = QtWidgets.QVBoxLayout()
		self.bragg_disk_settings_pane = BraggDiskSettings(main_window=self.main_window)
		self.bragg_disk_control_box = BraggDiskControlBox(main_window=self.main_window)
		leftpane.addWidget(self.bragg_disk_settings_pane)
		leftpane.addWidget(self.bragg_disk_control_box)
		layout.addLayout(leftpane)

		#rightpane = QtWidgets.QHBoxLayout()
		self.bragg_disk_preview_pane = BraggDiskPreviewPane(main_window=self.main_window)
		layout.addWidget(self.bragg_disk_preview_pane)

		self.setLayout(layout)


class BraggDiskSettings(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self,"Bragg Disk Detection Settings")

		form = QtWidgets.QFormLayout()

		self.corr_power_spinBox = QtWidgets.QDoubleSpinBox()
		self.corr_power_spinBox.setMinimum(0.0)
		self.corr_power_spinBox.setMaximum(1.0)
		self.corr_power_spinBox.setSingleStep(0.02)
		self.corr_power_spinBox.setValue(1.0)
		form.addRow("Correlation Power", self.corr_power_spinBox)

		self.sigma_spinBox = QtWidgets.QSpinBox()
		self.sigma_spinBox.setMinimum(0)
		self.sigma_spinBox.setMaximum(100)
		self.sigma_spinBox.setSingleStep(1)
		self.sigma_spinBox.setValue(2)
		form.addRow("Correlation Smoothing Sigma (px)", self.sigma_spinBox)

		self.edge_boundary_spinBox = QtWidgets.QSpinBox()
		self.edge_boundary_spinBox.setMinimum(0)
		self.edge_boundary_spinBox.setMaximum(1000)
		self.edge_boundary_spinBox.setSingleStep(1)
		self.edge_boundary_spinBox.setValue(20)
		form.addRow("Edge Boundary (px)", self.edge_boundary_spinBox)

		self.min_relative_intensity_spinBox = QtWidgets.QDoubleSpinBox()
		self.min_relative_intensity_spinBox.setMinimum(0.0)
		self.min_relative_intensity_spinBox.setMaximum(1.0)
		self.min_relative_intensity_spinBox.setSingleStep(0.001)
		self.min_relative_intensity_spinBox.setValue(0.005)
		self.min_relative_intensity_spinBox.setDecimals(4)
		form.addRow("Minimum Relative Intensity", self.min_relative_intensity_spinBox)

		self.relative_to_peak_spinBox = QtWidgets.QSpinBox()
		self.relative_to_peak_spinBox.setMinimum(0)
		self.relative_to_peak_spinBox.setMaximum(20)
		self.relative_to_peak_spinBox.setSingleStep(1)
		self.relative_to_peak_spinBox.setValue(1)
		form.addRow("Relative to peak #", self.relative_to_peak_spinBox)

		self.min_peak_spacing_spinBox = QtWidgets.QSpinBox()
		self.min_peak_spacing_spinBox.setMinimum(0)
		self.min_peak_spacing_spinBox.setMaximum(1000)
		self.min_peak_spacing_spinBox.setSingleStep(1)
		self.min_peak_spacing_spinBox.setValue(60)
		form.addRow("Minimum Peak Spacing (px)", self.min_peak_spacing_spinBox)

		self.max_num_peaks_spinBox = QtWidgets.QSpinBox()
		self.max_num_peaks_spinBox.setMinimum(0)
		self.max_num_peaks_spinBox.setMaximum(1000)
		self.max_num_peaks_spinBox.setSingleStep(1)
		self.max_num_peaks_spinBox.setValue(70)
		form.addRow("Max Number Peaks", self.max_num_peaks_spinBox)

		self.subpixel_chooser = QtWidgets.QComboBox()
		self.subpixel_chooser.addItem("None")
		self.subpixel_chooser.addItem("Parabolic")
		self.subpixel_chooser.addItem("Upsample DFT")
		form.addRow("Subpixel Mode", self.subpixel_chooser)

		self.upsample_factor_spinBox = QtWidgets.QSpinBox()
		self.upsample_factor_spinBox.setMinimum(1)
		self.upsample_factor_spinBox.setMaximum(256)
		self.upsample_factor_spinBox.setSingleStep(2)
		self.upsample_factor_spinBox.setValue(16)
		form.addRow("Upsample Factor", self.upsample_factor_spinBox)

		self.setLayout(form)


class BraggDiskControlBox(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self,"Peakfinding Progress")

		layout = QtWidgets.QHBoxLayout()

		self.start_peakfinding_button = QtWidgets.QPushButton("Find Bragg Disks")
		self.bragg_peak_progressbar = QtWidgets.QProgressBar()

		layout.addWidget(self.bragg_peak_progressbar)
		layout.addWidget(self.start_peakfinding_button)

		self.setLayout(layout)

class BraggDiskPreviewPane(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self,"Bragg Disk Detection Preview")

		# make the RS previews
		toprow = QtWidgets.QHBoxLayout()

		self.bragg_preview_realspace_1 = pg.ImageView()
		self.bragg_preview_realspace_1.setImage(np.zeros((25,25)))
		self.bragg_preview_realspace_1_selector = pg_point_roi(self.bragg_preview_realspace_1.getView())
		toprow.addWidget(self.bragg_preview_realspace_1)

		self.bragg_preview_realspace_2 = pg.ImageView()
		self.bragg_preview_realspace_2.setImage(np.zeros((25,25)))
		self.bragg_preview_realspace_2_selector = pg_point_roi(self.bragg_preview_realspace_2.getView())
		toprow.addWidget(self.bragg_preview_realspace_2)

		self.bragg_preview_realspace_3 = pg.ImageView()
		self.bragg_preview_realspace_3.setImage(np.zeros((25,25)))
		self.bragg_preview_realspace_3_selector = pg_point_roi(self.bragg_preview_realspace_3.getView())
		toprow.addWidget(self.bragg_preview_realspace_3)

		bottomrow = QtWidgets.QHBoxLayout()

		self.bragg_preview_DP_1 = pg.PlotWidget()
		# setup
		bottomrow.addWidget(self.bragg_preview_DP_1)

		self.bragg_preview_DP_2 = pg.PlotWidget()
		# setup
		bottomrow.addWidget(self.bragg_preview_DP_2)

		self.bragg_preview_DP_3 = pg.PlotWidget()
		# setup
		bottomrow.addWidget(self.bragg_preview_DP_3)

		layout = QtWidgets.QVBoxLayout()
		layout.addLayout(toprow)
		layout.addLayout(bottomrow)

		self.setLayout(layout)


class LatticeVectorTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window


class StrainMapTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window