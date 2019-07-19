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

		self.start_peakfinding_button = QtWidgets.QPushButton("Find All Bragg Disks")
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


###########################################
######## LATTICE VECTOR TAB ###############
###########################################

class LatticeVectorTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window

		layout = QtWidgets.QHBoxLayout()

		self.settings_pane = LatticeVectorSettingsPane(main_window=self.main_window)
		self.viz_pane = LatticeVectorVisualizationPane(main_window=self.main_window)

		layout.addWidget(self.settings_pane)
		layout.addWidget(self.viz_pane)

		self.setLayout(layout)

class LatticeVectorSettingsPane(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self, "Lattice Vector Determination")

		layout = QtWidgets.QVBoxLayout()

		radonbox = QtWidgets.QGroupBox("Radon Transform")
		radonform = QtWidgets.QFormLayout()

		self.radon_N_spinBox = QtWidgets.QSpinBox()
		self.radon_N_spinBox.setMinimum(1)
		self.radon_N_spinBox.setMaximum(7200)
		self.radon_N_spinBox.setValue(360)
		radonform.addRow("Number angles",self.radon_N_spinBox)

		self.radon_sigma_spinBox = QtWidgets.QSpinBox()
		self.radon_sigma_spinBox.setMinimum(0)
		self.radon_sigma_spinBox.setMaximum(500)
		self.radon_sigma_spinBox.setValue(2)
		radonform.addRow("Sigma",self.radon_sigma_spinBox)

		self.radon_min_spacing_spinBox = QtWidgets.QSpinBox()
		self.radon_min_spacing_spinBox.setMinimum(1)
		self.radon_min_spacing_spinBox.setMaximum(1200)
		self.radon_min_spacing_spinBox.setValue(2)
		radonform.addRow("Minimum Spacing",self.radon_min_spacing_spinBox)

		self.radon_min_rel_int_spinBox = QtWidgets.QDoubleSpinBox()
		self.radon_min_rel_int_spinBox.setMinimum(0)
		self.radon_min_rel_int_spinBox.setMaximum(1)
		self.radon_min_rel_int_spinBox.setValue(0.05)
		self.radon_min_rel_int_spinBox.setDecimals(3)
		self.radon_min_rel_int_spinBox.setSingleStep(0.02)
		radonform.addRow("Minimum Relative Intensity",self.radon_min_rel_int_spinBox)

		self.radon_update_button = QtWidgets.QPushButton("Calculate")
		radonform.addRow(self.radon_update_button)

		radonbox.setLayout(radonform)
		layout.addWidget(radonbox)

		directionbox = QtWidgets.QGroupBox("Lattice Vector Directions")
		directionform = QtWidgets.QFormLayout()

		self.direction_sigma_spinBox = QtWidgets.QSpinBox()
		self.direction_sigma_spinBox.setMinimum(0)
		self.direction_sigma_spinBox.setMaximum(1200)
		self.direction_sigma_spinBox.setValue(2)
		directionform.addRow("Sigma",self.direction_sigma_spinBox)

		self.directions_min_spacing_spinBox = QtWidgets.QSpinBox()
		self.directions_min_spacing_spinBox.setMinimum(1)
		self.directions_min_spacing_spinBox.setMaximum(1200)
		self.directions_min_spacing_spinBox.setValue(2)
		directionform.addRow("Minimum Spacing",self.directions_min_spacing_spinBox)

		self.directions_min_rel_int_spinBox = QtWidgets.QDoubleSpinBox()
		self.directions_min_rel_int_spinBox.setMinimum(0)
		self.directions_min_rel_int_spinBox.setMaximum(1)
		self.directions_min_rel_int_spinBox.setValue(0.05)
		self.directions_min_rel_int_spinBox.setSingleStep(0.005)
		self.directions_min_rel_int_spinBox.setDecimals(3)
		directionform.addRow("Minimum Relative Intensity",self.directions_min_rel_int_spinBox)

		self.directions_index1_spinbox = QtWidgets.QSpinBox()
		self.directions_index1_spinbox.setMinimum(0)
		self.directions_index1_spinbox.setMaximum(100)
		self.directions_index1_spinbox.setValue(0)
		directionform.addRow("Index 1",self.directions_index1_spinbox)

		self.directions_index2_spinBox = QtWidgets.QSpinBox()
		self.directions_index2_spinBox.setMinimum(0)
		self.directions_index2_spinBox.setMaximum(100)
		self.directions_index2_spinBox.setValue(0)
		directionform.addRow("Index 2",self.directions_index2_spinBox)

		directionbox.setLayout(directionform)
		layout.addWidget(directionbox)

		lengthbox = QtWidgets.QGroupBox("Lattice Vector Lengths")
		lengthform = QtWidgets.QFormLayout()

		self.lengths_spacing_thresh_spinBox = QtWidgets.QDoubleSpinBox()
		self.lengths_spacing_thresh_spinBox.setMinimum(0)
		self.lengths_spacing_thresh_spinBox.setMaximum(50)
		self.lengths_spacing_thresh_spinBox.setValue(1.5)
		self.lengths_spacing_thresh_spinBox.setSingleStep(0.05)
		self.lengths_spacing_thresh_spinBox.setDecimals(2)
		lengthform.addRow("Spacing Threshold",self.lengths_spacing_thresh_spinBox)

		self.lengths_sigma_spinBox = QtWidgets.QSpinBox()
		self.lengths_sigma_spinBox.setMinimum(0)
		self.lengths_sigma_spinBox.setMaximum(100)
		self.lengths_sigma_spinBox.setValue(1)
		lengthform.addRow("Sigma",self.lengths_sigma_spinBox)

		self.lengths_min_spacing_spinBox = QtWidgets.QSpinBox()
		self.lengths_min_spacing_spinBox.setMinimum(0)
		self.lengths_min_spacing_spinBox.setMaximum(100)
		self.lengths_min_spacing_spinBox.setValue(2)
		lengthform.addRow("Minimum Spacing",self.lengths_min_spacing_spinBox)

		self.lengths_min_rel_int_spinBox = QtWidgets.QDoubleSpinBox()
		self.lengths_min_rel_int_spinBox.setMinimum(0)
		self.lengths_min_rel_int_spinBox.setMaximum(1)
		self.lengths_min_rel_int_spinBox.setValue(0.1)
		self.lengths_min_rel_int_spinBox.setDecimals(3)
		self.lengths_min_rel_int_spinBox.setSingleStep(0.005)
		lengthform.addRow("Minimum Relative Intensity",self.lengths_min_rel_int_spinBox)

		lengthbox.setLayout(lengthform)
		layout.addWidget(lengthbox)

		self.setLayout(layout)

class LatticeVectorVisualizationPane(QtWidgets.QGroupBox):
	def __init__(self,main_window=None):
		QtWidgets.QGroupBox.__init__(self,"Lattice Vectors")

		toprow = QtWidgets.QHBoxLayout()
		radongroup = QtWidgets.QGroupBox("Radon Transform")
		self.radon_plot = pg.PlotWidget()
		toplayout = QtWidgets.QHBoxLayout()
		toplayout.addWidget(self.radon_plot)
		radongroup.setLayout(toplayout)
		toprow.addWidget(radongroup)

		bottomrow = QtWidgets.QHBoxLayout()

		crossinggroup = QtWidgets.QGroupBox("Crossings Plot")
		crossinglayout = QtWidgets.QHBoxLayout()
		self.crossings_plot = pg.PlotWidget()
		crossinglayout.addWidget(self.crossings_plot)
		crossinggroup.setLayout(crossinglayout)
		bottomrow.addWidget(crossinggroup)

		lvgroup = QtWidgets.QGroupBox("Bragg Vector Map with Lattice Vectors")
		lvlayout = QtWidgets.QHBoxLayout()
		self.lattice_plot = pg.PlotWidget()
		lvlayout.addWidget(self.lattice_plot)
		lvgroup.setLayout(lvlayout)
		bottomrow.addWidget(lvgroup)

		layout = QtWidgets.QVBoxLayout()
		layout.addLayout(toprow)
		layout.addLayout(bottomrow)
		self.setLayout(layout)

#################################################
################ STRAIN MAP TAB #################
#################################################

class StrainMapTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window

		layout = QtWidgets.QVBoxLayout()

		# virtual image row
		vimgrow = QtWidgets.QHBoxLayout()

		vimggroup = QtWidgets.QGroupBox("Virtual Image")
		vimglayout = QtWidgets.QHBoxLayout()

		self.DP_view = pg.ImageView()
		self.DP_ROI = pg.RectROI([256,256],[512,512], pen=(3,9))
		self.DP_view.getView().addItem(self.DP_ROI)
		vimglayout.addWidget(self.DP_view)

		self.RS_view = pg.ImageView()
		self.RS_ROI = pg_point_roi(self.RS_view.getView())
		vimglayout.addWidget(self.RS_view)
		vimggroup.setLayout(vimglayout)
		vimgrow.addWidget(vimggroup)

		layout.addLayout(vimgrow)

		# strain Group
		strainrow = QtWidgets.QHBoxLayout()
		straingroup = QtWidgets.QGroupBox("Strain Maps")
		strainlayout = QtWidgets.QVBoxLayout()

		strainTopRow = QtWidgets.QHBoxLayout()
		self.exx_view = pg.ImageView()
		strainTopRow.addWidget(self.exx_view)
		self.eyy_view = pg.ImageView()
		strainTopRow.addWidget(self.eyy_view)
		strainlayout.addLayout(strainTopRow)

		strainBottomRow = QtWidgets.QHBoxLayout()
		self.exy_view = pg.ImageView()
		strainBottomRow.addWidget(self.exy_view)
		self.theta_view = pg.ImageView()
		strainBottomRow.addWidget(self.theta_view)
		strainlayout.addLayout(strainBottomRow)

		straingroup.setLayout(strainlayout)
		strainrow.addWidget(straingroup)
		layout.addLayout(strainrow)

		# export group
		exportrow = QtWidgets.QHBoxLayout()
		exportgroup = QtWidgets.QGroupBox("Export")
		exportlayout = QtWidgets.QHBoxLayout()

		self.export_processing_button = QtWidgets.QPushButton("Export Processing")
		self.export_all_button = QtWidgets.QPushButton("Export All")

		exportlayout.addWidget(self.export_processing_button)
		exportlayout.addWidget(self.export_all_button)
		exportgroup.setLayout(exportlayout)
		exportrow.addWidget(exportgroup)

		layout.addLayout(exportrow)
		

		self.setLayout(layout)


