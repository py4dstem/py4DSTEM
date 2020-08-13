from __future__ import division, print_function
from PyQt5 import QtCore, QtWidgets
import numpy as np
import sys, os
import pyqtgraph as pg
import gc

from .panels import *
from ..gui_utils import sibling_path, pg_point_roi, LQCollection


class StrainMappingWindow(QtWidgets.QMainWindow):

	def __init__(self,main_window=None):

		QtWidgets.QMainWindow.__init__(self)

		self.main_window = main_window
		self.datacube = None

		#self.settings = LQCollection()

		self.strain_window = QtWidgets.QWidget()
		self.strain_window.setWindowTitle('py4DSTEM Strain Mapping')

		self.strain_window.setGeometry(100,100,1200,800)
		self.strain_window.show()

		# These are flags that will be flipped as we proceed,
		# to tell different draw functions if their data exists
		self.probe_kernel_accepted = False
		self.bragg_peaks_accepted = False
		self.BVM_accepted = False 
		self.sinogram_accepted = False
		self.lattice_vectors_accepted = False



	def copy_vac_DC_from_browser(self):
		try:
			self.vac_datacube = self.main_window.datacube
		except:
			print("Couldn't transfer datacube from Browser...")

		if self.vac_datacube is not None:
			self.probe_kernel_tab.update_views()



	def setup_tabs(self):

		self.layout = QtWidgets.QVBoxLayout(self.strain_window)

		self.tab_widget = QtWidgets.QTabWidget()

		self.probe_kernel_tab = ProbeKernelTab(main_window=self.main_window)
		self.bragg_disk_tab = BraggDiskTab(main_window=self.main_window)
		self.lattice_vector_tab = LatticeVectorTab(main_window=self.main_window)
		self.strain_map_tab = StrainMapTab(main_window=self.main_window)

		self.tab_widget.addTab(self.probe_kernel_tab,"Probe Kernel")
		self.tab_widget.addTab(self.bragg_disk_tab,"Bragg Disk Detection")
		self.tab_widget.addTab(self.lattice_vector_tab,"Lattice Vectors")
		self.tab_widget.addTab(self.strain_map_tab,"Strain Maps")

		self.probe_kernel_tab_index = self.tab_widget.indexOf(self.probe_kernel_tab)
		self.bragg_disk_tab_index = self.tab_widget.indexOf(self.bragg_disk_tab)
		self.lattice_vector_tab_index = self.tab_widget.indexOf(self.lattice_vector_tab)
		self.strain_map_tab_index = self.tab_widget.indexOf(self.strain_map_tab)

		# disable all the tabs not yet available (comment out for debugging)
		self.tab_widget.setTabEnabled(self.bragg_disk_tab_index, False)
		self.tab_widget.setTabEnabled(self.lattice_vector_tab_index, False)
		self.tab_widget.setTabEnabled(self.strain_map_tab_index, False)


		self.layout.addWidget(self.tab_widget)
		self.strain_window.setLayout(self.layout)
