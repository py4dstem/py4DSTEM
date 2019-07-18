from __future__ import division, print_function
from PyQt5 import QtCore, QtWidgets
import numpy as np
import sys, os
import pyqtgraph as pg
import gc

from .panels import *
from ..utils import sibling_path, pg_point_roi, LQCollection


class StrainMappingWindow(QtWidgets.QMainWindow):

	def __init__(self,main_window=None):

		QtWidgets.QMainWindow.__init__(self)

		self.main_window = main_window

		self.settings = LQCollection()

		self.strain_window = QtWidgets.QWidget()
		self.strain_window.setWindowTitle('py4DSTEM Strain Mapping')

		self.setup_tabs()

		#move this later:
		self.strain_window.setGeometry(100,100,1000,800)
		self.strain_window.show()






	def setup_tabs(self):

		self.layout = QtWidgets.QVBoxLayout(self.strain_window)

		self.tab_widget = QtWidgets.QTabWidget()

		self.probe_kernel_tab = ProbeKernelTab(main_window=self.main_window)
		self.bragg_disk_tab = BraggDiskTab(main_window=self.main_window)
		self.lattice_vector_tab = LatticeVectorTab(main_window=self.main_window)
		self.strain_map_tab = StrainMapTab(main_window=self.main_window)

		self.tab_widget.addTab(self.probe_kernel_tab,'Probe Kernel')
		self.tab_widget.addTab(self.bragg_disk_tab,"Bragg Disk Detection")
		self.tab_widget.addTab(self.lattice_vector_tab,"Lattice Vectors")
		self.tab_widget.addTab(self.strain_map_tab,"Strain Maps")


		self.layout.addWidget(self.tab_widget)
		self.strain_window.setLayout(self.layout)