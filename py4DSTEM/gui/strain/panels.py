import sys
from PyQt5 import QtCore, QtWidgets, QtGui


class ProbeKernelTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window

		# 


class BraggDiskTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window


class LatticeVectorTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window


class StrainMapTab(QtWidgets.QWidget):
	def __init__(self,main_window=None):
		QtWidgets.QWidget.__init__(self)

		self.main_window = main_window