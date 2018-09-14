#!/Users/Ben/Code/anaconda2/envs/py3/bin/python

import sys
from PySide2 import QtCore, QtWidgets, QtGui


class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Container widget        
        scrollableWidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self)

        ##### Make sub-widgets #####
        # For each, provide handles to connect to their widgets

        # File loading
        dataLoader = DataLoadingWidget()
        self.lineEdit_LoadFile = dataLoader.lineEdit_LoadFile
        self.pushButton_BrowseFiles = dataLoader.pushButton_BrowseFiles
        self.pushButton_Preprocess = dataLoader.pushButton_Preprocess

        # Data cube size and shape
        sizeAndShapeEditor = DataCubeSizeAndShapeWidget()
        self.spinBox_Nx = sizeAndShapeEditor.spinBox_Nx
        self.spinBox_Ny = sizeAndShapeEditor.spinBox_Ny
        self.lineEdit_Binning = sizeAndShapeEditor.lineEdit_Binning
        self.pushButton_Binning = sizeAndShapeEditor.pushButton_Binning
        self.pushButton_SetCropWindow = sizeAndShapeEditor.pushButton_SetCropWindow
        self.pushButton_CropData = sizeAndShapeEditor.pushButton_CropData

        # Create and set layout
        layout.addWidget(dataLoader)
        layout.addWidget(sizeAndShapeEditor)
        scrollableWidget.setLayout(layout)

        # Scroll Area Properties
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(scrollableWidget)
        scrollArea.setFrameStyle(QtWidgets.QFrame.NoFrame)

        # Set the scroll area container to fill the layout of the entire ControlPanel widget
        vLayout = QtWidgets.QVBoxLayout(self)
        vLayout.addWidget(scrollArea)
        self.setLayout(vLayout)

        # Set geometry
        #self.setFixedHeight(600)
        #self.setFixedWidth(300)



class DataLoadingWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Label, Line Edit, Browse Button
        self.label_Filename = QtWidgets.QLabel("Filename")
        self.lineEdit_LoadFile = QtWidgets.QLineEdit("")
        self.pushButton_BrowseFiles = QtWidgets.QPushButton("Browse")
        self.pushButton_Preprocess = QtWidgets.QPushButton("Preprocess")

        # Title
        title_row = QtWidgets.QLabel("Load Data")
        titleFont = QtGui.QFont()
        titleFont.setBold(True)
        title_row.setFont(titleFont)

        # Layout
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.label_Filename, stretch=0)
        top_row.addWidget(self.lineEdit_LoadFile, stretch=5)
        top_row.addWidget(self.pushButton_BrowseFiles, stretch=0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        layout.addLayout(top_row)
        layout.addWidget(self.pushButton_Preprocess,0,QtCore.Qt.AlignRight)

        self.setLayout(layout)
        self.setFixedWidth(260)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class DataCubeSizeAndShapeWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_spinBoxRow = QtWidgets.QHBoxLayout()
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Nx)
        layout_spinBoxRow.addWidget(QtWidgets.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Ny)

        layout_Reshaping = QtWidgets.QVBoxLayout()
        layout_Reshaping.addWidget(QtWidgets.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_spinBoxRow)

        # Binning
        self.lineEdit_Binning = QtWidgets.QLineEdit("")
        self.pushButton_Binning = QtWidgets.QPushButton("Bin Data")

        layout_binningRow = QtWidgets.QHBoxLayout()
        layout_binningRow.addWidget(QtWidgets.QLabel("Bin by:"))
        layout_binningRow.addWidget(self.lineEdit_Binning)
        layout_binningRow.addWidget(self.pushButton_Binning)

        layout_Binning = QtWidgets.QVBoxLayout()
        layout_Binning.addWidget(QtWidgets.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_binningRow)

        # Cropping
        self.pushButton_SetCropWindow = QtWidgets.QPushButton("Set Crop Window")
        self.pushButton_CropData = QtWidgets.QPushButton("Crop Data")

        layout_croppingRow = QtWidgets.QHBoxLayout()
        layout_croppingRow.addWidget(self.pushButton_SetCropWindow)
        layout_croppingRow.addWidget(self.pushButton_CropData)

        layout_Cropping = QtWidgets.QVBoxLayout()
        layout_Cropping.addWidget(QtWidgets.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_croppingRow)

        # Title
        title_row = QtWidgets.QLabel("Reshape, bin, and crop")
        titleFont = QtGui.QFont()
        titleFont.setBold(True)
        title_row.setFont(titleFont)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)

        self.setLayout(layout)
        self.setFixedWidth(260)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

class PreprocessingWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtWidgets.QSpinBox()
        self.spinBox_Ny = QtWidgets.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_Reshaping_Nx = QtWidgets.QHBoxLayout()
        layout_Reshaping_Nx.addWidget(QtWidgets.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_Reshaping_Nx.addWidget(self.spinBox_Nx)
        layout_Reshaping_Ny = QtWidgets.QHBoxLayout()
        layout_Reshaping_Ny.addWidget(QtWidgets.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_Reshaping_Ny.addWidget(self.spinBox_Ny)

        layout_Reshaping_N = QtWidgets.QVBoxLayout()
        layout_Reshaping_N.addLayout(layout_Reshaping_Nx)
        layout_Reshaping_N.addLayout(layout_Reshaping_Ny)

        layout_Reshaping = QtWidgets.QHBoxLayout()
        layout_Reshaping.addWidget(QtWidgets.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_Reshaping_N)

        # Binning
        self.spinBox_Binning_real = QtWidgets.QSpinBox()
        self.spinBox_Binning_diffraction = QtWidgets.QSpinBox()
        self.spinBox_Binning_real.setMaximum(1000)
        self.spinBox_Binning_diffraction.setMaximum(1000)

        layout_Binning_Diffraction = QtWidgets.QHBoxLayout()
        layout_Binning_Diffraction.addWidget(QtWidgets.QLabel("Q"),0,QtCore.Qt.AlignRight)
        layout_Binning_Diffraction.addWidget(self.spinBox_Binning_diffraction,0,QtCore.Qt.AlignRight)
        layout_Binning_Real = QtWidgets.QHBoxLayout()
        layout_Binning_Real.addWidget(QtWidgets.QLabel("R"),0,QtCore.Qt.AlignRight)
        layout_Binning_Real.addWidget(self.spinBox_Binning_real,0,QtCore.Qt.AlignRight)

        layout_Binning_RHS = QtWidgets.QVBoxLayout()
        layout_Binning_RHS.addLayout(layout_Binning_Diffraction)
        layout_Binning_RHS.addLayout(layout_Binning_Real)

        layout_Binning = QtWidgets.QHBoxLayout()
        layout_Binning.addWidget(QtWidgets.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_Binning_RHS)


        # Cropping
        self.checkBox_Crop_Real = QtWidgets.QCheckBox()
        self.checkBox_Crop_Diffraction = QtWidgets.QCheckBox()

        layout_Cropping_Diffraction = QtWidgets.QHBoxLayout()
        layout_Cropping_Diffraction.addWidget(QtWidgets.QLabel("Q"),0,QtCore.Qt.AlignRight)
        layout_Cropping_Diffraction.addWidget(self.checkBox_Crop_Diffraction,0,QtCore.Qt.AlignRight)
        layout_Cropping_Real = QtWidgets.QHBoxLayout()
        layout_Cropping_Real.addWidget(QtWidgets.QLabel("R"),0,QtCore.Qt.AlignRight)
        layout_Cropping_Real.addWidget(self.checkBox_Crop_Real,0,QtCore.Qt.AlignRight)

        layout_Cropping_RHS = QtWidgets.QVBoxLayout()
        layout_Cropping_RHS.addLayout(layout_Cropping_Diffraction)
        layout_Cropping_RHS.addLayout(layout_Cropping_Real)

        layout_Cropping = QtWidgets.QHBoxLayout()
        layout_Cropping.addWidget(QtWidgets.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_Cropping_RHS)

        # Excute
        self.pushButton_Execute = QtWidgets.QPushButton("Execute")
        self.pushButton_Cancel = QtWidgets.QPushButton("Cancel")

        layout_Execute = QtWidgets.QHBoxLayout()
        layout_Execute.addWidget(self.pushButton_Cancel)
        layout_Execute.addWidget(self.pushButton_Execute)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)
        layout.addLayout(layout_Execute)

        self.setLayout(layout)
        self.setFixedWidth(260)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    controlPanel = ControlPanel()
    controlPanel.show()

    app.exec_()





#app = QtWidgets.QApplication(sys.argv)
#controlPanel = ControlPanel()
#controlPanel.show()
#sys.exit(app.exec_())


