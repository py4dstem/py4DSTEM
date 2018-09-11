#!/Users/Ben/Code/anaconda2/envs/py3/bin/python

import sys
from PySide import QtCore, QtGui


class ControlPanel(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Container widget        
        scrollableWidget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout(self)

        ##### Make sub-widgets #####
        # For each, provide handles to connect to their widgets

        # File loading
        dataLoader = DataLoadingWidget()
        self.lineEdit_LoadFile = dataLoader.lineEdit_LoadFile
        self.pushButton_BrowseFiles = dataLoader.pushButton_BrowseFiles

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
        scrollArea = QtGui.QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(scrollableWidget)
        scrollArea.setFrameStyle(QtGui.QFrame.NoFrame)

        # Set the scroll area container to fill the layout of the entire ControlPanel widget
        vLayout = QtGui.QVBoxLayout(self)
        vLayout.addWidget(scrollArea)
        self.setLayout(vLayout)

        # Set geometry
        #self.setFixedHeight(600)
        #self.setFixedWidth(300)



class DataLoadingWidget(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Label, Line Edit, Browse Button
        self.label_Filename = QtGui.QLabel("Filename")
        self.lineEdit_LoadFile = QtGui.QLineEdit("")
        self.pushButton_BrowseFiles = QtGui.QPushButton("Browse")

        # Title
        title_row = QtGui.QLabel("Load Data")
        titleFont = QtGui.QFont()
        titleFont.setBold(True)
        title_row.setFont(titleFont)

        # Layout
        top_row = QtGui.QHBoxLayout()
        top_row.addWidget(self.label_Filename, stretch=0)
        top_row.addWidget(self.lineEdit_LoadFile, stretch=5)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        #verticalSpacer = QtGui.QSpacerItem(0, 10, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        #layout.addItem(verticalSpacer)
        layout.addLayout(top_row)
        layout.addWidget(self.pushButton_BrowseFiles,0,QtCore.Qt.AlignRight)

        self.setLayout(layout)
        self.setFixedWidth(260)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Fixed))

class DataCubeSizeAndShapeWidget(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        # Reshaping - Nx and Ny
        self.spinBox_Nx = QtGui.QSpinBox()
        self.spinBox_Ny = QtGui.QSpinBox()
        self.spinBox_Nx.setMaximum(100000)
        self.spinBox_Ny.setMaximum(100000)

        layout_spinBoxRow = QtGui.QHBoxLayout()
        layout_spinBoxRow.addWidget(QtGui.QLabel("Nx"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Nx)
        layout_spinBoxRow.addWidget(QtGui.QLabel("Ny"),0,QtCore.Qt.AlignRight)
        layout_spinBoxRow.addWidget(self.spinBox_Ny)

        layout_Reshaping = QtGui.QVBoxLayout()
        layout_Reshaping.addWidget(QtGui.QLabel("Scan shape"),0,QtCore.Qt.AlignCenter)
        layout_Reshaping.addLayout(layout_spinBoxRow)

        # Binning
        self.lineEdit_Binning = QtGui.QLineEdit("")
        self.pushButton_Binning = QtGui.QPushButton("Bin Data")

        layout_binningRow = QtGui.QHBoxLayout()
        layout_binningRow.addWidget(QtGui.QLabel("Bin by:"))
        layout_binningRow.addWidget(self.lineEdit_Binning)
        layout_binningRow.addWidget(self.pushButton_Binning)

        layout_Binning = QtGui.QVBoxLayout()
        layout_Binning.addWidget(QtGui.QLabel("Binning"),0,QtCore.Qt.AlignCenter)
        layout_Binning.addLayout(layout_binningRow)

        # Cropping
        self.pushButton_SetCropWindow = QtGui.QPushButton("Set Crop Window")
        self.pushButton_CropData = QtGui.QPushButton("Crop Data")

        layout_croppingRow = QtGui.QHBoxLayout()
        layout_croppingRow.addWidget(self.pushButton_SetCropWindow)
        layout_croppingRow.addWidget(self.pushButton_CropData)

        layout_Cropping = QtGui.QVBoxLayout()
        layout_Cropping.addWidget(QtGui.QLabel("Cropping"),0,QtCore.Qt.AlignCenter)
        layout_Cropping.addLayout(layout_croppingRow)

        # Title
        title_row = QtGui.QLabel("Reshape, bin, and crop")
        titleFont = QtGui.QFont()
        titleFont.setBold(True)
        title_row.setFont(titleFont)

        # Layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(title_row,0,QtCore.Qt.AlignCenter)
        layout.addLayout(layout_Reshaping)
        layout.addLayout(layout_Binning)
        layout.addLayout(layout_Cropping)

        self.setLayout(layout)
        self.setFixedWidth(260)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Fixed))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    controlPanel = ControlPanel()
    controlPanel.show()

    app.exec_()





#app = QtGui.QApplication(sys.argv)
#controlPanel = ControlPanel()
#controlPanel.show()
#sys.exit(app.exec_())


