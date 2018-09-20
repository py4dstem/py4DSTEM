from PySide import QtCore, QtGui
from PySide.QtCore import Qt
import numpy as np


#https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg17575.html
# plus more
class NumpyQTableModel(QtCore.QAbstractTableModel):
    def __init__(self, narray, col_names=None, row_names=None, fmt="%g", copy=True, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.copy = copy
        if self.copy:
            self._array = narray.copy()
        else:
            self._array = narray
        self.col_names = col_names
        self.row_names = row_names
        self.fmt=fmt

    def rowCount(self, parent=None):
        return self._array.shape[0]

    def columnCount(self, parent=None):
        return self._array.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role==Qt.EditRole:
                row = index.row()
                col = index.column()
                return self.fmt % self._array[row, col]
        return None

    def setData(self, index, value, role=Qt.EditRole):
        print index,value, role
        jj,ii = index.row(), index.column()
                     
        print 'setData', ii,jj
        #return QtCore.QAbstractTableModel.setData(self, *args, **kwargs)
        
        try:
            self._array[jj,ii] = value
            self.dataChanged.emit(index, index) # topLeft, bottomRight indexes of change
            return True
        except Exception as err:
            print "setData err:", err
            return False

    def set_array(self, narray):
        #print "set_array"
        if self.copy:
            self._array = narray.copy()
        else:
            self._array = narray
        self.layoutChanged.emit()
        self.dataChanged.emit((0,0), (self.rowCount(), self.columnCount()))
    
    
    def flags(self, *args, **kwargs):
        #return QtCore.QAbstractTableModel.flags(self, *args, **kwargs)
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal and self.col_names:
            return self.col_names[section]
        if role == Qt.DisplayRole and orientation == Qt.Vertical and self.row_names:
            return self.row_names[section]
        return QtCore.QAbstractTableModel.headerData(self, section-1, orientation, role)    
    
class ArrayLQ_QTableModel(NumpyQTableModel):
    def __init__(self, lq, col_names=None, row_names=None, parent=None):
        print lq.val
        NumpyQTableModel.__init__(self, lq.val, col_names=col_names, row_names=row_names, parent=parent)
        self.lq = lq
        self.lq.updated_value[()].connect(self.on_lq_updated_value)
        self.dataChanged.connect(self.on_dataChanged)

    def on_lq_updated_value(self):
        #print "ArrayLQ_QTableModel", self.lq.name, 'on_lq_updated_value'
        self.set_array(self.lq.val)
    
    def on_dataChanged(self,topLeft=None, bottomRight=None):
        #print "ArrayLQ_QTableModel", self.lq.name, 'on_dataChanged'
        self.lq.update_value(np.array(self._array))
        #self.lq.send_display_updates(force=True)
    
    
    
if __name__ == '__main__':
    qtapp = QtGui.QApplication([])
    
    import numpy as np
    
    A = np.random.rand(10,5)
    B = np.random.rand(12,4)
    
    table_view = QtGui.QTableView()
    table_view_model = NumpyQTableModel(narray=A, col_names=['Peak', 'FWHM','center', 'asdf', '!__!'])
    table_view.setModel(table_view_model)
    table_view.show()
    table_view.raise_()
    
    table_view_model.set_array(B)
    
    qtapp.exec_()
