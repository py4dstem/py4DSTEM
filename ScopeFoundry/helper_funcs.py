from PySide import QtCore, QtGui, QtUiTools
from collections import OrderedDict
import os

class OrderedAttrDict(object):

    def __init__(self):
        self._odict = OrderedDict()
        
    def add(self, name, obj):
        self._odict[name] = obj
        self.__dict__[name] = obj
        return obj
    
    def keys(self):
        return self._odict.keys()
    def values(self):
        return self._dict.values()
    def items(self):
        return self._odict.items()
    
    def __len__(self):
        return len(self._odict)

def sibling_path(a, b):
    return os.path.join(os.path.dirname(a), b)


def load_qt_ui_file(ui_filename):
    ui_loader = QtUiTools.QUiLoader()
    ui_file = QtCore.QFile(ui_filename)
    ui_file.open(QtCore.QFile.ReadOnly)
    ui = ui_loader.load(ui_file)
    ui_file.close()
    return ui

def confirm_on_close(widget, title="Close ScopeFoundry?", message="Do you wish to shut down ScopeFoundry?"):
    widget.closeEventEater = CloseEventEater(title, message)
    widget.installEventFilter(widget.closeEventEater)
    
class CloseEventEater(QtCore.QObject):
    
    def __init__(self, title="Close ScopeFoundry?", message="Do you wish to shut down ScopeFoundry?"):
        QtCore.QObject.__init__(self)
        self.title = title
        self.message = message
    
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Close:
            # eat close event
            print "close"
            reply = QtGui.QMessageBox.question(None, 
                                               self.title, 
                                               self.message,
                                               QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                QtGui.QApplication.quit()
                event.accept()
            else:
                event.ignore()
            return True
        else:
            # standard event processing            
            return QtCore.QObject.eventFilter(self,obj, event)