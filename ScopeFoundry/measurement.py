# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:25:48 2014

@author: esbarnard
"""

from PySide import QtCore, QtGui
import threading
import time
from ScopeFoundry.logged_quantity import LQCollection
from ScopeFoundry.helper_funcs import load_qt_ui_file
from collections import OrderedDict
import pyqtgraph as pg

class Measurement(QtCore.QObject):
    
    measurement_sucessfully_completed = QtCore.Signal(()) # signal sent when full measurement is complete
    measurement_interrupted = QtCore.Signal(()) # signal sent when  measurement is complete due to an interruption
    #measurement_state_changed = QtCore.Signal(bool) # signal sent when measurement started or stopped
    
    def __init__(self, app):
        """type app: MicroscopeApp
        """
        
        QtCore.QObject.__init__(self)

        self.app = app
        
        self.display_update_period = 0.1 # seconds
        self.display_update_timer = QtCore.QTimer(self)
        self.display_update_timer.timeout.connect(self.on_display_update_timer)
        self.acq_thread = None
        
        self.interrupt_measurement_called = False
        
        #self.logged_quantities = OrderedDict()
        self.settings = LQCollection()
        self.operations = OrderedDict()
        
        
        self.activation = self.settings.New('activation', dtype=bool, ro=False) # does the user want to the thread to be running
        self.running    = self.settings.New('running', dtype=bool, ro=True) # is the thread actually running?
        self.progress   = self.settings.New('progress', dtype=float, unit="%", si=False, ro=True)

        self.activation.updated_value.connect(self.start_stop)

        self.add_operation("start", self.start)
        self.add_operation("interrupt", self.interrupt)
        self.add_operation("setup", self.setup)
        self.add_operation("setup_figure", self.setup_figure)
        self.add_operation("update_display", self.update_display)
        self.add_operation('show_ui', self.show_ui)
        
        if hasattr(self, 'ui_filename'):
            self.load_ui()
        
        self.setup()
        
        try:
            self._add_control_widgets_to_measurements_tab()
        except Exception as err:
            print "MeasurementComponent: could not add to measurement tab", self.name,  err
        try:
            self._add_control_widgets_to_measurements_tree()
        except Exception as err:
            print "MeasurementComponent: could not add to measurement tree", self.name,  err


    def setup(self):
        """Override this to set up logged quantities and gui connections
        Runs during __init__, before the hardware connection is established
        Should generate desired LoggedQuantities"""
        pass
        #raise NotImplementedError()
        
    def setup_figure(self):
        print "Empty setup_figure called"
        pass
    
    @QtCore.Slot()
    def start(self):
        print "measurement", self.name, "start"
        self.interrupt_measurement_called = False
        if (self.acq_thread is not None) and self.is_measuring():
            raise RuntimeError("Cannot start a new measurement while still measuring")
        self.acq_thread = threading.Thread(target=self._thread_run)
        #self.measurement_state_changed.emit(True)
        self.running.update_value(True)
        self.acq_thread.start()
        self.t_start = time.time()
        self.display_update_timer.start(self.display_update_period*1000)

    def _run(self):
        raise NotImplementedError("Measurement {}._run() not defined".format(self.name))
    
    def _thread_run(self):
        #self.progress_updated.emit(50) # set progress bars to default run position at 50%
        self.set_progress(50)
        try:
            self._run()
            
        #except Exception as err:
        #    self.interrupt_measurement_called = True
        #    raise err
        finally:
            self.running.update_value(False)
            self.set_progress(0)  # set progress bars back to zero
            #self.measurement_state_changed.emit(False)
            if self.interrupt_measurement_called:
                self.measurement_interrupted.emit()
            else:
                self.measurement_sucessfully_completed.emit()

    def set_progress(self, pct):
        self.progress.update_value(pct)
                
    @QtCore.Slot()
    def interrupt(self):
        print "measurement", self.name, "interrupt"
        self.interrupt_measurement_called = True
        self.activation.update_value(False)
        #Make sure display is up to date        
        #self.on_display_update_timer()


    def start_stop(self, start):
        print self.name, "start_stop", start
        if start:
            self.start()
        else:
            self.interrupt()

        
    def is_measuring(self):
        if self.acq_thread is None:
            self.running.update_value(False)
            return False
        else:
            resp =  self.acq_thread.is_alive()
            self.running.update_value(resp)
            return resp
    
    def update_display(self):
        "Override this function to provide figure updates when the display timer runs"
        pass
    
    
    @QtCore.Slot()
    def on_display_update_timer(self):
        try:
            self.update_display()
        except Exception as err:
            pass
            print self.name, "Failed to update figure:", err            
        finally:
            if not self.is_measuring():
                self.display_update_timer.stop()

    def add_logged_quantity(self, name, **kwargs):
        lq = self.settings.New(name=name, **kwargs)
        return lq
    
    def add_operation(self, name, op_func):
        """type name: str
           type op_func: QtCore.Slot
        """
        self.operations[name] = op_func   
    
    def load_ui(self, ui_fname=None):
        # TODO destroy and rebuild UI if it already exists
        if ui_fname is not None:
            self.ui_filename = ui_fname
        # Load Qt UI from .ui file
        self.ui = load_qt_ui_file(self.ui_filename)
        self.show_ui()
        
    def show_ui(self):
        self.ui.show()
        self.ui.activateWindow()
        #self.ui.raise() #just to be sure it's on top
    
    def _add_control_widgets_to_measurements_tab(self):
        cwidget = self.app.ui.measurements_tab_scrollArea_content_widget
        
        self.controls_groupBox = QtGui.QGroupBox(self.name)
        self.controls_formLayout = QtGui.QFormLayout()
        self.controls_groupBox.setLayout(self.controls_formLayout)
        
        cwidget.layout().addWidget(self.controls_groupBox)
                
        self.control_widgets = OrderedDict()
        for lqname, lq in self.logged_quantities.items():
            #: :type lq: LoggedQuantity
            if lq.choices is not None:
                widget = QtGui.QComboBox()
            elif lq.dtype in [int, float]:
                if lq.si:
                    widget = pg.SpinBox()
                else:
                    widget = QtGui.QDoubleSpinBox()
            elif lq.dtype in [bool]:
                widget = QtGui.QCheckBox()  
            elif lq.dtype in [str]:
                widget = QtGui.QLineEdit()
            lq.connect_bidir_to_widget(widget)

            # Add to formlayout
            self.controls_formLayout.addRow(lqname, widget)
            self.control_widgets[lqname] = widget
            
            
        self.op_buttons = OrderedDict()
        for op_name, op_func in self.operations.items(): 
            op_button = QtGui.QPushButton(op_name)
            op_button.clicked.connect(op_func)
            self.controls_formLayout.addRow(op_name, op_button)
            
            
    def _add_control_widgets_to_measurements_tree(self, tree=None):
        if tree is None:
            tree = self.app.ui.measurements_treeWidget
        
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Measurements", "Value"])

        self.tree_item = QtGui.QTreeWidgetItem(tree, [self.name, ""])
        tree.insertTopLevelItem(0, self.tree_item)
        #self.tree_item.setFirstColumnSpanned(True)
        self.tree_progressBar = QtGui.QProgressBar()
        tree.setItemWidget(self.tree_item, 1, self.tree_progressBar)
        self.progress.updated_value.connect(self.tree_progressBar.setValue)

        # Add logged quantities to tree
        for lqname, lq in self.logged_quantities.items():
            #: :type lq: LoggedQuantity
            if lq.choices is not None:
                widget = QtGui.QComboBox()
            elif lq.dtype in [int, float]:
                if lq.si:
                    widget = pg.SpinBox()
                else:
                    widget = QtGui.QDoubleSpinBox()
            elif lq.dtype in [bool]:
                widget = QtGui.QCheckBox()  
            elif lq.dtype in [str]:
                widget = QtGui.QLineEdit()
            lq.connect_bidir_to_widget(widget)

            lq_tree_item = QtGui.QTreeWidgetItem(self.tree_item, [lqname, ""])
            self.tree_item.addChild(lq_tree_item)
            lq.hardware_tree_widget = widget
            tree.setItemWidget(lq_tree_item, 1, lq.hardware_tree_widget)
            #self.control_widgets[lqname] = widget
                
        # Add operation buttons to tree
        self.op_buttons = OrderedDict()
        for op_name, op_func in self.operations.items(): 
            op_button = QtGui.QPushButton(op_name)
            op_button.clicked.connect(op_func)
            self.op_buttons[op_name] = op_button
            #self.controls_formLayout.addRow(op_name, op_button)
            op_tree_item = QtGui.QTreeWidgetItem(self.tree_item, [op_name, ""])
            tree.setItemWidget(op_tree_item, 1, op_button)
