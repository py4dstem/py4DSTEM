from PySide import QtCore, QtGui
from logged_quantity import LQCollection#, LoggedQuantity
from collections import OrderedDict
import pyqtgraph as pg

class HardwareComponent(QtCore.QObject):

    def add_logged_quantity(self, name, **kwargs):
        #lq = LoggedQuantity(name=name, **kwargs)
        #self.logged_quantities[name] = lq
        #return lq
        return self.settings.New(name, **kwargs)

    def add_operation(self, name, op_func):
        """type name: str
           type op_func: QtCore.Slot
        """
        self.operations[name] = op_func   
            
    def __init__(self, app, debug=False):
        QtCore.QObject.__init__(self)

        self.app = app

        #self.logged_quantities = OrderedDict()
        self.settings = LQCollection()
        self.operations = OrderedDict()

        self.connected = self.add_logged_quantity("connected", dtype=bool)
        self.connected.updated_value.connect(self.enable_connection)
        
        self.debug_mode = self.add_logged_quantity("debug_mode", dtype=bool, initial=debug)
        
        self.setup()

        try:
            self._add_control_widgets_to_hardware_tab()
        except Exception as err:
            print "HardwareComponent: could not add to hardware tab", self.name,  err
        try:
            self._add_control_widgets_to_hardware_tree()
        except Exception as err:
            print "HardwareComponent: could not add to hardware tree", self.name,  err

        self.has_been_connected_once = False
        
        self.is_connected = False
        
    def setup(self):
        """
        Runs during __init__, before the hardware connection is established
        Should generate desired LoggedQuantities, operations
        """
        raise NotImplementedError()

    def _add_control_widgets_to_hardware_tab(self):
        cwidget = self.app.ui.hardware_tab_scrollArea_content_widget
        
        self.controls_groupBox = QtGui.QGroupBox(self.name)
        self.controls_formLayout = QtGui.QFormLayout()
        self.controls_groupBox.setLayout(self.controls_formLayout)
        
        cwidget.layout().addWidget(self.controls_groupBox)
        
        #self.connect_hardware_checkBox = QtGui.QCheckBox("Connect to Hardware")
        #self.controls_formLayout.addRow("Connect", self.connect_hardware_checkBox)
        #self.connect_hardware_checkBox.stateChanged.connect(self.enable_connection)

        
        self.control_widgets = OrderedDict()
        for lqname, lq in self.settings.as_dict().items():
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
        
        self.read_from_hardware_button = QtGui.QPushButton("Read From Hardware")
        self.read_from_hardware_button.clicked.connect(self.read_from_hardware)
        self.controls_formLayout.addRow("Logged Quantities:", self.read_from_hardware_button)
    
    def _add_control_widgets_to_hardware_tree(self):
        tree = self.app.ui.hardware_treeWidget
        #tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Hardware", "Value"])

        self.tree_item = QtGui.QTreeWidgetItem(tree, [self.name, "="*10])
        tree.insertTopLevelItem(0, self.tree_item)
        self.tree_item.setFirstColumnSpanned(True)
        
        for lqname, lq in self.settings.as_dict().items():
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
            #self.controls_formLayout.addRow(lqname, widget)
            lq_tree_item = QtGui.QTreeWidgetItem(self.tree_item, [lqname, ""])
            self.tree_item.addChild(lq_tree_item)
            lq.hardware_tree_widget = widget
            tree.setItemWidget(lq_tree_item, 1, lq.hardware_tree_widget)
            #self.control_widgets[lqname] = widget
                
        self.op_buttons = OrderedDict()
        for op_name, op_func in self.operations.items(): 
            op_button = QtGui.QPushButton(op_name)
            op_button.clicked.connect(op_func)
            self.op_buttons[op_name] = op_button
            #self.controls_formLayout.addRow(op_name, op_button)
            op_tree_item = QtGui.QTreeWidgetItem(self.tree_item, [op_name, ""])
            tree.setItemWidget(op_tree_item, 1, op_button)

        self.tree_read_from_hardware_button = QtGui.QPushButton("Read From\nHardware")
        self.tree_read_from_hardware_button.clicked.connect(self.read_from_hardware)
        #self.controls_formLayout.addRow("Logged Quantities:", self.read_from_hardware_button)
        self.read_from_hardware_button_tree_item = QtGui.QTreeWidgetItem(self.tree_item, ["Logged Quantities:", ""])
        self.tree_item.addChild(self.read_from_hardware_button_tree_item)
        tree.setItemWidget(self.read_from_hardware_button_tree_item, 1, self.tree_read_from_hardware_button)

    @QtCore.Slot()    
    def read_from_hardware(self):
        for name, lq in self.settings.as_dict().items():
            if self.debug_mode.val: print "read_from_hardware", name
            lq.read_from_hardware()
        
    
    def connect(self):
        """
        Opens a connection to hardware
        and connects hardware to associated LoggedQuantities
        """
        raise NotImplementedError()
        
        
    def disconnect(self):
        """
        Disconnects the hardware and severs hardware--LoggedQuantities link
        """
        
        raise NotImplementedError()
    
    @QtCore.Slot(bool)
    def enable_connection(self, enable=True):
        if enable:
            self.connect()
        else:
            self.disconnect()