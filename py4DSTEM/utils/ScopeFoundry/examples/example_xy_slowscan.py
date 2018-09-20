import sys
from PySide import QtGui

from ScopeFoundry import BaseMicroscopeGUI

# Import Hardware Components
from hardware_components.apd_counter import  APDCounterHardwareComponent
from ScopeFoundry.examples.hardware.dummy_xy_stage import DummyXYStage

# Import Measurement Components
from measurement_components.apd_optimizer_simple import APDOptimizerMeasurement
from measurement_components.simple_xy_scan import SimpleXYScan


class ExampleXYSlowscanGUI(BaseMicroscopeGUI):

    ui_filename = "../../ScopeFoundry/base_gui.ui"

    def setup(self):
        #Add hardware components
        print "Adding Hardware Components"
        self.add_hardware_component(APDCounterHardwareComponent(self))
        self.add_hardware_component(DummyXYStage(self))

        #Add measurement components
        print "Create Measurement objects"
        self.add_measurement_component(APDOptimizerMeasurement(self))
        self.add_measurement_component(SimpleXYScan(self))
        
        #set some default logged quantities
        self.hardware_components['apd_counter'].debug_mode.update_value(True)
        self.hardware_components['apd_counter'].dummy_mode.update_value(True)
        self.hardware_components['apd_counter'].connected.update_value(True)

        #Add additional logged quantities

        # Connect to custom gui



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName("Example XY slowscan App")

    gui = ExampleXYSlowscanGUI(app)
    gui.show()

    sys.exit(app.exec_())
