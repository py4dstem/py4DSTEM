import sys
from PySide import QtGui

from base_gui import BaseMicroscopeGUI

# Import Hardware Components
#from hardware_components.picoharp import PicoHarpHardwareComponent

# Import Measurement Components
#from measurement_components.powermeter_optimizer import PowerMeterOptimizerMeasurement

class ExampleMicroscopeGUI(BaseMicroscopeGUI):
    
    ui_filename = "example_gui.ui"
    
    def setup(self):
        #Add hardware components
        print "Adding Hardware Components"
        
        #self.picoharp_hc = self.add_hardware_component(PicoHarpHardwareComponent(self))

        #Add measurement components
        print "Create Measurement objects"
        #self.apd_optimizer_measure = self.add_measurement_component(APDOptimizerMeasurement(self))


        #Add additional logged quantities

        # Connect to custom gui



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName("Example Foundry Scope App")
    
    gui = ExampleMicroscopeGUI()
    gui.show()
    
    sys.exit(app.exec_())