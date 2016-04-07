from __future__ import division, print_function
import numpy as np
import dm3_lib as dm3
from ScopeFoundry import BaseApp
import pyqtgraph as pg
import sys
from PySide import QtGui


def pg_point_roi(view_box):
        circ_roi = pg.CircleROI( (0,0), (2,2) , movable=True, pen=(0,9))
        #self.circ_roi.removeHandle(self.circ_roi.getHandles()[0])
        h = circ_roi.addTranslateHandle((0.5,.5))
        h.pen = pg.mkPen('r')
        h.update()
        view_box.addItem(circ_roi)
        circ_roi.removeHandle(0)
        
        return circ_roi


class NCEM4DSTEMQuickViewApp(BaseApp):
    
    name = "SpecMapDataView"
    
    def setup(self):

        #self.ui = load_qt_ui_file(sibling_path(__file__, "x.ui"))

        #self.ui.show()
        #self.ui.raise_()
        
        self.dm3f = dm3.DM3("SEI-stack 1 CL300 bin 4 1 s ss15nm 40x40pixels.dm3",
                            debug=True)
        self.stem_Nx = 40
        self.stem_Ny = 40

        self.dm3f = dm3.DM3("NBED3by150_20nm_a0.65_CL380_30fsbi2_300kv_afterstrain65un.dm3", debug=True)
        self.stem_Nx = 3
        self.stem_Ny = 150

        self.stack_data = self.dm3f.imagedata
        self.stem_N, self.ccd_Ny, self.ccd_Nx = self.stack_data.shape
        
        
        self.data4D = self.stack_data.reshape(self.stem_Ny,self.stem_Nx,self.ccd_Ny, self.ccd_Nx)
        
        self.stack_imv = pg.ImageView()
        self.stack_imv.setImage(self.stack_data.swapaxes(1,2))
        self.stack_imv.setWindowTitle('Stack')
        self.stack_imv.show()
        
        self.stem_imv = pg.ImageView()
        self.stem_imv.setImage(self.data4D.sum(axis=(2,3)).T)
        
        self.stem_pt_roi = pg_point_roi(self.stem_imv.getView())
        
        self.stem_pt_roi.sigRegionChanged.connect(self.on_stem_pt_roi_change)
        
        self.virtual_aperture_roi = pg.RectROI([self.ccd_Nx/2, self.ccd_Ny/2], [50, 50], pen=(3,9))
        self.stack_imv.getView().addItem(self.virtual_aperture_roi)
        self.virtual_aperture_roi.sigRegionChanged.connect(self.on_virtual_aperture_roi_change)

        self.stem_imv.setWindowTitle('STEM image')        
        self.stem_imv.show()
        
        self.console_widget.show()
    
    def on_stem_pt_roi_change(self):
        roi_state = self.stem_pt_roi.saveState()
        #print roi_state
        #xc, y
        x0, y0 = roi_state['pos']
        xc = x0 + 1
        yc = y0 + 1
        #print(xc, yc)
        stack_num = self.stem_Nx*int(yc) + int(xc)
        self.stack_imv.setCurrentIndex(stack_num)
        
    def on_virtual_aperture_roi_change(self):
        roi_state = self.virtual_aperture_roi.saveState()
        x0, y0 = roi_state['pos']
        print(roi_state)
        slices, transforms = self.virtual_aperture_roi.getArraySlice(self.stack_data, self.stack_imv.getImageItem())
        slice_x, slice_y, slice_z = slices
        print(slices)
        self.stem_imv.setImage(self.data4D[:,:,slice_y, slice_x].sum(axis=(2,3)).T)
        
        
                
if __name__ == '__main__':
    app = NCEM4DSTEMQuickViewApp(sys.argv)
    
    sys.exit(app.exec_())