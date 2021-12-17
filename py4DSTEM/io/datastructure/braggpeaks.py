import numpy as np
from copy import copy
import h5py
from .dataobject import DataObject
from .pointlistarray import PointListArray
from .diffraction import DiffractionSlice
from .coordinates import Coordinates
from ...process.utils import tqdmnd

class BraggPeaks(DataObject):
    """
    Contains the positions and intensities of Bragg scattering, and methods
    for working with this data, including calibration and generation of
    Bragg vector maps.
    """
    def __init__(self, braggpeaks, coordinates, **kwargs):
        """
        Args:
            braggpeaks (PointListArray): a set of detected positions and intensities
            coordinates (Coordinates):
        """
        DataObject.__init__(self, **kwargs)
        assert(isinstance(braggpeaks,PointListArray))
        assert(isinstance(coordinates,Coordinates))

        # containers for calibrated and uncalibrated bragg positions 
        self.peaks = {
            'raw':braggpeaks,
            'cal':None
        }
        # and their bragg vector maps
        self.bvms = {
            'raw':None,
            'cal':None
        }

        # coodinates / calibrations
        self.coordinates = coordinates
        self.Q_Nx = coordinates.Q_Nx
        self.Q_Ny = coordinates.Q_Ny

        # vis params
        self.bvm_vis_params = {}
        self.set_bvm_vis_params(cmap='jet',scaling='log')


    ####### calibration methods ######

    def calibrate(self):
        """
        Calibrates using calibration metadata from self.coordinates.
        """
        print('Copying data...')
        peaks = self.peaks['raw'].copy()
        print('Done.')
        print('Looking for calibration measurements...')

        # Origin
        qx0,qy0 = self.coordinates.get_origin()
        if qx0 is not None and qy0 is not None:
            print('...calibrating origin...')
            from ...process.calibration import center_braggpeaks
            peaks = center_braggpeaks(peaks,(qx0,qy0))
        else:
            print('...origin not found, skipping...')

        # Elliptical distortions

        # Pixel size

        # Rotation

        del(self.peaks['cal'])
        self.peaks['cal'] = peaks






    ####### bvm methods #######

    def get_bvm_raw(self):
        """
        """
        from ...process.diskdetection import get_bvm_raw
        bvm = DiffractionSlice(
            data=get_bvm_raw(self.peaks['raw'],self.Q_Nx,self.Q_Ny),
            name='bvm_raw')
        self.bvms['raw'] = bvm
        return bvm

    def get_bvm_cal(self):
        """
        """
        from ...process.diskdetection import get_bvm
        assert(self.peaks['cal'] is not None), "Data has not been calibrated."
        bvm = DiffractionSlice(
            data=get_bvm(self.peaks['cal'],self.Q_Nx,self.Q_Ny),
            name='bvm_cal')
        self.bvms['cal'] = bvm
        return bvm

    def show_bvm_raw(self,**vis_params):
        """
        """
        from ...visualize import show
        bvm = self.bvms['raw'].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)

    def show_bvm_cal(self,**vis_params):
        """
        """
        from ...visualize import show
        assert(self.peaks['cal'] is not None), "Data has not been calibrated."
        bvm = self.bvms['cal'].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)

    def set_bvm_vis_params(self,**kwargs):
        self.bvm_vis_params = kwargs








### Read/Write

def save_braggpeaks_group(group, braggpeaks):
    """
    Expects an open .h5 group and a BraggPeaks instance; saves to the group
    """
    # TODO WRITE THESE FNS
    # save_pointlistarray_group below
    try:
        n_coords = len(pointlistarray.dtype.names)
    except:
        n_coords = 1
    #coords = np.string_(str([coord for coord in pointlistarray.dtype.names]))
    group.attrs.create("coordinates", np.string_(str(pointlistarray.dtype)))
    group.attrs.create("dimensions", n_coords)

    pointlist_dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    name = "data"
    dset = group.create_dataset(name,pointlistarray.shape,pointlist_dtype)

    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data

def get_pointlistarray_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlistarray in an open, correctly formatted H5 file,
        and returns a PointListArray.
    """
    name = g.name.split('/')[-1]
    dset = g['data']
    shape = g['data'].shape
    coordinates = g['data'][0,0].dtype
    pla = PointListArray(coordinates=coordinates,shape=shape,name=name)
    for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
        pla.get_pointlist(i,j).add_dataarray(dset[i,j])
    return pla


