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
            'cal':None,
            'cal_origin':None
        }
        # and their bragg vector maps
        self.bvms = {
            'raw':None,
            'cal':None,
            'cal_origin':None
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
            print('...origin calibrations not found, skipping...')

        # Elliptical distortions
        p_ellipse = self.coordinates.get_p_ellipse()
        if all([x is not None for x in p_ellipse]):
            print("...calibrating origin...")
            from ...process.calibration import correct_braggpeak_elliptical_distortions
            peaks = correct_braggpeak_elliptical_distortions(peaks,p_ellipse)
        else:
            print('...elliptical calibrations not found, skipping...')


        # Pixel size

        # Rotation

        del(self.peaks['cal'])
        self.peaks['cal'] = peaks

    def calibrate_origin(self,get_bvm=True):
        """
        Calibrates the origin using calibration metadata from self.coordinates.
        """
        print('Copying data...')
        peaks = self.peaks['raw'].copy()
        print('Done.')

        qx0,qy0 = self.coordinates.get_origin()
        assert(qx0 is not None and qy0 is not None), "Origin calibrations not found!"
        if qx0 is not None and qy0 is not None:
            from ...process.calibration import center_braggpeaks
            peaks = center_braggpeaks(peaks,(qx0,qy0))
            del(self.peaks['cal_origin'])
            self.peaks['cal_origin'] = peaks
            if get_bvm:
                self.get_bvm_origin()
        else:
            print('...origin not found, skipping...')

    def fit_elliptical_distortions(self,fitradii):
        """
        Fits the elliptical distortions using an annular region
        of a bragg vector map.

        TODO: update fn to use peaks, not bvm
        """
        from ...process.calibration import fit_ellipse_1D
        assert('cal_origin' in self.peaks.keys()), "Calibrate the origin!"
        assert('cal_origin' in self.bvms.keys()), "Compute the origin-corrected BVM!"
        peaks = self.peaks['cal_origin']
        bvm = self.bvms['cal_origin'].data
        p_ellipse = fit_ellipse_1D(bvm,(bvm.shape[0]/2,bvm.shape[1]/2),fitradii)
        _,_,a,b,theta = p_ellipse
        self.coordinates.set_ellipse((a,b,theta))
        return p_ellipse





    ####### bvm methods #######

    def get_bvm(self,calibrated=True):
        """
        Args:
            calibrated (bool): if True tries to use the calibrated
                peak positions; if they are not found or if False,
                uses the raw peak positions
        """
        if calibrated and self.peaks['cal'] is not None:
            bvm = self.get_bvm_cal()
        else:
            bvm = self.get_bvm_raw()
        return bvm

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

    def get_bvm_origin(self):
        """
        """
        from ...process.diskdetection import get_bvm
        assert(self.peaks['cal_origin'] is not None), "Data has not been calibrated."
        bvm = DiffractionSlice(
            data=get_bvm(self.peaks['cal_origin'],self.Q_Nx,self.Q_Ny),
            name='bvm_cal_origin')
        self.bvms['cal_origin'] = bvm
        return bvm

    def show_bvm(self,calibrated=True,**vis_params):
        """
        Args:
            calibrated (bool): if True tries to show the calibrated
                bvm; if it is not found or if False, shows the raw bvm
        """
        if calibrated and self.bvms['cal'] is not None:
            self.show_bvm_cal(**vis_params)
        else:
            self.show_bvm_raw(**vis_params)

    def show_bvm_raw(self,**vis_params):
        """
        """
        from ...visualize import show
        assert(self.bvms['raw'] is not None), "BVM not found"
        bvm = self.bvms['raw'].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)

    def show_bvm_cal(self,**vis_params):
        """
        """
        from ...visualize import show
        assert(self.bvms['cal'] is not None), "BVM not found"
        bvm = self.bvms['cal'].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)

    def set_bvm_vis_params(self,**kwargs):
        self.bvm_vis_params = kwargs











### Read/Write

# TODO TODO TODO

def save_braggpeaks_group(group, braggpeaks):
    """
    Expects an open .h5 group and a BraggPeaks instance; saves to the group
    """
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


