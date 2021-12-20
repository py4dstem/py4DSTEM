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
            'all':None,
            'origin':None,
            'ellipse':None
        }
        # and their bragg vector maps
        self.bvms = {
            'raw':None,
            'all':None,
            'origin':None,
            'ellipse':None
        }

        # coodinates / calibrations
        self.coordinates = coordinates
        self.Q_Nx = coordinates.Q_Nx
        self.Q_Ny = coordinates.Q_Ny

        # vis params
        self.bvm_vis_params = {}
        self.set_bvm_vis_params(cmap='jet',scaling='log')


    ####### calibration - applying calibrations ######

    def calibrate(self,which='all',get_bvm=True):
        """
        Calibrates the peak positions.  The new calibrated PointListArray
        of bragg peaks is both returned and stored in self.peaks[which].

        Args:
            which (str): Which calibrations to perform.  If 'all' is passed,
                performs applies all the measured calibrations found in
                self.coordinates. Otherwise, indicates that only calibrations up
                to this one should be applied - must be in
                ('origin','ellipse','pixel','all').
            get_bvms (bool): If True, computes the bvm of the calibrated data.

        Returns:
            (variable) if get_bvm is False, returns the calibrated PointListArray.
                if get_bvm is True, returns a 2-tuple of the calibrated data and the
                bvm.
        """
        assert(which in ('all','origin','ellipse','pixel')), "Invalid value for argument `which`, {}".format(which)
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
            if which == 'origin':
                del(self.peaks[which])
                self.peaks[which] = peaks
                if get_bvm:
                    bvm = self.get_bvm(which=which)
                    return peaks,bvm
                return peaks
        else:
            print('...origin calibrations not found, skipping...')
            if which == 'origin': return

        # Elliptical distortions
        p_ellipse = self.coordinates.get_p_ellipse()
        if all([x is not None for x in p_ellipse]):
            print("...calibrating origin...")
            from ...process.calibration import correct_braggpeak_elliptical_distortions
            peaks = correct_braggpeak_elliptical_distortions(peaks,p_ellipse)
            if which == 'ellipse':
                del(self.peaks[which])
                self.peaks[which] = peaks
                if get_bvm:
                    bvm = self.get_bvm(which=which)
                    return peaks,bvm
                return peaks
        else:
            print('...elliptical calibrations not found, skipping...')
            if which == 'ellipse': return


        # Pixel size

        # Rotation

        del(self.peaks['all'])
        self.peaks['all'] = peaks
        return peaks


    ####### calibration - measuring calibrations ######

    def fit_elliptical_distortions(self,fitradii):
        """
        Fits the elliptical distortions using an annular region
        of a bragg vector map.

        TODO: update fn to use peaks, not bvm
        """
        from ...process.calibration import fit_ellipse_1D
        assert('origin' in self.peaks.keys()), "Calibrate the origin!"
        assert('origin' in self.bvms.keys()), "Compute the origin-corrected BVM!"
        peaks = self.peaks['origin']
        bvm = self.bvms['origin'].data
        p_ellipse = fit_ellipse_1D(bvm,(bvm.shape[0]/2,bvm.shape[1]/2),fitradii)
        _,_,a,b,theta = p_ellipse
        self.coordinates.set_ellipse((a,b,theta))
        return p_ellipse





    ####### bvm methods #######

    def get_bvm(self,which='raw'):
        """
        Compute a bvm. Both returns and stores it in self.bvms[which].

        Args:
            which (str): Which bvm to compute, i.e. the Bragg peak positions
                at which stage of calibration to use. Must be in
                ('raw','origin','ellipse','pixel','all').

        Returns:
            (DiffractionSlice) the bvm
        """
        assert(which in ('raw','origin','ellipse','pixel','all')), "Invalid value for argument `which`, {}".format(which)
        assert(self.peaks[which] is not None), "This calibration state has not been computed, please calibrate the peak positions first with `self.calibrate(which = {} )`".format(which)
        peaks = self.peaks[which]
        if which=='raw':
            from ...process.diskdetection import get_bvm_raw as get_bvm
        else:
            from ...process.diskdetection import get_bvm
        bvm = DiffractionSlice(
            data=get_bvm(self.peaks[which],self.Q_Nx,self.Q_Ny),
            name=which)
        self.bvms[which] = bvm
        return bvm

    def show_bvm(self,which='raw',**vis_params):
        """
        Args:
            which (str): Which bvm to show, i.e. the Bragg peak positions
                at which stage of calibration to use. Must be in
                ('raw','origin','ellipse','pixel','all').
        """
        from ...visualize import show
        assert(which in ('raw','origin','ellipse','pixel','all')), "Invalid value for argument `which`, {}".format(which)
        assert(self.bvms[which] is not None), "This bvm has not been computed, please compute it first with `self.get_bvm(which = {} )`".format(which)
        bvm = self.bvms[which].data
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


