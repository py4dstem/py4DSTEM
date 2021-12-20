import numpy as np
from copy import copy
import h5py
import matplotlib.pyplot as plt
from .dataobject import DataObject
from .pointlistarray import PointListArray
from .pointlist import PointList
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
        self.radial_profiles = {}

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
        dq = self.coordinates.get_Q_pixel_size()
        units = self.coordinates.get_Q_pixel_units()
        if dq is not None:
            assert(units is not None)
            print("...calibrating Q pixel size...")
            from ...process.calibration import calibrate_Bragg_peaks_pixel_size
            peaks = calibrate_Bragg_peaks_pixel_size(peaks,q_pixel_size=dq)
            if which == 'pixel':
                del(self.peaks[which])
                self.peaks[which] = peaks
                if get_bvm:
                    bvm = self.get_bvm(which=which)
                    return peaks,bvm
                return peaks
        else:
            print('...pixel calibration not found, skipping...')


        # Rotation

        del(self.peaks['all'])
        self.peaks['all'] = peaks
        return peaks


    ####### calibration - measuring calibrations ######

    # Elliptical distortions

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

    # Pixel size

    def get_radial_integral(self,which='ellipse',dq=0.25):
        """

        """
        from ...process.utils import radial_integral
        assert(which in self.bvms.keys())
        bvm = self.bvms[which].data

        q,I = radial_integral(bvm,self.Q_Nx/2,self.Q_Ny/2,dr=dq)
        N = len(q)
        coords = [('q',float),('I',float)]
        data = np.zeros(N,coords)
        data['q'] = q
        data['I'] = I

        ans = PointList(data=data,coordinates=coords,name=which)
        self.radial_profiles[which] = ans
        return ans


    def fit_radial_peak(self,lims,which='ellipse',show=False,ymax=None):
        """

        """
        from ...process.fit import fit_1D_gaussian,gaussian
        assert(which in self.bvms.keys())
        bvm = self.bvms[which].data
        assert(len(lims)==2)
        assert(which in self.radial_profiles.keys()), "This radial profile hasn't been computed"
        profile = self.radial_profiles[which]

        q,I = profile.data['q'],profile.data['I']
        A,mu,sigma = fit_1D_gaussian(q,I,lims[0],lims[1])

        if show: self.show_radial_peak_fit((A,mu,sigma),lims,which,ymax)
        return mu

    def show_radial_profile(self,which='ellipse',ymax=None,
                            q_ref=None,returnfig=False):
        """
        Args:
            which (str): Which radial integral to show.  Must be in
                ('origin','ellipse','pixel','all').
            ymax (number or None): the upper limit of the y-axis
            q_ref (number or tuple/list of numbers or None): if not None, plot
                reference lines at these positions on the q-axis
        """
        from ...visualize import show_qprofile
        assert(which in self.radial_profiles.keys()), "This radial profile has not been computed"
        profile = self.radial_profiles[which]
        dq = self.coordinates.get_Q_pixel_size()
        units = self.coordinates.get_Q_pixel_units()
        if dq is not None:
            assert(units is not None)
            q = profile.data['q'] * dq
        if ymax is None:
            n = len(profile.data)
            ymax = np.max(profile.data['I'][n//4:]) * 1.2
        fig,ax = show_qprofile(q=q,intensity=profile.data['I'],ymax=ymax,
                      xlabel='q ('+units+')',returnfig=True)
        if q_ref is not None:
            ax.vlines(q_ref,0,ax.get_ylim()[1],color='r')
        if returnfig:
            return fig,ax
        else:
            import matplotlib.pyplot as plt
            plt.show()

    def show_radial_peak_fit(self,p,lims,which='ellipse',ymax=None):
        """
        Args:
            p (3-tuple): the params of the fit gaussian, (A,mu,sigma)
            lims (2-tuple): the fit window
            which (str): which calibration state to use
            ymax (number): the upper limit of the y-axis
        """
        from ...visualize import show_qprofile
        from ...process.fit import gaussian
        assert(which in self.radial_profiles.keys()), "This radial profile hasn't been computed"
        profile = self.radial_profiles[which]
        q,I = profile.data['q'],profile.data['I']
        A,mu,sigma = p
        if ymax is None:
            n = len(profile.data)
            ymax = np.max(I[n//4:]) * 1.2
        fig,ax = show_qprofile(q,I,ymax=ymax,returnfig=True)
        ax.vlines(lims,0,ax.get_ylim()[1],color='r')
        ax.vlines(mu,0,ax.get_ylim()[1],color='g')
        ax.plot(q,gaussian(q,A,mu,sigma),color='r')
        plt.show()

    def calculate_Q_pixel_size(self,q_meas,q_known,units='A'):
        """
        Computes the size of the Q-space pixels. Returns and also stores
        the answer in self.coordinates.

        Args:
            q_meas (number): a measured distance in q-space in pixels
            q_known (number): the corresponding known *real space* distance
            unit (str): the units of the real space value of `q_known`
        """
        dq = self.coordinates.calculate_Q_pixel_size(q_meas,q_known,units)
        return dq


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
        dq = self.coordinates.get_Q_pixel_size()
        if dq is not None:
            print(dq)
            xmax,ymax = int(np.round(self.Q_Nx*dq)),int(np.round(self.Q_Ny*dq))
        bvm = DiffractionSlice(
            data=get_bvm(self.peaks[which],xmax,ymax),
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


