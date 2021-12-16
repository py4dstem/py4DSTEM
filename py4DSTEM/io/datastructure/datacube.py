# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and
# methods pointing to processing functions - generally defined in other files in
# the ./process directory.

from collections.abc import Sequence
from tempfile import TemporaryFile

import numpy as np
import numba as nb
import h5py

from .dataobject import DataObject
from .diffraction import DiffractionSlice
from .real import RealSlice
from .coordinates import Coordinates
from .pointlist import PointList
from ...process import preprocess
from ...process import virtualimage
from ...process import virtualimage_viewer
from ...process.utils import tqdmnd, bin2D

class DataCube(DataObject):
    """
    A class storing a single 4D STEM dataset.

    Args:
        data (ndarray): the data, in a 4D array of shape ``(R_Nx,R_Ny,Q_Nx,Q_Ny)``
    """

    def __init__(self, data, **kwargs):
        """
        Instantiate a DataCube object. Set the data and scan dimensions.
        """
        # Initialize DataObject
        DataObject.__init__(self, **kwargs)
        self.data = data   #: the 4D dataset

        # Set shape
        assert (len(data.shape)==3 or len(data.shape)==4)
        if len(data.shape)==3:
            self.R_N, self.Q_Nx, self.Q_Ny = data.shape
            self.R_Nx, self.R_Ny = self.R_N, 1
            self.set_scan_shape(self.R_Nx,self.R_Ny)
        else:
            self.R_Nx = data.shape[0]  #: real space x pixels
            self.R_Ny = data.shape[1]  #: real space y pixels
            self.Q_Nx = data.shape[2]  #: diffraction space x pixels
            self.Q_Ny = data.shape[3]  #: diffraction space y pixels
            self.R_N = self.R_Nx*self.R_Ny  #: total number of real space pixels
        self.update_slice_parsers()

        # Containers
        self.diffractionslices = {}
        self.realslices = {}
        self.braggpeaks = {}
        self.coordinates = Coordinates(self.R_Nx,self.R_Ny,self.Q_Nx,self.Q_Ny)
        def add_diffractionslice(self,x):
            x.coordinates = self.coordinates
            self.diffractionslices[x.name] = x
        def add_realslice(self,x):
            x.coordinates = self.coordinates
            self.realslices[x.name] = x
        def add_braggpeaks(self,x):
            x.coordinates = self.coordinates
            self.braggpeaks[x.name] = x

        # initialize params
        # bvm visualization
        self.bvm_vis_params = {}
        self.set_bvm_vis_params(cmap='jet',scaling='log')


        # TODO delete this
        # Set bragg calibration state flags
        self.bragg_calstate_uncalibrated = False
        self.bragg_calstate_origin = False
        self.bragg_calstate_ellipse = False
        self.bragg_calstate_dq = False
        self.bragg_calstate_rotflip = False




    ################## virtual diffraction ##################

    def get_dp(self,where=(0,0),name=None):
        """
        Returns a single diffraction pattern at the specified scan position.

        Args:
            where (2-tuple of ints): the (rx,ry) scan position
            name (str or None): if specified, stores this pattern
            in the datacube's dictionary of DiffractionSlices.

        Returns:
            (DiffractionSlice): the diffraction pattern
        """
        dp = DiffractionSlice(data=virtualimage.get_dp(self,where))
        if name is not None:
            dp.name = name
            self.diffractionslices[name] = dp
        return dp

    def get_max_dp(self,where=None,name='max_dp'):
        """
        Computes the maximal diffraction pattern.

        Args:
            where (None, or 4-tuple of ints, or boolean array): specifies which
                diffraction patterns to compute the max dp over.  If `where` is
                None, uses the whole datase. If it is a 4-tuple, uses a rectangular
                region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
                shape must match real space, and only True pixels are used.

        Returns:
            (DiffractionSlice):
        """
        dp_max = DiffractionSlice(
            data=virtualimage.get_max_dp(self,where=where),
            name=name)
        self.diffractionslices[name] = dp_max
        return dp_max

    def get_mean_dp(self,where=None,name='mean_dp'):
        """
        Computes the mean diffraction pattern.

        Args:
            where (None, or 4-tuple of ints, or boolean array): specifies which
                diffraction patterns to compute the mean dp over.  If `where` is
                None, uses the whole datase. If it is a 4-tuple, uses a rectangular
                region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
                shape must match real space, and only True pixels are used.

        Returns:
            (DiffractionSlice):
        """
        dp_mean = DiffractionSlice(
            data=virtualimage.get_mean_dp(self,where=where),
            name=name)
        self.diffractionslices[name] = dp_mean
        return dp_mean

    def get_median_dp(self,where=None,name='median_dp'):
        """
        Computes the median diffraction pattern.

        Args:
            where (None, or 4-tuple of ints, or boolean array): specifies which
                diffraction patterns to compute the median dp over.  If `where` is
                None, uses the whole datase. If it is a 4-tuple, uses a rectangular
                region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
                shape must match real space, and only True pixels are used.

        Returns:
            (DiffractionSlice):
        """
        dp_median = DiffractionSlice(
            data=virtualimage.get_median_dp(self,where=where),
            name=name)
        self.diffractionslices[name] = dp_median
        return dp_median

    def position_realspace_lims(self,geometry,im=None,alpha=0.25,**kwargs):
        """
        Display a rectangle overlaid on an image-like array,
        specified with the `im` argument.

        Args:
            geometry (4-tuple): the detector limits (x0,xf,y0,yf)
            im (variable): (type of im) image used for overlay
                - (None) the sum im, if it exists, else, the virtual image from
                  a point detector at the center pixel
                - (2-tuple) the virtual image from a point detector at (rx,ry)
                - (string) the attached realslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show_rectangles
        im = self._get_best_im(im)
        show_rectangles(im,geometry,alpha=alpha,**kwargs)

    def position_realspace_mask(self,mask,im=None,alpha=0.25,**kwargs):
        """
        Display a boolean mask overlaid on an image-like array,
        specified with the `im` argument.

        Args:
            mask (2d boolean array):
            im (variable): (type of im) image used for overlay
                - (None) the sum im, if it exists, else, the virtual image from
                  a point detector at the center pixel
                - (2-tuple) the virtual image from a point detector at (rx,ry)
                - (string) the attached realslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show
        im = self._get_best_im(im)
        show(im,mask=np.logical_not(mask),mask_color='r',mask_alpha=alpha,**kwargs)




    ################## virtual imaging ##################

    def get_im(self,geometry,detector='point',name=None):
        """
        Computes a virtual image.  Which image depends on the arguments `geometry` and
        `detector`.

        Args:
            geometry (variable): the type and meaning of this argument depend on the
                value of the `detector` argument:
                    - 'point': (2-tuple of ints) the (qx,qy) position of a point detector
                    - 'rect': (4-tuple) the corners (qx0,qxf,qy0,qyf)
                    - 'square': (2-tuple) ((qx0,qy0),s) where s is the sidelength and
                      (qx0,qy0) is the upper left corner
                    - 'circ': (2-tuple) (center,radius) where center=(qx0,qy0)
                    - 'ann': (2-tuple) (center,radii) where center=(qx0,qy0) and
                      radii=(ri,ro)
                    - 'mask': (2D boolean array)
            detector (str): the detector type. Must be one of: 'point','rect','square',
                'circ', 'ann', or 'mask'
            name (str or None): if specified, stores this image in the datacube's
                dictionary of RealSlices.

        Returns:
            (RealSlice): the virtual image
        """
        im = RealSlice(data=virtualimage.get_im(self,geometry,detector=detector))
        if name is not None:
            im.name = name
            self.realslices[name] = im
        return im

    def get_sum_im(self,name='sum'):
        """
        Computes the virtual image integrating over the entire detector.

        Returns:
            (RealSlice): the virtual image
        """
        geometry = (0,self.R_Nx,0,self.R_Ny)
        im = RealSlice(data=virtualimage.get_im(self,geometry,'rect'),
                       name=name)
        self.realslices[name] = im
        return im

    def capture_circular_detector(self,geometry,name):
        """
        Computes the virtual image from a circular detector.

        Args:
            geometry (2-tuple): (center,radius) where center=(qx0,qy0)
            name (str): label for this image

        Returns:
            (RealSlice): the virtual image
        """
        vi = RealSlice(data=virtualimage.get_virtualimage_circ(self,geometry),
                       name=name)
        self.realslices[name] = vi
        return vi

    def capture_rectangular_detector(self,geometry,name):
        """
        Computes the virtual image from a rectangular detector.

        Args:
            geometry (4-tuple): (qx0,qxf,qy0,qyf)
            name (str): label for this image

        Returns:
            (RealSlice): the virtual image
        """
        vi = DiffractionSlice(data=virtualimage.get_virtualimage_rect(self,geometry),
                              name=name)
        self.realslices[name] = vi
        return vi

    def capture_point_detector(self,geometry,name):
        """
        Computes the virtual image from a point detector.

        Args:
            geometry (2-tuple): (qx,qy)
            name (str): label for this image

        Returns:
            (RealSlice): the virtual image
        """
        vi = DiffractionSlice(data=virtualimage.get_virtualimage_point(self,geometry),
                              name=name)
        self.realslices[name] = vi
        return vi

    def capture_annular_detector(self,geometry,name):
        """
        Computes the virtual image from an annular detector.

        Args:
            geometry (2-tuple): (center,radii) where center=(qx0,qy0) and radii=(ri,ro)
            name (str): label for this image

        Returns:
            (RealSlice): the virtual image
        """
        vi = DiffractionSlice(data=virtualimage.get_virtualimage_ann(self,geometry),
                              name=name)
        self.realslices[name] = vi
        return vi

    def position_circular_detector(self,geometry,dp=None,alpha=0.25,**kwargs):
        """
        Display a circular detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            geometry (2-tuple): the geometry of the detector, given by (center,radius),
                where center is the 2-tuple (qx0,qy0), and radius is a number
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached diffractionslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show_circles
        center,radius = geometry
        dp = self._get_best_dp(dp)
        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show_circles(dp,center=center,R=radius,alpha=alpha,**kwargs)

    def position_rectangular_detector(self,geometry,dp=None,alpha=0.25,**kwargs):
        """
        Display a rectangular detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            geometry (4-tuple): the detector limits (x0,xf,y0,yf)
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached diffractionslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show_rectangles
        dp = self._get_best_dp(dp)
        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show_rectangles(dp,geometry,alpha=alpha,**kwargs)

    def position_annular_detector(self,geometry,dp=None,alpha=0.25,**kwargs):
        """
        Display an annular detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            geometry (2-tuple): the geometry of the detector, given by (center,radii),
                where center is the 2-tuple (qx0,qy0), and radii is the 2-tuple (ri,ro)
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached diffractionslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show
        center,radii = geometry
        dp = self._get_best_dp(dp)
        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show(dp,
             annulus={'center':center,'radii':radii,'fill':True,'alpha':0.3},
             **kwargs)

    def position_point_detector(self,geometry,dp=None,alpha=0.25,**kwargs):
        """
        Display a point detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            geometry (2-tuple): the detector position (qx0,qy0)
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached diffractionslice of this name, if it exists
            alpha (number): the transparency of the overlay
        """
        from ...visualize import show_rectangles
        lims = (geometry[0],geometry[0]+1,geometry[1],geometry[1]+1)
        dp = self._get_best_dp(dp)
        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show_rectangles(dp,lims,alpha=alpha,**kwargs)




    ################## probe #################

    def add_probe_image(self,image):
        """
        Attach an image of the electron probe over vacuum to the data

        Args:
            image (2d array):
        """
        assert(image.shape == (self.Q_Nx,self.Q_Ny)), "probe shape must match the datacube's diffraction space shape"
        self.diffractionslices['probe_image'] = DiffractionSlice(
            data=image, name='probe_image')

    def get_probe_size(self, **kwargs):
        """
        Measures the convergence angle from a probe image in pixels.
        See process.calibration.origin.get_probe_size for more info.
        """
        from ...process.calibration.origin import get_probe_size
        assert('probe_image' in self.diffractionslices.keys()), "First add a probe image!"
        qr,qx0,qy0 = get_probe_size(self.diffractionslices['probe_image'].data,**kwargs)
        self.coordinates.set_alpha_pix(qr)
        self.coordinates.set_probe_center((qx0,qy0))

    def show_probe_size(self):
        """
        Show the measure size of the vacuum probe
        """
        from ...visualize import show_circles
        assert('probe_image' in self.diffractionslices.keys())
        show_circles(self.diffractionslices['probe_image'].data,
                     self.coordinates.probe_center,
                     self.coordinates.alpha_pix)

    def get_probe_kernel(self,method='sigmoid',**kwargs):
        """
        Compute a probe kernel from the probe image.

        Args:
            method (str): determines how the kernel is constructed. For
            details of each method, see the
            process.diskdetection.probe.get_probe_* fn docs.

            Methods:
                - 'none': normalizes, shifts center
                - 'gaussian': normalizes, shifts center, subtracts a gaussian
                - 'sigmoid' normalizes, shifts center, makes sigmoid trench
        """
        assert(method in ('none','gaussian','sigmoid'))
        assert('probe_image' in self.diffractionslices.keys())

        # get data
        probe = self.diffractionslices['probe_image'].data
        try:
            center = self.coordinates.probe_center
            alpha = self.coordinates.alpha_pix
        except AttributeError:
            from ...process.calibration.origin import get_probe_size
            alpha,qx0,qy0 = get_probe_size(probe)
        if 'origin' not in kwargs.keys():
            try:
                kwargs['origin'] = center
            except NameError:
                kwargs['origin'] = (qx0,qy0)

        # make the probe kernel
        if method == 'none':
            from ...process.diskdetection.probe import get_probe_kernel
            probe_kernel = get_probe_kernel(probe,**kwargs)

        if method == 'gaussian':
            from ...process.diskdetection.probe import get_probe_kernel_edge_gaussian
            if 'sigma' not in kwargs.keys():
                kwargs['sigma'] = alpha * 3.2                                       # discuss
            probe_kernel = get_probe_kernel_edge_gaussian(probe,**kwargs)

        if method == 'sigmoid':
            from ...process.diskdetection.probe import get_probe_kernel_edge_sigmoid
            if 'ri' not in kwargs.keys() or 'ro' not in kwargs.keys():
                kwargs['ri'] = alpha
                kwargs['ro'] = alpha * 2                                            # discuss
            probe_kernel = get_probe_kernel_edge_sigmoid(probe,**kwargs)

        # save
        self.diffractionslices['probe_kernel'] = DiffractionSlice(data=probe_kernel,
                                                                  name='probe_kernel')

    def show_probe_kernel(self,R=None,L=None,W=None):
        """
        Visualize the probe kernel.

        Args:
            R (int): side length of displayed image, in pixels
            L (int): the line profile length
            W (int): the line profile integration window width
        """
        from ...visualize import show_kernel
        assert('probe_kernel' in self.diffractionslices.keys())
        if R is None:
            try:
                R = int( 5*self.coordinates.alpha_pix )
            except NameError:
                R = self.Q_Nx // 4
        if L is None:
            L = 2*R
        if W is None:
            W = 1
        show_kernel(self.diffractionslices['probe_kernel'].data,R,L,W)





    ############# disk detection ###############

    def show_some_DPs(self,positions,im=None,colors=None,
                          HW=None,figsize_im=(6,6),figsize_dp=(4,4),
                          **kwargs):
        """
        Shows two plots: a real space image with the pixels at `positions`
        highlighted, and a grid of the diffraction patterns from these
        scan positions.

        Args:
            positions (len N list or tuple of 2-tuples): the scan positions
            im (variable): (type of dp) image used for overlay
                - (None) the sum image, if it exists, else, a virtual image
                  from a point detector at the center pixel
                - (2-tuple) the point detector virtual image from (qx,qy)
                - (string) the attached realslice of this name, if it exists
            colors (len N list of colors or None):
            HW (2-tuple of ints): diffraction pattern grid shape
            figsize_im (2-tuple): size of the image figure
            figsize_dp (2-tuple): size of each diffraction pattern panel
            **kwargs (dict): arguments passed to visualize.show for the
                *diffraction patterns*. Default is `scaling='log'`
        """
        from ...visualize.vis_special import show_selected_dps
        im = self._get_best_im(im)
        show_selected_dps(self,positions,im=im,colors=colors,
                HW=HW,figsize_im=figsize_im,figsize_dp=figsize_dp,**kwargs)

    def find_some_bragg_disks(self,positions,**kwargs):
        """
        Finds the bragg disks in the DPs at `positions`. For more info,
        see process.diskdetection.find_Bragg_disks_selected.

        Args:
            positions ((Nx2) shaped)
        """
        from ...process.diskdetection import find_Bragg_disks_selected
        assert('probe_kernel' in self.diffractionslices.keys())
        N = len(positions)
        x = [i[0] for i in positions]
        y = [i[1] for i in positions]
        braggpeaks = find_Bragg_disks_selected(
            self,self.diffractionslices['probe_kernel'].data,Rx=x,Ry=y,**kwargs)
        self.braggpeaks['_some_braggpeaks'] = {
            'peaks':braggpeaks,'positions':positions
        }

    def show_some_bragg_disks(self,im=None,colors=None,HW=None,figsize_dp=(4,4),
                              **kwargs):
        """
        Shows the positions of Bragg disks detected with
        self.find_some_bragg_disks.

        Args:
            colors (list of colors or None):
            HW (2-tuple of ints): diffraction pattern grid shape
            figsize_dp (2-tuple): size of each diffraction pattern panel
            **kwargs (dict): arguments passed to visualize.show for the
                *diffraction patterns*. Default is `scaling='log'`
        """
        from ...visualize.vis_special import show_selected_dps
        assert('_some_braggpeaks' in self.braggpeaks.keys()), "First run find_some_bragg_disks!"
        braggpeaks = self.braggpeaks['_some_braggpeaks']['peaks']
        positions = self.braggpeaks['_some_braggpeaks']['positions']
        im = self._get_best_im(im)
        show_selected_dps(self,positions,im=im,bragg_pos=braggpeaks,
                          colors=colors,HW=HW,figsize_dp=figsize_dp,**kwargs)

    def find_bragg_disks(self,name='braggpeaks',**disk_detec_params):
        """

        """
        from ...process.diskdetection import find_Bragg_disks
        assert('probe_kernel' in self.diffractionslices.keys())
        peaks = find_Bragg_disks(self,self.diffractionslices['probe_kernel'].data,
                                 **disk_detec_params)
        peaks.name = name
        self.braggpeaks[name] = {
            'raw':peaks,
        }




    ############## bragg vector maps ################

    def set_bvm_vis_params(self,**kwargs):
        self.bvm_vis_params = kwargs

    def show_bvm(self,name='bvm',**vis_params):
        """

        Args:
            name (str): which bvm to show. Passing 'bvm' shows the
                most calibrated bvm. Other options refer to which
                calibrations have been performed, and include
                'uncalibrated', 'origin','ellipse','dq','rotflip'
        """
        from ...visualize import show
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)






















###lksjdfsdlakjfdslkjfsdl###

    def calibrate_bragg_origins(self):
        """

        """
        from ...process.calibration import center_braggpeaks
        assert('uncalibrated' in self.braggpeaks.keys())
        assert(self.bragg_calstate_uncalibrated)
        assert(not self.bragg_calstate_origin)
        peaks = center_braggpeaks(self.braggpeaks['uncalibrated'],
                                  coords=self.coordinates)
        self.braggpeaks['origin'] = peaks
        self.bragg_calstate_origin = True

    def calibrate_bragg_elliptical_distortions(self,p_ellipse):
        """

        """
        from ...process.calibration import correct_braggpeak_elliptical_distortions
        assert('origin' in self.braggpeaks.keys())
        assert(self.bragg_calstate_origin)
        assert(not self.bragg_calstate_ellipse)
        peaks = correct_braggpeak_elliptical_distortions(
            self.braggpeaks['origin'],p_ellipse)
        self.braggpeaks['ellipse'] = peaks
        _,_,a,b,theta = p_ellipse
        self.coordinates.set_ellipse(a,b,theta)
        self.bragg_calstate_ellipse = True

    def calibrate_bragg_dq(self):
        """

        """
        from ...process.calibration import calibrate_Bragg_peaks_pixel_size
        assert('ellipse' in self.braggpeaks.keys())
        assert(self.bragg_calstate_ellipse)
        assert(not self.bragg_calstate_dq)
        peaks = calibrate_Bragg_peaks_pixel_size(
            self.braggpeaks['ellipse'],coords=self.coordinates)
        self.braggpeaks['dq'] = peaks
        self.bragg_calstate_dq = True

    def calibrate_bragg_rotation(self):
        """

        """
        from ...process.calibration import calibrate_Bragg_peaks_rotation
        assert('dq' in self.braggpeaks.keys())
        assert(self.bragg_calstate_dq)
        assert(not self.bragg_calstate_rotflip)
        #self.coordinates.set_QR_rotation(rot)
        #self.coordinates.set_QR_flip(flip)
        peaks = calibrate_Bragg_peaks_rotation(
            self.braggpeaks['dq'],coords=self.coordinates)
        self.braggpeaks['rotflip'] = peaks
        self.bragg_calstate_rotflip = True

    def fit_bvm_radial_peak(self,lims,name='bvm',ymax=None):
        """

        """
        from ...process.fit import fit_1D_gaussian,gaussian
        from ...visualize import show_qprofile
        assert(len(lims)==2)
        assert(name in self.diffractionslices.keys())
        assert(name[:3]=='bvm')
        name_profile = 'radial_integral_'+name
        assert(name_profile in self.pointlists.keys())
        profile = self.pointlists[name_profile]

        q,I = profile.data['q'],profile.data['I']
        A,mu,sigma = fit_1D_gaussian(q,I,lims[0],lims[1])

        if ymax is None:
            n = len(profile.data)
            ymax = np.max(profile.data['I'][n//4:]) * 1.2
        fig,ax = show_qprofile(q,I,ymax=ymax,returnfig=True)
        ax.vlines(lims,0,ax.get_ylim()[1],color='r')
        ax.vlines(mu,0,ax.get_ylim()[1],color='g')
        ax.plot(q,gaussian(q,A,mu,sigma),color='r')

        return mu





    ############# calibration #################

    def measure_origin(self, **kwargs):
        """
        Measure the position of the origin for data with no beamstop,
        and for which the center beam has the highest intensity. If
        the maximal diffraction pattern hasn't been computed, this
        function computes it. See process.calibration.origin.get_origin
        for more info.
        """
        from ...process.calibration.origin import get_origin
        if 'max_dp' not in self.diffractionslices.keys():
            self.get_max_dp()
        kwargs['dp_max'] = self.diffractionslices['max_dp'].data
        qx0,qy0 = get_origin(self, **kwargs)
        self.coordinates.set_origin_meas(qx0,qy0)

    def fit_origin(self, **kwargs):
        """
        Performs a fit to the measured origin positions. See
        process.calibration.origin.fit_origin for more info.
        """
        from ...process.calibration.origin import fit_origin
        try:
            origin_meas = self.coordinates.get_origin_meas()
        except AttributeError or KeyError:
            raise Exception('First run measure_origin!')
        qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = fit_origin(origin_meas, **kwargs)
        self.coordinates.set_origin(qx0_fit,qy0_fit)
        self.coordinates.set_origin_residuals(qx0_residuals,qy0_residuals)

    def fit_elliptical_distortions_bragg(self,fitradii,name='bvm_origin'):
        """
        Fits the elliptical distortions using an annular ragion
        of a bragg vector map
        """
        from ...process.calibration import fit_ellipse_1D
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        p_ellipse = fit_ellipse_1D(bvm,(bvm.shape[0]/2,bvm.shape[1]/2),fitradii)
        return p_ellipse

    def get_bvm_radial_integral(self,name='bvm',dq=0.25):
        """

        """
        from ...process.utils import radial_integral
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        assert(name[:3]=='bvm')
        name_profile = 'radial_integral_'+name
        q,I = radial_integral(bvm,self.Q_Nx/2,self.Q_Ny/2,dr=dq)
        N = len(q)
        coords = [('q',float),('I',float)]
        data = np.zeros(N,coords)
        data['q'] = q
        data['I'] = I
        radial_integral = PointList(coordinates=coords,data=data,
                                    name=name_profile)
        self.pointlists[name_profile] = radial_integral

    def calibrate_dq(self,method='single_measurement',**kwargs):
        """

        """
        methods = ['single_measurement',
                   ]
        assert(method in methods)
        if method == 'single_measurement':
            assert(all([x in kwargs.keys() for x in (
                   'q_pix','q_known','units')]))
            qp,qk,u = kwargs['q_pix'],kwargs['q_known'],kwargs['units']
            dq = 1. / ( qp * qk )
            self.coordinates.set_Q_pixel_size(dq)
            self.coordinates.set_Q_pixel_units(u+'^-1')

    def calibrate_rotation(self,theta,flip):
        """

        Args:
            theta (number): in radians
            flip (bool):
        """
        self.coordinates.set_QR_rotation(theta)
        self.coordinates.set_QR_flip(flip)

    def show_origin_meas(self):
        """
        Show the measured origin positions
        """
        from ...visualize import show_origin_meas
        show_origin_meas(self)

    def show_origin_fit(self):
        """
        Show the fit origin positions
        """
        from ...visualize import show_origin_fit
        show_origin_fit(self)






    ############## visualize ###################

    def show(self, dp=None, **kwargs):
        """
        Show a single *diffraction shaped* 2D slice of the dataset,
        passing any **kwargs to the visualize.show function.
        Default scaling is 'log'.

        Args:
            dp (None or 2-tuple or str): Specifies the data to show.
            Behavior depends on the argument type:
                - (None) the maximal dp, if it exists. else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached DiffractionSlice of this name, if it exists
        """
        from ...visualize import show
        if dp is None:
            try:
                dp = self.diffractionslices['max_dp'].data
                title = 'max_dp'
            except KeyError:
                dp = virtualimage.get_dp(self,(0,0))
                title = 'dp 0,0'
        elif isinstance(dp,(tuple,list)):
            assert(len(dp)==2)
            title = 'dp {},{}'.format(dp[0],dp[1])
            dp = virtualimage.get_dp(self,dp)
        elif isinstance(dp,str):
            try:
                title = dp
                dp = self.diffractionslices[dp].data
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(dp))
        else:
            raise Exception("Invalid type, {}".format(type(dp)))

        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        if 'title' not in kwargs:
            kwargs['title'] = title
        show(dp,**kwargs)

    def show_im(self, im=None, **kwargs):
        """
        Show a single *real shaped* 2D slice of the dataset,
        passing any **kwargs to the visualize.show function.

        Args:
            im (None or 2-tuple or str): Specifies the data to show.
            Behavior depends on the argument type:
                - (None) the sum image, if it exists. else, the virtual image
                  from a point detector at the center of the detector
                - (2-tuple) the virtual image from a point detector at (qx,qy)
                - (string) the attached RealSlice of this name, if it exists
        """
        from ...visualize import show
        if im is None:
            try:
                im = self.realslices['sum_im'].data
                title = 'sum_im'
            except KeyError:
                qx,qy = self.Q_Nx//2,self.Q_Ny//2
                im = virtualimage.get_im(self,
                                         geometry=(qx,qy),
                                         detector='point')
                title = 'im {},{}'.format(qx,qy)
        elif isinstance(im,(tuple,list)):
            assert(len(im)==2)
            title = 'im {},{}'.format(im[0],im[1])
            im = virtualimage.get_im(self,im,'point')
        elif isinstance(im,str):
            try:
                title = im
                im = self.realslices[im].data
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(im))
        else:
            raise Exception("Invalid type, {}".format(type(im)))

        if 'title' not in kwargs:
            kwargs['title'] = title
        show(im,**kwargs)

    def _get_best_dp(self,dp=None):
        if dp is None:
            try:
                dp = self.diffractionslices['max_dp'].data
            except KeyError:
                dp = self.data[0,0,:,:]
        elif isinstance(dp,tuple):
            assert(len(dp)==2)
            dp = self.data[dp[0],dp[1],:,:]
        elif isinstance(dp,str):
            try:
                dp = self.diffractionslices[dp].data
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(dp))

    def _get_best_im(self,im=None):
        if im is None:
            try:
                im = self.realslices['sum'].data
            except KeyError:
                im = self.data[:,:,self.Q_Nx//2,self.Q_Ny//2]
        elif isinstance(im,tuple):
            assert(len(im)==2)
            im = self.data[:,:,im[0],im[1]]
        elif isinstance(im,str):
            try:
                im = self.realslices[im].data
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(im))
        return im




    def bvm_fit_select_radii(self,radii,name='bvm',**vis_params):
        """

        Args:
            radii (2-tuple):
            name (str): which bvm to show. Passing 'bvm' shows the
                most calibrated bvm. Other options refer to which
                calibrations have been performed, and include
                'uncalibrated', 'origin','ellipse','dq','rotflip'
        """
        from ...visualize import show
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,
             annulus={'center':(bvm.shape[0]/2,bvm.shape[1]/2),
                      'radii':radii,'fill':True,
                      'alpha':0.3,'color':'y'},
             **vis_params)

    def show_elliptical_fit_bragg(self,radii,p_ellipse,name='bvm',**vis_params):
        """

        Args:
            radii (2-tuple):
            name (str): which bvm to show. Passing 'bvm' shows the
                most calibrated bvm. Other options refer to which
                calibrations have been performed, and include
                'uncalibrated', 'origin','ellipse','dq','rotflip'
        """
        from ...visualize import show
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        center = bvm.shape[0]/2,bvm.shape[1]/2
        _,_,a,b,theta = p_ellipse
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,
             annulus={'center':center,'radii':radii,'fill':True,
                      'alpha':0.2,'color':'y'},
             ellipse={'center':center,'a':a,'b':b,'theta':theta,
                       'color':'r','alpha':0.7,'linewidth':2},
             **vis_params)

    def show_bvm_radial_integral(self,name='bvm',ymax=None,q_ref=None,
                                 returnfig=False):
        """

        Args:
            name (str): which bvm to show. Passing 'bvm' shows the
                most calibrated bvm. Other options refer to which
                calibrations have been performed, and include
                'uncalibrated', 'origin','ellipse','dq','rotflip'
        """
        from ...visualize import show_qprofile
        assert(name in self.diffractionslices.keys())
        bvm = self.diffractionslices[name].data
        assert(name[:3]=='bvm')
        name_profile = 'radial_integral_'+name
        assert(name_profile in self.pointlists.keys())
        profile = self.pointlists[name_profile]
        dq = self.coordinates.get_Q_pixel_size()
        units = self.coordinates.get_Q_pixel_units()
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







    ############## diskdetection.py ############


    ############ braggvectormap.py #############

    def get_bvm(self,peaks='braggpeaks',calibrated=True,name='bvm'):
        """

        """
        from ...process.diskdetection import get_bvm
        assert(peaks in self.braggpeaks.keys())
        peaks = self.braggpeaks[peaks]
        if calibrated:
            peaks = peaks['cal']
        else:
            peaks = peaks['raw']
        bvm = DiffractionSlice(
            data=get_bvm(peaks,self.Q_Nx,self.Q_Ny),
            name=name)
        self.diffractionslices[name] = bvm






    ############### preprocess.py ##############

    def set_scan_shape(self,R_Nx,R_Ny):
        """
        Reshape the data given the real space scan shape.

        Args:
            R_Nx,R_Ny (int): the scan shape
        """
        self = preprocess.set_scan_shape(self,R_Nx,R_Ny)
        self.update_slice_parsers()
        self.coordinates.R_Nx = self.R_Nx
        self.coordinates.R_Ny = self.R_Ny

    def swap_RQ(self):
        """
        Swap real and reciprocal space coordinates.
        """
        self = preprocess.swap_RQ(self)
        self.update_slice_parsers()
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny
        self.coordinates.R_Nx = self.R_Nx
        self.coordinates.R_Ny = self.R_Ny

    def swap_Rxy(self):
        """
        Swap real space x and y coordinates.
        """
        self = preprocess.swap_Rxy(self)
        self.update_slice_parsers()
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny
        self.coordinates.R_Nx = self.R_Nx
        self.coordinates.R_Ny = self.R_Ny

    def swap_Qxy(self):
        """
        Swap reciprocal space x and y coordinates.
        """
        self = preprocess.swap_Qxy(self)
        Q_Nx,Q_Ny = self.coordinates.Q_Nx,self.coordinates.Q_Ny
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny

    def crop_data_diffraction(self,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max):
        self = preprocess.crop_data_diffraction(self,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max)
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny

    def crop_data_real(self,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max):
        self = preprocess.crop_data_real(self,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max)
        self.coordinates.R_Nx = self.R_Nx
        self.coordinates.R_Ny = self.R_Ny

    def bin_data_diffraction(self, bin_factor):
        self = preprocess.bin_data_diffraction(self, bin_factor)
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny

    def bin_data_mmap(self, bin_factor, dtype=np.float32):
        self = preprocess.bin_data_mmap(self, bin_factor, dtype=dtype)
        self.coordinates.Q_Nx = self.Q_Nx
        self.coordinates.Q_Ny = self.Q_Ny

    def bin_data_real(self, bin_factor):
        self = preprocess.bin_data_real(self, bin_factor)
        self.coordinates.R_Nx = self.R_Nx
        self.coordinates.R_Ny = self.R_Ny












    ################ Slice data #################

    def update_slice_parsers(self):
        # define index-sanitizing functions:
        self.normX = lambda x: np.maximum(0,np.minimum(self.R_Nx-1,x))
        self.normY = lambda x: np.maximum(0,np.minimum(self.R_Ny-1,x))

    def get_diffraction_space_view(self,Rx=0,Ry=0):
        """
        Returns the image in diffraction space, and a Bool indicating success or failure.
        """
        self.Rx,self.Ry = self.normX(Rx),self.normY(Ry)
        try:
            return self.data[self.Rx,self.Ry,:,:], 1
        except IndexError:
            return 0, 0
        except ValueError:
            return 0,0

    # Virtual images -- integrating

    def get_virtual_image_rect_integrate(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in integration
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_rect_integrate(self,slice_x,slice_y)

    def get_virtual_image_circ_integrate(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in integration
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_circ_integrate(self,slice_x,slice_y)

    def get_virtual_image_annular_integrate(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in integration
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virutalimage_viewer.get_virtual_image_annular_integrate(self,slice_x,slice_y,R)

    # Virtual images -- difference

    def get_virtual_image_rect_diffX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_rect_diffX(self,slice_x,slice_y)

    def get_virtual_image_rect_diffY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_rect_diffY(self,slice_x,slice_y)

    def get_virtual_image_circ_diffX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_circ_diffX(self,slice_x,slice_y)

    def get_virtual_image_circ_diffY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_circ_diffY(self,slice_x,slice_y)

    def get_virtual_image_annular_diffX(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virutalimage_viewer.get_virtual_image_annular_diffX(self,slice_x,slice_y,R)

    def get_virtual_image_annular_diffY(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virutalimage_viewer.get_virtual_image_annular_diffY(self,slice_x,slice_y,R)

    # Virtual images -- CoM

    def get_virtual_image_rect_CoMX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_rect_CoMX(self,slice_x,slice_y)

    def get_virtual_image_rect_CoMY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_rect_CoMY(self,slice_x,slice_y)

    def get_virtual_image_circ_CoMX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_circ_CoMX(self,slice_x,slice_y)

    def get_virtual_image_circ_CoMY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virutalimage_viewer.get_virtual_image_circ_CoMY(self,slice_x,slice_y)

    def get_virtual_image_annular_CoMX(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virutalimage_viewer.get_virtual_image_annular_CoMX(self,slice_x,slice_y,R)

    def get_virtual_image_annular_CoMY(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virutalimage_viewer.get_virtual_image_annular_CoMY(self,slice_x,slice_y,R)


### Read/Write

def save_datacube_group(group, datacube, use_compression=False):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    group.attrs.create("emd_group_type",1)
    if (isinstance(datacube.data,np.ndarray) or isinstance(datacube.data,h5py.Dataset)):
        if use_compression:
            data_datacube = group.create_dataset("data", data=datacube.data,
                chunks=(1,1,datacube.Q_Nx,datacube.Q_Ny),compression='gzip')
        else:
            data_datacube = group.create_dataset("data", data=datacube.data)
    else:
        # handle K2DataArray datacubes
        data_datacube = datacube.data._write_to_hdf5(group)

    # Dimensions
    assert len(data_datacube.shape)==4, "Shape of datacube is {}".format(len(data_datacube))
    R_Nx,R_Ny,Q_Nx,Q_Ny = data_datacube.shape
    data_R_Nx = group.create_dataset("dim1",(R_Nx,))
    data_R_Ny = group.create_dataset("dim2",(R_Ny,))
    data_Q_Nx = group.create_dataset("dim3",(Q_Nx,))
    data_Q_Ny = group.create_dataset("dim4",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))
    data_Q_Nx[...] = np.arange(0,Q_Nx)
    data_Q_Nx.attrs.create("name",np.string_("Q_x"))
    data_Q_Nx.attrs.create("units",np.string_("[pix]"))
    data_Q_Ny[...] = np.arange(0,Q_Ny)
    data_Q_Ny.attrs.create("name",np.string_("Q_y"))
    data_Q_Ny.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present


def get_datacube_from_grp(g,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts an h5py Group corresponding to a single datacube in an open, correctly formatted H5 file,
        and returns a DataCube.
    """
    # TODO: add binning
    assert binfactor == 1, "Bin on load is currently unsupported for EMD files."

    if (mem, binfactor) == ("RAM", 1):
        data = np.array(g['data'])
    elif (mem, binfactor) == ("MEMMAP", 1):
        data = g['data']
    name = g.name.split('/')[-1]
    return DataCube(data=data,name=name)




