# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and methods
# pointing to processing functions - generally defined in other files in the process directory.

from collections.abc import Sequence
from tempfile import TemporaryFile

import numpy as np
import numba as nb
import h5py

from .dataobject import DataObject
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

        # Containers for derivative data
        self.images = {} #: dict storing image-like 2D arrays derived from this data
        self.pointlists = {} #: dict storing pointlists and lists/tuples of pointlists
        self.bragg_positions = {} #: dict storing pointlistarrays of bragg disk positions
        self.coordinates = Coordinates(self.R_Nx,self.R_Ny,self.Q_Nx,self.Q_Ny)
            #: Coordinate instance storing calibration metadata

        # Set bragg calibration state flags
        self.bragg_calstate_uncalibrated = False
        self.bragg_calstate_origin = False
        self.bragg_calstate_ellipse = False
        self.bragg_calstate_dq = False
        self.bragg_calstate_rotflip = False

        # bvm visualization
        self.bvm_vis_params = {}
        self.set_bvm_vis_params(cmap='jet',scaling='log')





        # Set shape
        # TODO: look for shape in metadata
        # TODO: AND/OR look for R_Nx... in kwargs
        #self.R_Nx, self.R_Ny, self.Q_Nx, self.Q_Ny = self.data.shape
        #self.R_N = self.R_Nx*self.R_Ny
        #self.set_scan_shape(self.R_Nx,self.R_Ny)



    ########### Processing functions, organized by file in process directory ############



    ############## visualize ###################

    def show(self, im=None, **kwargs):
        """
        Show a single 2D slice of the dataset, passing any **kwargs to
        the visualize.show function. Default scaling is 'log'.

        Args:
            im (variable): (type of im) image dislayed
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (len 2 list) the point-detector virtual image from (qx,qy)
                - (string) the attached im of this name, if it exists
        """
        from ...visualize import show
        if im is None:
            try:
                im = self.images['dp_max']
            except KeyError:
                im = self.data[0,0,:,:]
        elif isinstance(im,tuple):
            assert(len(im)==2)
            im = self.data[im[0],im[1],:,:]
        elif isinstance(im,list):
            assert(len(im)==2)
            im = self.data[:,:,im[0],im[1]]
        elif isinstance(im,str):
            try:
                im = self.images[im]
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(im))
        else:
            raise Exception("Invalid type, {}".format(type(im)))

        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show(im,**kwargs)

    def position_circular_detector(self,center,radius,dp=None,alpha=0.25,**kwargs):
        """
        Display a circular detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached im of this name, if it exists
            center (2-tuple):
            radius (number):
        """
        from ...visualize import show_circles
        if dp is None:
            try:
                dp = self.images['dp_max']
            except KeyError:
                dp = self.data[0,0,:,:]
        elif isinstance(dp,tuple):
            assert(len(dp)==2)
            dp = self.data[dp[0],dp[1],:,:]
        elif isinstance(dp,str):
            try:
                dp = self.images[dp]
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(dp))

        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show_circles(dp,center=center,R=radius,alpha=alpha,**kwargs)

    def position_annular_detector(self,center,ri,ro,dp=None,alpha=0.25,**kwargs):
        """
        Display an annular detector overlaid on a diffraction-pattern-like array,
        specified with the `dp` argument.

        Args:
            dp (variable): (type of dp) image used for overlay
                - (None) the maximal dp, if it exists, else, the dp at (0,0)
                - (2-tuple) the diffraction pattern at (rx,ry)
                - (string) the attached im of this name, if it exists
            center (2-tuple):
            ri,ro: the inner and outer radii
        """
        from ...visualize import show
        if dp is None:
            try:
                dp = self.images['dp_max']
            except KeyError:
                dp = self.data[0,0,:,:]
        elif isinstance(dp,tuple):
            assert(len(dp)==2)
            dp = self.data[dp[0],dp[1],:,:]
        elif isinstance(dp,str):
            try:
                dp = self.images[dp]
            except KeyError:
                raise Exception("This datacube has no image called '{}'".format(dp))

        if 'scaling' not in kwargs:
            kwargs['scaling'] = 'log'
        show(dp,
             annulus={'center':center,'Ri':ri,'Ro':ro,'fill':True,'alpha':0.3},
             **kwargs)

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

    def show_probe_size(self):
        """
        Show the measure size of the vacuum probe
        """
        from ...visualize import show_circles
        assert('probe_image' in self.images.keys())
        show_circles(self.images['probe_image'],self.coordinates.probe_center,self.coordinates.alpha_pix)

    def show_probe_kernel(self,R=None,L=None,W=None):
        """
        Visualize the probe kernel.

        Args:
            R (int): side length of displayed image, in pixels
            L (int): the line profile length
            W (int): the line profile integration window width
        """
        from ...visualize import show_kernel
        assert('probe_kernel' in self.images.keys())
        if R is None:
            try:
                R = int( 5*self.coordinates.alpha_pix )
            except NameError:
                R = self.Q_Nx // 4
        if L is None:
            L = 2*R
        if W is None:
            W = 1
        show_kernel(self.images['probe_kernel'],R,L,W)

    def show_selected_dps(self,positions,im=None,colors=None,
                          HW=None,figsize_im=(6,6),figsize_dp=(4,4),
                          **kwargs):
        """
        Shows two plots: first, a real space image overlaid with colored dots
        at the specified positions; second, a grid of diffraction patterns
        corresponding to these scan positions.

        Args:
            positions (len N list or tuple of 2-tuples): the scan positions
            im (str or None): name of a real space image stored in datacube.
                Defaults to 'BF'.
            colors (len N list of colors or None):
            HW (2-tuple of ints): diffraction pattern grid shape
            figsize_im (2-tuple): size of the image figure
            figsize_dp (2-tuple): size of each diffraction pattern panel
            **kwargs (dict): arguments passed to visualize.show for the
                *diffraction patterns*. Default is `scaling='log'`
        """
        from ...visualize.vis_special import show_selected_dps
        show_selected_dps(self,positions,im=im,colors=colors,
                HW=HW,figsize_im=figsize_im,figsize_dp=figsize_dp,**kwargs)

    def show_some_bragg_disks(self,colors=None,HW=None,figsize_dp=(4,4),
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
        assert('bragg_positions_some_dps' in self.pointlists.keys()), "First run find_some_bragg_disks!"
        bragg_positions = self.pointlists['bragg_positions_some_dps']
        positions = self.pointlists['_bragg_positions_some_dps_positions']
        try:
            alpha = self.coordinates.alpha_pix
        except NameError:
            alpha = None
        show_selected_dps(self,positions,bragg_pos=bragg_positions,alpha=alpha,
                          colors=colors,HW=HW,figsize_dp=figsize_dp,**kwargs)

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
        assert(name in self.images.keys())
        bvm = self.images[name]
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,**vis_params)

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
        assert(name in self.images.keys())
        bvm = self.images[name]
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,
             annulus={'center':(bvm.shape[0]/2,bvm.shape[1]/2),
                      'Ri':radii[0],'Ro':radii[1],'fill':True,
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
        assert(name in self.images.keys())
        bvm = self.images[name]
        center = bvm.shape[0]/2,bvm.shape[1]/2
        _,_,a,b,theta = p_ellipse
        if len(vis_params)==0:
            vis_params = self.bvm_vis_params
        show(bvm,
             annulus={'center':center,'Ri':radii[0],'Ro':radii[1],'fill':True,
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
        assert(name in self.images.keys())
        bvm = self.images[name]
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


    ############## virtualimage.py #############

    def get_max_dp(self):
        """
        Computes the maximal diffraction pattern.
        """
        self.images['dp_max'] = virtualimage.get_max_dp(self)

    def capture_circular_detector(self,center,radius,name):
        """
        Computes the virtual image from a circular detector.

        Args:
            center (2-tuple): (x0,y0) of detector
            radius (number): radius of detector
            name (str): label for this image
        """
        self.images[name] = virtualimage.get_virtualimage_circ(
                                self,center[0],center[1],radius)



    ############## calibration ################

    def get_probe_size(self, **kwargs):
        """
        Measures the convergence angle from a probe image in pixels.
        See process.calibration.origin.get_probe_size for more info.
        """
        from ...process.calibration.origin import get_probe_size
        assert('probe_image' in self.images.keys()), "First add a probe image!"
        qr,qx0,qy0 = get_probe_size(self.images['probe_image'],**kwargs)
        self.coordinates.set_alpha_pix(qr)
        self.coordinates.set_probe_center((qx0,qy0))

    def measure_origin(self, **kwargs):
        """
        Measure the position of the origin for data with no beamstop,
        and for which the center beam has the highest intensity. If
        the maximal diffraction pattern hasn't been computed, this
        function computes it. See process.calibration.origin.get_origin
        for more info.
        """
        from ...process.calibration.origin import get_origin
        if 'dp_max' not in self.images.keys():
            self.get_max_dp()
        kwargs['dp_max'] = self.images['dp_max']
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
        assert(name in self.images.keys())
        bvm = self.images[name]
        p_ellipse = fit_ellipse_1D(bvm,(bvm.shape[0]/2,bvm.shape[1]/2),fitradii)
        return p_ellipse

    def get_bvm_radial_integral(self,name='bvm',dq=0.25):
        """

        """
        from ...process.utils import radial_integral
        assert(name in self.images.keys())
        bvm = self.images[name]
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





    ############## diskdetection.py ############

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
        assert('probe_image' in self.images.keys())

        # get data
        probe = self.images['probe_image']
        try:
            center = self.coordinates.probe_center
            alpha = self.coordinates.alpha_pix
        except AttributeError:
            from ...process.calibration.origin import get_probe_size
            assert('probe_image' in self.images.keys())
            alpha,qx0,qy0 = get_probe_size(self.images['probe_image'])
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
        self.images['probe_kernel'] = probe_kernel

    def find_some_bragg_disks(self,positions,**kwargs):
        """
        Finds the bragg disks in the DPs at `positions`. For more info,
        see process.diskdetection.find_Bragg_disks_selected.

        Args:
            positions ((Nx2) shaped)
        """
        from ...process.diskdetection import find_Bragg_disks_selected
        assert('probe_kernel' in self.images.keys())
        N = len(positions)
        x = [i[0] for i in positions]
        y = [i[1] for i in positions]
        bragg_positions = find_Bragg_disks_selected(
            self,self.images['probe_kernel'],Rx=x,Ry=y,**kwargs)
        self.pointlists['bragg_positions_some_dps'] = bragg_positions
        self.pointlists['_bragg_positions_some_dps_positions'] = positions

    def find_bragg_disks(self,**disk_detec_params):
        """

        """
        from ...process.diskdetection import find_Bragg_disks
        assert('probe_kernel' in self.images.keys())
        peaks = find_Bragg_disks(self,self.images['probe_kernel'],
                                 **disk_detec_params)
        self.bragg_positions['uncalibrated'] = peaks
        self.bragg_calstate_uncalibrated = True

        try:
            print('Calibrating origin positions')
            self.calibrate_bragg_origins()
        except AttributeError:
            print('Origin positions not found; skipping calibration')

    def calibrate_bragg_origins(self):
        """

        """
        from ...process.calibration import center_braggpeaks
        assert('uncalibrated' in self.bragg_positions.keys())
        assert(self.bragg_calstate_uncalibrated)
        assert(not self.bragg_calstate_origin)
        peaks = center_braggpeaks(self.bragg_positions['uncalibrated'],
                                  coords=self.coordinates)
        self.bragg_positions['origin'] = peaks
        self.bragg_calstate_origin = True

    def calibrate_bragg_elliptical_distortions(self,p_ellipse):
        """

        """
        from ...process.calibration import correct_braggpeak_elliptical_distortions
        assert('origin' in self.bragg_positions.keys())
        assert(self.bragg_calstate_origin)
        assert(not self.bragg_calstate_ellipse)
        peaks = correct_braggpeak_elliptical_distortions(
            self.bragg_positions['origin'],p_ellipse)
        self.bragg_positions['ellipse'] = peaks
        _,_,a,b,theta = p_ellipse
        self.coordinates.set_ellipse(a,b,theta)
        self.bragg_calstate_ellipse = True

    def calibrate_bragg_dq(self):
        """

        """
        from ...process.calibration import calibrate_Bragg_peaks_pixel_size
        assert('ellipse' in self.bragg_positions.keys())
        assert(self.bragg_calstate_ellipse)
        assert(not self.bragg_calstate_dq)
        peaks = calibrate_Bragg_peaks_pixel_size(
            self.bragg_positions['ellipse'],coords=self.coordinates)
        self.bragg_positions['dq'] = peaks
        self.bragg_calstate_dq = True

    def calibrate_bragg_rotation(self):
        """

        """
        from ...process.calibration import calibrate_Bragg_peaks_rotation
        assert('dq' in self.bragg_positions.keys())
        assert(self.bragg_calstate_dq)
        assert(not self.bragg_calstate_rotflip)
        #self.coordinates.set_QR_rotation(rot)
        #self.coordinates.set_QR_flip(flip)
        peaks = calibrate_Bragg_peaks_rotation(
            self.bragg_positions['dq'],coords=self.coordinates)
        self.bragg_positions['rotflip'] = peaks
        self.bragg_calstate_rotflip = True

    def fit_bvm_radial_peak(self,lims,name='bvm',ymax=None):
        """

        """
        from ...process.fit import fit_1D_gaussian,gaussian
        from ...visualize import show_qprofile
        assert(len(lims)==2)
        assert(name in self.images.keys())
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



    ############ braggvectormap.py #############

    def get_bvm(self,name='_best',overwrite=False):
        """

        """
        from ...process.diskdetection import get_bvm
        calstates = [self.bragg_calstate_uncalibrated,
                     self.bragg_calstate_origin,
                     self.bragg_calstate_ellipse,
                     self.bragg_calstate_dq,
                     self.bragg_calstate_rotflip]
        calstate = np.sum(calstates)
        assert(calstate!=0)
        d_calstate = {1:'uncalibrated',
                      2:'origin',
                      3:'ellipse',
                      4:'dq',
                      5:'rotflip'}
        name_best = d_calstate[calstate]
        if name == '_best':
            name = name_best
        else:
            assert(name in ('uncalibrated','origin','ellipse','dq','rotflip'))
        if name == name_best:
            set_bvm = True
        if 'bvm_'+name in self.images.keys():
            if not overwrite:
                raise Exception("This bvm has already been computed. To recompute and overwrite, pass `overwrite=True`")

        peaks = self.bragg_positions[name]
        bvm = get_bvm(peaks,self.Q_Nx,self.Q_Ny)
        self.images['bvm_'+name] = bvm
        if set_bvm:
            self.images['bvm'] = bvm






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






    ############ Miscellaneous methods #############

    def add_image(self,image,name):
        """
        Attach an image to the datacube

        Args:
            image (2d array):
            name (str): label for the data
        """
        assert(len(image.shape)==2)
        assert(isinstance(name,str))
        self.images[name] = image

    def add_probe_image(self,image):
        """
        Attach an image of the electron probe over vacuum to the data

        Args:
            image (2d array):
        """
        assert(image.shape == (self.Q_Nx,self.Q_Ny)), "probe shape must match the datacube's diffraction space shape"
        self.images['probe_image'] = image








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




