import numpy as np
from py4DSTEM.classes import DataCube




class PolarDatacube:

    """
    An interface to a 4D-STEM datacube under polar-elliptical transformation.
    """

    def __init__(
        self,
        datacube,
        qmin = 0.0,
        qmax = None,
        qstep = 1.0,
        n_annular = 180,
        qscale = None,
        mask = None,
        mask_thresh = 0.8,
        ellipse = True,
        friedel = False,
        ):
        """
        Parameters
        ----------
        datacube : DataCube
            The datacube in cartesian coordinates
        qmin : number
            Minumum radius of the polar transformation
        qmax : number or None
            Maximum radius of the polar transformation
        qstep : number
            Width of radial bins
        n_annular : integer
            Number of bins in the annular direction. Bins will each
            have a width of 360/num_annular_bins, in degrees
        qscale : number or None
            Radial scaling power to apply to polar transform
        mask : boolean array
            Cartesian space shaped mask to apply to all transforms
        mask_thresh : number
            Pixels below this value in the transformed mask are considered
            masked pixels
        ellipse : bool
            Setting to False forces a circular transform. Setting to True
            performs an elliptic transform iff elliptic calibrations are
            available.
        friedel : bool
            Setting to True computes the transform mod(pi), i.e. assumes
            Friedel symmetry is obeyed
        """

        # attach datacube
        assert(isinstance(datacube,DataCube))
        self._datacube = datacube

        # check for calibrations
        assert(hasattr(self._datacube,'calibration')), "No .calibration found"
        self.calibration = self._datacube.calibration

        # setup data getter
        self._set_polar_data_getter()


        # setup sampling

        # polar
        self._qscale = qscale
        if qmax is None:
            qmax = np.min(self._datacube.Qshape) / np.sqrt(2)
        self.set_annular_bins(n_annular)
        self.set_radial_bins(qmin,qmax,qstep)

        # cartesian
        self._xa,self._ya = np.meshgrid(
            np.arange(self._datacube.Q_Nx),
            np.arange(self._datacube.Q_Ny),
            indexing = 'ij'
        )

        # ellipse
        self.ellipse = ellipse

        # mask
        self._mask_thresh = mask_thresh
        self.mask = mask

        pass



    # sampling methods + properties

    def set_radial_bins(
        self,
        qmin,
        qmax,
        qstep,
    ):
        self._qmin = qmin
        self._qmax = qmax
        self._qstep = qstep

        self.radial_bins = np.arange(
            self._qmin,
            self._qmax,
            self._qstep
        )
        self.set_polar_shape()
        self.qscale = self._qscale

    @property
    def qmin(self):
        return self._qmin
    @qmin.setter
    def qmin(self, x):
        self.set_radial_bins(
            x,
            self._qmax,
            self._qstep
        )

    @property
    def qmax(self):
        return self._qmax
    @qmin.setter
    def qmax(self, x):
        self.set_radial_bins(
            self._qmin,
            x,
            self._qstep
        )

    @property
    def qstep(self):
        return self._qstep
    @qstep.setter
    def qstep(self, x):
        self.set_radial_bins(
            self._qmin,
            self._qmax,
            x
        )

    def set_annular_bins(
        self,
        n_annular
    ):
        self._n_annular = n_annular
        self._annular_bins = np.linspace(
            0,
            2*np.pi,
            self._n_annular,
            endpoint = False
        )
        self.set_polar_shape()
        self._annular_step = self.annular_bins[1] - self.annular_bins[0]

    @property
    def annular_bins(self):
        return self._annular_bins
    @property
    def annular_step(self):
        return self._annular_step

    @property
    def n_annular(self):
        return self._n_annular
    @n_annular.setter
    def n_annular(self, x):
        self.set_annular_bins(x)

    def set_polar_shape(self):
        if hasattr(self,'radial_bins') and hasattr(self,'annular_bins'):
            self.polar_shape = np.array((
                self.annular_bins.shape[0],
                self.radial_bins.shape[0]
            ))
            self.polar_size = np.prod(self.polar_shape)


    # scaling property
    @property
    def qscale(self):
        return self._qscale
    @qscale.setter
    def qscale(self,x):
        self._qscale = x
        if x is not None:
            self._qscale_ar = np.arange(self.polar_shape[1])**x


    # expose raw data
    @property
    def data_raw(self):
        return self._datacube


    # expose transformed data
    @property
    def data(self):
        return self._polar_data_getter

    def _set_polar_data_getter(self):
        self._polar_data_getter = PolarDataGetter(
            polarcube = self
        )


    # mask properties
    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self,x):
        if x is None:
            self._mask = x
        else:
            assert(x.shape == self._datacube.Qshape), "Mask shape must match diffraction space"
            self._mask = x
            self._mask_polar = self.transform(
                x
            )
    @property
    def mask_polar(self):
        return self._mask_polar
    @property
    def mask_thresh(self):
        return self._mask_thresh
    @mask_thresh.setter
    def mask_thresh(self,x):
        self._mask_thresh = x
        self.mask = self.mask


    # expose transformation
    @property
    def transform(self):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.

        Parameters
        ----------
        cartesian_data : array
            The data
        origin : tuple or list or None
            Variable behavior depending on the arg type. Length 2 tuples uses these
            values directly. Length 2 list of ints uses the calibrated origin value
            at this scan position. None uses the calibrated mean origin.
        mask : boolean array or None
            A mask applied to the data before transformation.  The value of masked
            pixels (0's) in the output is determined by `returnval`. Note that this
            mask is applied in combination with any mask at PolarData.mask.
        returnval : 'masked' or 'nan' or None
            Controls the returned data. 'masked' returns a numpy masked array. 'nan'
            returns a normal numpy array with masked pixels set to np.nan.  None
            returns a 2-tuple of numpy arrays - the transformed data with masked
            pixels set to 0, and the transformed mask.
        """
        return self._polar_data_getter._transform


    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += "Retrieves diffraction images in polar coordinates, using .data[x,y]"
        return string






class PolarDataGetter:

    def __init__(
        self,
        polarcube,
    ):
        self._polarcube = polarcube


    def __getitem__(self,pos):

        # unpack scan position
        x,y = pos
        # get the data
        cartesian_data = self._polarcube._datacube[x,y]
        # transform
        ans = self._transform(
            cartesian_data,
            origin = [x,y],
            returnval = 'masked'
        )
        # return
        return ans


    def _transform(
        self,
        cartesian_data,
        origin = None,
        ellipse = None,
        mask = None,
        mask_thresh = None,
        returnval = 'masked',
        ):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.

        Parameters
        ----------
        cartesian_data : array
            The data
        origin : tuple or list or None
            Variable behavior depending on the arg type. Length 2 tuples uses these
            values directly. Length 2 list of ints uses the calibrated origin value
            at this scan position. None uses the calibrated mean origin.
        ellipse : tuple or None
            Variable behavior depending on the arg type. Length 3 tuples uses these
            values directly (a,b,theta). None uses the calibrated value.
        mask : boolean array or None
            A mask applied to the data before transformation.  The value of masked
            pixels in the output is determined by `returnval`. Note that this
            mask is applied in combination with any mask at PolarData.mask.
        mask_thresh : number
            Pixels in the transformed mask with values below this number are
            considered masked, and will be populated by the values specified
            by `returnval`.
        returnval : 'masked' or 'nan' or None
            Controls the returned data. 'masked' returns a numpy masked array. 'nan'
            returns a normal numpy array with masked pixels set to np.nan.  None
            returns a 2-tuple of numpy arrays - the transformed data with masked
            pixels set to 0, and the transformed mask.
        """
        # get calibrations
        if origin is None:
            origin = self._polarcube.calibration.get_origin_mean()
        elif isinstance(origin,list):
            origin = self._polarcube.calibration.get_origin(origin[0],origin[1])
        elif isinstance(origin,tuple):
            pass
        else:
            raise Exception(f"Invalid type for `origin`, {type(origin)}")

        if ellipse is None:
            ellipse = self._polarcube.calibration.get_ellipse()
        elif isinstance(ellipse,tuple):
            pass
        else:
            raise Exception(f"Invalid type for `ellipse`, {type(ellipse)}")


        # combine passed mask with default mask
        mask0 = self._polarcube.mask
        if mask is None and mask0 is None:
            mask = np.ones_like(cartesian_data)
        elif mask is None:
            mask = mask0
        elif mask0 is None:
            mask = mask
        else:
            mask = mask*mask0

        if mask_thresh is None:
            mask_thresh = self._polarcube.mask_thresh


        # transform data
        ans = self._transform_array(
            cartesian_data,
            origin,
            ellipse
        )

        # transform mask
        norm = self._transform_array(
            mask,
            origin,
            ellipse
        )
        mask = norm>mask_thresh

        # normalize
        out = np.empty_like(norm)
        out[:] = np.nan
        ans = np.divide(ans, norm, out=out, where=mask!=0)

        # scaling
        if self._polarcube.qscale is not None:
            ans *= self._polarcube._qscale_ar[np.newaxis,:]

        # return
        if returnval == 'masked':
            ans = np.ma.array(
                data = ans,
                mask = np.isnan(ans)
            )
            return ans
        elif returnval == 'nan':
            return ans
        elif returnval is None:
            ans[np.isnan(ans)] = 0
            return (ans,norm)
        else:
            raise Exception(f"Unexpected value {returnval} encountered for `returnval`")



    def _transform_array(
        self,
        data,
        origin,
        ellipse
        ):

        # set origin
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]


        # circular
        if (ellipse is None) or (self._polarcube.ellipse) is False:

            # get polar coords
            rr = np.sqrt(x**2 + y**2)
            tt = np.arctan2(y, x) - np.pi/2

        # elliptical
        else:

            # unpack ellipse
            a,b,theta = ellipse

            # transformation matrix (elliptic cartesian -> circular cartesian)
            A = (a/b)*np.cos(theta)
            B = - np.sin(theta)
            C = (a/b)*np.sin(theta)
            D = np.cos(theta)
            det = 1 / (A*D - B*C)

            # get circular cartesian coords
            xc =  x*D - y*B
            yc = -x*C + y*A

            # get polar coords
            rr = det * np.hypot(xc,yc)
            tt = np.arctan2(yc,xc) - np.pi/2


        # transform to bin sampling
        r_ind = (rr - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = tt / self._polarcube.annular_step

        # get integers and increments
        r_ind_floor = np.floor(r_ind).astype('int')
        t_ind_floor = np.floor(t_ind).astype('int')
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor

        # resample
        sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self._polarcube.polar_shape[1])
        im = np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = data[sub] * (1 - dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = data[sub] * (1 - dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self._polarcube.polar_shape[1]-1)
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = data[sub] * (    dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = data[sub] * (    dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )

        # reshape and return
        ans = np.reshape(im, self._polarcube.polar_shape)
        return ans




    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += "Retrieves the diffraction pattern at scan position (x,y) in polar coordinates when sliced with [x,y]."
        return string



def fit_amorphous_ring(
    im,
    center,
    radial_range,
    coefs = None,
    show_fit_mask = False,
    verbose = False,
    plot_result = True,
    figsize = (8,8),
    return_all_coefs = False,
    ):
    """
    Fit an amorphous halo with a two-sided Gaussian model, plus a background 
    Gaussian function.

    TODO - move these functions to another 

    Parameters
    --------
    im: np.array
        2D image array to perform fitting on
    center: np.array
        (x,y) center coordinates for fitting mask
    radial_range: np.array
        (radius_inner, radius_outer) radial range to perform fitting over
    show_fit_mask: bool
        Set to true to preview the fitting mask and initial guess for the ellipse params
    verbose: bool
        Print fit results
    plot_result: bool
        Plot the result of the fitting
    figsize: tuple, list, np.array (optional)
        Figure size for plots
    return_all_coefs: bool
        Set to True to return the 11 parameter fit, rather than the 5 parameter ellipse
    Returns
    --------
    params_ellipse: np.array
        5 parameter elliptic fit coefficients
    params_ellipse_fit: np.array (optional)
        11 parameter elliptic fit coefficients
    """

    # coordinates
    xa,ya = np.meshgrid(
        np.arange(im.shape[0]),
        np.arange(im.shape[1]),
        indexing = 'ij',
        )

    # Make fitting mask
    ra2 = (xa - center[0])**2 + (ya - center[1])**2
    mask = np.logical_and(
        ra2 >= radial_range[0]**2,
        ra2 <= radial_range[1]**2,
        )
    vals = im[mask]
    basis = np.vstack((xa[mask],ya[mask]))

    # initial fitting parameters
    if coefs is None:
        # ellipse parameters
        x0 = center[0]
        y0 = center[1]
        R_mean = np.mean(radial_range)
        # A = 1/R_mean**2
        # B = 0
        # C = 1/R_mean**2
        a = R_mean
        b = R_mean
        t = 0

        # Gaussian model parameters
        int_min = np.min(vals)
        int_max = np.max(vals)
        int0 = (int_max - int_min)/2
        int12 = (int_max - int_min)/2
        k_bg = int_min
        sigma0 = np.mean(radial_range)
        sigma1 = (radial_range[1] - radial_range[0])/4
        sigma2 = (radial_range[1] - radial_range[0])/4

        coefs = (
            x0,y0,
            a,b,t,
            int0,int12,k_bg,
            sigma0,sigma1,sigma2)
        lb = (
            0,0,
            radial_range[0],radial_range[0],-2*np.pi,
            0,0,0,
            1,1,1)
        ub = (
            im.shape[0],im.shape[1],
            radial_range[1],radial_range[1],2*np.pi,
            np.inf,np.inf,np.inf,
            np.inf,np.inf,np.inf)

    if show_fit_mask:
        # show image preview of fitting mask
        int_scale = (-3,3)

        # Generate hybrid image for plotting
        int_med = np.median(vals)
        int_std = np.sqrt(np.median((vals - int_med)**2))
        int_range = (
            int_med + int_scale[0]*int_std, 
            int_med + int_scale[1]*int_std)
        im_plot = np.tile(np.clip(
            (im[:,:,None] - int_range[0]) / (int_range[1] - int_range[0]),
            0,1),(1,1,3))
        im_plot[:,:,0] *= 1-mask

        fig,ax = plt.subplots(figsize=figsize)
        ax.imshow(im_plot)

    else:
        # Perform elliptic fitting
        int_mean = np.mean(vals)
        coefs = curve_fit(
            amorphous_model, 
            basis, 
            vals / int_mean, 
            p0=coefs,
            xtol = 1e-12,
            bounds = (lb,ub),
        )[0]
        coefs[4] = np.mod(coefs[4],2*np.pi)
        coefs[5:8] *= int_mean
        # bounds=bounds

    if verbose:
        print('x0 = ' + str(np.round(coefs[0],3)) + ' px')
        print('y0 = ' + str(np.round(coefs[1],3)) + ' px')
        print('a  = ' + str(np.round(coefs[2],3)) + ' px')
        print('b  = ' + str(np.round(coefs[3],3)) + ' px')
        print('t  = ' + str(np.round(np.rad2deg(coefs[4]),3)) + ' deg')

    if plot_result:
        # Generate resulting best fit image
        im_fit = np.reshape(amorphous_model(
            np.vstack((xa.ravel(),ya.ravel())),
            coefs),im.shape)

        # plots
        phi = np.linspace(0,2*np.pi,360)
        cp = np.cos(phi)
        sp = np.sin(phi)

        fig,ax = plt.subplots(figsize=figsize)
        ax.imshow(
            im,
            vmin = 0,
            vmax = np.max(im_fit[mask]),
            cmap = 'gray',
            )

        x0 = coefs[0]
        y0 = coefs[1]
        a = coefs[2]
        b = coefs[3]
        t = coefs[4]
        s1 = coefs[9]
        s2 = coefs[10]

        ax.plot(
            y0 + np.array((-1,1))*a*np.sin(t),
            x0 + np.array((-1,1))*a*np.cos(t),
            c = 'r',
            ) 
        ax.plot(
            y0 + np.array((-1,1))*b*np.cos(t),
            x0 + np.array((1,-1))*b*np.sin(t),
            c = 'r',
            linestyle = 'dashed',
            )        

        ax.plot(
            y0 + a*np.sin(t)*cp + b*np.cos(t)*sp,
            x0 + a*np.cos(t)*cp - b*np.sin(t)*sp,
            c = 'r',
            )
        ax.plot(
            y0 + (a-s1)*np.sin(t)*cp + (b-s1*b/a)*np.cos(t)*sp,
            x0 + (a-s1)*np.cos(t)*cp - (b-s1*b/a)*np.sin(t)*sp,
            c = 'r',
            linestyle='dashed',
            )
        ax.plot(
            y0 + (a+s2)*np.sin(t)*cp + (b+s1*b/a)*np.cos(t)*sp,
            x0 + (a+s2)*np.cos(t)*cp - (b+s1*b/a)*np.sin(t)*sp,
            c = 'r',
            linestyle='dashed',
            )

    # Return fit parameters
    if return_all_coefs:
        return coefs
    else:
        return coefs[0:5]


def amorphous_model(basis, *coefs):
    coefs = np.squeeze(np.array(coefs))

    x0 = coefs[0]
    y0 = coefs[1]
    a = coefs[2]
    b = coefs[3]
    t = coefs[4]
    # A = coefs[2]
    # B = coefs[3]
    # C = coefs[4]
    int0 = coefs[5]
    int12 = coefs[6]
    k_bg = coefs[7]
    sigma0 = coefs[8]
    sigma1 = coefs[9]
    sigma2 = coefs[10]

    x0 = basis[0,:] - x0
    y0 = basis[1,:] - y0
    x = np.cos(t)*x0 - (b/a)*np.sin(t)*y0
    y = np.sin(t)*x0 + (b/a)*np.cos(t)*y0

    r2 = x**2 + y**2
    dr = np.sqrt(r2) - b
    dr2 = dr**2
    sub = dr < 0

    int_model = k_bg + \
        int0*np.exp(r2/(-2*sigma0**2))
    int_model[sub] += int12*np.exp(dr2[sub]/(-2*sigma1**2))
    sub = np.logical_not(sub)
    int_model[sub] += int12*np.exp(dr2[sub]/(-2*sigma2**2))

    return int_model

