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
        self._datacube.polar = self

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


