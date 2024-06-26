import numpy as np
from py4DSTEM.datacube import DataCube
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter1d


class PolarDatacube:
    """
    An interface to a 4D-STEM datacube under polar-elliptical transformation.
    """

    def __init__(
        self,
        datacube,
        qmin=0.0,
        qmax=None,
        qstep=1.0,
        n_annular=180,
        qscale=None,
        mask=None,
        mask_thresh=0.1,
        ellipse=True,
        two_fold_symmetry=False,
    ):
        """
        Parameters
        ----------
        datacube : DataCube
            The datacube in cartesian coordinates
        qmin : number
            Minumum radius of the polar transformation, in pixels
        qmax : number or None
            Maximum radius of the polar transformation, in pixels
        qstep : number
            Width of radial bins, in pixels
        n_annular : integer
            Number of bins in the annular direction. Bins will each
            have a width of 360/n_annular, or 180/n_annular if
            two_fold_rotation is set to True, in degrees
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
        two_fold_rotation : bool
            Setting to True computes the transform mod(theta,pi), i.e. assumes
            all patterns possess two-fold rotation (Friedel symmetry).  The
            output angular range in this case becomes [0, pi) as opposed to the
            default of [0,2*pi).
        """

        # attach datacube
        assert isinstance(datacube, DataCube)
        self._datacube = datacube
        self._datacube.polar = self

        # check for calibrations
        assert hasattr(self._datacube, "calibration"), "No .calibration found"
        self.calibration = self._datacube.calibration

        # setup data getter
        self._set_polar_data_getter()

        # setup sampling

        # polar
        self._qscale = qscale
        if qmax is None:
            qmax = np.min(self._datacube.Qshape) / np.sqrt(2)
        self._n_annular = n_annular
        self.two_fold_symmetry = two_fold_symmetry  # implicitly calls set_annular_bins
        self.set_radial_bins(qmin, qmax, qstep)

        # cartesian
        self._xa, self._ya = np.meshgrid(
            np.arange(self._datacube.Q_Nx),
            np.arange(self._datacube.Q_Ny),
            indexing="ij",
        )

        # ellipse
        self.ellipse = ellipse

        # mask
        self._mask_thresh = mask_thresh
        self.mask = mask

        pass

    from py4DSTEM.process.polar.polar_analysis import (
        calculate_radial_statistics,
        calculate_pair_dist_function,
        calculate_FEM_local,
        calculate_annular_symmetry,
        plot_radial_mean,
        plot_radial_var_norm,
        plot_annular_symmetry,
        plot_background_fits,
        plot_sf_estimate,
        plot_reduced_pdf,
        plot_pdf,
        background_pca,
    )
    from py4DSTEM.process.polar.polar_peaks import (
        find_peaks_single_pattern,
        find_peaks,
        refine_peaks_local,
        refine_peaks,
        plot_radial_peaks,
        plot_radial_background,
        model_radial_background,
        make_orientation_histogram,
    )

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

        self.radial_bins = np.arange(self._qmin, self._qmax, self._qstep)
        self._radial_step = self._datacube.calibration.get_Q_pixel_size() * self._qstep
        self.set_polar_shape()
        self.qscale = self._qscale

    @property
    def qmin(self):
        return self._qmin

    @qmin.setter
    def qmin(self, x):
        self.set_radial_bins(x, self._qmax, self._qstep)

    @property
    def qmax(self):
        return self._qmax

    @qmin.setter
    def qmax(self, x):
        self.set_radial_bins(self._qmin, x, self._qstep)

    @property
    def qstep(self):
        return self._qstep

    @qstep.setter
    def qstep(self, x):
        self.set_radial_bins(self._qmin, self._qmax, x)

    def set_annular_bins(self, n_annular):
        self._n_annular = n_annular
        self._annular_bins = np.linspace(
            0, self._annular_range, self._n_annular, endpoint=False
        )
        self._annular_step = self.annular_bins[1] - self.annular_bins[0]
        self.set_polar_shape()

    @property
    def annular_bins(self):
        return self._annular_bins

    @property
    def annular_step(self):
        return self._annular_step

    @property
    def two_fold_symmetry(self):
        return self._two_fold_symmetry

    @two_fold_symmetry.setter
    def two_fold_symmetry(self, x):
        assert isinstance(
            x, bool
        ), f"two_fold_symmetry must be boolean, not type {type(x)}"
        self._two_fold_symmetry = x
        if x:
            self._annular_range = np.pi
        else:
            self._annular_range = 2 * np.pi
        self.set_annular_bins(self._n_annular)

    @property
    def n_annular(self):
        return self._n_annular

    @n_annular.setter
    def n_annular(self, x):
        self.set_annular_bins(x)

    def set_polar_shape(self):
        if hasattr(self, "radial_bins") and hasattr(self, "annular_bins"):
            # set shape
            self.polar_shape = np.array(
                (self.annular_bins.shape[0], self.radial_bins.shape[0])
            )
            self.polar_size = np.prod(self.polar_shape)
            # set KDE params
            self._annular_bin_step = 1 / (
                self._annular_step * (self.radial_bins + self.qstep * 0.5)
            )
            self._sigma_KDE = self._annular_bin_step * 0.5
            # set array indices
            self._annular_indices = np.arange(self.polar_shape[0]).astype(int)
            self._radial_indices = np.arange(self.polar_shape[1]).astype(int)

    # coordinate grid properties
    @property
    def tt(self):
        return self._annular_bins

    @property
    def tt_deg(self):
        return self._annular_bins * 180 / np.pi

    @property
    def qq(self):
        return self.radial_bins * self.calibration.get_Q_pixel_size()

    # scaling property
    @property
    def qscale(self):
        return self._qscale

    @qscale.setter
    def qscale(self, x):
        self._qscale = x
        if x is not None:
            self._qscale_ar = (self.qq / self.qq[-1]) ** x

    # expose raw data
    @property
    def data_raw(self):
        return self._datacube

    # expose transformed data
    @property
    def data(self):
        return self._polar_data_getter

    def _set_polar_data_getter(self):
        self._polar_data_getter = PolarDataGetter(polarcube=self)

    # mask properties
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, x):
        if x is None:
            self._mask = x
        else:
            assert (
                x.shape == self._datacube.Qshape
            ), "Mask shape must match diffraction space"
            self._mask = x
            self._mask_polar = self.transform(x)

    @property
    def mask_polar(self):
        return self._mask_polar

    @property
    def mask_thresh(self):
        return self._mask_thresh

    @mask_thresh.setter
    def mask_thresh(self, x):
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
            Variable behavior depending on the arg type. Length 2 tuples uses
            these values directly. Length 2 list of ints uses the calibrated
            origin value at this scan position. None uses the calibrated mean
            origin.
        ellipse : tuple or None
            Variable behavior depending on the arg type. Length 3 tuples uses
            these values directly (a,b,theta). None uses the calibrated value.
        mask : boolean array or None
            A mask applied to the data before transformation.  The value of
            masked pixels in the output is determined by `returnval`. Note that
            this mask is applied in combination with any mask at PolarData.mask.
        mask_thresh : number
            Pixels in the transformed mask with values below this number are
            considered masked, and will be populated by the values specified
            by `returnval`.
        returnval : 'masked' or 'nan' or 'all' or 'zeros' or 'all_zeros'
            Controls the returned data, including how un-sampled points
            are handled.
              - 'masked' returns a numpy masked array.
              - 'nan' returns a normal numpy array with unsampled pixels set to
                np.nan.
              - 'all' returns a 4-tuple of numpy arrays - the transformed data
                with unsampled pixels set to 'nan', the normalization array, the
                normalization array scaled to account for the q-dependent
                sampling density, and the polar boolean mask
              - 'zeros' returns a normal numpy with unsampled pixels set to 0
              - 'all_zeros' returns the same 4-tuple as 'all', but with unsampled
                pixels in the transformed data array set to zeros.

         Returns
        --------
        variable
            see `returnval`, above. Default is a masked array representing
            the polar transformed data.
        """
        return self._polar_data_getter._transform

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += (
            "Retrieves diffraction images in polar coordinates, using .data[x,y] )"
        )
        return string


class PolarDataGetter:
    def __init__(
        self,
        polarcube,
    ):
        self._polarcube = polarcube

    def __getitem__(self, pos):
        # unpack scan position
        x, y = pos
        # get the data
        cartesian_data = self._polarcube._datacube[x, y]
        # transform
        ans = self._transform(cartesian_data, origin=[x, y], returnval="masked")
        # return
        return ans

    def _transform(
        self,
        cartesian_data,
        origin=None,
        ellipse=None,
        mask=None,
        mask_thresh=None,
        returnval="masked",
    ):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.

        Parameters
        ----------
        cartesian_data : array
            The data
        origin : tuple or list or None
            Variable behavior depending on the arg type. Length 2 tuples uses
            these values directly. Length 2 list of ints uses the calibrated
            origin value at this scan position. None uses the calibrated mean
            origin.
        ellipse : tuple or None
            Variable behavior depending on the arg type. Length 3 tuples uses
            these values directly (a,b,theta). None uses the calibrated value.
        mask : boolean array or None
            A mask applied to the data before transformation.  The value of
            masked pixels in the output is determined by `returnval`. Note that
            this mask is applied in combination with any mask at PolarData.mask.
        mask_thresh : number
            Pixels in the transformed mask with values below this number are
            considered masked, and will be populated by the values specified
            by `returnval`.
        returnval : 'masked' or 'nan' or 'all' or 'zeros' or 'all_zeros'
            Controls the returned data, including how un-sampled points
            are handled.
              - 'masked' returns a numpy masked array.
              - 'nan' returns a normal numpy array with unsampled pixels set to
                np.nan.
              - 'all' returns a 4-tuple of numpy arrays - the transformed data
                with unsampled pixels set to 'nan', the normalization array, the
                normalization array scaled to account for the q-dependent
                sampling density, and the polar boolean mask
              - 'zeros' returns a normal numpy with unsampled pixels set to 0
              - 'all_zeros' returns the same 4-tuple as 'all', but with unsampled
                pixels in the transformed data array set to zeros.

         Returns
        --------
        variable
            see `returnval`, above. Default is a masked array representing
            the polar transformed data.
        """

        # get calibrations
        if origin is None:
            origin = self._polarcube.calibration.get_origin_mean()
        elif isinstance(origin, list):
            origin = self._polarcube.calibration.get_origin(origin[0], origin[1])
        elif isinstance(origin, tuple):
            pass
        else:
            raise Exception(f"Invalid type for `origin`, {type(origin)}")

        if ellipse is None:
            ellipse = self._polarcube.calibration.get_ellipse()
        elif isinstance(ellipse, tuple):
            pass
        else:
            raise Exception(f"Invalid type for `ellipse`, {type(ellipse)}")

        # combine passed mask with default mask
        mask0 = self._polarcube.mask
        if mask is None and mask0 is None:
            mask = np.ones_like(cartesian_data, dtype=bool)
        elif mask is None:
            mask = mask0
        elif mask0 is None:
            mask = mask
        else:
            mask = mask * mask0

        if mask_thresh is None:
            mask_thresh = self._polarcube.mask_thresh

        # transform data
        ans = self._transform_array(
            cartesian_data * mask.astype("float"),
            origin,
            ellipse,
        )

        # transform normalization array
        ans_norm = self._transform_array(
            mask.astype("float"),
            origin,
            ellipse,
        )

        # scale the normalization array by the bin density
        norm_array = ans_norm * self._polarcube._annular_bin_step[np.newaxis]
        mask_bool = norm_array < mask_thresh

        # apply normalization
        ans = np.divide(
            ans,
            ans_norm,
            out=np.full_like(ans, np.nan),
            where=np.logical_not(mask_bool),
        )

        # radial power law scaling of output
        if self._polarcube.qscale is not None:
            ans *= self._polarcube._qscale_ar[np.newaxis, :]

        # return
        if returnval == "masked":
            ans = np.ma.array(data=ans, mask=mask_bool)
            return ans
        elif returnval == "nan":
            ans[mask_bool] = np.nan
            return ans
        elif returnval == "all":
            return ans, ans_norm, norm_array, mask_bool
        elif returnval == "zeros":
            ans[mask_bool] = 0
            return ans
        elif returnval == "all_zeros":
            ans[mask_bool] = 0
            return ans, ans_norm, norm_array, mask_bool
        else:
            raise Exception(f"Unexpected value {returnval} encountered for `returnval`")

    def _transform_array(
        self,
        data,
        origin,
        ellipse,
    ):
        # set origin
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]

        # circular
        if (ellipse is None) or (self._polarcube.ellipse) is False:
            # get polar coords
            rr = np.sqrt(x**2 + y**2)
            tt = np.mod(np.arctan2(y, x), self._polarcube._annular_range)

        # elliptical
        else:
            # unpack ellipse
            a, b, theta = ellipse

            # Get polar coords
            xc = x * np.cos(theta) + y * np.sin(theta)
            yc = (y * np.cos(theta) - x * np.sin(theta)) * (a / b)
            rr = (b / a) * np.hypot(xc, yc)
            tt = np.mod(np.arctan2(yc, xc) + theta, self._polarcube._annular_range)

        # transform to bin sampling
        r_ind = (rr - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = tt / self._polarcube.annular_step

        # get integers and increments
        r_ind_floor = np.floor(r_ind).astype("int")
        t_ind_floor = np.floor(t_ind).astype("int")
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor

        # resample
        sub = np.logical_and(
            r_ind_floor >= 0,
            r_ind_floor < self._polarcube.polar_shape[1],
        )
        im = np.bincount(
            r_ind_floor[sub]
            + np.mod(t_ind_floor[sub], self._polarcube.polar_shape[0])
            * self._polarcube.polar_shape[1],
            weights=data[sub] * (1 - dr[sub]) * (1 - dt[sub]),
            minlength=self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub]
            + np.mod(t_ind_floor[sub] + 1, self._polarcube.polar_shape[0])
            * self._polarcube.polar_shape[1],
            weights=data[sub] * (1 - dr[sub]) * (dt[sub]),
            minlength=self._polarcube.polar_size,
        )
        sub = np.logical_and(
            r_ind_floor >= -1, r_ind_floor < self._polarcube.polar_shape[1] - 1
        )
        im += np.bincount(
            r_ind_floor[sub]
            + 1
            + np.mod(t_ind_floor[sub], self._polarcube.polar_shape[0])
            * self._polarcube.polar_shape[1],
            weights=data[sub] * (dr[sub]) * (1 - dt[sub]),
            minlength=self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub]
            + 1
            + np.mod(t_ind_floor[sub] + 1, self._polarcube.polar_shape[0])
            * self._polarcube.polar_shape[1],
            weights=data[sub] * (dr[sub]) * (dt[sub]),
            minlength=self._polarcube.polar_size,
        )

        # reshape to 2D
        ans = np.reshape(im, self._polarcube.polar_shape)

        # apply KDE
        for a0 in range(self._polarcube.polar_shape[1]):
            # Use 5% (= exp(-(1/2*.1669)^2)) cutoff value
            # for adjacent pixel in kernel
            if self._polarcube._sigma_KDE[a0] > 0.1669:
                ans[:, a0] = gaussian_filter1d(
                    ans[:, a0],
                    sigma=self._polarcube._sigma_KDE[a0],
                    mode="wrap",
                )

        # return
        return ans

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += "Retrieves the diffraction pattern at scan position (x,y) in polar coordinates when sliced with [x,y]."
        return string
