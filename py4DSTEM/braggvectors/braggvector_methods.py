# BraggVectors methods
from __future__ import annotations
import inspect
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from emdfile import Array, Metadata, _read_metadata, tqdmnd
from py4DSTEM import show
from py4DSTEM.datacube import VirtualImage
from scipy.ndimage import gaussian_filter


class BraggVectorMethods:
    """
    A container for BraggVector object instance methods
    """

    # 2D histogram

    def histogram(
        self,
        mode="cal",
        sampling=1,
        weights=None,
        weights_thresh=0.005,
    ):
        """
        Returns a 2D histogram of Bragg vector intensities in diffraction space,
        aka a Bragg vector map.

        Parameters
        ----------
        mode : str
            Must be 'cal' or 'raw'. Use the calibrated or raw vector positions.
        sampling : number
            The sampling rate of the histogram, in units of the camera's sampling.
            `sampling = 2` upsamples and `sampling = 0.5` downsamples, each by a
            factor of 2.
        weights : None or array
            If None, use all real space scan positions.  Otherwise must be a real
            space shaped array representing a weighting factor applied to vector
            intensities from each scan position. If weights is boolean uses beam
            positions where weights is True. If weights is number-like, scales
            by the values, and skips positions where wieghts<weights_thresh.
        weights_thresh : number
            If weights is an array of numbers, pixels where weights>weight_thresh
            are skipped.

        Returns
        -------
        BraggVectorHistogram
            An Array with .data representing the data, and .dim[0] and .dim[1]
            representing the sampling grid.
        """
        # get vectors
        assert mode in ("cal", "raw"), f"Invalid mode {mode}!"
        if mode == "cal":
            v = self.cal
        else:
            v = self.raw

        # condense vectors into a single array for speed,
        # handling any weight factors
        if weights is None:
            vects = np.concatenate(
                [
                    v[i, j].data
                    for i in range(self.Rshape[0])
                    for j in range(self.Rshape[1])
                ]
            )
        elif weights.dtype == bool:
            x, y = np.nonzero(weights)
            vects = np.concatenate([v[i, j].data for i, j in zip(x, y)])
        else:
            l = []
            x, y = np.nonzero(weights > weights_thresh)
            for i, j in zip(x, y):
                d = v[i, j].data
                d["intensity"] *= weights[i, j]
                l.append(d)
            vects = np.concatenate(l)
        # get the vectors
        qx = vects["qx"]
        qy = vects["qy"]
        I = vects["intensity"]

        # Set up bin grid
        Q_Nx = np.round(self.Qshape[0] * sampling).astype(int)
        Q_Ny = np.round(self.Qshape[1] * sampling).astype(int)

        # transform vects onto bin grid
        if mode == "raw":
            qx *= sampling
            qy *= sampling
        # calibrated vects
        # to tranform to the bingrid we ~undo the calibrations,
        # then scale by the sampling factor
        else:
            # get pixel calibration
            if self.calstate["pixel"] is True:
                qpix = self.calibration.get_Q_pixel_size()
                qx /= qpix
                qy /= qpix
            # origin calibration
            if self.calstate["center"] is True:
                origin = self.calibration.get_origin_mean()
                qx += origin[0]
                qy += origin[1]
            # resample
            qx *= sampling
            qy *= sampling

        # round to nearest integer
        floorx = np.floor(qx).astype(np.int64)
        ceilx = np.ceil(qx).astype(np.int64)
        floory = np.floor(qy).astype(np.int64)
        ceily = np.ceil(qy).astype(np.int64)

        # Remove any points outside the bin grid
        mask = np.logical_and.reduce(
            ((floorx >= 0), (floory >= 0), (ceilx < Q_Nx), (ceily < Q_Ny))
        )
        qx = qx[mask]
        qy = qy[mask]
        I = I[mask]
        floorx = floorx[mask]
        floory = floory[mask]
        ceilx = ceilx[mask]
        ceily = ceily[mask]

        # Interpolate values
        dx = qx - floorx
        dy = qy - floory
        # Compute indices of the 4 neighbors to (qx,qy)
        # floor x, floor y
        inds00 = np.ravel_multi_index([floorx, floory], (Q_Nx, Q_Ny))
        # floor x, ceil y
        inds01 = np.ravel_multi_index([floorx, ceily], (Q_Nx, Q_Ny))
        # ceil x, floor y
        inds10 = np.ravel_multi_index([ceilx, floory], (Q_Nx, Q_Ny))
        # ceil x, ceil y
        inds11 = np.ravel_multi_index([ceilx, ceily], (Q_Nx, Q_Ny))

        # Compute the histogram by accumulating intensity in each
        # neighbor weighted by linear interpolation
        hist = (
            np.bincount(inds00, I * (1.0 - dx) * (1.0 - dy), minlength=Q_Nx * Q_Ny)
            + np.bincount(inds01, I * (1.0 - dx) * dy, minlength=Q_Nx * Q_Ny)
            + np.bincount(inds10, I * dx * (1.0 - dy), minlength=Q_Nx * Q_Ny)
            + np.bincount(inds11, I * dx * dy, minlength=Q_Nx * Q_Ny)
        ).reshape(Q_Nx, Q_Ny)

        # determine the resampled grid center and pixel size
        if mode == "cal" and self.calstate["center"] is True:
            x0 = sampling * origin[0]
            y0 = sampling * origin[1]
        else:
            x0, y0 = 0, 0
        if mode == "cal" and self.calstate["pixel"] is True:
            pixelsize = qpix / sampling
        else:
            pixelsize = 1 / sampling
        # find the dim vectors
        dimx = (np.arange(Q_Nx) - x0) * pixelsize
        dimy = (np.arange(Q_Ny) - y0) * pixelsize
        dim_units = self.calibration.get_Q_pixel_units()

        # wrap in a class
        ans = BraggVectorMap(
            name=f"2Dhist_{self.name}_{mode}_s={sampling}",
            data=hist,
            weights=weights,
            dims=[dimx, dimy],
            dim_units=dim_units,
            origin=(x0, y0),
            pixelsize=pixelsize,
        )

        # return
        return ans

    # aliases
    get_bvm = get_bragg_vector_map = histogram

    # bragg virtual imaging

    def get_virtual_image(
        self,
        mode=None,
        geometry=None,
        name="bragg_virtual_image",
        returncalc=True,
        center=True,
        ellipse=True,
        pixel=True,
        rotate=True,
    ):
        """
        Calculates a virtual image based on the values of the Braggvectors
        integrated over some detector function determined by `mode` and
        `geometry`.

        Parameters
        ----------
        mode : str
            defines the type of detector used. Options:
              - 'circular', 'circle': uses round detector, like bright field
              - 'annular', 'annulus': uses annular detector, like dark field
        geometry : variable
            expected value depends on the value of `mode`, as follows:
              - 'circle', 'circular': nested 2-tuple, ((qx,qy),radius)
              - 'annular' or 'annulus': nested 2-tuple,
                ((qx,qy),(radius_i,radius_o))
             Values can be in pixels or calibrated units. Note that (qx,qy)
             can be skipped, which assumes peaks centered at (0,0).
        center: bool
            Apply calibration - center coordinate.
        ellipse: bool
            Apply calibration - elliptical correction.
        pixel: bool
            Apply calibration - pixel size.
        rotate: bool
            Apply calibration - QR rotation.

        Returns
        -------
        virtual_im : VirtualImage
        """

        # parse inputs
        circle_modes = ["circular", "circle"]
        annulus_modes = ["annular", "annulus"]
        modes = circle_modes + annulus_modes + [None]
        assert mode in modes, f"Unrecognized mode {mode}"

        # set geometry
        if mode is None:
            if geometry is None:
                qxy_center = None
                radial_range = np.array((0, np.inf))
            else:
                if len(geometry[0]) == 0:
                    qxy_center = None
                else:
                    qxy_center = np.array(geometry[0])
                if isinstance(geometry[1], int) or isinstance(geometry[1], float):
                    radial_range = np.array((0, geometry[1]))
                elif len(geometry[1]) == 0:
                    radial_range = None
                else:
                    radial_range = np.array(geometry[1])
        elif mode == "circular" or mode == "circle":
            radial_range = np.array((0, geometry[1]))
            if len(geometry[0]) == 0:
                qxy_center = None
            else:
                qxy_center = np.array(geometry[0])
        elif mode == "annular" or mode == "annulus":
            radial_range = np.array(geometry[1])
            if len(geometry[0]) == 0:
                qxy_center = None
            else:
                qxy_center = np.array(geometry[0])

        # allocate space
        im_virtual = np.zeros(self.shape)

        # generate image
        for rx, ry in tqdmnd(
            self.shape[0],
            self.shape[1],
        ):
            # Get user-specified Bragg vectors
            p = self.get_vectors(
                rx,
                ry,
                center=center,
                ellipse=ellipse,
                pixel=pixel,
                rotate=rotate,
            )

            if p.data.shape[0] > 0:
                if radial_range is None:
                    im_virtual[rx, ry] = np.sum(p.I)
                else:
                    if qxy_center is None:
                        qr = np.hypot(p.qx, p.qy)
                    else:
                        qr = np.hypot(p.qx - qxy_center[0], p.qy - qxy_center[1])
                    sub = np.logical_and(qr >= radial_range[0], qr < radial_range[1])
                    if np.sum(sub) > 0:
                        im_virtual[rx, ry] = np.sum(p.I[sub])

        # wrap in Virtual Image class
        ans = VirtualImage(data=im_virtual, name=name)
        # add generating params as metadta
        ans.metadata = Metadata(
            name="gen_params",
            data={
                "_calling_method": inspect.stack()[0][3],
                "_calling_class": __class__.__name__,
                "mode": mode,
                "geometry": geometry,
                "name": name,
                "returncalc": returncalc,
            },
        )
        # attach to the tree
        self.attach(ans)

        # return
        if returncalc:
            return ans

    # calibration measurements

    def measure_origin(
        self,
        center_guess=None,
        score_method=None,
        findcenter="max",
    ):
        """
        Finds the diffraction shifts of the center beam using the raw Bragg
        vector measurements.

        If a center guess is not specified, first, a guess at the unscattered
        beam position is determined, either by taking the CoM of the Bragg vector
        map, or by taking its maximal pixel.  Once a unscattered beam position is
        determined, the Bragg peak closest to this position is identified. The
        shifts in these peaks positions from their average are returned as the
        diffraction shifts.

        Parameters
        ----------
        center_guess : 2-tuple
            initial guess for the center
        score_method : str
            Method used to find center peak
                - 'intensity': finds the most intense Bragg peak near the center
                - 'distance': finds the closest Bragg peak to the center
                - 'intensity weighted distance': determines center through a
                  combination of weighting distance and intensity
        findcenter (str): specifies the method for determining the unscattered beam
            position options: 'CoM', or 'max.' Only used if center_guess is None.
            CoM finds the center of mass of bragg ector map, 'max' uses its
            brightest pixel.

        Returns:
            (3-tuple): A 3-tuple comprised of:

                * **qx0** *((R_Nx,R_Ny)-shaped array)*: the origin x-coord
                * **qy0** *((R_Nx,R_Ny)-shaped array)*: the origin y-coord
                * **braggvectormap** *((R_Nx,R_Ny)-shaped array)*: the Bragg vector map of only
                  the Bragg peaks identified with the unscattered beam. Useful for diagnostic
                  purposes.
        """
        assert findcenter in ["CoM", "max"], "center must be either 'CoM' or 'max'"
        assert score_method in [
            "distance",
            "intensity",
            "intensity weighted distance",
            None,
        ], "center must be either 'distance' or 'intensity weighted distance'"

        R_Nx, R_Ny = self.Rshape
        Q_Nx, Q_Ny = self.Qshape

        # Default scoring method
        if score_method is None:
            if center_guess is None:
                score_method = "intensity"
            else:
                score_method = "distance"

        # Get guess at position of unscattered beam (x0,y0)
        if center_guess is None:
            bvm = self.histogram(mode="raw")
            if findcenter == "max":
                x0, y0 = np.unravel_index(
                    np.argmax(gaussian_filter(bvm, 10)), (Q_Nx, Q_Ny)
                )
            else:
                from py4DSTEM.process.utils import get_CoM

                x0, y0 = get_CoM(bvm)
        else:
            x0, y0 = center_guess

        # Get Bragg peak closest to unscattered beam at each scan position
        qx0 = np.zeros(self.Rshape)
        qy0 = np.zeros(self.Rshape)
        mask = np.ones(self.Rshape, dtype=bool)
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                vects = self.raw[Rx, Ry].data
                if len(vects) > 0:
                    if score_method == "distance":
                        r2 = (vects["qx"] - x0) ** 2 + (vects["qy"] - y0) ** 2
                        index = np.argmin(r2)
                    elif score_method == "intensity":
                        index = np.argmax(vects["intensity"])
                    elif score_method == "intensity weighted distance":
                        r2 = vects["intensity"] / (
                            1 + ((vects["qx"] - x0) ** 2 + (vects["qy"] - y0) ** 2)
                        )
                        index = np.argmax(r2)
                    qx0[Rx, Ry] = vects["qx"][index]
                    qy0[Rx, Ry] = vects["qy"][index]
                else:
                    mask = False
                    qx0[Rx, Ry] = x0
                    qy0[Rx, Ry] = y0

        # set calibration metadata
        self.calibration.set_origin_meas((qx0, qy0))
        self.calibration.set_origin_meas_mask(mask)

        # return
        return qx0, qy0, mask

    def measure_origin_beamstop(
        self, center_guess, radii, max_dist=None, max_iter=1, **kwargs
    ):
        """
        Find the origin from a set of braggpeaks assuming there is a beamstop, by identifying
        pairs of conjugate peaks inside an annular region and finding their centers of mass.

        Args:
            center_guess (2-tuple): qx0,qy0
            radii (2-tuple): the inner and outer radii of the specified annular region
            max_dist (number): the maximum allowed distance between the reflection of two
                peaks to consider them conjugate pairs
            max_iter (integer): for values >1, repeats the algorithm after updating center_guess

        Returns:
            (2d masked array): the origins
        """
        R_Nx, R_Ny = self.Rshape
        braggpeaks = self._v_uncal

        if max_dist is None:
            max_dist = radii[1]

        # remove peaks outside the annulus
        braggpeaks_masked = braggpeaks.copy()
        for rx in range(R_Nx):
            for ry in range(R_Ny):
                pl = braggpeaks_masked[rx, ry]
                qr = np.hypot(
                    pl.data["qx"] - center_guess[0], pl.data["qy"] - center_guess[1]
                )
                rm = np.logical_not(np.logical_and(qr >= radii[0], qr <= radii[1]))
                pl.remove(rm)

        # Find all matching conjugate pairs of peaks
        center_curr = center_guess
        for ii in range(max_iter):
            centers = np.zeros((R_Nx, R_Ny, 2))
            found_center = np.zeros((R_Nx, R_Ny), dtype=bool)
            for rx in range(R_Nx):
                for ry in range(R_Ny):
                    # Get data
                    pl = braggpeaks_masked[rx, ry]
                    is_paired = np.zeros(len(pl.data), dtype=bool)
                    matches = []

                    # Find matching pairs
                    for i in range(len(pl.data)):
                        if not is_paired[i]:
                            x, y = pl.data["qx"][i], pl.data["qy"][i]
                            x_r = -x + 2 * center_curr[0]
                            y_r = -y + 2 * center_curr[1]
                            dists = np.hypot(x_r - pl.data["qx"], y_r - pl.data["qy"])
                            dists[is_paired] = max_dist
                            matched = dists <= max_dist
                            if any(matched):
                                match = np.argmin(dists)
                                matches.append((i, match))
                                is_paired[i], is_paired[match] = True, True

                    # Find the center
                    if len(matches) > 0:
                        x0, y0 = [], []
                        for i in range(len(matches)):
                            x0.append(np.mean(pl.data["qx"][list(matches[i])]))
                            y0.append(np.mean(pl.data["qy"][list(matches[i])]))
                        x0, y0 = np.mean(x0), np.mean(y0)
                        centers[rx, ry, :] = x0, y0
                        found_center[rx, ry] = True
                    else:
                        found_center[rx, ry] = False

            # Update current center guess
            x0_curr = np.mean(centers[found_center, 0])
            y0_curr = np.mean(centers[found_center, 1])
            center_curr = x0_curr, y0_curr

        # collect answers
        mask = found_center
        qx0, qy0 = centers[:, :, 0], centers[:, :, 1]

        # set calibration metadata
        self.calibration.set_origin_meas((qx0, qy0))
        self.calibration.set_origin_meas_mask(mask)

        # return
        return qx0, qy0, mask

    def fit_origin(
        self,
        mask=None,
        fitfunction="plane",
        robust=False,
        robust_steps=3,
        robust_thresh=2,
        mask_check_data=True,
        plot=True,
        plot_range=None,
        cmap="RdBu_r",
        returncalc=True,
        **kwargs,
    ):
        """
        Fit origin of bragg vectors.

        Args:
            mask (2b boolean array, optional): ignore points where mask=True
            fitfunction (str, optional): must be 'plane' or 'parabola' or 'bezier_two'
            robust (bool, optional): If set to True, fit will be repeated with outliers
                removed.
            robust_steps (int, optional): Optional parameter. Number of robust iterations
                                    performed after initial fit.
            robust_thresh (int, optional): Threshold for including points, in units of
                root-mean-square (standard deviations) error of the predicted values after
                fitting.
            mask_check_data (bool):     Get mask from origin measurements equal to zero. (TODO - replace)
            plot (bool, optional): plot results
            plot_range (float):    min and max color range for plot (pixels)
            cmap (colormap): plotting colormap

        Returns:
            (variable): Return value depends on returnfitp. If ``returnfitp==False``
            (default), returns a 4-tuple containing:

                * **qx0_fit**: *(ndarray)* the fit origin x-position
                * **qy0_fit**: *(ndarray)* the fit origin y-position
                * **qx0_residuals**: *(ndarray)* the x-position fit residuals
                * **qy0_residuals**: *(ndarray)* the y-position fit residuals
        """
        q_meas = self.calibration.get_origin_meas()

        from py4DSTEM.process.calibration import fit_origin

        if mask_check_data is True:
            data_mask = np.logical_not(q_meas[0] == 0)
            if mask is None:
                mask = data_mask
            else:
                mask = np.logical_and(mask, data_mask)

        qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = fit_origin(
            tuple(q_meas),
            mask=mask,
            fitfunction=fitfunction,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )

        # try to add update calibration metadata
        try:
            self.calibration.set_origin((qx0_fit, qy0_fit))
            self.setcal()
        except AttributeError:
            warn(
                "No calibration found on this datacube - fit values are not being stored"
            )
            pass

        # show
        if plot:
            self.show_origin_fit(
                q_meas[0],
                q_meas[1],
                qx0_fit,
                qy0_fit,
                qx0_residuals,
                qy0_residuals,
                mask=mask,
                plot_range=plot_range,
                cmap=cmap,
                **kwargs,
            )

        # return
        if returncalc:
            return qx0_fit, qy0_fit, qx0_residuals, qy0_residuals

    def show_origin_fit(
        self,
        qx0_meas,
        qy0_meas,
        qx0_fit,
        qy0_fit,
        qx0_residuals,
        qy0_residuals,
        mask=None,
        plot_range=None,
        cmap="RdBu_r",
        **kwargs,
    ):
        # apply mask
        if mask is not None:
            qx0_meas = np.ma.masked_array(qx0_meas, mask=np.logical_not(mask))
            qy0_meas = np.ma.masked_array(qy0_meas, mask=np.logical_not(mask))
            qx0_residuals = np.ma.masked_array(qx0_residuals, mask=np.logical_not(mask))
            qy0_residuals = np.ma.masked_array(qy0_residuals, mask=np.logical_not(mask))
        qx0_mean = np.mean(qx0_fit)
        qy0_mean = np.mean(qy0_fit)

        # set range
        if plot_range is None:
            plot_range = max(
                (
                    1.5 * np.max(np.abs(qx0_fit - qx0_mean)),
                    1.5 * np.max(np.abs(qy0_fit - qy0_mean)),
                )
            )

        # set figsize
        imsize_ratio = np.sqrt(qx0_meas.shape[1] / qx0_meas.shape[0])
        axsize = (3 * imsize_ratio, 3 / imsize_ratio)
        axsize = kwargs.pop("axsize", axsize)

        # plot
        fig, ax = show(
            [
                [qx0_meas - qx0_mean, qx0_fit - qx0_mean, qx0_residuals],
                [qy0_meas - qy0_mean, qy0_fit - qy0_mean, qy0_residuals],
            ],
            cmap=cmap,
            axsize=axsize,
            title=[
                "measured origin, x",
                "fitorigin, x",
                "residuals, x",
                "measured origin, y",
                "fitorigin, y",
                "residuals, y",
            ],
            vmin=-1 * plot_range,
            vmax=1 * plot_range,
            intensity_range="absolute",
            show_cbar=True,
            returnfig=True,
            **kwargs,
        )
        plt.tight_layout()

        return

    def fit_p_ellipse(
        self, bvm, center, fitradii, mask=None, returncalc=False, **kwargs
    ):
        """
        Args:
            bvm (BraggVectorMap): a 2D array used for ellipse fitting
            center (2-tuple of floats): the center (x0,y0) of the annular fitting region
            fitradii (2-tuple of floats): inner and outer radii (ri,ro) of the fit region
            mask (ar-shaped ndarray of bools): ignore data wherever mask==True

        Returns:
            p_ellipse if returncal is True
        """
        from py4DSTEM.process.calibration import fit_ellipse_1D

        p_ellipse = fit_ellipse_1D(bvm, center, fitradii, mask)

        scaling = kwargs.get("scaling", "log")
        kwargs.pop("scaling", None)
        from py4DSTEM.visualize import show_elliptical_fit

        show_elliptical_fit(bvm, fitradii, p_ellipse, scaling=scaling, **kwargs)

        self.calibration.set_p_ellipse(p_ellipse)
        self.setcal()

        if returncalc:
            return p_ellipse

    def mask_in_Q(self, mask, update_inplace=False, returncalc=True):
        """
        Remove peaks which fall inside the diffraction shaped boolean array
        `mask`, in raw (uncalibrated) peak positions.

        Parameters
        ----------
        mask : 2d boolean array
            The mask. Must be diffraction space shaped
        update_inplace : bool
            If False (default) copies this BraggVectors instance and
            removes peaks from the copied instance. If True, removes
            peaks from this instance.
        returncalc : bool
            Toggles returning the answer

        Returns
        -------
        bvects : BraggVectors
        """
        # Copy peaks, if requested
        if update_inplace:
            v = self._v_uncal
        else:
            v = self._v_uncal.copy(name="_v_uncal")

        # Loop and remove masked peaks
        for rx in range(v.shape[0]):
            for ry in range(v.shape[1]):
                p = v[rx, ry]
                xs = np.round(p.data["qx"]).astype(int)
                ys = np.round(p.data["qy"]).astype(int)
                sub = mask[xs, ys]
                p.remove(sub)

        # assign the return value
        if update_inplace:
            ans = self
        else:
            ans = self.copy(name=self.name + "_masked")
            ans.set_raw_vectors(v)

        # return
        if returncalc:
            return ans
        else:
            return

    # alias
    def get_masked_peaks(self, mask, update_inplace=False, returncalc=True):
        """
        Alias for `mask_in_Q`.
        """
        warn(
            "`.get_masked_peaks` is deprecated and will be removed in a future version. Use `.mask_in_Q`"
        )
        return self.mask_in_Q(
            mask=mask, update_inplace=update_inplace, returncalc=returncalc
        )

    def mask_in_R(self, mask, update_inplace=False, returncalc=True):
        """
        Remove peaks which fall inside the real space shaped boolean array
        `mask`.

        Parameters
        ----------
        mask : 2d boolean array
            The mask. Must be real space shaped
        update_inplace : bool
            If False (default) copies this BraggVectors instance and
            removes peaks from the copied instance. If True, removes
            peaks from this instance.
        returncalc : bool
            Toggles returning the answer

        Returns
        -------
        bvects : BraggVectors
        """
        # Copy peaks, if requested
        if update_inplace:
            v = self._v_uncal
        else:
            v = self._v_uncal.copy(name="_v_uncal")

        # Loop and remove masked peaks
        for rx in range(v.shape[0]):
            for ry in range(v.shape[1]):
                if mask[rx, ry]:
                    p = v[rx, ry]
                    p.remove(np.ones(len(p), dtype=bool))

        # assign the return value
        if update_inplace:
            ans = self
        else:
            ans = self.copy(name=self.name + "_masked")
            ans.set_raw_vectors(v)

        # return
        if returncalc:
            return ans
        else:
            return

    def to_strainmap(self, name: str = None):
        """
        Generate a StrainMap object from the BraggVectors
        equivalent to py4DSTEM.StrainMap(braggvectors=braggvectors)

        Args:
            name (str, optional): The name of the strainmap. Defaults to None which reverts to default name 'strainmap'.

        Returns:
            py4DSTEM.StrainMap: A py4DSTEM StrainMap object generated from the BraggVectors
        """
        from py4DSTEM.process.strain import StrainMap

        return StrainMap(self, name) if name else StrainMap(self)

    def plot(
        self,
        index: tuple[int, int] | list[int],
        cal: str = "cal",
        returnfig: bool = False,
        **kwargs,
    ):
        """
        Plot Bragg vector, from a specified index.
        Calls py4DSTEM.process.diffraction.plot_diffraction_pattern(braggvectors.<cal/raw>[index], **kwargs).
        Optionally can return the figure.

        Parameters
        ----------
        index : tuple[int,int] | list[int]
            scan position for which Bragg vectors to plot
        cal : str, optional
            Choice to plot calibrated or raw Bragg vectors must be 'raw' or 'cal', by default 'cal'
        returnfig : bool, optional
            Boolean to return figure or not, by default False

        Returns
        -------
        tuple (figure, axes)
            matplotlib figure, axes returned if `returnfig` is True
        """
        cal = cal.lower()
        assert cal in (
            "cal",
            "raw",
        ), f"'cal' must be in ('cal', 'raw') {cal = } passed"
        from py4DSTEM.process.diffraction import plot_diffraction_pattern

        if cal == "cal":
            pl = self.cal[index]
        else:
            pl = self.raw[index]

        if returnfig:
            return plot_diffraction_pattern(
                pl,
                returnfig=returnfig,
                **kwargs,
            )
        else:
            plot_diffraction_pattern(
                pl,
                **kwargs,
            )


######### END BraggVectorMethods CLASS ########


class BraggVectorMap(Array):
    def __init__(self, name, data, weights, dims, dim_units, origin, pixelsize):
        Array.__init__(
            self,
            name=name,
            data=data,
            dims=dims,
            dim_units=[dim_units, dim_units],
        )
        self.metadata = Metadata(
            name="grid",
            data={"origin": origin, "pixelsize": pixelsize, "weights": weights},
        )

    @property
    def origin(self):
        return self.metadata["grid"]["origin"]

    @property
    def pixelsize(self):
        return self.metadata["grid"]["pixelsize"]

    @property
    def pixelunits(self):
        return self.dim_units[0]

    @property
    def weights(self):
        return self.metadata["grid"]["weights"]

    # read
    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        constr_args = Array._get_constructor_args(group)
        metadata = _read_metadata(group, "grid")
        args = {
            "name": constr_args["name"],
            "data": constr_args["data"],
            "weights": metadata["weights"],
            "dims": constr_args["dims"],
            "dim_units": constr_args["dim_units"],
            "origin": metadata["origin"],
            "pixelsize": metadata["pixelsize"],
        }
        return args
