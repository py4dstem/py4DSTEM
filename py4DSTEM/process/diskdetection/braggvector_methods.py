# BraggVectors methods

import numpy as np
from scipy.ndimage import gaussian_filter
from emdfile import Array,Metadata
from emdfile import _read_metadata
from py4DSTEM.process.calibration.origin import set_measured_origin, set_fit_origin
from py4DSTEM.process.utils import get_CoM


class BraggVectorMethods:
    """
    A container for BraggVector object instance methods
    """

    # 2D histogram

    def histogram(
        self,
        mode = 'cal',
        sampling = 1,
        weights = None,
        weights_thresh = 0.005,
        ):
        """
        Returns a 2D histogram of Bragg vector intensities in diffraction space.

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
        assert(mode in ('cal','raw')), f"Invalid mode {mode}!"
        if mode == 'cal':
            v = self.cal
        else:
            v = self.raw

        # condense vectors into a single array for speed,
        # handling any weight factors
        if weights is None:
            vects = np.concatenate(
                [v[i,j].data for i in range(self.Rshape[0]) for j in range(self.Rshape[1])])
        elif weights.dtype == bool:
            x,y = np.nonzero(weights)
            vects = np.concatenate(
                [v[i,j].data for i,j in zip(x,y)])
        else:
            l = []
            x,y = np.nonzero(weights>weights_thresh)
            for i,j in zip(x,y):
                d = v[i,j].data
                d['intensity'] *= weights[i,j]
                l.append(d)
            vects = np.concatenate(l)
        # get the vectors
        qx = vects['qx']
        qy = vects['qy']
        I = vects['intensity']

        # Set up bin grid
        Q_Nx = np.round(self.Qshape[0]*sampling).astype(int)
        Q_Ny = np.round(self.Qshape[1]*sampling).astype(int)

        # transform vects onto bin grid
        if mode == 'raw':
            qx *= sampling
            qy *= sampling
        # calibrated vects
        # to tranform to the bingrid we ~undo the calibrations,
        # then scale by the sampling factor
        else:
            # get pixel calibration
            if self.calstate['pixel']==True:
                qpix = self.calibration.get_Q_pixel_size()
                qx /= qpix
                qy /= qpix
            # origin calibration
            if self.calstate['center']==True:
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
            ((floorx>=0),(floory>=0),(ceilx<Q_Nx),(ceily<Q_Ny)))
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
        inds00 = np.ravel_multi_index([floorx,floory],(Q_Nx,Q_Ny))
        # floor x, ceil y
        inds01 = np.ravel_multi_index([floorx,ceily],(Q_Nx,Q_Ny))
        # ceil x, floor y
        inds10 = np.ravel_multi_index([ceilx,floory],(Q_Nx,Q_Ny))
        # ceil x, ceil y
        inds11 = np.ravel_multi_index([ceilx,ceily],(Q_Nx,Q_Ny))

        # Compute the histogram by accumulating intensity in each
        # neighbor weighted by linear interpolation
        hist = (
            np.bincount(inds00, I * (1.-dx) * (1.-dy), minlength=Q_Nx*Q_Ny) \
            + np.bincount(inds01, I * (1.-dx) * dy, minlength=Q_Nx*Q_Ny) \
            + np.bincount(inds10, I * dx * (1.-dy), minlength=Q_Nx*Q_Ny) \
            + np.bincount(inds11, I * dx * dy, minlength=Q_Nx*Q_Ny)
        ).reshape(Q_Nx,Q_Ny)

        # determine the resampled grid center and pixel size
        if mode == 'cal' and self.calstate['center']==True:
            x0 = sampling*origin[0]
            y0 = sampling*origin[1]
        else:
            x0,y0 = 0,0
        if mode == 'cal' and self.calstate['pixel']==True:
            pixelsize = qpix/sampling
        else:
            pixelsize = 1/sampling
        # find the dim vectors
        dimx = (np.arange(Q_Nx)-x0)*pixelsize
        dimy = (np.arange(Q_Ny)-y0)*pixelsize
        dim_units = self.calibration.get_Q_pixel_units()

        # wrap in a class
        ans = BraggVectorMap(
            name = f'2Dhist_{self.name}_{mode}_s={sampling}',
            data = hist,
            weights = weights,
            dims = [dimx,dimy],
            dim_units = dim_units,
            origin = (x0,y0),
            pixelsize = pixelsize
        )

        # return
        return ans

    # aliases
    get_bvm = get_bragg_vector_map = histogram



    # calibration measurements

    @set_measured_origin
    def measure_origin(
        self,
        center_guess = None,
        score_method = None,
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
        assert(findcenter in ["CoM", "max"]), "center must be either 'CoM' or 'max'"
        assert score_method in ["distance", "intensity", "intensity weighted distance", None], "center must be either 'distance' or 'intensity weighted distance'"

        R_Nx,R_Ny = self.Rshape
        Q_Nx,Q_Ny = self.Qshape

        # Default scoring method
        if score_method is None:
            if center_guess is None:
                score_method = "intensity"
            else:
                score_method = "distance"

        # Get guess at position of unscattered beam (x0,y0)
        if center_guess is None:
            bvm = self.histogram( mode='raw' )
            if findcenter == "max":
                x0, y0 = np.unravel_index(
                    np.argmax(gaussian_filter(bvm, 10)), (Q_Nx, Q_Ny)
                )
            else:
                x0, y0 = get_CoM(bvm)
        else:
            x0, y0 = center_guess

        # Get Bragg peak closest to unscattered beam at each scan position
        qx0 = np.zeros(self.Rshape)
        qy0 = np.zeros(self.Rshape)
        mask = np.ones(self.Rshape, dtype=bool)
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                vects = self.raw[Rx,Ry].data
                if len(vects) > 0:
                    if score_method == "distance":
                        r2 = (vects["qx"] - x0) ** 2 + (vects["qy"] - y0) ** 2
                        index = np.argmin(r2)
                    elif score_method == "intensity":
                        index = np.argmax(vects["intensity"])
                    elif score_method == "intensity weighted distance":
                        r2 = vects["intensity"]/(1+((vects["qx"] - x0) ** 2 + (vects["qy"] - y0) ** 2))
                        index = np.argmax(r2)
                    qx0[Rx, Ry] = vects["qx"][index]
                    qy0[Rx, Ry] = vects["qy"][index]
                else:
                    mask = False
                    qx0[Rx, Ry] = x0
                    qy0[Rx, Ry] = y0

        # return
        return qx0, qy0, mask


    @set_measured_origin
    def measure_origin_beamstop(
        self,
        center_guess,
        radii,
        max_dist=2,
        max_iter=1,
        **kwargs
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
        R_Nx,R_Ny = self.Rshape
        braggpeaks = self._v_uncal

        # remove peaks outside the annulus
        braggpeaks_masked = braggpeaks.copy()
        for rx in range(R_Nx):
            for ry in range(R_Ny):
                pl = braggpeaks_masked[rx,ry]
                qr = np.hypot(pl.data['qx']-center_guess[0],
                              pl.data['qy']-center_guess[1])
                rm = np.logical_not(np.logical_and(qr>=radii[0],qr<=radii[1]))
                pl.remove(rm)

        # Find all matching conjugate pairs of peaks
        center_curr = center_guess
        for ii in range(max_iter):
            centers = np.zeros((R_Nx,R_Ny,2))
            found_center = np.zeros((R_Nx,R_Ny),dtype=bool)
            for rx in range(R_Nx):
                for ry in range(R_Ny):

                    # Get data
                    pl = braggpeaks_masked[rx,ry]
                    is_paired = np.zeros(len(pl.data),dtype=bool)
                    matches = []

                    # Find matching pairs
                    for i in range(len(pl.data)):
                        if not is_paired[i]:
                            x,y = pl.data['qx'][i],pl.data['qy'][i]
                            x_r = -x+2*center_curr[0]
                            y_r = -y+2*center_curr[1]
                            dists = np.hypot(x_r-pl.data['qx'],y_r-pl.data['qy'])
                            dists[is_paired] = 2*max_dist
                            matched = dists<=max_dist
                            if(any(matched)):
                                match = np.argmin(dists)
                                matches.append((i,match))
                                is_paired[i],is_paired[match] = True,True

                    # Find the center
                    if len(matches)>0:
                        x0,y0 = [],[]
                        for i in range(len(matches)):
                            x0.append(np.mean(pl.data['qx'][list(matches[i])]))
                            y0.append(np.mean(pl.data['qy'][list(matches[i])]))
                        x0,y0 = np.mean(x0),np.mean(y0)
                        centers[rx,ry,:] = x0,y0
                        found_center[rx,ry] = True
                    else:
                        found_center[rx,ry] = False

            # Update current center guess
            x0_curr = np.mean(centers[found_center,0])
            y0_curr = np.mean(centers[found_center,1])
            center_curr = x0_curr,y0_curr

        # return
        mask = found_center
        qx0,qy0 = centers[:,:,0],centers[:,:,1]

        return qx0,qy0,mask


    @set_fit_origin
    def fit_origin(
        self,
        mask=None,
        fitfunction="plane",
        robust=False,
        robust_steps=3,
        robust_thresh=2,
        mask_check_data = True,
        plot = True,
        plot_range = None,
        returncalc = True,
        **kwargs
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
            # TODO - replace this bad hack for the mask for the origin fit
            mask = np.logical_not(q_meas[0]==0)
            qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = fit_origin(
                tuple(q_meas),
                mask = mask,
                )
        else:
            qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = fit_origin(
                tuple(q_meas))

        # try to add to calibration
        try:
            self.calibration.set_origin([qx0_fit,qy0_fit])
        except AttributeError:
            # should a warning be raised?
            pass
        if plot:
            from py4DSTEM.visualize import show_image_grid
            if mask is None:
                qx0_meas,qy0_meas = q_meas
                qx0_res_plot = qx0_residuals
                qy0_res_plot = qy0_residuals
            else:
                qx0_meas = np.ma.masked_array(q_meas[0], mask = np.logical_not(mask))
                qy0_meas = np.ma.masked_array(q_meas[1], mask = np.logical_not(mask))
                qx0_res_plot = np.ma.masked_array(qx0_residuals, mask = np.logical_not(mask))
                qy0_res_plot = np.ma.masked_array(qy0_residuals, mask = np.logical_not(mask))
            qx0_mean = np.mean(qx0_fit)
            qy0_mean = np.mean(qy0_fit)


            if plot_range is None:
                plot_range = 2*np.max(qx0_fit - qx0_mean)

            cmap = kwargs.get("cmap", "RdBu_r")
            kwargs.pop("cmap", None)
            axsize = kwargs.get("axsize", (6,2))
            kwargs.pop("axsize", None)

            show_image_grid(
                lambda i:[qx0_meas-qx0_mean,qx0_fit-qx0_mean,qx0_res_plot,
                          qy0_meas-qy0_mean,qy0_fit-qy0_mean,qy0_res_plot][i],
                H = 2,
                W = 3,
                cmap = cmap,
                axsize = axsize,
                vmin = -1*plot_range,
                vmax = 1*plot_range,
                intensity_range = "absolute",
                **kwargs,
            )

        self.setcal()

        if returncalc:
            return qx0_fit,qy0_fit,qx0_residuals,qy0_residuals

    def fit_p_ellipse(
        self,
        bvm,
        center,
        fitradii,
        mask=None,
        returncalc = False,
        **kwargs
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
        p_ellipse = fit_ellipse_1D(
            bvm,
            center,
            fitradii,
            mask
        )

        scaling = kwargs.get("scaling", "log")
        kwargs.pop("scaling", None)
        from py4DSTEM.visualize import show_elliptical_fit
        show_elliptical_fit(
            bvm,
            fitradii,
            p_ellipse,
            scaling = scaling,
            **kwargs
        )

        self.calibration.set_p_ellipse(p_ellipse)
        self.setcal()

        if returncalc:
            return p_ellipse


    # Deprecated??

    # Lattice vectors
    def choose_lattice_vectors(
        self,
        index_g0,
        index_g1,
        index_g2,
        mode = 'centered',
        plot = True,
        subpixel = 'multicorr',
        upsample_factor = 16,
        sigma=0,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=10,
        bvm_vis_params = {},
        returncalc = False,
        ):
        """
        Choose which lattice vectors to use for strain mapping.

        Args:
            index_g0 (int): origin
            index_g1 (int): second point of vector 1
            index_g2 (int): second point of vector 2
            mode (str): centered or raw bragg map
            plot (bool): plot bragg vector maps and vectors
            subpixel (str): specifies the subpixel resolution algorithm to use.
                must be in ('pixel','poly','multicorr'), which correspond
                to pixel resolution, subpixel resolution by fitting a
                parabola, and subpixel resultion by Fourier upsampling.
            upsample_factor: the upsampling factor for the 'multicorr'
                algorithm
            sigma: if >0, applies a gaussian filter
            maxNumPeaks: the maximum number of maxima to return
            minAbsoluteIntensity, minRelativeIntensity, relativeToPeak,
                minSpacing, edgeBoundary, maxNumPeaks: filtering applied
                after maximum detection and before subpixel refinement
        """
        from py4DSTEM.process.utils import get_maxima_2D

        if mode == "centered":
            bvm = self.bvm_centered
        else:
            bvm = self.bvm_raw

        g = get_maxima_2D(
            bvm,
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            sigma = sigma,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minSpacing = minSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
        )

        self.g = g

        from py4DSTEM.visualize import select_lattice_vectors
        g1,g2 = select_lattice_vectors(
            bvm,
            gx = g['x'],
            gy = g['y'],
            i0 = index_g0,
            i1 = index_g1,
            i2 = index_g2,
            **bvm_vis_params,
        )

        self.g1 = g1
        self.g2 = g2

        if returncalc:
            return g1, g2

    def index_bragg_directions(
        self,
        x0 = None,
        y0 = None,
        plot = True,
        bvm_vis_params = {},
        returncalc = False,
        ):
        """
        From an origin (x0,y0), a set of reciprocal lattice vectors gx,gy, and an pair of
        lattice vectors g1=(g1x,g1y), g2=(g2x,g2y), find the indices (h,k) of all the
        reciprocal lattice directions.

        Args:
            x0 (float): x-coord of origin
            y0 (float): y-coord of origin
            Plot (bool): plot results
        """

        if x0 is None:
            x0 = self.Qshape[0]/2
        if y0 is None:
            y0 = self.Qshape[0]/2

        from py4DSTEM.process.latticevectors import index_bragg_directions
        _, _, braggdirections = index_bragg_directions(
            x0,
            y0,
            self.g['x'],
            self.g['y'],
            self.g1,
            self.g2
        )

        self.braggdirections = braggdirections

        if plot:
            from py4DSTEM.visualize import show_bragg_indexing
            show_bragg_indexing(
                self.bvm_centered,
                **bvm_vis_params,
                braggdirections = braggdirections,
                points = True
            )

        if returncalc:
            return braggdirections



    def add_indices_to_braggpeaks(
        self,
        maxPeakSpacing,
        mask = None,
        returncalc = False,
        ):
        """
        Using the peak positions (qx,qy) and indices (h,k) in the PointList lattice,
        identify the indices for each peak in the PointListArray braggpeaks.
        Return a new braggpeaks_indexed PointListArray, containing a copy of braggpeaks plus
        three additional data columns -- 'h','k', and 'index_mask' -- specifying the peak
        indices with the ints (h,k) and indicating whether the peak was successfully indexed
        or not with the bool index_mask. If `mask` is specified, only the locations where
        mask is True are indexed.

        Args:
            maxPeakSpacing (float): Maximum distance from the ideal lattice points
                to include a peak for indexing
            qx_shift,qy_shift (number): the shift of the origin in the `lattice` PointList
                relative to the `braggpeaks` PointListArray
            mask (bool): Boolean mask, same shape as the pointlistarray, indicating which
                locations should be indexed. This can be used to index different regions of
                the scan with different lattices
        """
        from py4DSTEM.process.latticevectors import add_indices_to_braggpeaks

        bragg_peaks_indexed = add_indices_to_braggpeaks(
            self.vectors,
            self.braggdirections,
            maxPeakSpacing = maxPeakSpacing,
            qx_shift = self.Qshape[0]/2,
            qy_shift = self.Qshape[1]/2,
        )

        self.bragg_peaks_indexed = bragg_peaks_indexed

        if returncalc:
            return bragg_peaks_indexed


    def fit_lattice_vectors_all_DPs(self, returncalc = False):
        """
        Fits lattice vectors g1,g2 to each diffraction pattern in braggpeaks, given some
        known (h,k) indexing.


        """

        from py4DSTEM.process.latticevectors import fit_lattice_vectors_all_DPs
        g1g2_map = fit_lattice_vectors_all_DPs(self.bragg_peaks_indexed)
        self.g1g2_map = g1g2_map
        if returncalc:
            return g1g2_map

    def get_strain_from_reference_region(self, mask, returncalc = False):
        """
        Gets a strain map from the reference region of real space specified by mask and the
        lattice vector map g1g2_map.

        Args:
            mask (ndarray of bools): use lattice vectors from g1g2_map scan positions
                wherever mask==True

        """
        from py4DSTEM.process.latticevectors import get_strain_from_reference_region

        strainmap_median_g1g2 = get_strain_from_reference_region(
            self.g1g2_map,
            mask = mask,
        )

        self.strainmap_median_g1g2 = strainmap_median_g1g2

        if returncalc:
            return strainmap_median_g1g2


    def get_strain_from_reference_g1g2(self, mask, returncalc = False):
        """
        Gets a strain map from the reference lattice vectors g1,g2 and lattice vector map
        g1g2_map.


        Args:
            mask (ndarray of bools): use lattice vectors from g1g2_map scan positions
                wherever mask==True

        """
        from py4DSTEM.process.latticevectors import get_reference_g1g2
        g1_ref,g2_ref = get_reference_g1g2(self.g1g2_map, mask)

        from py4DSTEM.process.latticevectors import get_strain_from_reference_g1g2
        strainmap_reference_g1g2 = get_strain_from_reference_g1g2(self.g1g2_map, g1_ref, g2_ref)

        self.strainmap_reference_g1g2 = strainmap_reference_g1g2

        if returncalc:
            return strainmap_reference_g1g2

    def get_rotated_strain_map(self, mode, g_reference = None, returncalc = True, flip_theta = False):
        """
        Starting from a strain map defined with respect to the xy coordinate system of
        diffraction space, i.e. where exx and eyy are the compression/tension along the Qx
        and Qy directions, respectively, get a strain map defined with respect to some other
        right-handed coordinate system, in which the x-axis is oriented along (xaxis_x,
        xaxis_y).

        Args:
            g_referencce (tupe): reference coordinate system for xaxis_x and xaxis_y
        """

        assert mode in ("median","reference")
        if g_reference is None:
            g_reference = np.subtract(self.g1, self.g2)

        from py4DSTEM.process.latticevectors import get_rotated_strain_map

        if mode == "median":
            strainmap_raw = self.strainmap_median_g1g2
        elif mode == "reference":
            strainmap_raw = self.strainmap_reference_g1g2

        strainmap = get_rotated_strain_map(
            strainmap_raw,
            xaxis_x = g_reference[0],
            xaxis_y = g_reference[1],
            flip_theta = flip_theta
        )

        if returncalc:
            return strainmap



    def get_masked_peaks(
        self,
        mask,
        update_inplace = False,
        returncalc = True):
        """
        Removes all bragg peaks which fall inside `mask` in the raw
        (uncalibrated) positions.

        Args:
            mask (bool): binary image where peaks will be deleted
            update_inplace (bool): if True, removes peaks from this
                BraggVectors instance. If False, returns a new
                BraggVectors instance with the requested peaks removed
            returncalc (bool): if True, return the BraggVectors

        Returns:
            (BraggVectors or None)
        """

        # Copy peaks
        v = self._v_uncal.copy( name='_v_uncal' )

        # Loop over all peaks
        for rx in range(v.shape[0]):
            for ry in range(v.shape[1]):
                p = v.get_pointlist(rx,ry)
                sub = mask.ravel()[np.ravel_multi_index((
                    np.round(p.data["qx"]).astype('int'),
                    np.round(p.data["qy"]).astype('int')),
                    self.Qshape)]
                p.remove(sub)

        # if modifying this BraggVectors instance was requested
        if update_inplace:
            self._v_uncal = v
            ans = self

        # if a new instance was requested
        else:
            ans = self.copy( name=self.name+'_masked' )
            ans._v_uncal = v

        # re-calibrate
        ans.calibrate()

        # return
        if returncalc:
            return ans
        else:
            return




######### END BraggVectorMethods CLASS ########



class BraggVectorMap(Array):

    def __init__(
        self,
        name,
        data,
        weights,
        dims,
        dim_units,
        origin,
        pixelsize
    ):
        Array.__init__(
            self,
            name = name,
            data = data,
            dims = dims,
            dim_units = [dim_units,dim_units],
        )
        self.metadata = Metadata(
            name = 'grid',
            data = {
                'origin' : origin,
                'pixelsize' : pixelsize,
                'weights' : weights
            }
        )

    @property
    def origin(self):
        return self.metadata['grid']['origin']
    @property
    def pixelsize(self):
        return self.metadata['grid']['pixelsize']
    @property
    def pixelunits(self):
        return self.dim_units[0]
    @property
    def weights(self):
        return self.metadata['grid']['weights']


    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        constr_args = Array._get_constructor_args(group)
        metadata = _read_metadata(group,'grid')
        args = {
            'name' : constr_args['name'],
            'data' : constr_args['data'],
            'weights' : metadata['weights'],
            'dims' : constr_args['dims'],
            'dim_units' : constr_args['dim_units'],
            'origin' : metadata['origin'],
            'pixelsize' : metadata['pixelsize']
        }
        return args



