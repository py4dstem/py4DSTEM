# BraggVectors methods

import numpy as np
from ..emd import Metadata


# Bragg vector maps

def get_bvm(
    self,
    mode = 'centered',
    ):
    """
    Gets a Bragg vector map, a 2D histogram of Bragg scattering vectors.

    Args:
        Qshape (2 tuple): diffraction space shape
        mode (str): must be 'raw' or 'centered'. TODO, sampling selection

    Args:
        (2D array):
    """
    assert mode in ('raw','centered')

    # select peaks
    peaks = self.vectors if mode=='centered' else self.vectors_uncal

    # perform computation
    from ....process.diskdetection import get_bvm
    bvm = get_bvm(
        peaks,
        Qshape = self.Qshape,
        mode = mode,
    )
    if mode == 'centered':
        self.bvm_centered = bvm
    else:
        self.bvm_raw = bvm

    return bvm




def measure_origin(
    self,
    mode,
    returncalc = True,
    **kwargs,
    ):
    """
    Valid `mode` arguments are "beamstop" and "no_beamstop".
    Use-cases and input arguments:

    "no_beamstop" - A set of bragg peaks for data with no beamstop, and in which
        the center beam is brightest throughout. No required kwargs, optional
        kwargs are any accepted by process.calibration.origin.
        get_origin_from_braggpeaks.

    "beamstop" - A set of bragg peaks for data with a beamstop. Req'd kwargs
        are `center_guess` (2-tuple) and `radii` (2-tuple) specifying an annular
        region in which to search for conjugate pairs of Bragg peaks to use
        for calibrating. Optional kwargs are those accepted by
        process.calibration.origin.get_origin_beamstop_braggpeaks.

        Args:
            center_guess (2-tuple)
            radii   (2-tuple)

    """
    from ....process.calibration import measure_origin
    assert mode in ("beamstop", "no_beamstop")

    mode = "bragg_" + mode
    kwargs["Q_shape"] = self.Qshape

    # perform computation
    origin = measure_origin(
        self.vectors_uncal,
        mode = mode,
        **kwargs
    )

    # try to add to calibration
    try:
        self.calibration.set_origin_meas(origin)
    except AttributeError:
        # should a warning be raised?
        pass

    if returncalc:
        return origin


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
    fit_vis_params = None,
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

    from ....process.calibration import fit_origin
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
        from ....visualize import show_image_grid
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

        if fit_vis_params is None:
            fit_vis_params = {
                'H':2,
                'W':3,
                'cmap':'RdBu',
                'clipvals':'manual',
                'vmin':-1*plot_range,
                'vmax':1*plot_range,
                'axsize':(6,2),
                }

        show_image_grid(
            lambda i:[qx0_meas-qx0_mean,qx0_fit-qx0_mean,qx0_res_plot,
                      qy0_meas-qy0_mean,qy0_fit-qy0_mean,qy0_res_plot][i],
            **fit_vis_params
        )

    if returncalc:
        return qx0_fit,qy0_fit,qx0_residuals,qy0_residuals

# Calibrate
def calibrate(
    self,
    use_fitted_origin = True,
    returncalc = False,
    ):
    """
    Determines which calibrations are present in set.calibrations (of origin,
    elliptical, pixel, rotational), and applies any it finds to self.v_uncal,
    storing the output in self.v.

    Args:
        use_fitted_origin (bool): determine if using fitted origin or measured origin
    Returns:
        (PointListArray)
    """
    try:
        cal = self.calibration
    except AttributeError:
        raise Exception('No .calibration attribute found')

    from ....process.calibration.braggvectors import calibrate

    v = self.vectors_uncal.copy( name='_v_cal' )
    v = calibrate(
        v,
        cal,
        use_fitted_origin,
    )
    self._v_cal = v

    if returncalc:
        return v

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
    from ....process.utils import get_maxima_2D

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

    from ....visualize import select_lattice_vectors
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

    from ....process.latticevectors import index_bragg_directions
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
        from ....visualize import show_bragg_indexing
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
    from ....process.latticevectors import add_indices_to_braggpeaks

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

    from ....process.latticevectors import fit_lattice_vectors_all_DPs
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
    from ....process.latticevectors import get_strain_from_reference_region

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
    from ....process.latticevectors import get_reference_g1g2
    g1_ref,g2_ref = get_reference_g1g2(self.g1g2_map, mask)

    from ....process.latticevectors import get_strain_from_reference_g1g2
    strainmap_reference_g1g2 = get_strain_from_reference_g1g2(self.g1g2_map, g1_ref, g2_ref)

    self.strainmap_reference_g1g2 = strainmap_reference_g1g2

    if returncalc:
        return strainmap_reference_g1g2

def get_rotated_strain_map(self, mode, g_reference = None, returncalc = True):
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

    from ....process.latticevectors import get_rotated_strain_map

    if mode == "median":
        strainmap_raw = self.strainmap_median_g1g2
    elif mode == "reference":
        strainmap_raw = self.strainmap_reference_g1g2

    strainmap = get_rotated_strain_map(
        strainmap_raw,
        xaxis_x = g_reference[0],
        xaxis_y = g_reference[1],
    )

    if returncalc:
        return strainmap



def get_masked_peaks(
    self, 
    mask,
    update_inplace = False,
    returncalc = True):
    """
    This function applys a mask to bragg peaks .

    
    Args:
        mask (bool):    binary image where peaks will be deleted
        

    Returns:

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

    if update_inplace:
        self._v_uncal = v

        if returncalc: 
            return self

    if returncalc: 
        bragg_vector_update = self.copy()
        bragg_vector_update._v_uncal = v

        return bragg_vector_update


