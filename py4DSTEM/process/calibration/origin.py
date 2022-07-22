# Find the origin of diffraction space

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq

from .probe import get_probe_size
from ..fit import plane,parabola,bezier_two,fit_2D
from ..utils import get_CoM, add_to_2D_array_from_floats, get_maxima_2D
from ...utils.tqdmnd import tqdmnd
from ...io.datastructure import PointListArray, DataCube


def measure_origin(
        data,
        mode,
        **kwargs
    ):
    """
    Options for the `mode` argument, their uses-cases, and their expected additional input arguments are:

    "dc_no_beamstop" - A datacube with no beamstop, and in which the center beam
        is brightest throughout.

        Args:
            data (DataCube)

    "bragg_no_beamstop" - A set of bragg peaks for data with no beamstop, and in which
        the center beam is brightest throughout.

        Args:
            data (PointListArray)
            Q_shape (Qx, Qy) from braggvector

    "dc_beamstop" - A datacube with a beamstop

        Args:
            data (DataCube)
            mask (2d array)

    "bragg_beamstop" - A set of bragg peaks for data with a beamstop

        Args:
            data (PointListArray)
            center_guess (2-tuple)
            radii   (2-tuple)
            Q_Nx (int)
            Q_Ny (int)


    Returns:
        (3 real space shaped arrays) qx0, qy0, mask
    """
    # parse args
    modes = (
        "dc_no_beamstop",
        "bragg_no_beamstop",
        "dc_beamstop",
        "bragg_beamstop",
    )
    assert mode in modes, f"{mode} must be in {modes}"

    # select a fn
    fn_dict = {
        "dc_no_beamstop" : get_origin,
        "dc_beamstop" : get_origin_beamstop,
        "bragg_no_beamstop" : get_origin_from_braggpeaks,
        "bragg_beamstop" : get_origin_beamstop_braggpeaks,
    }
    fn = fn_dict[mode]

    # run
    ans = fn(
        data,
        **kwargs
    )

    # return
    return ans








### Functions for finding the origin

def get_origin_single_dp(dp, r, rscale=1.2):
    """
    Find the origin for a single diffraction pattern, assuming (a) there is no beam stop,
    and (b) the center beam contains the highest intensity.

    Args:
        dp (ndarray): the diffraction pattern
        r (number): the approximate disk radius
        rscale (number): factor by which `r` is scaled to generate a mask

    Returns:
        (2-tuple): The origin
    """
    Q_Nx, Q_Ny = dp.shape
    _qx0, _qy0 = np.unravel_index(np.argmax(gaussian_filter(dp, r)), (Q_Nx, Q_Ny))
    qyy, qxx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
    mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
    qx0, qy0 = get_CoM(dp * mask)
    return qx0, qy0


def get_origin(
    datacube,
    r=None,
    rscale=1.2,
    dp_max=None,
    mask=None
    ):
    """
    Find the origin for all diffraction patterns in a datacube, assuming (a) there is no
    beam stop, and (b) the center beam contains the highest intensity. Stores the origin
    positions in the Calibration associated with datacube, and optionally also returns
    them.

    Args:
        datacube (DataCube): the data
        r (number or None): the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (number): expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        dp_max (ndarray or None): the diffraction pattern or dp-shaped array used to
            compute the center disk radius, if r is left unspecified. Behavior depends
            on type:

                * if ``dp_max==None`` (default), computes and uses the maximal
                  diffraction pattern. Note that for a large datacube, this may be a
                  slow operation.
                * otherwise, this should be a (Q_Nx,Q_Ny) shaped array
        mask (ndarray or None): if not None, should be an (R_Nx,R_Ny) shaped
                    boolean array. Origin is found only where mask==True, and masked
                    arrays are returned for qx0,qy0

    Returns:
        (2-tuple of (R_Nx,R_Ny)-shaped ndarrays): the origin, (x,y) at each scan position
    """
    if r is None:
        if dp_max is None:
            dp_max = np.max(datacube.data, axis=(0, 1))
        else:
            assert dp_max.shape == (datacube.Q_Nx, datacube.Q_Ny)
        r, _, _ = get_probe_size(dp_max)

    qx0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
    qy0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
    qyy, qxx = np.meshgrid(np.arange(datacube.Q_Ny), np.arange(datacube.Q_Nx))

    if mask is None:
        for (rx, ry) in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            dp = datacube.data[rx, ry, :, :]
            _qx0, _qy0 = np.unravel_index(
                np.argmax(gaussian_filter(dp, r)), (datacube.Q_Nx, datacube.Q_Ny)
            )
            _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
            qx0[rx, ry], qy0[rx, ry] = get_CoM(dp * _mask)

    else:
        assert mask.shape == (datacube.R_Nx, datacube.R_Ny)
        assert mask.dtype == bool
        qx0 = np.ma.array(
            data=qx0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
        )
        qy0 = np.ma.array(
            data=qy0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
        )
        for (rx, ry) in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            if mask[rx, ry]:
                dp = datacube.data[rx, ry, :, :]
                _qx0, _qy0 = np.unravel_index(
                    np.argmax(gaussian_filter(dp, r)), (datacube.Q_Nx, datacube.Q_Ny)
                )
                _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
                qx0.data[rx, ry], qy0.data[rx, ry] = get_CoM(dp * _mask)
            else:
                qx0.mask, qy0.mask = True, True

    # return
    mask = np.ones(datacube.Rshape, dtype=bool)
    return qx0, qy0, mask


def get_origin_from_braggpeaks(
    braggpeaks,
    Q_shape,
    center_guess = None,
    score_method = None,
    findcenter="CoM",
    bvm=None,
    **kwargs
):
    """
    Gets the diffraction shifts using detected Bragg disk positions.

    First, a guess at the unscattered beam position is determined, either by taking the
    CoM of the Bragg vector map, or by taking its maximal pixel.  If the CoM is used, an
    additional refinement step is used where we take the CoM of a Bragg vector map
    contructed from a first guess at the central Bragg peaks (as opposed to the BVM of all
    BPs). Once a unscattered beam position is determined, the Bragg peak closest to this
    position is identified. The shifts in these peaks positions from their average are
    returned as the diffraction shifts.

    Args:
        braggpeaks (PointListArray): the Bragg peak positions
        Q_shape (tuple of ints): the shape of diffration space
        center_guess (tuple of ints):   initial guess for the center
        score_method (string):     Method used to find center peak
        findcenter (str): specifies the method for determining the unscattered beam
            position options: 'CoM', or 'max'
        bvm (array or None): the braggvector map. If None (default), the bvm is
            calculated

    Returns:
        (3-tuple): A 3-tuple comprised of:

            * **qx0** *((R_Nx,R_Ny)-shaped array)*: the origin x-coord
            * **qy0** *((R_Nx,R_Ny)-shaped array)*: the origin y-coord
            * **braggvectormap** *((R_Nx,R_Ny)-shaped array)*: the Bragg vector map of only
              the Bragg peaks identified with the unscattered beam. Useful for diagnostic
              purposes.
    """
    assert isinstance(braggpeaks, PointListArray), "braggpeaks must be a PointListArray"
    # assert all([isinstance(item, (int, np.integer)) for item in [Q_Nx, Q_Ny]])
    assert isinstance(findcenter, str), "center must be a str"
    assert findcenter in ["CoM", "max"], "center must be either 'CoM' or 'max'"
    assert score_method in ["distance", "intensity", "intensity weighted distance", None], "center must be either 'distance' or 'intensity weighted distance'"

    R_Nx, R_Ny = braggpeaks.shape
    Q_Nx, Q_Ny = Q_shape

    # Default scoring method
    if score_method is None:
        if center_guess is None:
            score_method = "intensity"
        else:
            score_method = "distance"


    # Get guess at position of unscattered beam (x0,y0)
    if center_guess is None:
        if bvm is None:
            from ..diskdetection.braggvectormap import get_bragg_vector_map_raw
            braggvectormap_all = get_bragg_vector_map_raw(braggpeaks, Q_Nx, Q_Ny)
        else:
            braggvectormap_all = bvm
        if findcenter == "max":
            x0, y0 = np.unravel_index(
                np.argmax(gaussian_filter(braggvectormap_all, 10)), (Q_Nx, Q_Ny)
            )
        else:
            x0, y0 = get_CoM(braggvectormap_all)
            braggvectormap = np.zeros_like(braggvectormap_all)
            for Rx in range(R_Nx):
                for Ry in range(R_Ny):
                    pointlist = braggpeaks.get_pointlist(Rx, Ry)
                    if pointlist.length > 0:
                        r2 = (pointlist.data["qx"] - x0) ** 2 + (
                            pointlist.data["qy"] - y0
                        ) ** 2
                        index = np.argmin(r2)
                        braggvectormap = add_to_2D_array_from_floats(
                            braggvectormap,
                            pointlist.data["qx"][index],
                            pointlist.data["qy"][index],
                            pointlist.data["intensity"][index],
                        )
            x0, y0 = get_CoM(braggvectormap)
    else:
        x0, y0 = center_guess

    # Get Bragg peak closest to unscattered beam at each scan position
    qx0 = np.zeros((R_Nx, R_Ny))
    qy0 = np.zeros((R_Nx, R_Ny))
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            if pointlist.length > 0:
                if score_method == "distance":
                    r2 = (pointlist.data["qx"] - x0) ** 2 + (pointlist.data["qy"] - y0) ** 2
                    index = np.argmin(r2)
                elif score_method == "intensity":
                    index = np.argmax(pointlist.data["intensity"])
                elif score_method == "intensity weighted distance":
                    r2 = pointlist.data["intensity"]/(1+((pointlist.data["qx"] - x0) ** 2 + (pointlist.data["qy"] - y0) ** 2))
                    index = np.argmax(r2)
                qx0[Rx, Ry] = pointlist.data["qx"][index]
                qy0[Rx, Ry] = pointlist.data["qy"][index]
            else:
                qx0[Rx, Ry] = x0
                qy0[Rx, Ry] = y0

    # return
    mask = np.ones(braggpeaks.shape, dtype=bool)
    return qx0, qy0, mask

def get_origin_single_dp_beamstop(DP: np.ndarray,mask: np.ndarray, **kwargs):
    """
    Find the origin for a single diffraction pattern, assuming there is a beam stop.

    Args:
        DP (np array): diffraction pattern
        mask (np array): boolean mask which is False under the beamstop and True
            in the diffraction pattern. One approach to generating this mask
            is to apply a suitable threshold on the average diffraction pattern
            and use binary opening/closing to remove and holes

    Returns:
        qx0, qy0 (tuple) measured center position of diffraction pattern
    """

    imCorr = np.real(
        np.fft.ifft2(
            np.fft.fft2(DP * mask)
            * np.conj(np.fft.fft2(np.rot90(DP, 2) * np.rot90(mask, 2)))
        )
    )

    xp, yp = np.unravel_index(np.argmax(imCorr), imCorr.shape)

    dx = ((xp + DP.shape[0] / 2) % DP.shape[0]) - DP.shape[0] / 2
    dy = ((yp + DP.shape[1] / 2) % DP.shape[1]) - DP.shape[1] / 2

    return (DP.shape[0] + dx) / 2, (DP.shape[1] + dy) / 2


def get_origin_beamstop(datacube: DataCube, mask: np.ndarray, **kwargs):
    """
    Find the origin for each diffraction pattern, assuming there is a beam stop.

    Args:
        datacube (DataCube)
        mask (np array): boolean mask which is False under the beamstop and True
            in the diffraction pattern. One approach to generating this mask
            is to apply a suitable threshold on the average diffraction pattern
            and use binary opening/closing to remove any holes

    Returns:
        qx0, qy0 (tuple of np arrays) measured center position of each diffraction pattern
    """

    qx0 = np.zeros(datacube.data.shape[:2])
    qy0 = np.zeros_like(qx0)

    for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny):
        x, y = get_origin_single_dp_beamstop(datacube.data[rx, ry, :, :], mask)

        qx0[rx,ry] = x
        qy0[rx,ry] = y

    return qx0, qy0

def get_origin_beamstop_braggpeaks(
    braggpeaks,
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
        braggpeaks (PointListArray):
        center_guess (2-tuple): qx0,qy0
        radii (2-tuple): the inner and outer radii of the specified annular region
        max_dist (number): the maximum allowed distance between the reflection of two
            peaks to consider them conjugate pairs
        max_iter (integer): for values >1, repeats the algorithm after updating center_guess

    Returns:
        (2d masked array): the origins
    """
    assert(isinstance(braggpeaks,PointListArray))
    R_Nx,R_Ny = braggpeaks.shape

    # remove peaks outside the annulus
    braggpeaks_masked = braggpeaks.copy()
    for rx in range(R_Nx):
        for ry in range(R_Ny):
            pl = braggpeaks_masked.get_pointlist(rx,ry)
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
                pl = braggpeaks_masked.get_pointlist(rx,ry)
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




### Functions for fitting the origin


def fit_origin(
    data,
    mask=None,
    fitfunction="plane",
    returnfitp=False,
    robust=False,
    robust_steps=3,
    robust_thresh=2,
):
    """
    Fits the position of the origin of diffraction space to a plane or parabola,
    given some 2D arrays (qx0_meas,qy0_meas) of measured center positions, optionally
    masked by the Boolean array `mask`. The 2D data arrays may be passed directly as
    a 2-tuple to the arg `data`, or, if `data` is either a DataCube or Calibration
    instance, they will be retreived automatically. If a DataCube or Calibration are
    passed, fitted origin and residuals are stored there directly.

    Args:
        data (2-tuple of 2d arrays): the measured origin position (qx0,qy0)
        mask (2b boolean array, optional): ignore points where mask=False
        fitfunction (str, optional): must be 'plane' or 'parabola' or 'bezier_two'
        returnfitp (bool, optional): if True, returns the fit parameters
        robust (bool, optional): If set to True, fit will be repeated with outliers
            removed.
        robust_steps (int, optional): Optional parameter. Number of robust iterations
                                performed after initial fit.
        robust_thresh (int, optional): Threshold for including points, in units of
            root-mean-square (standard deviations) error of the predicted values after
            fitting.

    Returns:
        (variable): Return value depends on returnfitp. If ``returnfitp==False``
        (default), returns a 4-tuple containing:

            * **qx0_fit**: *(ndarray)* the fit origin x-position
            * **qy0_fit**: *(ndarray)* the fit origin y-position
            * **qx0_residuals**: *(ndarray)* the x-position fit residuals
            * **qy0_residuals**: *(ndarray)* the y-position fit residuals

        If ``returnfitp==True``, returns a 2-tuple.  The first element is the 4-tuple
        described above.  The second element is a 4-tuple (popt_x,popt_y,pcov_x,pcov_y)
        giving fit parameters and covariance matrices with respect to the chosen
        fitting function.
    """
    assert isinstance(data,tuple) and len(data)==2
    qx0_meas,qy0_meas = data
    assert isinstance(qx0_meas, np.ndarray) and len(qx0_meas.shape) == 2
    assert isinstance(qx0_meas, np.ndarray) and len(qy0_meas.shape) == 2
    assert qx0_meas.shape == qy0_meas.shape
    assert mask is None or mask.shape == qx0_meas.shape and mask.dtype == bool
    assert fitfunction in ("plane", "parabola", "bezier_two")
    if fitfunction == "plane":
        f = plane
    elif fitfunction == "parabola":
        f = parabola
    elif fitfunction == "bezier_two":
        f = bezier_two
    else:
        raise Exception("Invalid fitfunction '{}'".format(fitfunction))

    # Check if mask for data is stored in (qx0_meax,qy0_meas) as a masked array
    if isinstance(qx0_meas, np.ma.MaskedArray):
        mask = np.ma.getmask(qx0_meas)

    # Fit data
    if mask is None:
        popt_x, pcov_x, qx0_fit = fit_2D(
            f,
            qx0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )
        popt_y, pcov_y, qy0_fit = fit_2D(
            f,
            qy0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )

    else:
        popt_x, pcov_x, qx0_fit = fit_2D(
            f,
            qx0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
            data_mask=mask == True,
        )
        popt_y, pcov_y, qy0_fit = fit_2D(
            f,
            qy0_meas,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
            data_mask=mask == True,
        )

    # Compute residuals
    qx0_residuals = qx0_meas - qx0_fit
    qy0_residuals = qy0_meas - qy0_fit

    # Return
    ans = (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals)
    if returnfitp:
        return ans,(popt_x,popt_y,pcov_x,pcov_y)
    else:
        return ans


### Older / soon-to-be-deprecated functions for finding the origin


def find_outlier_shifts(xshifts, yshifts, n_sigma=10, edge_boundary=0):
    """
    Finds outliers in the shift matrices.

    Gets a score function for each scan position Rx,Ry, given by the sum of the absolute values of
    the difference between the shifts at this position and all 8 NNs. Calculates a histogram of the
    scoring function, fits a gaussian to its initial peak, and sets a cutoff value to n_sigma times
    its standard deviation. Values beyond this cutoff are deemed outliers, as are scan positions
    within edge_boundary pixels of the edge of real space.

    Accepts:
        xshifts         ((R_Nx,R_Ny)-shaped array) the shifts in x
        yshifts         ((R_Nx,R_Ny)-shaped array) the shifts in y
        n_sigma         (float) the cutoff value for the score function, in number of std
        edge_boundary   (int) number of pixels near the mask edge to mark as outliers

    Returns:
        mask            ((R_nx,R_ny)-shaped array of bools) the outlier mask
        score           ((R_nx,R_ny)-shaped array) the outlier scores
        cutoff          (float) the score cutoff value
    """
    # Get score
    score = np.zeros_like(xshifts)
    score[:-1, :] += np.abs(
        xshifts[:-1, :] - np.roll(xshifts, (-1, 0), axis=(0, 1))[:-1, :]
    )
    score[1:, :] += np.abs(
        xshifts[1:, :] - np.roll(xshifts, (1, 0), axis=(0, 1))[1:, :]
    )
    score[:, :-1] += np.abs(
        xshifts[:, :-1] - np.roll(xshifts, (0, -1), axis=(0, 1))[:, :-1]
    )
    score[:, 1:] += np.abs(
        xshifts[:, 1:] - np.roll(xshifts, (0, 1), axis=(0, 1))[:, 1:]
    )
    score[:-1, :-1] += np.abs(
        xshifts[:-1, :-1] - np.roll(xshifts, (-1, -1), axis=(0, 1))[:-1, :-1]
    )
    score[1:, :-1] += np.abs(
        xshifts[1:, :-1] - np.roll(xshifts, (1, -1), axis=(0, 1))[1:, :-1]
    )
    score[:-1, 1:] += np.abs(
        xshifts[:-1, 1:] - np.roll(xshifts, (-1, 1), axis=(0, 1))[:-1, 1:]
    )
    score[1:, 1:] += np.abs(
        xshifts[1:, 1:] - np.roll(xshifts, (1, 1), axis=(0, 1))[1:, 1:]
    )
    score[:-1, :] += np.abs(
        yshifts[:-1, :] - np.roll(yshifts, (-1, 0), axis=(0, 1))[:-1, :]
    )
    score[1:, :] += np.abs(
        yshifts[1:, :] - np.roll(yshifts, (1, 0), axis=(0, 1))[1:, :]
    )
    score[:, :-1] += np.abs(
        yshifts[:, :-1] - np.roll(yshifts, (0, -1), axis=(0, 1))[:, :-1]
    )
    score[:, 1:] += np.abs(
        yshifts[:, 1:] - np.roll(yshifts, (0, 1), axis=(0, 1))[:, 1:]
    )
    score[:-1, :-1] += np.abs(
        yshifts[:-1, :-1] - np.roll(yshifts, (-1, -1), axis=(0, 1))[:-1, :-1]
    )
    score[1:, :-1] += np.abs(
        yshifts[1:, :-1] - np.roll(yshifts, (1, -1), axis=(0, 1))[1:, :-1]
    )
    score[:-1, 1:] += np.abs(
        yshifts[:-1, 1:] - np.roll(yshifts, (-1, 1), axis=(0, 1))[:-1, 1:]
    )
    score[1:, 1:] += np.abs(
        yshifts[1:, 1:] - np.roll(yshifts, (1, 1), axis=(0, 1))[1:, 1:]
    )
    score[1:-1, 1:-1] /= 8.0
    score[:1, 1:-1] /= 5.0
    score[-1:, 1:-1] /= 5.0
    score[1:-1, :1] /= 5.0
    score[1:-1, -1:] /= 5.0
    score[0, 0] /= 3.0
    score[0, -1] /= 3.0
    score[-1, 0] /= 3.0
    score[-1, -1] /= 3.0

    # Get mask and return
    cutoff = np.std(score) * n_sigma
    mask = score > cutoff
    if edge_boundary > 0:
        mask[:edge_boundary, :] = True
        mask[-edge_boundary:, :] = True
        mask[:, :edge_boundary] = True
        mask[:, -edge_boundary:] = True

    return mask, score, cutoff




