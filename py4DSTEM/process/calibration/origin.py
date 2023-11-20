# Find the origin of diffraction space

import functools
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq

from emdfile import tqdmnd, PointListArray
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.calibration.probe import get_probe_size
from py4DSTEM.process.fit import plane, parabola, bezier_two, fit_2D
from py4DSTEM.process.utils import get_CoM, add_to_2D_array_from_floats, get_maxima_2D


#
# # origin setting decorators
#
# def set_measured_origin(fun):
#     """
#     This is intended as a decorator function to wrap other functions which measure
#     the position of the origin.  If some function `get_the_origin` returns the
#     position of the origin as a tuple of two (R_Nx,R_Ny)-shaped arrays, then
#     decorating the function definition like
#
#         >>> @measure_origin
#         >>> def get_the_origin(...):
#
#     will make the function also save those arrays as the measured origin in the
#     calibration associated with the data used for the measurement. Any existing
#     measured origin value will be overwritten.
#
#     For the wrapper to work, the decorated function's first argument must have
#     a .calibration property, and its first two return values must be qx0,qy0.
#     """
#     @functools.wraps(fun)
#     def wrapper(*args,**kwargs):
#         ans = fun(*args,**kwargs)
#         data = args[0]
#         cali = data.calibration
#         cali.set_origin_meas((ans[0],ans[1]))
#         return ans
#     return wrapper
#
#
# def set_fit_origin(fun):
#     """
#     See docstring for `set_measured_origin`
#     """
#     @functools.wraps(fun)
#     def wrapper(*args,**kwargs):
#         ans = fun(*args,**kwargs)
#         data = args[0]
#         cali = data.calibration
#         cali.set_origin((ans[0],ans[1]))
#         return ans
#     return wrapper
#


# fit the origin


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
    given some 2D arrays (qx0_meas,qy0_meas) of measured center positions,
    optionally masked by the Boolean array `mask`. The 2D data arrays may be
    passed directly as a 2-tuple to the arg `data`, or, if `data` is either a
    DataCube or Calibration instance, they will be retreived automatically. If a
    DataCube or Calibration are passed, fitted origin and residuals are stored
    there directly.

    Args:
        data (2-tuple of 2d arrays): the measured origin position (qx0,qy0)
        mask (2b boolean array, optional): ignore points where mask=False
        fitfunction (str, optional): must be 'plane' or 'parabola' or 'bezier_two'
            or 'constant'
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
    assert isinstance(data, tuple) and len(data) == 2
    qx0_meas, qy0_meas = data
    assert isinstance(qx0_meas, np.ndarray) and len(qx0_meas.shape) == 2
    assert isinstance(qx0_meas, np.ndarray) and len(qy0_meas.shape) == 2
    assert qx0_meas.shape == qy0_meas.shape
    assert mask is None or mask.shape == qx0_meas.shape and mask.dtype == bool
    assert fitfunction in ("plane", "parabola", "bezier_two", "constant")
    if fitfunction == "constant":
        qx0_fit = np.mean(qx0_meas) * np.ones_like(qx0_meas)
        qy0_fit = np.mean(qy0_meas) * np.ones_like(qy0_meas)
    else:
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
            popt_x, pcov_x, qx0_fit, _ = fit_2D(
                f,
                qx0_meas,
                robust=robust,
                robust_steps=robust_steps,
                robust_thresh=robust_thresh,
            )
            popt_y, pcov_y, qy0_fit, _ = fit_2D(
                f,
                qy0_meas,
                robust=robust,
                robust_steps=robust_steps,
                robust_thresh=robust_thresh,
            )

        else:
            popt_x, pcov_x, qx0_fit, _ = fit_2D(
                f,
                qx0_meas,
                robust=robust,
                robust_steps=robust_steps,
                robust_thresh=robust_thresh,
                data_mask=mask == True,
            )
            popt_y, pcov_y, qy0_fit, _ = fit_2D(
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
        return ans, (popt_x, popt_y, pcov_x, pcov_y)
    else:
        return ans


### Functions for finding the origin

# for a diffraction pattern


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


# for a datacube


def get_origin(
    datacube,
    r=None,
    rscale=1.2,
    dp_max=None,
    mask=None,
    fast_center=False,
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
        fast_center: (bool)
            Skip the center of mass refinement step.

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
        for rx, ry in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            dp = datacube.data[rx, ry, :, :]
            _qx0, _qy0 = np.unravel_index(
                np.argmax(gaussian_filter(dp, r, mode="nearest")),
                (datacube.Q_Nx, datacube.Q_Ny),
            )
            if fast_center:
                qx0[rx, ry], qy0[rx, ry] = _qx0, _qy0
            else:
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
        for rx, ry in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding origins",
            unit="DP",
            unit_scale=True,
        ):
            if mask[rx, ry]:
                dp = datacube.data[rx, ry, :, :]
                _qx0, _qy0 = np.unravel_index(
                    np.argmax(gaussian_filter(dp, r, mode="nearest")),
                    (datacube.Q_Nx, datacube.Q_Ny),
                )
                if fast_center:
                    qx0[rx, ry], qy0[rx, ry] = _qx0, _qy0
                else:
                    _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
                    qx0.data[rx, ry], qy0.data[rx, ry] = get_CoM(dp * _mask)
            else:
                qx0.mask, qy0.mask = True, True

    # return
    mask = np.ones(datacube.Rshape, dtype=bool)
    return qx0, qy0, mask


def get_origin_single_dp_beamstop(DP: np.ndarray, mask: np.ndarray, **kwargs):
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

        qx0[rx, ry] = x
        qy0[rx, ry] = y

    return qx0, qy0
