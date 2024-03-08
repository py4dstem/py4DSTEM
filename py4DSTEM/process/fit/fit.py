# Fitting

import numpy as np
from scipy.optimize import curve_fit
from inspect import signature


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_1D_gaussian(xdata, ydata, xmin, xmax):
    """
    Fits a 1D gaussian to the subset of the 1D curve f(xdata)=ydata within the window
    (xmin,xmax). Returns A,mu,sigma.  Retrieve the full curve with

        >>> fit_gaussian = py4DSTEM.process.fit.gaussian(xdata,A,mu,sigma)
    """

    mask = (xmin <= xdata) * (xmax > xdata)
    inds = np.nonzero(mask)[0]
    _xdata = xdata[inds]
    _ydata = ydata[inds]
    scale = np.max(_ydata)
    _ydata = _ydata / scale

    p0 = [
        np.max(_ydata),
        _xdata[np.argmax(_ydata)],
        (xmax - xmin) / 8.0,
    ]  # TODO: better guess for std

    popt, pcov = curve_fit(gaussian, _xdata, _ydata, p0=p0)
    A, mu, sigma = scale * popt[0], popt[1], popt[2]
    return A, mu, sigma


def fit_2D(
    function,
    data,
    data_mask=None,
    popt=None,
    robust=False,
    robust_steps=3,
    robust_thresh=2,
):
    """
    Performs a 2D fit.

    TODO: make returning the mask optional

    Parameters
    ----------
    function : callable
        Some `function( xy, **p)` where `xy` is a length 2 vector (1D np array)
        specifying the pixel position (x,y), and `p` is the function parameters
    data : ndarray
        Some 2D array of any shape (n,m)
    data_mask : None or boolean array of shape (n,m), optional
        If specified, fits only the pixels in `data` where this array is True
    popt : dict
        Initial guess at the parameters `p` of `function`. Note that positions
        in pixels (i.e. the xy positions) are linearly scaled to the space [0,1]
    robust : bool
        Toggles robust fitting, which iteratively rejects outlier data points
        which have a root-mean-square error beyond `robust_thresh`
    robust_steps : int
        The number of robust fitting iterations to perform
    robust_thresh : int
        The robust fitting cutoff

    Returns:
    (popt,pcov,fit_at, mask) : 4-tuple
        The optimal fit parameters, the fitting covariance matrix, the
        the fit array with the returned `popt` params, and the mask
    """
    # get shape
    shape = data.shape
    shape1D = [1, np.prod(shape)]

    # x and y coordinates normalized from 0 to 1
    x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    ry, rx = np.meshgrid(y, x)
    rx_1D = rx.reshape((1, np.prod(shape)))
    ry_1D = ry.reshape((1, np.prod(shape)))
    xy = np.vstack((rx_1D, ry_1D))

    # if robust fitting is turned off, set number of robust iterations to 0
    if robust is False:
        robust_steps = 0

    # least squares fitting
    for k in range(robust_steps + 1):
        # in 1st iteration, set up params and mask
        if k == 0:
            if popt is None:
                popt = np.zeros((1, len(signature(function).parameters) - 1))
            if data_mask is not None:
                mask = data_mask
            else:
                mask = np.ones(shape, dtype=bool)

        # otherwise, get fitting error and add high error pixels to mask
        else:
            fit_mean_square_error = (function(xy, *popt).reshape(shape) - data) ** 2
            _mask = (
                fit_mean_square_error
                > np.mean(fit_mean_square_error) * robust_thresh**2
            )
            mask[_mask] = False

        # perform fitting
        popt, pcov = curve_fit(
            function,
            np.vstack((rx_1D[mask.reshape(shape1D)], ry_1D[mask.reshape(shape1D)])),
            data[mask],
            p0=popt,
        )

    fit_ar = function(xy, *popt).reshape(shape)
    return popt, pcov, fit_ar, mask


# Functions for fitting


def plane(xy, mx, my, b):
    return mx * xy[0] + my * xy[1] + b


def parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return (
        c0
        + cx1 * xy[0]
        + cy1 * xy[1]
        + cx2 * xy[0] ** 2
        + cy2 * xy[1] ** 2
        + cxy * xy[0] * xy[1]
    )


def bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    return (
        c00 * ((1 - xy[0]) ** 2) * ((1 - xy[1]) ** 2)
        + c10 * 2 * (1 - xy[0]) * xy[0] * ((1 - xy[1]) ** 2)
        + c20 * (xy[0] ** 2) * ((1 - xy[1]) ** 2)
        + c01 * 2 * ((1 - xy[0]) ** 2) * (1 - xy[1]) * xy[1]
        + c11 * 4 * (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1]
        + c21 * 2 * (xy[0] ** 2) * (1 - xy[1]) * xy[1]
        + c02 * ((1 - xy[0]) ** 2) * (xy[1] ** 2)
        + c12 * 2 * (1 - xy[0]) * xy[0] * (xy[1] ** 2)
        + c22 * (xy[0] ** 2) * (xy[1] ** 2)
    )


def polar_gaussian_2D(
    tq,
    I0,
    mu_t,
    mu_q,
    sigma_t,
    sigma_q,
    C,
):
    # unpack position
    t, q = tq
    # set theta value to its closest periodic reflection to mu_t
    # t = np.square(t-mu_t)
    # t2 = np.min(np.vstack([t,1-t]))
    t2 = np.square(t - mu_t)
    return (
        I0 * np.exp(-(t2 / (2 * sigma_t**2) + (q - mu_q) ** 2 / (2 * sigma_q**2))) + C
    )


def polar_twofold_gaussian_2D(
    tq,
    I0,
    mu_t,
    mu_q,
    sigma_t,
    sigma_q,
):
    # unpack position
    t, q = tq

    # theta periodicity
    dt = np.mod(t - mu_t + np.pi / 2, np.pi) - np.pi / 2

    # output intensity
    return I0 * np.exp(
        (dt**2 / (-2.0 * sigma_t**2)) + ((q - mu_q) ** 2 / (-2.0 * sigma_q**2))
    )


def polar_twofold_gaussian_2D_background(
    tq,
    I0,
    mu_t,
    mu_q,
    sigma_t,
    sigma_q,
    C,
):
    # unpack position
    t, q = tq

    # theta periodicity
    dt = np.mod(t - mu_t + np.pi / 2, np.pi) - np.pi / 2

    # output intensity
    return C + I0 * np.exp(
        (dt**2 / (-2.0 * sigma_t**2)) + ((q - mu_q) ** 2 / (-2.0 * sigma_q**2))
    )


def fit_2D_polar_gaussian(
    data,
    mask=None,
    p0=None,
    robust=False,
    robust_steps=3,
    robust_thresh=2,
    constant_background=False,
):
    """

    NOTE - this cannot work without using pixel coordinates - something is wrong in the workflow.


    Fits a 2D gaussian to the pixels in `data` which are set to True in `mask`.

    The gaussian is anisotropic and oriented along (t,q), centered at
    (mu_t,mu_q), has standard deviations (sigma_t,sigma_q), maximum of I0,
    and an optional constant offset of C, and is periodic in t.

        f(x,y) = I0 * exp( - (x-mu_x)^2/(2sig_x^2) + (y-mu_y)^2/(2sig_y^2) )
        or
        f(x,y) = I0 * exp( - (x-mu_x)^2/(2sig_x^2) + (y-mu_y)^2/(2sig_y^2) ) + C

    Parameters
    ----------
    data : 2d array
        the data to fit
    p0 : 6-tuple
        initial guess at fit parameters, (I0,mu_x,mu_y,sigma_x_sigma_y,C)
    mask : 2d boolean array
        ignore pixels where mask is False
    robust : bool
        toggle robust fitting
    robust_steps : int
        number of robust fit iterations
    robust_thresh : number
        the robust fitting threshold
    constant_background : bool
        whether or not to include constant background

    Returns
    -------
    (popt,pcov,fit_ar) : 3-tuple
        the optimal fit parameters, the covariance matrix, and the fit array
    """

    if constant_background:
        return fit_2D(
            polar_twofold_gaussian_2D_background,
            data=data,
            data_mask=mask,
            popt=p0,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )
    else:
        return fit_2D(
            polar_twofold_gaussian_2D,
            data=data,
            data_mask=mask,
            popt=p0,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )
