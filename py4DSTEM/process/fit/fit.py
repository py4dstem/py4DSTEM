# Fitting

import numpy as np
from scipy.optimize import curve_fit
from inspect import signature

def gaussian(x, A, mu, sigma):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)

def fit_1D_gaussian(xdata,ydata,xmin,xmax):
    """
    Fits a 1D gaussian to the subset of the 1D curve f(xdata)=ydata within the window
    (xmin,xmax). Returns A,mu,sigma.  Retrieve the full curve with

        >>> fit_gaussian = py4DSTEM.process.fit.gaussian(xdata,A,mu,sigma)
    """
    mask = (xmin<=xdata)*(xmax>xdata)
    inds = np.nonzero(mask)[0]
    _xdata = xdata[inds]
    _ydata = ydata[inds]
    scale = np.max(_ydata)
    _ydata = _ydata/scale

    p0 = [np.max(_ydata),_xdata[np.argmax(_ydata)],(xmax-xmin)/8.]  # TODO: better guess for std
    popt,pcov = curve_fit(gaussian,_xdata,_ydata,p0=p0)
    A,mu,sigma = scale*popt[0],popt[1],popt[2]
    return A,mu,sigma

def fit_2D(function, data, data_mask=None, popt=None,
    robust=False, robust_steps=3, robust_thresh=2):
    """
    Performs a 2D fit, where the fit function takes its first input in the form of a
    length 2 vector (ndarray) of (x,y) positions, followed by the remaining parameters,
    and the data to fit takes the form of an (n,m) shaped array. Robust fitting can be
    enabled to iteratively reject outlier data points, which have a root-mean-square
    error beyond the user-specified threshold.

    Args:
        function: First input should be a length 2 array xy, where (xy[0],xy[1]) are the
            (x,y) coordinates
        data: Data to fit, in an (n,m) shaped ndarray
        data_mask:  Optional parameter. If specified, must be a boolean array of the same
            shape as data, specifying which elements of data to use in the fit
        return_ar: Optional parameter. If False, only the fit parameters and covariance
            matrix are returned. If True, return an array  of the same shape as data with
            the fit values. Defaults to True
        popt: Optional parameter for input. If specified, should be a tuple of initial
            guesses for the fit parameters.
        robust: Optional parameter. If set to True, fit will be repeated with outliers
            removed.
        robust_steps: Optional parameter. Number of robust iterations performed after
            initial fit.
        robust_thresh: Optional parameter. Threshold for including points, in units of
            root-mean-square (standard deviations) error of the predicted values after
            fitting.

    Returns:
        (3-tuple) A 3-tuple containing:

            * **popt**: optimal fit parameters to function
            * **pcov**: the covariance matrix
            * **fit_ar**: optional. If return_ar==True, fit_ar is returned, and is an
              array of the same shape as data, containing the fit values
    """
    shape = data.shape
    shape1D = [1,np.prod(shape)]
    # x and y coordinates normalized from 0 to 1
    x,y = np.linspace(0, 1, shape[0]),np.linspace(0, 1, shape[1])
    ry,rx = np.meshgrid(y,x)
    rx_1D = rx.reshape((1,np.prod(shape)))
    ry_1D = ry.reshape((1,np.prod(shape)))
    xy = np.vstack((rx_1D, ry_1D))

    # if robust fitting is turned off, set number of robust iterations to 0
    if robust==False:
        robust_steps=0

    # least squares fitting - 1st iteration 
    for k in range(robust_steps+1):
        if k == 0:
            if popt is None:
                popt = np.zeros((1,len(signature(function).parameters)-1))
            if data_mask is not None:
                mask = data_mask
            else:
                mask = np.ones(shape,dtype=bool)
        else:
            fit_mean_square_error = (function(xy,*popt).reshape(shape) - data)**2
            mask = fit_mean_square_error <= np.mean(fit_mean_square_error) * robust_thresh**2
            # include user-specified mask if provided
            if data_mask is not None:
                mask[data_mask==False] = False

        # perform fitting
        popt, pcov = curve_fit(function,
            np.vstack((rx_1D[mask.reshape(shape1D)],ry_1D[mask.reshape(shape1D)])),
            data[mask],
            p0=popt)

    fit_ar = function(xy,*popt).reshape(shape)
    return popt, pcov, fit_ar


# Functions for fitting

def plane(xy, mx, my, b):
    return mx*xy[0] + my*xy[1] + b

def parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return c0 + \
        cx1*xy[0] + cy1*xy[1] + \
        cx2*xy[0]**2 + cy2*xy[1]**2 + cxy*xy[0]*xy[1]

def bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):

    return \
        c00  *((1-xy[0])**2)  * ((1-xy[1])**2) + \
        c10*2*(1-xy[0])*xy[0] * ((1-xy[1])**2) + \
        c20  *(xy[0]**2)      * ((1-xy[1])**2) + \
        c01*2*((1-xy[0])**2)  * (1-xy[1])*xy[1] + \
        c11*4*(1-xy[0])*xy[0] * (1-xy[1])*xy[1] + \
        c21*2*(xy[0]**2)      * (1-xy[1])*xy[1] + \
        c02  *((1-xy[0])**2)  * (xy[1]**2) + \
        c12*2*(1-xy[0])*xy[0] * (xy[1]**2) + \
        c22  *(xy[0]**2)      * (xy[1]**2)



