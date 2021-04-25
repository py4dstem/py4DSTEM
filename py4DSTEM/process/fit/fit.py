# Fitting

import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A*np.exp(-0.5*((x-mu)/sigma)**2)

def fit_1D_gaussian(xdata,ydata,xmin,xmax):
    """
    Fits a 1D gaussian to the subset of the 1D curve f(xdata)=ydata within the window (xmin,xmax).
    Returns A,mu,sigma.  Retrieve the full curve with
        fit_gaussian = py4DSTEM.process.fit.gaussian(xdata,A,mu,sigma)
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

def fit_2D(function, data, data_mask=None, return_ar=True, popt_guess=None, 
    robust=False, robust_steps=3, robust_thresh=2):
    """
    Performs a 2D fit, where the fit function takes its first input in the form of a length 2
    vector (ndarray) of (x,y) positions, followed by the remaining parameters, and the data to
    fit takes the form of an (n,m) shaped array. If robust fitting is specified, regression is 
    repeated using 

    Inputs:
        function     - First input should be a length 2 array xy, where (xy[0],xy[1]) are the (x,y)
                       coordinates
        data         - Data to fit, in an (n,m) shaped ndarray
        data_mask    - Optional parameter. If specified, must be a boolean array of the same shape
                       as data, specifying which elements of data to use in the fit
        return_ar    - Optional parameter. If False, only the fit parameters and covariance matrix
                       are returned. If True, return an array  of the same shape as data with the
                       fit values. Defaults to True
        popt_guess   - Optional parameter. If specified, should be a tuple of initial guesses for
                       the fit parameters.
        robust       - Optional parameter. If set to True, fit will be repeated with outliers removed.
        robust_steps - Optional parameter. Number of robust iterations performed after initial fit.
        robust_thresh- Optional parameter. Number of robust iterations performed after initial fit.

    Outputs:
        popt      - optimal fit parameters to function
        pcov      - the covariance matrix
        fit_ar    - optional. If return_ar==True, fit_ar is returned, and is an array of the
                    same shape as data, containing the fit values
    """
    shape = data.shape
    shape1D = [1,np.prod(shape)]
    x,y = np.arange(shape[0]),np.arange(shape[1])
    ry,rx = np.meshgrid(y,x)
    rx_1D = rx.reshape((1,np.prod(shape)))
    ry_1D = ry.reshape((1,np.prod(shape)))
    xy = np.vstack((rx_1D, ry_1D))

    # if robust fitting is turned off, set number of robust iterations to 0
    if robust==False:
        robust_steps=0

    # initial mask
    if data_mask is not None:
        mask = data_mask
    else:
        mask = np.ones(shape,dtype=bool)

    #  least squares fitting
    if popt_guess is not None:
        popt, pcov = curve_fit(function, 
            np.vstack((rx_1D[mask.reshape(shape1D)],ry_1D[mask.reshape(shape1D)])), 
            data[mask], 
            p0=popt_guess)
    else:
        popt, pcov = curve_fit(function,
            np.vstack((rx_1D[mask.reshape(shape1D)],ry_1D[mask.reshape(shape1D)])), 
            data[mask])

    # repeat fitting, with values beyond error threshold removed
    for x in range(robust_steps):
        fit_mean_square_error = (function(xy,*popt).reshape(shape) - data)**2
        mask = fit_mean_square_error <= np.mean(fit_mean_square_error) * robust_thresh**2
        # include user-specified mask if provided
        if data_mask is not None:
            mask[data_mask==False] = False
        # updated fit
        popt, pcov = curve_fit(function, 
            np.vstack((rx_1D[mask.reshape(shape1D)],ry_1D[mask.reshape(shape1D)])), 
            data[mask], 
            p0=popt)

    if return_ar:
        fit_ar = function(xy,*popt).reshape(shape)
        return popt, pcov, fit_ar
    else:
        return popt, pcov
