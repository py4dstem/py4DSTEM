# Fitting

import numpy as np
from scipy.optimize import curve_fit

def fit_2D(function, data, data_mask=None, return_ar=True, popt_guess=None):
    """
    Performs a 2D fit, where the fit function takes its first input in the form of a length 2
    vector (ndarray) of (x,y) positions, followed by the remaining parameters, and the data to
    fit takes the form of an (n,m) shaped array.

    Inputs:
        function  - First input should be a length 2 array xy, where (xy[0],xy[1]) are the (x,y)
                    coordinates
        data      - Data to fit, in an (n,m) shaped ndarray
        data_mask - Optional parameter. If specified, must be a boolean array of the same shape
                    as data, specifying which elements of data to use in the fit
        return_ar - Optional parameter. If False, only the fit parameters and covariance matrix
                    are returned. If True, return an array  of the same shape as data with the
                    fit values. Defaults to True
        popt_guess- Optional parameter. If specified, should be a tuple of initial guesses for
                    the fit parameters.

    Outputs:
        popt      - optimal fit parameters to function
        pcov      - the covariance matrix
        fit_ar    - optional. If return_ar==True, fit_ar is returned, and is an array of the
                    same shape as data, containing the fit values
    """
    shape = data.shape
    x,y = np.arange(shape[0]),np.arange(shape[1])
    ry,rx = np.meshgrid(y,x)
    rx_1D = rx.reshape((1,np.prod(shape)))
    ry_1D = ry.reshape((1,np.prod(shape)))
    xy = np.vstack((rx_1D, ry_1D))

    if data_mask is not None:
        rx_1D_known = rx_1D[data_mask.reshape((1,np.prod(shape)))]
        ry_1D_known = ry_1D[data_mask.reshape((1,np.prod(shape)))]
        xy_known = np.vstack((rx_1D_known,ry_1D_known))
        data_1D = data[data_mask]
        if popt_guess is not None:
            popt, pcov = curve_fit(function, xy_known, data_1D, p0=popt_guess)
        else:
            popt, pcov = curve_fit(function, xy_known, data_1D)
    else:
        data_1D = data.reshape(np.prod(shape))
        if popt_guess is not None:
            popt, pcov = curve_fit(function, xy, data_1D, p0=popt_guess)
        else:
            popt, pcov = curve_fit(function, xy, data_1D)

    if return_ar:
        fit_ar = function(xy,*popt).reshape(shape)
        return popt, pcov, fit_ar
    else:
        return popt, pcov

