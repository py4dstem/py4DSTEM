# Preprocessing utility functions

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np


def bin2D(array, factor, dtype=np.float64):
    """
    Bin a 2D ndarray by binfactor.

    Args:
        array (2D numpy array):
        factor (int): the binning factor
        dtype (numpy dtype): datatype for binned array. default is numpy default for
            np.zeros()

    Returns:
        the binned array
    """
    x, y = array.shape
    binx, biny = x // factor, y // factor
    xx, yy = binx * factor, biny * factor

    # Make a binned array on the device
    binned_ar = np.zeros((binx, biny), dtype=dtype)
    array = array.astype(dtype)

    # Collect pixel sums into new bins
    for ix in range(factor):
        for iy in range(factor):
            binned_ar += array[0 + ix : xx + ix : factor, 0 + iy : yy + iy : factor]
    return binned_ar


def make_Fourier_coords2D(Nx, Ny, pixelSize=1):
    """
    Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if hasattr(pixelSize, "__len__"):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_x = pixelSize[0]
        pixelSize_y = pixelSize[1]
    else:
        pixelSize_x = pixelSize
        pixelSize_y = pixelSize

    qx = np.fft.fftfreq(Nx, pixelSize_x)
    qy = np.fft.fftfreq(Ny, pixelSize_y)
    qy, qx = np.meshgrid(qy, qx)
    return qx, qy


def get_shifted_ar(ar, xshift, yshift, periodic=True, bilinear=False, device="cpu"):
    """
        Shifts array ar by the shift vector (xshift,yshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Args:
            ar (float): input array
            xshift (float): shift along axis 0 (x) in pixels
            yshift (float): shift along axis 1 (y) in pixels
            periodic (bool): flag for periodic boundary conditions
            bilinear (bool): flag for bilinear image shifts
            device(str): calculation device will be perfomed on. Must be 'cpu' or 'gpu'
        Returns:
            (array) the shifted array
    """
    if device == "cpu":
        xp = np

    elif device == "gpu":
        xp = cp

    ar = xp.asarray(ar)

    # Apply image shift
    if bilinear is False:
        nx, ny = xp.shape(ar)
        qx, qy = make_Fourier_coords2D(nx, ny, 1)
        qx = xp.asarray(qx)
        qy = xp.asarray(qy)

        w = xp.exp(-(2j * xp.pi) * ((yshift * qy) + (xshift * qx)))
        shifted_ar = xp.real(xp.fft.ifft2((xp.fft.fft2(ar)) * w))

    else:
        xF = xp.floor(xshift).astype(int).item()
        yF = xp.floor(yshift).astype(int).item()
        wx = xshift - xF
        wy = yshift - yF

        shifted_ar = (
            xp.roll(ar, (xF, yF), axis=(0, 1)) * ((1 - wx) * (1 - wy))
            + xp.roll(ar, (xF + 1, yF), axis=(0, 1)) * ((wx) * (1 - wy))
            + xp.roll(ar, (xF, yF + 1), axis=(0, 1)) * ((1 - wx) * (wy))
            + xp.roll(ar, (xF + 1, yF + 1), axis=(0, 1)) * ((wx) * (wy))
        )

    if periodic is False:
        # Rounded coordinates for boundaries
        xR = (xp.round(xshift)).astype(int)
        yR = (xp.round(yshift)).astype(int)

        if xR > 0:
            shifted_ar[0:xR, :] = 0
        elif xR < 0:
            shifted_ar[xR:, :] = 0
        if yR > 0:
            shifted_ar[:, 0:yR] = 0
        elif yR < 0:
            shifted_ar[:, yR:] = 0

    return shifted_ar


def get_maxima_2D(
    ar,
    subpixel="poly",
    upsample_factor=16,
    sigma=0,
    minAbsoluteIntensity=0,
    minRelativeIntensity=0,
    relativeToPeak=0,
    minSpacing=0,
    edgeBoundary=1,
    maxNumPeaks=1,
    _ar_FT=None,
):
    """
    Finds the maximal points of a 2D array.

    Args:
        ar (array) the 2D array
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
        _ar_FT (complex array) if 'multicorr' is used and this is not
            None, uses this argument as the Fourier transform of `ar`,
            instead of recomputing it

    Returns:
        a structured array with fields 'x','y','intensity'
    """
    from py4DSTEM.process.utils.multicorr import upsampled_correlation

    subpixel_modes = ("pixel", "poly", "multicorr")
    er = f"Unrecognized subpixel option {subpixel}. Must be in {subpixel_modes}"
    assert subpixel in subpixel_modes, er

    # gaussian filtering
    ar = ar if sigma <= 0 else gaussian_filter(ar, sigma)

    # local pixelwise maxima
    maxima_bool = (
        (ar >= np.roll(ar, (-1, 0), axis=(0, 1)))
        & (ar > np.roll(ar, (1, 0), axis=(0, 1)))
        & (ar >= np.roll(ar, (0, -1), axis=(0, 1)))
        & (ar > np.roll(ar, (0, 1), axis=(0, 1)))
        & (ar >= np.roll(ar, (-1, -1), axis=(0, 1)))
        & (ar > np.roll(ar, (-1, 1), axis=(0, 1)))
        & (ar >= np.roll(ar, (1, -1), axis=(0, 1)))
        & (ar > np.roll(ar, (1, 1), axis=(0, 1)))
    )

    # remove edges
    assert isinstance(edgeBoundary, (int, np.integer))
    if edgeBoundary < 1:
        edgeBoundary = 1
    maxima_bool[:edgeBoundary, :] = False
    maxima_bool[-edgeBoundary:, :] = False
    maxima_bool[:, :edgeBoundary] = False
    maxima_bool[:, -edgeBoundary:] = False

    # get indices
    # sort by intensity
    maxima_x, maxima_y = np.nonzero(maxima_bool)
    dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
    maxima = np.zeros(len(maxima_x), dtype=dtype)
    maxima["x"] = maxima_x
    maxima["y"] = maxima_y
    maxima["intensity"] = ar[maxima_x, maxima_y]
    maxima = np.sort(maxima, order="intensity")[::-1]

    if len(maxima) == 0:
        return maxima

    # filter
    maxima = filter_2D_maxima(
        maxima,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minSpacing=minSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
    )

    if subpixel == "pixel":
        return maxima

    # Parabolic subpixel refinement
    for i in range(len(maxima)):
        Ix1_ = ar[int(maxima["x"][i]) - 1, int(maxima["y"][i])].astype(np.float64)
        Ix0 = ar[int(maxima["x"][i]), int(maxima["y"][i])].astype(np.float64)
        Ix1 = ar[int(maxima["x"][i]) + 1, int(maxima["y"][i])].astype(np.float64)
        Iy1_ = ar[int(maxima["x"][i]), int(maxima["y"][i]) - 1].astype(np.float64)
        Iy0 = ar[int(maxima["x"][i]), int(maxima["y"][i])].astype(np.float64)
        Iy1 = ar[int(maxima["x"][i]), int(maxima["y"][i]) + 1].astype(np.float64)
        deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
        deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
        maxima["x"][i] += deltax
        maxima["y"][i] += deltay
        maxima["intensity"][i] = linear_interpolation_2D(
            ar, maxima["x"][i], maxima["y"][i]
        )

    if subpixel == "poly":
        return maxima

    # Fourier upsampling
    if _ar_FT is None:
        _ar_FT = np.fft.fft2(ar)
    for ipeak in range(len(maxima["x"])):
        xyShift = np.array((maxima["x"][ipeak], maxima["y"][ipeak]))
        # we actually have to lose some precision and go down to half-pixel
        # accuracy for multicorr
        xyShift[0] = np.round(xyShift[0] * 2) / 2
        xyShift[1] = np.round(xyShift[1] * 2) / 2

        subShift = upsampled_correlation(_ar_FT, upsample_factor, xyShift)
        maxima["x"][ipeak] = subShift[0]
        maxima["y"][ipeak] = subShift[1]

    maxima = np.sort(maxima, order="intensity")[::-1]
    return maxima


def filter_2D_maxima(
    maxima,
    minAbsoluteIntensity=0,
    minRelativeIntensity=0,
    relativeToPeak=0,
    minSpacing=0,
    edgeBoundary=1,
    maxNumPeaks=1,
):
    """
    Args:
        maxima : a numpy structured array with fields 'x', 'y', 'intensity'
        minAbsoluteIntensity : delete counts with intensity below this value
        minRelativeIntensity : delete counts with intensity below this value times
            the intensity of the i'th peak, where i is given by `relativeToPeak`
        relativeToPeak : see above
        minSpacing : if two peaks are within this euclidean distance from one
            another, delete the less intense of the two
        edgeBoundary : delete peaks within this distance of the image edge
        maxNumPeaks : an integer. defaults to 1

    Returns:
        a numpy structured array with fields 'x', 'y', 'intensity'
    """

    # Remove maxima which are too dim
    if minAbsoluteIntensity > 0:
        deletemask = maxima["intensity"] < minAbsoluteIntensity
        maxima = maxima[~deletemask]

    # Remove maxima which are too dim, compared to the n-th brightest
    if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
        assert isinstance(relativeToPeak, (int, np.integer))
        deletemask = (
            maxima["intensity"] / maxima["intensity"][relativeToPeak]
            < minRelativeIntensity
        )
        maxima = maxima[~deletemask]

    # Remove maxima which are too close
    if minSpacing > 0:
        deletemask = np.zeros(len(maxima), dtype=bool)
        for i in range(len(maxima)):
            if deletemask[i] == False:
                tooClose = (
                    (maxima["x"] - maxima["x"][i]) ** 2
                    + (maxima["y"] - maxima["y"][i]) ** 2
                ) < minSpacing**2
                tooClose[: i + 1] = False
                deletemask[tooClose] = True
        maxima = maxima[~deletemask]

    # Remove maxima in excess of maxNumPeaks
    if maxNumPeaks is not None:
        if len(maxima) > maxNumPeaks:
            maxima = maxima[:maxNumPeaks]

    return maxima


def linear_interpolation_2D(ar, x, y):
    """
    Calculates the 2D linear interpolation of array ar at position x,y using the four
    nearest array elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    dx = x - x0
    dy = y - y0
    return (
        (1 - dx) * (1 - dy) * ar[x0, y0]
        + (1 - dx) * dy * ar[x0, y1]
        + dx * (1 - dy) * ar[x1, y0]
        + dx * dy * ar[x1, y1]
    )
