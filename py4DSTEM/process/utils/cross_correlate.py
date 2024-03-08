# Cross correlation function

import numpy as np
from py4DSTEM.preprocess.utils import get_shifted_ar
from py4DSTEM.process.utils.multicorr import upsampled_correlation

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np


def get_cross_correlation(ar, template, corrPower=1, _returnval="real"):
    """
    Get the cross/phase/hybrid correlation of `ar` with `template`, where
    the latter is in real space.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """
    assert _returnval in ("real", "fourier")
    template_FT = np.conj(np.fft.fft2(template))
    return get_cross_correlation_FT(
        ar, template_FT, corrPower=corrPower, _returnval=_returnval
    )


def get_cross_correlation_FT(ar, template_FT, corrPower=1, _returnval="real"):
    """
    Get the cross/phase/hybrid correlation of `ar` with `template_FT`, where
    the latter is already in Fourier space (i.e. `template_FT` is
    `np.conj(np.fft.fft2(template))`.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """
    assert _returnval in ("real", "fourier")
    m = np.fft.fft2(ar) * template_FT
    if corrPower != 1:
        cc = np.abs(m) ** (corrPower) * np.exp(1j * np.angle(m))
    else:
        cc = m
    if _returnval == "real":
        cc = np.maximum(np.real(np.fft.ifft2(cc)), 0)
    return cc


def get_shift(ar1, ar2, corrPower=1):
    """
        Determine the relative shift between a pair of arrays giving the best overlap.

        Shift determination uses the brightest pixel in the cross correlation, and is
    thus limited to pixel resolution. corrPower specifies the cross correlation
    power, with 1 corresponding to a cross correlation and 0 a phase correlation.

        Args:
                ar1,ar2 (2D ndarrays):
        corrPower (float between 0 and 1, inclusive): 1=cross correlation, 0=phase
            correlation

    Returns:
                (2-tuple): (shiftx,shifty) - the relative image shift, in pixels
    """
    cc = get_cross_correlation(ar1, ar2, corrPower)
    xshift, yshift = np.unravel_index(np.argmax(cc), ar1.shape)
    return xshift, yshift


def align_images_fourier(
    G1,
    G2,
    upsample_factor,
    device="cpu",
):
    """
    Alignment of two images using DFT upsampling of cross correlation.

    Parameters
    -------
    G1: ndarray
        fourier transform of image 1
    G2: ndarray
        fourier transform of image 2
    upsample_factor: float
        upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns:
        xy_shift [pixels]
    """

    if device == "cpu":
        xp = np
    elif device == "gpu":
        xp = cp

    G1 = xp.asarray(G1)
    G2 = xp.asarray(G2)

    # cross correlation
    cc = G1 * xp.conj(G2)
    cc_real = xp.real(xp.fft.ifft2(cc))

    # local max
    x0, y0 = xp.unravel_index(cc_real.argmax(), cc.shape)

    # half pixel shifts
    x_inds = xp.mod(x0 + xp.arange(-1, 2), cc.shape[0]).astype("int")
    y_inds = xp.mod(y0 + xp.arange(-1, 2), cc.shape[1]).astype("int")

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]
    dx = (vx[2] - vx[0]) / (4 * vx[1] - 2 * vx[2] - 2 * vx[0])
    dy = (vy[2] - vy[0]) / (4 * vy[1] - 2 * vy[2] - 2 * vy[0])

    x0 = xp.round((x0 + dx) * 2.0) / 2.0
    y0 = xp.round((y0 + dy) * 2.0) / 2.0

    # subpixel shifts
    xy_shift = upsampled_correlation(
        cc, upsample_factor, xp.array([x0, y0]), device=device
    )

    return xy_shift


def align_and_shift_images(
    image_1,
    image_2,
    upsample_factor,
    device="cpu",
):
    """
    Alignment of two images using DFT upsampling of cross correlation.

    Parameters
    -------
    image_1: ndarray
        image 1
    image_2: ndarray
        image 2
    upsample_factor: float
        upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'.

    Returns:
        shifted image [pixels]
    """

    if device == "cpu":
        xp = np

    elif device == "gpu":
        xp = cp

    image_1 = xp.asarray(image_1)
    image_2 = xp.asarray(image_2)

    xy_shift = align_images_fourier(
        xp.fft.fft2(image_1),
        xp.fft.fft2(image_2),
        upsample_factor=upsample_factor,
        device=device,
    )
    dx = (
        xp.mod(xy_shift[0] + image_1.shape[0] / 2, image_1.shape[0])
        - image_1.shape[0] / 2
    )
    dy = (
        xp.mod(xy_shift[1] + image_1.shape[1] / 2, image_1.shape[1])
        - image_1.shape[1] / 2
    )

    image_2_shifted = get_shifted_ar(image_2, dx, dy, device=device)

    return image_2_shifted
