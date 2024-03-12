import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np



def get_shifted_ar(ar, xshift, yshift, periodic=True, bilinear=False, device="cpu"):
    """
    Shifts array ar by the shift vector (xshift,yshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Parameters
    ----------
    ar : float
        input array
    xshift : float
        shift along axis 0 (x) in pixels
    yshift : float
        shift along axis 1 (y) in pixels
    periodic : bool
        flag for periodic boundary conditions
    bilinear : bool
        flag for bilinear image shifts
    device : str
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns
    -------
    the shifted array
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


