import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np


def fourier_resample(
    array,
    scale=None,
    output_size=None,
    force_nonnegative=False,
    bandlimit_nyquist=None,
    bandlimit_power=2,
    dtype=np.float32,
):
    """
    Resize a 2D array along any dimension, using Fourier interpolation / extrapolation.
    For 4D input arrays, only the final two axes can be resized.

    The scaling of the array can be specified by passing either `scale`, which sets
    the scaling factor along both axes to be scaled; or by passing `output_size`,
    which specifies the final dimensions of the scaled axes (and allows for different
    scaling along the x,y or kx,ky axes.)

    Parameters
    ----------
    array : 2D/4D numpy array
        Input array, or 4D stack of arrays, to be resized.
    scale : float
        scalar value giving the scaling factor for all dimensions
    output_size : 2-tuple of ints
        two values giving either the (x,y) output size for 2D, or (kx,ky) for 4D
    force_nonnegative : bool
        Force all outputs to be nonnegative, after filtering
    bandlimit_nyquist : float
        Gaussian filter information limit in Nyquist units (0.5 max in both directions)
    bandlimit_power : float
        Gaussian filter power law scaling (higher is sharper)
    dtype : numpy dtype
        datatype for binned array. default is single precision float

    Returns
    -------
    the resized array (2D/4D numpy array)
    """

    # Verify input is 2D or 4D
    if np.size(array.shape) != 2 and np.size(array.shape) != 4:
        raise Exception(
            "Function does not support arrays with "
            + str(np.size(array.shape))
            + " dimensions"
        )

    # Get input size from last 2 dimensions
    input__size = array.shape[-2:]

    if scale is not None:
        assert (
            output_size is None
        ), "Cannot specify both a scaling factor and output size"
        assert np.size(scale) == 1, "scale should be a single value"
        scale = np.asarray(scale)
        output_size = (input__size * scale).astype("intp")
    else:
        assert scale is None, "Cannot specify both a scaling factor and output size"
        assert np.size(output_size) == 2, "output_size must contain two values"
        output_size = np.asarray(output_size)

    scale_output = np.prod(output_size) / np.prod(input__size)

    if bandlimit_nyquist is not None:
        kx = np.fft.fftfreq(output_size[0])
        ky = np.fft.fftfreq(output_size[1])
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2
        # Gaussian filter
        k_filt = np.exp(
            (k2 ** (bandlimit_power / 2)) / (-2 * bandlimit_nyquist**bandlimit_power)
        )

    # generate slices
    # named as {dimension}_{corner}_{in_/out},
    # where corner is ul, ur, ll, lr for {upper/lower}{left/right}

    # x slices
    if output_size[0] > input__size[0]:
        # x dimension increases
        x0 = int((input__size[0] + 1) // 2)
        x1 = int(input__size[0] // 2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0 - x1 + output_size[0], output_size[0])
        x_ll_in_ = slice(0 - x1 + input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0 - x1 + output_size[0], output_size[0])
        x_lr_in_ = slice(0 - x1 + input__size[0], input__size[0])

    elif output_size[0] < input__size[0]:
        # x dimension decreases
        x0 = int((output_size[0] + 1) // 2)
        x1 = int(output_size[0] // 2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0 - x1 + output_size[0], output_size[0])
        x_ll_in_ = slice(0 - x1 + input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0 - x1 + output_size[0], output_size[0])
        x_lr_in_ = slice(0 - x1 + input__size[0], input__size[0])

    else:
        # x dimension does not change
        x_ul_out = slice(None)
        x_ul_in_ = slice(None)

        x_ll_out = slice(None)
        x_ll_in_ = slice(None)

        x_ur_out = slice(None)
        x_ur_in_ = slice(None)

        x_lr_out = slice(None)
        x_lr_in_ = slice(None)

    # y slices
    if output_size[1] > input__size[1]:
        # y increases
        y0 = int((input__size[1] + 1) // 2)
        y1 = int(input__size[1] // 2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0 - y1 + output_size[1], output_size[1])
        y_ur_in_ = slice(0 - y1 + input__size[1], input__size[1])

        y_lr_out = slice(0 - y1 + output_size[1], output_size[1])
        y_lr_in_ = slice(0 - y1 + input__size[1], input__size[1])

    elif output_size[1] < input__size[1]:
        # y decreases
        y0 = int((output_size[1] + 1) // 2)
        y1 = int(output_size[1] // 2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0 - y1 + output_size[1], output_size[1])
        y_ur_in_ = slice(0 - y1 + input__size[1], input__size[1])

        y_lr_out = slice(0 - y1 + output_size[1], output_size[1])
        y_lr_in_ = slice(0 - y1 + input__size[1], input__size[1])

    else:
        # y dimension does not change
        y_ul_out = slice(None)
        y_ul_in_ = slice(None)

        y_ll_out = slice(None)
        y_ll_in_ = slice(None)

        y_ur_out = slice(None)
        y_ur_in_ = slice(None)

        y_lr_out = slice(None)
        y_lr_in_ = slice(None)

    if len(array.shape) == 2:
        # image array
        array_resize = np.zeros(output_size, dtype=np.complex64)
        array_fft = np.fft.fft2(array)

        # copy each quadrant into the resize array
        array_resize[x_ul_out, y_ul_out] = array_fft[x_ul_in_, y_ul_in_]
        array_resize[x_ll_out, y_ll_out] = array_fft[x_ll_in_, y_ll_in_]
        array_resize[x_ur_out, y_ur_out] = array_fft[x_ur_in_, y_ur_in_]
        array_resize[x_lr_out, y_lr_out] = array_fft[x_lr_in_, y_lr_in_]

        # Band limit if needed
        if bandlimit_nyquist is not None:
            array_resize *= k_filt

        # Back to real space
        array_resize = np.real(np.fft.ifft2(array_resize)).astype(dtype)

    elif len(array.shape) == 4:
        # This case is the same as the 2D case, but loops over the probe index arrays

        # init arrays
        array_resize = np.zeros((*array.shape[:2], *output_size), dtype)
        array_fft = np.zeros(input__size, dtype=np.complex64)
        array_output = np.zeros(output_size, dtype=np.complex64)

        for Rx, Ry in tqdmnd(
            array.shape[0],
            array.shape[1],
            desc="Resampling 4D datacube",
            unit="DP",
            unit_scale=True,
        ):
            array_fft[:, :] = np.fft.fft2(array[Rx, Ry, :, :])
            array_output[:, :] = 0

            # copy each quadrant into the resize array
            array_output[x_ul_out, y_ul_out] = array_fft[x_ul_in_, y_ul_in_]
            array_output[x_ll_out, y_ll_out] = array_fft[x_ll_in_, y_ll_in_]
            array_output[x_ur_out, y_ur_out] = array_fft[x_ur_in_, y_ur_in_]
            array_output[x_lr_out, y_lr_out] = array_fft[x_lr_in_, y_lr_in_]

            # Band limit if needed
            if bandlimit_nyquist is not None:
                array_output *= k_filt

            # Back to real space
            array_resize[Rx, Ry, :, :] = np.real(np.fft.ifft2(array_output)).astype(
                dtype
            )

    # Enforce positivity if needed, after filtering
    if force_nonnegative:
        array_resize = np.maximum(array_resize, 0)

    # Normalization
    array_resize = array_resize * scale_output

    return array_resize


