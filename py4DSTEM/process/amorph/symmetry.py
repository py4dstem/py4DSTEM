# move compute fem, symmetry functions here
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_polar_symmetries(dp):
    """
    This function will take in a polar transformed diffraction pattern (2D), compute the autocorrelation, and then the symmetries as well. 

    This function is to be used by the function which does this for the whole stack. 

    dp has theta along axis 0, and r along axis 1

    the normalized fourier coeffiecent for a certain symmetry order, a measure of symmetry, is then found by taking the average of the result in the radial bins desired. For example, two fold symmetry over the first five radial bins is equivalent to np.mean(dp_fft_normalized[2, 0:5])
    """
    dp_autocorrelated = np.fft.ifft(
        np.abs(np.fft.fft(dp, axis=0)) ** 2, axis=0
    )  # this emphasizes signal, but destroys any angular info
    dp_fft = np.abs(np.fft.fft(dp_autocorrelated, axis=0))
    # removes the effect of changing pattern intensity
    dp_fft_normalized = dp_fft / dp_fft[0, :]

    return dp_fft_normalized


def compute_polar_stack_symmetries(datacube_polar):
    """
    This function will take in a datacube of polar-transformed diffraction patterns, and do the autocorrelation, before taking the fourier transform along the theta direction, such that symmetries can be measured. They will be plotted by a different function

    Accepts:
        datacube_polar  - diffraction pattern cube that has been polar transformed

    Returns:
        datacube_symmetries - the normalized fft along the theta direction of the autocorrelated patterns in datacube_polar
    """
    datacube_symmetries = np.empty_like(datacube_polar.data)

    for i in tqdm(range(datacube_polar.R_Nx)):
        for j in range(datacube_polar.R_Ny):
            datacube_symmetries[i, j, :, :] = compute_polar_symmetries(
                datacube_polar.data[i, j, :, :]
            )

    return datacube_symmetries


def compute_FEM(data, method, mask=None):
    """
    implementing the four variance measurements from http://dx.doi.org/10.1016/j.ultramic.2010.05.010 Nanobeam diffraction fluctuation electron microscopy technique for structural characterization of disordered materials-Application to Al88-xY7Fe5Tix metallic glasses. Adapted from my Matlab code, but to only run on one dataset at a time.

    Inputs:
    data    - polar-transformed stacks (py4DSTEM dataobject). the shape is (R_Nx, R_Ny, theta, r)
    method  - integer, 0-3 corresponding to the four methods of computing FEM variance. 0 is the variance of annular mean, 1 is mean of ring variances, 2 is ring ensemble variance, and 3 is the annular mean of the variance
    mask    - real space mask that says which patterns to include
    """

    if mask is None:
        mask = np.ones(data.data.shape[0:2], dtype=bool)

    data = data.data[mask, :, :]  # this turns the data from 4D to 3D

    if method == 0:
        # this is variance of the annular mean
        ann_mean = np.mean(data, axis=1)
        ann_mean_var = (
            np.mean(ann_mean ** 2, axis=0) / np.mean(ann_mean, axis=0) ** 2 - 1
        )
        fem_result = ann_mean_var

    elif method == 1:
        # this is the mean of the ring variances
        ring_var = np.mean(data ** 2, axis=1) / np.mean(data, axis=1) ** 2 - 1
        fem_result = np.mean(ring_var, axis=0)

    elif method == 2:
        # this is the ring ensemble variance
        ring_ensemble_var = np.zeros(data.shape[2])
        for i in range(data.shape[2]):
            ring = np.ravel(data[:, :, i])
            ring_ensemble_var[i] = np.mean(ring ** 2) / np.mean(ring) ** 2 - 1
        fem_result = ring_ensemble_var

    elif method == 3:
        # this is the annular mean of the variance image
        var_im = np.mean(data ** 2, axis=0) / np.mean(data, axis=0) ** 2 - 1
        ann_mean_var_im = np.mean(var_im, axis=0)
        fem_result = ann_mean_var_im
    else:
        raise ValueError("Incorrect method input, must be int between 0 and 3.")

    return fem_result


def get_symmetries(datacube_symmetries, sym_order, r_range):
    """
    Accepts:
        datacube_symmetries - result of compute_polar_stack_symmetries, the stack of fft'd autocorrelated diffraction patterns. This is just a 4D numpy array
        sym_order           - symmetry order desired to plot
        r_range             - tuple of r indexes to avg over, indicating start, and stop
    Returns:
        array to be plotted using imshow
    """

    return np.mean(
        datacube_symmetries[:, :, sym_order, r_range[0] : r_range[1]], axis=2
    )
