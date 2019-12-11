import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import scipy.io as sio
from py4DSTEM.process.utils import print_progress_bar
from py4DSTEM.process.utils import polar_elliptical_transform
import matplotlib
from tqdm import tqdm

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()


def fit_stack(datacube, init_coefs):
    """
    This will fit an ellipse using the polar elliptical transform code to all the diffraction patterns. It will take in a datacube and return a coefficient array which can then be used to map strain, fit the centers, etc.

    Accepts:
        datacute    - a datacube of diffraction data
        init_coefs  - an initial starting guess for the fit
    Returns:
        coef_array  - an array of coefficients of the fit
    """
    coefs_array = np.zeros([i for i in datacube.data.shape[0:2]] + [len(init_coefs)])
    for i in tqdm(range(datacube.R_Nx)):
        for j in range(datacube.R_Ny):
            im = polar_elliptical_transform(datacube.data[i, j, :, :])
            im.fit_params_two_sided_gaussian(init_coef=init_coefs)
            coefs_array[i, j] = im.coef_opt

    return coefs_array


def calculate_coef_strain(coef_array, r_ref=None):
    """
    This function will calculate the strains from a 4D matrix output by fit_stack

    Cost function for two sided gaussian.
    xx      = x coords
    yy      = y coords
    coef[0] = N (linear constant)
    coef[1] = I_BG
    coef[2] = SD_BG
    coef[3] = I_ring
    coef[4] = SD_1
    coef[5] = SD_2
    coef[6] = X_center
    coef[7] = Y_center
    coef[8] = B
    coef[9] = C
    coef[10] = R

    Accepts:
        coef_array  - output from fit_stack
        r_ref       - reference radius ~0 strain. Default is none, and then will use median value
    Returns:
        exx         - strain in the major axis direction
        eyy         - strain in the minor axis direction
        exy         - shear

    """
    if r_ref is None:
        r_ref = np.median(coef_array[:, :, 10])
    r = coef_array[:, :, 10]
    b = coef_array[:, :, 8]
    c = coef_array[:, :, 9]

    d = r / r_ref
    a = 1 / d
    b /= d
    c /= d

    exx = 1 / 2 * (a - 1)
    eyy = 1 / 2 * (c - 1)
    exy = 1 / 2 * b

    return exx, eyy, exy


def convert_stack_polar(datacube, coef_array):
    """
    This function will take the coef_array from fit_stack and apply it to the image stack, to return polar transformed images.

    Accepts:
        datacube    - data in datacube format
        coef_array  - coefs from fit_stack

    Returns:
        datacube_polar - polar transformed datacube
    """

    return datacube_polar
