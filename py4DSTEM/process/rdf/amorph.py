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
        coef_cube  - an array of coefficients of the fit
    """
    coefs_array = np.zeros([i for i in datacube.data.shape[0:2]] + [len(init_coefs)])
    for i in tqdm(range(datacube.R_Nx)):
        for j in range(datacube.R_Ny):
            im = polar_elliptical_transform(datacube.data[i, j, :, :])
            im.fit_params_two_sided_gaussian(init_coef=init_coefs)
            coefs_array[i, j] = im.coef_opt

    return coefs_array


def calculate_coef_strain(coef_cube, r_ref=None, b_ref=None, c_ref=None):
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
        coef_cube   - output from fit_stack
        r_ref       - reference radius ~0 strain (coef[10]). Default is none, and then will use median value
        b_ref       - reference B value (coef[8]), from perhaps mean image
        c_ref       - reference C value (coef[9]), from perhaps mean image
    Returns:
        exx         - strain in the major axis direction
        eyy         - strain in the minor axis direction
        exy         - shear

    """
    if r_ref is None:
        r_ref = np.median(coef_cube[:, :, 10])
    if c_ref is None:
        c_ref = 1
    if b_ref is None:
        b_ref = 0
    r = coef_cube[:, :, 10]
    b = coef_cube[:, :, 8]
    c = coef_cube[:, :, 9]

    d = r / r_ref
    a = 1 / d
    b /= d
    c /= d

    exx = 1 / 2 * (a - 1)
    eyy = 1 / 2 * (c - c_ref)  # TODO - make sure this is ok to do
    exy = 1 / 2 * (b - b_ref)  # TODO - make sure this is ok to do

    return exx, eyy, exy


def plot_strains(strains, cmap="RdBu_r", vmin=None, vmax=None):
    """
    This function will plot strains with a unified color scale.

    Accepts:
        strains             - a collection of 3 arrays in the format (exx, eyy, exy)
        cmap, vmin, vmax    - imshow parameters
    """
    if vmin is None:
        vmin = np.min(strains)
    if vmax is None:
        vmax = np.max(strains)

    plt.figure(88, figsize=(9, 5.8))
    plt.clf()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, num=88)
    ax1.imshow(strains[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax1.set_title(r"$\epsilon_{AA}$")

    ax2.imshow(strains[1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax2.set_title(r"$\epsilon_{CC}$")

    im = ax3.imshow(strains[2], cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax3.set_title(r"$\epsilon_{B}$")

    cbar_ax = f.add_axes([0.125, 0.25, 0.775, 0.05])
    f.colorbar(im, cax=cbar_ax, orientation="horizontal")

    return


def compare_coef_cube(dp, coefs, mask=None, power=0.3):
    """
    This function will compare the fit to individual diffraction patterns. It is essentially a helper function to quickly build the right object.

    Accepts:
        dp      - a 2D diffraction pattern
        coefs   - coefs from coef_cube, corresponding to the diffraction pattern
        power   - the power to which the comparison is taken
    Returns:
        None
    """
    dp = polar_elliptical_transform(dp, mask=mask)
    dp.compare_coefs_two_sided_gaussian(coefs, power=power)
    return


def convert_stack_polar(datacube, coef_cube):
    """
    This function will take the coef_cube from fit_stack and apply it to the image stack, to return polar transformed images.

    Accepts:
        datacube    - data in datacube format
        coef_cube  - coefs from fit_stack

    Returns:
        datacube_polar - polar transformed datacube
    """

    return datacube_polar


def compute_polar_stack_symmetries(datacube_polar):
    """
    This function will take in a datacube of polar-transformed diffraction patterns, and do the autocorrelation, before taking the fourier transform along the theta direction, such that symmetries can be measured. They will be plotted by a different function

    Accepts:
        datacube_polar  - diffraction pattern cube that has been polar transformed

    Returns:
        datacube_symmetries - the normalized fft along the theta direction of the autocorrelated patterns in datacube_polar
    """

    return datacube_symmetries


def plot_symmetries(datacube_symmetries, sym_order):
    """
    This function will take in a datacube from compute_polar_stack_symmetries and plot a specific symmetry order. 

    Accepts:
        datacube_symmetries - result of compute_polar_stack_symmetries, the stack of fft'd autocorrelated diffraction patterns
        sym_order           - symmetry order desired to plot
    Returns:
        None
    """

    return None
