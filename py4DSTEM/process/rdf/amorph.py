import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import copy
import scipy.io as sio
from py4DSTEM.process.utils import print_progress_bar
from py4DSTEM.process.utils import polar_elliptical_transform
from py4DSTEM.process.utils.ellipticalCoords import *
import matplotlib
from tqdm import tqdm

# this fixes figure sizes on HiDPI screens
matplotlib.rcParams["figure.dpi"] = 200
plt.ion()


def make_mask_array(
    peak_pos_array, data_shape, peak_radius, bin_factor=1, universal_mask=None
):
    """
    This function needs a real home, I don't know where to put it yet. But this function will take in a peakListArray with all of the peaks in the diffraction pattern, and make a 4d array of masks such that they can be quickly applied to the diffraction patterns before fitting. 

    Accepts:
    peak_pos_array  - a peakListArray corresponding to all of the diffraction patterns in the dataset
    data_shape      - the 4-tuple shape of the data, essentially data.data.shape (qx, qy, x, y)
    peak_radius     - the peak radius
    bin_factor      - if the peak positions were measured at a different binning factor than the data, this will effectivly divide their location by bin_factor
    
    Returns:
    mask_array      - a 4D array of masks the same shape as the data
    """
    if universal_mask is None:
        universal_mask = np.ones(data_shape[2:])
    mask_array = np.ones(data_shape)
    yy, xx = np.meshgrid(np.arange(data_shape[3]), np.arange(data_shape[2]))

    for i in tqdm(np.arange(data_shape[0])):
        for j in np.arange(data_shape[1]):
            mask_array[i, j, :, :] = universal_mask
            for spot in peak_pos_array.get_pointlist(i, j).data:
                temp_inds = (
                    (xx - spot[0] / bin_factor) ** 2 + (yy - spot[1] / bin_factor) ** 2
                ) ** 0.5
                temp_inds = temp_inds < peak_radius
                mask_array[i, j, temp_inds] = 0
                # plt.figure(100,clear=True)
                # plt.imshow(temp_inds)
                # plt.pause(0.2)
    return mask_array.astype(bool)


def fit_stack(datacube, init_coefs, mask=None):
    """
    This will fit an ellipse using the polar elliptical transform code to all the diffraction patterns. It will take in a datacube and return a coefficient array which can then be used to map strain, fit the centers, etc.

    Accepts:
        datacute    - a datacube of diffraction data
        init_coefs  - an initial starting guess for the fit
        mask        - a mask, either 2D or 4D, for either one mask for the whole stack, or one per pattern. 
    Returns:
        coef_cube  - an array of coefficients of the fit
    """
    coefs_array = np.zeros([i for i in datacube.data.shape[0:2]] + [len(init_coefs)])
    for i in tqdm(range(datacube.R_Nx)):
        for j in tqdm(range(datacube.R_Ny)):
            if len(mask.shape) == 2:
                mask_current = mask
            elif len(mask.shape) == 4:
                mask_current = mask[i, j, :, :]

            coefs = fit_double_sided_gaussian(
                datacube.data[i, j, :, :], init_coefs, mask=mask_current
            )
            coefs_array[i, j] = coefs

    return coefs_array


def calculate_coef_strain(coef_cube, r_ref):
    """
    This function will calculate the strains from a 3D matrix output by fit_stack

    Coefs order:
        I0          the intensity of the first gaussian function
        I1          the intensity of the Janus gaussian
        sigma0      std of first gaussian
        sigma1      inner std of Janus gaussian
        sigma2      outer std of Janus gaussian
        c_bkgd      a constant offset
        R           center of the Janus gaussian
        x0,y0       the origin
        B,C         1x^2 + Bxy + Cy^2 = 1

    Accepts:
        coef_cube   - output from fit_stack
        r_ref       - a reference 0 strain radius - needed because we fit r as well as B and C
                      if r_ref is a number, it will assume that the reference ellipse is a perfect circle
                      r_ref can also be a 3-tuple, in which case it is (r_ref, B_ref, C_ref)
                      in this case, the reference 0 strain measurement is an ellipse defined by these numbers, which are defined as the outputs of fit_double_sided_gaussian
    Returns:
        exx         - strain in the x axis direction in image coordinates
        eyy         - strain in the y axis direction in image coordinates
        exy         - shear

    """
    # parse r_ref input
    if isinstance(r_ref, (int, float)):
        # r_ref is integer, and we will measure strain with circle
        A_ref, B_ref, C_ref = 1, 0, 1
    elif isinstance(r_ref, tuple):
        if len(r_ref) != 3:
            raise AssertionError(
                "r_ref must be a 3 element tuple with elements(r_ref, B_ref, C_ref)."
            )
        # r_ref is a tuple with (r_ref, B_ref, C_ref)
        A_ref, B_ref, C_ref = 1, r_ref[1], r_ref[2]
        r_ref = r_ref[0]
    else:
        raise ValueError("r_ref must be a number, or 3 element tuple")

    R = coef_cube[:, :, 6]
    r_ratio = (
        R / r_ref
    )  # this is a correction factor for what defines 0 strain, and must be applied to A, B and C. This has been found _experimentally_! TODO have someone else read this

    A = 1 / r_ratio ** 2
    B = coef_cube[:, :, 9] / r_ratio ** 2
    C = coef_cube[:, :, 10] / r_ratio ** 2

    # make reference transformation matrix
    m_ellipse_ref = np.asarray([[A_ref, B_ref / 2], [B_ref / 2, C_ref]])
    e_vals_ref, e_vecs_ref = np.linalg.eig(m_ellipse_ref)
    ang_ref = np.arctan2(e_vecs_ref[1, 0], e_vecs_ref[0, 0])

    rot_matrix_ref = np.asarray(
        [[np.cos(ang_ref), -np.sin(ang_ref)], [np.sin(ang_ref), np.cos(ang_ref)]]
    )

    transformation_matrix_ref = np.diag(np.sqrt(e_vals_ref))

    transformation_matrix_ref = (
        rot_matrix_ref @ transformation_matrix_ref @ rot_matrix_ref.T
    )

    transformation_matrix_ref_inv = np.linalg.inv(transformation_matrix_ref)

    exx, eyy, exy = np.empty_like(A), np.empty_like(C), np.empty_like(B)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            m_ellipse = np.asarray([[A[i, j], B[i, j] / 2], [B[i, j] / 2, C[i, j]]])
            e_vals, e_vecs = np.linalg.eig(m_ellipse)
            ang = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])
            rot_matrix = np.asarray(
                [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
            )
            transformation_matrix = np.diag(np.sqrt(e_vals))
            transformation_matrix = rot_matrix @ transformation_matrix @ rot_matrix.T

            transformation_matrix_from_ref = (
                transformation_matrix @ transformation_matrix_ref_inv
            )

            exx[i, j] = transformation_matrix_from_ref[0, 0] - 1
            eyy[i, j] = transformation_matrix_from_ref[1, 1] - 1
            exy[i, j] = 0.5 * (
                transformation_matrix_from_ref[0, 1]
                + transformation_matrix_from_ref[1, 0]
            )

    return exx, eyy, exy


def plot_strains(strains, cmap="RdBu_r", vmin=None, vmax=None, mask=None, fignum=88):
    """
    This function will plot strains with a unified color scale.

    Accepts:
        strains             - a collection of 3 arrays in the format (exx, eyy, exy)
        cmap, vmin, vmax    - imshow parameters
        mask                - real space mask of values not to show (black)
    """
    strains = copy.deepcopy(strains)
    cmap = matplotlib.cm.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(strains)
    if vmax is None:
        vmax = np.max(strains)
    if mask is None:
        mask = np.zeros_like(strains[0])

    cmap.set_under("black")
    cmap.set_over("black")
    cmap.set_bad("black")

    mask = mask.astype(bool)

    for i in strains:
        i[mask] = np.nan

    plt.figure(fignum, figsize=(9, 5.8), clear=True)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, num=fignum)
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
    ax1.set_title(r"$\epsilon_{xx}$")

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
    ax2.set_title(r"$\epsilon_{yy}$")

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
    ax3.set_title(r"$\epsilon_{xy}$")

    cbar_ax = f.add_axes([0.125, 0.25, 0.775, 0.05])
    f.colorbar(im, cax=cbar_ax, orientation="horizontal")

    return


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


def corr2d(im1, im2, mask=None):
    """
    This is the python version of matlab's corr2
    """

    if mask is not None:
        im1 = im1[mask]
        im2 = im2[mask]

    corr_val = np.sum((im1 - im1.mean()) * (im2 - im2.mean())) / np.sqrt(
        np.sum((im1 - im1.mean()) ** 2) * np.sum((im2 - im2.mean()) ** 2)
    )

    return corr_val


def compute_nn_corr(datacube, mask=None):
    """        
    the datacube is just a numpy array, and mask as well

    we will ignore the outer boundary where nearer neighbors aren't computed
    """
    corr_result = np.empty(datacube.shape[0:2])
    corr_result = corr_result[1:-1, 1:-1]

    for i in tqdm(range(corr_result.shape[0])):
        for j in range(corr_result.shape[1]):
            corr_result[i, j] = np.mean(
                [
                    corr2d(
                        datacube[i + 1, j + 1, :, :], datacube[i, j, :, :], mask=mask
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i, j + 1, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i, j + 2, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i + 1, j, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i + 1, j + 2, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i + 2, j, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i + 2, j + 1, :, :],
                        mask=mask,
                    ),
                    corr2d(
                        datacube[i + 1, j + 1, :, :],
                        datacube[i + 2, j + 2, :, :],
                        mask=mask,
                    ),
                ]
            )

    return corr_result


def plot_symmetries(datacube_symmetries, sym_order, r_range):
    """
    This function will take in a datacube from compute_polar_stack_symmetries and plot a specific symmetry order. 

    Accepts:
        datacube_symmetries - result of compute_polar_stack_symmetries, the stack of fft'd autocorrelated diffraction patterns. This is just a 4D numpy array
        sym_order           - symmetry order desired to plot
        r_range             - tuple of r indexes to sum/avg over, indicating start, and stop
    Returns:
        None
    """
    plt.figure(f"Symmetry order {sym_order}", clear=True)
    plt.imshow(
        np.mean(datacube_symmetries[:, :, sym_order, r_range[0] : r_range[1]], axis=2)
    )

    return None


def plot_nn(datacube, i, j, mask=None, **kwargs):
    """
    this will just plot a 3x3 grid of patterns
    datacube is a numpy array
    i is row,
    j is column
    mask is a numpy array
    kwargs is passed to plt.imshow
    """
    if mask is None:
        mask = np.ones(datacube.shape[2:4]).astype(bool)

    plt.figure(f"Nearest Neighbors of {i}, {j}", clear=True)
    tiled_image = np.concatenate(
        np.concatenate(datacube[i - 1 : i + 2, j - 1 : j + 2, :, :] * mask, axis=-2),
        axis=-1,
    )
    plt.imshow(tiled_image, **kwargs)

    return None


def nn_sum(data, rx, ry, weighting="gaussian", order=2, testing=False):
    """
    This function will take in a py4DSTEM datacube 'data', indices in real space, rx, ry, a weighting and an order, and return a single image, dp_sum. This image will be the nearest neighbor sum with either flat or gaussian weighting. 

    data        - py4DSTEM datacube
    rx, ry      - real space indices corresponding to row, column in the image
    weighting   - 'gaussian' or 'flat', corresponding to how patterns are summed
    order       - number, corresponding to how many nn orders for flat, or the gaussian width for gaussian weighting.
    """
    if rx >= data.R_Nx or rx < 0 or ry >= data.R_Ny or ry < 0:
        raise AssertionError(
            "invalid indices, rx and ry cannot be larger than their respective sizes - 1."
        )

    if weighting == "flat":
        inds_x = np.arange(rx - order, rx + order + 1, dtype=int)
        inds_y = np.arange(ry - order, ry + order + 1, dtype=int)
        inds_x, inds_y = np.meshgrid(inds_x, inds_y, indexing="ij")
        weights = np.ones_like(inds_x)
    elif weighting == "gaussian":
        inds_x = np.arange(rx - 2 * order, rx + 2 * order + 1, dtype=int)
        inds_y = np.arange(ry - 2 * order, ry + 2 * order + 1, dtype=int)
        inds_x, inds_y = np.meshgrid(inds_x, inds_y, indexing="ij")
        dist = np.sqrt((inds_x - rx) ** 2 + (inds_y - ry) ** 2)
        weights = np.exp(-((dist) ** 2) / (2 * order ** 2))
    else:
        raise AssertionError(
            "Weighting is not understood, must be either 'gaussian' or 'flat'"
        )

    # make inds/weights respect boundaries - decided I don't want to double count edges
    # inds_x[inds_x<0] = 0
    # inds_x[inds_x>data.R_Nx] = data.R_Nx
    # inds_y[inds_y<0] = 0
    # inds_y[inds_y>data.R_Ny] = data.R_Ny
    weights[inds_x < 0] = 0
    weights[inds_y < 0] = 0
    weights[inds_x > (data.R_Nx - 1)] = 0
    weights[inds_y > (data.R_Ny - 1)] = 0
    inds_x = inds_x % data.R_Nx
    inds_y = inds_y % data.R_Ny

    dp_sum = np.mean(
        data.data[inds_x, inds_y, :, :] * weights[:, :, None, None], axis=(0, 1)
    )

    if testing:
        mask = np.zeros(data.data.shape[0:2])
        mask[inds_x, inds_y] = 1
        plt.figure(1, clear=True)
        plt.imshow(mask)
        plt.title("Active indices")

        m_weights = np.zeros(data.data.shape[0:2])
        m_weights[inds_x, inds_y] = weights
        plt.figure(2, clear=True)
        plt.imshow(m_weights)
        plt.title("Active indices and weights")
        plt.figure(3, clear=True)
        plt.imshow(dist)
        plt.title("Distance in pixels")
        plt.figure(4, clear=True)
        plt.imshow(weights)
        plt.title("Weights used")

        plt.figure(5, clear=True)
        plt.imshow(dp_sum ** 0.25, cmap="inferno")
        plt.title("Summed diffraction pattern")
        plt.figure(6, clear=True)
        plt.imshow(data.data[rx, ry, :, :] ** 0.25, cmap="inferno")
        plt.title("Center diffraction pattern")

    return dp_sum


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
