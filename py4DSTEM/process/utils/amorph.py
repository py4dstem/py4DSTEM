# bring in corr2d, compute nn, nn_sum
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

