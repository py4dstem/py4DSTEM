import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
from py4DSTEM.process.rdf import amorph
import matplotlib
from py4DSTEM.process.utils.ellipticalCoords import *
from tqdm import tqdm
from scipy.signal import medfilt2d
from scipy.ndimage.morphology import binary_closing

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()

# flags to control which part of the script to run
run_test = False
run_data = False
analyze_data = True
# make ellipse
"""
The parameters in p are

    p[0] I0          the intensity of the first gaussian function
    p[1] I1          the intensity of the Janus gaussian
    p[2] sigma0      std of first gaussian
    p[3] sigma1      inner std of Janus gaussian
    p[4] sigma2      outer std of Janus gaussian
    p[5] c_bkgd      a constant offset
    p[6] R           center/radius of the Janus gaussian
    p[7:8] x0,y0       the origin
    p[9:11] A,B,C       Ax^2 + Bxy + Cy^2 = 1

"""
if run_test:
    yy, xx = np.meshgrid(np.arange(256), np.arange(256))
    coef = [250, 200, 1, 2, 2, 5, 60, 127, 125, 1.0, 0, 1.0]

    ring = double_sided_gaussian(coef, xx, yy)
    plt.figure(1)
    plt.clf()
    plt.imshow(ring)
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, R, x0, y0, A, B, C = coef
    r2 = A * (xx - x0) ** 2 + B * (xx - x0) * (yy - y0) + C * (yy - y0) ** 2

    mask = r2 ** 0.5 > 40

    coef_fit = [25, 200, 1, 0.2, 0.2, 2, 60, 128, 128, 1, 0, 1]
    fit = fit_double_sided_gaussian(ring, coef_fit, mask=mask)
    plt.figure(12)
    plt.clf()
    ring_fit = double_sided_gaussian(fit, xx, yy)
    plt.imshow(ring_fit)
    r_ratio = fit[6] ** 2 / coef[6] ** 2
    print(
        f"fitted parameters:\nI_center = {fit[0]:.2f}\nI_ring = {fit[1]:.2f}\nSTD_center = {fit[2]:.2f}\nSTD_inner = {fit[3]:.2f}\nSTD_outer = {fit[4]:.2f}\nBackground = {fit[5]:.2f}\nRadius = {fit[6]/r_ratio**.5:.2f}\nCenter = {fit[7]:.2f}, {fit[8]:.2f}\nA,B,C = {fit[9]/r_ratio:.2f}, {fit[10]/r_ratio:.2f}, {fit[11]/r_ratio:.2f}\n"
    )
    print(
        f"input parameters:\nI_center = {coef[0]:.2f}\nI_ring = {coef[1]:.2f}\nSTD_center = {coef[2]:.2f}\nSTD_inner = {coef[3]:.2f}\nSTD_outer = {coef[4]:.2f}\nBackground = {coef[5]:.2f}\nRadius = {coef[6]:.2f}\nCenter = {coef[7]:.2f}, {coef[8]:.2f}\nA,B,C = {coef[9]:.2f}, {coef[10]:.2f}, {coef[11]:.2f}"
    )

# Read the note in ellipticalCoords.py - the fit currently has a dependent variable, messing things up.
# the data is binned by 4 and is now [162,285,112,120]


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

    for i in tqdm(range(data_shape[0])):
        for j in range(data_shape[1]):
            mask_array[i, j, :, :] = universal_mask
            for spot in peak_pos_array.get_pointlist(i, j).data:
                temp_inds = (
                    (xx - spot[1] / bin_factor) ** 2 + (yy - spot[2] / bin_factor) ** 2
                ) ** 0.5
                temp_inds = temp_inds < peak_radius
                mask_array[i, j, temp_inds] = 0

    return mask_array.astype(bool)


load_data = False
if load_data:
    print("loading data")
    data = py4DSTEM.file.io.read(
        "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/binned_data.h5"
    )
    helper_data_browser = py4DSTEM.file.io.FileBrowser(
        "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/Dataset38_20190918_processing.h5"
    )
    # peaks = helper_data_browser.get_dataobject('braggpeaks_unshifted')
    mask_array = py4DSTEM.file.io.read(
        "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/data_mask_spots_and_radius.h5"
    )

if run_data:
    # make a mask of region to fit
    mean_dp = np.mean(data.data, axis=(0, 1))
    yy, xx = np.meshgrid(np.arange(mean_dp.shape[1]), np.arange(mean_dp.shape[0]))
    center = [52, 59]
    r = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) ** 0.5
    mask_r = np.ones_like(r)
    mask_r[r < 15] = 0
    mask_r[r > 40] = 0
    mask_r = mask_r.astype(bool)

    test_im = data.data[125, 173, :, :]
    plt.figure(12)
    plt.clf()
    # test_im = np.log(test_im + 0.01)
    plt.imshow(test_im ** 0.25 * mask_r)

    p_init = [10, 700, 5, 4, 4, 50, 30, 50, 60, 1, 0, 1]
    test_fit = fit_double_sided_gaussian(test_im, p_init, mask=mask_r)
    # compare_double_sided_gaussian(test_im, p_init, mask=mask_r)

    # mask_array = make_mask_array(peaks, data.data.shape, peak_radius=4.3, bin_factor=4, universal_mask=mask_r)
    np.seterr(all="ignore")
    coef_array = amorph.fit_stack(data, p_init, mask_array.data)

if analyze_data:
    # coef_array = py4DSTEM.file.io.read(
    #     "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/ellipse_coefs.h5"
    # )
    # coef_array = np.squeeze(coef_array.data)
    # coef_array[np.isnan(coef_array)] = 1

    coef_array = np.load(
        "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/coef_array_np.npy"
    )
    # # remove zeros/nan values
    coef_array[np.where(coef_array[:, :, 6] == 0)] = 10

    radius_ref = (
        29
    )  # used to set 0 strain value - remember that strain values are comparative!
    # mask of regions that are crystalline
    # strains = amorph.calculate_coef_strain(coef_array, r_ref=radius_ref, A_ref=np.median(coef_array[:,:,9]), B_ref=np.median(coef_array[:,:,10]), C_ref=np.median(coef_array[:,:,11]))
    strains = amorph.calculate_coef_strain(coef_array, r_ref=radius_ref)

    mask_strain = np.logical_or(np.abs(strains[0]) > 0.1, np.abs(strains[1]) > 0.1)
    mask_strain = binary_closing(mask_strain, iterations=3, border_value=1)

    normalized_strains = [
        medfilt2d(i) - np.median(medfilt2d(i)[135:, :]) for i in strains
    ]
    amorph.plot_strains(normalized_strains, vmax=0.05, vmin=-0.05, mask=mask_strain)
