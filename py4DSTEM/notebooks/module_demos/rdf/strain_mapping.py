# %%
import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import scipy.io as sio
import scipy.ndimage as spim
from py4DSTEM.process.utils import print_progress_bar
from py4DSTEM.process.utils import polar_elliptical_transform
from py4DSTEM.process.rdf import amorph
import matplotlib
from tqdm import tqdm

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()

# %%
# make ellipse
"""
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
"""
yy, xx = np.meshgrid(np.arange(256), np.arange(256))
coef = [10, 1, 1, 100, 10, 10, 128, 125, -2, 3, 85]
r = np.sqrt(
    (xx - coef[6]) ** 2
    + coef[8] * (xx - coef[6]) * (yy - coef[7])
    + coef[9] * (yy - coef[7]) ** 2
)
ring = (
    coef[0]
    + coef[1] * np.exp((-1 / (2 * coef[2] ** 2)) * r ** 2)
    + coef[3]
    * np.exp((-1 / (2 * coef[4] ** 2)) * (coef[10] - r) ** 2)
    * np.heaviside((coef[10] - r), 0.5)
    + coef[3]
    * np.exp((-1 / (2 * coef[5] ** 2)) * (coef[10] - r) ** 2)
    * np.heaviside((r - coef[10]), 0.5)
)
plt.imshow(ring)

# %%
data = polar_elliptical_transform(calibration_image=ring, r_range=200)
# params = data.fit_params(10,return_ans=True)
# plt.imshow(data.polar_ar)
init_coefs = [10, 1, 2, 105, 9, 11, 127, 127, 0.2, 1, 80]
data.fit_params_two_sided_gaussian(init_coefs)
data.compare_coefs_two_sided_gaussian()
# %%
# load data
fp = "/media/tom/Data/Stack5_SampleSi_50x50_ss1nm_10fps_spot9_alpha=1p5_cl380_300kv.dm4"
fp = "/media/tom/Data/test_data/Stack1_20170214_12x12_ss20nm_2s_spot9_alpha=0p51_bin4_cl=480_200kV.dm3"
datacube = py4DSTEM.file.io.read(fp)
datacube.set_scan_shape(12, 12)
mean_dp = np.mean(datacube.data, axis=(0, 1))
dp_beam_stop_mask = spim.median_filter(mean_dp > 100, 5)  # filter out the beam stop

rad_min = 120
center_guess = [datacube.Q_Nx / 2, datacube.Q_Ny / 2]
yy, xx = np.meshgrid(
    np.arange(datacube.Q_Nx) - center_guess[0],
    np.arange(datacube.Q_Ny) - center_guess[1],
)

rr = (xx ** 2 + yy ** 2) ** 0.5

dp_center_mask = rr > rad_min
dp_mask = np.logical_and(dp_beam_stop_mask, dp_center_mask)

# %%
# plot some images
fig, (ax1, ax2) = plt.subplots(1, 2, num=4)
ax1.imshow(mean_dp)
ax2.imshow(dp_mask)
# %%
# find initial good parameters
mean_fit = polar_elliptical_transform(mean_dp, mask=dp_mask)
init_coefs = [1, 1, 1, 900, 20, 20, 270, 270, 0, 1, 210]

# check init_coefs to make sure it is ballpark
mean_fit.compare_coefs_two_sided_gaussian(init_coefs)
mean_fit.fit_params_two_sided_gaussian(init_coef=init_coefs)

# check the fit
mean_fit.compare_coefs_two_sided_gaussian(power=0.2)

# %%
# now that we have a starting point, let's run it on the stack
# coef_cube = amorph.fit_stack(datacube, mean_fit.coef_opt)

coef_fp = "/media/tom/Data/test_data/Stack1_coefs.npy"
coef_cube = np.load(coef_fp)
# we should save this data as it takes some time to run
np.save(coef_fp, coef_cube)

fig, (ax1, ax2) = plt.subplots(1, 2, num=5)
ax1.imshow(coef_cube[:, :, 6], cmap="RdBu")  # x shifts
ax2.imshow(coef_cube[:, :, 7], cmap="RdBu")  # y shifts

# %%
# here, we notice from the previous plot that our data is somehow shifted incorrectly (set shift to 0), when we plot the central beam motion. TODO correct for shift?

shift = -12
fig, (ax1, ax2) = plt.subplots(1, 2, num=6)
ax1.imshow(
    np.reshape(
        np.roll(np.ravel(coef_cube[:, :, 6], order="f"), shift), (12, 12), order="f"
    ),
    cmap="RdBu",
)
ax1.set_title("x shifts")
ax2.imshow(
    np.reshape(
        np.roll(np.ravel(coef_cube[:, :, 7], order="f"), shift), (12, 12), order="f"
    ),
    cmap="RdBu",
)
ax2.set_title("y shifts")
# however, in this dataset, after it is shifted, we do not see any real outliers! So we will not fit a plane/parabola to the dataset, and just use the positions that we found.

# %%
eaa, ecc, eb = amorph.calculate_coef_strain(coef_cube)
amorph.plot_strains((eaa,ecc,eb))  # strains are in the fit ellipse directions.
