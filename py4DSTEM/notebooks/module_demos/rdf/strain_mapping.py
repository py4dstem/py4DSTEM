#%%
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
dp_mask = spim.median_filter(mean_dp < 100, 5)  # filter out the beam stop
