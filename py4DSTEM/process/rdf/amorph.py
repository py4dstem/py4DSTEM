#%%
import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import scipy.io as sio
from py4DSTEM.process.utils import print_progress_bar
from py4DSTEM.process.utils import polar_elliptical_transform
import matplotlib
from tqdm import tqdm
matplotlib.rcParams['figure.dpi'] = 100
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
yy,xx = np.meshgrid(np.arange(256), np.arange(256))
coef = [10,1,1,100,10,10,128,125,0.6,1,85]
r = np.sqrt( (xx - coef[6])**2 + coef[8]*(xx - coef[6])*(yy - coef[7]) + coef[9]*(yy - coef[7])**2)
ring = (coef[0] + coef[1] * np.exp( (-1/ (2*coef[2]**2)) * r**2) + coef[3] *np.exp( (-1/ (2*coef[4]**2)) * (coef[10] - r)**2) * np.heaviside((coef[10] - r),.5) + coef[3] * np.exp( (-1/ (2*coef[5]**2)) * (coef[10] - r)**2) * np.heaviside((r - coef[10]),.5) )
plt.imshow(ring)

# %%
data = polar_elliptical_transform(calibration_image=ring, r_range=200)
# params = data.fit_params(10,return_ans=True)
# plt.imshow(data.polar_ar)
data.fit_params_two_sided_gaussian()
data.compare_coefs_two_sided_gaussian()
# %%

def fit_stack(datacube, init_coefs):
    """
    This will fit an ellipse using the polar elliptical transform code to all the diffraction patterns. It will take in a datacube and return a coefficient array which can then be used to map strain, fit the centers, etc.

    Accepts:
        datacute    - a datacube of diffraction data
        init_coefs  - an initial starting guess for the fit
    Returns:
        coef_array  - an array of coefficients of the fit
    """
    coefs_array = np.zeros(datacube.data.shape[0:2])
    i = 0
    for im in tqdm(datacube):
        im = polar_elliptical_transform(im)
        im.fit_params_two_sided_gaussian(init_coef=init_coefs)
        coefs_array[i] = im.coef_opt
        i += 1
    return coef_array

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



