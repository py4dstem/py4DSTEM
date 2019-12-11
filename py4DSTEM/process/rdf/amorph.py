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



