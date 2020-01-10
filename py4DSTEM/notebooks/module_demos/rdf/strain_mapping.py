import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
import scipy.io as sio
import scipy.ndimage as spim
from py4DSTEM.process.utils import print_progress_bar
from py4DSTEM.process.utils import polar_elliptical_transform
from py4DSTEM.process.rdf import amorph
import matplotlib
from py4DSTEM.process.utils.ellipticalCoords import *
from tqdm import tqdm

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()

# make ellipse
"""
The parameters in p are

    p[0] I0          the intensity of the first gaussian function
    p[1] I1          the intensity of the Janus gaussian
    p[2] sigma0      std of first gaussian
    p[3] sigma1      inner std of Janus gaussian
    p[4] sigma2      outer std of Janus gaussian
    p[5] c_bkgd      a constant offset
    p[6] R           center of the Janus gaussian
    p[7:8] x0,y0       the origin
    p[9:11] A,B,C       Ax^2 + Bxy + Cy^2 = 1

"""
yy, xx = np.meshgrid(np.arange(256), np.arange(256))
coef = [250, 200, 1, 2, 2, 5, 60, 127, 125, 1.1, .2, 1]

ring = double_sided_gaussian(coef, xx, yy)
plt.imshow(ring)
I0,I1,sigma0,sigma1,sigma2,c_bkgd,R,x0,y0,A,B,C = coef
r2 = A*(xx-x0)**2 + B*(xx-x0)*(yy-y0) + C*(yy-y0)**2

mask = r2 > 40

coef_fit = [25, 200, 1, .2, .2, 2, 60, 128, 128, 1, 0, 1]
fit = fit_double_sided_gaussian(ring, coef_fit, mask=mask)
plt.figure(12)
plt.clf()
ring_fit = double_sided_gaussian(fit, xx, yy)
plt.imshow(ring_fit)
print(['fit=']+[f'{i:.2f}' for i in fit])
print(['coef=']+[f'{i:.2f}' for i in coef])

# Read the note in ellipticalCoords.py - the fit currently has a dependent variable, messing things up.
# the data is binned by 4 and is now [162,285,112,120]
print('loading data')
data = py4DSTEM.file.io.read('/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/binned_data.h5')
helper_data = py4DSTEM.file.io.read('/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/Dataset38_20190918_processing.h5')

# make a mask of region to fit
mean_dp = np.mean(data.data, axis=(0,1))
mask = np.ones_like(mean_dp)
mask[mean_dp<200] = 0
mask[mean_dp>2400] = 0

p_init = [10,400,5,2,2,70,35,50,60,1,0,1]

coefs_array = amorph.fit_stack(data, p_init, mask=mask.astype(bool))
#TODO run this, but also take care of warnings, make a better mask before running