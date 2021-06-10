# Module for extracting radial distribution functions g(r) from a series of diffraction
# images. Process follows closely to procedure covered in:
# Cockayne, D.H.,Annu. Rev. Mater. Res. 37:15987 (2007).

import numpy as np
from scipy.special import erf
from scipy.fftpack import dst, idst
from ..utils import single_atom_scatter

def get_radial_intensity(polar_img, polar_mask):
    """
    Takes in a radial transformed image and the radial mask (if any) applied to that image.
    Designed to be compatible with polar-elliptical transforms from utils
    """
    yMean = np.mean(polar_img,axis=0)
    yNorm = np.mean(polar_mask,axis=0)
    sub = yNorm > 1e-1
    yMean[sub] = yMean[sub] / yNorm[sub]

    return yMean

def fit_scattering_factor(scale, elements, composition, q_arr,units):
    """
    Scale is linear factor
    Elements is an 1D array of atomic numbers.
    Composition is a 1D array, same length as elements, describing the average atomic
    composition of the sample. If the Q_coords is a 1D array of Fourier coordinates,
    given in inverse Angstroms. Units is a string of 'VA' or 'A', which returns the
    scattering factor in volt angtroms or in angstroms.
    """

    ##TODO: actually do fitting
    scatter = single_atom_scatter(elements,composition,q_arr,units)
    scatter.get_scattering_factor()
    return scale*scatter.fe**2

def get_phi(radialIntensity,scatter,q_arr):
    """
    ymean
    scale*scatter.fe**2
    """
    return ((radialIntensity-scatter)/scatter)*q_arr

def get_mask(left,right,midpoint,slopes,q_arr):
    """
    start is float
    stop is float
    midpoint is float
    slopes is [float,float]
    """
    vec = q_arr
    mask_left = (erf(slopes[0]*(vec-left)) + 1) / 2
    mask_right = (erf(slopes[1]*(right-vec)) + 1) / 2
    mid_idx = np.max(np.where(q_arr < midpoint))
    mask_left[mid_idx:] = 0
    mask_right[0:mid_idx] = 0

    return mask_left + mask_right

def get_rdf(phi, q_arr):
    """
    phi can be masked or not masked
    """
    sample_freq = 1/(q_arr[1]-q_arr[0]) #this assumes regularly spaced samples in q-space
    radius = (np.arange(q_arr.shape[0])/q_arr.shape[0])*sample_freq
    radius = radius*0.5 #scaling factor 
    radius += radius[1] #shift by minimum frequency, since first frequency sampled is finite

    G_r = dst(phi,type=2)
    g_r = G_r/(4*np.pi*radius) +  1
    return g_r,radius


