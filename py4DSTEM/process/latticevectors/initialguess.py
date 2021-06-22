# Obtain an initial guess at the lattice vectors

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import radon

from ..utils import get_maxima_1D

def get_radon_scores(braggvectormap, mask=None, N_angles=200, sigma=2, minSpacing=2,
                                                           minRelativeIntensity=0.05):
    """
    Calculates a score function, score(angle), representing the likelihood that angle is
    a principle lattice direction of the lattice in braggvectormap.

    The procedure is as follows:
    If mask is not None, ignore any data in braggvectormap where mask is False. Useful
    for removing the unscattered beam, which can dominate the results.
    Take the Radon transform of the (masked) Bragg vector map.
    For each angle, get the corresponding slice of the sinogram, and calculate its score.
    If we let R_theta(r) be the sinogram slice at angle theta, and where r is the
    sinogram position coordinate, then the score of the slice is given by
         score(theta) = sum_i(R_theta(r_i)) / N_i
    Here, r_i are the positions r of all local maxima in R_theta(r), and N_i is the
    number of such maxima.  Thus the score is large when there are few maxima which are
    high intensity.

    Args:
        braggvectormap (ndarray): the Bragg vector map
        mask (ndarray of bools): ignore data in braggvectormap wherever mask==False
        N_angles (int): the number of angles at which to calculate the score
        sigma (float): smoothing parameter for local maximum identification
        minSpacing (float): if two maxima are found in a radon slice closer than
            minSpacing, the dimmer of the two is removed
        minRelativeIntensity (float): maxima in each radon slice dimmer than
            minRelativeIntensity compared to the most intense maximum are removed

    Returns:
        (3-tuple) A 3-tuple containing:

            * **scores**: *(ndarray, len N_angles, floats)* the scores for each angle
            * **thetas**: *(ndarray, len N_angles, floats)* the angles, in radians
            * **sinogram**: *(ndarray)* the radon transform of braggvectormap*mask
    """
    # Get sinogram
    thetas = np.linspace(0,180,N_angles)
    if mask is not None:
        sinogram = radon(braggvectormap*mask, theta=thetas, circle=False)
    else:
        sinogram = radon(braggvectormap, theta=thetas, circle=False)

    # Get scores
    N_maxima = np.empty_like(thetas)
    total_intensity = np.empty_like(thetas)
    for i in range(len(thetas)):
        theta = thetas[i]

        # Get radon transform slice
        ind = np.argmin(np.abs(thetas-theta))
        sinogram_theta = sinogram[:,ind]
        sinogram_theta = gaussian_filter(sinogram_theta,2)

        # Get maxima
        maxima = get_maxima_1D(sinogram_theta,sigma,minSpacing,minRelativeIntensity)

        # Calculate metrics
        N_maxima[i] = len(maxima)
        total_intensity[i] = np.sum(sinogram_theta[maxima])
    scores = total_intensity/N_maxima

    return scores, np.radians(thetas), sinogram

def get_lattice_directions_from_scores(thetas, scores, sigma=2, minSpacing=2,
                                       minRelativeIntensity=0.05, index1=0, index2=0):
    """
    Get the lattice directions from the scores of the radon transform slices.

    Args:
        thetas (ndarray): the angles, in radians
        scores (ndarray): the scores
        sigma (float): gaussian blur for local maxima identification
        minSpacing (float): minimum spacing for local maxima identification
        minRelativeIntensity (float): minumum intensity, relative to the brightest
            maximum, for local maxima identification
        index1 (int): specifies which local maximum to use for the first lattice
            direction, in order of maximum intensity
        index2 (int): specifies the local maximum for the second lattice direction

    Returns:
        (2-tuple) A 2-tuple containing:

            * **theta1**: *(float)* the first lattice direction, in radians
            * **theta2**: *(float)* the second lattice direction, in radians
    """
    assert len(thetas)==len(scores), "Size of thetas and scores must match"

    # Get first lattice direction
    maxima1 = get_maxima_1D(scores, sigma, minSpacing, minRelativeIntensity) # Get maxima
    thetas_max1 = thetas[maxima1]
    scores_max1 = scores[maxima1]
    dtype = np.dtype([('thetas',thetas.dtype),('scores',scores.dtype)]) # Sort by intensity
    ar_structured = np.empty(len(thetas_max1),dtype=dtype)
    ar_structured['thetas'] = thetas_max1
    ar_structured['scores'] = scores_max1
    ar_structured = np.sort(ar_structured, order='scores')[::-1]
    theta1 = ar_structured['thetas'][index1]                            # Get direction 1

    # Apply sin**2 damping
    scores_damped = scores*np.sin(thetas-theta1)**2

    # Get second lattice direction
    maxima2 = get_maxima_1D(scores_damped, sigma, minSpacing, minRelativeIntensity) # Get maxima
    thetas_max2 = thetas[maxima2]
    scores_max2 = scores[maxima2]
    dtype = np.dtype([('thetas',thetas.dtype),('scores',scores.dtype)]) # Sort by intensity
    ar_structured = np.empty(len(thetas_max2),dtype=dtype)
    ar_structured['thetas'] = thetas_max2
    ar_structured['scores'] = scores_max2
    ar_structured = np.sort(ar_structured, order='scores')[::-1]
    theta2 = ar_structured['thetas'][index2]                            # Get direction 2

    return theta1, theta2

def get_lattice_vector_lengths(u_theta, v_theta, thetas, sinogram, spacing_thresh=1.5,
                                        sigma=1, minSpacing=2, minRelativeIntensity=0.1):
    """
    Gets the lengths of the two lattice vectors from their angles and the sinogram.

    First, finds the spacing between peaks in the sinogram slices projected down the u-
    and v- directions, u_proj and v_proj.  Then, finds the lengths by taking::

        |u| = v_proj/sin(u_theta-v_theta)
        |v| = u_proj/sin(u_theta-v_theta)

    The most important thresholds for this function are spacing_thresh, which discards
    any detected spacing between adjacent radon projection peaks which deviate from the
    median spacing by more than this fraction, and minRelativeIntensity, which discards
    detected maxima (from which spacings are then calculated) below this threshold
    relative to the brightest maximum.

    Args:
        u_theta (float): the angle of u, in radians
        v_theta (float): the angle of v, in radians
        thetas (ndarray): the angles corresponding to the sinogram
        sinogram (ndarray): the sinogram
        spacing_thresh (float): ignores spacings which are greater than spacing_thresh
            times the median spacing
        sigma (float): gaussian blur for local maxima identification
        minSpacing (float): minimum spacing for local maxima identification
        minRelativeIntensity (float): minumum intensity, relative to the brightest
            maximum, for local maxima identification

    Returns:
        (2-tuple) A 2-tuple containing:

            * **u_length**: *(float)* the length of u, in pixels
            * **v_length**: *(float)* the length of v, in pixels
    """
    assert len(thetas)==sinogram.shape[1], "thetas must corresponding to the number of sinogram projection directions."

    # Get u projected spacing
    ind = np.argmin(np.abs(thetas-u_theta))
    sinogram_slice = sinogram[:,ind]
    maxima = get_maxima_1D(sinogram_slice, sigma, minSpacing, minRelativeIntensity)
    spacings = np.sort(np.arange(sinogram_slice.shape[0])[maxima])
    spacings = spacings[1:] - spacings[:-1]
    mask = np.array([max(i,np.median(spacings))/min(i,np.median(spacings)) for i in spacings]) < spacing_thresh
    spacings = spacings[mask]
    u_projected_spacing = np.mean(spacings)

    # Get v projected spacing
    ind = np.argmin(np.abs(thetas-v_theta))
    sinogram_slice = sinogram[:,ind]
    maxima = get_maxima_1D(sinogram_slice, sigma, minSpacing, minRelativeIntensity)
    spacings = np.sort(np.arange(sinogram_slice.shape[0])[maxima])
    spacings = spacings[1:] - spacings[:-1]
    mask = np.array([max(i,np.median(spacings))/min(i,np.median(spacings)) for i in spacings]) < spacing_thresh
    spacings = spacings[mask]
    v_projected_spacing = np.mean(spacings)

    # Get u and v lengths
    sin_uv = np.sin(np.abs(u_theta-v_theta))
    u_length = v_projected_spacing / sin_uv
    v_length = u_projected_spacing / sin_uv

    return u_length, v_length



