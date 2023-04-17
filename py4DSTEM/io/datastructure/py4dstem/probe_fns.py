# Functions to become Probe methods

import numpy as np
from py4DSTEM.io.datastructure.emd import Metadata




# Kernel generation

def get_kernel(
    self,
    mode = 'flat',
    returncalc = True,
    **kwargs
    ):
    """
    Creates a kernel from the probe for cross-correlative template matching.

    Precise behavior and valid keyword arguments depend on the `mode`
    selected.  In each case, the center of the probe is shifted to the
    origin and the kernel normalized such that it sums to 1. In 'flat'
    mode, this is the only processing performed. In the remaining modes,
    some additional processing is performed which adds a ring of
    negative intensity around the central probe, which results in
    edge-filetering-like behavior during cross correlation. Valid modes,
    and the required additional kwargs, if any, for each, are:

        - 'flat': creates a flat probe kernel. For bullseye or other
            structured probes, this mode is recommended. No required
            arguments, optional arg `origin` (2 tuple)
        - 'gaussian': subtracts a gaussian with a width of standard
            deviation `sigma`, which is a required argument. Optional
            arg `origin`.
        - 'sigmoid': subtracts an annulus with inner and outer radii
            of (ri,ro) and a sine-squared sigmoid radial profile from
            the probe template. Required arg: `radii` (2 tuple). Optional
            args `origin` (2-tuple)
        - 'sigmoid_log': subtracts an annulus with inner and outer radii
            of (ri,ro) and a logistic sigmoid radial profile from
            the probe template. Required arg: `radii` (2 tuple). Optional
            args `origin` (2-tuple)

    Returns:
        (2D array)
    """

    # perform computation
    from py4DSTEM.process.probe import get_kernel
    kern = get_kernel(
        self.probe,
        mode = mode,
        **kwargs
    )

    # add to the Probe
    self.kernel = kern

    # Set metadata
    md = Metadata(name='kernel')
    md['mode'] = mode
    for k,v in kwargs.items():
        md[k] = v
    self.metadata = md

    # return
    if returncalc:
        return kern


def get_probe_size(
    self,
    thresh_lower=0.01,
    thresh_upper=0.99,
    N=100,
    mode = None,
    plot = True,
    returncal = True,
    **kwargs,
    ):
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern
    DP with a linspace of N thresholds from thresh_lower to thresh_upper, measured
    relative to the maximum intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r
    should change very little over a wide range of intermediate values of the threshold.
    The range in which r is trustworthy is found by taking the derivative of r(thresh)
    and finding identifying where it is small.  The radius is taken to be the mean of
    these r values. Using the threshold corresponding to this r, a mask is created and
    the CoM of the DP times this mask it taken.  This is taken to be the origin x0,y0.

    Args:
        mode (str or array): specifies the diffraction pattern in which to find the 
            central disk. A position averaged, or shift-corrected and averaged,
            DP works best. If mode is None, the diffraction pattern stored in the
            tree from 'get_dp_mean' is used. If mode is a string it specifies the name of
            another virtual diffraction pattern in the tree. If mode is an array, the array
            is used to calculate probe size.
        thresh_lower (float, 0 to 1): the lower limit of threshold values
        thresh_upper (float, 0 to 1): the upper limit of threshold values
        N (int): the number of thresholds / masks to use
        plot (bool): if True plots results
        plot_params(dict): dictionary to modify defaults in plot
        return_calc (bool): if True returns 3-tuple described below

    Returns:
        (3-tuple): A 3-tuple containing:

            * **r**: *(float)* the central disk radius, in pixels
            * **x0**: *(float)* the x position of the central disk center
            * **y0**: *(float)* the y position of the central disk center
    """
    #perform computation        
    from py4DSTEM.process.calibration import get_probe_size
    from py4DSTEM.io.datastructure.py4dstem.calibration import Calibration

    x = get_probe_size(
        self.probe,
        thresh_lower = thresh_lower,
        thresh_upper = thresh_upper,
        N = N,
    )

    # try to add to calibration
    try:
        self.calibration.set_probe_param(x)
    except AttributeError:
        # should a warning be raised?
        pass

    #plot results 
    if plot:
        from py4DSTEM.visualize import show_circles
        show_circles(
            self.probe,
            (x[1], x[2]),
            x[0],
            vmin = 0,
            vmax = 1,
            **kwargs
        )

    # return
    if returncal:
        return x




