"""
Contains functions relating to polar-elliptical calculations.

This includes
  - transforming data from cartesian to polar-elliptical coordinates
  - converting between ellipse representations
  - radial and polar-elliptical radial integration

Functions for measuring/fitting elliptical distortions are found in
process/calibration/ellipse.py.  Functions for computing radial and
polar-elliptical radial backgrounds are found in process/preprocess/ellipse.py.

py4DSTEM uses 2 ellipse representations - one user-facing representation, and
one internal representation.  The user-facing represenation is in terms of the
following 5 parameters:

    x0,y0       the center of the ellipse
    a           the semimajor axis length
    b           the semiminor axis length
    theta       the (positive, right handed) tilt of the a-axis
                to the x-axis, in radians

Internally, fits are performed using the canonical ellipse parameterization,
in terms of the parameters (x0,y0,A,B,C):

    A(x-x0)^2 + B(x-x0)(y-y0) C(y-y0)^2 = 1

It is possible to convert between (a,b,theta) <--> (A,B,C) using
the convert_ellipse_params() and convert_ellipse_params_r() methods.

Transformation from cartesian to polar-elliptical space is done using

       x = x0 + a*r*cos(phi)*cos(theta) + b*r*sin(phi)*sin(theta)
       y = y0 + a*r*cos(phi)*sin(theta) - b*r*sin(phi)*cos(theta)

where (r,phi) are the polar-elliptical coordinates. All angular quantities are in
radians.
"""

import numpy as np

### Convert between representations

def convert_ellipse_params(A,B,C):
    """
    Converts ellipse parameters from canonical form (A,B,C) into semi-axis lengths and
    tilt (a,b,theta).
    See module docstring for more info.

    Args:
        A,B,C (floats): parameters of an ellipse in the form:
                             Ax^2 + Bxy + Cy^2 = 1

    Returns:
        (3-tuple): A 3-tuple consisting of:

        * **a**: (float) the semimajor axis length
        * **b**: (float) the semiminor axis length
        * **theta**: (float) the tilt of the ellipse semimajor axis with respect to
          the x-axis, in radians
    """
    val = np.sqrt((A-C)**2+B**2)
    b4a = B**2 - 4*A*C
    # Get theta
    if B == 0:
        if A<C:
            theta = 0
        else:
            theta = np.pi/2.
    else:
        theta = np.arctan2((C-A-val),B)
    # Get a,b
    a = - np.sqrt( -2*b4a*(A+C+val) ) / b4a
    b = - np.sqrt( -2*b4a*(A+C-val) ) / b4a
    a,b = max(a,b),min(a,b)
    return a,b,theta

def convert_ellipse_params_r(a,b,theta):
    """
    Converts from ellipse parameters (a,b,theta) to (A,B,C).
    See module docstring for more info.

    Args:
        a,b,theta (floats): parameters of an ellipse, where `a`/`b` are the
            semimajor/semiminor axis lengths, and theta is the tilt of the semimajor axis
            with respect to the x-axis, in radians.

    Returns:
        (3-tuple): A 3-tuple consisting of (A,B,C), the ellipse parameters in
            canonical form.
    """
    sin2,cos2 = np.sin(theta)**2,np.cos(theta)**2
    a2,b2 = a**2,b**2
    A = sin2/b2 + cos2/a2
    C = cos2/b2 + sin2/a2
    B = 2*(b2-a2)*np.sin(theta)*np.cos(theta)/(a2*b2)
    return A,B,C


### Polar elliptical transformation

def cartesian_to_polarelliptical_transform(
    cartesianData,
    p_ellipse,
    dr=1,
    dphi=np.radians(2),
    r_range=None,
    mask=None,
    maskThresh=0.99,
):
    """
    Transforms an array of data in cartesian coordinates into a data array in
    polar-elliptical coordinates.

    Discussion of the elliptical parametrization used can be found in the docstring
    for the process.utils.elliptical_coords module.

    Args:
        cartesianData (2D float array): the data in cartesian coordinates
        p_ellipse (5-tuple): specifies (qx0,qy0,a,b,theta), the parameters for the
            transformation. These are the same 5 parameters which are outputs
            of the elliptical fitting functions in the process.calibration
            module, e.g. fit_ellipse_amorphous_ring and fit_ellipse_1D. For
            more details, see the process.utils.elliptical_coords module docstring
        dr (float): sampling of the (r,phi) coords: the width of the bins in r
        dphi (float): sampling of the (r,phi) coords: the width of the bins in phi,
            in radians
        r_range (number or length 2 list/tuple or None): specifies the sampling of the
            (r,theta) coords.  Precise behavior which depends on the parameter type:
                * if None, autoselects max r value
                * if r_range is a number, specifies the maximum r value
                * if r_range is a length 2 list/tuple, specifies the min/max r values
        mask (2d array of bools): shape must match cartesianData; where mask==False,
            ignore these datapoints in making the polarElliptical data array
        maskThresh (float): the final data mask is calculated by converting mask (above)
            from cartesian to polar elliptical coords.  Due to interpolation, this
            results in some non-boolean values - this is converted back to a boolean
            array by taking polarEllipticalMask = polarTrans(mask) < maskThresh. Cells
            where polarTrans is less than 1 (i.e. has at least one masked NN) should
            generally be masked, hence the default value of 0.99.

    Returns:
        (3-tuple): A 3-tuple, containing:

            * **polarEllipticalData**: *(2D masked array)* a masked array containing
              the data and the data mask, in polarElliptical coordinates
            * **rr**: *(2D array)* meshgrid of the r coordinates
            * **pp**: *(2D array)* meshgrid of the phi coordinates
    """
    if mask is None:
        mask = np.ones_like(cartesianData, dtype=bool)
    assert (
        cartesianData.shape == mask.shape
    ), "Mask and cartesian data array shapes must match."
    assert len(p_ellipse) == 5, "p_ellipse must have length 5"

    # Get params
    qx0, qy0, a, b, theta = p_ellipse
    Nx, Ny = cartesianData.shape

    # Define r_range: 
    if r_range is None:
        #find corners of image
        corners = np.array([
                            [0,0],
                            [0,cartesianData.shape[0]],
                            [0,cartesianData.shape[1]],
                            [cartesianData.shape[0], cartesianData.shape[1]]
                            ])
        #find maximum corner distance
        r_min, r_max =0, np.ceil(
                            np.max(
                                np.sqrt(
                                    np.sum((corners -np.broadcast_to(np.array((qx0,qy0)), corners.shape))**2, axis = 1)
                                       )
                                   )
                                ).astype(int)
    else:
        try:
            r_min, r_max = r_range[0], r_range[1]
        except TypeError:
            r_min, r_max = 0, r_range

    # Define the r/phi coords
    r_bins = np.arange(r_min + dr / 2.0, r_max + dr / 2.0, dr)  # values are bin centers
    p_bins = np.arange(-np.pi + dphi / 2.0, np.pi + dphi / 2.0, dphi)
    rr, pp = np.meshgrid(r_bins, p_bins)
    Nr, Np = rr.shape

    # Get (qx,qy) corresponding to each (r,phi) in the newly defined coords
    xr = rr * np.cos(pp)
    yr = rr * np.sin(pp)
    qx = qx0 + xr * np.cos(theta) - yr * (b/a) * np.sin(theta)
    qy = qy0 + xr * np.sin(theta) + yr * (b/a) * np.cos(theta)

    # qx,qy are now shape (Nr,Np) arrays, such that (qx[r,phi],qy[r,phi]) is the point
    # in cartesian space corresponding to r,phi.  We now get the values for the final
    # polarEllipticalData array by interpolating values at these coords from the original
    # cartesianData array.

    transform_mask = (qx > 0) * (qy > 0) * (qx < Nx - 1) * (qy < Ny - 1)

    # Bilinear interpolation
    xF = np.floor(qx[transform_mask])
    yF = np.floor(qy[transform_mask])
    dx = qx[transform_mask] - xF
    dy = qy[transform_mask] - yF
    x_inds = np.vstack((xF, xF + 1, xF, xF + 1)).astype(int)
    y_inds = np.vstack((yF, yF, yF + 1, yF + 1)).astype(int)
    weights = np.vstack(
        ((1 - dx) * (1 - dy), (dx) * (1 - dy), (1 - dx) * (dy), (dx) * (dy))
    )
    transform_mask = transform_mask.ravel()
    polarEllipticalData = np.zeros(Nr * Np)
    polarEllipticalData[transform_mask] = np.sum(
        cartesianData[x_inds, y_inds] * weights, axis=0
    )
    polarEllipticalData = np.reshape(polarEllipticalData, (Nr, Np))

    # Transform mask
    polarEllipticalMask = np.zeros(Nr * Np)
    polarEllipticalMask[transform_mask] = np.sum(mask[x_inds, y_inds] * weights, axis=0)
    polarEllipticalMask = np.reshape(polarEllipticalMask, (Nr, Np))

    polarEllipticalData = np.ma.array(
        data=polarEllipticalData, mask=polarEllipticalMask < maskThresh
    )
    return polarEllipticalData, rr, pp


### Radial integration

def radial_elliptical_integral(ar, dr, p_ellipse):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of
    dr.

    Args:
        ar (2d array): the data
        dr (number): the r sampling
        p_ellipse (5-tuple): the parameters (x0,y0,a,b,theta) for the ellipse

    Returns:
        (2-tuple): A 2-tuple containing:

            * **rbin_centers**: *(1d array)* the bins centers of the radial integral
            * **radial_integral**: *(1d array)* the radial integral
        radial_integral (1d array) the radial integral
    """
    x0, y0 = p_ellipse[0], p_ellipse[1]
    rmax = int(
        max(
            (
                np.hypot(x0, y0),
                np.hypot(x0, ar.shape[1] - y0),
                np.hypot(ar.shape[0] - x0, y0),
                np.hypot(ar.shape[0] - x0, ar.shape[1] - y0),
            )
        )
    )
    polarAr, rr, pp = cartesian_to_polarelliptical_transform(
        ar, p_ellipse=p_ellipse, dr=dr, dphi=np.radians(2), r_range=rmax
    )
    radial_integral = np.sum(polarAr, axis=0)
    rbin_centers = rr[0, :]
    return rbin_centers,radial_integral


def radial_integral(ar, x0, y0, dr):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of dr.

    Args:
        ar (2d array): the data
        x0,y0 (floats): the origin
        dr (number): radial step size

    Returns:
        (2-tuple): A 2-tuple containing:

            * **rbin_centers**: *(1d array)* the bins centers of the radial integral
            * **radial_integral**: *(1d array)* the radial integral
    """
    return radial_elliptical_integral(ar, dr, (x0,y0,1,1,0))

