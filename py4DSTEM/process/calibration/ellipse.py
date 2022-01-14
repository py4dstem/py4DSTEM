"""
Functions related to elliptical calibration, such as fitting elliptical
distortions.

The user-facing representation of ellipses is in terms of the following 5
parameters:

    x0,y0       the center of the ellipse
    a           the semimajor axis length
    b           the semiminor axis length
    theta       the (positive, right handed) tilt of the a-axis
                to the x-axis, in radians

More details about the elliptical parameterization used can be found in
the module docstring for process/utils/elliptical_coords.py.
"""

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import gaussian_filter
from ..utils import convert_ellipse_params, convert_ellipse_params_r
from ..utils import get_CoM, radial_integral
from ...io import PointListArray

###### Fitting a 1d elliptical curve to a 2d array, e.g. a Bragg vector map ######

def fit_ellipse_1D(ar,center,fitradii,mask=None):
    """
    For a 2d array ar, fits a 1d elliptical curve to the data inside an annulus centered
    at `center` with inner and outer radii at `fitradii`.  The data to fit make optionally
    be additionally masked with the boolean array mask. See module docstring for more info.

    Args:
        ar (ndarray): array containing the data to fit
        center (2-tuple of floats): the center (x0,y0) of the annular fitting region
        fitradii (2-tuple of floats): inner and outer radii (ri,ro) of the fit region
        mask (ar-shaped ndarray of bools): ignore data wherever mask==True

    Returns:
        (5-tuple of floats): A 5-tuple containing the ellipse parameters:
            * **x0**: the center x-position
            * **y0**: the center y-position
            * **a**: the semimajor axis length
            * **b**: the semiminor axis length
            * **theta**: the tilt of the ellipse semimajor axis with respect to the
              x-axis, in radians
    """
    # Unpack inputs
    x0,y0 = center
    ri,ro = fitradii

    # Get the datapoints to fit
    yy,xx = np.meshgrid(np.arange(ar.shape[1]),np.arange(ar.shape[0]))
    rr = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    _mask = (rr>ri) * (rr<=ro)
    if mask is not None:
        _mask *= mask==False
    xs,ys = np.nonzero(_mask)
    vals = ar[_mask]

    # Get initial parameters guess
    p0 = [x0,y0,(2/(ri+ro))**2,0,(2/(ri+ro))**2]

    # Fit
    x,y,A,B,C = leastsq(ellipse_err, p0, args=(xs,ys,vals))[0]

    # Convert ellipse params
    a,b,theta = convert_ellipse_params(A,B,C)

    return x,y,a,b,theta

def ellipse_err(p, x, y, val):
    """
    For a point (x,y) in a 2d cartesian space, and a function taking the value
    val at point (x,y), and some 1d ellipse in this space given by
            ``A(x-x0)^2 + B(x-x0)(y-y0) + C(y-y0)^2 = 1``
    this function computes the error associated with the function's value at (x,y)
    given by its deviation from the ellipse times val.

    Note that this function is for internal use, and uses ellipse parameters `p`
    given in canonical form (x0,y0,A,B,C), which is different from the ellipse
    parameterization used in all the user-facing functions, for reasons of
    numerical stability.
    """
    x,y = x-p[0],y-p[1]
    return (p[2]*x**2 + p[3]*x*y + p[4]*y**2 - 1)*val


###### Fitting from amorphous diffraction rings ######

def fit_ellipse_amorphous_ring(data,center,fitradii,p0=None,mask=None):
    """
    Fit the amorphous halo of a diffraction pattern, including any elliptical distortion.

    The fit function is::

        f(x,y; I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C) =
            Norm(r; I0,sigma0,0) +
            Norm(r; I1,sigma1,R)*Theta(r-R)
            Norm(r; I1,sigma2,R)*Theta(R-r) + c_bkgd

    where

        * (x,y) are cartesian coordinates,
        * r is the radial coordinate,
        * (I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,R,B,C) are parameters,
        * Norm(x;I,s,u) is a gaussian in the variable x with maximum amplitude I,
          standard deviation s, and mean u
        * Theta(x) is a Heavyside step function
        * R is the radial center of the double sided gaussian, derived from (A,B,C)
          and set to the mean of the semiaxis lengths

    The function thus contains a pair of gaussian-shaped peaks along the radial
    direction of a polar-elliptical parametrization of a 2D plane. The first gaussian is
    centered at the origin. The second gaussian is centered about some finite R, and is
    'two-faced': it's comprised of two half-gaussians of different standard deviations,
    stitched together at their mean value of R. This Janus (two-faced ;p) gaussian thus
    comprises an elliptical ring with different inner and outer widths.

    The parameters of the fit function are

        * I0: the intensity of the first gaussian function
        * I1: the intensity of the Janus gaussian
        * sigma0: std of first gaussian
        * sigma1: inner std of Janus gaussian
        * sigma2: outer std of Janus gaussian
        * c_bkgd: a constant offset
        * x0,y0: the origin
        * A,B,C: The ellipse parameters, in the form Ax^2 + Bxy + Cy^2 = 1

    Args:
        data (2d array): the data
        center (2-tuple of numbers): the center (x0,y0)
        fitradii (2-tuple of numbers): the inner and outer radii of the fitting annulus
        p0 (11-tuple): initial guess parameters. If p0 is None, the function will compute
            a guess at all parameters. If p0 is a 11-tuple it must be populated by some
            mix of numbers and None; any parameters which are set to None will be guessed
            by the function.  The parameters are the 11 parameters of the fit function
            described above, p0 = (I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C).
            Note that x0,y0 are redundant; their guess values are the x0,y0 values passed
            to the main function, but if they are passed as elements of p0 these will
            take precendence.
        mask (2d array of bools): only fit to datapoints where mask is True

    Returns:
        (2-tuple comprised of a 5-tuple and an 11-tuple): Returns a 2-tuple.

        The first element is the ellipse parameters need to elliptically parametrize
        diffraction space, and is itself a 5-tuple:

            * **x0**: x center
            * **y0**: y center,
            * **a**: the semimajor axis length
            * **b**: the semiminor axis length
            * **theta**: tilt of a-axis w.r.t x-axis, in radians

        The second element is the full set of fit parameters to the double sided gaussian
        function, described above, and is an 11-tuple
    """
    if mask is None:
        mask = np.ones_like(data).astype(bool)
    assert data.shape == mask.shape, "data and mask must have same shapes."
    x0,y0 = center
    ri,ro = fitradii

    # Get data mask
    Nx,Ny = data.shape
    yy,xx = np.meshgrid(np.arange(Ny),np.arange(Nx))
    rr = np.hypot(xx-x0,yy-y0)
    _mask = ((rr>ri)*(rr<ro)).astype(bool)
    _mask *= mask

    # Make coordinates, get data values
    x_inds, y_inds = np.nonzero(_mask)
    vals = data[_mask]

    # Get initial parameter guesses
    I0 = np.max(data)
    I1 = np.max(data*mask)
    sigma0 = ri/2.
    sigma1 = (ro-ri)/4.
    sigma2 = (ro-ri)/4.
    c_bkgd = np.min(data)
    # To guess R, we take a radial integral
    q,radial_profile = radial_integral(data,x0,y0,1)
    R = q[(q>ri)*(q<ro)][np.argmax(radial_profile[(q>ri)*(q<ro)])]
    # Initial guess at A,B,C
    A,B,C = convert_ellipse_params_r(R,R,0)

    # Populate initial parameters
    p0_guess = tuple([I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C])
    if p0 is None:
        _p0 = p0_guess
    else:
        assert len(p0)==11
        _p0 = tuple([p0_guess[i] if p0[i] is None else p0[i] for i in range(len(p0))])

    # Perform fit
    p = leastsq(double_sided_gaussian_fiterr, _p0, args=(x_inds, y_inds, vals))[0]

    # Return
    _x0,_y0 = p[6],p[7]
    _A,_B,_C = p[8],p[9],p[10]
    _a,_b,_theta = convert_ellipse_params(_A,_B,_C)
    return (_x0,_y0,_a,_b,_theta),p

def double_sided_gaussian_fiterr(p, x, y, val):
    """
    Returns the fit error associated with a point (x,y) with value val, given parameters p.
    """
    return double_sided_gaussian(p, x, y) - val


def double_sided_gaussian(p, x, y):
    """
    Return the value of the double-sided gaussian function at point (x,y) given
    parameters p, described in detail in the fit_ellipse_amorphous_ring docstring.
    """
    # Unpack parameters
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C = p
    a,b,theta = convert_ellipse_params(A,B,C)
    R = np.mean((a,b))
    R2 = R**2
    A,B,C = A*R2,B*R2,C*R2
    r2 = A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2
    r = np.sqrt(r2) - R

    return (
        I0 * np.exp(-r2 / (2 * sigma0 ** 2))
        + I1 * np.exp(-r ** 2 / (2 * sigma1 ** 2)) * np.heaviside(-r, 0.5)
        + I1 * np.exp(-r ** 2 / (2 * sigma2 ** 2)) * np.heaviside(r, 0.5)
        + c_bkgd
    )


### Correct Bragg peak positions, making a circular coordinate system

def correct_braggpeak_elliptical_distortions(braggpeaks,p_ellipse,centered=True):
    """
    Given some elliptical distortions with ellipse parameters p and some measured
    PointListArray of Bragg peak positions braggpeaks, returns the elliptically corrected
    Bragg peaks.

    Args:
        braggpeaks (PointListArray): the Bragg peaks
        p_ellipse (5-tuple): the ellipse parameters (x0,y0,a,b,theta)
        centered (bool): if True, assumes that the braggpeaks PointListArray has been
            centered, and uses (x0,y0)=(0,0). Otherwise, uses the (x0,y0) from
            `p_ellipse`

    Returns:
        braggpeaks_corrected    (PointListArray) the corrected Bragg peaks
    """
    assert(isinstance(braggpeaks,PointListArray))

    # Unpack parameters
    x0,y0,a,b,theta = p_ellipse
    if centered:
        x0,y0 = 0,0

    # Get the transformation matrix
    e = b/a
    sint, cost = np.sin(theta-np.pi/2.), np.cos(theta-np.pi/2.)
    T = np.array(
            [
                [e*sint**2 + cost**2, sint*cost*(1-e)],
                [sint*cost*(1-e), sint**2 + e*cost**2]
            ]
        )

    # Correct distortions
    braggpeaks_corrected = braggpeaks.copy(name=braggpeaks.name + "_ellipsecorrected")
    for Rx in range(braggpeaks_corrected.shape[0]):
        for Ry in range(braggpeaks_corrected.shape[1]):
            pointlist = braggpeaks_corrected.get_pointlist(Rx, Ry)
            x, y = pointlist.data["qx"] - x0, pointlist.data["qy"] - y0
            xyar_i = np.vstack([x, y])
            xyar_f = np.matmul(T, xyar_i)
            pointlist.data["qx"] = xyar_f[0, :] + x0
            pointlist.data["qy"] = xyar_f[1, :] + y0
    return braggpeaks_corrected


### Fit an ellipse to crystalline scattering with a known angle between peaks

def constrain_degenerate_ellipse(data, p_ellipse, r_inner, r_outer, phi_known, fitrad=6):
    """
    When fitting an ellipse to data containing 4 diffraction spots in a narrow annulus
    about the central beam, the answer is degenerate: an infinite number of ellipses
    correctly fit this data.  Starting from one ellipse in the degenerate family of
    ellipses, this function selects the ellipse which will yield a final angle of
    phi_known between a pair of the diffraction peaks after performing elliptical
    distortion correction.

    Note that there are two possible angles which phi_known might refer to, because the
    angle of interest is well defined up to a complementary angle.  This function is
    written such that phi_known should be the smaller of these two angles.

    Args:
        data (ndarray) the data to fit, typically a Bragg vector map
        p_ellipse (5-tuple): the ellipse parameters (x0,y0,a,b,theta)
        r_inner (float): the fitting annulus inner radius
        r_outer (float): the fitting annulus outer radius
        phi_known (float): the known angle between a pair of diffraction peaks, in
            radians
        fitrad (float): the region about the fixed data point used to refine its position

    Returns:
        (2-tuple): A 2-tuple containing:

            * **a_constrained**: *(float)* the first semiaxis of the selected ellipse
            * **b_constrained**: *(float)* the second semiaxis of the selected ellipse
    """
    # Unpack ellipse params
    x,y,a,b,theta = p_ellipse

    # Get 4 constraining points
    xs,ys = np.zeros(4),np.zeros(4)
    yy,xx = np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
    rr = np.sqrt((xx-x)**2+(yy-y)**2)
    annular_mask = (rr>r_inner)*(rr<=r_outer)
    data_temp = np.zeros_like(data)
    data_temp[annular_mask] = data[annular_mask]
    for i in range(4):
        x_constr,y_constr = np.unravel_index(np.argmax(gaussian_filter(data_temp,2)),(data.shape[0],data.shape[1]))
        rr = np.sqrt((xx-x_constr)**2+(yy-y_constr)**2)
        mask = rr<fitrad
        xs[i],ys[i] = get_CoM(data*mask)
        data_temp[mask] = 0

    # Transform constraining points coordinate system
    xs -= x
    ys -= y
    T = np.squeeze(np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]))
    xs,ys = np.matmul(T,np.array([xs,ys]))

    # Get symmetrized constraining point
    angles = np.arctan2(ys,xs)
    distances = np.hypot(xs,ys)
    angle = np.mean(np.min(np.vstack([np.abs(angles),np.pi-np.abs(angles)]),axis=0))
    distance = np.mean(distances)
    x_fixed,y_fixed = distance*np.cos(angle),distance*np.sin(angle)

    # Get semiaxes a,b for the specified theta
    t = x_fixed/(a*np.cos(phi_known/2.))
    a_constrained = a*t
    b_constrained = np.sqrt(y_fixed**2/(1-(x_fixed/(a_constrained))**2))

    return a_constrained, b_constrained




