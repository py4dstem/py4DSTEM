# Contains methods relating to polar-elliptical calculations.
#
# This includes
#   - measuring / fitting elliptical distortions
#   - transforming data from cartesian to polar-elliptical coordinates
#   - correcting Bragg peak positions for elliptical distortions
#   - converting between ellipse representations
#
# We define the transformation from cartesian to polar-elliptical coordinates by:
#
#       x = x0 + A*r*cos(theta)*cos(phi) + B*r*sin(theta)*sin(phi)
#       y = y0 + B*r*sin(theta)*sin(phi) - A*r*cos(theta)*cos(phi)
#
# All angular quantities are in radians.

import numpy as np
from scipy.optimize import leastsq, least_squares
import matplotlib.pyplot as plt


def cartesianDataAr_to_polarEllipticalDataAr(
    cartesianData,
    params,
    dr=1,
    dtheta=np.radians(2),
    r_range=512,
    mask=None,
    maskThresh=0.99,
):
    """
    Transforms an array of data in cartesian coordinates into a data array in polar elliptical
    coordinates.

    If the cartesian coordinates are (qx,qy), then the parametrization for the polar-elliptical
    coordinate system is

        qx = qx0 + A*r*cos(theta)*cos(phi) - B*r*sin(theta)*sin(phi)
        qy = qy0 + B*r*cos(theta)*sin(phi) + A*r*sin(theta)*cos(phi)

    where the final coordinates are (r,theta), and the parameters are (qx0,qy0,A,B,phi). Physically,
    this corresponds to elliptical coordinates centered at (qx0,qy0), stretched by A/B along the
    semimajor axes, and with those axes tilted by phi with respect to the (qx,qy) axes.

    Accepts:
        cartesianData   (2D float array) the data in cartesian coordinates
        params          (5-tuple) specifies (qx0,qy0,A,B,phi), the parameters for the transformation
        dr              (float) sampling of the (r,theta) coords: the width of the bins in r
        dtheta          (float) sampling of the (r,theta) coords: the width of the bins in theta,
                        in radians
        r_range         (float) sampling of the (r,theta) coords:
                        if r_range is a number, specifies the maximum r value
                        if r_range is a length 2 list/tuple, specifies the min/max r values
                        if None, autoselects max r value
        mask            (2D bool array) shape must match cartesianData; where mask==False, ignore
                        these datapoints in making the polarElliptical data array
        maskThresh      (float) the final data mask is calculated by converting mask (above) from
                        cartesian to polar elliptical coords.  Due to interpolation, this results in
                        some non-boolean values - this is converted back to a boolean array by
                        taking polarEllipticalMask = polarTrans(mask) < maskThresh. Cells where
                        polarTrans is less than 1 (i.e. has at least one masked NN) should generally
                        be masked, hence the default value of 0.99.

    Returns:
        polarEllipticalData     (2D masked array) a masked array containing
                                    data: the data in polarElliptical coordinates
                                    mask: the data mask, in polarElliptical coordinates
        rr, tt                  (2D arrays) meshgrid of the (r,theta) coordinates
    """
    if mask is None:
        mask = np.ones_like(cartesianData, dtype=bool)
    assert (
        cartesianData.shape == mask.shape
    ), "Mask and cartesian data array shapes must match."
    assert len(params) == 5, "params must have length 5"
    try:
        r_min, r_max = r_range[0], r_range[1]
    except TypeError:
        r_min, r_max = 0, r_range
    Nx, Ny = cartesianData.shape

    # Get params
    qx0, qy0, A, B, phi = params

    # Define the r/theta coords
    r_bins = np.arange(r_min + dr / 2.0, r_max + dr / 2.0, dr)  # values are bin centers
    t_bins = np.arange(-np.pi + dtheta / 2.0, np.pi + dtheta / 2.0, dtheta)
    rr, tt = np.meshgrid(r_bins, t_bins)
    Nr, Nt = rr.shape

    # Get (qx,qy) corresponding to each (r,theta) in the newly defined coords
    xr = rr * np.cos(tt)
    yr = rr * np.sin(tt)
    qx = qx0 + xr * A * np.cos(phi) - yr * B * np.sin(phi)
    qy = qy0 + xr * A * np.sin(phi) + yr * B * np.cos(phi)

    # qx,qy are now shape (Nr,Ntheta) arrays, such that (qx[r,theta],qy[r,theta]) is the point
    # in cartesian space corresponding to r,theta.  We now get the values for the final
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
    polarEllipticalData = np.zeros(Nr * Nt)
    polarEllipticalData[transform_mask] = np.sum(
        cartesianData[x_inds, y_inds] * weights, axis=0
    )
    polarEllipticalData = np.reshape(polarEllipticalData, (Nr, Nt))

    # Transform mask
    polarEllipticalMask = np.zeros(Nr * Nt)
    polarEllipticalMask[transform_mask] = np.sum(mask[x_inds, y_inds] * weights, axis=0)
    polarEllipticalMask = np.reshape(polarEllipticalMask, (Nr, Nt))

    polarEllipticalData = np.ma.array(
        data=polarEllipticalData, mask=polarEllipticalMask < maskThresh
    )
    return polarEllipticalData, rr, tt


def convert_ellipse_params(A0, B0, C0):
    """
    Converts ellipse parameters from canonical form into semi-axis lengths and tilt.

    Accepts:
        A0,B0,C0    (floats) parameters of an ellipse in canonical form, i.e.:
                                A0*x^2 + B0*x*y + C0*y^2 = 1

    Returns:
        A,B         (floats) the semi-axis lengths
        phi         (float) the tilt of the ellipse semi-axes, in radians
    """
    if A0 == C0:
        x = B0
        phi = np.pi / 4.0 * np.sign(B0)
    else:
        x = (A0 - C0) * np.sqrt(1 + (B0 / (A0 - C0)) ** 2)
        phi = 0.5 * np.arctan(B0 / (A0 - C0))
    A = np.sqrt(2 / (A0 + C0 + x))
    B = np.sqrt(2 / (A0 + C0 - x))
    return A, B, phi


def fit_ellipse_inside_annulus(
    ar, x0, y0, r_inner, r_outer, datamask=None, output=None
):
    """
    Fits an ellipse to the data in an annular region of array ar, centered at (x0,y0) and with inner
    and outer radii of r_inner, r_outer, respectively.  Returns the ellipse parameters.

    Accepts:
        ar          (array) array containing the data to fit
        x0,y0       (floats) the center of the annular fitting region
        r_inner     (float) inner radius of the fit region
        r_outer     (float) outer radius of the fit region
        datamask    (ar-shaped array of bools) ignore datapoints where mask==False
        output      (str or None) controls the parametrization of the output

    Returns
        params      (5-tuple) the ellipse parameters. Parametrization depends on the flag 'output'.
                    if output != ('canonical' or 'both'):
                        returns (x0,y0,A,B,phi), using
                            x = x0 + A*r*cos(theta)*cos(phi) + B*r*sin(theta)*sin(phi)
                            y = y0 + B*r*sin(theta)*sin(phi) - A*r*cos(theta)*cos(phi)
                    if output == 'canonical':
                        returns (x0,y0,A0,B0,C0), using
                            A0*(x-x0)^2 + B0*(x-x0)*(y-y0) + C0*(y-y0)^2 = 1
                    if output == 'both':
                        returns both of the parametrizations above, in the order written.
    """
    # Get the datapoints to fit
    yy, xx = np.meshgrid(np.arange(ar.shape[1]), np.arange(ar.shape[0]))
    rr = np.hypot(xx - x0, yy - y0)
    mask = (rr > r_inner) * (rr <= r_outer)
    if datamask is not None:
        mask *= datamask
    xs, ys = np.nonzero(mask)
    vals = ar[mask]

    # Get initial parameters guess
    p0_0 = x0
    p0_1 = y0
    p0_2 = (2 / (r_inner + r_outer)) ** 2
    p0_3 = 0
    p0_4 = (2 / (r_inner + r_outer)) ** 2
    p0 = (p0_0, p0_1, p0_2, p0_3, p0_4)

    # Fit
    p2 = leastsq(ellipse_err, p0, args=(xs, ys, vals))[0]

    # Convert between parametrizations
    x, y, A0, B0, C0 = p2
    A, B, phi = convert_ellipse_params(A0, B0, C0)
    p1 = (x, y, A, B, phi)

    if output == "canonical":
        return p2
    elif output == "both":
        return p1, p2
    else:
        return p1


def ellipse_err(p, x, y, val):
    """
    Returns the error associated with point (x,y), which has value val, with respect to the ellipse
    given by parameters p.  p is a 5-tuple of the form (x,y,A0,B0,C0) specifying the ellipse
        A0*(x-x0)^2 + B0*(x-x0)*(y-y0) + C0*(y-y0)^2 = 1
    """
    x, y = x - p[0], y - p[1]
    return (p[2] * x ** 2 + p[3] * x * y + p[4] * y ** 2 - 1) * val


def correct_braggpeak_elliptical_distortions(braggpeaks, p):
    """
    Given some elliptical distortions with ellipse parameters p and some measured PointListArray
    of Bragg peak positions braggpeaks, returns the elliptically corrected Bragg peaks.

    Accepts:
        braggpeaks              (PointListArray) the Bragg peaks
        p                       (5-tuple) the parameters (x0,y0,A,B,phi) of an ellipse describing
                                the elliptical distortions in the data

    Returns:
        braggpeaks_corrected    (PointListArray) the corrected Bragg peaks
    """
    # Get the transformation matrix
    x0, y0, A, B, phi = p
    s = min(A, B) / max(
        A, B
    )  # scale the larger semiaxis down to the size of the smaller semiaxis
    phi += (
        np.pi / 2.0 * (np.argmin([A, B]) == 0)
    )  # if A was smaller, rotated phi by pi/2 to make A->B
    sinp, cosp = np.sin(phi), np.cos(phi)
    T = np.squeeze(
        np.array(
            [
                [sinp ** 2 + s * cosp ** 2, sinp * cosp * (s - 1)],
                [sinp * cosp * (s - 1), s * sinp ** 2 + cosp ** 2],
            ]
        )
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


def fit_double_sided_gaussian(data, p0, mask=None):
    """
    Fits a double sided gaussian to the data.

    The fit function is

        f(x,y; I0,I1,sigma0,sigma1,sigma2,c_bkgrd,R,x0,y0,B,C) =
            Norm(r; I0,sigma0,0) +
            Norm(r; I1,sigma1,R)*Theta(r-R)
            Norm(r; I1,sigma2,R)*Theta(R-r) + offset

    where (x,y) are coordinates and
    (I0,I1,sigma0,sigma1,sigma2,c_bkgd,R,x0,y0,B,C) are parameters.

    The function contains a pair of gaussian-shaped peaks along the radial direction of
    a polar-elliptical parametrization of a 2D plane.
    The first gaussian is centered at the origin.
    The second gaussian is centered about some finite R, and is 'two-faced': it's comprised of
    two half-gaussians of different widths, stitched together at R.
    The Janus-gaussian thus comprises an elliptical ring with different inner and
    outer widths.

    These should be empirically useful fit functions for amorphous diffraction data.
    A recommended proceedure, for starts:
    mask the central disk, and everything beyond some max q.  See if you can grab just the
    smooth plasmonic background as well as a single |q| ring.

    The parameters in p are

        I0          the intensity of the first gaussian function
        I1          the intensity of the Janus gaussian
        sigma0      std of first gaussian
        sigma1      inner std of Janus gaussian
        sigma2      outer std of Janus gaussian
        c_bkgd      a constant offset
        R           center of the Janus gaussian
        x0,y0       the origin
        B,C         1x^2 + Bxy + Cy^2 = 1

    Accepts:
        data        (2d array)
        p0          (12-tuple of floats) initial guess at parameters
        mask        ignore datapoints data[mask==False]

    Returns:
        p           (12-tuple of floats) the best fit parameters
    """
    if mask is None:
        mask = np.ones_like(data).astype(bool)
    assert data.shape == mask.shape, "data and mask must have same shapes."
    assert len(p0) == 11, "Initial guess needs 11 parameters."

    # Make coordinates, get data values
    x_inds, y_inds = np.nonzero(mask)
    vals = data[mask]

    # make bounds - speed things up
    # upper_bounds = [np.inf, np.inf, 1000, 1000, 1000, np.inf, np.max(data.shape), np.max(data.shape), np.max(data.shape), np.inf, np.inf, np.inf]
    # lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Fit
    p = leastsq(double_sided_gaussian_fiterr, p0, args=(x_inds, y_inds, vals))[0]
    return p


def compare_double_sided_gaussian(data, p, power=1, mask=None):
    """
    Plots a comparison between a diffraction pattern and a fit, given p. 
    """
    if mask is None:
        mask = np.ones_like(data)

    yy, xx = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    data_fit = double_sided_gaussian(p, xx, yy)

    theta = np.arctan2(xx - p[7], yy - p[8])
    theta_mask = np.cos(theta * 8) > 0
    data_combined = (data * theta_mask + data_fit * (1 - theta_mask)) ** power
    data_combined = mask * data_combined
    plt.figure(12, clear=True)
    plt.imshow(data_combined)

    return


def double_sided_gaussian_fiterr(p, x, y, val):
    """
    Returns the fit error associated with a point (x,y) with value val, given parameters p.
    """
    return double_sided_gaussian(p, x, y) - val


# @np.errstate(invalid='ignore') # activate if supressing warnings
def double_sided_gaussian(p, x, y):
    """
    Return the value of the double-sided gaussian function at point (x,y) given parameters p.
    """

    # Unpack parameters
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, R, x0, y0, B, C = p
    r2 = 1 * (x - x0) ** 2 + B * (x - x0) * (y - y0) + C * (y - y0) ** 2
    r = np.sqrt(r2) - R

    return (
        I0 * np.exp(-r2 / (2 * sigma0 ** 2))
        + I1 * np.exp(-r ** 2 / (2 * sigma1 ** 2)) * np.heaviside(-r, 0.5)
        + I1 * np.exp(-r ** 2 / (2 * sigma2 ** 2)) * np.heaviside(r, 0.5)
        + c_bkgd
    )


def radial_integral(ar, x0, y0, dr):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of dr.

    Accepts:
        ar              (2d array) the data
        x0,y0           (floats) the origin

    Returns:
        rbin_centers    (1d array) the bins centers of the radial integral
        radial_integral (1d array) the radial integral
    """
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
    polarAr, rr, tt = cartesianDataAr_to_polarEllipticalDataAr(
        ar, params=(x0, y0, 1, 1, 0), dr=dr, dtheta=np.radians(2), r_range=rmax
    )
    radial_integral = np.sum(polarAr, axis=0)
    rbin_centers = rr[0, :]
    return rbin_centers,radial_integral


def radial_elliptical_integral(ar, dr, ellipse_params):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of dr.

    Accepts:
        ar              (2d array) the data
        dr              (number) the r sampling
        ellipse_params  (5-tuple) the parameters (x0,y0,A,B,phi) for the ellipse

    Returns:
        rbin_centers    (1d array) the bins centers of the radial integral
        radial_integral (1d array) the radial integral
    """
    x0, y0 = ellipse_params[0], ellipse_params[1]
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
    polarAr, rr, tt = cartesianDataAr_to_polarEllipticalDataAr(
        ar, params=ellipse_params, dr=dr, dtheta=np.radians(2), r_range=rmax
    )
    radial_integral = np.sum(polarAr, axis=0)
    rbin_centers = rr[0, :]
    return rbin_centers,radial_integral
