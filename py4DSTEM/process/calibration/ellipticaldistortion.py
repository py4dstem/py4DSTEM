# Methods for correcting elliptical distortion

import numpy as np
from scipy.optimize import leastsq

def measure_elliptical_distortion(ar, x0, y0, r_inner, r_outer, datamask=None):
    """
    Fits an ellipse to the data in an annular region of array ar, centered at (x0,y0) and with inner
    and outer radii of r_inner, r_outer, respectively.

    Returns two sets of parameters, corresponding to two different representations of the same
    ellipse.  In the first, we write the ellipse as
        $ x^2/a^2 + y^2/b^2 = 1 $
    and in the second we write
        $ Ax'^2 + Bx'y' + Cy'^2 = 1 $
    The primed coordinates are orthogonal to the detector.  The unprimed coordinated are rotated
    counterclockwise by an angle theta with respect to the primed coordinates; thus in the unprimed
    system the semiaxes of the ellipse are oriented at angle theta.  Both coordinate systems take
    the center of the ellipse as the origin.

    Accepts:
        ar          (ndarry) array containing the data to fit
        x0,y0       (floats) the center of the annular fitting region
        r_inner     (float) inner radius of the fit region
        r_outer     (float) outer radius of the fit region
        datamask    (ar-shaped ndarray of bools) ignore datapoints where mask==False

    Returns:
        p1          (5-tuple) the parameters of the ellipse in the unprimed coordinates, with
                        p0 = (x,y,a,b,theta)
                    where (x,y) is the center of the ellipse, and a,b,theta are as decribed above.
        p2          (5-tuple) the parameters of the ellipse in the primed coordinates, with
                        p1 = (x,y,A,B,C)
    """
    # Get the datapoints to fit
    yy,xx = np.meshgrid(np.arange(ar.shape[1]),np.arange(ar.shape[1]))
    rr = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    mask = (rr>r_inner) * (rr<=r_outer)
    if datamask is not None:
        mask *= datamask
    xs,ys = np.nonzero(mask)
    vals = ar[mask]

    # Get initial parameters guess
    p0_0 = x0
    p0_1 = y0
    p0_2 = (2/(r_inner+r_outer))**2
    p0_3 = 0
    p0_4 = (2/(r_inner+r_outer))**2
    p0 = [p0_0,p0_1,p0_2,p0_3,p0_4]

    # Fit
    p2 = leastsq(ellipse_err, p0, args=(xs,ys,vals))[0]

    # Convert between representations
    x,y,A,B,C = p2
    a,b,theta = convert_ellipse_params(A,B,C)
    p1 = (x,y,a,b,theta)

    return p1, p2

def ellipse_err(p, x, y, val):
    """
    Returns the error associated with point (x,y), which has value val, with respect to the ellipse
    given by parameters p. p is a 5-tuple in the form of (x,y,A,B,C) specifying the ellipse
        $ Ax^2 + Bxy + Cy^2 = 1 $
    """
    x,y = x-p[0],y-p[1]
    return (p[2]*x**2 + p[3]*x*y + p[4]*y**2 - 1)*val

def convert_ellipse_params(A,B,C):
    """
    Takes A, B, and C for an ellipse given by
        $ Ax^2 + Bxy + Cy^2 = 1 $
    and return the semiaxes (a,b) and the tilt (theta) in radians.
    """
    x = (A-C)*np.sqrt(1+(B/(A-C))**2)
    a = np.sqrt(2/(A+C+x))
    b = np.sqrt(2/(A+C-x))
    theta = 0.5*np.arctan(B/(A-C))
    return a,b,theta

def correct_elliptical_distortion(braggpeaks, p):
    """
    Corrects the elliptical distortions described by the ellipse parameters p in the PointListArray
    of peak positions braggpeaks.

    Accepts:
        braggpeaks              (PointListArray) the Bragg peaks
        p                       (5-tuple) the parameters (x0,y0,a,b,theta) of an ellipse centered at
                                (x0,y0), with semiaxis of length a,b and tilted at angle theta.

    Returns:
        braggpeaks_corrected    (PointListArray) the corrected Bragg peaks
    """
    # Get the transformation matrix
    x0,y0,a,b,theta = p
    s = min(a,b)/max(a,b)
    theta += np.pi/2.*(np.argmin([a,b])==0)
    sint,cost = np.sin(theta),np.cos(theta)
    T = np.array([[sint**2 + s*cost**2, sint*cost*(s-1)],[sint*cost*(s-1), s*sint**2 + cost**2]])

    # Correct distortions
    braggpeaks_corrected = braggpeaks.copy(name=braggpeaks.name+"_ellipticalcorrected")
    for Rx in range(braggpeaks_corrected.shape[0]):
        for Ry in range(braggpeaks_corrected.shape[1]):
            pointlist = braggpeaks_corrected.get_pointlist(Rx,Ry)
            x,y = pointlist.data['qx']-x0, pointlist.data['qy']-y0
            xyar_i = np.vstack([x,y])
            xyar_f = np.matmul(T,xyar_i)
            pointlist.data['qx'] = xyar_f[0,:]+x0
            pointlist.data['qy'] = xyar_f[1,:]+y0
    return braggpeaks_corrected






