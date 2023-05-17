import numpy as np
from py4DSTEM.classes import DataCube
from py4DSTEM.process.polardata.polar_datacube import PolarDatacube



class PolarEllipticDatacube(PolarDatacube):

    """
    An interface to a 4D-STEM datacube under polar-elliptical transformation.
    """

    def __init__(
        self,
        datacube,
        qmin = 0.0,
        qmax = None,
        qstep = 1.0,
        n_annular = 180,
        ):
        """
        Parameters
        ----------
        datacube : DataCube
            The datacube in cartesian coordinates
        qmin : number
            Minumum radius of the polar transformation
        qmax : number or None
            Maximum radius of the polar transformation
        qstep : number
            Width of radial bins
        n_annular = 180
            Number of bins in the annular direction. Bins will each
            have a width of 360/num_annular_bins, in degrees
        """

        # initialize as a PolarDatacube
        PolarDatacube.__init__(
            self,
            datacube = datacube,
            qmin = qmin,
            qmax = qmax,
            qstep = qstep,
            n_annular = n_annular,
        )

        # overwrite PolarDatacube's getter
        self._set_polar_elliptic_data_getter()

    def _set_polar_elliptic_data_getter(self):
        self._polar_data_getter = PolarEllipticDataGetter(
            polarcube = self
        )



class PolarEllipticDataGetter:

    def __init__(
        self,
        polarcube,
    ):
        self._polarcube = polarcube

    def __getitem__(self,pos):
        x,y = pos
        ans = self._polarcube._datacube[x,y]
        ans = self._transform(
            cartesian_data = ans,
            cal = self._polarcube.calibration,
            scanxy = (x,y),
        )
        return ans

    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += "Retrieves the diffraction pattern at scan position (x,y) in polar elliptic coordinates when sliced with [x,y]."
        return string

    def _transform(
        self,
        cartesian_data,
        cal,
        scanxy,
        ):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.
        """
        # scan position
        x,y = scanxy

        # get calibrations
        assert(cal.get_origin(x,y) is not None), "No center found! Try setting the origin."
        assert(cal.get_ellipse(x,y) is not None), "No elliptical calibrations found!"
        origin = cal.get_origin(x,y)
        ellipse = cal.get_ellipse(x,y)
        a,b,theta = ellipse

        # shifted coordinates
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]


        # TODO

        # Get (qx,qy) corresponding to each (r,phi) in the newly defined coords
        #xr = rr * np.cos(pp)
        #yr = rr * np.sin(pp)
        #qx = qx0 + xr * np.cos(theta) - yr * (b / a) * np.sin(theta)
        #qy = qy0 + xr * np.sin(theta) + yr * (b / a) * np.cos(theta)


        # polar-elliptic coordinate indices
        r_ind = (np.sqrt(x**2 + y**2) - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = np.arctan2(y, x) / self._polarcube.annular_step
        r_ind_floor = np.floor(r_ind).astype('int')
        t_ind_floor = np.floor(t_ind).astype('int')
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor
        # t_ind_floor = np.mod(t_ind_floor, self.num_annular_bins)

        # polar transformation
        sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self._polarcube.polar_shape[1])
        im = np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (1 - dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (1 - dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self._polarcube.polar_shape[1]-1)
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (    dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (    dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )

        # output
        ans = np.reshape(im, self._polarcube.polar_shape)
        return ans






#### begin code from utils/cartesian_to_polarelliptical


# Utility functions

# convert (A,B,C) <-> (a,b,theta) representations

def convert_ellipse_params(A, B, C):
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
    val = np.sqrt((A - C) ** 2 + B**2)
    b4a = B**2 - 4 * A * C
    # Get theta
    if B == 0:
        if A < C:
            theta = 0
        else:
            theta = np.pi / 2.0
    else:
        theta = np.arctan2((C - A - val), B)
    # Get a,b
    a = -np.sqrt(-2 * b4a * (A + C + val)) / b4a
    b = -np.sqrt(-2 * b4a * (A + C - val)) / b4a
    a, b = max(a, b), min(a, b)
    return a, b, theta


def convert_ellipse_params_r(a, b, theta):
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
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    a2, b2 = a**2, b**2
    A = sin2 / b2 + cos2 / a2
    C = cos2 / b2 + sin2 / a2
    B = 2 * (b2 - a2) * np.sin(theta) * np.cos(theta) / (a2 * b2)
    return A, B, C







