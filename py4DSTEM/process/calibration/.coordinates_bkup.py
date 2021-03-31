import numpy as np
from numbers import Number

class Coordinates(object):
    """
    Defines coordinate systems for diffraction space in a 4D-STEM dataset.
    This includes cartesian and polar-elliptical coordinate systems.

    Each parameter may be a single number, for the case where the parameter is identical
    across all diffraction patterns, or it may be an (R_Nx,R_Ny)-shaped array, for the case
    where the parameter varies with scan position, e.g. the shifting of the optic axis as
    the beam is scanned.

	Storing and accessing some parameter p is accomplished with get/set methods, e.g.:
		```
		coords = Coordinates(R_Nx,R_Ny,Q_Nx,Q_Ny)
		coords.set_p(p)
		p = coords.get_p()
		```
	The get methods support retrieving numbers, arrays, or values at specified (rx,ry)
	positions.  The code
		```
        p = coords.get_p(rx,ry)
		```
    will retrieve the value p if p is a number, and the value p[rx,ry] if p is an array.
    """
    def __init__(self,R_Nx,R_Ny,Q_Nx,Q_Ny,
                 Q_pixel_size=1,Q_pixel_units='pixels',R_pixel_size=1,R_pixel_units='pixels',
                 qx0=None,qy0=None,e=None,theta=None):
        """
        Initialize a coordinate system.

        Accepts:
            Q_Nx,Q_Ny         (ints) the shape of diffraction space
            R_Nx,R_Ny         (ints) the shape of real space
            Q_pixel_size      (number)
            Q_pixel_units     (string)
            R_pixel_size      (number)
            R_pixel_units     (string)
            qx0,qy0           (numbers or ndarrays) the origin of diffraction space
            e                 (number) the ratio of lengths of the semiminor to
                              semimajor axes of the elliptical distortions
            theta             (number) the (positive, right handed) tilt of the
                              semimajor axis of the elliptical distortions with
                              respect to the x-axis, in radians

        """
        self.Q_Nx,self.Q_Ny = Q_Nx,Q_Ny
        self.R_Nx,self.R_Ny = R_Nx,R_Ny
        self.set_Q_pixel_size(Q_pixel_size)
        self.set_Q_pixel_units(Q_pixel_units)
        self.set_R_pixel_size(R_pixel_size)
        self.set_R_pixel_units(R_pixel_units)
        if qx0 is not None: self.set_qx0(qx0)
        if qy0 is not None: self.set_qy0(qy0)
        if e is not None: self.set_e(e)
        if theta is not None: self.set_theta(theta)


    def set_Q_pixel_size(self,Q_pixel_size):
        self.Q_pixel_size = Q_pixel_size
    def set_Q_pixel_units(self,Q_pixel_units):
        self.Q_pixel_units = Q_pixel_units
    def set_R_pixel_size(self,R_pixel_size):
        self.R_pixel_size = R_pixel_size
    def set_R_pixel_units(self,R_pixel_units):
        self.R_pixel_units = R_pixel_units
    def set_qx0(self,qx0):
        self._validate_input(qx0)
        self.qx0 = qx0
    def set_qy0(self,qy0):
        self._validate_input(qy0)
        self.qy0 = qy0
    def set_origin(self,qx0,qy0):
        self._validate_input(qx0)
        self._validate_input(qy0)
        self.qx0,self.qy0 = qx0,qy0
    def set_e(self,e):
        self._validate_input(e)
        self.e = e
    def set_theta(self,theta):
        self._validate_input(theta)
        self.theta = theta
    def set_ellipse(self,e,theta):
        self._validate_input(e)
        self._validate_input(theta)
        self.e,self.theta = e,theta

    def get_Q_pixel_size(self):
        return self._get_value(self.Q_pixel_size)
    def get_Q_pixel_units(self):
        return self._get_value(self.Q_pixel_units)
    def get_R_pixel_size(self):
        return self._get_value(self.R_pixel_size)
    def get_R_pixel_units(self):
        return self._get_value(self.R_pixel_units)
    def get_qx0(self,rx=None,ry=None):
        return self._get_value(self.qx0,rx,ry)
    def get_qy0(self,rx=None,ry=None):
        return self._get_value(self.qy0,rx,ry)
    def get_center(self,rx=None,ry=None):
        return self.get_qx0(rx,ry),self.get_qy0(rx,ry)
    def get_e(self,rx=None,ry=None):
        return self._get_value(self.e,rx,ry)
    def get_theta(self,rx=None,ry=None):
        return self._get_value(self.theta,rx,ry)
    def get_ellipse(self,rx=None,ry=None):
        return self.get_e(rx,ry),self.get_theta(rx,ry)

    def _validate_input(self,p):
        assert isinstance(p,Number) or isinstance(p,np.ndarray)
        if isinstance(p,np.ndarray):
            assert p.shape == (self.R_Nx,self.R_Ny)
    def _get_value(self,p,rx=None,ry=None):
        try:
            assert isinstance(p,Number) or isinstance(p,np.ndarray)
            if isinstance(p,Number):
                return p
            if rx is None and ry is None:
                return p
            else:
                assert np.all([isinstance(i,(int,np.integer)) for i in (rx,ry)])
                assert rx<self.R_Nx and ry<self.R_Ny
                return p[rx,ry]
        except (NameError,AttributeError):
            return None

