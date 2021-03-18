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
                 qx0=None,qy0=None,a=None,b=None,theta=None,
                 Q_pixel_size=1,Q_pixel_units='pixels',R_pixel_size=1,R_pixel_units='pixels'):
        """
        Initialize a coordinate system.

        Accepts:
            Q_Nx,Q_Ny         (ints) the shape of diffraction space
            R_Nx,R_Ny         (ints) the shape of real space
            qx0,qy0           (numbers or ndarrays) the origin of diffraction space
            a,b               (numbers or ndarrays) the stretch along the major/minor
                              axes of the polar-elliptical coordinate system
            theta             (number or ndarray) the tilt angle, in radians, of the
                              polar-elliptical coordinate system
            Q_pixel_size      (number)
            Q_pixel_units     (string)
            R_pixel_size      (number)
            R_pixel_units     (string)

        """
        self.Q_Nx,self.Q_Ny = Q_Nx,Q_Ny
        self.R_Nx,self.R_Ny = R_Nx,R_Ny
        self.Q_pixel_size,self.Q_pixel_units = Q_pixel_size,Q_pixel_units
        self.R_pixel_size,self.R_pixel_units = R_pixel_size,R_pixel_units
        self.qx0,self.qy0 = qx0,qy0
        self.a,self.b,self.theta = a,b,theta

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
    def set_a(self,a):
        self._validate_input(a)
        self.a = a
    def set_b(self,b):
        self._validate_input(b)
        self.b = b
    def set_theta(self,theta):
        self._validate_input(theta)
        self.theta = theta
    def set_ellipse(self,a,b,theta):
        self._validate_input(a)
        self._validate_input(b)
        self._validate_input(theta)
        self.a,self.b,self.theta = a,b,theta
    def set_Q_pixel_size(self,Q_pixel_size):
        self.Q_pixel_size = Q_pixel_size
    def set_Q_pixel_units(self,Q_pixel_units):
        self.Q_pixel_units = Q_pixel_units
    def set_R_pixel_size(self,R_pixel_size):
        self.R_pixel_size = R_pixel_size
    def set_R_pixel_units(self,R_pixel_units):
        self.R_pixel_units = R_pixel_units

    def get_qx0(self,rx=None,ry=None):
        return self._get_value(self.qx0,rx,ry)
    def get_qy0(self,rx=None,ry=None):
        return self._get_value(self.qy0,rx,ry)
    def get_center(self,rx=None,ry=None):
        return self.get_qx0(rx,ry),self.get_qy0(rx,ry)
    def get_a(self,rx=None,ry=None):
        return self._get_value(self.a,rx,ry)
    def get_b(self,rx=None,ry=None):
        return self._get_value(self.b,rx,ry)
    def get_theta(self,rx=None,ry=None):
        return self._get_value(self.theta,rx,ry)
    def get_ellipse(self,rx=None,ry=None):
        return self.get_a(rx,ry),self.get_b(rx,ry),self.get_theta(rx,ry)
    def get_Q_pixel_size(self):
        return self._get_value(self.Q_pixel_size)
    def get_Q_pixel_units(self):
        return self._get_value(self.Q_pixel_units)
    def get_R_pixel_size(self):
        return self._get_value(self.R_pixel_size)
    def get_R_pixel_units(self):
        return self._get_value(self.R_pixel_units)

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
        except NameError:
            return None

