# Defines the Calibration class, which stores calibration metadata

import numpy as np
from numbers import Number
from typing import Optional
import h5py

from ..emd.metadata import Metadata


class Calibration(Metadata):
    """
    Stores calibration measurements.

    Usage:

        >>> c = Calibration()
        >>> c.set_p(p)
        >>> p = c.get_p()

    If the parameter has not been set, the getter methods return None. For
    parameters with multiple values, they're returned as a tuple. If any of
    the multiple values can't be found, a single None is returned instead.
    Some parameters may have distinct values for each scan position; these
    are stored as 2D arrays, and

        >>> c.get_p()

    will return the entire 2D array, while

        >>> c.get_p(rx,ry)

    will return the value of `p` at position `rx,ry`.
    """
    def __init__(
        self,
        name: Optional[str] ='calibration'
        ):
        """
        Args:
            name (optional, str):
        """
        Metadata.__init__(
            self,
            name=name)

        # set initial pixel values
        self.set_Q_pixel_size(1)
        self.set_R_pixel_size(1)
        self.set_Q_pixel_units('pixels')
        self.set_R_pixel_units('pixels')


    ### getter/setter methods

    # datacube shape
    def set_R_Nx(self,x):
        self._params['R_Nx'] = x
    def get_R_Nx(self):
        return self._get_value('R_Nx')
    def set_R_Ny(self,x):
        self._params['R_Ny'] = x
    def get_R_Ny(self):
        return self._get_value('R_Ny')
    def set_Q_Nx(self,x):
        self._params['Q_Nx'] = x
    def get_Q_Nx(self):
        return self._get_value('Q_Nx')
    def set_Q_Ny(self,x):
        self._params['Q_Ny'] = x
    def get_Q_Ny(self):
        return self._get_value('Q_Ny')
    def set_datacube_shape(self,x):
        """
        Args:
            x (4-tuple): (R_Nx,R_Ny,Q_Nx,Q_Ny)
        """
        R_Nx,R_Ny,Q_Nx,Q_Ny = x
        self._params['R_Nx'] = R_Nx
        self._params['R_Ny'] = R_Ny
        self._params['Q_Nx'] = Q_Nx
        self._params['Q_Ny'] = Q_Ny
    def get_datacube_shape(self):
        """ (R_Nx,R_Ny,Q_Nx,Q_Ny)
        """
        R_Nx = self.get_R_Nx()
        R_Ny = self.get_R_Ny()
        Q_Nx = self.get_Q_Nx()
        Q_Ny = self.get_Q_Ny()
        shape = (R_Nx,R_Ny,Q_Nx,Q_Ny)
        if any([x is None for x in shape]):
            shape = None
        return shape
    def set_Qshape(self,x):
        """
        Args:
            x (2-tuple): (Q_Nx,Q_Ny)
        """
        Q_Nx,Q_Ny = x
        self._params['Q_Nx'] = Q_Nx
        self._params['Q_Ny'] = Q_Ny
    def get_Qshape(self,x):
        Q_Nx = self._params['Q_Nx']
        Q_Ny = self._params['Q_Ny']
        shape = (Q_Nx,Q_Ny)
        if any([x is None for x in shape]):
            shape = None
        return shape
    def set_Rshape(self,x):
        """
        Args:
            x (2-tuple): (R_Nx,R_Ny)
        """
        R_Nx,R_Ny = x
        self._params['R_Nx'] = R_Nx
        self._params['R_Ny'] = R_Ny
    def get_Rshape(self,x):
        R_Nx = self._params['R_Nx']
        R_Ny = self._params['R_Ny']
        shape = (R_Nx,R_Ny)
        if any([x is None for x in shape]):
            shape = None
        return shape

    # pixel sizes
    def set_Q_pixel_size(self,x):
        self._params['Q_pixel_size'] = x
    def get_Q_pixel_size(self):
        return self._get_value('Q_pixel_size')
    def set_R_pixel_size(self,x):
        self._params['R_pixel_size'] = x
    def get_R_pixel_size(self):
        return self._get_value('R_pixel_size')
    def set_Q_pixel_units(self,x):
        pix = ('pixels','A^-1')
        assert(x in pix), f"{x} must be in {pix}"
        self._params['Q_pixel_units'] = x
    def get_Q_pixel_units(self):
        return self._get_value('Q_pixel_units')
    def set_R_pixel_units(self,x):
        self._params['R_pixel_units'] = x
    def get_R_pixel_units(self):
        return self._get_value('R_pixel_units')

    # origin
    def set_qx0(self,x):
        self._params['qx0'] = x
    def get_qx0(self,rx=None,ry=None):
        return self._get_value('qx0',rx,ry)
    def set_qy0(self,x):
        self._params['qy0'] = x
    def get_qy0(self,rx=None,ry=None):
        return self._get_value('qy0',rx,ry)
    def set_qx0_meas(self,x):
        self._params['qx0_meas'] = x
    def get_qx0_meas(self,rx=None,ry=None):
        return self._get_value('qx0_meas',rx,ry)
    def set_qy0_meas(self,x):
        self._params['qy0_meas'] = x
    def get_qy0_meas(self,rx=None,ry=None):
        return self._get_value('qy0_meas',rx,ry)
    def set_origin_meas_mask(self,x):
        self._['origin_meas_mask'] = x
    def get_origin_meas_mask(self,rx=None,ry=None):
        return self._get_value('origin_meas_mask',rx,ry)
    def set_origin(self,x):
        """
        Args:
            x (2-tuple of numbers or of 2D, R-shaped arrays): the origin
        """
        qx0,qy0 = x
        self.set_qx0(qx0)
        self.set_qy0(qy0)
    def get_origin(self,rx=None,ry=None):
        qx0 = self._get_value('qx0',rx,ry)
        qy0 = self._get_value('qy0',rx,ry)
        ans = (qx0,qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans
    def set_origin_meas(self,x):
        """
        Args:
            x (2-tuple or 3 uple of 2D R-shaped arrays): qx0,qy0,[mask]
        """
        qx0,qy0 = x[0],x[1]
        self.set_qx0_meas(qx0)
        self.set_qy0_meas(qy0)
        try:
            m = x[2]
            self.set_origin_meas_mask(m)
        except IndexError:
            pass
    def get_origin_meas(self,rx=None,ry=None):
        qx0 = self._get_value('qx0_meas',rx,ry)
        qy0 = self._get_value('qy0_meas',rx,ry)
        ans = (qx0,qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans
    def set_probe_semiangle(self,x):
        self._params['probe_semiangle'] = x
    def get_probe_semiangle(self):
        return self._get_value('probe_semiangle')
    def set_probe_param(self, x):
        """
        Args:
            x (3-tuple): (probe size, x0, y0)
        """
        probe_semiangle, qx0, qy0 = x
        self.set_probe_semiangle(probe_semiangle)
        self.set_qx0(qx0)
        self.set_qy0(qy0)
    def get_probe_param(self):
        probe_semiangle = self._get_value('probe_semiangle')
        qx0 = self._get_value('qx0')
        qy0 = self._get_value('qy0')
        ans = (probe_semiangle,qx0,qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans
        

    # ellipse
    def set_a(self,x):
        self._params['a'] = x
    def get_a(self,rx=None,ry=None):
        return self._get_value('a',rx,ry)
    def set_b(self,x):
        self._params['b'] = x
    def get_b(self,rx=None,ry=None):
        return self._get_value('b',rx,ry)
    def set_theta(self,x):
        self._params['theta'] = x
    def get_theta(self,rx=None,ry=None):
        return self._get_value('theta',rx,ry)
    def set_ellipse(self,x):
        """
        Args:
            x (3-tuple): (a,b,theta)
        """
        a,b,theta = x
        self._params['a'] = a
        self._params['b'] = b
        self._params['theta'] = theta
    def set_p_ellipse(self,x):
        """
        Args:
            x (5-tuple): (qx0,qy0,a,b,theta) NOTE: does *not* change qx0,qy0!
        """
        _,_,a,b,theta = x
        self._params['a'] = a
        self._params['b'] = b
        self._params['theta'] = theta
    def get_ellipse(self,rx=None,ry=None):
        a = self.get_a(rx,ry)
        b = self.get_b(rx,ry)
        theta = self.get_theta(rx,ry)
        ans = (a,b,theta)
        if any([x is None for x in ans]):
            ans = None
        return ans
    def get_p_ellipse(self,rx=None,ry=None):
        qx0,qy0 = self.get_origin(rx,ry)
        a,b,theta = self.get_ellipse(rx,ry)
        return (qx0,qy0,a,b,theta)

    # Q/R-space rotation and flip
    def set_QR_rotation_degrees(self,x):
        self._params['QR_rotation_degrees'] = x
    def get_QR_rotation_degrees(self):
        return self._get_value('QR_rotation_degrees')
    def set_QR_flip(self,x):
        self._params['QR_flip'] = x
    def get_QR_flip(self):
        return self._get_value('QR_flip')
    def set_QR_rotflip(self, rot_flip):
        """
        Args:
            rot_flip (tuple), (rot, flip) where:
                rot (number): rotation in degrees
                flip (bool): True indicates a Q/R axes flip
        """
        rot,flip = rot_flip
        self.set_QR_rotation_degrees(rot)
        self.set_QR_flip(flip)
    def get_QR_rotflip(self):
        rot = self.get_QR_rotation_degrees()
        flip = self.get_QR_flip()
        if rot is None or flip is None:
            return None
        return (rot,flip)


    # probe
    def set_convergence_semiangle_pixels(self,x):
        self._params['convergence_semiangle_pixels'] = x
    def get_convergence_semiangle_pixels(self):
        return self._get_value('convergence_semiangle_pixels')
    def set_convergence_semiangle_pixels(self,x):
        self._params['convergence_semiangle_mrad'] = x
    def get_convergence_semiangle_pixels(self):
        return self._get_value('convergence_semiangle_mrad')
    def set_probe_center(self,x):
        self._params['probe_center'] = x
    def get_probe_center(self):
        return self._get_value('probe_center')


    # For parameters which can have 2D or (2+n)D array values,
    # this function enables returning the value(s) at a 2D position,
    # rather than the whole array
    def _get_value(self,p,rx=None,ry=None):
        """ Enables returning the value of a pixel (rx,ry),
            if these are passed and `p` is an appropriate array
        """
        v = self._params.get(p)

        if v is None:
            return v

        if (rx is None) or (ry is None) or (not isinstance(v,np.ndarray)):
            return v

        else:
            er = f"`rx` and `ry` must be ints; got values {rx} and {ry}"
            assert np.all([isinstance(i,(int,np.integer)) for i in (rx,ry)]), er
            return v[rx,ry]



    def copy(self,name=None):
        """
        """
        if name is None: name = self.name+"_copy"
        cal = Calibration(name=name)
        cal._params.update(self._params)
        return cal



    # HDF5 read/write

    # write inherited from Metadata

    # read
    def from_h5(group):
        from .io import Calibration_from_h5
        return Calibration_from_h5(group)




########## End of class ##########





