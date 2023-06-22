# Defines the Calibration class, which stores calibration metadata

import numpy as np
from numbers import Number
from typing import Optional

from emdfile import Metadata
from py4DSTEM.data.propagating_calibration import propagating_calibration

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

    The Calibration object is capable of automatically calling the ``calibrate`` method
    of any other py4DSTEM objects when certain calibrations are updated. The methods
    that trigger propagation of calibration information are tagged with the
    @propagating_calibration decorator. Use the ``register_target`` method
    to set up an object to recieve calls to ``calibrate``

    """
    def __init__(
        self,
        name: Optional[str] = 'calibration',
        datacube = None
        ):
        """
        Args:
            name (optional, str):
        """
        Metadata.__init__(
            self,
            name=name)

        # List to hold objects that will re-`calibrate` when
        # certain properties are changed
        self._targets = []

        # set datacube
        self._datacube = datacube

        # set initial pixel values
        self.set_Q_pixel_size(1)
        self.set_R_pixel_size(1)
        self.set_Q_pixel_units('pixels')
        self.set_R_pixel_units('pixels')



    # datacube

    @property
    def datacube(self):
        return self._datacube




    ### getter/setter methods



    # pixel size/units

    @propagating_calibration
    def set_Q_pixel_size(self,x):
        if self._has_datacube():
            self.datacube.set_dim(2,x)
            self.datacube.set_dim(3,x)
        self._params['Q_pixel_size'] = x
    def get_Q_pixel_size(self):
        return self._get_value('Q_pixel_size')

    @propagating_calibration
    def set_R_pixel_size(self,x):
        if self._has_datacube():
            self.datacube.set_dim(0,x)
            self.datacube.set_dim(1,x)
        self._params['R_pixel_size'] = x
    def get_R_pixel_size(self):
        return self._get_value('R_pixel_size')

    @propagating_calibration
    def set_Q_pixel_units(self,x):
        assert(x in ('pixels','A^-1','mrad')), f"Q pixel units must be 'A^-1', 'mrad' or 'pixels'."
        if self._has_datacube():
            self.datacube.set_dim_units(2,x)
            self.datacube.set_dim_units(3,x)
        self._params['Q_pixel_units'] = x
    def get_Q_pixel_units(self):
        return self._get_value('Q_pixel_units')

    @propagating_calibration
    def set_R_pixel_units(self,x):
        if self._has_datacube():
            self.datacube.set_dim_units(0,x)
            self.datacube.set_dim_units(1,x)
        self._params['R_pixel_units'] = x
    def get_R_pixel_units(self):
        return self._get_value('R_pixel_units')



    # datacube shape

    def get_R_Nx(self):
        self._validate_datacube()
        return self.datacube.R_Nx
    def get_R_Ny(self):
        self._validate_datacube()
        return self.datacube.R_Ny
    def get_Q_Nx(self):
        self._validate_datacube()
        return self.datacube.Q_Nx
    def get_Q_Ny(self):
        self._validate_datacube()
        return self.datacube.Q_Ny
    def get_datacube_shape(self):
        self._validate_datacube()
        """ (R_Nx,R_Ny,Q_Nx,Q_Ny)
        """
        return self.datacube.data.dshape
    def get_Qshape(self,x):
        self._validate_datacube()
        return self.data.Qshape
    def get_Rshape(self,x):
        self._validate_datacube()
        return self.data.Rshape

    # is there a datacube?
    def _validate_datacube(self):
        assert(self.datacube is not None), "Can't find shape attr because Calibration doesn't point to a DataCube"
    def _has_datacube(self):
        return(self.datacube is not None)




    # origin
    def set_qx0(self,x):
        self._params['qx0'] = x
        x = np.asarray(x)
        qx0_mean = np.mean(x)
        qx0_shift = x-qx0_mean
        self._params['qx0_mean'] = qx0_mean
        self._params['qx0_shift'] = qx0_shift
    def set_qx0_mean(self,x):
        self._params['qx0_mean'] = x
    def get_qx0(self,rx=None,ry=None):
        return self._get_value('qx0',rx,ry)
    def get_qx0_mean(self):
        return self._get_value('qx0_mean')
    def get_qx0shift(self,rx=None,ry=None):
        return self._get_value('qx0_shift',rx,ry)

    def set_qy0(self,x):
        self._params['qy0'] = x
        x = np.asarray(x)
        qy0_mean = np.mean(x)
        qy0_shift = x-qy0_mean
        self._params['qy0_mean'] = qy0_mean
        self._params['qy0_shift'] = qy0_shift
    def set_qy0_mean(self,x):
        self._params['qy0_mean'] = x
    def get_qy0(self,rx=None,ry=None):
        return self._get_value('qy0',rx,ry)
    def get_qy0_mean(self):
        return self._get_value('qy0_mean')
    def get_qy0shift(self,rx=None,ry=None):
        return self._get_value('qy0_shift',rx,ry)

    def set_qx0_meas(self,x):
        self._params['qx0_meas'] = x
    def get_qx0_meas(self,rx=None,ry=None):
        return self._get_value('qx0_meas',rx,ry)

    def set_qy0_meas(self,x):
        self._params['qy0_meas'] = x
    def get_qy0_meas(self,rx=None,ry=None):
        return self._get_value('qy0_meas',rx,ry)

    def set_origin_meas_mask(self,x):
        self._params['origin_meas_mask'] = x
    def get_origin_meas_mask(self,rx=None,ry=None):
        return self._get_value('origin_meas_mask',rx,ry)

    @propagating_calibration
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
    def get_origin_mean(self):
        qx0 = self._get_value('qx0_mean')
        qy0 = self._get_value('qy0_mean')
        return qx0,qy0
    def get_origin_shift(self,rx=None,ry=None):
        qx0 = self._get_value('qx0_shift',rx,ry)
        qy0 = self._get_value('qy0_shift',rx,ry)
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
        self.set_qx0_mean(qx0)
        self.set_qy0_mean(qy0)
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

    @propagating_calibration
    def set_ellipse(self,x):
        """
        Args:
            x (3-tuple): (a,b,theta)
        """
        a,b,theta = x
        self._params['a'] = a
        self._params['b'] = b
        self._params['theta'] = theta

    @propagating_calibration
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

    @propagating_calibration
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


    # Methods for assigning objects which will be
    # auto-calibrated when the Calibration instance is updated

    def register_target(self,new_target):
        """
        Register an object to recieve calls to it `calibrate`
        method when certain calibrations get updated
        """
        self._targets.append(new_target)

    def unregister_target(self,target):
        """
        Unlink an object from recieving calls to `calibrate` when
        certain calibration values are changed
        """
        if target in self._targets:
            self._targets.remove(target)


    # HDF5 i/o

    # write is inherited from Metadata

    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in
        read mode. Determines if it's a valid Metadata representation, and
        if so loads and returns it as a Calibration instance. Otherwise,
        raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A Calibration instance
        """
        # load the group as a Metadata instance
        metadata = Metadata.from_h5(group)

        # convert it to a Calibration instance
        cal = Calibration(name = metadata.name)
        cal._params.update(metadata._params)

        # return
        return cal




########## End of class ##########


