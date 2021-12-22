import numpy as np
from numbers import Number
import h5py
from .dataobject import DataObject

class Calibrations(DataObject):
    """
    Stores and furinishes calibration measurements.

    Usage:

        >>> c = Calibrations(R_Nx,R_Ny,Q_Nx,Q_Ny)
        >>> c.set_p(p)
        >>> p = c.get_p()

    If the parameter has not been set, the getter methods return None
    The value of a parameter may be a number, representing the entire dataset,
    or an (R_Nx,R_Ny) shaped array, representing values at each detector pixel.

        >>> pxy = c.get_p(rx,ry)

    will return the value of `p` at pixel `rx,ry`.
    """
    def __init__(self,R_Nx,R_Ny,Q_Nx,Q_Ny):
        """
        Args:
            R_Nx,R_Ny (int): the shape of real space
            Q_Nx,Q_Ny (int): the shape of diffraction space
        """
        DataObject.__init__(self)
        self.R_Nx,self.R_Ny,self.Q_Nx,self.Q_Ny = R_Nx,R_Ny,Q_Nx,Q_Ny

        # Set attributes
        self.params = {
            'Q_Nx':Q_Nx,
            'Q_Ny':Q_Ny,
            'R_Nx':R_Nx,
            'R_Ny':R_Ny,
            'Q_pixel_size':1,
            'Q_pixel_units':'pixels',
            'R_pixel_size':1,
            'R_pixel_units':'pixels'
        }


    def _get_value(self,p,rx=None,ry=None):
        """ Enables returning the value of a pixel (rx,ry),
            if these are passed and `p` is an appropriate array
        """
        try:
            v = self.params[p]
            if isinstance(v,np.ndarray) and rx is not None and ry is not None:
                assert np.all([isinstance(i,(int,np.integer)) for i in (rx,ry)])
                assert rx<self.R_Nx and ry<self.R_Ny
                return p[rx,ry]
            return v
        except KeyError:
            return None


    # origin
    def set_origin(self,origin):
        """
        Args:
            origin (2-tuple): (qx0,qy0)
        """
        qx0,qy0 = origin
        self.params['qx0'] = qx0
        self.params['qy0'] = qy0
    def get_origin(self,rx=None,ry=None):
        qx0 = self._get_value('qx0',rx,ry)
        qy0 = self._get_value('qy0',rx,ry)
        return (qx0,qy0)
    def set_origin_meas(self,x):
        """
        Args:
            x (2-tuple): (qx0,qy0)
        """
        qx0,qy0 = x
        self.params['qx0_meas'] = qx0
        self.params['qy0_meas'] = qy0
    def get_origin_meas(self,rx=None,ry=None):
        qx0 = self._get_value('qx0_meas',rx,ry)
        qy0 = self._get_value('qy0_meas',rx,ry)
        return (qx0,qy0)
    def set_origin_residuals(self,x):
        """
        Args:
            x (2-tuple): (qx0,qy0)
        """
        qx0,qy0 = x
        self.params['qx0_residuals'] = qx0
        self.params['qy0_residuals'] = qy0
    def get_origin_residuals(self,rx=None,ry=None):
        qx0 = self._get_value('qx0_residuals',rx,ry)
        qy0 = self._get_value('qy0_residuals',rx,ry)
        return (qx0,qy0)

    # ellipse
    def set_a(self,x):
        self.params['a'] = x
    def get_a(self,rx=None,ry=None):
        return self._get_value('a',rx,ry)
    def set_b(self,x):
        self.params['b'] = x
    def get_b(self,rx=None,ry=None):
        return self._get_value('b',rx,ry)
    def set_theta(self,x):
        self.params['theta'] = x
    def get_theta(self,rx=None,ry=None):
        return self._get_value('theta',rx,ry)
    def set_ellipse(self,x):
        """
        Args:
            x (3-tuple): (a,b,theta)
        """
        a,b,theta = x
        self.params['a'] = a
        self.params['b'] = b
        self.params['theta'] = theta
    def set_p_ellipse(self,x):
        """
        Args:
            x (5-tuple): (qx0,qy0,a,b,theta) NOTE: does *not* change qx0,qy0
        """
        _,_,a,b,theta = x
        self.params['a'] = a
        self.params['b'] = b
        self.params['theta'] = theta
    def get_ellipse(self,rx=None,ry=None):
        a = self._get_value('a',rx,ry)
        b = self._get_value('b',rx,ry)
        theta = self._get_value('theta',rx,ry)
        return (a,b,theta)
    def get_p_ellipse(self,rx=None,ry=None):
        qx0 = self._get_value('qx0',rx,ry)
        qy0 = self._get_value('qy0',rx,ry)
        a = self._get_value('a',rx,ry)
        b = self._get_value('b',rx,ry)
        theta = self._get_value('theta',rx,ry)
        return (qx0,qy0,a,b,theta)

    # pixel sizes
    def set_Q_pixel_size(self,x):
        self.params['Q_pixel_size'] = x
    def get_Q_pixel_size(self):
        return self._get_value('Q_pixel_size')
    def set_R_pixel_size(self,x):
        self.params['R_pixel_size'] = x
    def get_R_pixel_size(self):
        return self._get_value('R_pixel_size')
    def set_Q_pixel_units(self,x):
        self.params['Q_pixel_units'] = x
    def get_Q_pixel_units(self):
        return self._get_value('Q_pixel_units')
    def set_R_pixel_units(self,x):
        self.params['R_pixel_units'] = x
    def get_R_pixel_units(self):
        return self._get_value('R_pixel_units')

    # Q/R-space rotation and flip
    def set_QR_rotation_degrees(self,x):
        self.params['QR_rotation_degrees'] = x
    def get_QR_rotation_degrees(self):
        return self._get_value('QR_rotation_degrees')
    def set_QR_flip(self,x):
        self.params['QR_flip'] = x
    def get_QR_flip(self):
        return self._get_value('QR_flip')

    # probe
    def set_convergence_semiangle_pixels(self,x):
        self.params['convergence_semiangle_pixels'] = x
    def get_convergence_semiangle_pixels(self):
        return self._get_value('convergence_semiangle_pixels')
    def set_probe_center(self,x):
        self.params['probe_center'] = x
    def get_probe_center(self):
        return self._get_value('probe_center')



    # show
    def show(self):
        if 'name' in vars(self).keys():
            print('{0:<16}\t{1:<16}'.format('name',self.name))
        for k,v in vars(self).items():
            if k != 'name':
                if isinstance(v,np.ndarray):
                    v = 'array'
                print('{0:<16}\t{1:<16}'.format(k,v))


    # calibration methods
    def calculate_Q_pixel_size(self,q_meas,q_known,units='A'):
        """
        Computes the size of the Q-space pixels. Returns and also stores
        the answer.

        Args:
            q_meas (number): a measured distance in q-space in pixels
            q_known (number): the corresponding known *real space* distance
            unit (str): the units of the real space value of `q_known`
        """
        dq = 1. / ( q_meas * q_known )
        self.set_Q_pixel_size(dq)
        self.set_Q_pixel_units(units+'^-1')
        return dq





### Read/Write

def save_calibrations_group(group, calibrations):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    keys,datakeys = dir(calibrations),[]
    for key in keys:
        if (key[0]!='_') and (key[:4] not in ('get_','set_','sort','show','name')):
            datakeys.append(key)

    for key in datakeys:
        data = vars(calibrations)[key]
        group.create_dataset(key, data=data)

def get_calibrations_from_grp(g):
    """ Accepts an h5py Group corresponding to a Calibrations instance in an open,
        correctly formatted H5 file, and returns the instance.
    """
    name = g.name.split('/')[-1]
    data = dict()
    for key in g.keys():
        data[key] = np.array(g[key])
        if len(data[key].shape) == 0:
            data[key] = data[key].item()
    for key in ('R_Nx','R_Ny','Q_Nx','Q_Ny'):
        if key not in data.keys():
            data[key] = 1
    calibrations = Calibrations(data['R_Nx'],data['R_Ny'],data['Q_Nx'],data['Q_Ny'])
    calibrations.R_pixel_units = str(data['R_pixel_units'])
    calibrations.Q_pixel_units = str(data['Q_pixel_units'])
    for key in ('R_Nx','R_Ny','Q_Nx','Q_Ny','R_pixel_units','Q_pixel_units'):
        del data[key]
    for key in data.keys():
        getattr(calibrations,'set_'+key)(data[key])

    calibrations.name = name
    return calibrations






