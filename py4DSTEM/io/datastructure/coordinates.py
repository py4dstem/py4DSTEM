import numpy as np
from numbers import Number
import h5py
from .dataobject import DataObject

class Coordinates(DataObject):
    """
    Stores and furinishes calibration measurements.

    Usage:

        >>> c = Coordinates(R_Nx,R_Ny,Q_Nx,Q_Ny)
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

    def set_qx0(self,x):
        self.params['qx0'] = x
    def get_qx0(self,rx=None,ry=None):
        return self._get_value('qx0',rx,ry)
    def set_qy0(self,x):
        self.params['qy0'] = x
    def get_qy0(self,rx=None,ry=None):
        return self._get_value('qy0',rx,ry)

    def set_qx0_meas(self,x):
        self.params['qx0_meas'] = x
    def get_qx0_meas(self,rx=None,ry=None):
        return self._get_value('qx0_meas',rx,ry)
    def set_qy0_meas(self,x):
        self.params['qy0_meas'] = x
    def get_qy0_meas(self,rx=None,ry=None):
        return self._get_value('qy0_meas',rx,ry)

    def set_qx0_residuals(self,x):
        self.params['qx0_residuals'] = x
    def get_qx0_residuals(self,rx=None,ry=None):
        return self._get_value('qx0_residuals',rx,ry)
    def set_qy0_residuals(self,x):
        self.params['qy0_residuals'] = x
    def get_qy0_residuals(self,rx=None,ry=None):
        return self._get_value('qy0_residuals',rx,ry)

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









    # Special get/set functions
    def set_R_Nx(self,R_Nx):
        self.R_Nx = R_Nx
    def set_R_Ny(self,R_Ny):
        self.R_Ny = R_Ny
    def set_Q_Nx(self,Q_Nx):
        self.Q_Nx = Q_Nx
    def set_Q_Ny(self,Q_Ny):
        self.Q_Ny = Q_Ny
    def set_Q_pixel_size(self,Q_pixel_size):
        self._validate_input(Q_pixel_size)
        self.Q_pixel_size = Q_pixel_size
    def set_Q_pixel_units(self,Q_pixel_units):
        self.Q_pixel_units = Q_pixel_units
    def set_R_pixel_size(self,R_pixel_size):
        self._validate_input(R_pixel_size)
        self.R_pixel_size = R_pixel_size
    def set_R_pixel_units(self,R_pixel_units):
        self.R_pixel_units = R_pixel_units
    def set_alpha_pix(self,alpha_pix):
        self.alpha_pix = alpha_pix
    def set_probe_center(self,probe_center):
        self.probe_center = probe_center
    def set_QR_rotation(self,QR_rotation):
        self._validate_input(QR_rotation)
        self.QR_rotation = QR_rotation
    def set_QR_flip(self,QR_flip):
        assert(isinstance(QR_flip,(bool,np.bool_)))
        self.QR_flip = QR_flip

    def get_R_Nx(self):
        return self.R_Nx
    def get_R_Ny(self):
        return self.R_Ny
    def get_Q_Nx(self):
        return self.Q_Nx
    def get_Q_Ny(self):
        return self.Q_Ny
    def get_Q_pixel_size(self):
        return self._get_value(self.Q_pixel_size)
    def get_Q_pixel_units(self):
        return self.Q_pixel_units
    def get_R_pixel_size(self):
        return self._get_value(self.R_pixel_size)
    def get_R_pixel_units(self):
        return self.R_pixel_units
    def get_alpha_pix(self):
        return self.alpha_pix
    def get_probe_center(self):
        return self.probe_center
        return (qx0,qy0,a,b,theta)
    def get_QR_rotation(self):
        return self.QR_rotation
    def get_QR_flip(self):
        return self.QR_flip




    def show(self):
        if 'name' in vars(self).keys():
            print('{0:<16}\t{1:<16}'.format('name',self.name))
        for k,v in vars(self).items():
            if k != 'name':
                if isinstance(v,np.ndarray):
                    v = 'array'
                print('{0:<16}\t{1:<16}'.format(k,v))




### Read/Write

def save_coordinates_group(group, coordinates):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    keys,datakeys = dir(coordinates),[]
    for key in keys:
        if (key[0]!='_') and (key[:4] not in ('get_','set_','sort','show','name')):
            datakeys.append(key)

    for key in datakeys:
        data = vars(coordinates)[key]
        group.create_dataset(key, data=data)

def get_coordinates_from_grp(g):
    """ Accepts an h5py Group corresponding to a Coordinates instance in an open,
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
    coordinates = Coordinates(data['R_Nx'],data['R_Ny'],data['Q_Nx'],data['Q_Ny'])
    coordinates.R_pixel_units = str(data['R_pixel_units'])
    coordinates.Q_pixel_units = str(data['Q_pixel_units'])
    for key in ('R_Nx','R_Ny','Q_Nx','Q_Ny','R_pixel_units','Q_pixel_units'):
        del data[key]
    for key in data.keys():
        getattr(coordinates,'set_'+key)(data[key])

    coordinates.name = name
    return coordinates






