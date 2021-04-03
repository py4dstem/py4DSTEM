import numpy as np
from numbers import Number
import h5py
from .dataobject import DataObject

class Coordinates(DataObject):
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
                 Q_pixel_size=1,Q_pixel_units='pixels',
                 R_pixel_size=1,R_pixel_units='pixels',
                 **kwargs):
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
        DataObject.__init__(self, **kwargs)

        # Define the data items that can be stored
        datakeys = ('R_Nx',
                    'R_Ny',
                    'Q_Nx',
                    'Q_Ny',
                    'R_pixel_size',
                    'Q_pixel_size',
                    'qx0',
                    'qy0',
                    'e',
                    'theta')

        # Construct get/set functions
        for key in datakeys:
            setattr(self,'set_'+key,self.set_constructor(key))
            setattr(self,'get_'+key,self.get_constructor(key))

        # Set attributes
        self.set_R_Nx(R_Nx)
        self.set_R_Ny(R_Ny)
        self.set_Q_Nx(Q_Nx)
        self.set_Q_Ny(Q_Ny)
        self.set_Q_pixel_size(Q_pixel_size)
        self.set_Q_pixel_units(Q_pixel_units)
        self.set_R_pixel_size(R_pixel_size)
        self.set_R_pixel_units(R_pixel_units)

        # Set attributes passed as kwargs
        for key,val in kwargs.items():
            try:
                vars(self)['set_'+key](val)
            except:
                KeyError

    # Special get/set functions
    def set_Q_pixel_units(self,Q_pixel_units):
        self.Q_pixel_units = Q_pixel_units
    def set_R_pixel_units(self,R_pixel_units):
        self.R_pixel_units = R_pixel_units
    def set_origin(self,qx0,qy0):
        self._validate_input(qx0)
        self._validate_input(qy0)
        self.qx0,self.qy0 = qx0,qy0
    def set_ellipse(self,e,theta):
        self._validate_input(e)
        self._validate_input(theta)
        self.e,self.theta = e,theta

    def get_Q_pixel_units(self):
        return self._get_value(self.Q_pixel_units)
    def get_R_pixel_units(self):
        return self._get_value(self.R_pixel_units)
    def get_center(self,rx=None,ry=None):
        return self.get_qx0(rx,ry),self.get_qy0(rx,ry)
    def get_ellipse(self,rx=None,ry=None):
        return self.get_e(rx,ry),self.get_theta(rx,ry)

    # Get/set constructors
    def set_constructor(self,key):
        def fn(val):
            self._validate_input(val)
            vars(self)[key] = val
        return fn
    def get_constructor(self,key):
        def fn(rx=None,ry=None):
            return self._get_value(vars(self)[key],rx,ry)
        return fn

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
        if len(data[key].shape)==0:
            data[key] = data[key][0]
    dimensionkeys = ('R_Nx','R_Ny','Q_Nx','Q_Ny')
    for key in dimensionkeys:
        if key not in data.keys():
            data[key] = 1
    coordinates = Coordinates(dimensionkeys[0],dimensionkeys[1],
                              dimensionkeys[2],dimensionkeys[2])
    for key in dimensionkeys:
        del data[key]
    #for key in data.keys():
    #    coordinates.




    coordinates.name = name
    return coordinates






