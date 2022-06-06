import numpy as np
from numbers import Number
from typing import Optional
import h5py

from .ioutils import determine_group_name
from .ioutils import EMD_group_exists, EMD_group_types


class Calibration:
    """
    Stores calibration measurements.

    Usage:

        >>> c = Calibration()
        >>> c.set_p(p)
        >>> p = c.get_p()

    If the parameter has not been set, the getter methods return None
    The value of a parameter may be a number, representing the entire dataset,
    or a 2D typically (R_Nx,R_Ny)-shaped array, representing values at each
    detector pixel. For parameters with 2D array values,

        >>> c.get_p()

    will return the entire 2D array, and

        >>> c.get_p(rx,ry)

    will return the value of `p` at position `rx,ry`.
    """
    def __init__(
        self,
        datacube_shape: Optional[tuple] = None,
        name: Optional[str] ='calibration'
        ):
        """
         Args:
            datacube_shape (Optional, 4-tuple): The datacube shape, (R_Nx,R_Ny,Q_Nx,Q_Ny)
        """
        self.name = name

        # create parameter dictionary
        self._params = {
            'Q_pixel_size':1,
            'Q_pixel_units':'pixels',
            'R_pixel_size':1,
            'R_pixel_units':'pixels'
        }

        if datacube_shape is not None:
            self.set_datacube_shape(datacube_shape)


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
        return (R_Nx,R_Ny,Q_Nx,Q_Ny)

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
        self._params['Q_pixel_units'] = x
    def get_Q_pixel_units(self):
        return self._get_value('Q_pixel_units')
    def set_R_pixel_units(self,x):
        self._params['R_pixel_units'] = x
    def get_R_pixel_units(self):
        return self._get_value('R_pixel_units')


    # origin
    def set_origin(self,x):
        """
        Args:
            x (3D array of shape (R_Nx,R_Ny,2): the origin
        """
        self._params['origin'] = x
    def get_origin(self,rx=None,ry=None):
        return self._get_value('origin',rx,ry)
    def set_origin_meas(self,x):
        """
        Args:
            x (3D array of shape (R_Nx,R_Ny,2): the measured origin
        """
        self._params['origin_meas'] = x
    def get_origin_meas(self,rx=None,ry=None):
        return self._get_value('origin_meas',rx,ry)
    def set_origin_residuals(self,x):
        """
        Args:
            x (3D array of shape (R_Nx,R_Ny,2): the residuals of fitting the origin
        """
        self._params['origin_residuals'] = x
    def get_origin_residuals(self,rx=None,ry=None):
        return self._get_value('origin_residuals',rx,ry)

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
        a = self._get_value('a',rx,ry)
        b = self._get_value('b',rx,ry)
        theta = self._get_value('theta',rx,ry)
        return (a,b,theta)
    def get_p_ellipse(self,rx=None,ry=None):
        origin = self._get_origin('origin',rx,ry)
        qx0,qy0 = origin[...,0],origin[...,1]
        if qx0.ndim == 0: qx0 = qx0.item()
        if qy0.ndim == 0: qy0 = qy0.item()
        a = self._get_value('a',rx,ry)
        b = self._get_value('b',rx,ry)
        theta = self._get_value('theta',rx,ry)
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



    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A Calibration metadata instance called '{self.name}', containing the following fields:"
        string += "\n"

        maxlen = 0
        for k in self._params.keys():
            if len(k)>maxlen: maxlen=len(k)

        for k,v in self._params.items():
            if isinstance(v,np.ndarray):
                v = f"{v.ndim}D-array"
            string += "\n"+space+f"{k}:{(maxlen-len(k)+3)*' '}{str(v)}"
        string += "\n)"

        return string



    ## Writing to an HDF5 file

    def to_h5(self,group):
        """
        Takes a valid HDF5 group for an HDF5 file object which is open in write or append
        mode. Writes a new group with a name given by this Calibration instance's .name
        field nested inside the passed group, and saves the data there.

        If the Calibration instance has no name, it will be assigned the name
        Calibration"#" where # is the lowest available integer.  If the instance has a name
        which already exists here in this file, raises and exception.

        TODO: add overwite option.

        Accepts:
            group (HDF5 group)
        """

        # Detemine the name of the group
        # if current name is invalid, raises and exception
        # TODO: add overwrite option
        determine_group_name(self, group)


        ## Write

        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",0) # this tag indicates a Calibration dictionary
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # Save data
        for k,v in self._params.items():
            if isinstance(v,str): v = np.string_(v)
            grp.create_dataset(k, data=v)


## Read Calibration objects

def Calibration_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid Calibration object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A Calibration instance
    """
    er = f"No Calibration instance called {name} could be found in group {group} of this HDF5 file."
    assert(EMD_group_exists(
            group,
            EMD_group_types['Calibration'],
            name)), er
    grp = group[name]


    # Get metadata
    name = grp.name.split('/')[-1]

    # Get data
    data = {}
    for k,v in grp.items():
        v = np.array(v)
        if v.ndim==0: v=v.item()
        if isinstance(v,bytes):
            v = v.decode('utf-8')
        data[k] = v

    cal = Calibration(name=name)
    cal._params.update(data)

    return cal



