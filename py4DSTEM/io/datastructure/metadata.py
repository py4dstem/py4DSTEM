from . import DataObject

class Metadata(DataObject):
    """
    A class for storing metadata.

    Items of metadata are stored in five dictionaries inside a Metadata
    instance. Access, adding, and editing can be accomplished directly
    from the dictionaries, or can be accomplished using the get/set
    methods.

    The dictionaries are: 'microscope', 'calibration', 'sample', 'user', 'comments'
    They are reserved for the following uses:
    'microscope': everything from the raw / original file goes here.
    'calibration': all calibrations added later by the user go here.
    'sample': information about the sample and sample prep.
    'user': information about the microscope operator who acquired the data,
            as well as the user who performed the computational analysis.
    'comments': general use space for any other information

    Note that certain pieces of metadata may exist in two places - 'microscope'
    and 'calibration'.  For instance, this would occur with the pixel sizes if
    (1) the microscope's pixel size calibrations were attached to the original
    file and automatically added to the 'microscope' dictionary, and then (2)
    during processing the user re-calibrates the pixel sizes manually, e.g. using
    a reference sample to achieve optimal accuracy/precision, and stores their
    new calibrations using the set_Q_pixel_size method, which will store the
    values in 'calibration'. This has the advantage of keeping all the original
    data, while also allowing more refined calibrations.  Retreiving metadata with
    the get methods (e.g. get_Q_pixel_size) will default to using the values in
    'calibration' if they are present and 'microscope' if they are not, unless
    the keyword argument 'where' is specified.
    """
    def __init__(self):
        """
        Initializes a new class instance with empty dicts for the
        py4DSTEM metadata types.
        """
        DataObject.__init__(self)

        self.microscope = {}
        self.sample = {}
        self.user = {}
        self.calibration = {}
        self.comments = {}

        self.dicts = ('microscope','sample','user','calibration','comments')

    ####### Begin get/set methods #######

    # Pixel sizes
    def set_R_pixel_size(self,val):
        self.calibration['R_pixel_size'] = val
    def set_R_pixel_size_units(self,val):
        self.calibration['R_pixel_size_units'] = val
    def get_R_pixel_size_calibration(self):
        return self.calibration['R_pixel_size']
    def get_R_pixel_size_microscope(self):
        return self.microscope['R_pixel_size']
    def get_R_pixel_size(self,where=False):
        key = 'R_pixel_size'
        if key in self.calibration.keys():
            _w='calibration'
            val = self.calibration[key]
        elif key in self.microscope.keys():
            _w ='microscope'
            val = self.microscope[key]
        else:
            raise KeyError("'R_pixel_size' not found in metadata")
        if where:
            print("R_pixel_size retrieved from {}".format(_w))
        return val
    def get_R_pixel_size_units_microscope(self):
        return self.microscope['R_pixel_size_units']
    def get_R_pixel_size_units_calibration(self):
        return self.calibration['R_pixel_size_units']
    def get_R_pixel_size_units(self,where=False):
        key = 'R_pixel_size_units'
        if key in self.calibration.keys():
            _w='calibration'
            val = self.calibration[key]
        elif key in self.microscope.keys():
            _w ='microscope'
            val = self.microscope[key]
        else:
            raise KeyError("'R_pixel_size_units' not found in metadata")
        if where:
            print("R_pixel_size_units retrieved from {}".format(_w))
        return val

    def set_Q_pixel_size(self,val):
        self.calibration['Q_pixel_size'] = val
    def set_Q_pixel_size_units(self,val):
        self.calibration['Q_pixel_size_units'] = val
    def get_Q_pixel_size_calibration(self):
        return self.calibration['Q_pixel_size']
    def get_Q_pixel_size_microscope(self):
        return self.microscope['Q_pixel_size']
    def get_Q_pixel_size(self,where=False):
        key = 'Q_pixel_size'
        if key in self.calibration.keys():
            _w='calibration'
            val = self.calibration[key]
        elif key in self.microscope.keys():
            _w ='microscope'
            val = self.microscope[key]
        else:
            raise KeyError("'{}' not found in metadata".format(key))
        if where:
            print("'{}' retrieved from {}".format(key,_w))
        return val
    def get_Q_pixel_size_units_microscope(self):
        return self.microscope['Q_pixel_size_units']
    def get_Q_pixel_size_units_calibration(self):
        return self.calibration['Q_pixel_size_units']
    def get_Q_pixel_size_units(self,where=False):
        key = 'Q_pixel_size_units'
        if key in self.calibration.keys():
            _w='calibration'
            val = self.calibration[key]
        elif key in self.microscope.keys():
            _w ='microscope'
            val = self.microscope[key]
        else:
            raise KeyError("'{}' not found in metadata",key)
        if where:
            print("{} retrieved from {}".format(key,_w))
        return val

    # Elliptical distortions
    def get_elliptical_distortions(self):
        """ a,b,theta
        """
        return self.get_a(),self.get_b(),self.get_theta()
    def set_elliptical_distortions(self,a,b,theta):
        """ a,b,theta
        """
        self.set_a(a)
        self.set_b(b)
        self.set_theta(theta)
    def get_a(self):
        return self.calibration['a']
    def get_b(self):
        return self.calibration['b']
    def get_theta(self):
        return self.calibration['theta']
    def set_a(self,a):
        self.calibration['a'] = a
    def set_b(self,b):
        self.calibration['b'] = b
    def set_theta(self,theta):
        self.calibration['theta'] = theta





#    def get_accelerating_voltage():
#        return
#    def set_accelerating_voltage():
#        return
#    def get_accelerating_voltage_units():
#        return
#    def set_accelerating_voltage_units():
#        return







