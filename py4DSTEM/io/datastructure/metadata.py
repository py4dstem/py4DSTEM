
class Metadata(object):
    """
    A class for storing metadata.

    Items of metadata are stored in five dictionaries inside a Metadata
    instance. Access, adding, and editing can be accomplished directly
    from the dictionaries, or can be accomplished using the get/set
    methods.
    """
    def __init__(self):
        """
        Initializes a new class instance with empty dicts for the
        py4DSTEM metadata types.
        """
        self.microscope = {}
        self.sample = {}
        self.user = {}
        self.calibration = {}
        self.comments = {}

    ### Begin get/set methods ###

    def set_R_pixel_size(self,val):
        self.calibration['R_pixel_size'] = val
    def set_R_pixel_size_units(self,val):
        self.calibration['R_pixel_size_units'] = val
    def get_R_pixel_size_calibration():
        return self.calibration['R_pixel_size']
    def get_R_pixel_size_microscope():
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
    def get_R_pixel_size_units_microscope():
        return self.microscope['R_pixel_size_units']
    def get_R_pixel_size_units_calibration():
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
    def get_Q_pixel_size_calibration():
        return self.calibration['Q_pixel_size']
    def get_Q_pixel_size_microscope():
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
    def get_Q_pixel_size_units_microscope():
        return self.microscope['Q_pixel_size_units']
    def get_Q_pixel_size_units_calibration():
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






#    def get_elliptical_distortions():
#        return
#    def set_elliptical_distortions():
#        return
#
#    def get_centerposition():
#        return
#    def set_centerposition():
#        return
#
#    def get_centerposition_meas():
#        return
#    def set_centerposition_meas():
#        return
#
#    def get_centerposition_fit():
#        return
#    def set_centerposition_fit():
#        return
#
#    def get_accelerating_voltage():
#        return
#    def set_accelerating_voltage():
#        return
#    def get_accelerating_voltage_units():
#        return
#    def set_accelerating_voltage_units():
#        return







