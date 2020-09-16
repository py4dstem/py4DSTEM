import h5py
from os.path import exists
from .h5rw import h5write,h5read,h5append,h5info
from ..read_utils import is_py4DSTEM_file,get_py4DSTEM_topgroups
from ..write import save

class Metadata(object):
    """
    A class for reading, writing, and storing metadata. This is the
    pythonic interface for metadata in py4DSTEM.  It enables metadata
    to be read/written from/to HDF5, read from non-native files, and
    read/written from the python interpretter in-program.

    External file methods:
        to_h5(fp)           # Writes to a py4DSTEM .h5 file
        from_h5(fp)         # Reads from a py4DSTEM .h5
        from_nonnative(fp)  # Reads from nonnative files

    In-program metadata access and editing can be accomplished directly
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

    def to_h5(self, fp, overwrite=False, topgroup=None):
        """
        If fp points to an existing .h5 file, and if no metadata is
        present, writes it there.
        If fp points to an existing .h5 file which already contains
        some metadata, appends any new metadata from the object
        instance to the .h5, and also checks if any conflicting
        metadata exists (same keys, different values) -- if it does,
        overwrites iff overwrite=True, otherwise skips these items.
        If there is no file at fp, writes a new .h5 file there,
        formatted as a py4DSTEM file, and containing only the metadata.
        If an existing .h5 file has more than one topgroup, the
        'topgroup' argument should be passed which one to write to.
        """
        if exists(fp):
            assert is_py4DSTEM_file(fp), "File found at path {} is not recognized as a py4DSTEM file.".format(fp)
        else:
            save(fp,[])
        tgs = get_py4DSTEM_topgroups(fp)
        if topgroup is not None:
            assert(topgroup in tgs), "Error: specified topgroup, {}, not found.".format(topgroup)
        elif len(tgs)==1:
            tg = tgs[0]
        else:
            print("Multiple topgroups detected.  Please specify one by passing the 'topgroup' keyword argument.")
            print("")
            print("Topgroups found:")
            for tg in tgs:
                print(tg)

        with h5py.File(fp,'r+') as f:
            if 'metadata' not in f[tg]: f[tg].create_group('metadata')
            mgp = f[tg + '/metadata']
            if 'microscope' not in mgp: mgp.create_group('microscope')
            if 'sample' not in mgp: mgp.create_group('sample')
            if 'user' not in mgp: mgp.create_group('user')
            if 'calibration' not in mgp: mgp.create_group('calibration')
            if 'comments' not in mgp: mgp.create_group('comments')

        h5append(fp,tg+'/metadata/microscope',self.microscope)
        h5append(fp,tg+'/metadata/calibration',self.calibration)
        h5append(fp,tg+'/metadata/sample',self.sample)
        h5append(fp,tg+'/metadata/user',self.user)
        h5append(fp,tg+'/metadata/comments',self.comments)

        return

    def from_h5(self, fp, topgroup=None):
        """
        Reads the metadata in a py4DSTEM formatted .h5 file and stores it in the
        Metadata instance.
        """
        assert is_py4DSTEM_file(fp), "File found at path {} is not recognized as a py4DSTEM file.".format(fp)
        tgs = get_py4DSTEM_topgroups(fp)
        if topgroup is not None:
            assert(topgroup in tgs), "Error: specified topgroup, {}, not found.".format(topgroup)
        elif len(tgs)==1:
            tg = tgs[0]
        else:
            print("Multiple topgroups detected.  Please specify one by passing the 'topgroup' keyword argument.")
            print("")
            print("Topgroups found:")
            for tg in tgs:
                print(tg)

        self.microscope = h5read(fp,tg+'/metadata/microscope')
        self.calibration = h5read(fp,tg+'/metadata/calibration')
        self.sample = h5read(fp,tg+'/metadata/sample')
        self.user = h5read(fp,tg+'/metadata/user')
        self.comments = h5read(fp,tg+'/metadata/comments')

        return self


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







