

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

    def to_h5(fp, overwrite=False):
        """
        If no metadata is present in the .h5 at fp, writes it there.
        If metadata is already present, appends any new metadata from
        the object instance to the .h5, and also checks if any
        conflicting metadata exists (same keys, different values).
        In this case, overwrites iff overwrite=True, otherwise skips
        these items.
        """
        return

    def from_h5(fp):
        """
        Reads the metadata in a py4DSTEM formatted .h5 file and stores
        it in the Metadata instance.
        """
        return

    def from_nonnative(fp):
        """
        Reads the metadata in a non-native file format and stores
        it in the Metadata instance.
        """
        return

    ### Begin get/set methods ###

    def get_R_pixel_size():
        return
    def set_R_pixel_size():
        return

    def get_R_pixel_size_microscope():
        return
    def set_R_pixel_size_microscope():
        return

    def get_R_pixel_size_calibrated():
        return
    def set_R_pixel_size_calibrated():
        return

    def get_Q_pixel_size():
        return
    def set_Q_pixel_size():
        return

    def get_Q_pixel_size_microscope():
        return
    def set_Q_pixel_size_microscope():
        return

    def get_Q_pixel_size_calibrated():
        return
    def set_Q_pixel_size_calibrated():
        return

    def get_elliptical_distortions():
        return
    def set_elliptical_distortions():
        return

    def get_centerposition():
        return
    def set_centerposition():
        return

    def get_centerposition_meas():
        return
    def set_centerposition_meas():
        return

    def get_centerposition_fit():
        return
    def set_centerposition_fit():
        return

    def get_accelerating_voltage():
        return
    def set_accelerating_voltage():
        return







