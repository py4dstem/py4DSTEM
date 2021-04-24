from . import DataObject

class Metadata(DataObject):
    """
    A class for storing metadata.

    Items of metadata are stored in five dictionaries inside a Metadata
    instance. Access, adding, and editing can be accomplished directly
    from the dictionaries, or can be accomplished using the get/set
    methods.

    The dictionaries are: 'microscope', 'calibration', 'sample', 'user', 'comments'
    They are intended for the following uses:
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

        self.dicts = {'microscope':self.microscope,
                      'sample':self.sample,
                      'user':self.user,
                      'calibration':self.calibration,
                      'comments':self.comments}

        # Add new metadata items here. The get/set methods are generated from this table.
        # Keys are the metadata items name, values are the dictionaries they belong in
        # Multiple dictionaries will create multiple get/set methods, one for each dict
        # with the LAST entry in the tuple specifying the default dictionary for this key
        getset_lookup = {
            'R_pixel_size':('microscope','calibration'),
            'R_pixel_size_units':('microscope','calibration'),
            'Q_pixel_size':('microscope','calibration'),
            'Q_pixel_size_units':('microscope','calibration'),
            'e':('calibration',),   # elliptical distortion
            'theta':('calibration',),
            'beam_energy':('microscope',),
            'QR_rotation':('microscope','calibration'),
            'QR_rotation_units':('microscope','calibration')
        }

        # Make the get/set methods
        for k in getset_lookup:
            dics = getset_lookup[k]
            if len(dics)==1:
                dic = dics[0]
                # Construct normal get/set functions
                setattr(self,'set_'+k,self.set_constructor(self.dicts[dic],k))
                setattr(self,'get_'+k,self.get_constructor(self.dicts[dic],k))
            else:
                for dic in dics:
                    # get/set fns specifying one of multiple possible dicts
                    setattr(self,'set_'+k+'__'+dic,self.set_constructor(self.dicts[dic],k))
                    setattr(self,'get_'+k+'__'+dic,self.get_constructor(self.dicts[dic],k))
                # get/set fns which draw from the 'best' available source
                setattr(self,'set_'+k,self.set_constructor(self.dicts[dics[-1]],k))
                setattr(self,'get_'+k,self.get_constructor_multiDict([self.dicts[dic] for dic in dics],k))

    def set_constructor(self,dic,key):
        def fn(val):
            dic[key] = val
        return fn
    def get_constructor(self,dic,key):
        def fn():
            return dic[key]
        return fn
    def get_constructor_multiDict(self,dics,key):
        def fn():
            for dic in dics[::-1]:
                try:
                    return dic[key]
                except KeyError:
                    pass
            raise Exception('Metadata not found')
        return fn


    # Additional convenience get/set methods

    def get_elliptical_distortions(self):
        """ a,b,theta
        """
        return self.get_a(),self.get_b(),self.get_theta()
    def set_elliptical_distortions(self,a,b,theta):
        """ a,b,theta
        """
        self.set_a(a),self.set_b(b),self.set_theta(theta)










