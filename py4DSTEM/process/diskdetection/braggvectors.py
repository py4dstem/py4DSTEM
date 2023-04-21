# Defines the BraggVectors class

from typing import Optional,Union
import numpy as np
from os.path import basename

from py4DSTEM.classes import Data
from py4DSTEM.process.diskdetection.braggvector_methods import BraggVectorMethods
from emdfile import Custom,PointListArray,Metadata





class BraggVectors(Custom,BraggVectorMethods,Data):
    """
    Stores localized bragg scattering positions and intensities
    for a 4D-STEM datacube.

    Raw (detector coordinate) vectors are accessible as

        >>> braggvectors.raw[ scan_x, scan_y ]

    and calibrated vectors as

        >>> braggvectors.cal[ scan_x, scan_y ]

    To set which calibrations are being applied, call

        >>> braggvectors.setcal(
        >>>     center = bool,
        >>>     ellipse = bool,
        >>>     pixel = bool
        >>> )

    If .setcal is not called, calibrations will be automatically selected based
    based on the contents of the instance's `calibrations` property. The
    calibrations performed in the last call to `braggvectors.cal` are exposed as

        >>> braggvectors.calstate

    After grabbing some vectors

        >>> vects = braggvectors.raw[ scan_x,scan_y ]

    the values themselves are accessible as

        >>> vects.qx,vects.qy,vects.I
        >>> vects['qx'],vects['qy'],vects['intensity']

    """

    def __init__(
        self,
        Rshape,
        Qshape,
        name = 'braggvectors'
        ):
        Custom.__init__(self,name=name)

        self.Rshape = Rshape
        self.shape = self.Rshape
        self.Qshape = Qshape

        self._v_uncal = PointListArray(
            dtype = [
                ('qx',np.float64),
                ('qy',np.float64),
                ('intensity',np.float64)
            ],
            shape = Rshape,
            name = '_v_uncal'
        )

        # calibrated vector getters
        # nested list of 2**3 vector getters for [center][ellipse][pixel]-like access
        # where entries 0,1 correspond to False,True values of the params
        self._cal_vector_getters = [[[
            None for cen in range(2)] for ell in range(2)] for pix in range(2)]
        self._calstate = None

    @property
    def calstate(self):
        return self._calstate



    # raw vectors

    @property
    def raw(self):
        """
        Calling

            >>> raw[ scan_x, scan_y ]

        returns those bragg vectors.
        """

        # check if a raw vector getter exists
        # if it doesn't, make it
        if not hasattr(self,'_raw_vector_getter'):
            self._raw_vector_getter = RawVectorGetter(
                data = self._v_uncal
            )

        # use the vector getter to grab the vector
        return self._raw_vector_getter



    # calibrated vectors

    @property
    def cal(self):
        """
        Calling

            >>> cal[ scan_x, scan_y ]

        retrieves data.  Use `.setcal` to set the calibrations to be applied, or
        `.calstate` to see which calibrations are currently set.  Requesting data
        before setting calibrations will automatically select values based
        calibrations that are available.
        """

        # check if a calibration state is set
        # if not, autoselect calibrations and make a getter
        if self.calstate is None:
            self.setcal()

        # retrieve the getter and return
        getter = self._cal_vector_getters[
            self.calstate['center']][
            self.calstate['ellipse']][
            self.calstate['pixel']]
        return getter


    # set calibration state

    def setcal(
        self,
        center = None,
        ellipse = None,
        pixel = None
    ):
        """
        Calling

            >>> braggvectors.setcal(
            >>>     center = bool,
            >>>     ellipse = bool,
            >>>     pixel = bool
            >>> )

        sets the calibrations that will be applied to vectors subsequently
        retrieved with

            >>> braggvectors.cal[ scan_x, scan_y ]

        Any arguments left as `None` will be automatically set based on
        the calibration measurements available.
        """

        # autodetect
        if center is None:
            # TODO center = ...
            pass
        if ellipse is None:
            # TODO
            pass
        if pixel is None:
            # TODO
            pass

        # make the requested vector getter
        # and set the calstate
        transform = get_transform(  # TODO - write this.....!
            calibration = self.calibration,
            center = center,
            ellipse = ellipse,
            pixel = pixel
        )
        calstate = {
            "center" : center,
            "ellipse" : ellipse,
            "pixel" : pixel
        }
        getter = CaliVectorGetter(
            data = self._v_uncal,
            transform = transform,
            calstate = calstate
        )
        self._cal_vector_getters[center][ellipse][pixel] = getter
        self._calstate = calstate

        pass




    # copy

    def copy(self, name=None):
        name = name if name is not None else self.name+"_copy"
        braggvector_copy = BraggVectors(self.Rshape, self.Qshape, name=name)
        braggvector_copy._v_uncal = self._v_uncal.copy()
        try:
            braggvector_copy._v_cal = self._v_cal.copy()
        except AttributeError:
            pass
        for k in self.metadata.keys():
            braggvector_copy.metadata = self.metadata[k].copy()
        self.root.tree(braggvector_copy)

        return braggvector_copy

    # write

    def to_h5(self,group):
        """ Constructs the group, adds the bragg vector pointlists,
        and adds metadata describing the shape
        """
        md = Metadata( name = '_braggvectors_shape' )
        md['Rshape'] = self.Rshape
        md['Qshape'] = self.Qshape
        self.metadata = md
        grp = Custom.to_h5(self,group)

    # read

    @classmethod
    def _get_constructor_args(cls,group):
        """
        """
        # Get shape metadata from the metadatabundle group
        assert('metadatabundle' in group.keys()), "No metadata found, can't get Rshape and Qshape"
        grp_metadata = group['metadatabundle']
        assert('_braggvectors_shape' in grp_metadata.keys()), "No _braggvectors_shape metadata found"
        md = Metadata.from_h5(grp_metadata['_braggvectors_shape'])
        # Populate args and return
        kwargs = {
            'name' : basename(group.name),
            'Rshape' : md['Rshape'],
            'Qshape' : md['Qshape']
        }
        return kwargs

    def _populate_instance(self,group):
        """
        """
        dic = self._get_emd_attr_data(group)
        assert('_v_uncal' in dic.keys()), "Uncalibrated bragg vectors not found!"
        self._v_uncal = dic['_v_uncal']
        if '_v_cal' in dic.keys():
            self._v_cal = dic['_v_cal']

    # standard output display

    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string





# Support classes


class RawVectorGetter(
    def __init__(
        self,
        data
    ):
        self._data = _data

    def __getitem__(self,pos):
        x,y = pos
        ans = self._data[x,y]
        return BVects(ans)


class CaliVectorGetter:
    def __init__(
        self,
        data,
        transform,
        calstate
    ):
        self._data = data
        self._transform = transform
        self.calstate = calstate

    def __getitem__(self,pos):
        x,y = pos
        ans = self._data[x,y]
        ans = self._transform(ans)
        return BVects(ans)


class BVects:
    """
    Enables

        >>> v.qx,v.qy,v.I

    -like access to a collection of Bragg vector.
    """

    def __init__(
        self,
        pointlist
        ):
        """ pointlist must have fields 'qx', 'qy', and 'intensity'
        """
        assert(isinstance(pointlist,PointList))
        self._data = poinstlist.data

    @property
    def qx(self):
        return self._data['qx']

    @property
    def qy(self):
        return self._data['qy']

    @property
    def I(self):
        return self._data['intensity']




