# Defines the Strain class

import numpy as np
from typing import Optional
from py4DSTEM.data import RealSlice, Data
from py4DSTEM.braggvectors import BraggVectors



class StrainMap(RealSlice,Data):
    """
    Stores strain map.

    TODO add docs

    """

    def __init__(
        self,
        braggvectors: BraggVectors,
        name: Optional[str] = 'strainmap'
        ):
        """
        TODO
        """
        assert(isinstance(braggvectors,BraggVectors)), f"braggvectors myst be BraggVectors, not type {type(braggvectors)}"

        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            name = name,
            data = np.empty((
                6,
                braggvectors.Rshape[0],
                braggvectors.Rshape[1],
            )),
            slicelabels = [
                'exx',
                'eyy',
                'exy',
                'theta',
                'mask',
                'error'
            ]
        )

        # set up braggvectors
        self.braggvectors = braggvectors
        # TODO - how to handle changes to braggvectors
        #   option: register with calibrations and add a .calibrate method
        #   which {{does something}} when origin changes
        # TODO - include ellipse cal or no?

        assert(self.root is not None)

        # initialize as Data
        Data.__init__(
            self,
            calibration = self.braggvectors.calibration
        )


    # braggvector properties

    @property
    def braggvectors(self):
        return self._braggvectors
    @braggvectors.setter
    def braggvectors(self,x):
        assert(isinstance(x,BraggVectors)), f".braggvectors must be BraggVectors, not type {type(x)}"
        assert(x.calibration.origin is not None), f"braggvectors must have a calibrated origin"
        self._braggvectors = x
        self._braggvectors.tree(self,force=True)



    # Class methods

    def choose_lattice_vectors(
        self,
        index_g0,
        index_g1,
        index_g2,
        mode = 'centered',
        plot = True,
        subpixel = 'multicorr',
        upsample_factor = 16,
        sigma=0,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=10,
        bvm_vis_params = {},
        returncalc = False,
        ):
        """
        Choose which lattice vectors to use for strain mapping.

        Args:
            index_g0 (int): origin
            index_g1 (int): second point of vector 1
            index_g2 (int): second point of vector 2
            mode (str): centered or raw bragg map
            plot (bool): plot bragg vector maps and vectors
            subpixel (str): specifies the subpixel resolution algorithm to use.
                must be in ('pixel','poly','multicorr'), which correspond
                to pixel resolution, subpixel resolution by fitting a
                parabola, and subpixel resultion by Fourier upsampling.
            upsample_factor: the upsampling factor for the 'multicorr'
                algorithm
            sigma: if >0, applies a gaussian filter
            maxNumPeaks: the maximum number of maxima to return
            minAbsoluteIntensity, minRelativeIntensity, relativeToPeak,
                minSpacing, edgeBoundary, maxNumPeaks: filtering applied
                after maximum detection and before subpixel refinement
        """
        from py4DSTEM.process.utils import get_maxima_2D

        if mode == "centered":
            bvm = self.bvm_centered
        else:
            bvm = self.bvm_raw

        g = get_maxima_2D(
            bvm,
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            sigma = sigma,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minSpacing = minSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
        )

        self.g = g

        from py4DSTEM.visualize import select_lattice_vectors
        g1,g2 = select_lattice_vectors(
            bvm,
            gx = g['x'],
            gy = g['y'],
            i0 = index_g0,
            i1 = index_g1,
            i2 = index_g2,
            **bvm_vis_params,
        )

        self.g1 = g1
        self.g2 = g2

        if returncalc:
            return g1, g2











    # IO methods

    # TODO - copy method

    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = RealSlice._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
        }
        return args



