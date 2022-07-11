# BraggVectors methods

import numpy as np
from ..emd import Metadata






# Bragg vector maps

def get_bvm(
    self,
    Qshape,
    mode = 'centered',
    ):
    """
    Gets a Bragg vector map, a 2D histogram of Bragg scattering vectors.

    Args:
        Qshape (2 tuple): diffraction space shape
        mode (str): must be 'raw' or 'centered'. TODO, sampling selection

    Args:
        (2D array):
    """
    assert mode in ('raw','centered')

    # select peaks
    peaks = self.v if mode=='centered' else self.v_uncal

    # perform computation
    from ....process.diskdetection import get_bvm
    bvm = get_bvm(
        peaks,
        Qshape = Qshape,
        mode = mode,
    )

    return bvm




def measure_origin(
    self,
    mode,
    returncalc = True,
    **kwargs,
    ):
    """
    Modes of operation are 2 or 5.  Use-cases and input arguments:

    2 - A set of bragg peaks for data with no beamstop, and in which
        the center beam is brightest throughout.

        Args:
            data (PointListArray)
            Q_Nx (int)
            Q_Ny (int)

    5 - A set of bragg peaks for data with a beamstop

        Args:
            data (PointListArray)
            center_guess (2-tuple)
            radii   (2-tuple)
            Q_Nx (int)
            Q_Ny (int)

    """
    assert mode in (2,5)

    # perform computation
    from ....process.calibration import measure_origin
    origin = measure_origin(
        self.v_uncal,
        mode = mode,
        **kwargs
    )

    # try to add to calibration
    try:
        self.calibration.set_origin_meas(origin)
    except AttributeError:
        # should a warning be raised?
        pass

    if returncalc:
        return origin




# Calibrate

def calibrate(
    self,
    returncalc = False
    ):
    """
    Determines which calibrations are present in set.calibrations (of origin,
    elliptical, pixel, rotational), and applies any it finds to self.v_uncal,
    storing the output in self.v.

    Returns:
        (PointListArray)
    """
    try:
        cal = self.calibration
    except AttributeError:
        raise Exception('No .calibration attribute found')

    from ....process.calibration.braggvectors import calibrate

    v = self.v_uncal.copy( name='v_cal' )
    v = calibrate(
        v,
        cal
    )
    self._v_cal = v

    if returncalc:
        return v









