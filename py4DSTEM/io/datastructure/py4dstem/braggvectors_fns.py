# BraggVectors methods

import numpy as np
from ..emd import Metadata






# Bragg vector maps

def get_bvm(
    self,
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
    peaks = self.vectors if mode=='centered' else self.vectors_uncal

    # perform computation
    from ....process.diskdetection import get_bvm
    bvm = get_bvm(
        peaks,
        Qshape = self.Qshape,
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

    "bragg_no_beamstop" - A set of bragg peaks for data with no beamstop, and in which
        the center beam is brightest throughout.

        Args:
            data (PointListArray)

    "bragg_beam_stop" - A set of bragg peaks for data with a beamstop

        Args:
            data (PointListArray)
            center_guess (2-tuple)
            radii   (2-tuple)

    """
    assert mode in ("bragg_no_beamstop","bragg_beamstop")

    # perform computation
    from ....process.calibration import measure_origin
    
    kwargs["Q_shape"] = self.Qshape
    
    origin = measure_origin(
        self.vectors_uncal,
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


def fit_origin(
    self,
    mask=None,
    fitfunction="plane",
    robust=False,
    robust_steps=3,
    robust_thresh=2,
    plot = True,
    returncalc = True,
    ):
    """

    
    """
    q_meas = self.calibration.get_origin_meas()
    from ....process.calibration import fit_origin
    qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = fit_origin(tuple(q_meas))

    # try to add to calibration
    try:
        self.calibration.set_origin([qx0_fit,qy0_fit])
    except AttributeError:
        # should a warning be raised?
        pass
    if plot: 
        from ....visualize import show_image_grid
        qx0_meas,qy0_meas = q_meas
        qx0_mean = np.mean(qx0_fit)
        qy0_mean = np.mean(qy0_fit)

        show_image_grid(
            lambda i:[qx0_meas-qx0_mean,qx0_fit-qx0_mean,qx0_residuals,
                      qy0_meas-qy0_mean,qy0_fit-qy0_mean,qy0_residuals][i],
            H = 2,
            W = 3,
            cmap = 'RdBu',
            clipvals = 'manual',
            vmin = -1,
            vmax = 1,
            axsize = (6,2),
        )

    if returncalc:
        return qx0_fit,qy0_fit,qx0_residuals,qy0_residuals 

# Calibrate
def calibrate(
    self,
    use_fitted_origin = True,
    returncalc = False,
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

    v = self.vectors_uncal.copy( name='v_cal' )
    v = calibrate( 
        v,
        cal,
        use_fitted_origin,
    )
    self._v_cal = v

    if returncalc:
        return v









