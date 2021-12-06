import numpy as np
from scipy import linalg
from typing import Union, Optional
from time import time
from tqdm import tqdm

from ...io.datastructure import PointList


def estimate_thickness(
    self, bragg_peaks: PointList, bloch_beams: PointList, thickness: np.ndarray
) -> float:
    """
    Estimate thickness of diffraction pattern encoded in ``bragg_peaks`` by computing
    a thickness series of dynamical patters using ``bloch_beams`` and comparing
    relative intensities.

    Args:
        bragg_peaks (PointList):        experimental measured disk intensities, with hkl indices
        bloch_beams (PointList):        beams to include in the Bloch wave dynamical diffraction
                                        calculation
        thickness (ndarray):            Array of thickness values to compare against
    """
    return


def index_pattern(
    self,
    bragg_peaks: PointList,
    orientation: np.ndarray,
    tol_distance: float,
    signal_excitation_error: float,
) -> PointList:
    """
    Given a set of experimental Bragg disk locations and the orientation matrix
    computed in ``match_single_pattern``, select the experimental peaks that are
    within ``tol_distance`` of the kinematically predicted ones and return a
    new PointList containing (qx,qy,Intensity,h,k,l) for the matching peaks.
    """
    return
