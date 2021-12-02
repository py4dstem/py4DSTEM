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
