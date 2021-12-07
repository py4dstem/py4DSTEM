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


def index_Bragg_peaks_from_orientation(
    self,
    bragg_peaks: PointList,
    orientation: np.ndarray,
    tol_distance: float = 0.05,
    sigma_excitation_error: float = 0.02,
    tol_excitation_error_mult: float = 3,
    tol_intensity: float = 0.1,
    k_max: float = None,
) -> PointList:
    """
    Given a set of experimental Bragg disk locations and the orientation matrix
    computed in ``match_single_pattern``, select the experimental peaks that are
    within ``tol_distance`` of the kinematically predicted ones and return a
    new PointList containing (qx,qy,Intensity,h,k,l) for the matching peaks.

    Args:
        bragg_peaks:        (PointList) peaks to index, with (qx,qy,intensity) fields
        orientation:        (tuple/array) orientation to generate comparison peaks from.
                                Can be 3-element for a zone axis or [3x3] matrix to include
                                in-plane rotation
        tol_distance        (float) distance threshold from ideal peaks to index an experimental peak
        The remaining args are passed on to Crystal.generate_diffraction_pattern:
        sigma_excitation_error
        tol_excitation_error_mult
        tol_intensity
        k_max

    Note that to index kinematically forbidden peaks present in a pattern, you will
    likely need to set tol_intensity=0

    """
    match_dtype = np.dtype(
        [
            ("qx", np.float64),
            ("qy", np.float64),
            ("intensity", np.float64),
            ("h", np.int64),
            ("k", np.int64),
            ("l", np.int64),
        ]
    )

    sim_peaks = self.generate_diffraction_pattern(
        zone_axis=orientation,
        sigma_excitation_error=sigma_excitation_error,
        tol_excitation_error_mult=tol_excitation_error_mult,
        tol_intensity=tol_intensity,
        k_max=k_max,
    )

    if sim_peaks.length == 0:
        print("Warning! No kinematic peaks found!")
        return PointList(match_dtype)

    # Accumulate matches as a list of len-1 arrays, then concatenate later
    # TODO: do this a smarter way
    matches = []
    # loop over all experimental peaks
    for i in range(bragg_peaks.length):
        # get current peak
        qx, qy = bragg_peaks.data["qx"][i], bragg_peaks.data["qy"][i]

        # get find closest peak, and check if it's within tol_distance
        dq = np.hypot(sim_peaks.data["qx"] - qx, sim_peaks.data["qy"] - qy)
        if np.min(dq) < tol_distance:
            idx = np.argmin(dq)
            matches.append(
                np.array(
                    [
                        (
                            sim_peaks.data["qx"][idx],
                            sim_peaks.data["qy"][idx],
                            bragg_peaks.data["intensity"][i],
                            sim_peaks.data["h"][idx],
                            sim_peaks.data["k"][idx],
                            sim_peaks.data["l"][idx],
                        )
                    ],
                    dtype=match_dtype,
                )
            )

    return PointList(match_dtype, np.array(matches))
