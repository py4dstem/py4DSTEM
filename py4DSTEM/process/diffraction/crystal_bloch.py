import warnings
import numpy as np
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray

def generate_dynamical_diffraction_pattern(
    self,
    beams: PointList,
    thickness: Union[float,list,tuple,np.ndarray],
    zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
    foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
):
    """
    Generate a dynamical diffraction pattern (or thickness series of patterns) 
    using the Bloch wave method.

    The beams to be included in the Bloch calculation must be pre-calculated
    and passed as a PointList containing at least (qx, qy, h, k, l) fields.

    If ``thickness`` is a single value, one new PointList will be returned.
    If ``thickness`` is a sequence of values, a list of PointLists will be returned, 
        corresponding to each thickness value in the input.
    
    Frequent reference will be made to 

    Args:
        beams (PointList):              PointList from the kinematical diffraction generator
                                        which will define the beams included in the Bloch calculation
        thickness (float or list/array) thickness to evaluate diffraction patterns at.
                                        The main Bloch calculation can be reused for multiple thicknesses
                                        without much overhead. 
        zone_axis (np float vector):     3 element projection direction for sim pattern
                                         Can also be a 3x3 orientation matrix (zone axis 3rd column)
        foil_normal:                     3 element foil normal - set to None to use zone_axis
        proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)

    Returns:
        bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, h, k, l]
    """

    # Compute the reduced structure matrix \hat{A} in DeGraef 5.52

    # Compute eigen-decomposition of \hat{A} to yield C (the matrix containing the eigenvectors
    # as its columns) and gamma (the reduced eigenvalues), as in DeGraef 5.52

    # Compute thickness matrix/matrices E (DeGraef 5.60)

    # Compute diffraction intensities by calculating exit wave \Psi in DeGraef 5.60, and collect
    # values into PointLists