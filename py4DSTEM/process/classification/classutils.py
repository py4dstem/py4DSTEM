# Utility functions for classification routines

import numpy as np
from ..utils import get_shifted_ar
from ...file.datastructure import DataCube

def get_class_DP(datacube, class_image, xshifts=None, yshifts=None):
    """
    Get the average diffraction pattern for the class described in real space by class_image.

    Accepts:
        datacube        (DataCube) a datacube
        class_image     (2D array) the weight of the class at each position in real space
        xshifts         (2D array, or None) the x diffraction shifts at each real space pixel.
                        If None, no shifting is performed.
        yshifts         (2D array, or None) the y diffraction shifts at each real space pixel.
                        If None, no shifting is performed.

    Returns:
        class_DP        (2D array) the average diffraction pattern for the class
    """
    assert isinstance(datacube,DataCube)
    assert class_image.shape == (datacube.R_Nx,datacube.R_Ny)
    if xshifts is not None:
        assert xshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if yshifts is not None:
        assert yshifts.shape == (datacube.R_Nx,datacube.R_Ny)

    N_DPs = 0
    class_DP = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    for Rx in range(datacube.R_Nx):
        for Ry in range(datacube.R_Ny):
            if class_image[Rx,Ry] != 0:
                curr_DP = class_image[Rx,Ry]*datacube.data4D[Rx,Ry]
                if xshifts is not None and yshifts is not None:
                    xshift = xshifts[Rx,Ry]
                    yshift = yshifts[Rx,Ry]
                    curr_DP = get_shifted_ar(curr_DP,-xshift,-yshift)
                class_DP += curr_DP
                N_DPs += 1
    class_DP /= N_DPs
    return class_DP


