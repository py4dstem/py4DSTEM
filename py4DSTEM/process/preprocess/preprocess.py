# Preprocessing functions
#
# These functions generally accept DataCube objects as arguments, and return a new, modified
# DataCube.
# Most of these functions are also included as DataCube class methods.  Thus
#       datacube = preprocess_function(datacube, *args)
# will be identical to
#       datacube.preprocess_function(*args)

import numpy as np

def set_scan_shape(datacube,R_Ny,R_Nx):
    """
    Reshape the data given the real space scan shape.
    """
    try:
        datacube.data4D = datacube.data4D.reshape(datacube.R_N,datacube.Q_Ny,datacube.Q_Nx).reshape(R_Ny,R_Nx,datacube.Q_Ny,datacube.Q_Nx)
        datacube.R_Ny,datacube.R_Nx = R_Ny,R_Nx
        return datacube
    except ValueError:
        return datacube

def crop_data_diffraction(datacube,crop_Qy_min,crop_Qy_max,crop_Qx_min,crop_Qx_max):
    datacube.data4D = datacube.data4D[:,:,crop_Qy_min:crop_Qy_max,crop_Qx_min:crop_Qx_max]
    datacube.Q_Ny, datacube.Q_Nx = crop_Qy_max-crop_Qy_min, crop_Qx_max-crop_Qx_min
    return datacube

def crop_data_real(datacube,crop_Ry_min,crop_Ry_max,crop_Rx_min,crop_Rx_max):
    datacube.data4D = datacube.data4D[crop_Ry_min:crop_Ry_max,crop_Rx_min:crop_Rx_max,:,:]
    datacube.R_Ny, datacube.R_Nx = crop_Ry_max-crop_Ry_min, crop_Rx_max-crop_Rx_min
    datacube.R_N = datacube.R_Ny*datacube.R_Nx
    return datacube


