# Preprocessing functions
#
# These functions generally accept DataCube objects as arguments, and return a new, modified
# DataCube.
# Most of these functions are also included as DataCube class methods.  Thus
#       datacube = preprocess_function(datacube, *args)
# will be identical to
#       datacube.preprocess_function(*args)

import numpy as np
from ...file.log import log


### Editing datacube shape ###

@log
def set_scan_shape(datacube,R_Nx,R_Ny):
    """
    Reshape the data given the real space scan shape.
    """
    try:
        datacube.data4D = datacube.data4D.reshape(datacube.R_N,datacube.Q_Nx,datacube.Q_Ny).reshape(R_Nx,R_Ny,datacube.Q_Nx,datacube.Q_Ny)
        datacube.R_Nx,datacube.R_Ny = R_Nx,R_Ny
        return datacube
    except ValueError:
        print("Can't reshape {} scan positions into a {}x{} array.".format(datacube.R_N, R_Nx, R_Ny))
        return datacube

@log
def swap_RQ(datacube):
    """
    Swaps real and reciprocal space coordinates, so that if
        datacube.data4D.shape = (Rx,Ry,Qx,Qy)
    Then
        swap_RQ(datacube).data4D.shape = (Qx,Qy,Rx,Ry)
    """
    datacube.data4D = np.moveaxis(np.moveaxis(datacube.data4D,2,0),3,1)
    datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.Q_Nx,datacube.Q_Ny,datacube.R_Nx,datacube.R_Ny

@log
def swap_Rxy(datacube):
    """
    Swaps real space x and y coordinates, so that if
        datacube.data4D.shape = (Ry,Rx,Qx,Qy)
    Then
        swap_Rxy(datacube).data4D.shape = (Rx,Ry,Qx,Qy)
    """
    datacube.data4D = np.moveaxis(datacube.data4D,1,0)
    datacube.R_Nx,datacube.R_Ny = datacube.R_Ny,datacube.R_Nx

@log
def swap_Qxy(datacube):
    """
    Swaps reciprocal space x and y coordinates, so that if
        datacube.data4D.shape = (Rx,Ry,Qy,Qx)
    Then
        swap_Qxy(datacube).data4D.shape = (Rx,Ry,Qx,Qy)
    """
    datacube.data4D = np.moveaxis(datacube.data4D,3,2)
    datacube.Q_Nx,datacube.Q_Ny = datacube.Q_Ny,datacube.Q_Nx


### Cropping and binning ###

@log
def crop_data_diffraction(datacube,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max):
    datacube.data4D = datacube.data4D[:,:,crop_Qx_min:crop_Qx_max,crop_Qy_min:crop_Qy_max]
    datacube.Q_Nx, datacube.Q_Ny = crop_Qx_max-crop_Qx_min, crop_Qy_max-crop_Qy_min
    return datacube

@log
def crop_data_real(datacube,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max):
    datacube.data4D = datacube.data4D[crop_Rx_min:crop_Rx_max,crop_Ry_min:crop_Ry_max,:,:]
    datacube.R_Nx, datacube.R_Ny = crop_Rx_max-crop_Rx_min, crop_Ry_max-crop_Ry_min
    datacube.R_N = datacube.R_Nx*datacube.R_Ny
    return datacube

@log
def bin_data_diffraction(datacube, bin_factor):
    """
    Performs diffraction space binning of data by bin_factor.
    """
    if bin_factor <= 1:
        return datacube
    else:
        assert type(bin_factor) is int, "Error: binning factor {} is not an int.".format(bin_factor)
        R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny
        # Crop to make data binnable if necessary
        if ((Q_Nx%bin_factor == 0) and (Q_Ny%bin_factor == 0)):
            pass
        elif (Q_Nx%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:,:,:-(Q_Ny%bin_factor)]
        elif (Q_Ny%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:,:-(Q_Nx%bin_factor),:]
        else:
            datacube.data4D = datacube.data4D[:,:,:-(Q_Nx%bin_factor),:-(Q_Ny%bin_factor)]
        datacube.data4D = datacube.data4D.reshape(R_Nx,R_Ny,int(Q_Nx/bin_factor),bin_factor,int(Q_Ny/bin_factor),bin_factor).sum(axis=(3,5))
        datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.data4D.shape
        return datacube

@log
def bin_data_real(datacube, bin_factor):
    """
    Performs diffraction space binning of data by bin_factor.
    """
    if bin_factor <= 1:
        return datacube
    else:
        assert type(bin_factor) is int, "Error: binning factor {} is not an int.".format(bin_factor)
        R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny
        # Crop to make data binnable if necessary
        if ((R_Nx%bin_factor == 0) and (R_Ny%bin_factor == 0)):
            pass
        elif (R_Nx%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:-(R_Ny%bin_factor),:,:]
        elif (R_Ny%bin_factor == 0):
            datacube.data4D = datacube.data4D[:-(R_Nx%bin_factor),:,:,:]
        else:
            datacube.data4D = datacube.data4D[:-(R_Nx%bin_factor),:-(R_Ny%bin_factor),:,:]
        datacube.data4D = datacube.data4D.reshape(int(R_Nx/bin_factor),bin_factor,int(R_Ny/bin_factor),bin_factor,Q_Nx,Q_Ny).sum(axis=(1,3))
        datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.data4D.shape
        return datacube



