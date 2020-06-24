# Preprocessing functions
#
# These functions generally accept DataCube objects as arguments, and return a new, modified
# DataCube.
# Most of these functions are also included as DataCube class methods.  Thus
#       datacube = preprocess_function(datacube, *args)
# will be identical to
#       datacube.preprocess_function(*args)

import numpy as np
from ..utils import bin2D

### Editing datacube shape ###

def set_scan_shape(datacube,R_Nx,R_Ny):
    """
    Reshape the data given the real space scan shape.
    """
    try:
        datacube.data = datacube.data.reshape(datacube.R_N,datacube.Q_Nx,datacube.Q_Ny).reshape(R_Nx,R_Ny,datacube.Q_Nx,datacube.Q_Ny)
        datacube.R_Nx,datacube.R_Ny = R_Nx,R_Ny
        return datacube
    except ValueError:
        print("Can't reshape {} scan positions into a {}x{} array.".format(datacube.R_N, R_Nx, R_Ny))
        return datacube
    except AttributeError:
        print(f"Can't reshape {datacube.data.__class__.__name__} datacube.")
        return datacube

def swap_RQ(datacube):
    """
    Swaps real and reciprocal space coordinates, so that if
        datacube.data.shape = (Rx,Ry,Qx,Qy)
    Then
        swap_RQ(datacube).data.shape = (Qx,Qy,Rx,Ry)
    """
    datacube.data = np.transpose(datacube.data, axes=(2, 3, 0, 1))
    datacube.R_Nx, datacube.R_Ny, datacube.Q_Nx, datacube.Q_Ny = datacube.Q_Nx, datacube.Q_Ny, datacube.R_Nx, datacube.R_Ny
    return datacube

def swap_Rxy(datacube):
    """
    Swaps real space x and y coordinates, so that if
        datacube.data.shape = (Ry,Rx,Qx,Qy)
    Then
        swap_Rxy(datacube).data.shape = (Rx,Ry,Qx,Qy)
    """
    datacube.data = np.moveaxis(datacube.data, 1, 0)
    datacube.R_Nx, datacube.R_Ny = datacube.R_Ny, datacube.R_Nx
    return datacube

def swap_Qxy(datacube):
    """
    Swaps reciprocal space x and y coordinates, so that if
        datacube.data.shape = (Rx,Ry,Qy,Qx)
    Then
        swap_Qxy(datacube).data.shape = (Rx,Ry,Qx,Qy)
    """
    datacube.data = np.moveaxis(datacube.data, 3, 2)
    datacube.Q_Nx, datacube.Q_Ny = datacube.Q_Ny, datacube.Q_Nx
    return datacube


### Cropping and binning ###

def crop_data_diffraction(datacube,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max):
    datacube.data = datacube.data[:,:,crop_Qx_min:crop_Qx_max,crop_Qy_min:crop_Qy_max]
    datacube.Q_Nx, datacube.Q_Ny = crop_Qx_max-crop_Qx_min, crop_Qy_max-crop_Qy_min
    return datacube

def crop_data_real(datacube,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max):
    datacube.data = datacube.data[crop_Rx_min:crop_Rx_max,crop_Ry_min:crop_Ry_max,:,:]
    datacube.R_Nx, datacube.R_Ny = crop_Rx_max-crop_Rx_min, crop_Ry_max-crop_Ry_min
    datacube.R_N = datacube.R_Nx*datacube.R_Ny
    return datacube

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
            datacube.data = datacube.data[:,:,:,:-(Q_Ny%bin_factor)]
        elif (Q_Ny%bin_factor == 0):
            datacube.data = datacube.data[:,:,:-(Q_Nx%bin_factor),:]
        else:
            datacube.data = datacube.data[:,:,:-(Q_Nx%bin_factor),:-(Q_Ny%bin_factor)]
        datacube.data = datacube.data.reshape(R_Nx,R_Ny,int(Q_Nx/bin_factor),bin_factor,int(Q_Ny/bin_factor),bin_factor).sum(axis=(3,5))
        datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.data.shape
        return datacube

def bin_data_mmap(datacube, bin_factor):
    """
    Performs diffraction space binning of data by bin_factor.

    Note that this function casts data
    """
    assert type(bin_factor) is int, "Error: binning factor {} is not an int.".format(bin_factor)
    R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny

    data = np.zeros((datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx//bin_factor,datacube.Q_Ny//bin_factor),dtype=np.float64)
    for Rx in range(datacube.R_Nx):
        for Ry in range(datacube.R_Ny):
            data[Rx,Ry,:,:] = bin2D(datacube.data[Rx,Ry,:,:],bin_factor,dtype=np.float64)

    datacube.data = data
    datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.data.shape
    return datacube

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
            datacube.data = datacube.data[:,:-(R_Ny%bin_factor),:,:]
        elif (R_Ny%bin_factor == 0):
            datacube.data = datacube.data[:-(R_Nx%bin_factor),:,:,:]
        else:
            datacube.data = datacube.data[:-(R_Nx%bin_factor),:-(R_Ny%bin_factor),:,:]
        datacube.data = datacube.data.reshape(int(R_Nx/bin_factor),bin_factor,int(R_Ny/bin_factor),bin_factor,Q_Nx,Q_Ny).sum(axis=(1,3))
        datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny = datacube.data.shape
        return datacube



