# Preprocessing functions
#
# These functions generally accept DataCube objects as arguments, and return a new, modified
# DataCube.
# Most of these functions are also included as DataCube class methods.  Thus
#       datacube = preprocess_function(datacube, *args)
# will be identical to
#       datacube.preprocess_function(*args)

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

def bin_data_diffraction(datacube, bin_factor):
    """
    Performs diffraction space binning of data by bin_factor.
    """
    if bin_factor <= 1:
        return datacube
    else:
        assert type(bin_factor) is int, "Error: binning factor {} is not an int.".format(bin_factor)
        R_Ny,R_Nx,Q_Ny,Q_Nx = datacube.R_Ny,datacube.R_Nx,datacube.Q_Ny,datacube.Q_Nx
        # Crop to make data binnable if necessary
        if ((Q_Ny%bin_factor == 0) and (Q_Nx%bin_factor == 0)):
            pass
        elif (Q_Nx%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:,:-(Q_Ny%bin_factor),:]
        elif (Q_Ny%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:,:,:-(Q_Nx%bin_factor)]
        else:
            datacube.data4D = datacube.data4D[:,:,:-(Q_Ny%bin_factor),:-(Q_Nx%bin_factor)]
        datacube.data4D = datacube.data4D.reshape(R_Ny,R_Nx,int(Q_Ny/bin_factor),bin_factor,int(Q_Nx/bin_factor),bin_factor).sum(axis=(3,5))
        datacube.R_Ny,datacube.R_Nx,datacube.Q_Ny,datacube.Q_Nx = datacube.data4D.shape
        return datacube

def bin_data_real(datacube, bin_factor):
    """
    Performs diffraction space binning of data by bin_factor.
    """
    if bin_factor <= 1:
        return datacube
    else:
        assert type(bin_factor) is int, "Error: binning factor {} is not an int.".format(bin_factor)
        R_Ny,R_Nx,Q_Ny,Q_Nx = datacube.R_Ny,datacube.R_Nx,datacube.Q_Ny,datacube.Q_Nx
        # Crop to make data binnable if necessary
        if ((R_Ny%bin_factor == 0) and (R_Nx%bin_factor == 0)):
            pass
        elif (R_Nx%bin_factor == 0):
            datacube.data4D = datacube.data4D[:-(R_Ny%bin_factor),:,:,:]
        elif (R_Ny%bin_factor == 0):
            datacube.data4D = datacube.data4D[:,:-(R_Nx%bin_factor),:,:]
        else:
            datacube.data4D = datacube.data4D[:-(R_Ny%bin_factor),:-(R_Nx%bin_factor),:,:]
        datacube.data4D = datacube.data4D.reshape(int(R_Ny/bin_factor),bin_factor,int(R_Nx/bin_factor),bin_factor,Q_Ny,Q_Nx).sum(axis=(1,3))
        datacube.R_Ny,datacube.R_Nx,datacube.Q_Ny,datacube.Q_Nx = datacube.data4D.shape
        return datacube



