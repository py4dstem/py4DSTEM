import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np



def make_Fourier_coords2D(Nx, Ny, pixelSize=1):
    """
    Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if hasattr(pixelSize, "__len__"):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_x = pixelSize[0]
        pixelSize_y = pixelSize[1]
    else:
        pixelSize_x = pixelSize
        pixelSize_y = pixelSize

    qx = np.fft.fftfreq(Nx, pixelSize_x)
    qy = np.fft.fftfreq(Ny, pixelSize_y)
    qy, qx = np.meshgrid(qy, qx)
    return qx, qy




def get_qx_qy_1d(M, dx=[1, 1], fft_shifted=False):
    """
    Generates 1D Fourier coordinates for a (Nx,Ny)-shaped 2D array.
    Specifying the dx argument sets a unit size.

    Args:
        M: (2,) shape of the returned array
        dx: (2,) tuple, pixel size
        fft_shifted: True if result should be fft_shifted to have the origin in the center of the array
    """
    qxa = np.fft.fftfreq(M[0], dx[0])
    qya = np.fft.fftfreq(M[1], dx[1])
    if fft_shifted:
        qxa = np.fft.fftshift(qxa)
        qya = np.fft.fftshift(qya)
    return qxa, qya



