import numpy as np


def get_ewpc_filter_function(Q_Nx, Q_Ny):
    """
    Returns a function for computing the exit wave power cepstrum of a diffraction
    pattern using a Hanning window. This can be passed as the filter_function in the
    Bragg disk detection functions (with the probe an array of ones) to find the lattice
    vectors by the EWPC method (but be careful as the lengths are now in realspace
    units!) See https://arxiv.org/abs/1911.00984
    """
    h = np.hanning(Q_Nx)[:, np.newaxis] * np.hanning(Q_Ny)[np.newaxis, :]
    return (
        lambda x: np.abs(np.fft.fftshift(np.fft.fft2(h * np.log(np.maximum(x, 0.01)))))
        ** 2
    )


