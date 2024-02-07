import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np


def get_CoM(ar, device="cpu", corner_centered=False):
    """
    Finds and returns the center of mass of array ar.
    If corner_centered is True, uses fftfreq for indices.
    """
    if device == "cpu":
        xp = np
    elif device == "gpu":
        xp = cp

    ar = xp.asarray(ar)
    nx, ny = ar.shape

    if corner_centered:
        ry, rx = xp.meshgrid(xp.fft.fftfreq(ny, 1 / ny), xp.fft.fftfreq(nx, 1 / nx))
    else:
        ry, rx = xp.meshgrid(xp.arange(ny), xp.arange(nx))

    tot_intens = xp.sum(ar)
    xCoM = xp.sum(rx * ar) / tot_intens
    yCoM = xp.sum(ry * ar) / tot_intens
    return xCoM, yCoM


