import numpy as np
from math import *
import math as m
import cmath as cm
from numba import cuda
import torch as th
import torch.nn as nn
import numpy as np
from PIL import Image
import sigpy as sp
from py4DSTEM.process.utils import Param

def cartesian_aberrations(qx, qy, lam, C):
    """
    Aberrations defined with dimensionless cartesian coordinates

    Args:
        qx: array_like, 2d qx vector
        qy: array_like, 2d qy vector
        lam: wavelength
        C: aberration coefficients

    Returns:
        the aberration surface chi
    """

    u = qx * lam
    v = qy * lam
    u2 = u ** 2
    u3 = u ** 3
    u4 = u ** 4

    v2 = v ** 2
    v3 = v ** 3
    v4 = v ** 4

    aberr = Param()
    aberr.C1 = C[0]
    aberr.C12a = C[1]
    aberr.C12b = C[2]
    aberr.C21a = C[3]
    aberr.C21b = C[4]
    aberr.C23a = C[5]
    aberr.C23b = C[6]
    aberr.C3 = C[7]
    aberr.C32a = C[8]
    aberr.C32b = C[9]
    aberr.C34a = C[10]
    aberr.C34b = C[11]

    chi = 1 / 2 * aberr.C1 * (u2 + v2)
    + 1 / 2 * (aberr.C12a * (u2 - v2) + 2 * aberr.C12b * u * v)
    + 1 / 3 * (aberr.C23a * (u3 - 3 * u * v2) + aberr.C23b * (3 * u2 * v - v3))
    + 1 / 3 * (aberr.C21a * (u3 + u * v2) + aberr.C21b * (v3 + u2 * v))
    + 1 / 4 * aberr.C3 * (u4 + v4 + 2 * u2 * v2)
    + 1 / 4 * aberr.C34a * (u4 - 6 * u2 * v2 + v4)
    + 1 / 4 * aberr.C34b * (4 * u3 * v - 4 * u * v3)
    + 1 / 4 * aberr.C32a * (u4 - v4)
    + 1 / 4 * aberr.C32b * (2 * u3 * v + 2 * u * v3)

    chi *= 2 * np.pi / lam

    return chi

def aperture3(qx, qy, lam, alpha_max):
    """
       Return a boolean where (qx,qy) is within a sharp aperture given wavelength and convergence semi-angle

        Args:
            qx (float): Qx components
            qy (float, float, 1D): Qy components
            lam (float): wavelength in angstrom
            alpha_max (float): convergence semi-angle in rad

        Returns:
            bool, True if sqrt(qx**2 + qy**2) * lam < alpha_max
    """

    xp = sp.get_array_module(qx)
    qx2 = qx ** 2
    qy2 = qy ** 2
    q = xp.sqrt(qx2 + qy2)
    ktheta = xp.arcsin(q * lam)
    return ktheta < alpha_max


def aperture_xp(qx, qy, lam, alpha_max, edge=2):
    """
           Return a boolean where (qx,qy) is within a sharp aperture given wavelength and convergence semi-angle

            Args:
                qx (float): Qx components
                qy (float, float, 1D): Qy components
                lam (float): wavelength in angstrom
                alpha_max (float): convergence semi-angle in rad

            Returns:
                bool, True if sqrt(qx**2 + qy**2) * lam < alpha_max
    """

    xp = sp.get_array_module(qx)
    q = xp.sqrt(qx ** 2 + qy ** 2)
    ktheta = xp.arcsin(q * lam)
    qmax = alpha_max / lam
    dk = qx[0][1]

    arr = xp.zeros_like(qx)
    arr[ktheta < alpha_max] = 1
    # riplot(arr,'arr')
    if edge > 0:
        dEdge = edge / (qmax / dk);  # fraction of aperture radius that will be smoothed
        # some fancy indexing: pull out array elements that are within
        #    our smoothing edges
        ind = (ktheta / alpha_max > (1 - dEdge)) * (ktheta / alpha_max < (1 + dEdge))
        arr[ind] = 0.5 * (1 - xp.sin(np.pi / (2 * dEdge) * (ktheta[ind] / alpha_max - 1)))
    return arr

def fftshift_checkerboard(w, h):
    """
       Return a checkerboard array of size (w*2,h*2), where the values +1 and -1 are alternating.
       This can be used to perform an fftshift operation in-place, when used in conjunction with a Fourier transform.

        Args:
            w (int): width of the array / 2
            h (int): height of the array / 2

        Returns:
            checkerboard array of size (w*2,h*2)
    """

    re = np.r_[w * [-1, 1]]  # even-numbered rows
    ro = np.r_[w * [1, -1]]  # odd-numbered rows
    return np.row_stack(h * (re, ro))

def fourier_coordinates_2D(N, dx=[1.0, 1.0], centered=True):
    """
        Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    qxx = np.fft.fftfreq(N[1], dx[1])
    qyy = np.fft.fftfreq(N[0], dx[0])
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qx, qy = np.meshgrid(qxx, qyy)
    q = np.array([qy, qx]).astype(np.float32)
    return q

def cartesian_aberrations_single(qx, qy, lam, C):
    """
       Zernike polynomials in the cartesian coordinate system

        Args:
            qx (array, 2D float): 2D array of qx values
            qy (array, 2D float): 2D array of qy values
            lam (int): wavelength in Angstrom
            C (array 1D, float): shape (12,)
        Returns:
            aberration surface chi on the grid given by (qx,qy)
    """

    u = qx * lam
    v = qy * lam
    u2 = u ** 2
    u3 = u ** 3
    u4 = u ** 4
    # u5 = u ** 5

    v2 = v ** 2
    v3 = v ** 3
    v4 = v ** 4
    # v5 = v ** 5

    chi = 0

    # r-2 = x-2 +y-2.
    chi += 1 / 2 * C[0] * (u2 + v2) # r^2
    #r-2 cos(2*phi) = x"2 -y-2.
    # r-2 sin(2*phi) = 2*x*y.
    chi += 1 / 2 * (C[1] * (u2 - v2) + 2 * C[2] * u * v) # r^2 cos(2 phi) + r^2 sin(2 phi)
    # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
    chi += 1 / 3 * (C[5] * (u3 - 3 * u * v2) + C[6] * (3 * u2 * v - v3))# r^3 cos(3phi) + r^3 sin(3 phi)
    # r-3 cos(phi) = x-3 +x*y-2.
    # r-3 sin(phi) = y*x-2 +y-3.
    chi += 1 / 3 * (C[3] * (u3 + u * v2) + C[4] * (v3 + u2 * v))# r^3 cos(phi) + r^3 sin(phi)
    # r-4 = x-4 +2*x-2*y-2 +y-4.
    chi += 1 / 4 * C[7] * (u4 + v4 + 2 * u2 * v2)# r^4
    # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
    chi += 1 / 4 * C[10] * (u4 - 6 * u2 * v2 + v4)# r^4 cos(4 phi)
    # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
    chi += 1 / 4 * C[11] * (4 * u3 * v - 4 * u * v3) # r^4 sin(4 phi)
    # r-4 cos(2*phi) = x-4 -y-4.
    chi += 1 / 4 * C[8] * (u4 - v4)
    # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
    chi += 1 / 4 * C[9] * (2 * u3 * v + 2 * u * v3)
    # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
    # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
    # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
    # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
    # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
    # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

    chi *= 2 * np.pi / lam

    return chi


@cuda.jit
def single_sideband_kernel_cartesian(G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha,
                                     Psi_Qp, Psi_Qp_left_sb, Psi_Qp_right_sb, eps, lam, scale):
    """
    Kernel to perform single-sideband reconstruction

    Args:
        G: array_like , 4D G-function used in ptychography
        strides: array_like shape (4,) contains the strides of the G array
        Qx_all: array_like, shape (G.shape[1],)
        Qy_all: array_like, shape (G.shape[0],)
        Kx_all: array_like, shape (G.shape[3],)
        Ky_all: array_like, shape (G.shape[2],)
        aberrations: array_like, shape (16,)
        aberration_angles: array_like, shape (12,)
        theta_rot: float, STEM rotation angle in radians
        alpha: float, converge half-angle in radians
        Psi_Qp: array_like, shape (G.shape[0], G.shape[1],) to store the result of the weak phase approximation
        Psi_Qp_left_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the left sideband
        Psi_Qp_right_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the right sideband
        eps: float, threshold for which Γ-function values to include in the summation Γ_abs > eps
        lam: float, wavelength in Angstrom
        scale: float, scale factor to scale the aperture to the measured intensity
    """

    def aperture2(qx, qy, lam, alpha_max, scale):
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = m.sqrt(qx2 + qy2)
        ktheta = m.asin(q * lam)
        return (ktheta < alpha_max) * scale

    def chi3(qy, qx, lam, C):
        """
        Zernike polynomials in the cartesian coordinate system

        Args:
            qx (array, 2D float): 2D array of qx values
            qy (array, 2D float): 2D array of qy values
            lam (int): wavelength in Angstrom
            C (array 1D, float): shape (12,)

        Returns:
            aberration surface chi on the grid given by (qx,qy)
        """

        u = qx * lam
        v = qy * lam
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        # u5 = u ** 5

        v2 = v ** 2
        v3 = v ** 3
        v4 = v ** 4
        # v5 = v ** 5

        #C1 = C[0]
        #C12a = C[1]
        #C12b = C[2]
        #C21a = C[3]
        #C21b = C[4]
        #C23a = C[5]
        #C23b = C[6]
        #C3 = C[7]
        #C32a = C[8]
        #C32b = C[9]
        #C34a = C[10]
        #C34b = C[11]

        chi = 0

        # r-2 = x-2 +y-2.
        chi += 1 / 2 * C[0] * (u2 + v2)  # r^2
        # r-2 cos(2*phi) = x"2 -y-2.
        # r-2 sin(2*phi) = 2*x*y.
        chi += 1 / 2 * (C[1] * (u2 - v2) + 2 * C[2] * u * v)  # r^2 cos(2 phi) + r^2 sin(2 phi)
        # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
        chi += 1 / 3 * (C[5] * (u3 - 3 * u * v2) + C[6] * (3 * u2 * v - v3))  # r^3 cos(3phi) + r^3 sin(3 phi)
        # r-3 cos(phi) = x-3 +x*y-2.
        # r-3 sin(phi) = y*x-2 +y-3.
        chi += 1 / 3 * (C[3] * (u3 + u * v2) + C[4] * (v3 + u2 * v))  # r^3 cos(phi) + r^3 sin(phi)
        # r-4 = x-4 +2*x-2*y-2 +y-4.
        chi += 1 / 4 * C[7] * (u4 + v4 + 2 * u2 * v2)  # r^4
        # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
        chi += 1 / 4 * C[10] * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
        # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
        chi += 1 / 4 * C[11] * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
        # r-4 cos(2*phi) = x-4 -y-4.
        chi += 1 / 4 * C[8] * (u4 - v4)
        # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
        chi += 1 / 4 * C[9] * (2 * u3 * v + 2 * u * v3)
        # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
        # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
        # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
        # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
        # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
        # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

        chi *= 2 * np.pi / lam

        return chi

    gs = G.shape
    N = gs[0] * gs[1] * gs[2] * gs[3]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iqy = n // strides[0]
    iqx = (n - iqy * strides[0]) // strides[1]
    iky = (n - (iqy * strides[0] + iqx * strides[1])) // strides[2]
    ikx = (n - (iqy * strides[0] + iqx * strides[1] + iky * strides[2])) // strides[3]

    if n < N:

        Qx = Qx_all[iqx]
        Qy = Qy_all[iqy]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * m.cos(theta_rot) - Qy * m.sin(theta_rot)
        Qy_rot = Qx * m.sin(theta_rot) + Qy * m.cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        A = aperture2(Ky, Kx, lam, alpha, scale) * cm.exp(-1j * chi3(Ky, Kx, lam, aberrations))
        chi_KplusQ = chi3(Ky + Qy, Kx + Qx, lam, aberrations)
        A_KplusQ = aperture2(Ky + Qy, Kx + Qx, lam, alpha, scale) * cm.exp(-1j * chi_KplusQ)
        chi_KminusQ = chi3(Ky - Qy, Kx - Qx, lam, aberrations)
        A_KminusQ = aperture2(Ky - Qy, Kx - Qx, lam, alpha, scale) * cm.exp(-1j * chi_KminusQ)

        Gamma = A.conjugate() * A_KminusQ - A * A_KplusQ.conjugate()

        Kplus = sqrt((Kx + Qx) ** 2 + (Ky + Qy) ** 2)
        Kminus = sqrt((Kx - Qx) ** 2 + (Ky - Qy) ** 2)
        K = sqrt(Kx ** 2 + Ky ** 2)
        bright_field = K < alpha / lam
        double_overlap1 = (Kplus < alpha / lam) * bright_field * (Kminus > alpha / lam)
        double_overlap2 = (Kplus > alpha / lam) * bright_field * (Kminus < alpha / lam)

        Gamma_abs = abs(Gamma)
        take = Gamma_abs > eps and bright_field
        if take:
            val = G[iqy, iqx, iky, ikx] * Gamma.conjugate()
            cuda.atomic.add(Psi_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Psi_Qp.imag, (iqy, iqx), val.imag)
        if double_overlap1:
            val = G[iqy, iqx, iky, ikx] * Gamma.conjugate()
            cuda.atomic.add(Psi_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Psi_Qp_left_sb.imag, (iqy, iqx), val.imag)
        if double_overlap2:
            val = G[iqy, iqx, iky, ikx] * Gamma.conjugate()
            cuda.atomic.add(Psi_Qp_right_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Psi_Qp_right_sb.imag, (iqy, iqx), val.imag)
        if iqx == 0 and iqy == 0:
            val = abs(G[iqy, iqx, iky, ikx]) + 1j * 0
            cuda.atomic.add(Psi_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Psi_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Psi_Qp_right_sb.real, (iqy, iqx), val.real)
