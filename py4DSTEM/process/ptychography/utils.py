import numpy as np
from math import *
import math as m
import cmath as cm
from numba import cuda

@cuda.jit
def single_sideband_kernel(G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations,
                           aberration_angles, theta_rot, alpha, Ψ_Qp, Ψ_Qp_left_sb,
                           Ψ_Qp_right_sb, eps, lam, scale):
    """
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
        Ψ_Qp: array_like, shape (G.shape[0], G.shape[1],) to store the result of the weak phase approximation
        Ψ_Qp_left_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the left sideband
        Ψ_Qp_right_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the right sideband
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

    def chi3(qx, qy, lam, a, phi):
        """
        Creates an aberration surface from aberration coefficients.

        Args:
            qx: M1 x M2 tensor of x coefficients of reciprocal space
            qy: M1 x M2 tensor of y coefficients of reciprocal space
            lam: wavelength in Angstrom
            a10: shift                               in Angstrom
            a11: shift                               in Angstrom
            a20: defocus                             in Angstrom
            a22: two-fold Astigmatism                in Angstrom
            a31: axial coma                          in Angstrom
            a33: three-fold astigmatism              in Angstrom
            a40: spherical aberration                in um
            a42: star aberration                     in um
            a44: four-fold astigmatism               in um
            a51: fourth order axial coma             in um
            a53: three-lobe aberration               in um
            a55: five-fold astigmatism               im um
            a60: fifth-order spherical               in mm
            a62: fifth-order two-fold astigmatism    in mm
            a64: fifth-order four-fold astigmatism   in mm
            a66: siz-fold astigmatism                in mm
            phi11: shift angle
            phi22: two-fold Astigmatism angle
            phi33: three-fold astigmatism angle
            phi44: four-fold astigmatism angle
            phi55: five-fold astigmatism angle
            phi51: fourth order axial coma angle
            phi66: Chromatic aberration angle
            phi62: fifth-order two-fold astigmatism angle
            phi64: fifth-order four-fold astigmatism angle
            phi31: axial coma angle
            phi42: star aberration angle
            phi53: three-lobe aberration angle

        Returns:
            M1 x M1 x (maximum size of all aberration tensors)
        """
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = sqrt(qx2 + qy2)
        ktheta = m.asin(q * lam)

        kphi = m.atan2(qy, qx)
        # print(a0.shape, scales.shape)
        cos = m.cos
        chi = 2.0 * np.pi / lam * (1.0 * (a[1] * cos(2 * (kphi - phi[0])) + a[0]) * ktheta +
                                   1.0 / 2 * (a[3] * cos(2 * (kphi - phi[1])) + a[2]) * ktheta ** 2 +
                                   1.0 / 3 * (a[5] * cos(3 * (kphi - phi[2])) + a[4] * cos(
                    1 * (kphi - phi[3]))) * ktheta ** 3 +
                                   1.0 / 4 * (a[8] * cos(4 * (kphi - phi[4])) + a[7] * cos(2 * (kphi - phi[5])) + a[
                    6]) * ktheta ** 4 +
                                   1.0 / 5 * (a[11] * cos(5 * (kphi - phi[6])) + a[10] * cos(3 * (kphi - phi[7])) + a[
                    9] * cos(
                    1 * (kphi - phi[8]))) * ktheta ** 5 +
                                   1.0 / 6 * (a[15] * cos(6 * (kphi - phi[9])) + a[14] * cos(4 * (kphi - phi[10])) + a[
                    13] * cos(
                    2 * (kphi - phi[11])) + a[12]) * ktheta ** 6)

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

        Qx_rot = Qx * cos(theta_rot) - Qy * sin(theta_rot)
        Qy_rot = Qx * sin(theta_rot) + Qy * cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        A = aperture2(Ky, Kx, lam, alpha, scale) * cm.exp(1j * chi3(Ky, Kx, lam, aberrations, aberration_angles))
        A_KplusQ = aperture2(Ky + Qy, Kx + Qx, lam, alpha, scale) * cm.exp(1j *
                                                                           chi3(Ky + Qy, Ky + Qx, lam, aberrations,
                                                                                aberration_angles))
        A_KminusQ = aperture2(Ky - Qy, Kx - Qx, lam, alpha, scale) * cm.exp(1j *
                                                                            chi3(Ky - Qy, Kx - Qx, lam, aberrations,
                                                                                 aberration_angles))

        Γ = A.conjugate() * A_KminusQ - A * A_KplusQ.conjugate()

        Kplus = sqrt((Kx + Qx) ** 2 + (Ky + Qy) ** 2)
        Kminus = sqrt((Kx - Qx) ** 2 + (Ky - Qy) ** 2)
        K = sqrt(Kx ** 2 + Ky ** 2)
        bright_field = K < alpha / lam
        double_overlap1 = (Kplus < alpha / lam) * bright_field * (Kminus > alpha / lam)
        double_overlap2 = (Kplus > alpha / lam) * bright_field * (Kminus < alpha / lam)

        # print(Γ.real, Γ.imag)
        # Gamma[iqy, iqx, iky, ikx] = Γ
        Γ_abs = abs(Γ)
        take = Γ_abs > eps and bright_field
        if take:
            val = G[iqy, iqx, iky, ikx] * Γ.conjugate()
            cuda.atomic.add(Ψ_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp.imag, (iqy, iqx), val.imag)
        if double_overlap1:
            val = G[iqy, iqx, iky, ikx]
            cuda.atomic.add(Ψ_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_left_sb.imag, (iqy, iqx), val.imag)
        if double_overlap2:
            val = G[iqy, iqx, iky, ikx]
            cuda.atomic.add(Ψ_Qp_right_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_right_sb.imag, (iqy, iqx), val.imag)
        if iqx == 0 and iqy == 0:
            val = abs(G[iqy, iqx, iky, ikx]) + 1j * 0
            cuda.atomic.add(Ψ_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_right_sb.real, (iqy, iqx), val.real)
