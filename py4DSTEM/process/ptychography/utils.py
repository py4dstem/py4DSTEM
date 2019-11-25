import numpy as np
from math import *
import math as m
import cmath as cm
from numba import cuda

@cuda.jit
def single_sideband_kernel(G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, aberration_angles, theta_rot, alpha,
                           Ψ_Qp, Ψ_Qp_left_sb, Ψ_Qp_right_sb, eps, lam, scale):
    """

    :param G: array_like , 4D G-function used in ptychography
    :param strides: array_like shape (4,) contains the strides of the G array
    :param Qx_all: array_like, shape (G.shape[1],)
    :param Qy_all: array_like, shape (G.shape[0],)
    :param Kx_all: array_like, shape (G.shape[3],)
    :param Ky_all: array_like, shape (G.shape[2],)
    :param aberrations: array_like, shape (16,)
    :param aberration_angles: array_like, shape (12,)
    :param theta_rot: float, STEM rotation angle in radians
    :param alpha: float, converge half-angle in radians
    :param Ψ_Qp: array_like, shape (G.shape[0], G.shape[1],) to store the result of the weak phase approximation
    :param Ψ_Qp_left_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the left sideband
    :param Ψ_Qp_right_sb: array_like, shape (G.shape[0], G.shape[1],) to store the result of the right sideband
    :param eps: float, threshold for which Γ-function values to include in the summation Γ_abs > eps
    :param lam: float, wavelength in Angstrom
    :param scale: float, scale factor to scale the aperture to the measured intensity
    :return: nothing
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

        :param qx: M1 x M2 tensor of x coefficients of reciprocal space
        :param qy: M1 x M2 tensor of y coefficients of reciprocal space
        :param lam: wavelength in Angstrom
        :param a10: shift                               in Angstrom
        :param a11: shift                               in Angstrom
        :param a20: defocus                             in Angstrom
        :param a22: two-fold Astigmatism                in Angstrom
        :param a31: axial coma                          in Angstrom
        :param a33: three-fold astigmatism              in Angstrom
        :param a40: spherical aberration                in um
        :param a42: star aberration                     in um
        :param a44: four-fold astigmatism               in um
        :param a51: fourth order axial coma             in um
        :param a53: three-lobe aberration               in um
        :param a55: five-fold astigmatism               im um
        :param a60: fifth-order spherical               in mm
        :param a62: fifth-order two-fold astigmatism    in mm
        :param a64: fifth-order four-fold astigmatism   in mm
        :param a66: siz-fold astigmatism                in mm
        :param phi11: shift angle
        :param phi22: two-fold Astigmatism angle
        :param phi33: three-fold astigmatism angle
        :param phi44: four-fold astigmatism angle
        :param phi55: five-fold astigmatism angle
        :param phi51: fourth order axial coma angle
        :param phi66: Chromatic aberration angle
        :param phi62: fifth-order two-fold astigmatism angle
        :param phi64: fifth-order four-fold astigmatism angle
        :param phi31: axial coma angle
        :param phi42: star aberration angle
        :param phi53: three-lobe aberration angle
        :return: M1 x M1 x (maximum size of all aberration tensors)
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
