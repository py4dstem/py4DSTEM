import time

import sigpy as sp
from sigpy import config

from py4DSTEM.io import DataCube

if config.cupy_enabled:
    import cupy as cp
import torch as th
from numba import cuda
import numpy as np
from numpy.fft import fftfreq, fftshift

import torch.nn as nn

from py4DSTEM.process.ptychography.utils import cartesian_aberrations_single, fftshift_checkerboard, aperture3, \
    cartesian_aberrations, aperture_xp, single_sideband_kernel_cartesian

from py4DSTEM.process.utils import get_qx_qy_1d, electron_wavelength_angstrom

from tqdm import trange
import math as m
from math import sin, cos
import cmath as cm


class ZernikeProbeSingle(nn.Module):
    def __init__(self, q: th.Tensor, lam, fft_shifted=True):
        """
        Creates an aberration surface from aberration coefficients. The output can be used with error backpropgation for neural network training.

        Args:
            q (th.tensor): shape (2, MY, MX), contains qy and qx
            lam (float):
            fft_shifted (optional, bool): whether to apply a checkerboard to the aberration surface, so that the real-space probe is centered after ifft2. default: True
        """

        super(ZernikeProbeSingle, self).__init__()
        self.q = q
        self.lam = lam
        self.fft_shifted = fft_shifted

        if self.fft_shifted:
            cb = fftshift_checkerboard(self.q.shape[1] // 2, self.q.shape[2] // 2)
            self.cb = th.from_numpy(cb).float().to(q.device)

    def forward(self, C, A):
        """
        Given aberrations C and a condenser aperture A, returns an aberration surface on the coordinate grid given at initialization.

        Args:
            C (th.tensor, float): shape (12,) twelve cartesian aberration coefficients
            A (th.tensor, float): shape (MY, MX) array containing the condenser aperture, in fftshifted order (zero on the top left)
        Returns:
            (2D th.tensor, shape (MY, MX), complex) the probe in the aperture plane
        """
        chi = cartesian_aberrations_single(self.q[1], self.q[0], self.lam, C)
        Psi = th.exp(-1j * chi) * A.expand_as(chi)

        if self.fft_shifted:
            Psi = Psi * self.cb

        return Psi


def disk_overlap_function(Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
    """
    Create the disk-overlap function Gamma, given detector K-space coordinates (Kx_all, Ky_all),
    scan-space reciprocal coordinates (Qx_all, Qy_all),
    aberrations, STEM rotation theta_rot, beam convergence semi-angle alpha, and electron wavelength lam in Angstrom

    Args:
        Qx_all (th.tensor, float, 1D): Qx components
        Qy_all (th.tensor, float, 1D): Qy components
        Kx_all (th.tensor, float, 1D): Kx components
        Ky_all (th.tensor, float, 1D): Ky components
        aberrations (th.tensor, float, 1D): (12,) aberration coefficients
        aberrations (float): STEM rotation angle in rad
        alpha (float): convergence semi-angle in rad
        lam (float): wavelength in angstrom

    Returns:
        (2D th.tensor, shape (MY, MX), complex) the probe in the aperture plane
    """

    n_batch = Qx_all.shape[0]
    xp = sp.backend.get_array_module(aberrations)
    Gamma = xp.zeros((n_batch,) + (Ky_all.shape[0], Kx_all.shape[0]), dtype=xp.complex64)
    gs = Gamma.shape
    threadsperblock = 2 ** 8
    blockspergrid = m.ceil(np.prod(gs) / threadsperblock)
    strides = xp.array((np.array(Gamma.strides) / (Gamma.nbytes / Gamma.size)).astype(np.int))
    disk_overlap_kernel[blockspergrid, threadsperblock](Gamma, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations,
                                                        theta_rot, alpha, lam)
    return Gamma


@cuda.jit
def disk_overlap_kernel(gamma, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
    """
    Kernel to create the disk-overlap function gamma, given detector K-space coordinates (Kx_all, Ky_all),
    scan-space reciprocal coordinates (Qx_all, Qy_all),
    aberrations, STEM rotation theta_rot, beam convergence semi-angle alpha, and electron wavelength lam in Angstrom

    Args:
        gamma (th.tensor, float, 4D): 4D tensor of disk overlap functions
        strides (th.tensor, float, 1D): strides of gamma
        Qx_all (th.tensor, float, 1D): Qx components
        Qy_all (th.tensor, float, 1D): Qy components
        Kx_all (th.tensor, float, 1D): Kx components
        Ky_all (th.tensor, float, 1D): Ky components
        aberrations (th.tensor, float, 1D): (12,) aberration coefficients
        theta_rot (float): STEM rotation angle in rad
        alpha (float): convergence semi-angle in rad
        lam (float): wavelength in angstrom

    Returns:
        no returns, results is in gamma
    """

    def aperture(qx, qy, lam, alpha_max):
        """
       Return a boolean whether (qx,qy) is within a sharp aperture given wavelength and convergence semi-angle

        Args:
            qx (float): Qx components
            qy (float, float, 1D): Qy components
            lam (float): wavelength in angstrom
            alpha_max (float): convergence semi-angle in rad

        Returns:
            bool, True if sqrt(qx**2 + qy**2) * lam < alpha_max
        """
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = m.sqrt(qx2 + qy2)
        ktheta = m.asin(q * lam)
        return ktheta < alpha_max

    def chi3(qy, qx, lam, C):
        """
        Return the phase of the aberration function chi at (qx,qy), given aberrations C

        Args:
            qx (float): Qx components
            qy (float, float, 1D): Qy components
            lam (float): wavelength in angstrom
            C (array, 1D): shape (12,) convergence semi-angle in rad

        Returns:
            phase of the aberration function at (qx,qy)
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

    gs = gamma.shape
    N = gs[0] * gs[1] * gs[2]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = n // strides[0]
    iky = (n - j * strides[0]) // strides[1]
    ikx = (n - (j * strides[0] + iky * strides[1])) // strides[2]

    if n < N:
        Qx = Qx_all[j]
        Qy = Qy_all[j]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * cos(theta_rot) - Qy * sin(theta_rot)
        Qy_rot = Qx * sin(theta_rot) + Qy * cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        chi = chi3(Ky, Kx, lam, aberrations)
        A = aperture(Ky, Kx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky + Qy, Kx + Qx, lam, aberrations)
        Ap = aperture(Ky + Qy, Kx + Qx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky - Qy, Kx - Qx, lam, aberrations)
        Am = aperture(Ky - Qy, Kx - Qx, lam, alpha) * cm.exp(-1j * chi)

        gamma[j, iky, ikx] = A.conjugate() * Am - A * Ap.conjugate()


def double_overlap_intensitities_in_range(G_max, thetas, Qx_max, Qy_max, Kx, Ky, aberrations, alpha_rad, lam):
    """
    Calculates the summed intensities of the double disk overlaps for an array of STEM rotation angles given in thetas.
    This function is useful for finding the STEM rotation from a finely scanned ptychography dataset.

    Args:
        G_max (th.tensor, float, 4D): 4D tensor of disk overlap functions
        thetas (th.tensor, float, 1D): STEM rotation angles in rad
        Qx_max (th.tensor, float, 1D): Qx components
        Qy_max (th.tensor, float, 1D): Qy components
        Kx (th.tensor, float, 1D): Kx components
        Ky (th.tensor, float, 1D): Ky components
        aberrations (th.tensor, float, 1D): (12,) aberration coefficients
        alpha_rad (float): convergence semi-angle in rad
        lam (float): wavelength in angstrom

    Returns:
        (array, 1D) sum over double overlap intensities at STEM rotations given in thetas
    """

    xp = sp.backend.get_array_module(G_max)
    intensities = np.zeros((len(thetas)))
    for i, theta_rot in enumerate(thetas):
        if th.cuda.is_available():
            Gamma = disk_overlap_function(Qx_max, Qy_max, Kx, Ky, aberrations, theta_rot, alpha_rad, lam)

        intensities[i] = xp.sum(xp.abs(G_max * Gamma.conj()))

    return intensities


def find_rotation_angle_with_double_disk_overlap(G, lam, dx, dscan, alpha_rad, mask=None, n_fit=6, ranges=[360, 30],
                                                 partitions=[144, 120], verbose=False, manual_frequencies=None,
                                                 aberrations=None):
    """
    Finds the best rotation angle by maximizing the double disk overlap intensity of the 4D dataset. Only valid
    for datasets where the scan step size is roughly on the same length scale as the illumination half-angle alp

    Args:
        G (th.tensor, float, 4D): (NY, NX, MY, MX) 4D tensor of disk overlap functions
        lam (float): wavelength in Angstrom
        dx (float, float): 1/(2 * k_max) real_space sampling determined from maximum sampled detector angle, in angstrom
        dscan (float, float): real-space sampling of the scan, in angstrom
        alpha_rad (float): convergence semi-angle in rad
        mask (th.tensor, float, 2D): (NY, NX) mask to apply to G
        n_fit (int): number of "trotters" to use for summation
        ranges (list): list of angle ranges in degrees to try and rotate the disk overlap function to, default [360, 30]
        partitions (list): list of numbers of partitions the range of angles should be split into, default [144,120]
        verbose (bool): optional, talk to me or not
        manual_frequencies (list of 2-tuples): optional, indices into (NY, NX) that pick out spatial frequencies at which the G-function has bragg-peaks/maxima
        aberrations (th.tensor, float, 1D): (12,) aberration coefficients

    Returns:
        tuple (max_ind, thetas, intensities) max_ind: index into thetas and intensities that gives the maximum intensity in the double overlap sum --> the best STEM rotation angle
    """
    ny, nx, nky, nkx = G.shape
    xp = sp.backend.get_array_module(G)

    Kx, Ky = get_qx_qy_1d([nkx, nky], dx, fft_shifted=True)
    Qx, Qy = get_qx_qy_1d([nx, ny], dscan, fft_shifted=False)

    Kx = xp.array(Kx, dtype=G[0, 0, 0, 0].real.dtype)
    Ky = xp.array(Ky, dtype=G[0, 0, 0, 0].real.dtype)
    Qx = xp.array(Qx, dtype=G[0, 0, 0, 0].real.dtype)
    Qy = xp.array(Qy, dtype=G[0, 0, 0, 0].real.dtype)

    if aberrations is None:
        aberrations = xp.zeros((12))

    if manual_frequencies is None:
        Gabs = xp.sum(xp.abs(G), (2, 3))
        if mask is not None:
            gg = Gabs * mask
            inds = xp.argsort((gg).ravel()).get()
        else:
            inds = xp.argsort(Gabs.ravel()).get()
        strongest_object_frequencies = np.unravel_index(inds[-1 - n_fit:-1], G.shape[:2])

        G_max = G[strongest_object_frequencies]
        Qy_max = Qy[strongest_object_frequencies[0]]
        Qx_max = Qx[strongest_object_frequencies[1]]
    else:
        strongest_object_frequencies = manual_frequencies
        G_max = G[strongest_object_frequencies]
        Qy_max = Qy[strongest_object_frequencies[0]]
        Qx_max = Qx[strongest_object_frequencies[1]]

    if verbose:
        print(f"strongest_object_frequencies: {strongest_object_frequencies}")

    best_angle = 0

    for j, (range, parts) in enumerate(zip(ranges, partitions)):
        thetas = np.linspace(best_angle - np.deg2rad(range / 2), best_angle + np.deg2rad(range / 2), parts)
        intensities = double_overlap_intensitities_in_range(G_max, thetas, Qx_max, Qy_max, Kx, Ky, aberrations,
                                                            alpha_rad, lam)

        sortind = np.argsort(intensities)
        max_ind0 = sortind[-1]
        max_ind1 = sortind[0]
        best_angle = thetas[max_ind0]

    max_ind = np.argsort(intensities)[-1]

    return max_ind, thetas, intensities


def weak_phase_reconstruction(dc: DataCube, aberrations=None, verbose=False, use_cuda=True):
    """
    Perform a ptychographic reconstruction of the datacube assuming a weak phase object.
    In the weak phase object approximation, the dataset in double Fourier-space
    coordinates can be described as [1]::

        G(r',\rho') = |A(r')|^2 \delta(\rho') + A(r')A*(r'+\rho')Ψ*(-\rho')+ A*(r')A(r'-\rho')Ψ(\rho')

    We solve this equation for Ψ*(\rho') in two different ways:

    1) collect all the signal in the bright-field by multiplying G with::

        A(r')A*(r'+\rho')+ A*(r')A(r'-\rho')[2]

    2) collect only the signal in the double-overlap region [1]

    References:
        * [1] Rodenburg, J. M., McCallum, B. C. & Nellist, P. D. Experimental tests on
          double-resolution coherent imaging via STEM. Ultramicroscopy 48, 304–314 (1993).
        * [2] Yang, H., Ercius, P., Nellist, P. D. & Ophus, C. Enhanced phase contrast
          transfer using ptychography combined with a pre-specimen phase plate in a
          scanning transmission electron microscope. Ultramicroscopy 171, 117–125 (2016).

    Args:
        dc: py4DSTEM datacube
        aberrations: optional array shape (12,), cartesian aberration coefficients
        verbose: optional bool, default: False
        use_cuda: optional bool, default: True

    Returns:
        (Psi_Rp, Psi_Rp_left_sb, Psi_Rp_right_sb)
        Psi_Rp is the result of method 1) and Psi_Rp_left_sb, Psi_Rp_right_sb are the results
        of method 2)
    """

    assert 'beam_energy' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: beam_energy'
    assert 'convergence_semiangle_mrad' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: convergence_semiangle_mrad'

    assert 'Q_pixel_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: Q_pixel_size'
    assert 'R_pixel_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: R_pixel_size'
    assert 'QR_rotation' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: QR_rotation'
    assert 'QR_rotation_units' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: QR_rotation_units'

    M = dc.data

    ny, nx, nky, nkx = M.shape

    E = dc.metadata.microscope['beam_energy']
    alpha_rad = dc.metadata.microscope['convergence_semiangle_mrad'] * 1e-3
    lam = electron_wavelength_angstrom(E)
    eps = 1e-3
    k_max = dc.metadata.calibration['Q_pixel_size']
    dxy = dc.metadata.calibration['R_pixel_size']
    theta = dc.metadata.calibration['QR_rotation']
    if dc.metadata.calibration['QR_rotation_units'] == 'deg':
        theta = np.deg2rad(theta)

    cuda_is_available = config.cupy_enabled

    if verbose:
        print(f"E               = {E}             eV")
        print(f"λ               = {lam * 1e2:2.2}   pm")
        print(f"dR              = {dxy}             Å")
        print(f"dK              = {k_max}           Å")
        print(f"scan       size = {[ny, nx]}")
        print(f"detector   size = {[nky, nkx]}")

    if cuda_is_available:
        M = cp.array(M, dtype=M.dtype)

    xp = sp.get_array_module(M)

    Kx, Ky = get_qx_qy_1d([nkx, nky], k_max, fft_shifted=True)
    Qx, Qy = get_qx_qy_1d([nx, ny], dxy, fft_shifted=False)

    Kx = Kx.astype(M.dtype)
    Ky = Ky.astype(M.dtype)
    Qx = Qx.astype(M.dtype)
    Qy = Qy.astype(M.dtype)

    ap = aperture3(Kx, Ky, lam, alpha_rad).astype(xp.float32)
    scale = 1  # math.sqrt(mean_intensity / aperture_intensity)
    ap *= scale

    start = time.perf_counter()

    G = xp.fft.fft2(M, axes=(0, 1), norm='ortho')
    end = time.perf_counter()
    print(f"FFT along scan coordinate took {end - start}s")

    if aberrations is None:
        aberrations = xp.zeros((12))

    Psi_Qp = xp.zeros((ny, nx), dtype=G.dtype)
    Psi_Qp_left_sb = xp.zeros((ny, nx), dtype=xp.complex64)
    Psi_Qp_right_sb = xp.zeros((ny, nx), dtype=xp.complex64)

    start = time.perf_counter()
    if cuda_is_available:
        threadsperblock = 2 ** 8
        blockspergrid = m.ceil(np.prod(G.shape) / threadsperblock)
        strides = cp.array((np.array(G.strides) / (G.nbytes / G.size)).astype(np.int))

        single_sideband_kernel_cartesian[blockspergrid, threadsperblock](G, strides, Qx, Qy, Kx, Ky, aberrations,
                                                                         theta, alpha_rad, Psi_Qp, Psi_Qp_left_sb,
                                                                         Psi_Qp_right_sb, eps, lam, scale)
    else:
        def get_qx_qy(M, dx, fft_shifted=False):
            qxa = fftfreq(M[0], dx[0])
            qya = fftfreq(M[1], dx[1])
            [qxn, qyn] = np.meshgrid(qxa, qya)
            if fft_shifted:
                qxn = fftshift(qxn)
                qyn = fftshift(qyn)
            return qxn, qyn

        Kx, Ky = get_qx_qy([nkx, nky], k_max, fft_shifted=True)
        # reciprocal in scanning space
        Qx, Qy = get_qx_qy([nx, ny], dxy)

        Kplus = np.sqrt((Kx + Qx[:, :, None, None]) ** 2 + (Ky + Qy[:, :, None, None]) ** 2)
        Kminus = np.sqrt((Kx - Qx[:, :, None, None]) ** 2 + (Ky - Qy[:, :, None, None]) ** 2)
        K = np.sqrt(Kx ** 2 + Ky ** 2)

        A_KplusQ = np.zeros_like(G)
        A_KminusQ = np.zeros_like(G)

        C = np.zeros((12))
        A = np.exp(1j * cartesian_aberrations(Kx, Ky, lam, C)) * aperture_xp(Kx, Ky, lam, alpha_rad, edge=0)

        print('Creating aperture overlap functions')
        for ix, qx in enumerate(Qx[0]):
            print(f"{ix} / {Qx[0].shape}")
            for iy, qy in enumerate(Qy[:, 0]):
                x = Kx + qx
                y = Ky + qy
                A_KplusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                  edge=0)
                # A_KplusQ *= 1e4

                x = Kx - qx
                y = Ky - qy
                A_KminusQ[iy, ix] = np.exp(1j * cartesian_aberrations(x, y, lam, C)) * aperture_xp(x, y, lam, alpha_rad,
                                                                                                   edge=0)
                # A_KminusQ *= 1e4

        # [1] Equ. (4): Γ = A*(Kf)A(Kf-Qp) - A(Kf)A*(Kf+Qp)
        Gamma = A.conj() * A_KminusQ - A * A_KplusQ.conj()

        double_overlap1 = (Kplus < alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus > alpha_rad / lam)
        double_overlap2 = (Kplus > alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus < alpha_rad / lam)

        Psi_Qp = np.zeros((ny, nx), dtype=np.complex64)
        Psi_Qp_left_sb = np.zeros((ny, nx), dtype=np.complex64)
        Psi_Qp_right_sb = np.zeros((ny, nx), dtype=np.complex64)
        print(f"Now summing over K-space.")
        for y in trange(ny):
            for x in range(nx):
                Γ_abs = np.abs(Gamma[y, x])
                take = Γ_abs > eps
                Psi_Qp[y, x] = np.sum(G[y, x][take] * Gamma[y, x][take].conj())
                Psi_Qp_left_sb[y, x] = np.sum(G[y, x][double_overlap1[y, x]])
                Psi_Qp_right_sb[y, x] = np.sum(G[y, x][double_overlap2[y, x]])

                # direct beam at zero spatial frequency
                if x == 0 and y == 0:
                    Psi_Qp[y, x] = np.sum(np.abs(G[y, x]))
                    Psi_Qp_left_sb[y, x] = np.sum(np.abs(G[y, x]))
                    Psi_Qp_right_sb[y, x] = np.sum(np.abs(G[y, x]))

    end = time.perf_counter()
    print(f"SSB took {end - start}")

    Psi_Rp = xp.fft.ifft2(Psi_Qp, norm='ortho')
    Psi_Rp_left_sb = xp.fft.ifft2(Psi_Qp_left_sb, norm='ortho')
    Psi_Rp_right_sb = xp.fft.ifft2(Psi_Qp_right_sb, norm='ortho')

    if cuda_is_available:
        Psi_Rp = Psi_Rp.get()
        Psi_Rp_left_sb = Psi_Rp_left_sb.get()
        Psi_Rp_right_sb = Psi_Rp_right_sb.get()

    return Psi_Rp, Psi_Rp_left_sb, Psi_Rp_right_sb


def single_sideband_reconstruction(G, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha_rad,
                                   Psi_Qp, Psi_Qp_left_sb, Psi_Qp_right_sb, eps, lam):
    """
        Perform a ptychographic reconstruction directly from a provided G function:

            G(r',\rho') = |A(r')|^2 \delta(\rho') + A(r')A*(r'+\rho')Ψ*(-\rho')+ A*(r')A(r'-\rho')Ψ(\rho')

        We solve this equation for Ψ*(\rho') in two different ways:

        1) collect all the signal in the bright-field by multiplying G with::

            A(r')A*(r'+\rho')+ A*(r')A(r'-\rho')[2]

        2) collect only the signal in the double-overlap region [1]

        References:
            * [1] Rodenburg, J. M., McCallum, B. C. & Nellist, P. D. Experimental tests on
              double-resolution coherent imaging via STEM. Ultramicroscopy 48, 304–314 (1993).
            * [2] Yang, H., Ercius, P., Nellist, P. D. & Ophus, C. Enhanced phase contrast
              transfer using ptychography combined with a pre-specimen phase plate in a
              scanning transmission electron microscope. Ultramicroscopy 171, 117–125 (2016).

        Args:
            G (th.tensor, float, 4D): (NY, NX, MY, MX) 4D tensor of disk overlap functions, the datacube fourier-transformed along the scan dimensions
            Qx_all (th.tensor, float, 1D): Qx components
            Qy_all (th.tensor, float, 1D): Qy components
            Kx_all (th.tensor, float, 1D): Kx components
            Ky_all (th.tensor, float, 1D): Ky components
            aberrations (th.tensor, float, 1D): (12,) aberration coefficients
            theta_rot (float): convergence semi-angle in rad
            alpha_rad (float): convergence semi-angle in rad
            Psi_Qp (th.tensor, float, 2D): (NY, NX) storage tensor for the resulting exit wave
            Psi_Qp_left_sb (th.tensor, float, 2D): (NY, NX) storage tensor for the resulting exit wave, left sideband
            Psi_Qp_right_sb (th.tensor, float, 2D): (NY, NX) storage tensor for the resulting exit wave, right sideband
            eps (float): small value, typically 1e-3
            lam (float): wavelength in angstrom

        Returns:
            None
        """

    xp = sp.backend.get_array_module(G)
    threadsperblock = 2 ** 8
    blockspergrid = m.ceil(np.prod(G.shape) / threadsperblock)
    strides = xp.array((np.array(G.strides) / (G.nbytes / G.size)).astype(np.int))
    scale = 1
    single_sideband_kernel_cartesian[blockspergrid, threadsperblock](G, strides, Qx_all, Qy_all, Kx_all, Ky_all,
                                                                     aberrations,
                                                                     theta_rot, alpha_rad, Psi_Qp, Psi_Qp_left_sb,
                                                                     Psi_Qp_right_sb, eps, lam, scale)
    xp.cuda.Device(Psi_Qp.device).synchronize()
