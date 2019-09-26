import time
from py4DSTEM.file.datastructure import DataCube
import sigpy as cp
import torch as th
import numpy as np
from .utils import *
from py4DSTEM.process.utils import *
from py4DSTEM.process.utils import plot
from tqdm import trange


def aperture3(qx, qy, lam, alpha_max):
    xp = cp.get_array_module(qx)
    qx2 = qx ** 2
    qy2 = qy ** 2
    q = xp.sqrt(qx2 + qy2)
    ktheta = xp.arcsin(q * lam)
    return ktheta < alpha_max


def aperture_xp(qx, qy, lam, alpha_max, edge=2):
    xp = cp.get_array_module(qx)
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


def cartesian_aberrations(qx, qy, lam, C):
    """
    Aberrations defined with dimensionless cartesian coordinates

    :param qx: array_like, 2d qx vector
    :param qy: array_like, 2d qy vector
    :param lam: wavelength
    :param C: aberration coefficients
    :return: the aberration surface chi
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


def weak_phase_reconstruction(dc: DataCube, verbose=False, use_cuda=True):
    """
    Perform a ptychographic reconstruction of the datacube assuming a weak phase object. In the weak phase object
    approximation, the dataset in double Fourier-space coordinates can be described as [1]

    G(r',\rho') = |A(r')|^2 \delta(\rho') + A(r')A*(r'+\rho')Ψ*(-\rho')+ A*(r')A(r'-\rho')Ψ(\rho')

    We solve this equation for Ψ*(\rho') in two different ways:

    1) collect all the signal in the bright-field by multiplying G with A(r')A*(r'+\rho')+ A*(r')A(r'-\rho')[2]
    2) collect only the signal in the double-overlap region [1]

    References:
    [1] Rodenburg, J. M., McCallum, B. C. & Nellist, P. D. Experimental tests on double-resolution coherent imaging via
        STEM. Ultramicroscopy 48, 304–314 (1993).
    [2] Yang, H., Ercius, P., Nellist, P. D. & Ophus, C. Enhanced phase contrast transfer using ptychography combined
        with a pre-specimen phase plate in a scanning transmission electron microscope. Ultramicroscopy 171, 117–125 (2016).

    :param dc: py4DSTEM datacube
    :return: (Ψ_Rp, Ψ_Rp_left_sb, Ψ_Rp_right_sb)
            Ψ_Rp is the result of method 1) and Ψ_Rp_left_sb, Ψ_Rp_right_sb are the results of method 2)
    """

    assert 'accelerating_voltage' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: accelerating_voltage'
    assert 'convergence_semiangle_mrad' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: convergence_semiangle_mrad'

    assert 'K_pix_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: K_pix_size'
    assert 'R_pix_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: R_pix_size'
    assert 'R_to_K_rotation_degrees' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: R_to_K_rotation_degrees'

    complex_dtype = {"float32": np.complex64, "float64": np.complex128}
    M = dc.data
    ny, nx, nky, nkx = M.shape

    E = dc.metadata.microscope['accelerating_voltage']
    alpha_rad = dc.metadata.microscope['convergence_semiangle_mrad'] * 1e-3
    lam = electron_wavelength_angstrom(E)
    eps = 1e-3
    k_max = dc.metadata.calibration['K_pix_size']
    dxy = dc.metadata.calibration['R_pix_size']
    theta = np.deg2rad(dc.metadata.calibration['R_to_K_rotation_degrees'])

    cuda_is_available = th.cuda.is_available() if use_cuda else False

    if verbose:
        print(f"E               = {E}             eV")
        print(f"λ               = {lam * 1e2:2.2}   pm")
        print(f"dR              = {dxy}             Å")
        print(f"dK              = {k_max}           Å")
        print(f"scan       size = {[ny, nx]}")
        print(f"detector   size = {[nky, nkx]}")

    if cuda_is_available:
        M = cp.array(M, dtype=M.dtype)

    xp = cp.get_array_module(M)

    def get_qx_qy_1D(M, dx, dtype, fft_shifted=False):
        qxa = xp.fft.fftfreq(M[0], dx[0]).astype(dtype)
        qya = xp.fft.fftfreq(M[1], dx[1]).astype(dtype)
        if fft_shifted:
            qxa = xp.fft.fftshift(qxa)
            qya = xp.fft.fftshift(qya)
        return qxa, qya

    Kx, Ky = get_qx_qy_1D([nkx, nky], k_max, M.dtype, fft_shifted=True)
    Qx, Qy = get_qx_qy_1D([nx, ny], dxy, M.dtype, fft_shifted=False)

    pacbed = xp.mean(M, (0, 1))
    mean_intensity = xp.sum(pacbed)
    print(mean_intensity)
    ap = aperture3(Kx, Ky, lam, alpha_rad).astype(xp.float32)
    aperture_intensity = float(xp.sum(ap))
    print(aperture_intensity)
    scale = 1  # math.sqrt(mean_intensity / aperture_intensity)
    ap *= scale

    if verbose:
        if cuda_is_available:
            plot(pacbed.get(), 'PACBED')
        else:
            plot(pacbed, 'PACBED')

    start = time.perf_counter()

    # M = xp.pad(M, ((ny // 2, ny // 2), (nx // 2, nx // 2), (0, 0), (0, 0)), mode='constant', constant_values=xp.mean(M).get())

    G = xp.fft.fft2(M, axes=(0, 1), norm='ortho')
    end = time.perf_counter()
    print(f"FFT along scan coordinate took {end - start}s")

    aberrations = xp.zeros((16))
    aberration_angles = xp.zeros((12))

    Ψ_Qp = xp.zeros((ny, nx), dtype=G.dtype)
    Ψ_Qp_left_sb = xp.zeros((ny, nx), dtype=np.complex64)
    Ψ_Qp_right_sb = xp.zeros((ny, nx), dtype=np.complex64)

    start = time.perf_counter()
    if cuda_is_available:
        gs = G.shape
        threadsperblock = 2 ** 8
        blockspergrid = m.ceil(np.prod(G.shape) / threadsperblock)
        strides = cp.array((np.array(G.strides) / (G.nbytes / G.size)).astype(np.int))

        # Gamma = xp.zeros_like(G)
        single_sideband_kernel[blockspergrid, threadsperblock](G, strides, Qx, Qy, Kx, Ky, aberrations,
                                                               aberration_angles, theta, alpha_rad, Ψ_Qp, Ψ_Qp_left_sb,
                                                               Ψ_Qp_right_sb, eps, lam, scale)
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

        a20 = th.Tensor([20])

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
        Γ = A.conj() * A_KminusQ - A * A_KplusQ.conj()

        double_overlap1 = (Kplus < alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus > alpha_rad / lam)
        double_overlap2 = (Kplus > alpha_rad / lam) * (K < alpha_rad / lam) * (Kminus < alpha_rad / lam)

        Ψ_Qp = np.zeros((ny, nx), dtype=np.complex64)
        Ψ_Qp_left_sb = np.zeros((ny, nx), dtype=np.complex64)
        Ψ_Qp_right_sb = np.zeros((ny, nx), dtype=np.complex64)
        print(f"Now summing over K-space.")
        for y in trange(ny):
            for x in range(nx):
                Γ_abs = np.abs(Γ[y, x])
                take = Γ_abs > eps
                Ψ_Qp[y, x] = np.sum(G[y, x][take] * Γ[y, x][take].conj())
                Ψ_Qp_left_sb[y, x] = np.sum(G[y, x][double_overlap1[y, x]])
                Ψ_Qp_right_sb[y, x] = np.sum(G[y, x][double_overlap2[y, x]])

                # direct beam at zero spatial frequency
                if x == 0 and y == 0:
                    Ψ_Qp[y, x] = np.sum(np.abs(G[y, x]))
                    Ψ_Qp_left_sb[y, x] = np.sum(np.abs(G[y, x]))
                    Ψ_Qp_right_sb[y, x] = np.sum(np.abs(G[y, x]))

    end = time.perf_counter()
    print(f"SSB took {end - start}")

    Ψ_Rp = xp.fft.ifft2(Ψ_Qp, norm='ortho')
    Ψ_Rp_left_sb = xp.fft.ifft2(Ψ_Qp_left_sb, norm='ortho')
    Ψ_Rp_right_sb = xp.fft.ifft2(Ψ_Qp_right_sb, norm='ortho')

    if cuda_is_available:
        Ψ_Rp = Ψ_Rp.get()
        Ψ_Rp_left_sb = Ψ_Rp_left_sb.get()
        Ψ_Rp_right_sb = Ψ_Rp_right_sb.get()

    return Ψ_Rp, Ψ_Rp_left_sb, Ψ_Rp_right_sb
