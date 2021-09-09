import time
from py4DSTEM.io.datastructure import DataCube
import sigpy as sp
from sigpy import config
if config.cupy_enabled:
    import cupy as cp
import torch as th
import numpy as np
import torch.nn as nn
from .utils import *
from py4DSTEM.process.utils import *
from py4DSTEM.process.utils import plot
from tqdm import trange

def sector_mask(shape, centre, radius, angle_range=(0,360)):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask * anglemask

def HSV_to_RGB(cin):
    """\
    HSV to RGB transformation.
    """

    # HSV channels
    h, s, v = cin

    i = (6. * h).astype(int)
    f = (6. * h) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    imout = np.zeros(h.shape + (3,), dtype=h.dtype)
    imout[:, :, 0] = 255 * (i0 * v + i1 * q + i2 * p + i3 * p + i4 * t + i5 * v)
    imout[:, :, 1] = 255 * (i0 * t + i1 * v + i2 * v + i3 * q + i4 * p + i5 * p)
    imout[:, :, 2] = 255 * (i0 * p + i1 * p + i2 * t + i3 * v + i4 * v + i5 * q)

    return imout
def P1A_to_HSV(cin, vmin=None, vmax=None):
    """\
    Transform a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.
    """
    # HSV channels
    h = .5 * np.angle(cin) / np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    assert vmin < vmax
    v = (v.clip(vmin, vmax) - vmin) / (vmax - vmin)

    return HSV_to_RGB((h, s, v))


def imsave(a, filename=None, vmin=None, vmax=None, cmap=None):
    """
    imsave(a) converts array a into, and returns a PIL image
    imsave(a, filename) returns the image and also saves it to filename
    imsave(a, ..., vmin=vmin, vmax=vmax) clips the array to values between vmin and vmax.
    imsave(a, ..., cmap=cmap) uses a matplotlib colormap.
    """

    if a.dtype.kind == 'c':
        # Image is complex
        if cmap is not None:
            print('imsave: Ignoring provided cmap - input array is complex')
        i = P1A_to_HSV(a, vmin, vmax)
        im = Image.fromarray(np.uint8(i), mode='RGB')

    else:
        if vmin is None:
            vmin = a.min()
        if vmax is None:
            vmax = a.max()
        im = Image.fromarray((255 * (a.clip(vmin, vmax) - vmin) / (vmax - vmin)).astype('uint8'))
        if cmap is not None:
            r = im.point(lambda x: cmap(x / 255.0)[0] * 255)
            g = im.point(lambda x: cmap(x / 255.0)[1] * 255)
            b = im.point(lambda x: cmap(x / 255.0)[2] * 255)
            im = Image.merge("RGB", (r, g, b))
            # b = (255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8')
            # im = Image.fromstring('L', a.shape[-1::-1], b.tostring())

    if filename is not None:
        im.save(filename)
    return im

def fourier_coordinates_2D(N, dx=[1.0, 1.0], centered=True):
    qxx = np.fft.fftfreq(N[1], dx[1])
    qyy = np.fft.fftfreq(N[0], dx[0])
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qx, qy = np.meshgrid(qxx, qyy)
    q = np.array([qy, qx]).astype(np.float32)
    return q

def fftshift_checkerboard(w, h):
    re = np.r_[w * [-1, 1]]  # even-numbered rows
    ro = np.r_[w * [1, -1]]  # odd-numbered rows
    return np.row_stack(h * (re, ro))

def cartesian_aberrations_single(qx, qy, lam, C):
    """
    Zernike polynomials in the cartesian coordinate system
    :param qx:
    :param qy:
    :param lam: wavelength in Angstrom
    :param C:   (12 ,)
    :return:
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

class ZernikeProbeSingle(nn.Module):
    def __init__(self, q: th.Tensor, lam, fft_shifted=True):
        """
        Creates an aberration surface from aberration coefficients. The output is backpropable

        :param q: 2 x M1 x M2 tensor of x coefficients of reciprocal space
        :param lam: wavelength in Angstrom
        :param C: aberration coefficients
        :return: (maximum size of all aberration tensors) x MY x MX
        """

        super(ZernikeProbeSingle, self).__init__()
        self.q = q
        self.lam = lam
        self.fft_shifted = fft_shifted

        if self.fft_shifted:
            cb = fftshift_checkerboard(self.q.shape[1] // 2, self.q.shape[2] // 2)
            self.cb = th.from_numpy(cb).float().to(q.device)

    def forward(self, C, A):
        chi = cartesian_aberrations_single(self.q[1], self.q[0], self.lam, C)
        Psi = th.exp(-1j*chi) * A.expand_as(chi)

        if self.fft_shifted:
            Psi = Psi * self.cb

        return Psi

def aperture3(qx, qy, lam, alpha_max):
    xp = sp.get_array_module(qx)
    qx2 = qx ** 2
    qy2 = qy ** 2
    q = xp.sqrt(qx2 + qy2)
    ktheta = xp.arcsin(q * lam)
    return ktheta < alpha_max


def aperture_xp(qx, qy, lam, alpha_max, edge=2):
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

def get_qx_qy_1D(M, dx, dtype=np.float32, fft_shifted=False):
    xp = sp.get_array_module(M)
    qxa = xp.fft.fftfreq(M[0], dx[0]).astype(dtype)
    qya = xp.fft.fftfreq(M[1], dx[1]).astype(dtype)
    if fft_shifted:
        qxa = xp.fft.fftshift(qxa)
        qya = xp.fft.fftshift(qya)
    return qxa, qya

def disk_overlap_function(Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
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
def disk_overlap_kernel(Γ, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
    def aperture2(qx, qy, lam, alpha_max):
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = m.sqrt(qx2 + qy2)
        ktheta = m.asin(q * lam)
        return ktheta < alpha_max

    def chi3(qy, qx, lam, C):
        """
        Zernike polynomials in the cartesian coordinate system
        :param qx:
        :param qy:
        :param lam: wavelength in Angstrom
        :param C:   (12 ,)
        :return:
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

        # aberr = Param()
        # aberr.C1 = C[0]
        # aberr.C12a = C[1]
        # aberr.C12b = C[2]
        # aberr.C21a = C[3]
        # aberr.C21b = C[4]
        # aberr.C23a = C[5]
        # aberr.C23b = C[6]
        # aberr.C3 = C[7]
        # aberr.C32a = C[8]
        # aberr.C32b = C[9]
        # aberr.C34a = C[10]
        # aberr.C34b = C[11]

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

    gs = Γ.shape
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
        A = aperture2(Ky, Kx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky + Qy, Kx + Qx, lam, aberrations)
        Ap = aperture2(Ky + Qy, Kx + Qx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky - Qy, Kx - Qx, lam, aberrations)
        Am = aperture2(Ky - Qy, Kx - Qx, lam, alpha) * cm.exp(-1j * chi)

        Γ[j, iky, ikx] = A.conjugate() * Am - A * Ap.conjugate()

def double_overlap_intensitities_in_range(G_max, thetas, Qx_max, Qy_max, Kx, Ky, aberrations,
                                          aberration_angles, alpha_rad, lam, do_plot=False):
    xp = sp.backend.get_array_module(G_max)
    intensities = np.zeros((len(thetas)))
    for i, theta_rot in enumerate(thetas):
        if th.cuda.is_available():
            Gamma = disk_overlap_function(Qx_max, Qy_max, Kx, Ky, aberrations, theta_rot, alpha_rad,lam)

        intensities[i] = xp.sum(xp.abs(G_max * Gamma.conj()))

    if do_plot:
        f, ax = plt.subplots()
        ax.scatter(np.rad2deg(thetas), intensities)
        plt.show()

    return intensities

def find_rotation_angle_with_double_disk_overlap(G, lam, k_max, dxy, alpha_rad, mask=None, n_fit=6, ranges=[360, 30],
                                                 partitions=[144, 120], verbose=False, manual_frequencies=None, aberrations=None):
    """
        Finds the best rotation angle by maximizing the double disk overlap intensity of the 4D dataset. Only valid
        for datasets where the scan step size is roughly on the same length scale as the illumination half-angle alpha.

    :param G: G function. 4DSTEM dataset Fourier transformed along the scan coordinates
    :param lam:
    :param k_max:
    :param dxy:
    :param alpha_rad:
    :param n_fit: number of object spatial frequencies to fit
    :param ranges:
    :param verbose:
    :return: the best rotation angle in radians.
    """
    ny, nx, nky, nkx = G.shape
    xp = sp.backend.get_array_module(G)

    def get_qx_qy_1D(M, dx, dtype, fft_shifted=False):
        qxa = xp.fft.fftfreq(M[0], dx[0]).astype(dtype)
        qya = xp.fft.fftfreq(M[1], dx[1]).astype(dtype)
        if fft_shifted:
            qxa = xp.fft.fftshift(qxa)
            qya = xp.fft.fftshift(qya)
        return qxa, qya

    Kx, Ky = get_qx_qy_1D([nkx, nky], k_max, G[0, 0, 0, 0].real.dtype, fft_shifted=True)
    Qx, Qy = get_qx_qy_1D([nx, ny], dxy, G[0, 0, 0, 0].real.dtype, fft_shifted=False)

    if aberrations is None:
        aberrations = xp.zeros((12))
    aberration_angles = xp.zeros((12))

    if manual_frequencies is None:
        Gabs = xp.sum(xp.abs(G), (2, 3))
        if mask is not None:
            gg = Gabs * mask
#             plot(gg.get(), 'Gabs * mask')
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
                                                            aberration_angles, alpha_rad, lam, do_plot=False)

        sortind = np.argsort(intensities)
        max_ind0 = sortind[-1]
        max_ind1 = sortind[0]
        best_angle = thetas[max_ind0]
        best_angle1 = thetas[max_ind1]
        if verbose:
            A = xp.zeros(G_max.shape[1:], dtype=xp.complex64)
            Ap = xp.zeros(G_max.shape[1:], dtype=xp.complex64)
            Am = xp.zeros(G_max.shape[1:], dtype=xp.complex64)
            print(f"Iteration {j}: current best rotation angle: {np.rad2deg(best_angle)}")
            Gamma = disk_overlap_function(Qx_max, Qy_max, Kx, Ky, aberrations, best_angle, alpha_rad,lam)

    max_ind = np.argsort(intensities)[-1]

    return max_ind, thetas, intensities

def weak_phase_reconstruction(dc: DataCube, verbose=False, use_cuda=True):
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

    Returns:
        (Ψ_Rp, Ψ_Rp_left_sb, Ψ_Rp_right_sb)
        Ψ_Rp is the result of method 1) and Ψ_Rp_left_sb, Ψ_Rp_right_sb are the results
        of method 2)
    """

    assert 'beam_energy' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: beam_energy'
    assert 'convergence_semiangle_mrad' in dc.metadata.microscope, 'metadata.microscope dictionary missing key: convergence_semiangle_mrad'

    assert 'Q_pixel_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: Q_pixel_size'
    assert 'R_pixel_size' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: R_pixel_size'
    assert 'QR_rotation' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: QR_rotation'
    assert 'QR_rotation_units' in dc.metadata.calibration, 'metadata.calibration dictionary missing key: QR_rotation_units'

    complex_dtype = {"float32": np.complex64, "float64": np.complex128}
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

    Kx, Ky = get_qx_qy_1D([nkx, nky], k_max, M.dtype, fft_shifted=True)
    Qx, Qy = get_qx_qy_1D([nx, ny], dxy, M.dtype, fft_shifted=False)

    pacbed = xp.mean(M, (0, 1))
    mean_intensity = xp.sum(pacbed)
    # print(mean_intensity)
    ap = aperture3(Kx, Ky, lam, alpha_rad).astype(xp.float32)
    aperture_intensity = float(xp.sum(ap))
    # print(aperture_intensity)
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
    Ψ_Qp_left_sb = xp.zeros((ny, nx), dtype=xp.complex64)
    Ψ_Qp_right_sb = xp.zeros((ny, nx), dtype=xp.complex64)

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

def single_sideband_reconstruction(G, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha_rad,
                                   Ψ_Qp, Ψ_Qp_left_sb, Ψ_Qp_right_sb, eps, lam):
    xp = sp.backend.get_array_module(G)
    threadsperblock = 2 ** 8
    blockspergrid = m.ceil(np.prod(G.shape) / threadsperblock)
    strides = xp.array((np.array(G.strides) / (G.nbytes / G.size)).astype(np.int))
    scale = 1
    single_sideband_kernel_cartesian[blockspergrid, threadsperblock](G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations,
                                                           theta_rot, alpha_rad, Ψ_Qp, Ψ_Qp_left_sb,
                                                           Ψ_Qp_right_sb, eps, lam, scale)
    xp.cuda.Device(Ψ_Qp.device).synchronize()