# Defines utility functions used by other functions in the /process/ directory.

import numpy as np
from numpy.fft import fftfreq, fftshift
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi
import math as ma
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from emdfile import tqdmnd

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np


def radial_reduction(ar, x0, y0, binsize=1, fn=np.mean, coords=None):
    """
    Evaluate a reduction function on pixels within annular rings centered on (x0,y0),
    with a ring width of binsize.

    By default, returns the mean value of pixels within each annulus.
    Some other useful reductions include: np.sum, np.std, np.count, np.median, ...

    When running in a loop, pre-compute the pixel coordinates and pass them in
    for improved performance, like so:
        coords = np.mgrid[0:ar.shape[0],0:ar.shape[1]]
        radial_sums = radial_reduction(ar, x0,y0, coords=coords)
    """
    qx, qy = coords if coords else np.mgrid[0 : ar.shape[0], 0 : ar.shape[1]]

    r = (
        np.floor(np.hypot(qx - x0, qy - y0).ravel() / binsize).astype(np.int64)
        * binsize
    )
    edges = np.cumsum(np.bincount(r)[::binsize])
    slices = [slice(0, edges[0])] + [
        slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
    ]
    rargsort = np.argsort(r)
    sorted_ar = ar.ravel()[rargsort]
    reductions = np.array([fn(sorted_ar[s]) for s in slices])

    return reductions


def plot(
    img,
    title="Image",
    savePath=None,
    cmap="inferno",
    show=True,
    vmax=None,
    figsize=(10, 10),
    scale=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, interpolation="nearest", cmap=plt.get_cmap(cmap), vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(title)
    fontprops = fm.FontProperties(size=18)
    if scale is not None:
        scalebar = AnchoredSizeBar(
            ax.transData,
            scale[0],
            scale[1],
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=img.shape[0] / 40,
            fontproperties=fontprops,
        )

        ax.add_artist(scalebar)
    ax.grid(False)
    if savePath is not None:
        fig.savefig(savePath + ".png", dpi=600)
        fig.savefig(savePath + ".eps", dpi=600)
    if show:
        plt.show()


def electron_wavelength_angstrom(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = h / ma.sqrt(2 * m * e * E_eV) / ma.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    return lam


def electron_interaction_parameter(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34
    lam = h / ma.sqrt(2 * m * e * E_eV) / ma.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    sigma = (2 * np.pi / lam / E_eV) * (m * c**2 + e * E_eV) / (2 * m * c**2 + e * E_eV)
    return sigma


def sector_mask(shape, centre, radius, angle_range=(0, 360)):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.

    Args:
        shape: 2D shape of the mask
        centre: 2D center of the circular sector
        radius: radius of the circular mask
        angle_range: angular range of the circular mask
    """
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= 2 * np.pi

    # circular mask
    circmask = r2 <= radius * radius

    # print 'radius - ', radius

    # angular mask
    anglemask = theta < (tmax - tmin)

    return circmask * anglemask


def get_qx_qy_1d(M, dx=[1, 1], fft_shifted=False):
    """
    Generates 1D Fourier coordinates for a (Nx,Ny)-shaped 2D array.
    Specifying the dx argument sets a unit size.

    Args:
        M: (2,) shape of the returned array
        dx: (2,) tuple, pixel size
        fft_shifted: True if result should be fft_shifted to have the origin in the center of the array
    """

    qxa = fftfreq(M[0], dx[0])
    qya = fftfreq(M[1], dx[1])
    if fft_shifted:
        qxa = fftshift(qxa)
        qya = fftshift(qya)
    return qxa, qya


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


def get_maxima_1D(ar, sigma=0, minSpacing=0, minRelativeIntensity=0, relativeToPeak=0):
    """
    Finds the indices where 1D array ar is a local maximum.
    Optional parameters allow blurring the array and filtering the output;
    setting each to 0 (default) turns off these functions.

    Args:
        ar (1D array):
        sigma (number): gaussian blur std to apply to ar before finding maxima
        minSpacing (number): if two maxima are found within minSpacing, the dimmer one
            is removed
        minRelativeIntensity (number): maxima dimmer than minRelativeIntensity compared
            to the relativeToPeak'th brightest maximum are removed
        relativeToPeak (int): 0=brightest maximum. 1=next brightest, etc.

    Returns:
        (array of ints): An array of indices where ar is a local maximum, sorted by intensity.
    """
    assert len(ar.shape) == 1, "ar must be 1D"
    assert isinstance(
        relativeToPeak, (int, np.integer)
    ), "relativeToPeak must be an int"
    if sigma > 0:
        ar = gaussian_filter(ar, sigma)

    # Get maxima and intensity arrays
    maxima_bool = np.logical_and((ar > np.roll(ar, -1)), (ar >= np.roll(ar, +1)))
    x = np.arange(len(ar))[maxima_bool]
    intensity = ar[maxima_bool]

    # Sort by intensity
    temp_ar = np.array(
        [(x, inten) for inten, x in sorted(zip(intensity, x), reverse=True)]
    )
    x, intensity = temp_ar[:, 0], temp_ar[:, 1]

    # Remove points which are too close
    if minSpacing > 0:
        deletemask = np.zeros(len(x), dtype=bool)
        for i in range(len(x)):
            if not deletemask[i]:
                delete = np.abs(x[i] - x) < minSpacing
                delete[: i + 1] = False
                deletemask = deletemask | delete
        x = np.delete(x, deletemask.nonzero()[0])
        intensity = np.delete(intensity, deletemask.nonzero()[0])

    # Remove points which are too dim
    if minRelativeIntensity > 0:
        deletemask = intensity / intensity[relativeToPeak] < minRelativeIntensity
        x = np.delete(x, deletemask.nonzero()[0])
        intensity = np.delete(intensity, deletemask.nonzero()[0])

    return x.astype(int)


def linear_interpolation_1D(ar, x):
    """
    Calculates the 1D linear interpolation of array ar at position x using the two
    nearest elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    dx = x - x0
    return (1 - dx) * ar[x0] + dx * ar[x1]


def add_to_2D_array_from_floats(ar, x, y, I):
    """
    Adds the values I to array ar, distributing the value between the four pixels nearest
    (x,y) using linear interpolation.  Inputs (x,y,I) may be floats or arrays of floats.

    Note that if the same [x,y] coordinate appears more than once in the input array,
    only the *final* value of I at that coordinate will get added.
    """
    Nx, Ny = ar.shape
    x0, x1 = (np.floor(x)).astype(int), (np.ceil(x)).astype(int)
    y0, y1 = (np.floor(y)).astype(int), (np.ceil(y)).astype(int)
    mask = np.logical_and(
        np.logical_and(np.logical_and((x0 >= 0), (y0 >= 0)), (x1 < Nx)), (y1 < Ny)
    )
    dx = x - x0
    dy = y - y0
    ar[x0[mask], y0[mask]] += (1 - dx[mask]) * (1 - dy[mask]) * I[mask]
    ar[x0[mask], y1[mask]] += (1 - dx[mask]) * (dy[mask]) * I[mask]
    ar[x1[mask], y0[mask]] += (dx[mask]) * (1 - dy[mask]) * I[mask]
    ar[x1[mask], y1[mask]] += (dx[mask]) * (dy[mask]) * I[mask]
    return ar


def get_voronoi_vertices(voronoi, nx, ny, dist=10):
    """
    From a scipy.spatial.Voronoi instance, return a list of ndarrays, where each array
    is shape (N,2) and contains the (x,y) positions of the vertices of a voronoi region.

    The problem this function solves is that in a Voronoi instance, some vertices outside
    the field of view of the tesselated region are left unspecified; only the existence
    of a point beyond the field is referenced (which may or may not be 'at infinity').
    This function specifies all points, such that the vertices and edges of the
    tesselation may be directly laid over data.

    Args:
        voronoi (scipy.spatial.Voronoi): the voronoi tesselation
        nx (int): the x field-of-view of the tesselated region
        ny (int): the y field-of-view of the tesselated region
        dist (float, optional): place new vertices by extending new voronoi edges outside
            the frame by a distance of this factor times the distance of its known vertex
            from the frame edge

    Returns:
        (list of ndarrays of shape (N,2)): the (x,y) coords of the vertices of each
        voronoi region
    """
    assert isinstance(
        voronoi, Voronoi
    ), "voronoi must be a scipy.spatial.Voronoi instance"

    vertex_list = []

    # Get info about ridges containing an unknown vertex.  Include:
    #   -the index of its known vertex, in voronoi.vertices, and
    #   -the indices of its regions, in voronoi.point_region
    edgeridge_vertices_and_points = []
    for i in range(len(voronoi.ridge_vertices)):
        ridge = voronoi.ridge_vertices[i]
        if -1 in ridge:
            edgeridge_vertices_and_points.append(
                [max(ridge), voronoi.ridge_points[i, 0], voronoi.ridge_points[i, 1]]
            )
    edgeridge_vertices_and_points = np.array(edgeridge_vertices_and_points)

    # Loop over all regions
    for index in range(len(voronoi.regions)):
        # Get the vertex indices
        vertex_indices = voronoi.regions[index]
        vertices = np.array([0, 0])
        # Loop over all vertices
        for i in range(len(vertex_indices)):
            index_current = vertex_indices[i]
            if index_current != -1:
                # For known vertices, just add to a running list
                vertices = np.vstack((vertices, voronoi.vertices[index_current]))
            else:
                # For unknown vertices, get the first vertex it connects to,
                # and the two voronoi points that this ridge divides
                index_prev = vertex_indices[(i - 1) % len(vertex_indices)]
                edgeridge_index = int(
                    np.argwhere(edgeridge_vertices_and_points[:, 0] == index_prev)
                )
                index_vert, region0, region1 = edgeridge_vertices_and_points[
                    edgeridge_index, :
                ]
                x, y = voronoi.vertices[index_vert]
                # Only add new points for unknown vertices if the known index it connects to
                # is inside the frame.  Add points by finding the line segment starting at
                # the known point which is perpendicular to the segment connecting the two
                # voronoi points, and extending that line segment outside the frame.
                if (x > 0) and (x < nx) and (y > 0) and (y < ny):
                    x_r0, y_r0 = voronoi.points[region0]
                    x_r1, y_r1 = voronoi.points[region1]
                    m = -(x_r1 - x_r0) / (y_r1 - y_r0)
                    # Choose the direction to extend the ridge
                    ts = np.array([-x, -y / m, nx - x, (ny - y) / m])
                    x_t = lambda t: x + t
                    y_t = lambda t: y + m * t
                    t = ts[np.argmin(np.hypot(x - x_t(ts), y - y_t(ts)))]
                    x_new, y_new = x_t(dist * t), y_t(dist * t)
                    vertices = np.vstack((vertices, np.array([x_new, y_new])))
                else:
                    # If handling unknown points connecting to points outside the frame is
                    # desired, add here
                    pass

                # Repeat for the second vertec the unknown vertex connects to
                index_next = vertex_indices[(i + 1) % len(vertex_indices)]
                edgeridge_index = int(
                    np.argwhere(edgeridge_vertices_and_points[:, 0] == index_next)
                )
                index_vert, region0, region1 = edgeridge_vertices_and_points[
                    edgeridge_index, :
                ]
                x, y = voronoi.vertices[index_vert]
                if (x > 0) and (x < nx) and (y > 0) and (y < ny):
                    x_r0, y_r0 = voronoi.points[region0]
                    x_r1, y_r1 = voronoi.points[region1]
                    m = -(x_r1 - x_r0) / (y_r1 - y_r0)
                    # Choose the direction to extend the ridge
                    ts = np.array([-x, -y / m, nx - x, (ny - y) / m])
                    x_t = lambda t: x + t
                    y_t = lambda t: y + m * t
                    t = ts[np.argmin(np.hypot(x - x_t(ts), y - y_t(ts)))]
                    x_new, y_new = x_t(dist * t), y_t(dist * t)
                    vertices = np.vstack((vertices, np.array([x_new, y_new])))
                else:
                    pass

        # Remove regions with insufficiently many vertices
        if len(vertices) < 4:
            vertices = np.array([])
        # Remove initial dummy point
        else:
            vertices = vertices[1:, :]
        # Update vertex list with this region's vertices
        vertex_list.append(vertices)

    return vertex_list


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


def fourier_resample(
    array,
    scale=None,
    output_size=None,
    force_nonnegative=False,
    bandlimit_nyquist=None,
    bandlimit_power=2,
    dtype=np.float32,
    conserve_array_sums=False,
):
    """
    Resize a 2D array along any dimension, using Fourier interpolation / extrapolation.
    For 4D input arrays, only the final two axes can be resized.

    The scaling of the array can be specified by passing either `scale`, which sets
    the scaling factor along both axes to be scaled; or by passing `output_size`,
    which specifies the final dimensions of the scaled axes (and allows for different
    scaling along the x,y or kx,ky axes.)

    Args:
        array (2D/4D numpy array): Input array, or 4D stack of arrays, to be resized.
        scale (float): scalar value giving the scaling factor for all dimensions
        output_size (2-tuple of ints): two values giving either the (x,y) output size for 2D, or (kx,ky) for 4D
        force_nonnegative (bool): Force all outputs to be nonnegative, after filtering
        bandlimit_nyquist (float): Gaussian filter information limit in Nyquist units (0.5 max in both directions)
        bandlimit_power (float): Gaussian filter power law scaling (higher is sharper)
        dtype (numpy dtype): datatype for binned array. default is single precision float
        conserve_arrray_sums (bool): If True, the sums of the array are conserved

    Returns:
        the resized array (2D/4D numpy array)
    """

    # Verify input is 2D or 4D
    if np.size(array.shape) != 2 and np.size(array.shape) != 4:
        raise Exception(
            "Function does not support arrays with "
            + str(np.size(array.shape))
            + " dimensions"
        )

    # Get input size from last 2 dimensions
    input__size = array.shape[-2:]

    if scale is not None:
        assert (
            output_size is None
        ), "Cannot specify both a scaling factor and output size"
        assert np.size(scale) == 1, "scale should be a single value"
        scale = np.asarray(scale)
        output_size = (input__size * scale).astype("intp")
    else:
        assert scale is None, "Cannot specify both a scaling factor and output size"
        assert np.size(output_size) == 2, "output_size must contain two values"
        output_size = np.asarray(output_size)

    scale_output = np.prod(output_size) / np.prod(input__size)

    if bandlimit_nyquist is not None:
        kx = np.fft.fftfreq(output_size[0])
        ky = np.fft.fftfreq(output_size[1])
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2
        # Gaussian filter
        k_filt = np.exp(
            (k2 ** (bandlimit_power / 2)) / (-2 * bandlimit_nyquist**bandlimit_power)
        )

    # generate slices
    # named as {dimension}_{corner}_{in_/out},
    # where corner is ul, ur, ll, lr for {upper/lower}{left/right}

    # x slices
    if output_size[0] > input__size[0]:
        # x dimension increases
        x0 = int((input__size[0] + 1) // 2)
        x1 = int(input__size[0] // 2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0 - x1 + output_size[0], output_size[0])
        x_ll_in_ = slice(0 - x1 + input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0 - x1 + output_size[0], output_size[0])
        x_lr_in_ = slice(0 - x1 + input__size[0], input__size[0])

    elif output_size[0] < input__size[0]:
        # x dimension decreases
        x0 = int((output_size[0] + 1) // 2)
        x1 = int(output_size[0] // 2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0 - x1 + output_size[0], output_size[0])
        x_ll_in_ = slice(0 - x1 + input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0 - x1 + output_size[0], output_size[0])
        x_lr_in_ = slice(0 - x1 + input__size[0], input__size[0])

    else:
        # x dimension does not change
        x_ul_out = slice(None)
        x_ul_in_ = slice(None)

        x_ll_out = slice(None)
        x_ll_in_ = slice(None)

        x_ur_out = slice(None)
        x_ur_in_ = slice(None)

        x_lr_out = slice(None)
        x_lr_in_ = slice(None)

    # y slices
    if output_size[1] > input__size[1]:
        # y increases
        y0 = int((input__size[1] + 1) // 2)
        y1 = int(input__size[1] // 2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0 - y1 + output_size[1], output_size[1])
        y_ur_in_ = slice(0 - y1 + input__size[1], input__size[1])

        y_lr_out = slice(0 - y1 + output_size[1], output_size[1])
        y_lr_in_ = slice(0 - y1 + input__size[1], input__size[1])

    elif output_size[1] < input__size[1]:
        # y decreases
        y0 = int((output_size[1] + 1) // 2)
        y1 = int(output_size[1] // 2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0 - y1 + output_size[1], output_size[1])
        y_ur_in_ = slice(0 - y1 + input__size[1], input__size[1])

        y_lr_out = slice(0 - y1 + output_size[1], output_size[1])
        y_lr_in_ = slice(0 - y1 + input__size[1], input__size[1])

    else:
        # y dimension does not change
        y_ul_out = slice(None)
        y_ul_in_ = slice(None)

        y_ll_out = slice(None)
        y_ll_in_ = slice(None)

        y_ur_out = slice(None)
        y_ur_in_ = slice(None)

        y_lr_out = slice(None)
        y_lr_in_ = slice(None)

    if len(array.shape) == 2:
        # image array
        array_resize = np.zeros(output_size, dtype=np.complex64)
        array_fft = np.fft.fft2(array)

        # copy each quadrant into the resize array
        array_resize[x_ul_out, y_ul_out] = array_fft[x_ul_in_, y_ul_in_]
        array_resize[x_ll_out, y_ll_out] = array_fft[x_ll_in_, y_ll_in_]
        array_resize[x_ur_out, y_ur_out] = array_fft[x_ur_in_, y_ur_in_]
        array_resize[x_lr_out, y_lr_out] = array_fft[x_lr_in_, y_lr_in_]

        # Band limit if needed
        if bandlimit_nyquist is not None:
            array_resize *= k_filt

        # Back to real space
        array_resize = np.real(np.fft.ifft2(array_resize)).astype(dtype)

    elif len(array.shape) == 4:
        # This case is the same as the 2D case, but loops over the probe index arrays

        # init arrays
        array_resize = np.zeros((*array.shape[:2], *output_size), dtype)
        array_fft = np.zeros(input__size, dtype=np.complex64)
        array_output = np.zeros(output_size, dtype=np.complex64)

        for Rx, Ry in tqdmnd(
            array.shape[0],
            array.shape[1],
            desc="Resampling 4D datacube",
            unit="DP",
            unit_scale=True,
        ):
            array_fft[:, :] = np.fft.fft2(array[Rx, Ry, :, :])
            array_output[:, :] = 0

            # copy each quadrant into the resize array
            array_output[x_ul_out, y_ul_out] = array_fft[x_ul_in_, y_ul_in_]
            array_output[x_ll_out, y_ll_out] = array_fft[x_ll_in_, y_ll_in_]
            array_output[x_ur_out, y_ur_out] = array_fft[x_ur_in_, y_ur_in_]
            array_output[x_lr_out, y_lr_out] = array_fft[x_lr_in_, y_lr_in_]

            # Band limit if needed
            if bandlimit_nyquist is not None:
                array_output *= k_filt

            # Back to real space
            array_resize[Rx, Ry, :, :] = np.real(np.fft.ifft2(array_output)).astype(
                dtype
            )

    # Enforce positivity if needed, after filtering
    if force_nonnegative:
        array_resize = np.maximum(array_resize, 0)

    # Normalization
    if not conserve_array_sums:
        array_resize = array_resize * scale_output

    return array_resize


# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import matplotlib.font_manager as fm
#
#
# try:
#    from IPython.display import clear_output
# except ImportError:
#    def clear_output(wait=True):
#        pass
#
# def plot(img, title='Image', savePath=None, cmap='inferno', show=True, vmax=None,
#                                                        figsize=(10, 10), scale=None):
#    fig, ax = plt.subplots(figsize=figsize)
#    im = ax.imshow(img, interpolation='nearest', cmap=plt.get_cmap(cmap), vmax=vmax)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    ax.set_title(title)
#    fontprops = fm.FontProperties(size=18)
#    if scale is not None:
#        scalebar = AnchoredSizeBar(ax.transData,
#                                   scale[0], scale[1], 'lower right',
#                                   pad=0.1,
#                                   color='white',
#                                   frameon=False,
#                                   size_vertical=img.shape[0] / 40,
#                                   fontproperties=fontprops)
#
#        ax.add_artist(scalebar)
#    ax.grid(False)
#    if savePath is not None:
#        fig.savefig(savePath + '.png', dpi=600)
#        fig.savefig(savePath + '.eps', dpi=600)
#    if show:
#        plt.show()
