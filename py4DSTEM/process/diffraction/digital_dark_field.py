"""
A general note on all these functions is that they are designed for use with rotation calibration into the pointslist.
However, they have to date only been used with the Qx and Qy in pixels and not calibrated into reciprocal units.  
There is no reason why this should not work, but the default tolerance would need adjustment.
"""

import numpy as np
from matplotlib.collections import EllipseCollection
from emdfile import tqdmnd
from py4DSTEM import show


def aperture_array_generator(
    shape,
    center,
    pad,
    mode,
    g1,
    g2=(0, 0),
    s1=0,
    s2=0,
    r1=0,
    r2=250,
    n1lims=(-5, 5),
    n2lims=(-5, 5),
    returns="both",
    plot_result=False,
    plot_image=None,
    plot_marker_size=100,
    plot_marker_radius_pixels=None,
    plot_marker_color="g",
    figsize=(6, 6),
    returnfig=False,
    **kwargs,
):
    """
    shape is a tuple describing the shape of the diffraction patterns

    center is a tuple of the centre position (vertical, horizontal)

    pad is any edge boundary desired (i.e. no aperture positions within pad pixels of edge)

    mode tells what kind of calculation is desired.  Which parameters are required and exactly what they mean
    depends on this choice:

        'single': just one aperture at a specified position:
            g1 and g2, lattice vectors (non-colinear)
            s1 and s2, multiples of these used to find the required lattice position
                i.e. aperture placed at s1*g1 + s2*g2
            r1, r2 unused

        '2-beam': just a line of apertures along spots for a 2-beam condition:
            g1: lattice vector
            g2: unused
            n1lims: tuple of integers giving the largest multiples of this lattice vector to be used,
                negative and positive
            r1 and r2, inner and outerradii in pixels over which aperture points will be found (optional)
                this is a good way to exclude the central spot by setting r > disc radius

        'array': an array defined by g1 and g2 centred on s1*g1+s2*g2
            r1 and r2, inner and outerradii in pixels as for '2-beam'
            n1lims and n2lims: tuple of integers giving the largest multiples of each lattice vector to be used,
                as for 2-beam

    returns sets whether the function returns:

        'both' = a centered array and an array in raw pixel numbers (uncentered)
        'centered' = just the version centered on [0,0]
        in all cases, the return is a list of (Qx,Qy) tuples

    Parameters
    ----------
    shape: tuple, list, np.array
        2-element vector of the diffraction space shape.
    center: tuple, list, np.array
        2-element vector of the center coordinate.
    pad: float
        Spacing around the boundaries where no apertures are generated.
    mode: string
        'single', '2-beam' or 'array' depending on desired aperture configuration.
    g1: tuple, list, np.array
        2-element vector for first g vector.
    g2: tuple, list, np.array
        2-element vector for second g vector.
    s1: int
        Multiples of g1 to position aperture.
    s2: int
        Multiples of g2 to position aperture.
    r1: float
        inner radius
    r2: float
        outer radius
    n1lims: (int,int)
        Limits for the g1 vector.
    n2lims=(int,int)
        Limits for the g2 vector.
    returns:
        What function returns.
    plot_result: bool
        Plot the aperture array
    plot_image: bool
        Image to show in background of the aperture array
    plot_marker_size: float
        Marker size in points (standard matplotlib)
    plot_marker_radius_pixels: float
        Marker radius in pixels    
	plot_marker_color: 3-tuple or string
		Any sensible python color definition
    figsize: (float, float)
        Figure size.
    returnfig: bool
        Set to true to return the figure handles.

    Returns
    ----------
    aperture_positions:
        (N,2) array containing the aperture positions in the image coordinate system.
    centered_aperture_positions:
        (N,2) array containing the aperture positions in a centered coordinate system.
    fig, ax:
        Figure and axes handles for plot.

    """

    V, H = shape[0], shape[1]

    if mode == "single":
        aperture_positions = [
            (center[0] + s1 * g1[0] + s2 * g2[0], center[1] + s1 * g1[1] + s2 * g2[1])
        ]
        centered_aperture_positions = [
            (s1 * g1[0] + s2 * g2[0], s1 * g1[1] + s2 * g2[1])
        ]

    elif mode == "2-beam":
        aperture_positions = []
        centered_aperture_positions = []

        for i in np.arange(n1lims[0], n1lims[1] + 1):
            v = center[0] + i * g1[0]
            h = center[1] + i * g1[1]
            vc = i * g1[0]
            hc = i * g1[1]
            r = (vc**2 + hc**2) ** 0.5
            if pad < v < V - pad and pad < h < H - pad and r1 <= r <= r2:
                aperture_positions += [(v, h)]
                centered_aperture_positions += [(vc, hc)]

    elif mode == "array":
        aperture_positions = []
        centered_aperture_positions = []

        for i in np.arange(n1lims[0], n1lims[1] + 1):
            for j in np.arange(n2lims[0], n2lims[1] + 1):
                v = center[0] + i * g1[0] + j * g2[0] + s1 * g1[0] + s2 * g2[0]
                h = center[1] + i * g1[1] + j * g2[1] + s1 * g1[1] + s2 * g2[1]
                vc = i * g1[0] + j * g2[0] + s1 * g1[0] + s2 * g2[0]
                hc = i * g1[1] + j * g2[1] + s1 * g1[1] + s2 * g2[1]
                r = (vc**2 + hc**2) ** 0.5
                if pad < v < V - pad and pad < h < H - pad and r1 <= r <= r2:
                    aperture_positions += [(v, h)]
                    centered_aperture_positions += [(vc, hc)]
    else:
        print("incorrect mode selection")

    # Convert lists to numpy arrays
    aperture_positions = np.array(aperture_positions)
    centered_aperture_positions = np.array(centered_aperture_positions)

    # plotting
    if plot_result:
        if plot_image is None:
            plot_image = np.zeros(shape)

        fig, ax = show(
            plot_image,
            ticks=False,
            returnfig=True,
            **kwargs,
        )
        if plot_marker_size is None or plot_marker_radius_pixels is not None:
            offsets = list(
                zip(
                    aperture_positions[:, 1],
                    aperture_positions[:, 0],
                ),
            )
            ax.add_collection(
                EllipseCollection(
                    widths=2.0 * plot_marker_radius_pixels,
                    heights=2.0 * plot_marker_radius_pixels,
                    angles=0,
                    units="xy",
                    facecolors=plot_marker_color,
                    alpha=0.3,
                    offsets=offsets,
                    transOffset=ax.transData,
                ),
            )
        else:
            ax.scatter(
                aperture_positions[:, 1],
                aperture_positions[:, 0],
                color=(0.0, 1.0, 0.0, 0.3),
                s=plot_marker_size,
            )

    if returns == "both":
        if returnfig:
            return aperture_positions, centered_aperture_positions, fig, ax
        else:
            return aperture_positions, centered_aperture_positions
    elif returns == "centered":
        if returnfig:
            return centered_aperture_positions, fig, ax
        else:
            return centered_aperture_positions
    else:
        print("incorrect selection of return parameter")


def aperture_array_subtract(
    aperture_positions,
    aperture_positions_delete,
    shape,
    tol=1.0,
    plot_result=False,
    plot_image=None,
    plot_marker_size=100,
    figsize=(6, 6),
    returnfig=False,
    **kwargs,
):
    """
    This function takes in a set of aperture positions, and removes apertures within
    the user-specified tolerance from aperture_array_delete.

    Parameters
    ----------
    aperture_positions: tuple, list, np.array
        2-element vector(s) of the diffraction space shape of positions of apertures
    aperture_positions: tuple, list, np.array
        2-element vector(s) of the diffraction space shape of positions of apertures to remove from the list
    shape: tuple, list, np.array
        2-element vector of the diffraction space shape
    tol: float
        a single number giving the tolerance for a maximum distance between aperture positions in the two lists to still be considered a match
    plot_result: bool
        Plot the aperture array
    plot_marker_size: float
        Marker size in points (standard matplotlib)
    plot_marker_radius_pixels: float
        Marker radius in pixels.
    figsize: (float, float)
        Figure size.
    returnfig: bool
        Set to true to return the figure handles.

    Returns
    ----------
    aperture_positions:
        (N,2) array containing the aperture positions in the image coordinate system.
    fig, ax:
        Figure and axes handles for plot.
    """

    # Determine which apertures to keep
    keep = np.zeros(aperture_positions.shape[0], dtype="bool")
    tol2 = tol**2
    for a0 in range(aperture_positions.shape[0]):
        dist2_min = np.min(
            np.sum(
                (aperture_positions[a0] - aperture_positions_delete) ** 2,
                axis=1,
            ),
        )
        if dist2_min > tol2:
            keep[a0] = True

    aperture_positions_new = aperture_positions[keep]

    # plotting
    if plot_result:
        aperture_positions_del = aperture_positions[np.logical_not(keep)]

        if plot_image is None:
            plot_image = np.zeros(shape)

        fig, ax = show(
            plot_image,
            ticks=False,
            returnfig=True,
            **kwargs,
        )
        ax.scatter(
            aperture_positions_del[:, 1],
            aperture_positions_del[:, 0],
            color=(1.0, 0.0, 0.0, 0.3),
            s=plot_marker_size,
        )
        ax.scatter(
            aperture_positions_new[:, 1],
            aperture_positions_new[:, 0],
            color=(0.0, 1.0, 0.0, 0.3),
            s=plot_marker_size,
        )

    return aperture_positions_new


def pointlist_to_array(
    bragg_peaks,
    center=None,
    ellipse=None,
    pixel=None,
    rotate=None,
):
    """
    This function turns the py4DSTEM BraggVectors to a simple numpy array that is more
    convenient for rapid array processing in numpy

    Parameters
    ----------
    bragg_peaks: BraggVectors
        py4DSTEM BraggVectors
    center: bool
        If True, applies center calibration to bragg_peaks
    ellipse: bool
        if True, applies elliptical calibration to bragg_peaks
    pixel: bool
        if True, applies pixel calibration to bragg_peaks
    rotate: bool
        if True, applies rotational calibration to bragg_peaks

    Returns
    ----------
    points_array: numpy array
         This will be an 2D numpy array of n points x 5 columns:
            qx
            qy
            I
            Rx
            Ry
    """
    if center is None:
        center = bragg_peaks.calstate["center"]

    if ellipse is None:
        ellipse = bragg_peaks.calstate["ellipse"]

    if pixel is None:
        pixel = bragg_peaks.calstate["pixel"]

    if rotate is None:
        rotate = bragg_peaks.calstate["rotate"]

    for i, j in tqdmnd(bragg_peaks.Rshape[0], bragg_peaks.Rshape[1]):
        vectors = bragg_peaks.get_vectors(
            scan_x=i,
            scan_y=j,
            center=center,
            ellipse=ellipse,
            pixel=pixel,
            rotate=rotate,
        )

        if i == j == 0:
            points_array = np.array(
                [
                    vectors.qx,
                    vectors.qy,
                    vectors.I,
                    vectors.qx.shape[0] * [i],
                    vectors.qx.shape[0] * [j],
                ]
            ).T
        else:
            nps = np.array(
                [
                    vectors.qx,
                    vectors.qy,
                    vectors.I,
                    vectors.qx.shape[0] * [i],
                    vectors.qx.shape[0] * [j],
                ]
            ).T
            points_array = np.vstack((points_array, nps))

    return points_array


def pointlist_differences(aperture_position, points_array):
    """
    calculates Euclidean distances between a specific aperture position
    and a whole list of detected points for a dataset

    Parameters
    ----------
    aperture_position: tuple
        2-element vector of the diffraction space shape of a position of an aperture
    points_array: numpy array
        as produced by pointlist_to_array and defined in docstring for that function

    Returns
    ----------
    diff: numpy array
        the Euclidean distances as a 1D numpy array
    """
    subtractor = np.array(
        [[aperture_position[0], aperture_position[1]] * points_array.shape[0]]
    ).reshape((points_array.shape[0], 2))
    diff = ((points_array[:, :2] - subtractor) ** 2).sum(axis=1) ** 0.5
    return diff


def DDFimage(points_array, aperture_positions, Rshape=None, tol=1):
    """
    Calculates a Digital Dark Field image from a list of detected diffraction peak positions in a points_array and a list of aperture_positions, within a defined matching tolerance

    This does rely on the pointslist_differences function for the calculation

    Parameters
    ----------
    points_array: numpy array
        as produced by pointlist_to_array and defined in docstring for that function
    aperture_positions: tuple, list, np.array
        2-element vector(s) of the diffraction space shape of positions of apertures
    Rshape: tuple, list, array
        a 2 element vector giving the real space dimensions.  If not specified, this is determined from the max along points_array
    tol: float
        the tolerance in pixels or calibrated units for a point in the points_array to be considered to match to an aperture position in the aperture_positions array

    Returns
    ----------
    image: numpy array
        2D numpy array with dimensions determined by Rshape

    """

    if Rshape is None:
        Rshape = (
            np.max(np.max(points_array[:, 3])).astype("int") + 1,
            np.max(np.max(points_array[:, 4])).astype("int") + 1,
        )

    image = np.zeros(Rshape)
    for aperture_index in tqdmnd(len(aperture_positions)):
        aperture_position = aperture_positions[aperture_index]
        intensities = np.vstack(
            (
                points_array[:, 2:].T,
                pointlist_differences(aperture_position, points_array),
            )
        ).T
        intensities2 = np.delete(intensities, np.where(intensities[:, 3] > tol), axis=0)
        for row in range(intensities2[:, 0].shape[0]):
            image[
                intensities2[row, 1].astype(int), intensities2[row, 2].astype(int)
            ] += intensities2[row, 0]
    return image
