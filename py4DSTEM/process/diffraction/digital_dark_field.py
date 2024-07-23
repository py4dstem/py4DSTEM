
"""
A general note on all these functions is that they are designed for use with rotation calibration into the pointslist.
However, they have to date only been used with the Qx and Qy in pixels and not calibrated into reciprocal units.  
There is no reason why this should not work, but the default tolerance would need adjustment.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    plot_result = False,
    plot_image = None,
    plot_marker_size = 100,
    figsize = (6,6),
    returnfig = False,
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
    """

    V, H = shape[0], shape[1]

    if mode == "single":
        aperture_position = [
            (center[0] + s1 * g1[0] + s2 * g2[0], center[1] + s1 * g1[1] + s2 * g2[1])
        ]
        centered_aperture_position = [
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

        fig,ax = show(
            plot_image,
            ticks = False,
            returnfig = True,
            **kwargs,
        )
        ax.scatter(
            aperture_positions[:,1],
            aperture_positions[:,0],
            color = (0.0,1.0,0.0,0.3),
            s = plot_marker_size,
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
    tol = 1.0,
    plot_result = False,
    plot_image = None,
    plot_marker_size = 100,
    figsize = (6,6),
    returnfig = False,
    **kwargs,
    ):
    """
    This function takes in a set of aperture positions, and removes apertures within
    the user-specified tolerance from aperture_array_delete.
    """

    # Determine which apertures to keep 
    keep = np.zeros(aperture_positions.shape[0],dtype='bool')
    tol2 = tol**2
    for a0 in range(aperture_positions.shape[0]):
        dist2_min = np.min(
            np.sum(
                (aperture_positions[a0] - aperture_positions_delete)**2,
                axis = 1,
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

        fig,ax = show(
            plot_image,
            ticks = False,
            returnfig = True,
            **kwargs,
        )
        ax.scatter(
            aperture_positions_del[:,1],
            aperture_positions_del[:,0],
            color = (1.0,0.0,0.0,0.3),
            s = plot_marker_size,
        )
        ax.scatter(
            aperture_positions_new[:,1],
            aperture_positions_new[:,0],
            color = (0.0,1.0,0.0,0.3),
            s = plot_marker_size,
        )

    return aperture_positions_new



def pointlist_to_array(
    bplist, 
    # idim, 
    # jdim
    ):
    """
    This function turns the py4dstem pointslist object to a simple numpy array that is more
    convenient for rapid array processing in numpy

    idim and jdim are the dimensions in the Rx and Ry directions

    returns an array called points_array
        This will be an 2D numpy array of n points x 5 columns:
            qx
            qy
            I
            Rx
            Ry
    """
    for i, j in tqdmnd(bplist.Rshape[0], bplist.Rshape[1]):
        if i == j == 0:
            points_array = np.array(
                [
                    bplist.cal[i, j].qx,
                    bplist.cal[i, j].qy,
                    bplist.cal[i, j].I,
                    bplist.cal[i, j].qx.shape[0] * [i],
                    bplist.cal[i, j].qx.shape[0] * [j],
                ]
            ).T
        else:
            nps = np.array(
                [
                    bplist.cal[i, j].qx,
                    bplist.cal[i, j].qy,
                    bplist.cal[i, j].I,
                    bplist.cal[i, j].qx.shape[0] * [i],
                    bplist.cal[i, j].qx.shape[0] * [j],
                ]
            ).T
            points_array = np.vstack((points_array, nps))
    return points_array


def pointlist_differences(aperture_position, points_array):
    """
    calculates differences between a specific aperture position
    and a whole list of detected points for a dataset (as an array)

    returns the Euclidean distances as a 1D numpy array
    """
    subtractor = np.array(
        [[aperture_position[0], aperture_position[1]] * points_array.shape[0]]
    ).reshape((points_array.shape[0], 2))
    diff = ((points_array[:, :2] - subtractor) ** 2).sum(axis=1) ** 0.5
    return diff


def DDFimage(
    points_array, 
    aperture_positions, 
    Rshape = None,
    tol=1
):
    """
    points_array is an array of points as calculated by pointlist_to_array

    aperture_positions is a numpy.array of aperture centers generated by aperture_array_generator,
    with dimensions 2xN for N apertures.

    Rshape is a 2 element vector giving the real space dimensions.  If not specified, we
    take these value from the max along points_array.

    tol is the tolerance for a displacement between points and centers (in pixels)

    this does rely on the pointslist_differences function

    returns a the DDF image as a 2D numpy array
    """

    if Rshape is None:
        Rshape = (
            np.max(np.max(points_array[:,3])).astype('int')+1,
            np.max(np.max(points_array[:,4])).astype('int')+1,
        )

    image = np.zeros(Rshape)
    for aperture_index in tqdmnd(len(aperture_positions)):
        aperture_position = aperture_positions[aperture_index]
        intensities = np.vstack(
            (points_array[:, 2:].T, pointlist_differences(aperture_position, points_array))
        ).T
        intensities2 = np.delete(intensities, np.where(intensities[:, 3] > tol), axis=0)
        for row in range(intensities2[:, 0].shape[0]):
            image[
                intensities2[row, 1].astype(int), intensities2[row, 2].astype(int)
            ] += intensities2[row, 0]
    return image
