"""
A general note on all these functions is that they are designed for use with rotation calibration into the pointslist.
However, they have to date only been used with the Qx and Qy in pixels and not calibrated into reciprocal units.  
There is no reason why this should not work, but the default tolerance would need adjustment.
"""

# import numpy as np
# from emdfile import tqdmnd


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
        apertureposition = [
            (center[0] + s1 * g1[0] + s2 * g2[0], center[1] + s1 * g1[1] + s2 * g2[1])
        ]
        centeredapertureposition = [(s1 * g1[0] + s2 * g2[0], s1 * g1[1] + s2 * g2[1])]
        if returns == "both":
            return apertureposition, centeredapertureposition
        elif returns == "centered":
            return centeredapertureposition
        else:
            print("incorrect selection of return parameter")

    if mode == "2-beam":
        aperturepositions = []
        centeredaperturepositions = []

        for i in np.arange(n1lims[0], n1lims[1] + 1):
            v = center[0] + i * g1[0]
            h = center[1] + i * g1[1]
            vc = i * g1[0]
            hc = i * g1[1]
            r = (vc**2 + hc**2) ** 0.5
            if pad < v < V - pad and pad < h < H - pad and r1 <= r <= r2:
                aperturepositions += [(v, h)]
                centeredaperturepositions += [(vc, hc)]
        if returns == "both":
            return aperturepositions, centeredaperturepositions
        elif returns == "centered":
            return centeredaperturepositions
        else:
            print("incorrect selection of return parameter")

    if mode == "array":
        aperturepositions = []
        centeredaperturepositions = []

        for i in np.arange(n1lims[0], n1lims[1] + 1):
            for j in np.arange(n2lims[0], n2lims[1] + 1):
                v = center[0] + i * g1[0] + j * g2[0] + s1 * g1[0] + s2 * g2[0]
                h = center[1] + i * g1[1] + j * g2[1] + s1 * g1[1] + s2 * g2[1]
                vc = i * g1[0] + j * g2[0] + s1 * g1[0] + s2 * g2[0]
                hc = i * g1[1] + j * g2[1] + s1 * g1[1] + s2 * g2[1]
                r = (vc**2 + hc**2) ** 0.5
                if pad < v < V - pad and pad < h < H - pad and r1 <= r <= r2:
                    aperturepositions += [(v, h)]
                    centeredaperturepositions += [(vc, hc)]
        if returns == "both":
            return aperturepositions, centeredaperturepositions
        elif returns == "centered":
            return centeredaperturepositions
        else:
            print("incorrect selection of return parameter")

    else:
        print("incorrect mode selection")


def pointlist_to_array(bplist, idim, jdim):
    """
    This function turns the py4dstem pointslist object to a simple numpy array that is more
    convenient for rapid array processing in numpy

    idim and jdim are the dimensions in the Rx and Ry directions

    returns an array called pointsarray
        This will be an 2D numpy array of n points x 5 columns:
            qx
            qy
            I
            Rx
            Ry
    """
    for i, j in tqdmnd(idim, jdim):
        if i == j == 0:
            pointsarray = np.array(
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
            pointsarray = np.vstack((pointsarray, nps))
    return pointsarray


def pointlist_differences(apertureposition, pointsarray):
    """
    calculates differences between a specific aperture position
    and a whole list of detected points for a dataset (as an array)

    returns the Euclidean distances as a 1D numpy array
    """
    subtractor = np.array(
        [[apertureposition[0], apertureposition[1]] * pointsarray.shape[0]]
    ).reshape((pointsarray.shape[0], 2))
    diff = ((pointsarray[:, :2] - subtractor) ** 2).sum(axis=1) ** 0.5
    return diff


def DDFimage(dataset, pointsarray, aperturearray, tol=1):
    """
    dataset is a 4DSTEM dataset as a numpy array, only used to get the size of image

    pointsarray is an array of points as calculated by pointlist_to_array

    aperturearray is a list of tuples of aperture centers generated by aperture_array_generator

    tol is the tolerance for a displacement between points and centers (in pixels)

    this does rely on the pointslist_differences function

    returns a the DDF image as a 2D numpy array
    """
    image = np.zeros_like(dataset[:, :, 0, 0])
    for aperture_index in tqdmnd(len(aperturearray)):
        apertureposition = aperturearray[aperture_index]
        intensities = np.vstack(
            (pointsarray[:, 2:].T, pointlist_differences(apertureposition, pointsarray))
        ).T
        intensities2 = np.delete(intensities, np.where(intensities[:, 3] > tol), axis=0)
        for row in range(intensities2[:, 0].shape[0]):
            image[
                intensities2[row, 1].astype(int), intensities2[row, 2].astype(int)
            ] += intensities2[row, 0]
    return image
