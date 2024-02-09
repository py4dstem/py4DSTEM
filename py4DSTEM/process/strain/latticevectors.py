# Functions for indexing the Bragg directions

import numpy as np
from emdfile import PointList, PointListArray, tqdmnd
from numpy.linalg import lstsq
from py4DSTEM.data import RealSlice


def index_bragg_directions(x0, y0, gx, gy, g1, g2):
    """
    From an origin (x0,y0), a set of reciprocal lattice vectors gx,gy, and an pair of
    lattice vectors g1=(g1x,g1y), g2=(g2x,g2y), find the indices (h,k) of all the
    reciprocal lattice directions.

    The approach is to solve the matrix equation
            ``alpha = beta * M``
    where alpha is the 2xN array of the (x,y) coordinates of N measured bragg directions,
    beta is the 2x2 array of the two lattice vectors u,v, and M is the 2xN array of the
    h,k indices.

    Args:
        x0 (float): x-coord of origin
        y0 (float): y-coord of origin
        gx (1d array): x-coord of the reciprocal lattice vectors
        gy (1d array): y-coord of the reciprocal lattice vectors
        g1 (2-tuple of floats): g1x,g1y
        g2 (2-tuple of floats): g2x,g2y

    Returns:
        (3-tuple) A 3-tuple containing:

            * **h**: *(ndarray of ints)* first index of the bragg directions
            * **k**: *(ndarray of ints)* second index of the bragg directions
            * **bragg_directions**: *(PointList)* a 4-coordinate PointList with the
              indexed bragg directions; coords 'qx' and 'qy' contain bragg_x and bragg_y
              coords 'g1_ind' and 'g2_ind' contain g1_ind and g2_ind.
    """
    # Get beta, the matrix of lattice vectors
    beta = np.array([[g1[0], g2[0]], [g1[1], g2[1]]])

    # Get alpha, the matrix of measured bragg angles
    alpha = np.vstack([gx - x0, gy - y0])

    # Calculate M, the matrix of peak positions
    M = lstsq(beta, alpha, rcond=None)[0].T
    M = np.round(M).astype(int)

    # Get g1_ind,g2_ind
    g1_ind = M[:, 0]
    g2_ind = M[:, 1]

    # Store in a PointList
    coords = [("qx", float), ("qy", float), ("g1_ind", int), ("g2_ind", int)]
    temp_array = np.zeros([], dtype=coords)
    bragg_directions = PointList(data=temp_array)
    bragg_directions.add_data_by_field((gx, gy, g1_ind, g2_ind))
    mask = np.zeros(bragg_directions["qx"].shape[0])
    mask[0] = 1
    bragg_directions.remove(mask)

    return g1_ind, g2_ind, bragg_directions


def add_indices_to_braggvectors(
    braggpeaks, lattice, maxPeakSpacing, qx_shift=0, qy_shift=0, mask=None
):
    """
    Using the peak positions (qx,qy) and indices (g1_ind,g2_ind) in the PointList lattice,
    identify the indices for each peak in the PointListArray braggpeaks.
    Return a new braggpeaks_indexed PointListArray, containing a copy of braggpeaks plus
    three additional data columns -- 'g1_ind','g2_ind', and 'index_mask' -- specifying the peak
    indices with the ints (g1_ind,g2_ind) and indicating whether the peak was successfully indexed
    or not with the bool index_mask. If `mask` is specified, only the locations where
    mask is True are indexed.

    Args:
        braggpeaks (PointListArray): the braggpeaks to index. Must contain
            the coordinates 'qx', 'qy', and 'intensity'
        lattice (PointList): the positions (qx,qy) of the (g1_ind,g2_ind) lattice points.
            Must contain the coordinates 'qx', 'qy', 'g1_ind', and 'g2_ind'
        maxPeakSpacing (float): Maximum distance from the ideal lattice points
            to include a peak for indexing
        qx_shift,qy_shift (number): the shift of the origin in the `lattice` PointList
            relative to the `braggpeaks` PointListArray
        mask (bool): Boolean mask, same shape as the pointlistarray, indicating which
            locations should be indexed. This can be used to index different regions of
            the scan with different lattices

    Returns:
        (PointListArray): The original braggpeaks pointlistarray, with new coordinates
        'g1_ind', 'g2_ind', containing the indices of each indexable peak.
    """

    # assert isinstance(braggpeaks,BraggVectors)
    # assert isinstance(lattice, PointList)
    # assert np.all([name in lattice.dtype.names for name in ('qx','qy','h','k')])

    if mask is None:
        mask = np.ones(braggpeaks.Rshape, dtype=bool)

    assert (
        mask.shape == braggpeaks.Rshape
    ), "mask must have same shape as pointlistarray"
    assert mask.dtype == bool, "mask must be boolean"

    coords = [
        ("qx", float),
        ("qy", float),
        ("intensity", float),
        ("g1_ind", int),
        ("g2_ind", int),
    ]

    indexed_braggpeaks = PointListArray(
        dtype=coords,
        shape=braggpeaks.Rshape,
    )

    calstate = braggpeaks.calstate

    # loop over all the scan positions
    for Rx, Ry in tqdmnd(mask.shape[0], mask.shape[1]):
        if mask[Rx, Ry]:
            pl = braggpeaks.get_vectors(
                Rx,
                Ry,
                center=True,
                ellipse=calstate["ellipse"],
                rotate=calstate["rotate"],
                pixel=False,
            )
            for i in range(pl.data.shape[0]):
                r2 = (pl.data["qx"][i] - lattice.data["qx"] + qx_shift) ** 2 + (
                    pl.data["qy"][i] - lattice.data["qy"] + qy_shift
                ) ** 2
                ind = np.argmin(r2)
                if r2[ind] <= maxPeakSpacing**2:
                    indexed_braggpeaks[Rx, Ry].add_data_by_field(
                        (
                            pl.data["qx"][i],
                            pl.data["qy"][i],
                            pl.data["intensity"][i],
                            lattice.data["g1_ind"][ind],
                            lattice.data["g2_ind"][ind],
                        )
                    )

    return indexed_braggpeaks


def fit_lattice_vectors(braggpeaks, x0=0, y0=0, minNumPeaks=5):
    """
    Fits lattice vectors g1,g2 to braggpeaks given some known (g1_ind,g2_ind) indexing.

    Args:
        braggpeaks (PointList): A 6 coordinate PointList containing the data to fit.
            Coords are 'qx','qy' (the bragg peak positions), 'intensity' (used as a
            weighting factor when fitting), 'g1_ind','g2_ind' (indexing). May optionally also
            contain 'index_mask' (bool), indicating which peaks have been successfully
            indixed and should be used.
        x0 (float): x-coord of the origin
        y0 (float): y-coord of the origin
        minNumPeaks (int): if there are fewer than minNumPeaks peaks found in braggpeaks
            which can be indexed, return None for all return parameters

    Returns:
        (7-tuple) A 7-tuple containing:

            * **x0**: *(float)* the x-coord of the origin of the best-fit lattice.
            * **y0**: *(float)* the y-coord of the origin
            * **g1x**: *(float)* x-coord of the first lattice vector
            * **g1y**: *(float)* y-coord of the first lattice vector
            * **g2x**: *(float)* x-coord of the second lattice vector
            * **g2y**: *(float)* y-coord of the second lattice vector
            * **error**: *(float)* the fit error
    """
    assert isinstance(braggpeaks, PointList)
    assert np.all(
        [
            name in braggpeaks.dtype.names
            for name in ("qx", "qy", "intensity", "g1_ind", "g2_ind")
        ]
    )
    braggpeaks = braggpeaks.copy()

    # Remove unindexed peaks
    if "index_mask" in braggpeaks.dtype.names:
        deletemask = braggpeaks.data["index_mask"] == False  # noqa:E712
        braggpeaks.remove(deletemask)

    # Check to ensure enough peaks are present
    if braggpeaks.length < minNumPeaks:
        return None, None, None, None, None, None, None

    # Get M, the matrix of (g1_ind,g2_ind) indices
    g1_ind, g2_ind = braggpeaks.data["g1_ind"], braggpeaks.data["g2_ind"]
    M = np.vstack((np.ones_like(g1_ind, dtype=int), g1_ind, g2_ind)).T

    # Get alpha, the matrix of measured Bragg peak positions
    alpha = np.vstack((braggpeaks.data["qx"] - x0, braggpeaks.data["qy"] - y0)).T

    # Get weighted matrices
    weights = braggpeaks.data["intensity"]
    weighted_M = M * weights[:, np.newaxis]
    weighted_alpha = alpha * weights[:, np.newaxis]

    # Solve for lattice vectors
    beta = lstsq(weighted_M, weighted_alpha, rcond=None)[0]
    x0, y0 = beta[0, 0], beta[0, 1]
    g1x, g1y = beta[1, 0], beta[1, 1]
    g2x, g2y = beta[2, 0], beta[2, 1]

    # Calculate the error
    alpha_calculated = np.matmul(M, beta)
    error = np.sqrt(np.sum((alpha - alpha_calculated) ** 2, axis=1))
    error = np.sum(error * weights) / np.sum(weights)

    return x0, y0, g1x, g1y, g2x, g2y, error


def fit_lattice_vectors_all_DPs(braggpeaks, x0=0, y0=0, minNumPeaks=5):
    """
    Fits lattice vectors g1,g2 to each diffraction pattern in braggpeaks, given some
    known (h,k) indexing.

    Args:
        braggpeaks (PointList): A 6 coordinate PointList containing the data to fit.
            Coords are 'qx','qy' (the bragg peak positions), 'intensity' (used as a
            weighting factor when fitting), 'g1_ind','g2_ind' (indexing). May optionally also
            contain 'index_mask' (bool), indicating which peaks have been successfully
            indixed and should be used.
        x0 (float): x-coord of the origin
        y0 (float): y-coord of the origin
        minNumPeaks (int): if there are fewer than minNumPeaks peaks found in braggpeaks
            which can be indexed, return None for all return parameters

    Returns:
        (RealSlice): A RealSlice ``g1g2map`` containing the following 8 arrays:

            * ``g1g2_map.get_slice('x0')``     x-coord of the origin of the best fit lattice
            * ``g1g2_map.get_slice('y0')``     y-coord of the origin
            * ``g1g2_map.get_slice('g1x')``    x-coord of the first lattice vector
            * ``g1g2_map.get_slice('g1y')``    y-coord of the first lattice vector
            * ``g1g2_map.get_slice('g2x')``    x-coord of the second lattice vector
            * ``g1g2_map.get_slice('g2y')``    y-coord of the second lattice vector
            * ``g1g2_map.get_slice('error')``  the fit error
            * ``g1g2_map.get_slice('mask')``   1 for successful fits, 0 for unsuccessful
              fits
    """
    assert isinstance(braggpeaks, PointListArray)
    assert np.all(
        [
            name in braggpeaks.dtype.names
            for name in ("qx", "qy", "intensity", "g1_ind", "g2_ind")
        ]
    )

    # Make RealSlice to contain outputs
    slicelabels = ("x0", "y0", "g1x", "g1y", "g2x", "g2y", "error", "mask")
    g1g2_map = RealSlice(
        data=np.zeros((8, braggpeaks.shape[0], braggpeaks.shape[1])),
        slicelabels=slicelabels,
        name="g1g2_map",
    )

    # Fit lattice vectors
    for Rx, Ry in tqdmnd(
        braggpeaks.shape[0],
        braggpeaks.shape[1],
        desc="Fitting lattice vectors",
        unit="DP",
        unit_scale=True,
    ):
        braggpeaks_curr = braggpeaks.get_pointlist(Rx, Ry)
        qx0, qy0, g1x, g1y, g2x, g2y, error = fit_lattice_vectors(
            braggpeaks_curr, x0, y0, minNumPeaks
        )
        # Store data
        if g1x is not None:
            g1g2_map.get_slice("x0").data[Rx, Ry] = qx0
            # Assume this is a correct change
            g1g2_map.get_slice("y0").data[Rx, Ry] = qy0
            g1g2_map.get_slice("g1x").data[Rx, Ry] = g1x
            g1g2_map.get_slice("g1y").data[Rx, Ry] = g1y
            g1g2_map.get_slice("g2x").data[Rx, Ry] = g2x
            g1g2_map.get_slice("g2y").data[Rx, Ry] = g2y
            g1g2_map.get_slice("error").data[Rx, Ry] = error
            g1g2_map.get_slice("mask").data[Rx, Ry] = 1

    return g1g2_map


def get_reference_g1g2(g1g2_map, mask):
    """
    Gets a pair of reference lattice vectors from a region of real space specified by
    mask. Takes the median of the lattice vectors in g1g2_map within the specified
    region.

    Args:
        g1g2_map (RealSlice): the lattice vector map; contains 2D arrays in g1g2_map.data
            under the keys 'g1x', 'g1y', 'g2x', and 'g2y'.  See documentation for
            fit_lattice_vectors_all_DPs() for more information.
        mask (ndarray of bools): use lattice vectors from g1g2_map scan positions wherever
            mask==True

    Returns:
        (2-tuple of 2-tuples) A 2-tuple containing:

            * **g1**: *(2-tuple)* first reference lattice vector (x,y)
            * **g2**: *(2-tuple)* second reference lattice vector (x,y)
    """
    assert isinstance(g1g2_map, RealSlice)
    assert np.all(
        [name in g1g2_map.slicelabels for name in ("g1x", "g1y", "g2x", "g2y")]
    )
    assert mask.dtype == bool
    g1x = np.median(g1g2_map.get_slice("g1x").data[mask])
    g1y = np.median(g1g2_map.get_slice("g1y").data[mask])
    g2x = np.median(g1g2_map.get_slice("g2x").data[mask])
    g2y = np.median(g1g2_map.get_slice("g2y").data[mask])
    return (g1x, g1y), (g2x, g2y)


def get_strain_from_reference_g1g2(g1g2_map, g1, g2):
    """
    Gets a strain map from the reference lattice vectors g1,g2 and lattice vector map
    g1g2_map.

    Note that this function will return the strain map oriented with respect to the x/y
    axes of diffraction space - to rotate the coordinate system, use
    get_rotated_strain_map(). Calibration of the rotational misalignment between real and
    diffraction space may also be necessary.

    Args:
        g1g2_map (RealSlice): the lattice vector map; contains 2D arrays in g1g2_map.data
            under the keys 'g1x', 'g1y', 'g2x', and 'g2y'.  See documentation for
            fit_lattice_vectors_all_DPs() for more information.
        g1 (2-tuple): first reference lattice vector (x,y)
        g2 (2-tuple): second reference lattice vector (x,y)

    Returns:
        (RealSlice) the strain map; contains the elements of the infinitessimal strain
        matrix, in the following 5 arrays:

            * ``strain_map.get_slice('e_xx')``: change in lattice x-components with respect
              to x
            * ``strain_map.get_slice('e_yy')``: change in lattice y-components with respect
              to y
            * ``strain_map.get_slice('e_xy')``: change in lattice x-components with respect
              to y
            * ``strain_map.get_slice('theta')``: rotation of lattice with respect to
              reference
            * ``strain_map.get_slice('mask')``: 0/False indicates unknown values

        Note 1: the strain matrix has been symmetrized, so e_xy and e_yx are identical
    """
    assert isinstance(g1g2_map, RealSlice)
    assert np.all(
        [name in g1g2_map.slicelabels for name in ("g1x", "g1y", "g2x", "g2y", "mask")]
    )

    # Get RealSlice for output storage
    R_Nx, R_Ny = g1g2_map.get_slice("g1x").shape
    strain_map = RealSlice(
        data=np.zeros((5, R_Nx, R_Ny)),
        slicelabels=("e_xx", "e_yy", "e_xy", "theta", "mask"),
        name="strain_map",
    )

    # Get reference lattice matrix
    g1x, g1y = g1
    g2x, g2y = g2
    M = np.array([[g1x, g1y], [g2x, g2y]])

    for Rx, Ry in tqdmnd(
        R_Nx,
        R_Ny,
        desc="Calculating strain",
        unit="DP",
        unit_scale=True,
    ):
        # Get lattice vectors for DP at Rx,Ry
        alpha = np.array(
            [
                [
                    g1g2_map.get_slice("g1x").data[Rx, Ry],
                    g1g2_map.get_slice("g1y").data[Rx, Ry],
                ],
                [
                    g1g2_map.get_slice("g2x").data[Rx, Ry],
                    g1g2_map.get_slice("g2y").data[Rx, Ry],
                ],
            ]
        )
        # Get transformation matrix
        beta = lstsq(M, alpha, rcond=None)[0].T

        # Get the infinitesimal strain matrix
        strain_map.get_slice("e_xx").data[Rx, Ry] = 1 - beta[0, 0]
        strain_map.get_slice("e_yy").data[Rx, Ry] = 1 - beta[1, 1]
        strain_map.get_slice("e_xy").data[Rx, Ry] = -(beta[0, 1] + beta[1, 0]) / 2.0
        strain_map.get_slice("theta").data[Rx, Ry] = (beta[0, 1] - beta[1, 0]) / 2.0
        strain_map.get_slice("mask").data[Rx, Ry] = g1g2_map.get_slice("mask").data[
            Rx, Ry
        ]
    return strain_map


def get_rotated_strain_map(unrotated_strain_map, xaxis_x, xaxis_y, flip_theta):
    """
    Starting from a strain map defined with respect to the xy coordinate system of
    diffraction space, i.e. where exx and eyy are the compression/tension along the Qx
    and Qy directions, respectively, get a strain map defined with respect to some other
    right-handed coordinate system, in which the x-axis is oriented along (xaxis_x,
    xaxis_y).

    Args:
        xaxis_x,xaxis_y (float): diffraction space (x,y) coordinates of a vector
            along the new x-axis
        unrotated_strain_map (RealSlice): a RealSlice object containing 2D arrays of the
            infinitessimal strain matrix elements, stored at
                * unrotated_strain_map.get_slice('e_xx')
                * unrotated_strain_map.get_slice('e_xy')
                * unrotated_strain_map.get_slice('e_yy')
                * unrotated_strain_map.get_slice('theta')

    Returns:
        (RealSlice) the rotated counterpart to unrotated_strain_map, with the
        rotated_strain_map.get_slice('e_xx') element oriented along the new coordinate
        system
    """
    assert isinstance(unrotated_strain_map, RealSlice)
    assert np.all(
        [
            key in ["e_xx", "e_xy", "e_yy", "theta", "mask"]
            for key in unrotated_strain_map.slicelabels
        ]
    )
    theta = -np.arctan2(xaxis_y, xaxis_x)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cost2 = cost**2
    sint2 = sint**2

    Rx, Ry = unrotated_strain_map.get_slice("e_xx").data.shape
    rotated_strain_map = RealSlice(
        data=np.zeros((5, Rx, Ry)),
        slicelabels=["e_xx", "e_xy", "e_yy", "theta", "mask"],
        name=unrotated_strain_map.name + "_rotated".format(np.degrees(theta)),
    )

    rotated_strain_map.data[0, :, :] = (
        cost2 * unrotated_strain_map.get_slice("e_xx").data
        - 2 * cost * sint * unrotated_strain_map.get_slice("e_xy").data
        + sint2 * unrotated_strain_map.get_slice("e_yy").data
    )
    rotated_strain_map.data[1, :, :] = (
        cost
        * sint
        * (
            unrotated_strain_map.get_slice("e_xx").data
            - unrotated_strain_map.get_slice("e_yy").data
        )
        + (cost2 - sint2) * unrotated_strain_map.get_slice("e_xy").data
    )
    rotated_strain_map.data[2, :, :] = (
        sint2 * unrotated_strain_map.get_slice("e_xx").data
        + 2 * cost * sint * unrotated_strain_map.get_slice("e_xy").data
        + cost2 * unrotated_strain_map.get_slice("e_yy").data
    )
    if flip_theta is True:
        rotated_strain_map.data[3, :, :] = -unrotated_strain_map.get_slice("theta").data
    else:
        rotated_strain_map.data[3, :, :] = unrotated_strain_map.get_slice("theta").data
    rotated_strain_map.data[4, :, :] = unrotated_strain_map.get_slice("mask").data
    return rotated_strain_map
