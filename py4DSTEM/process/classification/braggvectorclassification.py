# Functions for classification using a method which initially classifies the detected bragg
# vectors, then uses these labels to classify real space scan positions

import numpy as np
from itertools import permutations
from ...file.datastructure import PointListArray

def get_braggpeak_labels_by_scan_position(braggpeaks, Qx, Qy):
    """
    For each scan position, gets a set of integers, specifying the bragg peaks at this scan
    position.

    From a set of positions in diffraction space (Qx,Qy), assign each detected bragg peak in the
    PointListArray braggpeaks a label corresponding to the index of the closest position; thus
    for a bragg peak at (qx,qy), if the closest position in (Qx,Qy) is (Qx[i],Qy[i]), assign
    this peak the label i. This is equivalent to assigning each bragg peak (qx,qy) a label
    according to the Voronoi region it lives in, given a voronoi tesselation seeded from the
    points (Qx,Qy).

    For each scan position, get the set of all indices i for all bragg peaks found at this scan
    position.

    Accepts:
        braggpeaks          (PointListArray) Bragg peaks; must have coords 'qx' and 'qy'
        Qx                  (ndarray of floats) x-coords of the voronoi points
        Qy                  (ndarray of floats) y-coords of the voronoi points

    Returns:
        braggpeak_labels    (list of lists of sets) the labels found at each scan position.
                            Scan position (Rx,Ry) is accessed via braggpeak_labels[Rx][Ry]
    """
    assert isinstance(braggpeaks,PointListArray), "braggpeaks must be a PointListArray"
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy')]), "braggpeaks must contain coords 'qx' and 'qy'"

    braggpeak_labels = [[set() for i in range(braggpeaks.shape[1])] for j in range(braggpeaks.shape[0])]
    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            s = braggpeak_labels[Rx][Ry]
            pointlist = braggpeaks.get_pointlist(Rx,Ry)
            for i in range(pointlist.length):
                label = np.argmin(np.hypot(Qx-pointlist.data['qx'][i],Qy-pointlist.data['qy'][i]))
                s.add(label)

    return braggpeak_labels


def get_initial_classes(braggpeak_labels, N, thresh=0.3, BP_fraction_thresh=0.1, max_iterations=200):
    """
    From the sets of Bragg peaks present at each scan position, get an initial guess classes at
    which Bragg peaks should be grouped together into classes.

    The algorithm is as follows:
    1. Calculate a 3-point correlation function, i.e. f(i,j,k) = the probability that Bragg
    peaks i, j, and k are all in the same DP.
    2. Find the BP triplet maximizing the 3-point function; include these three BPs in a class.
    3. Get all DPs containing the class BPs. From these, find the next most likely BP to also
    be present.  If its probability of coexisting with the known class BPs is greater than
    thresh, add it to the class and repeat this step. Otherwise, proceed to the next step.
    4. Check: if the new class is the same as a class that has already been found, OR if the
    fraction of BPs which have not yet been placed in a class is less than BP_fraction_thresh,
    or more than max_iterations have been attempted, finish, returning all classes. Otherwise,
    set all slices of the 3-point function containing the BPs in the new class to zero, and
    begin a new iteration, starting at step 2 using the new, altered 3-point function.

    Accepts:
        braggpeak_labels    (list of lists of sets) the bragg peak labels found at each scan
                            position; see get_braggpeak_labels_by_scan_position().
        thresh              (float in [0,1]) threshold for adding new BPs to a class
        BP_fraction_thresh  (float in [0,1]) algorithm terminates if fewer than this fraction
                            of the BPs have not been assigned to a class
        max_iterations      (int) algorithm terminates after this many iterations

    Returns:
        BP_sets             (list of sets) the sets of Bragg peaks constituting the classes
    """
    assert isinstance(braggpeak_labels[0][0],set)
    assert thresh >= 0 and thresh <= 1
    assert BP_fraction_thresh >= 0 and BP_fraction_thresh <= 1
    assert isinstance(max_iterations,(int,np.integer))
    R_Nx = len(braggpeak_labels)
    R_Ny = len(braggpeak_labels[0])

    # Get three-point function
    threepoint_function = np.zeros((N,N,N))
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            s = braggpeak_labels[Rx][Ry]
            perms = permutations(s,3)
            for perm in perms:
                threepoint_function[perm[0],perm[1],perm[2]] += 1
    threepoint_function /= R_Nx*R_Ny

    # Main loop
    BP_sets = []
    iteration = 0
    unused_BPs = np.ones(N,dtype=bool)
    seed_new_class = True
    while seed_new_class:
        ind1,ind2,ind3 = np.unravel_index(np.argmax(threepoint_function),(N,N,N))
        BP_set = set([ind1,ind2,ind3])
        grow_class = True
        while grow_class:
            frequencies = np.zeros(N)
            N_elements = 0
            for Rx in range(R_Nx):
                for Ry in range(R_Ny):
                    s = braggpeak_labels[Rx][Ry]
                    if BP_set.issubset(s):
                        N_elements += 1
                        for i in s:
                            frequencies[i] += 1
            frequencies /= N_elements
            for i in BP_set:
                frequencies[i] = 0
            ind_new = np.argmax(frequencies)
            if frequencies[ind_new] > thresh:
                BP_set.add(ind_new)
            else:
                grow_class = False

        # Modify 3-point function, add new BP set to list, and decide to continue or stop
        for i in BP_set:
            threepoint_function[i,:,:] = 0
            threepoint_function[:,i,:] = 0
            threepoint_function[:,:,i] = 0
            unused_BPs[i] = 0
        for s in BP_sets:
            if len(s) == len(s.union(BP_set)):
                seed_new_class = False
        if seed_new_class is True:
            BP_sets.append(BP_set)
        iteration += 1
        N_unused_BPs = np.sum(unused_BPs)
        if iteration > max_iterations or N_unused_BPs < N*BP_fraction_thresh:
            seed_new_class = False

    return BP_sets














