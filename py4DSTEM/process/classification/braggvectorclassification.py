# Functions for classification using a method which initially classifies the detected bragg
# vectors, then uses these labels to classify real space scan positions

import numpy as np
from numpy.linalg import lstsq
from itertools import permutations
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from skimage.measure import label
from sklearn.decomposition import NMF

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
        N                   (N) the total number of indexed Bragg peaks in the 4D-STEM dataset
        braggpeak_labels    (list of lists of sets) the Bragg peak labels found at each scan
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


class BraggVectorClassification(object):
    """
    A class for classifying 4D-STEM data based on which BPs are found at each diffraction pattern.

    A BraggVectorClassification instance enables classification using several methods; a brief
    overview is provided here, with more details in each individual method's documentation.

    Initialization methods:

        __init__:
            Determine the initial classes. The approach here involves first segmenting diffraction
            space, using maxima of a Bragg vector map.

    Class refinement methods:
    Each of these methods creates a new set of candidate classes, *but does not yet overwrite the
    old classes*. This enables the new classes to be viewed and compared to the old classes before
    deciding whether to accept or reject them. Thus running two of these methods in succession,
    without accepting changes in between, simply discards the first set of candidate classes.

        nmf:
            Nonnegative matrix factorization (X = WH) to refine the classes.  Briefly, after
            constructing a matrix X which describes which Bragg peaks were observed in each
            diffraction pattern, we factor X into two smaller matrices, W and H.  Physically, W and H
            describe a small set of classes, each of which corresponds to some subset of (or, more
            strictly, weights for) the Bragg peaks and the scan positions. We additionally impose
            the contraint that, on physical grounds, all the elements of X, W, and H must be
            nonnegative.
        split:
            If any classes contain multiple non-contiguous segments in real space, divide these into
            distinct classes.
        merge:
            If any classes contain sufficient overlap in both scan positions and BPs, merge them
            into a single class.

    Accepting/rejecting changes:

        accept:
            Updates classes (the W and H matrices) with the current candidate classes.
        reject:
            Discard the current candidate classes.

    Class examination methods:

        get_class:
            get a single class, returning both its BP weights and scan position weights
        get_class_BPs:
            get the BP weights for a single class
        get_class_image:
            get the image, i.e. scan position weights, associated with a single class
        get_candidate_class:
            as above, for the current candidate class
        get_candidate_class_BPs:
            as above, for the current candidate class
        get_candidate_class_image:
            as above, for the current candidate class

    """

    def __init__(self, braggpeaks, Qx, Qy, thresh=0.3, BP_fraction_thresh=0.1, max_iterations=200,
                       X_is_boolean=True):
        """
        Initializes a BraggVectorClassification instance, by
        1. Getting integer labels for all of the detected Bragg peaks according to which (Qx,Qy) is
           closest, then generating a corresponding set of integers for each scan position.
           See get_braggpeak_labels_by_scan_position() docstring for more info.
        2. From these sets, get initial classes described by BP sets by determining which labels
           are most likely to co-occur with each other. See get_initial_classes() docstring for more
           info.
        3. Generate the three matrices X, W, and H. See nmf() doscstring for discussion.

        Accepts:
            braggpeaks          (PointListArray) Bragg peaks; must have coords 'qx' and 'qy'
            Qx                  (ndarray of floats) x-coords of the voronoi points
            Qy                  (ndarray of floats) y-coords of the voronoi points
            thresh              (float in [0,1]) threshold for adding new BPs to a class
            BP_fraction_thresh  (float in [0,1]) algorithm terminates if fewer than this fraction
                                of the BPs have not been assigned to a class
            max_iterations      (int) algorithm terminates after this many iterations
            X_is_boolean        (bool) if True, populate X with bools (BP is or is not present)
                                if False, populate X with floats (BP c.c. intensities)
        """
        assert isinstance(braggpeaks,PointListArray), "braggpeaks must be a PointListArray"
        assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy')]), "braggpeaks must contain coords 'qx' and 'qy'"
        assert len(Qx)==len(Qy), "Qx and Qy must have same length"
        assert isinstance(X_is_boolean, bool)
        self.R_Nx = braggpeaks.shape[0]
        self.R_Ny = braggpeaks.shape[1]
        self.Qx = Qx
        self.Qy = Qy

        # Get the sets of Bragg peaks present at each scan position
        braggpeak_labels = get_braggpeak_labels_by_scan_position(braggpeaks, Qx, Qy)

        # Get sets of integers representing the initial classes
        BP_sets = get_initial_classes(braggpeak_labels, N=len(Qx), thresh=thresh,
                                      BP_fraction_thresh=BP_fraction_thresh,
                                      max_iterations=max_iterations)

        # Construct X, W, H matrices
        self.N_c = len(BP_sets)
        self.N_feat = len(Qx)
        self.N_meas = np.prod(braggpeaks.shape)

        # X
        self.X = np.zeros((self.N_feat,self.N_meas))
        for Rx in range(self.R_Nx):
            for Ry in range(self.R_Ny):
                R = Rx*self.R_Ny + Ry
                s = braggpeak_labels[Rx][Ry]
                pointlist = braggpeaks.get_pointlist(Rx,Ry)
                for i in s:
                    if X_is_boolean:
                        self.X[i,R] = True
                    else:
                        ind = np.argmin(np.hypot(pointlist.data['qx']-Qx[i],
                                                 pointlist.data['qy']-Qy[i]))
                        self.X[i,R] = pointlist.data['intensity'][ind]

        # W
        self.W = np.zeros((self.N_feat,self.N_c))
        for i in range(self.N_c):
            BP_set = BP_sets[i]
            for j in BP_set:
                self.W[j,i] = 1

        # H
        self.H = lstsq(self.W,self.X,rcond=None)[0]
        self.H = np.where(self.H<0,0,self.H)

        self.W_next = None
        self.H_next = None
        self.N_c_next = None

        return

    def nmf(self, max_iterations=1):
        """
        Nonnegative matrix factorization to refine the classes.

        In the matrix factorization performed here, X = WH, where

            X       is the data matrix. It has shape (N_feat,N_meas), where N_feat is the number
                    of Bragg peak integer labels (i.e. len(Qx)) and N_meas is the number of
                    diffraction patterns (i.e. R_Nx*R_Ny).  Element X[i,j] represents the value
                    of the i'th BP in the j'th DP.  The values depend on the flag
                    datamatrix_is_boolean: if True, X[i,j] is 1 if this BP was present in this
                    DP, or 0 if not; if False, X[i,j] is the cross correlation intensity of this
                    BP in this DP.
            W       the class matrix. It has shape (N_feat,N_c), where N_c is the number of
                    classes. The i'th column vector, w_i = W[:,i], describes the weight of each
                    Bragg peak in the i'th class.  w_i has length N_feat, and w_i[j]
                    describes how strongly the j'th BP is associated with the i'th class.
            H       the coefficient matrix. It has shape (N_c,N_meas).  The i'th column vector
                    H[:,i] describes the contribution of each class to scan position i.

                    Alternatively, we can completely equivalently think of H as a class matrix,
                    and W as a coeffient matrix.  In this picture, the i'th row vector of H,
                    h_i = H[i,:], describes the weight of each scan position in the i'th class.
                    h_i has length N_meas, and h_i[j] describes how strongly the j'th scan
                    position is associated with the i'th class.  The row vector W[i,:] is then
                    a coefficient vector, which gives the contributions each of the (H) classes
                    to the measured values of the i'th BP.  These pictures are related by a
                    transpose: X = WH is equivalent to X.T = (H.T)(W.T).

        Here, we use nonnegative matrix factorization, i.e. we impose the constrain that, on
        physical grounds, all elements of X, W, and H should be nonnegative.

        The computation itself is performed using the sklearn nmf class. When this method is called,
        the three relevant matrices should already be defined. This method refines W and H, with up to max_iterations NMF steps.

        Accepts:
            max_iterations      (int) the maximum number of NMF steps to take
        """
        sklearn_nmf = NMF(n_components=self.N_c, init='custom', max_iter=max_iterations)
        self.W_next = sklearn_nmf.fit_transform(self.X, W=self.W, H=self.H)
        self.H_next = sklearn_nmf.components_
        self.N_c_next = self.W_next.shape[1]

        return

    def split(self, sigma=2, threshold_split=0.25, expand_mask=1):
        """
        If any classes contain multiple non-contiguous segments in real space, divide these regions
        into distinct classes.

        Algorithm is as follows:
        First, an image of each class is obtained from its scan position weights.
        Then, the image is convolved with a gaussian of std sigma.
        This is then turned into a binary mask, by thresholding with threshold_split.
        Stray pixels are eliminated by performing a one pixel binary closing, then binary opening.
        The mask is then expanded by expand_mask pixels.
        Finally, the contiguous regions of the resulting mask are found. These become the new class
        components by scan position.

        Accepts:
            sigma           (float) std of gaussian kernel used to smooth the class images before
                            thresholding and splitting.
            threshold_split (float) used to threshold the class image to create a binary mask.
            expand_mask     (int) number of pixels by which to expand the mask before separating
                            into contiguous regions.
        """
        assert isinstance(expand_mask,(int,np.integer))

        W_next = np.zeros((self.N_feat,1))
        H_next = np.zeros((1,self.N_meas))
        for i in range(self.N_c):
            # Get the class in real space
            class_image = self.get_class_image(i)

            # Turn into a binary mask
            class_image = gaussian_filter(class_image,sigma)
            mask = class_image > (np.max(class_image)*threshold_split)
            mask = binary_opening(mask, iterations=1)
            mask = binary_closing(mask, iterations=1)
            mask = binary_dilation(mask, iterations=expand_mask)

            # Get connected regions
            labels, nlabels = label(mask, background=0, return_num=True, connectivity=2)

            # Add each region to the new W and H matrices
            for j in range(nlabels):
                mask = (labels == (j+1))
                mask = binary_erosion(mask, iterations=expand_mask)

                # Leave the Bragg peak weightings the same
                W_next = np.hstack((W_next,self.W[:,i,np.newaxis]))

                # Use the existing real space pixel weightings
                h_i = np.zeros(self.N_meas)
                h_i[mask.ravel()] = self.H[i,:][mask.ravel()]
                H_next = np.vstack((H_next,h_i[np.newaxis,:]))

        self.W_next = W_next[:,1:]
        self.H_next = H_next[1:,:]
        self.N_c_next = self.W_next.shape[1]

        return

    def merge(self, params):
        """
        If any classes contain sufficient overlap in both scan positions and BPs, merge them
        into a single class.
        """
        pass

    def accept(self):
        """
        Updates classes (the W and H matrices) with the current candidate classes.
        """
        if self.W_next is None or self.H_next is None:
            return
        else:
            self.W = self.W_next
            self.H = self.H_next
            self.N_c = self.N_c_next
            self.W_next = None
            self.H_next = None
            self.N_c_next = None

    def reject(self):
        """
        Discard the current candidate classes.
        """
        self.W_next = None
        self.H_next = None
        self.N_c_next = None

    def get_class(self, i):
        """
        Get a single class, returning both its BP weights and scan position weights.

        Accepts:
            i           (int) the class index

        Returns:
            class_BPs   (length N_feat array of floats) the weights of the N_feat Bragg peaks for
                        this class
            class_image (shape (R_Nx,R_Ny) array of floats) the weights of each scan position in this
                        class
        """
        class_BPs = self.W[:,i]
        class_image = self.H[i,:].reshape((self.R_Nx,self.R_Ny))
        return class_BPs, class_image

    def get_class_BPs(self, i):
        """
        Get a single class, returning its BP weights.

        Accepts:
            i       (int) the class index

        Returns:
            class_BPs   (length N_feat array of floats) the weights of the N_feat Bragg peaks for
                        this class
        """
        return self.W[:,i]

    def get_class_image(self, i):
        """
        Get a single class, returning its scan position weights.

        Accepts:
            i       (int) the class index

        Returns:
            class_image (shape (R_Nx,R_Ny) array of floats) the weights of each scan position in this
                        class
        """
        return self.H[i,:].reshape((self.R_Nx,self.R_Ny))

    def get_candidate_class(self, i):
        """
        Get a single candidate class, returning both its BP weights and scan position weights.

        Accepts:
            i           (int) the class index

        Returns:
            class_BPs   (length N_feat array of floats) the weights of the N_feat Bragg peaks for
                        this class
            class_image (shape (R_Nx,R_Ny) array of floats) the weights of each scan position in this
                        class
        """
        assert self.W_next is not None, "W_next is not assigned."
        assert self.H_next is not None, "H_next is not assigned."

        class_BPs = self.W_next[:,i]
        class_image = self.H_next[i,:].reshape((self.R_Nx,self.R_Ny))
        return class_BPs, class_image

    def get_candidate_class_BPs(self, i):
        """
        Get a single candidate class, returning its BP weights.

        Accepts:
            i           (int) the class index

        Returns:
            class_BPs   (length N_feat array of floats) the weights of the N_feat Bragg peaks for
                        this class
        """
        assert self.W_next is not None, "W_next is not assigned."

        return self.W_next[:,i]

    def get_candidate_class_image(self, i):
        """
        Get a single candidate class, returning its scan position weights.

        Accepts:
            i           (int) the class index

        Returns:
            class_image (shape (R_Nx,R_Ny) array of floats) the weights of each scan position in this
                        class
        """
        assert self.H_next is not None, "H_next is not assigned."

        return self.H_next[i,:].reshape((self.R_Nx,self.R_Ny))











