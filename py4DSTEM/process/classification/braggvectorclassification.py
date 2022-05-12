# Functions for classification using a method which initially classifies the detected bragg
# vectors, then uses these labels to classify real space scan positions

import numpy as np
from numpy.linalg import lstsq
from itertools import permutations
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from skimage.measure import label
from sklearn.decomposition import NMF

from ...io import PointListArray


class BraggVectorClassification(object):
    """
    A class for classifying 4D-STEM data based on which Bragg peaks are found at each
    diffraction pattern.

    A BraggVectorClassification instance enables classification using several methods; a brief
    overview is provided here, with more details in each individual method's documentation.

    Initialization methods:

        __init__:
            Determine the initial classes. The approach here involves first segmenting diffraction
            space, using maxima of a Bragg vector map.

        get_initial_classes_by_cooccurrence:

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

    Args:
        braggpeaks (PointListArray): Bragg peaks; must have coords 'qx' and 'qy'
        Qx (ndarray of floats): x-coords of the voronoi points
        Qy (ndarray of floats): y-coords of the voronoi points
        X_is_boolean (bool): if True, populate X with bools (BP is or is not present).
            if False, populate X with floats (BP c.c. intensities)
        max_dist (None or number): maximum distance from a given voronoi point a peak
            can be and still be associated with this label
    """

    def __init__(self, braggpeaks, Qx, Qy, X_is_boolean=True, max_dist=None):
        """
        Initializes a BraggVectorClassification instance.

        This method:
        1.  Gets integer labels for all of the detected Bragg peaks, according to which
            (Qx,Qy) is closest, then generating a corresponding set of integers for each scan
            position.  See get_braggpeak_labels_by_scan_position() docstring for more info.
        2.  Generates the data matrix X.  See the nmf() method docstring for more info.

        This method should be followed by one of the methods which populates the initial classes -
        currently, either get_initial_classes_by_cooccurrence() or get_initial_classes_from_images.
        These methods generate the W and H matrices -- i.e. the decompositions of the X matrix in
        terms of scan positions and Bragg peaks -- which are necessary for any subsequent
        processing.

        Args:
            braggpeaks (PointListArray): Bragg peaks; must have coords 'qx' and 'qy'
            Qx (ndarray of floats): x-coords of the voronoi points
            Qy (ndarray of floats): y-coords of the voronoi points
            X_is_boolean (bool): if True, populate X with bools (BP is or is not present).
                if False, populate X with floats (BP c.c. intensities)
            max_dist (None or number): maximum distance from a given voronoi point a peak
                can be and still be associated with this label
        """
        assert isinstance(braggpeaks,PointListArray), "braggpeaks must be a PointListArray"
        assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy')]), "braggpeaks must contain coords 'qx' and 'qy'"
        assert len(Qx)==len(Qy), "Qx and Qy must have same length"
        self.braggpeaks = braggpeaks
        self.R_Nx = braggpeaks.shape[0]  #: shape of real space (x)
        self.R_Ny = braggpeaks.shape[1]  #: shape of real space (y)
        self.Qx = Qx  #: x-coordinates of the voronoi points
        self.Qy = Qy  #: y-coordinates of the voronoi points 

        #: the sets of Bragg peaks present at each scan position
        self.braggpeak_labels = get_braggpeak_labels_by_scan_position(braggpeaks, Qx, Qy, max_dist)

        # Construct X matrix
        #: first dimension of the data matrix; the number of bragg peaks
        self.N_feat = len(self.Qx)
        #: second dimension of the data matrix; the number of scan positions
        self.N_meas = self.R_Nx*self.R_Ny

        self.X = np.zeros((self.N_feat,self.N_meas))  #: the data matrix
        for Rx in range(self.R_Nx):
            for Ry in range(self.R_Ny):
                R = Rx*self.R_Ny + Ry
                s = self.braggpeak_labels[Rx][Ry]
                pointlist = self.braggpeaks.get_pointlist(Rx,Ry)
                for i in s:
                    if X_is_boolean:
                        self.X[i,R] = True
                    else:
                        ind = np.argmin(np.hypot(pointlist.data['qx']-Qx[i],
                                                 pointlist.data['qy']-Qy[i]))
                        self.X[i,R] = pointlist.data['intensity'][ind]

        return

    def get_initial_classes_by_cooccurrence(self, thresh=0.3, BP_fraction_thresh=0.1,
                                                              max_iterations=200,
                                                              X_is_boolean=True,
                                                              n_corr_init=2):
        """
        Populate the initial classes by finding sets of Bragg peaks that tend to co-occur
        in the
        same diffraction patterns.

        Beginning from the sets of Bragg peaks labels for each scan position (determined
        in __init__), this method gets initial classes by determining which labels are
        most likely to co-occur with each other -- see get_initial_classes() docstring
        for more info.  Then the matrices W and H are generated -- see nmf() doscstring
        for discussion.

        Args:
            thresh (float in [0,1]): threshold for adding new BPs to a class
            BP_fraction_thresh (float in [0,1]): algorithm terminates if fewer than this
                fraction of the BPs have not been assigned to a class
            max_iterations (int): algorithm terminates after this many iterations
            n_corr_init (int): seed new classes by finding maxima of the n-point joint
                probability function.  Must be 2 or 3.
        """
        assert isinstance(X_is_boolean, bool)
        assert isinstance(max_iterations, (int,np.integer))
        assert n_corr_init in (2,3)

        # Get sets of integers representing the initial classes
        BP_sets = get_initial_classes(self.braggpeak_labels, N=len(self.Qx), thresh=thresh,
                                      BP_fraction_thresh=BP_fraction_thresh,
                                      max_iterations=max_iterations,
                                      n_corr_init=n_corr_init)

        # Construct W, H matrices
        self.N_c = len(BP_sets)

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

    def get_initial_classes_from_images(self, class_images):
        """
        Populate the initial classes using a set of user-defined class images.

        Args:
            class_images (ndarray): must have shape (R_Nx,R_Ny,N_c), where N_c is the
                number of classes, and class_images[:,:,i] is the image of class i.
        """
        assert class_images.shape[0]==self.R_Nx
        assert class_images.shape[1]==self.R_Ny

        # Construct W, H matrices
        self.N_c = class_images.shape[2]

        # H
        H = np.zeros((self.N_c,self.N_meas))
        for i in range(self.N_c):
            H[i,:] = class_images[:,:,i].ravel()
        self.H = np.copy(H, order='C')

        # W
        W = lstsq(self.H.T, self.X.T,rcond=None)[0].T
        W = np.where(W<0,0,W)
        self.W = np.copy(W, order='C')

        self.W_next = None
        self.H_next = None
        self.N_c_next = None

        return

    def nmf(self, max_iterations=1):
        """
        Nonnegative matrix factorization to refine the classes.

        The data matrix ``X`` is factored into two smaller matrices, ``W`` and ``H``::

            X = WH

        Here,

            * ``X``is the data matrix. It has shape (N_feat,N_meas), where N_feat is the
              number of Bragg peak integer labels (i.e. len(Qx)) and N_meas is the number
              of diffraction patterns (i.e. R_Nx*R_Ny).  Element X[i,j] represents the
              value of the i'th BP in the j'th DP.  The values depend on the flag
              datamatrix_is_boolean: if True, X[i,j] is 1 if this BP was present in this
              DP, or 0 if not; if False, X[i,j] is the cross correlation intensity of
              this BP in this DP.
            * ``W`` is the class matrix. It has shape (N_feat,N_c), where N_c is the
              number of classes. The i'th column vector, w_i = W[:,i], describes the
              weight of each Bragg peak in the i'th class.  w_i has length N_feat, and
              w_i[j] describes how strongly the j'th BP is associated with the i'th
              class.
            * ``H`` is the coefficient matrix. It has shape (N_c,N_meas).  The i'th
              column vector H[:,i] describes the contribution of each class to scan
              position i.

        Alternatively, we can completely equivalently think of H as a class matrix,
        and W as a coeffient matrix.  In this picture, the i'th row vector of H,
        h_i = H[i,:], describes the weight of each scan position in the i'th class.
        h_i has length N_meas, and h_i[j] describes how strongly the j'th scan
        position is associated with the i'th class.  The row vector W[i,:] is then
        a coefficient vector, which gives the contributions each of the (H) classes
        to the measured values of the i'th BP.  These pictures are related by a
        transpose: X = WH is equivalent to X.T = (H.T)(W.T).

        In nonnegative matrix factorization we impose the constrain, here on
        physical grounds, that all elements of X, W, and H should be nonnegative.

        The computation itself is performed using the sklearn nmf class. When this method
        is called, the three relevant matrices should already be defined. This method
        refines W and H, with up to max_iterations NMF steps.

        Args:
            max_iterations (int): the maximum number of NMF steps to take
        """
        sklearn_nmf = NMF(n_components=self.N_c, init='custom', max_iter=max_iterations)
        self.W_next = sklearn_nmf.fit_transform(self.X, W=self.W, H=self.H)
        self.H_next = sklearn_nmf.components_
        self.N_c_next = self.W_next.shape[1]

        return

    def split(self, sigma=2, threshold_split=0.25, expand_mask=1, minimum_pixels=1):
        """
        If any classes contain multiple non-contiguous segments in real space, divide
        these regions into distinct classes.

        Algorithm is as follows:
        First, an image of each class is obtained from its scan position weights.
        Then, the image is convolved with a gaussian of std sigma.
        This is then turned into a binary mask, by thresholding with threshold_split.
        Stray pixels are eliminated by performing a one pixel binary closing, then binary
        opening.
        The mask is then expanded by expand_mask pixels.
        Finally, the contiguous regions of the resulting mask are found. These become the
        new class components by scan position.

        The splitting itself involves creating two classes - i.e. adding a column to W
        and a row to H.  The new BP classes (W columns) have exactly the same values as
        the old BP class. The two new scan position classes (H rows) divide up the
        non-zero entries of the old scan position class into two or more non-intersecting
        subsets, each of which becomes its own new class.

        Args:
            sigma (float): std of gaussian kernel used to smooth the class images before
                thresholding and splitting.
            threshold_split (float): used to threshold the class image to create a binary mask.
            expand_mask (int): number of pixels by which to expand the mask before separating
                into contiguous regions.
            minimum_pixels (int): if, after splitting, a potential new class contains fewer than
                this number of pixels, ignore it
        """
        assert isinstance(expand_mask,(int,np.integer))
        assert isinstance(minimum_pixels,(int,np.integer))

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

                if np.sum(mask) >= minimum_pixels:

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

    def merge(self, threshBPs=0.1, threshScanPosition=0.1, return_params=True):
        """
        If any classes contain sufficient overlap in both scan positions and BPs, merge
        them into a single class.

        The algorithm is as follows:
        First, the Pearson correlation coefficient matrix is calculated for the classes
        according to both their diffraction space, Bragg peak representations (i.e. the
        correlations of the columns of W) and according to their real space, scan
        position representations (i.e. the correlations of the rows of H). Class pairs
        whose BP correlation coefficient exceeds threshBPs and whose scan position
        correlation coefficient exceed threshScanPosition are deemed 'sufficiently
        overlapped', and are marked as merge candidates. To account for intransitivity
        issues (e.g. class pairs 1/2 and 2/3 are merge candidates, but class pair 1/3 is
        not), merging is then performed beginning with candidate pairs with the greatest
        product of the two correlation coefficients, skipping later merge candidate pairs
        if one of the two classes has already been merged.

        The algorithm can be looped until no more merge candidates satisfying the
        specified thresholds remain with the merge_iterative method.

        The merging itself involves turning two classes into one by combining a pair of
        W columns (i.e. the BP representations of the classes) and the corresponding pair
        of H rows (i.e. the scan position representation of the class) into a single W
        column / H row. In terms of scan positions, the new row of H is generated by
        simply adding the two old H rows. In terms of Bragg peaks, the new column of W is
        generated by adding the two old columns of W, while weighting each by its total
        intensity in real space (i.e. the sum of its H row).

        Args:
            threshBPs (float): the threshold for the bragg peaks correlation coefficient,
                above which the two classes are considered candidates for merging
            threshScanPosition (float): the threshold for the scan position correlation
                coefficient, above which two classes are considered candidates for
                merging
            return_params (bool): if True, returns W_corr, H_corr, and merge_candidates.
                Otherwise, returns nothing. Incompatible with iterative=True.
        """

    def merge_by_class_index(self, i, j):
        """
        Merge classes i and j into a single class.

        Columns i and j of W  pair of W (i.e. the BP representations of the classes) and
        the corresponding pair of H rows (i.e. the scan position representation of the
        class) are mergedinto a single W column / H row. In terms of scan positions, the
        new row of H is generated by simply adding the two old H rows. In terms of Bragg
        peaks, the new column of W is generated by adding the two old columns of W, while
        weighting each by its total intensity in real space (i.e. the sum of its H row).

        Args:
            i (int): index of the first class to merge
            j (int): index of the second class to merge
        """
        assert np.all([isinstance(ind,(int,np.integer)) for ind in [i,j]]), "i and j must be ints"

        # Get merged class
        weight_i = np.sum(self.H[i,:])
        weight_j = np.sum(self.H[j,:])
        W_new = (self.W[:,i]*weight_i + self.W[:,j]*weight_j)/(weight_i+weight_j)
        H_new = self.H[i,:] + self.H[j,:]

        # Remove old classes and add in new class
        self.W_next = np.delete(self.W,j,axis=1)
        self.H_next = np.delete(self.H,j,axis=0)
        self.W_next[:,i] = W_new
        self.H_next[i,:] = H_new
        self.N_c_next = self.N_c-1

        return

    def split_by_class_index(self, i, sigma=2, threshold_split=0.25, expand_mask=1,
                                                                     minimum_pixels=1):
        """
        If class i contains multiple non-contiguous segments in real space, divide these
        regions into distinct classes.

        Algorithm is as described in the docstring for self.split.

        Args:
            i (int): index of the class to split
            sigma (float): std of gaussian kernel used to smooth the class images before
                thresholding and splitting.
            threshold_split (float): used to threshold the class image to create a binary
                mask.
            expand_mask (int): number of pixels by which to expand the mask before
                separating into contiguous regions.
            minimum_pixels (int): if, after splitting, a potential new class contains
                fewer than this number of pixels, ignore it
        """
        assert isinstance(i,(int,np.integer))
        assert isinstance(expand_mask,(int,np.integer))
        assert isinstance(minimum_pixels,(int,np.integer))
        W_next = np.zeros((self.N_feat,1))
        H_next = np.zeros((1,self.N_meas))

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

            if np.sum(mask) >= minimum_pixels:

                # Leave the Bragg peak weightings the same
                W_next = np.hstack((W_next,self.W[:,i,np.newaxis]))

                # Use the existing real space pixel weightings
                h_i = np.zeros(self.N_meas)
                h_i[mask.ravel()] = self.H[i,:][mask.ravel()]
                H_next = np.vstack((H_next,h_i[np.newaxis,:]))

        W_prev = np.delete(self.W,i,axis=1)
        H_prev = np.delete(self.H,i,axis=0)
        self.W_next = np.concatenate((W_next[:,1:],W_prev),axis=1)
        self.H_next = np.concatenate((H_next[1:,:],H_prev),axis=0)
        self.N_c_next = self.W_next.shape[1]

        return

    def remove_class(self, i):
        """
        Remove class i.

        Args:
            i (int): index of the class to remove
        """
        assert isinstance(i,(int,np.integer))

        self.W_next = np.delete(self.W,i,axis=1)
        self.H_next = np.delete(self.H,i,axis=0)
        self.N_c_next = self.W_next.shape[1]

        return

    def merge_iterative(self, threshBPs=0.1, threshScanPosition=0.1):
        """
        If any classes contain sufficient overlap in both scan positions and BPs, merge
        them into a single class.

        Identical to the merge method, with the addition of iterating until no new merge
        pairs are found.

        Args:
            threshBPs (float): the threshold for the bragg peaks correlation coefficient,
                above which the two classes are considered candidates for merging
            threshScanPosition (float): the threshold for the scan position correlation
                coefficient, above which two classes are considered candidates for
                merging
        """
        proceed = True
        W_ = np.copy(self.W)
        H_ = np.copy(self.H)
        Nc_ = W_.shape[1]

        while proceed:

            # Get correlation coefficients
            W_corr = np.corrcoef(W_.T)
            H_corr = np.corrcoef(H_)

            # Get merge candidate pairs
            mask_BPs = W_corr > threshBPs
            mask_ScanPosition = H_corr > threshScanPosition
            mask_upperright = np.zeros((Nc_,Nc_),dtype=bool)
            for i in range(Nc_):
                mask_upperright[i,i+1:] = 1
            merge_mask = mask_BPs * mask_ScanPosition * mask_upperright
            merge_i,merge_j = np.nonzero(merge_mask)

            # Sort merge candidate pairs
            merge_candidates = np.zeros(len(merge_i),dtype=[('i',int),('j',int),('cc_w',float),
                                                            ('cc_h',float),('score',float)])
            merge_candidates['i'] = merge_i
            merge_candidates['j'] = merge_j
            merge_candidates['cc_w'] = W_corr[merge_i,merge_j]
            merge_candidates['cc_h'] = H_corr[merge_i,merge_j]
            merge_candidates['score'] = W_corr[merge_i,merge_j]*H_corr[merge_i,merge_j]
            merge_candidates = np.sort(merge_candidates,order='score')[::-1]

            # Perform merging
            merged = np.zeros(Nc_,dtype=bool)
            W_merge = np.zeros((self.N_feat,1))
            H_merge = np.zeros((1,self.N_meas))
            for index in range(len(merge_candidates)):
                i = merge_candidates['i'][index]
                j = merge_candidates['j'][index]
                if not (merged[i] or merged[j]):
                    weight_i = np.sum(H_[i,:])
                    weight_j = np.sum(H_[j,:])
                    W_new = (W_[:,i]*weight_i + W_[:,j]*weight_j)/(weight_i+weight_j)
                    H_new = H_[i,:] + H_[j,:]
                    W_merge = np.hstack((W_merge,W_new[:,np.newaxis]))
                    H_merge = np.vstack((H_merge,H_new[np.newaxis,:]))
                merged[i] = True
                merged[j] = True
            W_merge = W_merge[:,1:]
            H_merge = H_merge[1:,:]

            W_ = np.hstack((W_[:,merged==False],W_merge))
            H_ = np.vstack((H_[merged==False,:],H_merge))
            Nc_ = W_.shape[1]

            if len(merge_candidates)==0:
                proceed = False

        self.W_next = W_
        self.H_next = H_
        self.N_c_next = self.W_next.shape[1]

        return

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

        Args:
            i (int): the class index

        Returns:
            (2-tuple): A 2-tuple containing:

                * **class_BPs**: *(length N_feat array of floats)* the weights of the
                  N_feat Bragg peaks for this class
                * **class_image**: *(shape (R_Nx,R_Ny) array of floats)* the weights of
                  each scan position in this class
        """
        class_BPs = self.W[:,i]
        class_image = self.H[i,:].reshape((self.R_Nx,self.R_Ny))
        return class_BPs, class_image

    def get_class_BPs(self, i):
        """
        Get a single class, returning its BP weights.

        Args:
            i (int): the class index

        Returns:
            (length N_feat array of floats): the weights of the N_feat Bragg peaks for
            this class
        """
        return self.W[:,i]

    def get_class_image(self, i):
        """
        Get a single class, returning its scan position weights.

        Args:
            i (int): the class index

        Returns:
            (shape (R_Nx,R_Ny) array of floats): the weights of each scan position in
            this class
        """
        return self.H[i,:].reshape((self.R_Nx,self.R_Ny))

    def get_candidate_class(self, i):
        """
        Get a single candidate class, returning both its BP weights and scan position weights.

        Args:
            i           (int) the class index

        Returns:
            (2-tuple): A 2-tuple containing:

                * **class_BPs**: *(length N_feat array of floats)* the weights of the
                  N_feat Bragg peaks for this class
                * **class_image**: *(shape (R_Nx,R_Ny) array of floats)* the weights of
                  each scan position in this class
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

        Args:
            i (int): the class index

        Returns:
            (shape (R_Nx,R_Ny) array of floats): the weights of each scan position in
            this class
        """
        assert self.H_next is not None, "H_next is not assigned."

        return self.H_next[i,:].reshape((self.R_Nx,self.R_Ny))


### Functions for initial class determination ###

def get_braggpeak_labels_by_scan_position(braggpeaks, Qx, Qy, max_dist=None):
    """
    For each scan position, gets a set of integers, specifying the bragg peaks at this
    scan position.

    From a set of positions in diffraction space (Qx,Qy), assign each detected bragg peak
    in the PointListArray braggpeaks a label corresponding to the index of the closest
    position; thus for a bragg peak at (qx,qy), if the closest position in (Qx,Qy) is
    (Qx[i],Qy[i]), assign this peak the label i. This is equivalent to assigning each
    bragg peak (qx,qy) a label according to the Voronoi region it lives in, given a
    voronoi tesselation seeded from the points (Qx,Qy).

    For each scan position, get the set of all indices i for all bragg peaks found at
    this scan position.

    Args:
        braggpeaks (PointListArray): Bragg peaks; must have coords 'qx' and 'qy'
        Qx (ndarray of floats): x-coords of the voronoi points
        Qy (ndarray of floats): y-coords of the voronoi points
        max_dist (None or number): maximum distance from a given voronoi point a peak
            can be and still be associated with this label

    Returns:
        (list of lists of sets) the labels found at each scan position. Scan position
        (Rx,Ry) is accessed via braggpeak_labels[Rx][Ry]
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
                if max_dist is not None:
                    if np.hypot(Qx[label]-pointlist.data['qx'][i],Qy[label]-pointlist.data['qy'][i]) < max_dist:
                        s.add(label)
                else:
                    s.add(label)

    return braggpeak_labels


def get_initial_classes(braggpeak_labels, N, thresh=0.3, BP_fraction_thresh=0.1,
                        max_iterations=200, n_corr_init=2):
    """
    From the sets of Bragg peaks present at each scan position, get an initial guess
    classes at which Bragg peaks should be grouped together into classes.

    The algorithm is as follows:
    1. Calculate an n-point correlation function, i.e. the joint probability of any given
    n BPs coexisting in a diffraction pattern.  n is controlled by n_corr_init, and must
    be 2 or 3. peaks i, j, and k are all in the same DP.
    2. Find the BP triplet maximizing the 3-point function; include these three BPs in a
    class.
    3. Get all DPs containing the class BPs. From these, find the next most likely BP to
    also be present.  If its probability of coexisting with the known class BPs is
    greater than thresh, add it to the class and repeat this step. Otherwise, proceed to
    the next step.
    4. Check: if the new class is the same as a class that has already been found, OR if
    the fraction of BPs which have not yet been placed in a class is less than
    BP_fraction_thresh, or more than max_iterations have been attempted, finish,
    returning all classes. Otherwise, set all slices of the 3-point function containing
    the BPs in the new class to zero, and begin a new iteration, starting at step 2 using
    the new, altered 3-point function.

    Args:
        N (int): the total number of indexed Bragg peaks in the 4D-STEM dataset
        braggpeak_labels (list of lists of sets): the Bragg peak labels found at each
            scan position; see get_braggpeak_labels_by_scan_position().
        thresh (float in [0,1]): threshold for adding new BPs to a class
        BP_fraction_thresh (float in [0,1]): algorithm terminates if fewer than this
            fraction of the BPs have not been assigned to a class
        max_iterations (int): algorithm terminates after this many iterations
        n_corr_init (int): seed new classes by finding maxima of the n-point joint
            probability function.  Must be 2 or 3.

    Returns:
        (list of sets): the sets of Bragg peaks constituting the classes
    """
    assert isinstance(braggpeak_labels[0][0],set)
    assert thresh >= 0 and thresh <= 1
    assert BP_fraction_thresh >= 0 and BP_fraction_thresh <= 1
    assert isinstance(max_iterations,(int,np.integer))
    assert n_corr_init in (2,3)
    R_Nx = len(braggpeak_labels)
    R_Ny = len(braggpeak_labels[0])

    if n_corr_init == 2:
        # Get two-point function
        n_point_function = np.zeros((N,N))
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                s = braggpeak_labels[Rx][Ry]
                perms = permutations(s,2)
                for perm in perms:
                    n_point_function[perm[0],perm[1]] += 1
        n_point_function /= R_Nx*R_Ny

        # Main loop
        BP_sets = []
        iteration = 0
        unused_BPs = np.ones(N,dtype=bool)
        seed_new_class = True
        while seed_new_class:
            ind1,ind2 = np.unravel_index(np.argmax(n_point_function),(N,N))
            BP_set = set([ind1,ind2])
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

            # Modify 2-point function, add new BP set to list, and decide to continue or stop
            for i in BP_set:
                n_point_function[i,:] = 0
                n_point_function[:,i] = 0
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

    else:
        # Get three-point function
        n_point_function = np.zeros((N,N,N))
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                s = braggpeak_labels[Rx][Ry]
                perms = permutations(s,3)
                for perm in perms:
                    n_point_function[perm[0],perm[1],perm[2]] += 1
        n_point_function /= R_Nx*R_Ny

        # Main loop
        BP_sets = []
        iteration = 0
        unused_BPs = np.ones(N,dtype=bool)
        seed_new_class = True
        while seed_new_class:
            ind1,ind2,ind3 = np.unravel_index(np.argmax(n_point_function),(N,N,N))
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
                n_point_function[i,:,:] = 0
                n_point_function[:,i,:] = 0
                n_point_function[:,:,i] = 0
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









