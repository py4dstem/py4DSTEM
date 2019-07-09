# A class, polar_elliptical_transform, which finds the circular or elliptical parametrization of
# a dataset which best aligns it to a diffraction space image.

import numpy as np
from py4DSTEM.process.utils import print_progress_bar

class polar_elliptical_transform(object):
    """
    This class facilitates coordinate transformation from Cartesian coordinates to polar elliptical
    coordinates.  The notation / parametrization here is:

        x = x0 + A*r*cos(theta)*cos(phi) + B*r*sin(theta)*sin(phi)
        y = y0 + B*r*sin(theta)*sin(phi) - A*r*cos(theta)*cos(phi)

    where (x,y) is a point in the cartesian plane, (r,theta) is the corresponding point in the
    polar elliptical cooridanate system, and is (x0,y0,A,B,phi) are parameters describing the
    polar elliptical cooridinates.  In particular, (x0,y0) give the origin, (A,B) give the semi-axis
    lengths, and phi gives the tilt of semiaxis A with respect to the x-axis.  For A=B, the
    (r,theta) parameterization describes a polar coordinate system.

    The parameteris (x0,y0,A,B,phi) are stored as elements of a length-5 array, in this order, at
    self.coefs.  theta is passed in degrees, phi is in radians (i'm sorry (0_0,)...)

    Examples
    --------
    For two arrays (ar,mask) representing an image and a boolean mask:

        >>> pet = polar_elliptical_transform(calibration_image=ar, mask=mask)
        >>> pet.get_polar_transform()

    will generate and store the polar transformation of ar in an array at pet.polar_ar.  Note that
    by default this is now a polar transform with A=B=1, C=0, and (x0,y0) at the center of ar.
    The parameters of the transformation can be refined against elliptical features of the
    calibration image ar -- appropriate data includes a single DP with amorphous rings, a position
    averaged DP over many nanocrystalline grains, etc. with -- by invoking:

        >>> pet.fit_origin()        # Refines x0,y0
        >>> pet.fit_ellipticity()   # Refines x0,y0,A,B,phi

    and the polar transform can then be recalculated with

        >>> pet.get_polar_transform()
    """

    def __init__(self, calibration_image, mask=None, x0_init=None, y0_init=None,
                                                                   dr=1, dtheta=2, r_range=500):
        """
        Initializes a polar_elliptical_trasform_instance.

        Accepts:
            calibration_image   (2D float array) image used to calibrate elliptical distoration
            mask                (2D bool array) False pixels to ignore / mask
            x0_init             (float) initial center x coord. If None, initialize at center
            y0_init             (float) initial center y coord. If None, initialize at center
            dr                  (float) radial bin size
            dtheta              (float) angular bin size, in degrees
            r_range             (float or list/tuple of two floats) the min/max r values.
                                If r_range is a number, it is taken as max and min is set to 0
        """
        # Setup transformation parameters
        self.calibration_image = calibration_image
        if mask is None:
            self.mask = np.ones_like(calibration_image, dtype=bool)
        else:
            self.mask = mask
        self.dr = dr
        self.dtheta = np.radians(dtheta)  # blasphemy!
        try:
            self.r_range = [r_range[0],r_range[1]]
        except TypeError:
            self.r_range = [0,r_range]

        # Cartesian coordinates
        self.Nx = calibration_image.shape[0]
        self.Ny = calibration_image.shape[1]
        self.yy,self.xx = np.meshgrid(np.arange(self.Ny),np.arange(self.Nx))

        # Polar coordinates
        r_bins = np.arange(self.r_range[0]+self.dr/2.,self.r_range[1]+self.dr/2.,self.dr) # values
        t_bins = np.arange(-np.pi+self.dtheta/2.,np.pi+self.dtheta/2.,self.dtheta) # are bin centers
        self.Nr, self.Nt = len(r_bins),len(t_bins)
        self.rr, self.tt = np.meshgrid(r_bins, t_bins)

        # Initial parameters
        self.coefs = np.zeros(5)
        if x0_init is not None:
            self.coefs[0] = x0_init
        else:
            self.coefs[0] = 0.5*self.Nx
        if y0_init is not None:
            self.coefs[1] = y0_init
        else:
            self.coefs[1] = 0.5*self.Ny
        self.coefs[2] = 1
        self.coefs[3] = 1
        self.coefs[4] = 0

        return

    def get_polar_transform(self, ar=None, mask=None, return_ans=False):
        """
        Get the polar transformation of an array ar, or if ar is None, of self.calibration_image.

        Accepts:
            ar         (2D array or None) the array to transform; if None, use self.calibration_image
            mask       (2D array or None) mask indicating pixels to ignore; if None, use self.mask
            return_ans (bool) if True, return ar and mask

        Returns (if return_ars is True):
            polar_ar   (2D array) the polar transformation of ar
            polar_mask (2D array) the polar transformation of mask
        """
        if ar is None:
            ar = self.calibration_image
        if mask is None:
            mask = self.mask

        # Define coordinate system
        xr = self.rr * np.cos(self.tt)
        yr = self.rr * np.sin(self.tt)
        x = self.coefs[0] + xr*self.coefs[2]*np.cos(self.coefs[4]) - \
                            yr*self.coefs[3]*np.sin(self.coefs[4])
        y = self.coefs[1] + yr*self.coefs[3]*np.cos(self.coefs[4]) + \
                            xr*self.coefs[2]*np.sin(self.coefs[4])

        # To map between (x,y) and (r,theta), we've specified the set of (r,theta) coordinates
        # we'd like in meshgrid arrays (self.rr,self.tt), then converted those values
        # into two arrays (x,y) of the same shape - i.e. shape (Nr,Nt) - so that x[r0,theta0]
        # is some floating point number representing value of the cartesian abscissa for the
        # point at (r0,theta0). Values of self.polar_ar are then determined by bilinear
        # interpolation, i.e. using a wieghted sum of the values of the four pixels 
        # ar[x_,y_], ar[x_+1,y_], ar[x_,y_+1], ar[x_+1,y_+1]
        # The next few blocks of code prepare the data for bilinear interpolation

        transform_mask = (x>0)*(y>0)*(x<self.Nx-1)*(y<self.Ny-1)
        xF = np.floor(x[transform_mask])
        yF = np.floor(y[transform_mask])
        dx = x[transform_mask] - xF
        dy = y[transform_mask] - yF

        x_inds = np.vstack((xF,xF+1,xF  ,xF+1)).astype(int)
        y_inds = np.vstack((yF,yF,  yF+1,yF+1)).astype(int)
        weights = np.vstack(((1-dx)*(1-dy),
                             (  dx)*(1-dy),
                             (1-dx)*(  dy),
                             (  dx)*(  dy)))

        transform_mask = transform_mask.ravel()
        self.polar_ar = np.zeros(np.prod(self.rr.shape))        # Bilinear interpolation happens
        self.polar_ar[transform_mask] = np.sum(ar[x_inds,y_inds]*weights,axis=0) #    <-----here 
        self.polar_ar = np.reshape(self.polar_ar,self.rr.shape)

        self.polar_mask = np.zeros(np.prod(self.rr.shape))
        self.polar_mask[transform_mask] = np.sum(mask[x_inds,y_inds]*weights,axis=0)
        self.polar_mask = np.reshape(self.polar_mask,self.rr.shape)

        if return_ans:
            return self.polar_ar, self.polar_mask
        else:
            return

    def get_polar_score(self, return_ans=False):
        """
        Get the score - the RMSD divided by the mean along the theta direction - for self.polar_ar
        while masking off self.polar_mask.

        Accepts:
            return_ans  (bool) if True, return score, RMSD, and mean

        Returns (if return_ans is True):
            score       (float) the score
            RMSD        (1D array) the RMSD of self.polar_ar[self.polar_mask] along theta
            mean        (1D array) the mean of self.polar_ar[self.polar_mask] along theta
        """
        # Mean along theta
        self.mean = np.sum(self.polar_ar*self.polar_mask, axis=0)
        mask_sum = np.sum(self.polar_mask, axis=0)
        divmask = mask_sum > 1e-2
        self.mean[divmask] = self.mean[divmask] / mask_sum[divmask]
        self.mean[divmask==False] = 0

        # RMS along theta
        self.RMSD = (self.polar_ar - np.tile(self.mean,(self.Nt,1)))**2
        self.RMSD = np.sum(self.RMSD, axis=0)
        self.RMSD[divmask] = np.sqrt(self.RMSD[divmask] / self.mean[divmask])
        self.RMSD[divmask==False] = 0

        # Score
        self.score = np.sum(self.RMSD[divmask] / self.mean[divmask]) / self.Nr

        if return_ans:
            return self.score, self.RMSD, self.mean
        else:
            return

    def fit_params(self, n_iter, step_sizes_init=[0.1,0.1,0.1,0.01,0.01], step_scale=0.9,
                                                                        return_ans=False):
        """
        Find the polar elliptical transformation parameters x0,y0,A,B,C which best describe the data
        by minimizing a cost function (theta-integrated RMSD of the polar transform).

        Accepts:
            n_iter              (int) number of iterations
            step_sizes_init     (length 2 list of numbers) initial step sizes [dx, dy] by which to
                                move the polar origin
            step_scale          (float <= 1) if no step is taken, reduce step size by this fraction
            return_ans          (bool) if True, return x0, y0, scores, x0_vals, y0_vals, A_vals,
                                and B_vals

        Returns (if return_ans is True):
            x0                  (float) x0
            y0                  (float) y0
            scores              (array) scores at each iteration
            x0_vals             (array) x0 values at each iteration
            y0_vals             (array) y0 values at each iteration
            A_vals              (array) A values at each iteration
            B_vals              (array) B values at each iteration
            C_vals              (array) C values at each iteration
        """
        scores, x0_vals, y0_vals, A_vals, B_vals, C_vals = [],[],[],[],[],[]

        # Initial step sizes
        step_sizes = np.zeros(5)
        try:
            step_sizes[0] = step_sizes_init[0]
            step_sizes[1] = step_sizes_init[1]
            step_sizes[2] = step_sizes_init[2]
            step_sizes[3] = step_sizes_init[3]
            step_sizes[4] = step_sizes_init[4]
        except TypeError:
            raise Exception("step_sizes_init should be a length 5 array/list/tuple, giving the initial step sizes of (x0,y0,A,B,C)")
        coef_inds = np.nonzero(step_sizes)[0]   # Don't iterate over coefs with step size = 0

        # Initial polar transform and score
        self.get_polar_transform()
        score,_,_ = self.get_polar_score(return_ans=True)
        scores.append(score)
        x0_vals.append(self.coefs[0])
        y0_vals.append(self.coefs[1])
        A_vals.append(self.coefs[2])
        B_vals.append(self.coefs[3])
        C_vals.append(self.coefs[4])

        # Main loop
        for i in range(n_iter):

            # Loop over x0,y0, testing and accepting/rejecting steps for each
            # based on polar transform score values
            n_steps = 0
            for j in coef_inds:
                # Test new coefficient and update
                self.coefs[j] += step_sizes[j]
                self.get_polar_transform()
                test_score,_,_ = self.get_polar_score(return_ans=True)

                if test_score < score:
                    score = test_score
                    n_steps += 1
                else:
                    self.coefs[j] -= 2*step_sizes[j]
                    self.get_polar_transform()
                    test_score,_,_ = self.get_polar_score(return_ans=True)
                    if test_score < score:
                        score = test_score
                        n_steps += 1
                    else:
                        self.coefs[j] += step_sizes[j]
                        self.get_polar_transform()

            # If neither x0 nor y0 was updated, reduce the step size
            if n_steps == 0:
                step_sizes = step_sizes*step_scale

            scores.append(score)
            x0_vals.append(self.coefs[0])
            y0_vals.append(self.coefs[1])
            print_progress_bar(i+1, n_iter, prefix='Analyzing:', suffix='Complete', length=50)

        if return_ans:
            return np.array(scores), np.array(x0_vals), np.array(y0_vals), np.array(A_vals), np.array(B_vals), np.array(C_vals)
        else:
            return

    def fit_origin(self, n_iter, step_sizes_init=[1,1], step_scale=0.9, return_ans=False):
        """
        Find the origin of polar coordinates which best describe the data
        by minimizing a cost function (theta-integrated RMSD of the polar transform).

        Accepts:
            n_iter              (int) number of iterations
            step_sizes_init     (length 2 list of numbers) initial step sizes [dx, dy] by which to
                                move the polar origin
            step_scale          (float <= 1) if no step is taken, reduce step size by this fraction
            return_ans          (bool) if True, return x0,y0,scores,x0_vals, and y0_vals

        Returns (if return_ans is True):
            x0                  (float) x0
            y0                  (float) y0
            scores              (array) scores at each iteration
            x0_vals             (array) x0 values at each iteration
            y0_vals             (array) y0 values at each iteration
        """
        # Initial step sizes
        step_sizes = np.zeros(2)
        try:
            step_sizes[0] = step_sizes_init[0]
            step_sizes[1] = step_sizes_init[1]
        except TypeError:
            raise Exception("step_sizes_init should be a length 2 array/list/tuple, giving the initial step sizes of x0 and y0, respectively")

        # Call fit_params
        if return_ans:
            scores,x0_vals,y0_vals,_,_,_ = self.fit_params(n_iter=n_iter,
                                               step_sizes_init=[step_sizes[0],step_sizes[1],0,0,0],
                                               step_scale=0.9,
                                               return_ans=True)
            return scores, x0_vals, y0_vals
        else:
            self.fit_params(n_iter=n_iter,
                            step_sizes_init=[step_sizes[0],step_size[1],0,0,0],
                            step_scale=0.9,
                            return_ans=False)
            return




