# A class, polar_elliptical_transform, which finds the circular or elliptical parametrization of
# a dataset which best aligns it to a diffraction space image.

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from py4DSTEM.process.utils import print_progress_bar
from scipy.signal import convolve2d


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

        >>> pet.fit_origin(n_iter=100)        # Refines x0,y0
        >>> pet.fit_ellipticity(n_iter=100)   # Refines x0,y0,A,B,phi

    and the polar transform can then be recalculated with

        >>> pet.get_polar_transform()
    """

    def __init__(
        self,
        calibration_image,
        mask=None,
        x0_init=None,
        y0_init=None,
        dr=1,
        dtheta=2,
        r_range=500,
    ):
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
            self.r_range = [r_range[0], r_range[1]]
        except TypeError:
            self.r_range = [0, r_range]

        # Cartesian coordinates
        self.Nx = calibration_image.shape[0]
        self.Ny = calibration_image.shape[1]
        self.yy, self.xx = np.meshgrid(np.arange(self.Ny), np.arange(self.Nx))

        # Polar coordinates
        r_bins = np.arange(
            self.r_range[0] + self.dr / 2.0, self.r_range[1] + self.dr / 2.0, self.dr
        )  # values
        t_bins = np.arange(
            -np.pi + self.dtheta / 2.0, np.pi + self.dtheta / 2.0, self.dtheta
        )  # are bin centers
        self.Nr, self.Nt = len(r_bins), len(t_bins)
        self.rr, self.tt = np.meshgrid(r_bins, t_bins)

        # Initial parameters
        self.coefs = np.zeros(5)
        if x0_init is not None:
            self.coefs[0] = x0_init
        else:
            self.coefs[0] = 0.5 * self.Nx
        if y0_init is not None:
            self.coefs[1] = y0_init
        else:
            self.coefs[1] = 0.5 * self.Ny
        self.coefs[2] = 1
        self.coefs[3] = 1
        self.coefs[4] = 0

        return

    def get_polar_transform(self, ar=None, mask=None, return_ans=False):
        """
        Get the polar transformation of an array ar, or if ar is None, of self.calibration_image.

        Note that the flag return_ans controls two things: first, if the outputs of the method
        are returned, and second, if the outputs of the method are stored in self.polar_ar and
        self.polar_mask.  If return_ans is True, the outputs are returned and self.polar_ar and
        self.polar_mask are NOT updated.  If return_ans is False, the outputs are not returned
        and self.polar_ar and self.polar_mask ARE updated.

        Accepts:
            ar         (2D array or None) the array to transform; if None, use self.calibration_image
            mask       (2D array or None) mask indicating pixels to ignore; if None, use self.mask
            return_ans (bool) if True, return ar and mask. If False, instead update self.polar_ar
                       and self.polar_mask.

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
        x = (
            self.coefs[0]
            + xr * self.coefs[2] * np.cos(self.coefs[4])
            - yr * self.coefs[3] * np.sin(self.coefs[4])
        )
        y = (
            self.coefs[1]
            + yr * self.coefs[3] * np.cos(self.coefs[4])
            + xr * self.coefs[2] * np.sin(self.coefs[4])
        )

        # To map between (x,y) and (r,theta), we've specified the set of (r,theta) coordinates
        # we'd like in meshgrid arrays (self.rr,self.tt), then converted those values
        # into two arrays (x,y) of the same shape - i.e. shape (Nr,Nt) - so that x[r0,theta0]
        # is some floating point number representing value of the cartesian abscissa for the
        # point at (r0,theta0). Values of self.polar_ar are then determined by bilinear
        # interpolation, i.e. using a wieghted sum of the values of the four pixels
        # ar[x_,y_], ar[x_+1,y_], ar[x_,y_+1], ar[x_+1,y_+1]
        # The next few blocks of code prepare the data for bilinear interpolation

        transform_mask = (x > 0) * (y > 0) * (x < self.Nx - 1) * (y < self.Ny - 1)
        xF = np.floor(x[transform_mask])
        yF = np.floor(y[transform_mask])
        dx = x[transform_mask] - xF
        dy = y[transform_mask] - yF

        x_inds = np.vstack((xF, xF + 1, xF, xF + 1)).astype(int)
        y_inds = np.vstack((yF, yF, yF + 1, yF + 1)).astype(int)
        weights = np.vstack(
            ((1 - dx) * (1 - dy), (dx) * (1 - dy), (1 - dx) * (dy), (dx) * (dy))
        )

        transform_mask = transform_mask.ravel()
        polar_ar = np.zeros(np.prod(self.rr.shape))  # Bilinear interpolation happens
        polar_ar[transform_mask] = np.sum(
            ar[x_inds, y_inds] * weights, axis=0
        )  #    <-----here
        polar_ar = np.reshape(polar_ar, self.rr.shape)

        polar_mask = np.zeros(np.prod(self.rr.shape))
        polar_mask[transform_mask] = np.sum(mask[x_inds, y_inds] * weights, axis=0)
        polar_mask = np.reshape(polar_mask, self.rr.shape)

        if return_ans:
            return polar_ar, polar_mask
        else:
            self.polar_ar, self.polar_mask = polar_ar, polar_mask
            return

    def get_polar_transform_two_sided_gaussian(
        self, ar=None, mask=None, r_sigma=0.1, t_sigma=0.1, return_ans=False, coef=None
    ):
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

        ar = ar * mask
        if coef is None:
            coef = self.coef_opt

        # Define coordinate system
        ya, xa = self.yy, self.xx
        xa = xa - coef[6]
        ya = ya - coef[7]

        # Convert general ellipse coefficients into canonical form
        phi = 0
        if np.abs(coef[8]) > 0:
            phi = -1 * np.arctan(
                (1 - coef[9] + np.sqrt((coef[9] - 1) ** 2 + coef[8] ** 2)) / coef[8]
            )

        a0 = np.sqrt(
            2
            * (1 + coef[9] + np.sqrt((coef[9] - 1) ** 2 + coef[8] ** 2))
            / (4 * coef[9] - coef[8] ** 2)
        )
        b0 = np.sqrt(
            2
            * (1 + coef[9] - np.sqrt((coef[9] - 1) ** 2 + coef[8] ** 2))
            / (4 * coef[9] - coef[8] ** 2)
        )
        ratio = b0 / a0
        m = np.asarray(
            [
                [
                    ratio * np.cos(phi) ** 2 + np.sin(phi) ** 2,
                    (ratio - 1) * np.cos(phi) * np.sin(phi),
                ],
                [
                    (ratio - 1) * np.cos(phi) * np.sin(phi),
                    np.cos(phi) ** 2 + ratio * np.sin(phi) ** 2,
                ],
            ]
        )

        # arrays of polar elliptical coordinates
        ta = np.arctan2(m[1, 0] * xa + m[1, 1] * ya, m[0, 0] * xa + m[0, 1] * ya)
        ra = (
            np.sqrt(xa ** 2 + coef[8] * xa * ya + coef[9] * ya ** 2) * b0
        )  # normalize by b0 because major axis is retained in projection

        # resample coordinates
        r_ind = np.round((ra - self.r_range[0]) / self.dr).astype(dtype=int)
        t_ind = np.mod(np.round((ta / self.dtheta)), self.Nt).astype(dtype=int)
        sub = np.logical_and(r_ind < self.Nr, r_ind >= 1)
        rt_inds = np.transpose(np.stack((t_ind[sub], r_ind[sub])))

        # and set up KDE kernels
        sr = r_sigma / self.dr
        vr = np.arange(-np.ceil(4 * sr), np.ceil(4 * sr) + 1, 1)
        kr = np.exp(-vr ** 2 / (2 * sr ** 2))
        kr = np.expand_dims(kr, axis=0)

        dt = self.dtheta * 180 / np.pi
        st = t_sigma / dt
        vt = np.arange(-np.ceil(4 * st), np.ceil(4 * st) + 1, 1)
        kt = np.exp(-vt ** 2 / (2 * st ** 2))
        kt = np.expand_dims(kt, axis=1)

        # convolve by kr
        kNorm = convolve2d(np.ones((self.Nt, self.Nr)), kr, mode="same")

        # convolve by kt
        kNorm = convolve2d(kNorm, kt, mode="same", boundary="wrap")

        # normalize
        kNorm = np.divide(1, kNorm, where=kNorm != 0)

        # get polar array and convolve with KDE kernels, then normalize
        polarNorm = accumarray(rt_inds, np.ones((np.sum(sub))), (self.Nt, self.Nr))
        polarNorm = convolve2d(polarNorm, kr, mode="same")
        polarNorm = convolve2d(polarNorm, kt, mode="same", boundary="wrap")
        polarNorm = kNorm * polarNorm

        polarMask = accumarray(rt_inds, mask[sub], (self.Nt, self.Nr))
        polarMask = convolve2d(polarMask, kr, mode="same")
        polarMask = convolve2d(polarMask, kt, mode="same", boundary="wrap")
        self.polar_mask = kNorm * polarMask

        polarCBED = accumarray(rt_inds, ar[sub], (self.Nt, self.Nr))
        polarCBED = convolve2d(polarCBED, kr, mode="same")
        polarCBED = convolve2d(polarCBED, kt, mode="same", boundary="wrap")
        polarCBED = kNorm * polarCBED
        polar_ar = np.divide(polarCBED, polarNorm, where=polarNorm != 0)
        self.polar_ar = polar_ar

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
        self.mean = np.sum(self.polar_ar * self.polar_mask, axis=0)
        mask_sum = np.sum(self.polar_mask, axis=0)
        divmask = mask_sum > 1e-2
        self.mean[divmask] = self.mean[divmask] / mask_sum[divmask]
        self.mean[divmask == False] = 0

        # RMS along theta
        self.RMSD = (self.polar_ar - np.tile(self.mean, (self.Nt, 1))) ** 2
        self.RMSD = np.sum(self.RMSD, axis=0)
        divmask = self.mean > 1e-10
        self.RMSD[divmask] = np.sqrt(self.RMSD[divmask] / self.mean[divmask])
        self.RMSD[divmask == False] = 0

        # Score
        self.score = np.sum(self.RMSD[divmask] / self.mean[divmask]) / self.Nr

        if return_ans:
            return self.score, self.RMSD, self.mean
        else:
            return

    def fit_params(
        self,
        n_iter,
        step_sizes_init=[0.1, 0.1, 0.1, 0.01, 0.01],
        step_scale=0.9,
        return_ans=False,
    ):
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
        scores, x0_vals, y0_vals, A_vals, B_vals, C_vals = [], [], [], [], [], []

        # Initial step sizes
        step_sizes = np.zeros(5)
        try:
            step_sizes[0] = step_sizes_init[0]
            step_sizes[1] = step_sizes_init[1]
            step_sizes[2] = step_sizes_init[2]
            step_sizes[3] = step_sizes_init[3]
            step_sizes[4] = step_sizes_init[4]
        except TypeError:
            raise Exception(
                "step_sizes_init should be a length 5 array/list/tuple, giving the initial step sizes of (x0,y0,A,B,C)"
            )
        coef_inds = np.nonzero(step_sizes)[
            0
        ]  # Don't iterate over coefs with step size = 0

        # Initial polar transform and score
        self.get_polar_transform()
        score, _, _ = self.get_polar_score(return_ans=True)
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
                test_score, _, _ = self.get_polar_score(return_ans=True)

                if test_score < score:
                    score = test_score
                    n_steps += 1
                else:
                    self.coefs[j] -= 2 * step_sizes[j]
                    self.get_polar_transform()
                    test_score, _, _ = self.get_polar_score(return_ans=True)
                    if test_score < score:
                        score = test_score
                        n_steps += 1
                    else:
                        self.coefs[j] += step_sizes[j]
                        self.get_polar_transform()

            # If neither x0 nor y0 was updated, reduce the step size
            if n_steps == 0:
                step_sizes = step_sizes * step_scale

            scores.append(score)
            x0_vals.append(self.coefs[0])
            y0_vals.append(self.coefs[1])
            print_progress_bar(
                i + 1, n_iter, prefix="Analyzing:", suffix="Complete", length=50
            )

        if return_ans:
            return (
                np.array(scores),
                np.array(x0_vals),
                np.array(y0_vals),
                np.array(A_vals),
                np.array(B_vals),
                np.array(C_vals),
            )
        else:
            return

    def fit_origin(
        self, n_iter, step_sizes_init=[1, 1], step_scale=0.9, return_ans=False
    ):
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
            raise Exception(
                "step_sizes_init should be a length 2 array/list/tuple, giving the initial step sizes of x0 and y0, respectively"
            )

        # Call fit_params
        if return_ans:
            scores, x0_vals, y0_vals, _, _, _ = self.fit_params(
                n_iter=n_iter,
                step_sizes_init=[step_sizes[0], step_sizes[1], 0, 0, 0],
                step_scale=0.9,
                return_ans=True,
            )
            return scores, x0_vals, y0_vals
        else:
            self.fit_params(
                n_iter=n_iter,
                step_sizes_init=[step_sizes[0], step_sizes[1], 0, 0, 0],
                step_scale=0.9,
                return_ans=False,
            )
            return

    def fit_params_two_sided_gaussian(self, init_coef=None):
        """
        Instead fit the form I(r) = I_BG * exp(- r^2 / (2 * SD_BG) ^2)
                                + I_ring * exp(- (R-r^2) / (2*SD_1) ^2) * U(R-r)
                                + I_ring * exp(- (R-r^2) / (2*SD_2) ^2) * U(r-R)
                                + N

        where I(r) is the radial intensity function, I_BG, I_ring are the background and amorphous ring intensities,
        SD_BG is the background intensity std. dev., SD_1, and SD_2 are fitting parameters, and U(R-r) is the heaviside step function,
        and N is the bkgd noise level

        For elliptical rings, r is replaced by sqrt(x^2 + B*xy ^2 + C*y^2), to assert the canonical form of the ellipse as follows:
        R^2 = x^2 + Bxy + Cy^2 where A = 1

        Accepts:
            init_coef - initial coefficients to start from.
        Returns (if return_ans is True):

        """
        if init_coef is None:
            if hasattr(self, "coef_opt"):
                init_coef = self.coef_opt
            else:
                raise Exception("You must initialize coefficients if first time.")
        else:
            coef = init_coef

        lb = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, -10, -10, 0])
        ub = np.asarray(
            [
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                self.Nx,
                self.Ny,
                10,
                10,
                np.inf,
            ]
        )

        # TODO: pass arguments to set non default function optimization
        xdata = np.asarray(
            [
                self.xx[self.mask.astype(dtype=bool)],
                self.yy[self.mask.astype(dtype=bool)],
            ]
        )
        ydata = self.calibration_image[self.mask.astype(dtype=bool)]
        coef_opt, _ = sciopt.curve_fit(
            f=two_sided_gaussian_fun_wrapper,
            xdata=xdata,
            ydata=ydata,
            p0=coef,
            bounds=(lb, ub),
        )
        self.coef_opt = coef_opt

        # initializes this to None so that full array transform is perfomed on next grab
        self.polarNorm = None
        return

    def compare_coefs_two_sided_gaussian(self, coef_opt=None, power=1):
        """
        This function will only take in itself, and compare the data to the fit. You can also put in coef_opt yourself to test out parameters
        
        Accepts:
            coef_opt - a vector of 11 elements, corresponding to two_sided_gaussian_fun
        Returns:
            nothing
        """
        if coef_opt is None and hasattr(self, "coef_opt"):
            coef_opt = self.coef_opt
        elif coef_opt is None:
            raise Exception("you must put in coefficients!")

        im = self.calibration_image

        xdata = np.asarray([self.xx, self.yy])
        im_fit = two_sided_gaussian_fun(xdata, coef_opt)

        theta = np.arctan2(self.xx - coef_opt[7], self.yy - coef_opt[6])
        theta_mask = np.cos(theta * 8) > 0
        im_fit = np.reshape(im_fit, self.calibration_image.shape)
        im_combined = (im * theta_mask + im_fit * (1 - theta_mask)) ** power
        im_combined = self.mask * im_combined
        plt.figure(12)
        plt.clf()
        plt.imshow(im_combined)

        return


####################
def two_sided_gaussian_fun_wrapper(X, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    return two_sided_gaussian_fun(X, [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])


def two_sided_gaussian_fun(X, coef):
    """
    This function accepts a set of x,y coordinates X=(x,y) and a set of 11 parameters coef, and
    returns the value at x,y of a function constructed of three Gaussians, as follows:

        f(x,y; I0,I1,sigma0,sigma1,sigma2,R,offset,x0,y0,B,C) =
            Norm(r; I0,sigma0,0) +
            Norm(r; I1,sigma1,R)*Theta(r-R)
            Norm(r; I1,sigma2,R)*Theta(R-r) + offset

    where Norm(I,sigma,R) is a gaussian with max amplitude I, standard deviation sigma, and
    center R; where Theta(x) is a Heavyside function; and where r is the radial coordinate of a
    polar elliptical coordinate system.  In particular, Norm() and r are given by

        Norm(r; I,sigma,R) = I*exp(-(r-R)^2/(2*sigma^2))
        r^2 = (x-x0)^2 + B(x-x0)(y-y0) + C(y-y0)^2

    The input parameters are summarized below:
        X[0] = x-coordinates
        X[1] = y-coordinates
        coef[0] = offset
        coef[1] = I0
        coef[2] = sigma0
        coef[3] = I1
        coef[4] = sigma1
        coef[5] = sigma2
        coef[6] = x0
        coef[7] = y0
        coef[8] = B
        coef[9] = C
        coef[10] = R
    """
    r = np.sqrt( (X[0] - coef[6])**2 + coef[8]*(X[0] - coef[6])*(X[1] - coef[7]) + \
                                                        coef[9]*(X[1] - coef[7])**2 )
    return coef[1] * np.exp( (-1/ (2*coef[2]**2)) * r**2) + \
           coef[3] * np.exp( (-1/ (2*coef[4]**2)) * (coef[10] - r)**2) * \
           np.heaviside((coef[10] - r),0) + \
           coef[3] * np.exp( (-1/ (2*coef[5]**2)) * (coef[10] - r)**2) * \
           np.heaviside((r - coef[10]),0) + \
           coef[0]

def accumarray(indices,values,size):
    """
    Helper function to mimic matlab accum array function.

    Accepts:
        indices     shape (N,M) array, where N is the number of values and M is the dimentionality
                    of the source/destination arrays
        values      shape (N,1) array
        size        int, must be equal to M

    Returns
        output      shape (M,1) array.
    """
    assert indices.shape[1] == len(size), "Size and array mismatch"

    output = np.zeros(size)
    np.add.at(output,(indices[:,0],indices[:,1]),values)
    return dest
