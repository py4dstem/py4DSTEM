# Defines the Probe class

import numpy as np
from typing import Optional
from warnings import warn

from emdfile import Metadata
from py4DSTEM.data import DiffractionSlice, Data
from py4DSTEM.visualize import show
from scipy.ndimage import binary_opening, binary_dilation, distance_transform_edt


class Probe(DiffractionSlice, Data):
    """
    Stores a vacuum probe.

    Both a vacuum probe and a kernel for cross-correlative template matching
    derived from that probe are stored and can be accessed at

        >>> p.probe
        >>> p.kernel

    respectively, for some Probe instance `p`. If a kernel has not been computed
    the latter expression returns None.


    """

    def __init__(self, data: np.ndarray, name: Optional[str] = "probe"):
        """
        Accepts:
            data (2D or 3D np.ndarray): the vacuum probe, or
                the vacuum probe + kernel
            name (str): a name

        Returns:
            (Probe)
        """
        # if only the probe is passed, make space for the kernel
        if data.ndim == 2:
            data = np.stack([data, np.zeros_like(data)])

        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self, name=name, data=data, slicelabels=["probe", "kernel"]
        )

        # initialize metadata params
        self.metadata = Metadata(name="params")
        self.alpha = None
        self.origin = None

    ## properties

    @property
    def probe(self):
        return self.get_slice("probe").data

    @probe.setter
    def probe(self, x):
        assert x.shape == (self.data.shape[1:])
        self.data[0, :, :] = x

    @property
    def kernel(self):
        return self.get_slice("kernel").data

    @kernel.setter
    def kernel(self, x):
        assert x.shape == (self.data.shape[1:])
        self.data[1, :, :] = x

    @property
    def alpha(self):
        return self.metadata["params"]["alpha"]

    @alpha.setter
    def alpha(self, x):
        self.metadata["params"]["alpha"] = x

    @property
    def origin(self):
        return self.metadata["params"]["origin"]

    @origin.setter
    def origin(self, x):
        self.metadata["params"]["origin"] = x

    # read
    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = DiffractionSlice._get_constructor_args(group)
        args = {
            "data": ar_constr_args["data"],
            "name": ar_constr_args["name"],
        }
        return args

    # generation methods

    @classmethod
    def from_vacuum_data(cls, data, mask=None, threshold=0.2, expansion=12, opening=3):
        """
        Generates and returns a vacuum probe Probe instance from either a
        2D vacuum image or a 3D stack of vacuum diffraction patterns.

        The probe is multiplied by `mask`, if it's passed.  An additional
        masking step zeros values outside of a mask determined by `threshold`,
        `expansion`, and `opening`, generated by first computing the binary image
        probe < max(probe)*threshold, then applying a binary expansion and
        then opening to this image. No alignment is performed - i.e. it is assumed
        that the beam was stationary during acquisition of the stack. To align
        the images, use the DataCube .get_vacuum_probe method.

        Parameters
        ----------
        data : 2D or 3D array
            the vacuum diffraction data. For 3D stacks, use shape (N,Q_Nx,Q_Ny)
        mask : boolean array, optional
            mask applied to the probe
        threshold : float
            threshold determining mask which zeros values outside of probe
        expansion : int
            number of pixels by which the zeroing mask is expanded to capture
            the full probe
        opening : int
            size of binary opening used to eliminate stray bright pixels

        Returns
        -------
        probe : Probe
            the vacuum probe
        """
        assert isinstance(data, np.ndarray)
        if data.ndim == 3:
            probe = np.average(data, axis=0)
        elif data.ndim == 2:
            probe = data
        else:
            raise Exception(f"data must be 2- or 3-D, not {data.ndim}-D")

        if mask is not None:
            probe *= mask

        mask = probe > np.max(probe) * threshold
        mask = binary_opening(mask, iterations=opening)
        mask = binary_dilation(mask, iterations=1)
        mask = (
            np.cos(
                (np.pi / 2)
                * np.minimum(
                    distance_transform_edt(np.logical_not(mask)) / expansion, 1
                )
            )
            ** 2
        )

        probe = cls(probe * mask)
        return probe

    @classmethod
    def generate_synthetic_probe(cls, radius, width, Qshape):
        """
        Makes a synthetic probe, with the functional form of a disk blurred by a
        sigmoid (a logistic function).

        Parameters
        ----------
        radius : float
            the probe radius
        width : float
            the blurring of the probe edge. width represents the
            full width of the blur, with x=-w/2 to x=+w/2 about the edge
            spanning values of ~0.12 to 0.88
        Qshape : 2 tuple
            the diffraction plane dimensions

        Returns
        -------
        probe : Probe
            the probe
        """
        # Make coords
        Q_Nx, Q_Ny = Qshape
        qy, qx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
        qy, qx = qy - Q_Ny / 2.0, qx - Q_Nx / 2.0
        qr = np.sqrt(qx**2 + qy**2)

        # Shift zero to disk edge
        qr = qr - radius

        # Calculate logistic function
        probe = 1 / (1 + np.exp(4 * qr / width))

        return cls(probe)

    # calibration methods

    def measure_disk(
        self,
        thresh_lower=0.01,
        thresh_upper=0.99,
        N=100,
        data=None,
        zero_vacuum=True,
        alpha_max=1.2,
        returncalc=True,
        plot=True,
    ):
        """
        Finds the center and radius of an average probe image.

        A naive algorithm. Creates a series of N binary masks by thresholding
        the probe image a linspace of N thresholds from thresh_lower to
        thresh_upper, relative to the image max/min. For each mask, we find the
        square root of the number of True valued pixels divided by pi to
        estimate a radius. Because the central disk is intense relative to the
        remainder of the image, the computed radii are expected to vary very
        little over a wider range threshold values. A range of r values
        considered trustworthy is estimated by taking the derivative
        r(thresh)/dthresh identifying where it is small, and the mean of this
        range is returned as the radius. A center is estimated using a binary
        thresholded image in combination with the center of mass operator.

        Parameters
        ----------
        thresh_lower : float, 0 to 1
            the lower limit of threshold values
        thresh_upper : float, 0 to 1)
            the upper limit of threshold values
        N : int
            the number of thresholds / masks to use
        data : 2d array, optional
            if passed, uses this 2D array in place of the probe image when
            performing the computation. This also supresses storing the
            results in the Probe's calibration metadata
        zero_vacuum : bool
            if True, sets pixels beyond alpha_max * the semiconvergence angle
            to zero. Ignored if `data` is not None
        alpha_max : number
            sets the maximum scattering angle in the probe image, beyond which
            values are set to zero if `zero_vacuum` is True
        returncalc : True
            toggles returning the answer
        plot : bool
            toggles visualizing results

        Returns
        -------
        r, x0, y0 : (3-tuple)
            the radius and origin
        """
        from py4DSTEM.process.utils import get_CoM

        # set the image
        im = self.probe if data is None else data

        # define the thresholds
        thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
        r_vals = np.zeros(N)

        # get binary images and compute a radius for each
        immax = np.max(im)
        for i, val in enumerate(thresh_vals):
            mask = im > immax * val
            r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

        # Get derivative and determine trustworthy r-values
        dr_dtheta = np.gradient(r_vals)
        mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * np.median(dr_dtheta))
        r = np.mean(r_vals[mask])

        # Get origin
        thresh = np.mean(thresh_vals[mask])
        mask = im > immax * thresh
        x0, y0 = get_CoM(im * mask)

        # Store metadata
        ans = r, x0, y0
        if data is None:
            self.alpha = r
            self.origin = (x0, y0)
            try:
                self.calibration.set_probe_param(ans)
            except AttributeError:
                warn(
                    f"Couldn't store the probe parameters in metadata as no calibration was found for this Probe instance, {self}"
                )
                pass

        if data is None and zero_vacuum:
            self.zero_vacuum(alpha_max=alpha_max)

        # show result
        if plot:
            show(im, circle={"center": (x0, y0), "R": r, "fill": True, "alpha": 0.36})

        # return
        if returncalc:
            return ans

    def zero_vacuum(
        self,
        alpha_max=1.2,
    ):
        """
        Sets pixels outside of the probe's central disk to zero.

        The probe origin and convergence semiangle must be set for this
        method to run - these can be set using `measure_disk`. Pixels are
        defined as outside the central disk if their distance from the origin
        exceeds the semiconvergence angle * alpha_max.

        Parameters
        ----------
        alpha_max : number
            Pixels farther than this number times the semiconvergence angle
            from the origin are set to zero
        """
        # validate inputs
        assert (
            self.alpha is not None
        ), "no probe semiconvergence angle found; try running `Probe.measure_disk`"
        assert (
            self.origin is not None
        ), "no probe origin found; try running `Probe.measure_disk`"
        # make a mask
        qyy, qxx = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        qrr = np.hypot(qxx - self.origin[0], qyy - self.origin[1])
        mask = qrr < self.alpha * alpha_max
        # zero the vacuum
        self.probe *= mask
        pass

    # Kernel generation methods

    def get_kernel(
        self, mode="flat", origin=None, data=None, returncalc=True, **kwargs
    ):
        """
        Creates a cross-correlation kernel from the vacuum probe.

        Specific behavior and valid keyword arguments depend on the `mode`
        specified.  In each case, the center of the probe is shifted to the
        origin and the kernel normalized such that it sums to 1. This is the
        only processing performed if mode is 'flat'. Otherwise, a centrosymmetric
        region of negative intensity is added around the probe intended to promote
        edge-filtering-like behavior during cross correlation, with the
        functional form of the subtracted region defined by `mode` and the
        relevant **kwargs. For normalization, flat probes integrate to 1, and the
        remaining probes integrate to 1 before subtraction and 0 after.  Required
        keyword arguments are:

          - 'flat': No required arguments. This mode is recommended for bullseye
            or other structured probes
          - 'gaussian': Required arg `sigma` (number), the width (standard
            deviation) of a centered gaussian to be subtracted.
          - 'sigmoid': Required arg `radii` (2-tuple), the inner and outer radii
            (ri,ro) of an annular region with a sine-squared sigmoidal radial
            profile to be subtracted.
          - 'sigmoid_log': Required arg `radii` (2-tuple), the inner and outer radii
            (ri,ro) of an annular region with a logistic sigmoidal radial
            profile to be subtracted.

        Parameters
        ----------
        mode : str
            must be in 'flat','gaussian','sigmoid','sigmoid_log'
        origin : 2-tuple, optional
            specify the origin. If not passed, looks for a value for the probe
            origin in metadata. If not found there, calls .measure_disk.
        data : 2d array, optional
            if specified, uses this array instead of the probe image to compute
            the kernel
        **kwargs
            see descriptions above

        Returns
        -------
        kernel : 2D array
        """

        modes = ["flat", "gaussian", "sigmoid", "sigmoid_log"]

        # parse args
        assert mode in modes, f"mode must be in {modes}. Received {mode}"

        # get function
        function_dict = {
            "flat": self.get_probe_kernel_flat,
            "gaussian": self.get_probe_kernel_edge_gaussian,
            "sigmoid": self._get_probe_kernel_edge_sigmoid_sine_squared,
            "sigmoid_log": self._get_probe_kernel_edge_sigmoid_sine_squared,
        }
        fn = function_dict[mode]

        # check for the origin
        if origin is None:
            try:
                x = self.calibration.get_probe_params()
            except AttributeError:
                x = None
            finally:
                if x is None:
                    origin = None
                else:
                    r, x, y = x
                    origin = (x, y)

        # get the data
        probe = data if data is not None else self.probe

        # compute
        kern = fn(probe, origin=origin, **kwargs)

        # add to the Probe
        self.kernel = kern

        # return
        if returncalc:
            return kern

    @staticmethod
    def get_probe_kernel_flat(probe, origin=None, bilinear=False):
        """
        Creates a cross-correlation kernel from the vacuum probe by normalizing
        and shifting the center.

        Parameters
        ----------
        probe : 2d array
            the vacuum probe
        origin : 2-tuple (optional)
            the origin of diffraction space. If not specified, finds the origin
            using get_probe_radius.
        bilinear : bool (optional)
            By default probe is shifted via a Fourier transform. Setting this to
            True overrides it and uses bilinear shifting. Not recommended!

        Returns
        -------
        kernel : ndarray
            the cross-correlation kernel corresponding to the probe, in real
            space
        """
        from py4DSTEM.process.utils import get_shifted_ar

        Q_Nx, Q_Ny = probe.shape

        # Get CoM
        if origin is None:
            from py4DSTEM.process.calibration import get_probe_size

            _, xCoM, yCoM = get_probe_size(probe)
        else:
            xCoM, yCoM = origin

        # Normalize
        probe = probe / np.sum(probe)

        # Shift center to corners of array
        probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM, bilinear=bilinear)

        # Return
        return probe_kernel

    @staticmethod
    def get_probe_kernel_edge_gaussian(
        probe,
        sigma,
        origin=None,
        bilinear=True,
    ):
        """
        Creates a cross-correlation kernel from the probe, subtracting a
        gaussian from the normalized probe such that the kernel integrates to
        zero, then shifting the center of the probe to the array corners.

        Parameters
        ----------
        probe : ndarray
            the diffraction pattern corresponding to the probe over vacuum
        sigma : float
            the width of the gaussian to subtract, relative to the standard
            deviation of the probe
        origin : 2-tuple (optional)
            the origin of diffraction space. If not specified, finds the origin
            using get_probe_radius.
        bilinear : bool
            By default probe is shifted via a Fourier transform. Setting this to
            True overrides it and uses bilinear shifting. Not recommended!

        Returns
        -------
        kernel : ndarray
            the cross-correlation kernel
        """
        from py4DSTEM.process.utils import get_shifted_ar

        Q_Nx, Q_Ny = probe.shape

        # Get CoM
        if origin is None:
            from py4DSTEM.process.calibration import get_probe_size

            _, xCoM, yCoM = get_probe_size(probe)
        else:
            xCoM, yCoM = origin

        # Shift probe to origin
        probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM, bilinear=bilinear)

        # Generate normalization kernel
        # Coordinates
        qy, qx = np.meshgrid(
            np.mod(np.arange(Q_Ny) + Q_Ny // 2, Q_Ny) - Q_Ny // 2,
            np.mod(np.arange(Q_Nx) + Q_Nx // 2, Q_Nx) - Q_Nx // 2,
        )
        qr2 = qx**2 + qy**2
        # Calculate Gaussian normalization kernel
        qstd2 = np.sum(qr2 * probe_kernel) / np.sum(probe_kernel)
        kernel_norm = np.exp(-qr2 / (2 * qstd2 * sigma**2))

        # Output normalized kernel
        probe_kernel = probe_kernel / np.sum(probe_kernel) - kernel_norm / np.sum(
            kernel_norm
        )

        return probe_kernel

    @staticmethod
    def get_probe_kernel_edge_sigmoid(
        probe,
        radii,
        origin=None,
        type="sine_squared",
        bilinear=True,
    ):
        """
        Creates a convolution kernel from an average probe, subtracting an annular
        trench about the probe such that the kernel integrates to zero, then
        shifting the center of the probe to the array corners.

        Parameters
        ----------
        probe : ndarray
            the diffraction pattern corresponding to the probe over vacuum
        radii : 2-tuple
            the sigmoid inner and outer radii
        origin : 2-tuple (optional)
            the origin of diffraction space. If not specified, finds the origin
            using get_probe_radius.
        type : string
            must be 'logistic' or 'sine_squared'
        bilinear : bool
            By default probe is shifted via a Fourier transform. Setting this to
            True overrides it and uses bilinear shifting. Not recommended!

        Returns
        -------
        kernel : 2d array
            the cross-correlation kernel
        """
        from py4DSTEM.process.utils import get_shifted_ar

        # parse inputs
        if isinstance(probe, Probe):
            probe = probe.probe

        valid_types = ("logistic", "sine_squared")
        assert type in valid_types, "type must be in {}".format(valid_types)
        Q_Nx, Q_Ny = probe.shape
        ri, ro = radii

        # Get CoM
        if origin is None:
            from py4DSTEM.process.calibration import get_probe_size

            _, xCoM, yCoM = get_probe_size(probe)
        else:
            xCoM, yCoM = origin

        # Shift probe to origin
        probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM, bilinear=bilinear)

        # Generate normalization kernel
        # Coordinates
        qy, qx = np.meshgrid(
            np.mod(np.arange(Q_Ny) + Q_Ny // 2, Q_Ny) - Q_Ny // 2,
            np.mod(np.arange(Q_Nx) + Q_Nx // 2, Q_Nx) - Q_Nx // 2,
        )
        qr = np.sqrt(qx**2 + qy**2)
        # Calculate sigmoid
        if type == "logistic":
            r0 = 0.5 * (ro + ri)
            sigma = 0.25 * (ro - ri)
            sigmoid = 1 / (1 + np.exp((qr - r0) / sigma))
        elif type == "sine_squared":
            sigmoid = (qr - ri) / (ro - ri)
            sigmoid = np.minimum(np.maximum(sigmoid, 0.0), 1.0)
            sigmoid = np.cos((np.pi / 2) * sigmoid) ** 2
        else:
            raise Exception("type must be in {}".format(valid_types))

        # Output normalized kernel
        probe_kernel = probe_kernel / np.sum(probe_kernel) - sigmoid / np.sum(sigmoid)

        return probe_kernel

    def _get_probe_kernel_edge_sigmoid_sine_squared(
        self,
        probe,
        radii,
        origin=None,
        **kwargs,
    ):
        return self.get_probe_kernel_edge_sigmoid(
            probe,
            radii,
            origin=origin,
            type="sine_squared",
            **kwargs,
        )

    def _get_probe_kernel_edge_sigmoid_logistic(
        self,
        probe,
        radii,
        origin=None,
        **kwargs,
    ):
        return self.get_probe_kernel_edge_sigmoid(
            probe, radii, origin=origin, type="logistic", **kwargs
        )
