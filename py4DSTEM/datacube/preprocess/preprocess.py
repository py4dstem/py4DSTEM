import numpy as np
from typing import Optional
from scipy.ndimage import median_filter
import warnings

from emdfile import tqdmnd, Metadata
from py4DSTEM.utils import bin2D, get_shifted_ar, fourier_resample
from py4DSTEM.data import DiffractionSlice, Data



# DataCube methods

class Preprocessor:
    def __init__(self):
        pass


    ### Set & manipulate dimensions, bin, crop

    def set_scan_shape(self, Rshape):
        """
        Reshape the data given the real space scan shape. Accepts: Rshape (2-tuple)
        """
        assert len(Rshape) == 2, "Rshape must have a length of 2"
        try:
            # reshape
            self.data = self.data.reshape(
                Rshape[0], Rshape[1], self.Q_Nx, self.Q_Ny)

            # TODO - restruct
            # set dim vectors
            Rpixsize = self.calibration.get_R_pixel_size()
            Rpixunits = self.calibration.get_R_pixel_units()
            self.set_dim(0, [0, Rpixsize], units=Rpixunits)
            self.set_dim(1, [0, Rpixsize], units=Rpixunits)

            # return
            return self

        except ValueError:
            print(f"Can't reshape {self.R_N} scan positions into a {Rshape} shaped array.  Returning")
            return self
        except AttributeError:
            print(f"Can't reshape self.")
            return self

    def swap_RQ(self):
        """
        Swaps the first and last two dimensions of the 4D self.
        """
        self.data = np.transpose(self.data, axes=(2, 3, 0, 1))

        # TODO
        # set dim vectors
        Rpixsize = self.calibration.get_R_pixel_size()
        Rpixunits = self.calibration.get_R_pixel_units()
        Qpixsize = self.calibration.get_Q_pixel_size()
        Qpixunits = self.calibration.get_Q_pixel_units()
        self.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        self.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")
        self.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        self.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

        # return
        return self

    def swap_Rxy(self):
        """
        Swaps the real space x and y coordinates.
        """
        # swap
        self.data = np.moveaxis(self.data, 1, 0)

        # TODO
        # set dim vectors
        Rpixsize = self.calibration.get_R_pixel_size()
        Rpixunits = self.calibration.get_R_pixel_units()
        self.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        self.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")

        # return
        return self

    def swap_Qxy(self):
        """
        Swaps the diffraction space x and y coordinates.
        """
        self.data = np.moveaxis(self.data, 3, 2)
        return self

    def crop_Q(self, ROI):
        """
        Crops the data in diffraction space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Qx_min,Qx_max,Qy_min,Qy_max)
        """
        assert len(ROI) == 4, "Crop region `ROI` must have length 4"
        self.data = self.data[ :, :, ROI[0]:ROI[1], ROI[2]:ROI[3]]

        # TODO
        # set dim vectors
        Qpixsize = self.calibration.get_Q_pixel_size()
        Qpixunits = self.calibration.get_Q_pixel_units()
        self.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        self.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

        # return
        return self

    def crop_R(self, ROI):
        """
        Crops the data in real space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Rx_min,Rx_max,Ry_min,Ry_max)
        """
        assert len(ROI) == 4, "Crop region `ROI` must have length 4"
        self.data = self.data[ ROI[0]:ROI[1] , ROI[2]:ROI[3] ]

        # TODO
        # set dim vectors
        Rpixsize = self.calibration.get_R_pixel_size()
        Rpixunits = self.calibration.get_R_pixel_units()
        self.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        self.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")

        # return
        return self

    def bin_Q(self, N, dtype=None):
        """
        Bins the data in diffraction space by bin factor N

        Parameters
        ----------
        N : int
            The binning factor
        dtype : a datatype (optional)
            Specify the datatype for the output. If not passed, the datatype
            is left unchanged

        Returns
        ------
        self : DataCube
        """
        # validate inputs
        assert type(bin_factor) is int, f"Error: binning factor {bin_factor} is not an int."
        if bin_factor == 1:
            return self
        if dtype is None:
            dtype = self.data.dtype

        # get shape
        R_Nx, R_Ny, Q_Nx, Q_Ny = (
            self.R_Nx,
            self.R_Ny,
            self.Q_Nx,
            self.Q_Ny,
        )
        # crop edges if necessary
        if (Q_Nx % bin_factor == 0) and (Q_Ny % bin_factor == 0):
            pass
        elif Q_Nx % bin_factor == 0:
            self.data = self.data[:, :, :, : -(Q_Ny % bin_factor)]
        elif Q_Ny % bin_factor == 0:
            self.data = self.data[:, :, : -(Q_Nx % bin_factor), :]
        else:
            self.data = self.data[
                :, :, : -(Q_Nx % bin_factor), : -(Q_Ny % bin_factor)
            ]

        # bin
        self.data = (
            self.data.reshape(
                R_Nx,
                R_Ny,
                int(Q_Nx / bin_factor),
                bin_factor,
                int(Q_Ny / bin_factor),
                bin_factor,
            )
            .sum(axis=(3, 5))
            .astype(dtype)
        )

        # TODO
        # set dim vectors
        Qpixsize = self.calibration.get_Q_pixel_size() * bin_factor
        Qpixunits = self.calibration.get_Q_pixel_units()
        self.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        self.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")
        # set calibration pixel size
        self.calibration.set_Q_pixel_size(Qpixsize)

        # return
        return self

    def pad_Q(self, N=None, output_size=None):
        """
        Pads the data in diffraction space by pad factor N, or to match output_size.

        Accepts:
            N (float, or Sequence[float]): the padding factor
            output_size ((int,int)): the padded output size
        """
        Qx, Qy = self.shape[-2:]

        if pad_factor is not None:
            if output_size is not None:
                raise ValueError(
                    "Only one of 'pad_factor' or 'output_size' can be specified."
                )

            pad_factor = np.array(pad_factor)
            if pad_factor.shape == ():
                pad_factor = np.tile(pad_factor, 2)

            if np.any(pad_factor < 1):
                raise ValueError("'pad_factor' needs to be larger than 1.")

            pad_kx = np.round(Qx * (pad_factor[0] - 1) / 2).astype("int")
            pad_kx = (pad_kx, pad_kx)
            pad_ky = np.round(Qy * (pad_factor[1] - 1) / 2).astype("int")
            pad_ky = (pad_ky, pad_ky)

        else:
            if output_size is None:
                raise ValueError(
                    "At-least one of 'pad_factor' or 'output_size' must be specified."
                )

            if len(output_size) != 2:
                raise ValueError(
                    f"'output_size' must have length 2, not {len(output_size)}"
                )

            Sx, Sy = output_size

            if Sx < Qx or Sy < Qy:
                raise ValueError(f"'output_size' must be at-least as large as {(Qx,Qy)}.")

            pad_kx = Sx - Qx
            pad_kx = (pad_kx // 2, pad_kx // 2 + pad_kx % 2)

            pad_ky = Sy - Qy
            pad_ky = (pad_ky // 2, pad_ky // 2 + pad_ky % 2)

        pad_width = (
            (0, 0),
            (0, 0),
            pad_kx,
            pad_ky,
        )

        self.data = np.pad(self.data, pad_width=pad_width, mode="constant")

        # TODO
        Qpixsize = self.calibration.get_Q_pixel_size()
        Qpixunits = self.calibration.get_Q_pixel_units()
        self.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        self.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

        self.calibrate()

        return self

    def resample_Q(self, N=None, output_size=None, method="bilinear"):
        """
        Resamples the data in diffraction space by resampling factor N, or to match output_size,
        using either 'fourier' or 'bilinear' interpolation.

        Accepts:
            N (float, or Sequence[float]): the resampling factor
            output_size ((int,int)): the resampled output size
            method (str): 'fourier' or 'bilinear' (default)
        """
        if method == "fourier":
            if np.size(resampling_factor) != 1:
                warnings.warn(
                    (
                        "Fourier resampling currently only accepts a scalar resampling_factor. "
                        f"'resampling_factor' set to {resampling_factor[0]}."
                    ),
                    UserWarning,
                )
                resampling_factor = resampling_factor[0]

            old_size = self.data.shape

            self.data = fourier_resample(
                self.data, scale=resampling_factor, output_size=output_size
            )

            if not resampling_factor:
                resampling_factor = output_size[0] / old_size[2]
            if self.calibration.get_Q_pixel_size() is not None:
                self.calibration.set_Q_pixel_size(
                    self.calibration.get_Q_pixel_size() / resampling_factor
                )

        elif method == "bilinear":
            from scipy.ndimage import zoom

            if resampling_factor is not None:
                if output_size is not None:
                    raise ValueError(
                        "Only one of 'resampling_factor' or 'output_size' can be specified."
                    )

                resampling_factor = np.array(resampling_factor)
                if resampling_factor.shape == ():
                    resampling_factor = np.tile(resampling_factor, 2)

            else:
                if output_size is None:
                    raise ValueError(
                        "At-least one of 'resampling_factor' or 'output_size' must be specified."
                    )

                if len(output_size) != 2:
                    raise ValueError(
                        f"'output_size' must have length 2, not {len(output_size)}"
                    )

                resampling_factor = np.array(output_size) / np.array(self.shape[-2:])

            resampling_factor = np.concatenate(((1, 1), resampling_factor))
            self.data = zoom(
                self.data, resampling_factor, order=1, mode="grid-wrap", grid_mode=True
            )
            self.calibration.set_Q_pixel_size(
                self.calibration.get_Q_pixel_size() / resampling_factor[2]
            )

        else:
            raise ValueError(
                f"'method' needs to be one of 'bilinear' or 'fourier', not {method}."
            )

        return self

    def bin_Q_mmap(self, N, dtype=np.float32):
        """
        Bins the data in diffraction space by bin factor N for memory mapped data

        Accepts:
            N (int): the binning factor
            dtype: the data type
        """
        # validate inputs
        assert type(bin_factor) is int, f"Error: binning factor {bin_factor} is not an int."
        if bin_factor == 1:
            return self

        # get shape
        R_Nx, R_Ny, Q_Nx, Q_Ny = (
            self.R_Nx,
            self.R_Ny,
            self.Q_Nx,
            self.Q_Ny,
        )
        # allocate space
        data = np.zeros(
            (
                self.R_Nx,
                self.R_Ny,
                self.Q_Nx // bin_factor,
                self.Q_Ny // bin_factor,
            ),
            dtype=dtype,
        )
        # bin
        for Rx, Ry in tqdmnd(self.R_Ny, self.R_Ny):
            data[Rx, Ry, :, :] = bin2D(self.data[Rx, Ry, :, :], bin_factor, dtype=dtype)
        self.data = data

        # TODO
        # set dim vectors
        Qpixsize = self.calibration.get_Q_pixel_size() * bin_factor
        Qpixunits = self.calibration.get_Q_pixel_units()
        self.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        self.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")
        # set calibration pixel size
        self.calibration.set_Q_pixel_size(Qpixsize)

        # return
        return self

    def bin_R(self, N):
        """
        Bins the data in real space by bin factor N

        Accepts:
            N (int): the binning factor
        """
        # validate inputs
        assert type(bin_factor) is int, f"Bin factor {bin_factor} is not an int."
        if bin_factor <= 1:
            return self

        # set shape
        R_Nx, R_Ny, Q_Nx, Q_Ny = (
            self.R_Nx,
            self.R_Ny,
            self.Q_Nx,
            self.Q_Ny,
        )
        # crop edges if necessary
        if (R_Nx % bin_factor == 0) and (R_Ny % bin_factor == 0):
            pass
        elif R_Nx % bin_factor == 0:
            self.data = self.data[:, : -(R_Ny % bin_factor), :, :]
        elif R_Ny % bin_factor == 0:
            self.data = self.data[: -(R_Nx % bin_factor), :, :, :]
        else:
            self.data = self.data[
                : -(R_Nx % bin_factor), : -(R_Ny % bin_factor), :, :
            ]
        # bin
        self.data = self.data.reshape(
            int(R_Nx / bin_factor),
            bin_factor,
            int(R_Ny / bin_factor),
            bin_factor,
            Q_Nx,
            Q_Ny,
        ).sum(axis=(1, 3))

        # TODO
        # set dim vectors
        Rpixsize = self.calibration.get_R_pixel_size() * bin_factor
        Rpixunits = self.calibration.get_R_pixel_units()
        self.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        self.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")
        # set calibration pixel size
        self.calibration.set_R_pixel_size(Rpixsize)

        # return
        return self

    def thin_R(self, N):
        """
        Reduces the data in real space by skipping every N patterns in the x and y directions.

        Accepts:
            N (int): the thinning factor
        """
        # get shapes
        Rshape0 = self.Rshape
        Rshapef = tuple([x // thinning_factor for x in Rshape0])

        # allocate memory
        data = np.empty(
            (Rshapef[0], Rshapef[1], self.Qshape[0], self.Qshape[1]),
            dtype=self.data.dtype,
        )

        # populate data
        for rx, ry in tqdmnd(Rshapef[0], Rshapef[1]):
            rx0 = rx * thinning_factor
            ry0 = ry * thinning_factor
            data[rx, ry, :, :] = self[rx0, ry0, :, :]

        self.data = data

        # TODO
        # set dim vectors
        Rpixsize = self.calibration.get_R_pixel_size() * thinning_factor
        Rpixunits = self.calibration.get_R_pixel_units()
        self.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        self.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")
        # set calibration pixel size
        self.calibration.set_R_pixel_size(Rpixsize)

        # return
        return self


    ### Denoising

    def filter_hot_pixels(self, thresh, ind_compare=1, return_mask=False):
        """
        This function performs pixel filtering to remove hot / bright pixels.
        A mean diffraction pattern is calculated, then a moving local ordering filter
        is applied to it, finding and sorting the intensities of the 21 pixels nearest
        each pixel (where 21 = (the pixel itself) + (nearest neighbors) + (next
        nearest neighbors) = (1) + (8) + (12) = 21; the next nearest neighbors
        exclude the corners of the NNN square of pixels). This filter then returns
        a single value at each pixel given by the N'th highest value of these 21
        sorted values, where N is specified by `ind_compare`.  ind_compare=0
        specifies the highest intensity, =1 is the second hightest, etc. Next, a mask
        is generated which is True for all pixels which are least a value `thresh`
        higher than the local ordering filter output. Thus for the default
        `ind_compare` value of 1, the mask will be True wherever the mean diffraction
        pattern is higher than the second brightest pixel in it's local window by
        at least a value of `thresh`. Finally, we loop through all diffraction
        images, and any pixels defined by mask are replaced by their 3x3 local
        median.

        Parameters
        ----------
        thresh : float
            Threshold for replacing hot pixels, if pixel value minus local ordering
            filter exceeds it.
        ind_compare : int
            Which median filter value to compare against. 0 = brightest pixel,
            1 = next brightest, etc.
        return_mask : bool
            If True, returns the filter mask

        Returns
        -------
        self : Datacube
        mask : bool
            (optional) the bad pixel mask
        """

        # Mean image over all probe positions
        diff_mean = np.mean(self.data, axis=(0, 1))
        shape = diff_mean.shape

        # Moving local ordered pixel values
        diff_local_med = np.sort(
            np.vstack(
                [
                    np.roll(diff_mean, (-1, -1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (0, -1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (1, -1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-1, 0), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (0, 0), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (1, 0), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-1, 1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (0, 1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (1, 1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-1, -2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (0, -2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (1, -2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-1, 2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (0, 2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (1, 2), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-2, -1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-2, 0), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (-2, 1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (2, -1), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (2, 0), axis=(0, 1)).ravel(),
                    np.roll(diff_mean, (2, 1), axis=(0, 1)).ravel(),
                ]
            ),
            axis=0,
        )
        # arry of the ind_compare'th pixel intensity
        diff_compare = np.reshape(diff_local_med[-ind_compare - 1, :], shape)

        # Generate mask
        mask = diff_mean - diff_compare > thresh

        # If the mask is empty, return
        if np.sum(mask) == 0:
            print("No hot pixels detected")
            if return_mask is True:
                return self, mask
            else:
                return self

        # Otherwise, apply filtering

        # Get masked indices
        x_ma, y_ma = np.nonzero(mask)

        # Get local windows for each masked pixel
        xslices, yslices = [], []
        for xm, ym in zip(x_ma, y_ma):
            xslice, yslice = slice(xm - 1, xm + 2), slice(ym - 1, ym + 2)
            if xslice.start < 0:
                xslice = slice(0, xslice.stop)
            elif xslice.stop > shape[0]:
                xslice = slice(xslice.start, shape[0])
            if yslice.start < 0:
                yslice = slice(0, yslice.stop)
            elif yslice.stop > shape[1]:
                yslice = slice(yslice.start, shape[1])
            xslices.append(xslice)
            yslices.append(yslice)

        # Loop and replace pixels
        for ax, ay in tqdmnd(
            *(self.R_Nx, self.R_Ny), desc="Cleaning pixels", unit=" images"
        ):
            for xm, ym, xs, ys in zip(x_ma, y_ma, xslices, yslices):
                self.data[ax, ay, xm, ym] = np.median(self.data[ax, ay, xs, ys])

            # Calculate local 3x3 median images
            # im_med = median_filter(self.data[ax, ay, :, :], size=3, mode="nearest")
            # self.data[ax, ay, :, :][mask] = im_med[mask]

        # Return
        if return_mask is True:
            return self, mask
        else:
            return self


    ### Background subtraction

    def get_radial_bkgrnd(self, rx, ry, sigma=2):
        """
        Computes and returns a background image for the diffraction
        pattern at (rx,ry), populated by radial rings of constant intensity
        about the origin, with the value of each ring given by the median
        value of the diffraction pattern at that radial distance.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        sigma : number
            If >0, applying a gaussian smoothing in the radial direction
            before returning

        Returns
        -------
        background : ndarray
            The radial background
        """
        # ensure a polar cube and origin exist
        assert self.polar is not None, "No polar self found!"
        assert self.calibration.get_origin() is not None, "No origin found!"

        # get the 1D median background
        bkgrd_ma_1d = np.ma.median(self.polar.data[rx, ry], axis=0)
        bkgrd_1d = bkgrd_ma_1d.data
        bkgrd_1d[bkgrd_ma_1d.mask] = 0

        # smooth
        if sigma > 0:
            bkgrd_1d = gaussian_filter1d(bkgrd_1d, sigma)

        # define the 2D cartesian coordinate system
        origin = self.calibration.get_origin()
        origin = origin[0][rx, ry], origin[1][rx, ry]
        qxx, qyy = self.qxx_raw - origin[0], self.qyy_raw - origin[1]

        # get distance qr in polar-elliptical coords
        ellipse = self.calibration.get_ellipse()
        ellipse = (1, 1, 0) if ellipse is None else ellipse
        a, b, theta = ellipse

        qrr = np.sqrt(
            ((qxx * np.cos(theta)) + (qyy * np.sin(theta))) ** 2
            + ((qxx * np.sin(theta)) - (qyy * np.cos(theta))) ** 2 / (b / a) ** 2
        )

        # make an interpolation function and get the 2D background
        f = interp1d(self.polar.radial_bins, bkgrd_1d, fill_value="extrapolate")
        background = f(qrr)

        # return
        return background

    def get_radial_bksb_dp(self, rx, ry, sigma=2):
        """
        Computes and returns the diffraction pattern at beam position (rx,ry)
        with a radial background subtracted.  See the docstring for
        self.get_radial_background for more info.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        sigma : number
            If >0, applying a gaussian smoothing in the radial direction
            before returning

        Returns
        -------
        data : ndarray
            The radial background subtracted diffraction image
        """
        # get 2D background
        background = self.get_radial_bkgrnd(rx, ry, sigma)

        # subtract, zero negative values, return
        ans = self.data[rx, ry] - background
        ans[ans < 0] = 0
        return ans

    ### Local averaging

    def get_local_ave_dp(
        self,
        rx,
        ry,
        radial_bksb=False,
        sigma=2,
        braggmask=False,
        braggvectors=None,
        braggmask_radius=None,
    ):
        """
        Computes and returns the diffraction pattern at beam position (rx,ry)
        after weighted local averaging with its nearest-neighbor patterns,
        using a 3x3 gaussian kernel for the weightings.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        radial_bksb : bool
            It True, apply a radial background subtraction to each pattern
            before averaging
        sigma : number
            If radial_bksb is True, use this sigma for radial smoothing of
            the background
        braggmask : bool
            If True, masks bragg scattering at each scan position before
            averaging. `braggvectors` and `braggmask_radius` must be
            specified.
        braggvectors : BraggVectors
            The Bragg vectors to use for masking
        braggmask_radius : number
            The radius about each Bragg point to mask

        Returns
        -------
        data : ndarray
            The radial background subtracted diffraction image
        """
        # define the kernel
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

        # get shape and check for valid inputs
        nx, ny = self.data.shape[:2]
        assert rx >= 0 and rx < nx, "rx outside of scan range"
        assert ry >= 0 and ry < ny, "ry outside of scan range"

        # get the subcube, checking for edge patterns
        # and modifying the kernel as needed
        if rx != 0 and rx != (nx - 1) and ry != 0 and ry != (ny - 1):
            subcube = self.data[rx - 1 : rx + 2, ry - 1 : ry + 2, :, :]
        elif rx == 0 and ry == 0:
            subcube = self.data[:2, :2, :, :]
            kernel = kernel[1:, 1:]
        elif rx == 0 and ry == (ny - 1):
            subcube = self.data[:2, -2:, :, :]
            kernel = kernel[1:, :-1]
        elif rx == (nx - 1) and ry == 0:
            subcube = self.data[-2:, :2, :, :]
            kernel = kernel[:-1, 1:]
        elif rx == (nx - 1) and ry == (ny - 1):
            subcube = self.data[-2:, -2:, :, :]
            kernel = kernel[:-1, :-1]
        elif rx == 0:
            subcube = self.data[:2, ry - 1 : ry + 2, :, :]
            kernel = kernel[1:, :]
        elif rx == (nx - 1):
            subcube = self.data[-2:, ry - 1 : ry + 2, :, :]
            kernel = kernel[:-1, :]
        elif ry == 0:
            subcube = self.data[rx - 1 : rx + 2, :2, :, :]
            kernel = kernel[:, 1:]
        elif ry == (ny - 1):
            subcube = self.data[rx - 1 : rx + 2, -2:, :, :]
            kernel = kernel[:, :-1]
        else:
            raise Exception(f"Invalid (rx,ry) = ({rx},{ry})...")

        # normalize the kernel
        kernel /= np.sum(kernel)

        # compute...

        # ...in the simple case
        if not (radial_bksb) and not (braggmask):
            ans = np.tensordot(subcube, kernel, axes=((0, 1), (0, 1)))

        # ...with radial background subtration
        elif radial_bksb and not (braggmask):
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx != 0 else 0
            _ys = 1 if ry != 0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            for (i, j), w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                ans += self.get_radial_bksb_dp(x, y, sigma) * w

        # ...with bragg masking
        elif not (radial_bksb) and braggmask:
            assert (
                braggvectors is not None
            ), "`braggvectors` must be specified or `braggmask` must be turned off!"
            assert (
                braggmask_radius is not None
            ), "`braggmask_radius` must be specified or `braggmask` must be turned off!"
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx != 0 else 0
            _ys = 1 if ry != 0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            weights = np.zeros(self.Qshape)
            for (i, j), w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                mask = self.get_braggmask(braggvectors, x, y, braggmask_radius)
                weights_curr = mask * w
                ans += self.data[x, y] * weights_curr
                weights += weights_curr
            # normalize
            out = np.full_like(ans, np.nan)
            ans_mask = weights > 0
            ans = np.divide(ans, weights, out=out, where=ans_mask)
            # make masked array
            ans = np.ma.array(data=ans, mask=np.logical_not(ans_mask))
            pass

        # ...with both radial background subtraction and bragg masking
        else:
            assert (
                braggvectors is not None
            ), "`braggvectors` must be specified or `braggmask` must be turned off!"
            assert (
                braggmask_radius is not None
            ), "`braggmask_radius` must be specified or `braggmask` must be turned off!"
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx != 0 else 0
            _ys = 1 if ry != 0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            weights = np.zeros(self.Qshape)
            for (i, j), w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                mask = self.get_braggmask(braggvectors, x, y, braggmask_radius)
                weights_curr = mask * w
                ans += self.get_radial_bksb_dp(x, y, sigma) * weights_curr
                weights += weights_curr
            # normalize
            out = np.full_like(ans, np.nan)
            ans_mask = weights > 0
            ans = np.divide(ans, weights, out=out, where=ans_mask)
            # make masked array
            ans = np.ma.array(data=ans, mask=np.logical_not(ans_mask))
            pass

        # return
        return ans


    ### Bragg masking

    def get_braggmask(self, braggvectors, rx, ry, radius):
        """
        Returns a boolean mask which is False in a radius of `radius` around
        each bragg scattering vector at scan position (rx,ry).

        Parameters
        ----------
        braggvectors : BraggVectors
            The bragg vectors
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        radius : number
            mask pixels about each bragg vector to this radial distance

        Returns
        -------
        mask : boolean ndarray
        """
        # allocate space
        mask = np.ones(self.Qshape, dtype=bool)
        # get the vectors
        vects = braggvectors.raw[rx, ry]
        # loop
        for idx in range(len(vects.data)):
            qr = np.hypot(self.qxx_raw - vects.qx[idx], self.qyy_raw - vects.qy[idx])
            mask = np.logical_and(mask, qr > radius)
        return mask

    ### Shift patterns

    def align_diffraction(
        self,
        xshifts,
        yshifts,
        periodic=True,
        bilinear=False,
    ):
        """
        This function shifts each 2D diffraction image by the values defined by
        (xshifts,yshifts). The shift values can be scalars (same shift for all
        images) or arrays with the same dimensions as the probe positions in
        self.

        Args:
                self (DataCube):   py4DSTEM DataCube
                xshifts (float):       Array or scalar value for the x dim shifts
                yshifts (float):       Array or scalar value for the y dim shifts
                periodic (bool):       Flag for periodic boundary conditions. If set to false, boundaries are assumed to be periodic.
                bilinear (bool):       Flag for bilinear image shifts. If set to False, Fourier shifting is used.

            Returns:
                self (DataCube):   py4DSTEM DataCube
        """

        # if the shift values are constant, expand to arrays
        xshifts = np.array(xshifts)
        yshifts = np.array(yshifts)
        if xshifts.ndim == 0:
            xshifts = xshifts * np.ones((self.R_Nx, self.R_Ny))
        if yshifts.ndim == 0:
            yshifts = yshifts * np.ones((self.R_Nx, self.R_Ny))

        # Loop over all images
        for ax, ay in tqdmnd(
            *(self.R_Nx, self.R_Ny), desc="Shifting images", unit=" images"
        ):
            self.data[ax, ay, :, :] = get_shifted_ar(
                self.data[ax, ay, :, :],
                xshifts[ax, ay],
                yshifts[ax, ay],
                periodic=periodic,
                bilinear=bilinear,
            )

        return self


