# Defines the DataCube class, which stores 4D-STEM datacubes

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import (
    binary_opening,
    binary_dilation,
    distance_transform_edt,
    binary_fill_holes,
    gaussian_filter1d,
    gaussian_filter,
)
from typing import Optional, Union

from emdfile import Array, Metadata, Node, Root, tqdmnd
from py4DSTEM.data import Data, Calibration
from py4DSTEM.datacube.virtualimage import DataCubeVirtualImager
from py4DSTEM.datacube.virtualdiffraction import DataCubeVirtualDiffraction


class DataCube(
    Array,
    Data,
    DataCubeVirtualImager,
    DataCubeVirtualDiffraction,
):
    """
    Storage and processing methods for 4D-STEM datasets.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "datacube",
        slicelabels: Optional[Union[bool, list]] = None,
        calibration: Optional[Union[Calibration, None]] = None,
    ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            calibration (None or Calibration or 'pass'): default (None)
                creates and attaches a new Calibration instance to root
                metadata, or, passing a Calibration instance uses this instead.
            slicelabels (None or list): names for slices if this is a
                stack of datacubes

        Returns:
            A new DataCube instance.
        """
        # initialize as an Array
        Array.__init__(
            self,
            data=data,
            name=name,
            units="pixel intensity",
            dim_names=["Rx", "Ry", "Qx", "Qy"],
            slicelabels=slicelabels,
        )

        # initialize as Data
        Data.__init__(self, calibration)

        # register with calibration
        self.calibration.register_target(self)

        # cartesian coords
        self.calibrate()

        # polar coords
        self.polar = None

    def calibrate(self):
        """
        Calibrate the coordinate axes of the datacube. Using the calibrations
        at self.calibration, sets the 4 dim vectors (Qx,Qy,Rx,Ry) according
        to the pixel size, units and origin positions, then updates the
        meshgrids representing Q and R space.
        """
        assert self.calibration is not None, "No calibration found!"

        # Get calibration values
        rpixsize = self.calibration.get_R_pixel_size()
        rpixunits = self.calibration.get_R_pixel_units()
        qpixsize = self.calibration.get_Q_pixel_size()
        qpixunits = self.calibration.get_Q_pixel_units()
        origin = self.calibration.get_origin_mean()
        if origin is None or origin == (None, None):
            origin = (0, 0)

        # Calc dim vectors
        dim_rx = np.arange(self.R_Nx) * rpixsize
        dim_ry = np.arange(self.R_Ny) * rpixsize
        dim_qx = -origin[0] + np.arange(self.Q_Nx) * qpixsize
        dim_qy = -origin[1] + np.arange(self.Q_Ny) * qpixsize

        # Set dim vectors
        self.set_dim(0, dim_rx, units=rpixunits)
        self.set_dim(1, dim_ry, units=rpixunits)
        self.set_dim(2, dim_qx, units=qpixunits)
        self.set_dim(3, dim_qy, units=qpixunits)

        # Set meshgrids
        self._qxx, self._qyy = np.meshgrid(dim_qx, dim_qy)
        self._rxx, self._ryy = np.meshgrid(dim_rx, dim_ry)

        self._qyy_raw, self._qxx_raw = np.meshgrid(
            np.arange(self.Q_Ny), np.arange(self.Q_Nx)
        )
        self._ryy_raw, self._rxx_raw = np.meshgrid(
            np.arange(self.R_Ny), np.arange(self.R_Nx)
        )

    # coordinate meshgrids
    @property
    def rxx(self):
        return self._rxx

    @property
    def ryy(self):
        return self._ryy

    @property
    def qxx(self):
        return self._qxx

    @property
    def qyy(self):
        return self._qyy

    @property
    def rxx_raw(self):
        return self._rxx_raw

    @property
    def ryy_raw(self):
        return self._ryy_raw

    @property
    def qxx_raw(self):
        return self._qxx_raw

    @property
    def qyy_raw(self):
        return self._qyy_raw

    # coordinate meshgrids with shifted origin
    def qxxs(self, rx, ry):
        qx0_shift = self.calibration.get_qx0shift(rx, ry)
        if qx0_shift is None:
            raise Exception(
                "Can't compute shifted meshgrid - origin shift is not defined"
            )
        return self.qxx - qx0_shift

    def qyys(self, rx, ry):
        qy0_shift = self.calibration.get_qy0shift(rx, ry)
        if qy0_shift is None:
            raise Exception(
                "Can't compute shifted meshgrid - origin shift is not defined"
            )
        return self.qyy - qy0_shift

    # shape properties

    ## shape

    # FOV
    @property
    def R_Nx(self):
        return self.data.shape[0]

    @property
    def R_Ny(self):
        return self.data.shape[1]

    @property
    def Q_Nx(self):
        return self.data.shape[2]

    @property
    def Q_Ny(self):
        return self.data.shape[3]

    @property
    def Rshape(self):
        return (self.data.shape[0], self.data.shape[1])

    @property
    def Qshape(self):
        return (self.data.shape[2], self.data.shape[3])

    @property
    def R_N(self):
        return self.R_Nx * self.R_Ny

    # aliases
    qnx = Q_Nx
    qny = Q_Ny
    rnx = R_Nx
    rny = R_Ny
    rshape = Rshape
    qshape = Qshape
    rn = R_N

    ## pixel size / units

    # Q
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()

    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()

    # R
    @property
    def R_pixel_size(self):
        return self.calibration.get_R_pixel_size()

    @property
    def R_pixel_units(self):
        return self.calibration.get_R_pixel_units()

    # aliases
    qpixsize = Q_pixel_size
    qpixunit = Q_pixel_units
    rpixsize = R_pixel_size
    rpixunit = R_pixel_units

    def copy(self):
        """
        Copys datacube
        """
        from py4DSTEM import DataCube

        new_datacube = DataCube(
            data=self.data.copy(),
            name=self.name,
            calibration=self.calibration.copy(),
            slicelabels=self.slicelabels,
        )

        Qpixsize = new_datacube.calibration.get_Q_pixel_size()
        Qpixunits = new_datacube.calibration.get_Q_pixel_units()
        Rpixsize = new_datacube.calibration.get_R_pixel_size()
        Rpixunits = new_datacube.calibration.get_R_pixel_units()

        new_datacube.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        new_datacube.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")

        new_datacube.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        new_datacube.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

        return new_datacube

    # I/O

    # to_h5 is inherited from Array

    # read
    @classmethod
    def _get_constructor_args(cls, group):
        """Construct a datacube with no calibration / metadata"""
        # We only need some of the Array constructors;
        # dim vector/units are passed through when Calibration
        # is loaded, and the runtim dim vectors are then set
        # in _add_root_links
        ar_args = Array._get_constructor_args(group)

        args = {
            "data": ar_args["data"],
            "name": ar_args["name"],
            "slicelabels": ar_args["slicelabels"],
            "calibration": None,
        }

        return args

    def _add_root_links(self, group):
        """When reading from file, link to calibration metadata,
        then use it to populate the datacube dim vectors
        """
        # Link to the datacube
        self.calibration._datacube = self

        # Populate dim vectors
        self.calibration.set_Q_pixel_size(self.calibration.get_Q_pixel_size())
        self.calibration.set_R_pixel_size(self.calibration.get_R_pixel_size())
        self.calibration.set_Q_pixel_units(self.calibration.get_Q_pixel_units())
        self.calibration.set_R_pixel_units(self.calibration.get_R_pixel_units())

        return

    # Class methods

    def add(self, data, name=""):
        """
        Adds a block of data to the DataCube's tree. If `data` is an instance of
        an EMD/py4DSTEM class, add it to the tree.  If it's a numpy array,
        turn it into an Array instance, then save to the tree.
        """
        if isinstance(data, np.ndarray):
            data = Array(data=data, name=name)
        self.attach(data)

    def set_scan_shape(self, Rshape):
        """
        Reshape the data given the real space scan shape.

        Accepts:
            Rshape (2-tuple)
        """
        from py4DSTEM.preprocess import set_scan_shape

        assert len(Rshape) == 2, "Rshape must have a length of 2"
        d = set_scan_shape(self, Rshape[0], Rshape[1])
        return d

    def swap_RQ(self):
        """
        Swaps the first and last two dimensions of the 4D datacube.
        """
        from py4DSTEM.preprocess import swap_RQ

        d = swap_RQ(self)
        return d

    def swap_Rxy(self):
        """
        Swaps the real space x and y coordinates.
        """
        from py4DSTEM.preprocess import swap_Rxy

        d = swap_Rxy(self)
        return d

    def swap_Qxy(self):
        """
        Swaps the diffraction space x and y coordinates.
        """
        from py4DSTEM.preprocess import swap_Qxy

        d = swap_Qxy(self)
        return d

    def crop_Q(self, ROI):
        """
        Crops the data in diffraction space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Qx_min,Qx_max,Qy_min,Qy_max)
        """
        from py4DSTEM.preprocess import crop_data_diffraction

        assert len(ROI) == 4, "Crop region `ROI` must have length 4"
        d = crop_data_diffraction(self, ROI[0], ROI[1], ROI[2], ROI[3])
        return d

    def crop_R(self, ROI):
        """
        Crops the data in real space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Rx_min,Rx_max,Ry_min,Ry_max)
        """
        from py4DSTEM.preprocess import crop_data_real

        assert len(ROI) == 4, "Crop region `ROI` must have length 4"
        d = crop_data_real(self, ROI[0], ROI[1], ROI[2], ROI[3])
        return d

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
        datacube : DataCube
        """
        from py4DSTEM.preprocess import bin_data_diffraction

        d = bin_data_diffraction(self, N, dtype)
        return d

    def pad_Q(self, N=None, output_size=None):
        """
        Pads the data in diffraction space by pad factor N, or to match output_size.

        Accepts:
            N (float, or Sequence[float]): the padding factor
            output_size ((int,int)): the padded output size
        """
        from py4DSTEM.preprocess import pad_data_diffraction

        d = pad_data_diffraction(self, pad_factor=N, output_size=output_size)
        return d

    def resample_Q(self, N=None, output_size=None, method="bilinear"):
        """
        Resamples the data in diffraction space by resampling factor N, or to match output_size,
        using either 'fourier' or 'bilinear' interpolation.

        Accepts:
            N (float, or Sequence[float]): the resampling factor
            output_size ((int,int)): the resampled output size
            method (str): 'fourier' or 'bilinear' (default)
        """
        from py4DSTEM.preprocess import resample_data_diffraction

        d = resample_data_diffraction(
            self, resampling_factor=N, output_size=output_size, method=method
        )
        return d

    def bin_Q_mmap(self, N, dtype=np.float32):
        """
        Bins the data in diffraction space by bin factor N for memory mapped data

        Accepts:
            N (int): the binning factor
            dtype: the data type
        """
        from py4DSTEM.preprocess import bin_data_mmap

        d = bin_data_mmap(self, N)
        return d

    def bin_R(self, N):
        """
        Bins the data in real space by bin factor N

        Accepts:
            N (int): the binning factor
        """
        from py4DSTEM.preprocess import bin_data_real

        d = bin_data_real(self, N)
        return d

    def thin_R(self, N):
        """
        Reduces the data in real space by skipping every N patterns in the x and y directions.

        Accepts:
            N (int): the thinning factor
        """
        from py4DSTEM.preprocess import thin_data_real

        d = thin_data_real(self, N)
        return d

    def filter_hot_pixels(self, thresh, ind_compare=1, return_mask=False):
        """
        This function performs pixel filtering to remove hot / bright pixels. We first compute a moving local ordering filter,
        applied to the mean diffraction image. This ordering filter will return a single value from the local sorted intensity
        values, given by ind_compare. ind_compare=0 would be the highest intensity, =1 would be the second hightest, etc.
        Next, a mask is generated for all pixels which are least a value thresh higher than the local ordering filter output.
        Finally, we loop through all diffraction images, and any pixels defined by mask are replaced by their 3x3 local median.

        Args:
            datacube (DataCube):
            thresh (float): threshold for replacing hot pixels, if pixel value minus local ordering filter exceeds it.
            ind_compare (int): which median filter value to compare against. 0 = brightest pixel, 1 = next brightest, etc.
            return_mask (bool): if True, returns the filter mask

        Returns:
            datacube (DataCube)
            mask (optional, boolean Array) the bad pixel mask
        """
        from py4DSTEM.preprocess import filter_hot_pixels

        d = filter_hot_pixels(
            self,
            thresh,
            ind_compare,
            return_mask,
        )
        return d

    # Probe

    def get_vacuum_probe(
        self,
        ROI=None,
        align=True,
        mask=None,
        threshold=0.2,
        expansion=12,
        opening=3,
        verbose=False,
        returncalc=True,
    ):
        """
        Computes a vacuum probe.

        Which diffraction patterns are included in the calculation is specified
        by the `ROI` parameter.  Diffraction patterns are aligned before averaging
        if `align` is True (default). A global mask is applied to each diffraction
        pattern before aligning/averaging if `mask` is specified. After averaging,
        a final masking step is applied according to the parameters `threshold`,
        `expansion`, and `opening`.

        Parameters
        ----------
        ROI : optional, boolean array or len 4 list/tuple
            If unspecified, uses the whole datacube. If a boolean array is
            passed must be real-space shaped, and True pixels are used. If a
            4-tuple is passed, uses the region inside the limits
            (rx_min,rx_max,ry_min,ry_max)
        align : optional, bool
            if True, aligns the probes before averaging
        mask : optional, array
            mask applied to each diffraction pattern before alignment and
            averaging
        threshold : float
            in the final masking step, values less than max(probe)*threshold
            are considered outside the probe
        expansion : int
            number of pixels by which the final mask is expanded after
            thresholding
        opening : int
            size of binary opening applied to the final mask to eliminate stray
            bright pixels
        verbose : bool
            toggles verbose output
        returncalc : bool
            if True, returns the answer

        Returns
        -------
        probe : Probe, optional
            the vacuum probe
        """
        from py4DSTEM.process.utils import get_shifted_ar, get_shift
        from py4DSTEM.braggvectors import Probe

        # parse region to use
        if ROI is None:
            ROI = np.ones(self.Rshape, dtype=bool)
        elif isinstance(ROI, tuple):
            assert len(ROI) == 4, "if ROI is a tuple must be length 4"
            _ROI = np.ones(self.Rshape, dtype=bool)
            ROI = _ROI[ROI[0] : ROI[1], ROI[2] : ROI[3]]
        else:
            assert isinstance(ROI, np.ndarray)
            assert ROI.shape == self.Rshape
        xy = np.vstack(np.nonzero(ROI))
        length = xy.shape[1]

        # setup global mask
        if mask is None:
            mask = 1
        else:
            assert mask.shape == self.Qshape

        # compute average probe
        probe = self.data[xy[0, 0], xy[1, 0], :, :]
        for n in tqdmnd(range(1, length)):
            curr_DP = self.data[xy[0, n], xy[1, n], :, :] * mask
            if align:
                xshift, yshift = get_shift(probe, curr_DP)
                curr_DP = get_shifted_ar(curr_DP, xshift, yshift)
            probe = probe * (n - 1) / n + curr_DP / n

        # mask
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
        probe *= mask

        # make a probe, add to tree, and return
        probe = Probe(probe)
        self.attach(probe)
        if returncalc:
            return probe

    def get_probe_size(
        self,
        dp=None,
        thresh_lower=0.01,
        thresh_upper=0.99,
        N=100,
        plot=False,
        returncal=True,
        write_to_cal=True,
        **kwargs,
    ):
        """
        Gets the center and radius of the probe in the diffraction plane.

        The algorithm is as follows:
        First, create a series of N binary masks, by thresholding the diffraction
        pattern DP with a linspace of N thresholds from thresh_lower to
        thresh_upper, measured relative to the maximum intensity in DP.
        Using the area of each binary mask, calculate the radius r of a circular
        probe. Because the central disk is typically very intense relative to
        the rest of the DP, r should change very little over a wide range of
        intermediate values of the threshold. The range in which r is trustworthy
        is found by taking the derivative of r(thresh) and finding identifying
        where it is small.  The radius is taken to be the mean of these r values.
        Using the threshold corresponding to this r, a mask is created and the
        CoM of the DP times this mask it taken.  This is taken to be the origin
        x0,y0.

        Args:
            dp (str or array): specifies the diffraction pattern in which to
                find the central disk. A position averaged, or shift-corrected
                and averaged, DP works best. If mode is None, the diffraction
                pattern stored in the tree from 'get_dp_mean' is used. If mode
                is a string it specifies the name of another virtual diffraction
                pattern in the tree. If mode is an array, the array is used to
                calculate probe size.
            thresh_lower (float, 0 to 1): the lower limit of threshold values
            thresh_upper (float, 0 to 1): the upper limit of threshold values
            N (int): the number of thresholds / masks to use
            plot (bool): if True plots results
            plot_params(dict): dictionary to modify defaults in plot
            return_calc (bool): if True returns 3-tuple described below
            write_to_cal (bool): if True, looks for a Calibration instance
                and writes the measured probe radius there

        Returns:
            (3-tuple): A 3-tuple containing:

                * **r**: *(float)* the central disk radius, in pixels
                * **x0**: *(float)* the x position of the central disk center
                * **y0**: *(float)* the y position of the central disk center
        """
        # perform computation
        from py4DSTEM.process.calibration import get_probe_size

        if dp is None:
            assert (
                "dp_mean" in self.treekeys
            ), "calculate .get_dp_mean() or pass a `dp` arg"
            DP = self.tree("dp_mean").data
        elif type(dp) == str:
            assert dp in self.treekeys, f"mode {dp} not found in the tree"
            DP = self.tree(dp)
        elif type(dp) == np.ndarray:
            assert dp.shape == self.Qshape, "must be a diffraction space shape 2D array"
            DP = dp

        x = get_probe_size(
            DP,
            thresh_lower=thresh_lower,
            thresh_upper=thresh_upper,
            N=N,
        )

        # try to add to calibration
        if write_to_cal:
            try:
                self.calibration.set_probe_param(x)
            except AttributeError:
                raise Exception(
                    "writing to calibrations were requested, but could not be completed"
                )

        # plot results
        if plot:
            from py4DSTEM.visualize import show_circles

            show_circles(DP, (x[1], x[2]), x[0], vmin=0, vmax=1, **kwargs)

        # return
        if returncal:
            return x



    ### Bragg disks

    def find_bragg_vectors(
        self,
        template,
        data=None,
        preprocess=None,
        corr = {
            'corrPower' : 1,
            'sigma' : 2,
            'subpixel' : 'poly',
        },
        thresh = {
            'minAbsoluteIntensity' : 0,
            'minRelativeIntensity' : 0.005,
            'relativeToPeak' : 0,
            'minPeakSpacing' : 60,
            'edgeBoundary' : 20,
            'maxNumPeaks' : 70,
        },
        device=None,
        ML=None,
        return_cc = False
        name = 'braggvectors',
        returncalc = True
    ):
        """
        Finds Bragg scattering vectors.

        Template matching is performed by (1) optionally preprocessing the data,
        (2) cross-correlating it with the template and finding the local maxima,
        and (3) thresholding and returning.  These three steps are controlled
        using the `preprocess`, `corr`, and `thresh` arguments, respectively.
        The `data` argument controls which data is analyzed, as well as the
        return value.

        Template matching may be performed on the CPU, one or more GPUs, or
        distributed to a cluster of CPUs.  Device configuration is controlled
        with the `device` argument.

        Vectors may be localized using a neural network, FCU-net (Fourier space,
        complex valued U-net) instead of traditional template matching. FCU-net
        is enabled using the `ML` argument. If you use FCU-net in your work,
        please reference "Munshi, Joydeep, et al. npj Computational Materials
        8.1 (2022): 254".


        Examples (CPU + cross-correlation)
        ----------------------------------

        >>> datacube.get_bragg_vectors( template )

        will find bragg scattering for the entire datacube using cross-
        correlative template matching on the CPU with a correlation power of 1,
        gaussian blurring on each correlagram of 2 pixels, polynomial subpixel
        refinement, and the default thresholding parameters.

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     corr = {
        >>>         'corrPower' : 1,
        >>>         'sigma' : 2,
        >>>         'subpixel' : 'multicorr',
        >>>         'upsample_factor' : 16
        >>>     },
        >>> )

        will perform the same computation but use Fourier upsampling for
        subpixel refinement, and

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     thresh = {
        >>>         'minAboluteIntensity' : 100,
        >>>         'minPeakSpacing' : 18,
        >>>         'edgeBoundary' : 10,
        >>>         'maxNumPeaks' : 100,
        >>>     },
        >>> )

        will perform the same computation but threshold the detected
        maxima using an absolute rather than relative intensity threshhold,
        and modify the other threshold params as above.

        Using

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (5,6)
        >>> )

        will perform template matching against the diffraction image at
        scan position (5,6), and using

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (np.array([4,5,6]),np.array([10,11,12]))
        >>> )

        will perform template matching against the 3 diffraction images at
        scan positions (4,10), (5,11), (6,12).

        Using

        >>> datacube.fing_bragg_vectors(
        >>>     template = None,
        >>>     corr = {
        >>>         'sigma' : 5,
        >>>         'subpixel' : 'poly'
        >>>     },
        >>> )

        will not cross-correlate at all, and will instead perform maximum
        detection on the raw data of the entire datacube, after applying
        a gaussian blur to each diffraction image of 5 pixels, and using
        polynomial subpixel refinement.

        Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'sigma' : 2,
        >>>     },
        >>>     corr = {
        >>>         'sigma' : 4,
        >>>     }
        >>> )

        will apply a 2-pixel gaussian blur to the diffraction image, then
        cross correlate, then apply a 4-pixel gaussian blur to the cross-
        correlation before finding maxima. Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'radial_bkgrd' : True,
        >>>         'localave' : True,
        >>>     },
        >>> )

        will subtract the radial median from each diffraction image, then
        obtain the weighted average diffraction image with a 3x3 gaussian
        footprint in real space (i.e.

            [[ 1, 2, 1 ],
             [ 2, 4, 2 ],  *  (1/16)
             [ 1, 2, 1 ]]

        ) and perform template matching against the resulting images.
        Using

        >>> def preprocess_fn(data):
        >>>     return py4DSTEM.preprocess.bin2D(data,2)
        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'filter_function' : preprocess_fn
        >>>     },
        >>> )

        will bin each diffraction image by 2 before cross-correlating.
        Note that the template shape must match the final, preprocessed
        data shape.


        Examples (GPU acceleration, cluster acceleration, and ML)
        -------------------------------------------------------
        # TODO!


        Parameters
        ----------
        template : qshaped 2d np.ndarray or Probe or None
            The matching template. If an ndarray is passed, must be centered
            about the origin. If a Probe is passed, probe.kernel must be
            populated. If None is passed, cross correlation is skipped.
        data : None or 2-tuple or 2D numpy ndarray
            Specifies the data in which to find the Bragg scattering. Valid
            entries and their behavoirs are:
                * None: use the entire DataCube, and return a BraggVectors
                  instance
                * 2-tuple of ints: use the diffraction pattern at scan
                  position (rx,ry), and return a QPoints instance
                * 2-tuple of arrays of ints: use the diffraction patterns
                  at scan positions (rxs,rys), and return a list of QPoints
                  instances
                * 2D numpy array, real-space shaped, boolean: run on the
                  diffraction images specified by the True pixels in the
                  input array, and return a list of QPoints instances
                * 2D numpy array, diffraction-space shaped: run on the
                  input array, and return a QPoints instance
        preprocess : None or dict
            If None, no preprocessing is performed.  Otherwise, should be a
            dictionary with the following valid keys:
                * radial_bkgrd (bool): if True, finds and subtracts the local
                  median of each diffraction image.  Origin must be calibrated.
                * localave (bool): if True, takes the local 3x3 gaussian
                  average of each diffraction image
                * sigma (number): if >0, applies a gaussian blur to the data
                  before cross correlating
                * filter_function (callable): function applied to each
                  diffraction image before peak finding. Must be a function of
                  only one argument (the diffr image) which returns the pre-
                  processed image. The shape of the returned DP must match
                  the shape of the probe. If using distributed disk detection,
                  the function must be able to be pickled with dill.
            If more than one key is passed then all requested preprocessing
            steps are performed, in the order they're listed here.
        corr : None or dict
            If None, no cross correlation is performed, and maximum detection
            is performed on the (possibly preprocessed) diffraction data with
            no subpixel refinement applied. Otherwise, should be a dictionary
            with valid keys:
                * corrPower (number, 0-1): type of correlation to perform,
                  where 1 is a cross correlation, 0 is a phase correlation,
                  and values in between are hybrid correlations. Pure cross
                  correlation is recommend to minimize noise
                * sigma (number): if >0, apply a gaussian blur to the cross
                  correlation before detecting maxima
                * subpixel ('none' or 'poly' or 'multicorr'): controls
                  subpixel refinement of maxima. 'none' returns the values
                  to pixel precision. 'poly' performs polynomial (2D
                  parabolic) numerical refinement. 'multicorr' performs
                  Fourier upsampling subpixel refinement, and requires
                  the `upsample_factor` keyword also be specified. 'poly'
                  is fast; 'multicorr' is much slower but allows greater
                  precision.
                * upsample_factor (int): the upsampling factor used for
                  'multicorr' subpixel refinement and defining the precision
                  of the refinement. Ignored if `subpixel` is not 'multicorr'
            Note that passing `template=None` skips cross-correlation but not
            maxmimum detection - in this case, `corrPower` is ignored, but all
            parameters in this dictionary are used.
            Note also that if this dictionary is specified (i.e. is not None)
            but corrPower or sigma or subpixel are not specified, their default
            values (corrPower=1, sigma=2, subpixel='poly') are used.
        thresh : None or dict
            If None, no thresholding is performed (not recommended!).  Otherwise,
            should be a dictionary with valid keys:
                * minAbsoluteIntensity (number): maxima with intensities below
                  `this value are removed. Ignored if set to 0.
                * minRelativeIntensity (number): maxima with intensities below a
                  reference maximum * (this value) are removed. The refernce
                  maximum is selected for each diffraction image according to
                  the `relativeToPeak` argument: 0 specifies the brightest
                  maximum, 1 specifies the second brightest, etc.
                * relativeToPeak (int): specifies the reference maximum used
                  int the `minRelativeIntensity` threshold
                * minPeakSpacing (number): if two maxima are closer together than
                  this number of pixels, the dimmer maximum is removed
                * edgeBoundary (number): maxima closer to the edge of the
                  diffraction image than this value are removed
                * maxNumPeaks (int): only the brightest `maxNumPeaks` maxima
                  are kept
        device : None or dict
            If None, uses the CPU.  Otherwise, should be a dictionary with
            valid keys:
                * CUDA (bool): enable GPU acceleration
                * CUDA_batched (bool): enable batched GPU computation
                * ipyparallel (bool): enable multinode distribution using
                  ipyparallel. Must also specify the "client" and "data_file",
                  and optionally "cluster_path" keywords
                * dask (bool): enable multinode distribution using
                  dask. Must also specify the "client" and "data_file",
                  and optionally "cluster_path" keywords
                * client (str or obj): when used with ipyparralel, must be a
                  path to a client json for connecting to your existing
                  IPyParallel cluster. When used with dask, must be a dask
                  client that connects to your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): working directory for cluster processing,
                  defaults to current directory
            Note that only one of 'CUDA' or 'ipyparallel' or 'dask' may be set
            to True, and that 'client' and 'data_file' and optionally
            'cluster_path' must be specified if either 'ipyparallel' or 'dask'
            computation is selected. Note also that preprocessing is currently
            not performed for any device accelerated computations.
        ML : None or dict
            If None, does cross correlative template matching.  Otherwise, should
            be a dictionary with valid keys:
                * num_attempts (int): Number of attempts to predict the Bragg
                  disks. More attempts (ideally) results in a more confident
                  prediction, as FCU-net uses Monte Carlo dropout to estimate
                  the model uncertainty.  Note that increasing num_attempts will
                  increase the compute time and it is adviced to use GPU (CUDA)
                  acceleration for num_attempts > 1.
                  # TODO: @alex-rakowski - you had "Recommended: 5" but also
                  # had the default set to 1, and this comment that using >1
                  # is not recommended without CUDA.  Can we clarify?
                * batch_size (int): number of diffraction images to send to
                  the model at once for prediction. For CPU a batch size of 1
                  is recommended.  For GPU the batch size may be selected based
                  on the available GPU RAM and the size of the diffraction
                  images, with larger batch sizes accelerating the computation
                  and increasing the required memory.
                * model_path (None or str): if None, py4DSTEM will check if a
                  model is available locally, then download and update the model
                  if one is not available or the local model is not up-to-date.
                  Otherwise, must be a filepath to a Tensorflow model of weights
                  to use with FCU-net. Leaving this as None is recommended.
                  # TODO: @alex-rakowski - is this correct? This is what I think
                  # it should do, but looking quickly at the current method it
                  # looks like None always downloads the latest? Is that right?
            Note that to use GPU / batched GPU computation, the "CUDA" and
            "CUDA_batched" flags should be set to True in the `device` arguement.
        return_cc : bool
            If True, returns the cross correlation in addition to the detected
            peaks.
        name : str
            Name for the output BraggVectors instance
        returncalc : bool
            If True, return the answer
        """





        ### OLD CODE

        # parse device config
        if ML:
            mode = "dc_ml"
        # ML arguments
        if ML == True:
            kws["CUDA"] = CUDA
            kws["model_path"] = ml_model_path
            kws["num_attempts"] = ml_num_attempts
            kws["batch_size"] = ml_batch_size

        elif mode == "datacube":
            if distributed is None and CUDA == False:
                mode = "dc_CPU"
            elif distributed is None and CUDA == True:
                if CUDA_batched == False:
                    mode = "dc_GPU"
                else:
                    mode = "dc_GPU_batched"
            else:
                x = _parse_distributed(distributed)
                connect, data_file, cluster_path, distributed_mode = x
                if distributed_mode == "dask":
                    mode = "dc_dask"
                elif distributed_mode == "ipyparallel":
                    mode = "dc_ipyparallel"
                else:
                    er = f"unrecognized distributed mode {distributed_mode}"
                    raise Exception(er)
        # overwrite if ML selected

        # select a function
        fn_dict = {
            "dp": _find_Bragg_disks_single,
            "dp_stack": _find_Bragg_disks_stack,
            "dc_CPU": _find_Bragg_disks_CPU,
            "dc_GPU": _find_Bragg_disks_CUDA_unbatched,
            "dc_GPU_batched": _find_Bragg_disks_CUDA_batched,
            "dc_dask": _find_Bragg_disks_dask,
            "dc_ipyparallel": _find_Bragg_disks_ipp,
            "dc_ml": find_Bragg_disks_aiml,
        }
        fn = fn_dict[mode]

        # prepare kwargs
        kws = {}
        # distributed kwargs
        if distributed is not None:
            kws["connect"] = connect
            kws["data_file"] = data_file
            kws["cluster_path"] = cluster_path
        # ML arguments
        if ML == True:
            kws["CUDA"] = CUDA
            kws["model_path"] = ml_model_path
            kws["num_attempts"] = ml_num_attempts
            kws["batch_size"] = ml_batch_size

        # if radial background subtraction is requested, add to args
        if radial_bksb and mode == "dc_CPU":
            kws["radial_bksb"] = radial_bksb

        # run and return
        ans = fn(
            data,
            template,
            filter_function=filter_function,
            corrPower=corrPower,
            sigma_dp=sigma_dp,
            sigma_cc=sigma_cc,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minPeakSpacing=minPeakSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
            **kws,
        )
        return ans



        # prepare probe



        # parse `data` and compute








        # FROM o.g. datacube.find_Bragg_disks

        from py4DSTEM.braggvectors import find_Bragg_disks

        sigma_cc = sigma if sigma is not None else sigma_cc

        # parse args
        if data is None:
            x = self
        elif isinstance(data, tuple):
            x = self, data[0], data[1]
        elif isinstance(data, np.ndarray):
            assert data.dtype == bool, "array must be boolean"
            assert data.shape == self.Rshape, "array must be Rspace shaped"
            x = self.data[data, :, :]
        else:
            raise Exception(f"unexpected type for `data` {type(data)}")

        # compute
        peaks = find_Bragg_disks(
            data=x,
            template=template,
            radial_bksb=radial_bksb,
            filter_function=filter_function,
            corrPower=corrPower,
            sigma_dp=sigma_dp,
            sigma_cc=sigma_cc,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minPeakSpacing=minPeakSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
            CUDA=CUDA,
            CUDA_batched=CUDA_batched,
            distributed=distributed,
            ML=ML,
            ml_model_path=ml_model_path,
            ml_num_attempts=ml_num_attempts,
            ml_batch_size=ml_batch_size,
        )

        if isinstance(peaks, Node):
            # add metadata
            peaks.name = name
            peaks.metadata = Metadata(
                name="gen_params",
                data={
                    #'gen_func' :
                    "template": template,
                    "filter_function": filter_function,
                    "corrPower": corrPower,
                    "sigma_dp": sigma_dp,
                    "sigma_cc": sigma_cc,
                    "subpixel": subpixel,
                    "upsample_factor": upsample_factor,
                    "minAbsoluteIntensity": minAbsoluteIntensity,
                    "minRelativeIntensity": minRelativeIntensity,
                    "relativeToPeak": relativeToPeak,
                    "minPeakSpacing": minPeakSpacing,
                    "edgeBoundary": edgeBoundary,
                    "maxNumPeaks": maxNumPeaks,
                    "CUDA": CUDA,
                    "CUDA_batched": CUDA_batched,
                    "distributed": distributed,
                    "ML": ML,
                    "ml_model_path": ml_model_path,
                    "ml_num_attempts": ml_num_attempts,
                    "ml_batch_size": ml_batch_size,
                },
            )

            # add to tree
            if data is None:
                self.attach(peaks)

        # return
        if returncalc:
            return peaks

    # aliases
    find_disks = find_bragg = find bragg_disks = find_bragg_scattering = find_bragg_vectors




    #### BEGIN from diskdetection

    def find_Bragg_disks(
        data,
        template,
        radial_bksb=False,
        filter_function=None,
        corrPower=1,
        sigma=None,
        sigma_dp=0,
        sigma_cc=2,
        subpixel="multicorr",
        upsample_factor=16,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0.005,
        relativeToPeak=0,
        minPeakSpacing=60,
        edgeBoundary=20,
        maxNumPeaks=70,
        CUDA=False,
        CUDA_batched=True,
        distributed=None,
        ML=False,
        ml_model_path=None,
        ml_num_attempts=1,
        ml_batch_size=8,
    ):
        """
        """

        # parse args
        sigma_cc = sigma if sigma is not None else sigma_cc

        # `data` type
        if isinstance(data, DataCube):
            mode = "datacube"
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                mode = "dp"
            elif data.ndim == 3:
                mode = "dp_stack"
            else:
                er = f"if `data` is an array, must be 2- or 3-D, not {data.ndim}-D"
                raise Exception(er)
        else:
            try:
                # when a position (rx,ry) is passed, get those patterns
                # and put them in a stack
                dc, rx, ry = data[0], data[1], data[2]

                # h5py datasets have different rules for slicing than
                # numpy arrays, so we have to do this manually
                if "h5py" in str(type(dc.data)):
                    data = np.zeros((len(rx), dc.Q_Nx, dc.Q_Ny))
                    # no background subtraction
                    if not radial_bksb:
                        for i, (x, y) in enumerate(zip(rx, ry)):
                            data[i] = dc.data[x, y]
                    # with bksubtr
                    else:
                        for i, (x, y) in enumerate(zip(rx, ry)):
                            data[i] = dc.get_radial_bksb_dp(rx, ry)
                else:
                    # no background subtraction
                    if not radial_bksb:
                        data = dc.data[np.array(rx), np.array(ry), :, :]
                    # with bksubtr
                    else:
                        data = np.zeros((len(rx), dc.Q_Nx, dc.Q_Ny))
                        for i, (x, y) in enumerate(zip(rx, ry)):
                            data[i] = dc.get_radial_bksb_dp(x, y)
                if data.ndim == 2:
                    mode = "dp"
                elif data.ndim == 3:
                    mode = "dp_stack"
            except:
                er = f"entry {data} for `data` could not be parsed"
                raise Exception(er)

        # CPU/GPU/cluster/ML-AI

        if ML:
            mode = "dc_ml"

        elif mode == "datacube":
            if distributed is None and CUDA == False:
                mode = "dc_CPU"
            elif distributed is None and CUDA == True:
                if CUDA_batched == False:
                    mode = "dc_GPU"
                else:
                    mode = "dc_GPU_batched"
            else:
                x = _parse_distributed(distributed)
                connect, data_file, cluster_path, distributed_mode = x
                if distributed_mode == "dask":
                    mode = "dc_dask"
                elif distributed_mode == "ipyparallel":
                    mode = "dc_ipyparallel"
                else:
                    er = f"unrecognized distributed mode {distributed_mode}"
                    raise Exception(er)
        # overwrite if ML selected

        # select a function
        fn_dict = {
            "dp": _find_Bragg_disks_single,
            "dp_stack": _find_Bragg_disks_stack,
            "dc_CPU": _find_Bragg_disks_CPU,
            "dc_GPU": _find_Bragg_disks_CUDA_unbatched,
            "dc_GPU_batched": _find_Bragg_disks_CUDA_batched,
            "dc_dask": _find_Bragg_disks_dask,
            "dc_ipyparallel": _find_Bragg_disks_ipp,
            "dc_ml": find_Bragg_disks_aiml,
        }
        fn = fn_dict[mode]

        # prepare kwargs
        kws = {}
        # distributed kwargs
        if distributed is not None:
            kws["connect"] = connect
            kws["data_file"] = data_file
            kws["cluster_path"] = cluster_path
        # ML arguments
        if ML == True:
            kws["CUDA"] = CUDA
            kws["model_path"] = ml_model_path
            kws["num_attempts"] = ml_num_attempts
            kws["batch_size"] = ml_batch_size

        # if radial background subtraction is requested, add to args
        if radial_bksb and mode == "dc_CPU":
            kws["radial_bksb"] = radial_bksb

        # run and return
        ans = fn(
            data,
            template,
            filter_function=filter_function,
            corrPower=corrPower,
            sigma_dp=sigma_dp,
            sigma_cc=sigma_cc,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minPeakSpacing=minPeakSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
            **kws,
        )
        return ans


    # Single diffraction pattern


    def _find_Bragg_disks_single(
        DP,
        template,
        filter_function=None,
        corrPower=1,
        sigma_dp=0,
        sigma_cc=2,
        subpixel="poly",
        upsample_factor=16,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minPeakSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=100,
        _return_cc=False,
        _template_space="real",
    ):
        # apply filter function
        er = "filter_function must be callable"
        if filter_function:
            assert callable(filter_function), er
        DP = DP if filter_function is None else filter_function(DP)

        # check for a template
        if template is None:
            cc = DP
        else:
            # fourier transform the template
            assert _template_space in ("real", "fourier")
            if _template_space == "real":
                template_FT = np.conj(np.fft.fft2(template))
            else:
                template_FT = template

            # apply any smoothing to the data
            if sigma_dp > 0:
                DP = gaussian_filter(DP, sigma_dp)

            # Compute cross correlation
            # _returnval = 'fourier' if subpixel == 'multicorr' else 'real'
            cc = get_cross_correlation_FT(
                DP,
                template_FT,
                corrPower,
                "fourier",
            )

        # Get maxima
        maxima = get_maxima_2D(
            np.maximum(np.real(np.fft.ifft2(cc)), 0),
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            sigma=sigma_cc,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minSpacing=minPeakSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
            _ar_FT=cc,
        )

        # Wrap as QPoints instance
        maxima = QPoints(maxima)

        # Return
        if _return_cc is True:
            return maxima, cc
        return maxima


    # def _get_cross_correlation_FT(
    #    DP,
    #    template_FT,
    #    corrPower = 1,
    #    _returnval = 'real'
    #    ):
    #    """
    #    if _returnval is 'real', returns the real-valued cross-correlation.
    #    otherwise, returns the complex valued result.
    #    """
    #
    #    m = np.fft.fft2(DP) * template_FT
    #    cc = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
    #    if _returnval == 'real':
    #        cc = np.maximum(np.real(np.fft.ifft2(cc)),0)
    #    return cc


    # 3D stack of DPs


    def _find_Bragg_disks_stack(
        dp_stack,
        template,
        filter_function=None,
        corrPower=1,
        sigma_dp=0,
        sigma_cc=2,
        subpixel="poly",
        upsample_factor=16,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minPeakSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=100,
        _template_space="real",
    ):
        ans = []

        for idx in range(dp_stack.shape[0]):
            dp = dp_stack[idx, :, :]
            peaks = _find_Bragg_disks_single(
                dp,
                template,
                filter_function=filter_function,
                corrPower=corrPower,
                sigma_dp=sigma_dp,
                sigma_cc=sigma_cc,
                subpixel=subpixel,
                upsample_factor=upsample_factor,
                minAbsoluteIntensity=minAbsoluteIntensity,
                minRelativeIntensity=minRelativeIntensity,
                relativeToPeak=relativeToPeak,
                minPeakSpacing=minPeakSpacing,
                edgeBoundary=edgeBoundary,
                maxNumPeaks=maxNumPeaks,
                _template_space=_template_space,
                _return_cc=False,
            )
            ans.append(peaks)

        return ans


    # Whole datacube, CPU


    def _find_Bragg_disks_CPU(
        datacube,
        probe,
        filter_function=None,
        corrPower=1,
        sigma_dp=0,
        sigma_cc=2,
        subpixel="multicorr",
        upsample_factor=16,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0.005,
        relativeToPeak=0,
        minPeakSpacing=60,
        edgeBoundary=20,
        maxNumPeaks=70,
        radial_bksb=False,
    ):
        # Make the BraggVectors instance
        braggvectors = BraggVectors(datacube.Rshape, datacube.Qshape)

        # Get the template's Fourier Transform
        probe_kernel_FT = np.conj(np.fft.fft2(probe)) if probe is not None else None

        # Loop over all diffraction patterns
        # Compute and populate BraggVectors data
        for rx, ry in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding Bragg Disks",
            unit="DP",
            unit_scale=True,
        ):
            # Get a diffraction pattern

            # without background subtraction
            if not radial_bksb:
                dp = datacube.data[rx, ry, :, :]
            # and with
            else:
                dp = datacube.get_radial_bksb_dp(rx, ry)

            # Compute
            peaks = _find_Bragg_disks_single(
                dp,
                template=probe_kernel_FT,
                filter_function=filter_function,
                corrPower=corrPower,
                sigma_dp=sigma_dp,
                sigma_cc=sigma_cc,
                subpixel=subpixel,
                upsample_factor=upsample_factor,
                minAbsoluteIntensity=minAbsoluteIntensity,
                minRelativeIntensity=minRelativeIntensity,
                relativeToPeak=relativeToPeak,
                minPeakSpacing=minPeakSpacing,
                edgeBoundary=edgeBoundary,
                maxNumPeaks=maxNumPeaks,
                _return_cc=False,
                _template_space="fourier",
            )

            # Populate data
            braggvectors._v_uncal[rx, ry] = peaks

        # Return
        return braggvectors




    #### END from diskdetection

    # in the process of deprecation, as of sep. 3 2023 (-> v0.14.4)
    def find_Bragg_disks(
        self,
        template,
        data=None,
        radial_bksb=False,
        filter_function=None,
        corrPower=1,
        sigma=None,
        sigma_dp=0,
        sigma_cc=2,
        subpixel="multicorr",
        upsample_factor=16,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0.005,
        relativeToPeak=0,
        minPeakSpacing=60,
        edgeBoundary=20,
        maxNumPeaks=70,
        CUDA=False,
        CUDA_batched=True,
        distributed=None,
        ML=False,
        ml_model_path=None,
        ml_num_attempts=1,
        ml_batch_size=8,
        name="braggvectors",
        returncalc=True,
    ):
        """
        Finds the Bragg disks in the diffraction patterns represented by `data` by
        cross/phase correlatin with `template`.

        Behavior depends on `data`. If it is None (default), runs on the whole DataCube,
        and stores the output in its tree. Otherwise, nothing is stored in tree,
        but some value is returned. Valid entries are:

            - a 2-tuple of numbers (rx,ry): run on this diffraction image,
                and return a QPoints instance
            - a 2-tuple of arrays (rx,ry): run on these diffraction images,
                and return a list of QPoints instances
            - an Rspace shapped 2D boolean array: run on the diffraction images
                specified by the True counts and return a list of QPoints
                instances

        For disk detection on a full DataCube, the calculation can be performed
        on the CPU, GPU or a cluster. By default the CPU is used.  If `CUDA` is set
        to True, tries to use the GPU.  If `CUDA_batched` is also set to True,
        batches the FFT/IFFT computations on the GPU. For distribution to a cluster,
        distributed must be set to a dictionary, with contents describing how
        distributed processing should be performed - see below for details.


        For each diffraction pattern, the algorithm works in 4 steps:

        (1) any pre-processing is performed to the diffraction image. This is
            accomplished by passing a callable function to the argument
            `filter_function`, a bool to the argument `radial_bksb`, or a value >0
            to `sigma_dp`. If none of these are passed, this step is skipped.
        (2) the diffraction image is cross correlated with the template.
            Phase/hybrid correlations can be used instead by setting the
            `corrPower` argument. Cross correlation can be skipped entirely,
            and the subsequent steps performed directly on the diffraction
            image instead of the cross correlation, by passing None to
            `template`.
        (3) the maxima of the cross correlation are located and their
            positions and intensities stored. The cross correlation may be
            passed through a gaussian filter first by passing the `sigma_cc`
            argument. The method for maximum detection can be set with
            the `subpixel` parameter. Options, from something like fastest/least
            precise to slowest/most precise are 'pixel', 'poly', and 'multicorr'.
        (4) filtering is applied to remove untrusted or undesired positive counts,
            based on their intensity (`minRelativeIntensity`,`relativeToPeak`,
            `minAbsoluteIntensity`) their proximity to one another or the
            image edge (`minPeakSpacing`, `edgeBoundary`), and the total
            number of peaks per pattern (`maxNumPeaks`).


        Parameters
        ----------
        template : 2D array
            the vacuum probe template, in real space. For Probe instances,
            this is `probe.kernel`.  If None, does not perform a cross
            correlation.
        data : variable
            see above
        radial_bksb : bool
            if True, computes a radial background given by the median of the
            (circular) polar transform of each each diffraction pattern, and
            subtracts this background from the pattern before applying any
            filter function and computing the cross correlation. The origin
            position must be set in the datacube's calibrations. Currently
            only supported for full datacubes on the CPU.
        filter_function : callable
            filtering function to apply to each diffraction pattern before
            peak finding. Must be a function of only one argument (the
            diffraction pattern) and return the filtered diffraction pattern.
            The shape of the returned DP must match the shape of the probe
            kernel (but does not need to match the shape of the input
            diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        corrPower : float between 0 and 1, inclusive
            the cross correlation power. A value of 1 corresponds to a cross
            correlation, 0 corresponds to a phase correlation, and intermediate
            values correspond to hybrid correlations.
        sigma : float
            alias for `sigma_cc`
        sigma_dp : float
            if >0, a gaussian smoothing filter with this standard deviation
            is applied to the diffraction pattern before maxima are detected
        sigma_cc : float
            if >0, a gaussian smoothing filter with this standard deviation
            is applied to the cross correlation before maxima are detected
        subpixel : str
            Whether to use subpixel fitting, and which algorithm to use.
            Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor : int
            upsampling factor for subpixel fitting (only used when
            subpixel='multicorr')
        minAbsoluteIntensity : float
            the minimum acceptable correlation peak intensity, on an absolute scale
        minRelativeIntensity : float
            the minimum acceptable correlation peak intensity, relative to the
            intensity of the brightest peak
        relativeToPeak : int
            specifies the peak against which the minimum relative intensity is
            measured -- 0=brightest maximum. 1=next brightest, etc.
        minPeakSpacing : float
            the minimum acceptable spacing between detected peaks
        edgeBoundary (int): minimum acceptable distance for detected peaks from
            the diffraction image edge, in pixels.
        maxNumPeaks : int
            the maximum number of peaks to return
        CUDA : bool
            If True, import cupy and use an NVIDIA GPU to perform disk detection
        CUDA_batched : bool
            If True, and CUDA is selected, the FFT and IFFT steps of disk detection
            are performed in batches to better utilize GPU resources.
        distributed : dict
            contains information for parallel processing using an IPyParallel or
            Dask distributed cluster.  Valid keys are:
                * ipyparallel (dict):
                * client_file (str): path to client json for connecting to your
                    existing IPyParallel cluster
                * dask (dict): client (object): a dask client that connects to
                    your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): defaults to the working directory during
                    processing
            if distributed is None, which is the default, processing will be in
            serial
        name : str
            name for the output BraggVectors
        returncalc : bool
            if True, returns the answer

        Returns
        -------
        variable
            See above.
        """
        from py4DSTEM.braggvectors import find_Bragg_disks

        sigma_cc = sigma if sigma is not None else sigma_cc

        # parse args
        if data is None:
            x = self
        elif isinstance(data, tuple):
            x = self, data[0], data[1]
        elif isinstance(data, np.ndarray):
            assert data.dtype == bool, "array must be boolean"
            assert data.shape == self.Rshape, "array must be Rspace shaped"
            x = self.data[data, :, :]
        else:
            raise Exception(f"unexpected type for `data` {type(data)}")

        # compute
        peaks = find_Bragg_disks(
            data=x,
            template=template,
            radial_bksb=radial_bksb,
            filter_function=filter_function,
            corrPower=corrPower,
            sigma_dp=sigma_dp,
            sigma_cc=sigma_cc,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minPeakSpacing=minPeakSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
            CUDA=CUDA,
            CUDA_batched=CUDA_batched,
            distributed=distributed,
            ML=ML,
            ml_model_path=ml_model_path,
            ml_num_attempts=ml_num_attempts,
            ml_batch_size=ml_batch_size,
        )

        if isinstance(peaks, Node):
            # add metadata
            peaks.name = name
            peaks.metadata = Metadata(
                name="gen_params",
                data={
                    #'gen_func' :
                    "template": template,
                    "filter_function": filter_function,
                    "corrPower": corrPower,
                    "sigma_dp": sigma_dp,
                    "sigma_cc": sigma_cc,
                    "subpixel": subpixel,
                    "upsample_factor": upsample_factor,
                    "minAbsoluteIntensity": minAbsoluteIntensity,
                    "minRelativeIntensity": minRelativeIntensity,
                    "relativeToPeak": relativeToPeak,
                    "minPeakSpacing": minPeakSpacing,
                    "edgeBoundary": edgeBoundary,
                    "maxNumPeaks": maxNumPeaks,
                    "CUDA": CUDA,
                    "CUDA_batched": CUDA_batched,
                    "distributed": distributed,
                    "ML": ML,
                    "ml_model_path": ml_model_path,
                    "ml_num_attempts": ml_num_attempts,
                    "ml_batch_size": ml_batch_size,
                },
            )

            # add to tree
            if data is None:
                self.attach(peaks)

        # return
        if returncalc:
            return peaks

    def get_beamstop_mask(
        self,
        threshold=0.25,
        distance_edge=2.0,
        include_edges=True,
        sigma=0,
        use_max_dp=False,
        scale_radial=None,
        name="mask_beamstop",
        returncalc=True,
    ):
        """
        This function uses the mean diffraction pattern plus a threshold to
        create a beamstop mask.

        Args:
            threshold (float):  Value from 0 to 1 defining initial threshold for
                beamstop mask, taken from the sorted intensity values - 0 is the
                dimmest pixel, while 1 uses the brighted pixels.
            distance_edge (float): How many pixels to expand the mask.
            include_edges (bool): If set to True, edge pixels will be included
                in the mask.
            sigma (float):
                Gaussain blur std to apply to image before thresholding.
            use_max_dp (bool):
                Use the max DP instead of the mean DP.
            scale_radial (float):
                Scale from center of image by this factor (can help with edge)
            name (string): Name of the output array.
            returncalc (bool): Set to true to return the result.

        Returns:
            (Optional): if returncalc is True, returns the beamstop mask

        """

        if scale_radial is not None:
            x = np.arange(self.data.shape[2]) * 2.0 / self.data.shape[2]
            y = np.arange(self.data.shape[3]) * 2.0 / self.data.shape[3]
            ya, xa = np.meshgrid(y - np.mean(y), x - np.mean(x))
            im_scale = 1.0 + np.sqrt(xa**2 + ya**2) * scale_radial

        # Get image for beamstop mask
        if use_max_dp:
            # if not "dp_mean" in self.tree.keys():
            #     self.get_dp_max();
            # im = self.tree["dp_max"].data.astype('float')
            if not "dp_max" in self._branch.keys():
                self.get_dp_max()
            im = self.tree("dp_max").data.copy().astype("float")
        else:
            if not "dp_mean" in self._branch.keys():
                self.get_dp_mean()
            im = self.tree("dp_mean").data.copy()

            # if not "dp_mean" in self.tree.keys():
            #     self.get_dp_mean();
            # im = self.tree["dp_mean"].data.astype('float')

        # smooth and scale if needed
        if sigma > 0.0:
            im = gaussian_filter(im, sigma, mode="nearest")
        if scale_radial is not None:
            im *= im_scale

        # Calculate beamstop mask
        int_sort = np.sort(im.ravel())
        ind = np.round(
            np.clip(int_sort.shape[0] * threshold, 0, int_sort.shape[0])
        ).astype("int")
        intensity_threshold = int_sort[ind]
        mask_beamstop = im >= intensity_threshold

        # clean up mask
        mask_beamstop = np.logical_not(binary_fill_holes(np.logical_not(mask_beamstop)))
        mask_beamstop = binary_fill_holes(mask_beamstop)

        # Edges
        if include_edges:
            mask_beamstop[0, :] = False
            mask_beamstop[:, 0] = False
            mask_beamstop[-1, :] = False
            mask_beamstop[:, -1] = False

        # Expand mask
        mask_beamstop = distance_transform_edt(mask_beamstop) < distance_edge

        # Wrap beamstop mask in a class
        x = Array(data=mask_beamstop, name=name)

        # Add metadata
        x.metadata = Metadata(
            name="gen_params",
            data={
                #'gen_func' :
                "threshold": threshold,
                "distance_edge": distance_edge,
                "include_edges": include_edges,
                "name": "mask_beamstop",
                "returncalc": returncalc,
            },
        )

        # Add to tree
        self.tree(x)

        # return
        if returncalc:
            return mask_beamstop

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
        assert self.polar is not None, "No polar datacube found!"
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
        datacube.get_radial_background for more info.

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
