# Virtual imaging from a datacube. Includes:
#  * VirtualImage - a container for virtual image data + metadata
#  * DataCubeVirtualImager - methods inherited by DataCube for virt imaging
#
# for bragg virtual imaging methods, goto diskdetection.virtualimage.py

import numpy as np
import dask.array as da
from typing import Optional
import inspect

from emdfile import tqdmnd, Metadata
from py4DSTEM.data import Calibration, RealSlice, Data, DiffractionSlice
from py4DSTEM.preprocess import get_shifted_ar
from py4DSTEM.visualize import show


# Virtual image container class


class VirtualImage(RealSlice, Data):
    """
    A container for storing virtual image data and metadata,
    including the real-space shaped 2D image and metadata
    indicating how this image was generated from a datacube.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "virtualimage",
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            the 2D data
        name : str
            the name
        """
        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            data=data,
            name=name,
        )

    # read
    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = RealSlice._get_constructor_args(group)
        args = {
            "data": ar_constr_args["data"],
            "name": ar_constr_args["name"],
        }
        return args


# DataCube virtual imaging methods


class DataCubeVirtualImager:
    def __init__(self):
        pass

    def get_virtual_image(
        self,
        mode,
        geometry,
        centered=False,
        calibrated=False,
        shift_center=False,
        subpixel=False,
        verbose=True,
        dask=False,
        return_mask=False,
        name="virtual_image",
        returncalc=True,
        test_config=False,
    ):
        """
        Calculate a virtual image.

        The detector is determined by the combination of the `mode` and
        `geometry` arguments, supporting point, circular, rectangular,
        annular, and custom mask detectors. The values passed to geometry
        may be given with respect to an origin at the corner of the detector
        array or with respect to the calibrated center position, and in units of
        pixels or real calibrated units, depending on the values of the
        `centered` and `calibrated` arguments, respectively.  The mask may be
        shifted pattern-by-pattern to account for diffraction scan shifts using
        the `shift_center` argument.

        The computed virtual image is stored in the datacube's tree, and is
        also returned by default.

        Parameters
        ----------
        mode : str
            defines geometry mode for calculating virtual image, and the
            expected input for the `geometry` argument. options:
              - 'point': uses a single pixel detector
              - 'circle', 'circular': uses a round detector, like bright
                field
              - 'annular', 'annulus': uses an annular detector, like dark
                field
              - 'rectangle', 'square', 'rectangular': uses rectangular
                detector
              - 'mask': any diffraction-space shaped 2D array, representing
                a flexible detector
        geometry : variable
            the expected value of this argument is determined by `mode` as
            follows:
              - 'point': 2-tuple, (qx,qy), ints
              - 'circle', 'circular': nested 2-tuple, ((qx,qy),radius),
              - 'annular', 'annulus': nested 2-tuple,
                ((qx,qy),(radius_i,radius_o)),
              - 'rectangle', 'square', 'rectangular': 4-tuple,
                (xmin,xmax,ymin,ymax)
              - `mask`: any boolean or floating point 2D array with the same
                  size as datacube.Qshape
        centered : bool
            if False, the origin is in the upper left corner. If True, the origin
            is set to the mean origin in the datacube calibrations, so that a
            bright-field image could be specified with, e.g., geometry=((0,0),R).
            The origin can set with datacube.calibration.set_origin().  For
            `mode="mask"`, has no effect. Default is False.
        calibrated : bool
            if True, geometry is specified in units of 'A^-1' instead of pixels.
            The datacube's calibrations must have its `"Q_pixel_units"` parameter
            set to "A^-1". For `mode="mask"`, has no effect. Default is False.
        shift_center : bool
            if True, the mask is shifted at each real space position to account
            for any shifting of the origin of the diffraction images. The
            datacube's calibration['origin'] parameter must be set. The shift
            applied to each pattern is the difference between the local origin
            position and the mean origin position over all patterns, rounded to
            the nearest integer for speed. Default is False. If `shift_center` is
            True, `centered` is automatically set to True.
        subpixel : bool
            if True, applies subpixel shifts to virtual image
        verbose : bool
            toggles a progress bar
        dask : bool
            if True, use dask to distribute the calculation
        return_mask : bool
            if False (default) returns a virtual image as usual. Otherwise does
            *not* compute or return a virtual image, instead finding and
            returning the mask that will be used in subsequent calls to this
            function using these same parameters. In this case, must be either
            `True` or a 2-tuple of integers corresponding to `(rx,ry)`. If True
            is passed, returns the mask used if `shift_center` is set to False.
            If a 2-tuple is passed, returns the mask used at scan position
            (rx,ry) if `shift_center` is set to True. Nothing is added to the
            datacube's tree.
        name : str
            the output object's name
        returncalc : bool
            if True, returns the output
        test_config : bool
            if True, prints the Boolean values of
            (`centered`,`calibrated`,`shift_center`). Does not compute the
            virtual image.

        Returns
        -------
        virt_im : VirtualImage (optional, if returncalc is True)
        """
        # parse inputs
        assert mode in (
            "point",
            "circle",
            "circular",
            "annulus",
            "annular",
            "rectangle",
            "square",
            "rectangular",
            "mask",
        ), "check doc strings for supported modes"

        if test_config:
            for x, y in zip(
                ["centered", "calibrated", "shift_center"],
                [centered, calibrated, shift_center],
            ):
                print(f"{x} = {y}")

        # Get geometry
        g = self.get_calibrated_detector_geometry(
            self.calibration, mode, geometry, centered, calibrated
        )

        # Get mask
        mask = self.make_detector(self.Qshape, mode, g)
        # if return_mask is True, skip computation
        if return_mask is True and shift_center is False:
            return mask

        # Calculate virtual image

        # no center shifting
        if shift_center is False:
            # single CPU
            if not dask:
                # allocate space
                if mask.dtype == "complex":
                    virtual_image = np.zeros(self.Rshape, dtype="complex")
                else:
                    virtual_image = np.zeros(self.Rshape)
                # compute
                for rx, ry in tqdmnd(
                    self.R_Nx,
                    self.R_Ny,
                    disable=not verbose,
                ):
                    virtual_image[rx, ry] = np.sum(self.data[rx, ry] * mask)

            # dask
            if dask is True:
                # set up a generalized universal function for dask distribution
                def _apply_mask_dask(self, mask):
                    virtual_image = np.sum(
                        np.multiply(self.data, mask), dtype=np.float64
                    )

                apply_mask_dask = da.as_gufunc(
                    _apply_mask_dask,
                    signature="(i,j),(i,j)->()",
                    output_dtypes=np.float64,
                    axes=[(2, 3), (0, 1), ()],
                    vectorize=True,
                )

                # compute
                virtual_image = apply_mask_dask(self.data, mask)

        # with center shifting
        else:
            # get shifts
            assert (
                self.calibration.get_origin_shift() is not None
            ), "origin need to be calibrated"
            qx_shift, qy_shift = self.calibration.get_origin_shift()
            if subpixel is False:
                qx_shift = qx_shift.round().astype(int)
                qy_shift = qy_shift.round().astype(int)

            # if return_mask is True, get+return the mask and skip the computation
            if return_mask is not False:
                try:
                    rx, ry = return_mask
                except TypeError:
                    raise Exception(
                        f"if `shift_center=True`, return_mask must be a 2-tuple of \
                        ints or False, but revieced inpute value of {return_mask}"
                    )
                if subpixel:
                    _mask = get_shifted_ar(
                        mask, qx_shift[rx, ry], qy_shift[rx, ry], bilinear=True
                    )
                else:
                    _mask = np.roll(
                        mask, (qx_shift[rx, ry], qy_shift[rx, ry]), axis=(0, 1)
                    )
                return _mask

            # allocate space
            if mask.dtype == "complex":
                virtual_image = np.zeros(self.Rshape, dtype="complex")
            else:
                virtual_image = np.zeros(self.Rshape)

            # loop
            for rx, ry in tqdmnd(
                self.R_Nx,
                self.R_Ny,
                disable=not verbose,
            ):
                # get shifted mask
                if subpixel:
                    _mask = get_shifted_ar(
                        mask, qx_shift[rx, ry], qy_shift[rx, ry], bilinear=True
                    )
                else:
                    _mask = np.roll(
                        mask, (qx_shift[rx, ry], qy_shift[rx, ry]), axis=(0, 1)
                    )
                # add to output array
                virtual_image[rx, ry] = np.sum(self.data[rx, ry] * _mask)

        # data handling

        # wrap with a py4dstem class
        ans = VirtualImage(
            data=virtual_image,
            name=name,
        )

        # add generating params as metadata
        ans.metadata = Metadata(
            name="gen_params",
            data={
                "_calling_method": inspect.stack()[0][3],
                "_calling_class": __class__.__name__,
                "mode": mode,
                "geometry": geometry,
                "centered": centered,
                "calibrated": calibrated,
                "shift_center": shift_center,
                "subpixel": subpixel,
                "verbose": verbose,
                "dask": dask,
                "return_mask": return_mask,
                "name": name,
                "returncalc": True,
                "test_config": test_config,
            },
        )

        # add to the tree
        self.attach(ans)

        # return
        if returncalc:
            return ans

    # Position detector

    def position_detector(
        self,
        mode,
        geometry,
        data=None,
        centered=None,
        calibrated=None,
        shift_center=False,
        subpixel=True,
        scan_position=None,
        invert=False,
        color="r",
        alpha=0.7,
        **kwargs,
    ):
        """
        Position a virtual detector by displaying a mask over a diffraction
        space image.  Calling `.get_virtual_image()` using the same `mode`
        and `geometry` parameters will compute a virtual image using this
        detector.

        Parameters
        ----------
        mode : str
            see the DataCube.get_virtual_image docstring
        geometry : variable
            see the DataCube.get_virtual_image docstring
        data : None or 2d-array or 2-tuple of ints
            The diffraction image to overlay the mask on. If `None` (default),
            looks for a max or mean or median diffraction image in this order
            and if found, uses it, otherwise, uses the diffraction pattern at
            scan position (0,0).  If a 2d array is passed, must be diffraction
            space shaped array.  If a 2-tuple is passed, uses the diffraction
            pattern at scan position (rx,ry).
        centered : bool
            see the DataCube.get_virtual_image docstring
        calibrated : bool
            see the DataCube.get_virtual_image docstring
        shift_center : None or bool or 2-tuple of ints
            If `None` (default) and `data` is either None or an array, the mask
            is not shifted.  If `None` and `data` is a 2-tuple, shifts the mask
            according to the origin at the scan position (rx,ry) specified in
            `data`.  If False, does not shift the mask.  If True and `data` is
            a 2-tuple, shifts the mask accordingly, and if True and `data` is
            any other value, raises an error.  If `shift_center` is a 2-tuple,
            shifts the mask according to the origin value at this 2-tuple
            regardless of the value of `data` (enabling e.g. overlaying the
            mask for a specific scan position on a max or mean diffraction
            image.)
        subpixel : bool
            if True, applies subpixel shifts to virtual image
        invert : bool
            if True, invert the masked pixel (i.e. pixels *outside* the detector
            are overlaid with a mask)
        color : any matplotlib color specification
            the mask color
        alpha : number
            the mask transparency
        kwargs : dict
            Any additional arguments are passed on to the show() function
        """
        # parse inputs

        # mode
        assert mode in (
            "point",
            "circle",
            "circular",
            "annulus",
            "annular",
            "rectangle",
            "square",
            "rectangular",
            "mask",
        ), "check doc strings for supported modes"

        # data
        if data is None:
            image = None
            keys = ["dp_mean", "dp_max", "dp_median"]
            for k in keys:
                try:
                    image = self.tree(k)
                    break
                except:
                    pass
            if image is None:
                image = self[0, 0]
        elif isinstance(data, np.ndarray):
            assert (
                data.shape == self.Qshape
            ), f"Can't position a detector over an image with a shape that is different \
                from diffraction space.  Diffraction space in this dataset has shape {self.Qshape} \
                but the image passed has shape {data.shape}"
            image = data
        elif isinstance(data, DiffractionSlice):
            assert (
                data.shape == self.Qshape
            ), f"Can't position a detector over an image with a shape that is different \
                from diffraction space.  Diffraction space in this dataset has shape {self.Qshape} \
                but the image passed has shape {data.shape}"
            image = data.data
        elif isinstance(data, tuple):
            rx, ry = data[:2]
            image = self[rx, ry]
        else:
            raise Exception(
                f"Invalid argument passed to `data`. Expected None or np.ndarray or \
                    tuple, not type {type(data)}"
            )

        # shift center
        if shift_center is None:
            shift_center = False
        elif shift_center is True:
            assert isinstance(
                data, tuple
            ), "If shift_center is set to True, `data` should be a 2-tuple (rx,ry). \
                To shift the detector mask while using some other input for `data`, \
                set `shift_center` to a 2-tuple (rx,ry)"
        elif isinstance(shift_center, tuple):
            rx, ry = shift_center[:2]
            shift_center = True
        else:
            shift_center = False

        # Get the mask

        # Get geometry
        g = self.get_calibrated_detector_geometry(
            calibration=self.calibration,
            mode=mode,
            geometry=geometry,
            centered=centered,
            calibrated=calibrated,
        )

        # Get mask
        mask = self.make_detector(image.shape, mode, g)
        if not (invert):
            mask = np.logical_not(mask)

        # Shift center
        if shift_center:
            try:
                rx, ry
            except NameError:
                raise Exception(
                    "if `shift_center` is True then `data` must be the 3-tuple (DataCube,rx,ry)"
                )
            # get shifts
            assert (
                self.calibration.get_origin_shift() is not None
            ), "origin shifts need to be calibrated"
            qx_shift, qy_shift = self.calibration.get_origin_shift()
            if subpixel:
                mask = get_shifted_ar(
                    mask, qx_shift[rx, ry], qy_shift[rx, ry], bilinear=True
                )
            else:
                qx_shift = int(np.round(qx_shift[rx, ry]))
                qy_shift = int(np.round(qy_shift[rx, ry]))
                mask = np.roll(mask, (qx_shift, qy_shift), axis=(0, 1))

        # Show
        show(image, mask=mask, mask_color=color, mask_alpha=alpha, **kwargs)
        return

    @staticmethod
    def get_calibrated_detector_geometry(
        calibration, mode, geometry, centered, calibrated
    ):
        """
        Determine the detector geometry in pixels, given some mode and geometry
        in calibrated units, where the calibration state is specified by {
        centered, calibrated}

        Parameters
        ----------
        calibration : Calibration
            Used to retrieve the center positions. If `None`, confirms that
            centered and calibrated are False then passes, otherwise raises
            an exception
        mode : str
            see the DataCube.get_virtual_image docstring
        geometry : variable
            see the DataCube.get_virtual_image docstring
        centered : bool
            see the DataCube.get_virtual_image docstring
        calibrated : bool
            see the DataCube.get_virtual_image docstring

        Returns
        -------
        geo : tuple
            the geometry in detector pixels
        """
        # Parse inputs
        g = geometry
        if calibration is None:
            assert (
                calibrated is False and centered is False
            ), "No calibration found - set a calibration or set `centered` and `calibrated` to False"
            return g
        else:
            assert isinstance(calibration, Calibration)
            cal = calibration

        # Get calibration metadata
        if centered:
            assert cal.get_qx0_mean() is not None, "origin needs to be calibrated"
            x0_mean, y0_mean = cal.get_origin_mean()

        if calibrated:
            assert (
                cal["Q_pixel_units"] == "A^-1"
            ), "check calibration - must be calibrated in A^-1 to use `calibrated=True`"
            unit_conversion = cal.get_Q_pixel_size()

        # Convert units into detector pixels

        # Shift center
        if centered is True:
            if mode == "point":
                g = (g[0] + x0_mean, g[1] + y0_mean)
            if mode in ("circle", "circular", "annulus", "annular"):
                g = ((g[0][0] + x0_mean, g[0][1] + y0_mean), g[1])
            if mode in ("rectangle", "square", "rectangular"):
                g = (g[0] + x0_mean, g[1] + x0_mean, g[2] + y0_mean, g[3] + y0_mean)

        # Scale by the detector pixel size
        if calibrated is True:
            if mode == "point":
                g = (g[0] / unit_conversion, g[1] / unit_conversion)
            if mode in ("circle", "circular"):
                g = (
                    (g[0][0] / unit_conversion, g[0][1] / unit_conversion),
                    (g[1] / unit_conversion),
                )
            if mode in ("annulus", "annular"):
                g = (
                    (g[0][0] / unit_conversion, g[0][1] / unit_conversion),
                    (g[1][0] / unit_conversion, g[1][1] / unit_conversion),
                )
            if mode in ("rectangle", "square", "rectangular"):
                g = (
                    g[0] / unit_conversion,
                    g[1] / unit_conversion,
                    g[2] / unit_conversion,
                    g[3] / unit_conversion,
                )

        return g

    @staticmethod
    def make_detector(
        shape,
        mode,
        geometry,
    ):
        """
        Generate a 2D mask representing a detector function.

        Parameters
        ----------
        shape : 2-tuple
            defines shape of mask. Should be the shape of diffraction space.
        mode : str
            defines geometry mode for calculating virtual image. See the
            docstring for DataCube.get_virtual_image
        geometry : variable
            defines geometry for calculating virtual image. See the
            docstring for DataCube.get_virtual_image

        Returns
        -------
        detector_mask : 2d array
        """
        g = geometry

        # point mask
        if mode == "point":
            assert (
                isinstance(g, tuple) and len(g) == 2
            ), "specify qx and qy as tuple (qx, qy)"
            mask = np.zeros(shape, dtype=bool)

            qx = int(g[0])
            qy = int(g[1])

            mask[qx, qy] = 1

        # circular mask
        if mode in ("circle", "circular"):
            assert (
                isinstance(g, tuple)
                and len(g) == 2
                and len(g[0]) == 2
                and isinstance(g[1], (float, int))
            ), "specify qx, qy, radius_i as ((qx, qy), radius)"

            qxa, qya = np.indices(shape)
            mask = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1] ** 2

        # annular mask
        if mode in ("annulus", "annular"):
            assert (
                isinstance(g, tuple)
                and len(g) == 2
                and len(g[0]) == 2
                and len(g[1]) == 2
            ), "specify qx, qy, radius_i, radius_0 as ((qx, qy), (radius_i, radius_o))"

            assert g[1][1] > g[1][0], "Inner radius must be smaller than outer radius"

            qxa, qya = np.indices(shape)
            mask1 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 > g[1][0] ** 2
            mask2 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1][1] ** 2
            mask = np.logical_and(mask1, mask2)

        # rectangle mask
        if mode in ("rectangle", "square", "rectangular"):
            assert (
                isinstance(g, tuple) and len(g) == 4
            ), "specify x_min, x_max, y_min, y_max as (x_min, x_max, y_min, y_max)"
            mask = np.zeros(shape, dtype=bool)

            xmin = int(np.round(g[0]))
            xmax = int(np.round(g[1]))
            ymin = int(np.round(g[2]))
            ymax = int(np.round(g[3]))

            mask[xmin:xmax, ymin:ymax] = 1

        # flexible mask
        if mode == "mask":
            assert type(g) == np.ndarray, "`geometry` type should be `np.ndarray`"
            assert g.shape == shape, "mask and diffraction pattern shapes do not match"
            mask = g
        return mask

    # TODO where should this go?
    def make_bragg_mask(
        self,
        Qshape,
        g1,
        g2,
        radius,
        origin,
        max_q,
        return_sum=True,
        include_origin=True,
        rotation_deg=0,
        **kwargs,
    ):
        """
        Creates and returns a mask consisting of circular disks
        about the points of a 2D lattice.

        Args:
            Qshape (2 tuple): the shape of diffraction space
            g1,g2 (len 2 array or tuple): the lattice vectors
            radius (number): the disk radius
            origin (len 2 array or tuple): the origin
            max_q (nuumber): the maxima distance to tile to
            return_sum (bool): if False, return a 3D array, where each
                slice contains a single disk; if False, return a single
                2D masks of all disks
            include_origin (bool) : if False, removes origin disk
            rotation_deg (float) : rotate g1 and g2 vectors

        Returns:
            (2 or 3D array) the mask
        """
        g1, g2, origin = np.asarray(g1), np.asarray(g2), np.asarray(origin)

        rotation_rad = np.deg2rad(rotation_deg)
        cost = np.cos(rotation_rad)
        sint = np.sin(rotation_rad)
        rotation_matrix = np.array(((cost, sint), (-sint, cost)))

        g1 = np.dot(g1, rotation_matrix)
        g2 = np.dot(g2, rotation_matrix)

        # Get N,M, the maximum indices to tile out to
        L1 = np.sqrt(np.sum(g1**2))
        H = int(max_q / L1) + 1
        L2 = np.hypot(-g2[0] * g1[1], g2[1] * g1[0]) / np.sqrt(np.sum(g1**2))
        K = int(max_q / L2) + 1

        # Compute number of points
        N = 0
        for h in range(-H, H + 1):
            for k in range(-K, K + 1):
                v = h * g1 + k * g2
                if np.sqrt(v.dot(v)) < max_q:
                    N += 1

        # create mask
        mask = np.zeros((Qshape[0], Qshape[1], N), dtype=bool)
        N = 0
        for h in range(-H, H + 1):
            for k in range(-K, K + 1):
                if h == 0 and k == 0 and include_origin is False:
                    continue
                else:
                    v = h * g1 + k * g2
                    if np.sqrt(v.dot(v)) < max_q:
                        center = origin + v
                        mask[:, :, N] = self.make_detector(
                            Qshape,
                            mode="circle",
                            geometry=(center, radius),
                        )
                        N += 1

        if return_sum:
            mask = np.sum(mask, axis=2)
        return mask
