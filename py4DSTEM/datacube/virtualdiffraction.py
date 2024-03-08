# Virtual diffraction from a self. Includes:
#  * VirtualDiffraction - a container for virtual diffraction data + metadata
#  * DataCubeVirtualDiffraction - methods inherited by DataCube for virt diffraction

import numpy as np
from typing import Optional
import inspect

from emdfile import tqdmnd, Metadata
from py4DSTEM.data import DiffractionSlice, Data
from py4DSTEM.preprocess import get_shifted_ar

# Virtual diffraction container class


class VirtualDiffraction(DiffractionSlice, Data):
    """
    Stores a diffraction-space shaped 2D image with metadata
    indicating how this image was generated from a self.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "virtualdiffraction",
    ):
        """
        Args:
            data (np.ndarray) : the 2D data
            name (str) : the name

        Returns:
            A new VirtualDiffraction instance
        """
        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
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
        ar_constr_args = DiffractionSlice._get_constructor_args(group)
        args = {
            "data": ar_constr_args["data"],
            "name": ar_constr_args["name"],
        }
        return args


# DataCube virtual diffraction methods


class DataCubeVirtualDiffraction:
    def __init__(self):
        pass

    def get_virtual_diffraction(
        self,
        method,
        mask=None,
        shift_center=False,
        subpixel=False,
        verbose=True,
        name="virtual_diffraction",
        returncalc=True,
    ):
        """
        Function to calculate virtual diffraction images.

        Parameters
        ----------
        method : str
            defines method used for averaging/combining diffraction patterns.
            Options are ('mean', 'median', 'max')
        mask : None or 2D array
            if None (default), all pixels are used. Otherwise, must be a boolean
            or floating point or complex array with the same shape as real space.
            For bool arrays, only True pixels are used in the computation.
            Otherwise a weighted average is performed.
        shift_center : bool
            toggles shifting the diffraction patterns to account for beam shift.
            Currently only supported for 'max' and 'mean' modes. Default is
            False.
        subpixel : bool
            if shift_center is True, toggles subpixel shifts via Fourier
            interpolation. Ignored if shift_center is False.
        verbose : bool
            toggles progress bar
        name : string
            name for the output DiffractionImage instance
        returncalc : bool
            toggles returning the output

        Returns
        -------
        diff_im : DiffractionImage
        """
        # parse inputs
        assert method in (
            "max",
            "median",
            "mean",
        ), "check doc strings for supported types"
        assert (
            mask is None or mask.shape == self.Rshape
        ), "mask must be None or real-space shaped"

        # Calculate

        # ...with no center shifting
        if shift_center is False:
            # ...for the whole pattern
            if mask is None:
                if method == "mean":
                    virtual_diffraction = np.mean(self.data, axis=(0, 1))
                elif method == "max":
                    virtual_diffraction = np.max(self.data, axis=(0, 1))
                else:
                    virtual_diffraction = np.median(self.data, axis=(0, 1))

            # ...for boolean masks
            elif mask.dtype == bool:
                mask_indices = np.nonzero(mask)
                if method == "mean":
                    virtual_diffraction = np.mean(
                        self.data[mask_indices[0], mask_indices[1], :, :], axis=0
                    )
                elif method == "max":
                    virtual_diffraction = np.max(
                        self.data[mask_indices[0], mask_indices[1], :, :], axis=0
                    )
                else:
                    virtual_diffraction = np.median(
                        self.data[mask_indices[0], mask_indices[1], :, :], axis=0
                    )

            # ...for complex and floating point masks
            else:
                # allocate space
                if mask.dtype == "complex":
                    virtual_diffraction = np.zeros(self.Qshape, dtype="complex")
                else:
                    virtual_diffraction = np.zeros(self.Qshape)
                # set computation method
                if method == "mean":
                    fn = np.sum
                elif method == "max":
                    fn = np.max
                else:
                    fn = np.median
                # loop
                for qx, qy in tqdmnd(
                    self.Q_Nx,
                    self.Q_Ny,
                    disable=not verbose,
                ):
                    virtual_diffraction[qx, qy] = fn(
                        np.squeeze(self.data[:, :, qx, qy]) * mask
                    )
                # normalize weighted means
                if method == "mean":
                    virtual_diffraction /= np.sum(mask)

        # ...with center shifting
        else:
            assert method in (
                "max",
                "mean",
            ), "only 'mean' and 'max' are supported for center-shifted virtual diffraction"

            # Get calibration metadata
            assert self.calibration.get_origin() is not None, "origin is not calibrated"
            x0, y0 = self.calibration.get_origin()
            x0_mean, y0_mean = self.calibration.get_origin_mean()

            # get shifts
            qx_shift = x0_mean - x0
            qy_shift = y0_mean - y0

            if subpixel is False:
                # round shifts -> int
                qx_shift = qx_shift.round().astype(int)
                qy_shift = qy_shift.round().astype(int)

            # ...for boolean masks and unmasked
            if mask is None or mask.dtype == bool:
                # get scan points
                mask = np.ones(self.Rshape, dtype=bool) if mask is None else mask
                mask_indices = np.nonzero(mask)
                # allocate space
                virtual_diffraction = np.zeros(self.Qshape)
                # loop
                for rx, ry in zip(mask_indices[0], mask_indices[1]):
                    # get shifted DP
                    if subpixel:
                        DP = get_shifted_ar(
                            self.data[
                                rx,
                                ry,
                                :,
                                :,
                            ],
                            qx_shift[rx, ry],
                            qy_shift[rx, ry],
                        )
                    else:
                        DP = np.roll(
                            self.data[
                                rx,
                                ry,
                                :,
                                :,
                            ],
                            (qx_shift[rx, ry], qy_shift[rx, ry]),
                            axis=(0, 1),
                        )
                    # compute
                    if method == "mean":
                        virtual_diffraction += DP
                    elif method == "max":
                        virtual_diffraction = np.maximum(virtual_diffraction, DP)
                # normalize means
                if method == "mean":
                    virtual_diffraction /= len(mask_indices[0])

            # ...for floating point and complex masks
            else:
                # allocate space
                if mask.dtype == "complex":
                    virtual_diffraction = np.zeros(self.Qshape, dtype="complex")
                else:
                    virtual_diffraction = np.zeros(self.Qshape)
                # loop
                for rx, ry in tqdmnd(
                    self.R_Nx,
                    self.R_Ny,
                    disable=not verbose,
                ):
                    # get shifted DP
                    if subpixel:
                        DP = get_shifted_ar(
                            self.data[
                                rx,
                                ry,
                                :,
                                :,
                            ],
                            qx_shift[rx, ry],
                            qy_shift[rx, ry],
                        )
                    else:
                        DP = np.roll(
                            self.data[
                                rx,
                                ry,
                                :,
                                :,
                            ],
                            (qx_shift[rx, ry], qy_shift[rx, ry]),
                            axis=(0, 1),
                        )

                    # compute
                    w = mask[rx, ry]
                    if method == "mean":
                        virtual_diffraction += DP * w
                    elif method == "max":
                        virtual_diffraction = np.maximum(virtual_diffraction, DP * w)
                if method == "mean":
                    virtual_diffraction /= np.sum(mask)

        # wrap, add to tree, and return

        # wrap in DiffractionImage
        ans = VirtualDiffraction(data=virtual_diffraction, name=name)

        # add the args used to gen this dp as metadata
        ans.metadata = Metadata(
            name="gen_params",
            data={
                "_calling_method": inspect.stack()[0][3],
                "_calling_class": __class__.__name__,
                "method": method,
                "mask": mask,
                "shift_center": shift_center,
                "subpixel": subpixel,
                "verbose": verbose,
                "name": name,
                "returncalc": returncalc,
            },
        )

        # add to the tree
        self.attach(ans)

        # return
        if returncalc:
            return ans

    # additional interfaces

    def get_dp_max(
        self,
        returncalc=True,
    ):
        """
        Calculates the max diffraction pattern.

        Calls `DataCube.get_virtual_diffraction` - see that method's docstring
        for more custimizable virtual diffraction.

        Parameters
        ----------
        returncalc : bool
            toggles returning the answer

        Returns
        -------
        max_dp : VirtualDiffraction
        """
        return self.get_virtual_diffraction(
            method="max",
            mask=None,
            shift_center=False,
            subpixel=False,
            verbose=True,
            name="dp_max",
            returncalc=True,
        )

    def get_dp_mean(
        self,
        returncalc=True,
    ):
        """
        Calculates the mean diffraction pattern.

        Calls `DataCube.get_virtual_diffraction` - see that method's docstring
        for more custimizable virtual diffraction.

        Parameters
        ----------
        returncalc : bool
            toggles returning the answer

        Returns
        -------
        mean_dp : VirtualDiffraction
        """
        return self.get_virtual_diffraction(
            method="mean",
            mask=None,
            shift_center=False,
            subpixel=False,
            verbose=True,
            name="dp_mean",
            returncalc=True,
        )

    def get_dp_median(
        self,
        returncalc=True,
    ):
        """
        Calculates the max diffraction pattern.

        Calls `DataCube.get_virtual_diffraction` - see that method's docstring
        for more custimizable virtual diffraction.

        Parameters
        ----------
        returncalc : bool
            toggles returning the answer

        Returns
        -------
        max_dp : VirtualDiffraction
        """
        return self.get_virtual_diffraction(
            method="median",
            mask=None,
            shift_center=False,
            subpixel=False,
            verbose=True,
            name="dp_median",
            returncalc=True,
        )
