# Defines the Strain class

import warnings
from typing import Optional, List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from py4DSTEM import PointList, PointListArray, tqdmnd
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.data import Data, RealSlice
from py4DSTEM.preprocess.utils import get_maxima_2D
from py4DSTEM.process.strain.latticevectors import (
    fit_lattice_vectors_all_DPs,
    get_reference_g1g2,
    get_rotated_strain_map,
    get_strain_from_reference_g1g2,
    index_bragg_directions,
)
from py4DSTEM.visualize import (
    show,
    add_bragg_index_labels,
    add_pointlabels,
    add_vector,
)


class StrainMap(RealSlice, Data):
    """
    Storage and processing methods for 4D-STEM datasets.

    """

    def __init__(self, braggvectors: BraggVectors, name: Optional[str] = "strainmap"):
        """
        Parameters
        ----------
        braggvectors : BraggVectors
            The Bragg vectors
        name : str
            The name of the strainmap

        Returns
        -------
        A new StrainMap instance.
        """
        assert isinstance(
            braggvectors, BraggVectors
        ), f"braggvectors must be BraggVectors, not type {type(braggvectors)}"

        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            name=name,
            data=np.empty(
                (
                    6,
                    braggvectors.Rshape[0],
                    braggvectors.Rshape[1],
                )
            ),
            slicelabels=["e_xx", "e_yy", "e_xy", "theta", "mask", "error"],
        )

        # set up braggvectors
        # this assigns the bvs, ensures the origin is calibrated,
        # and adds the strainmap to the bvs' tree
        self.braggvectors = braggvectors

        # initialize as Data
        Data.__init__(self)

        # set calstate
        # this property is used only to check to make sure that
        # the braggvectors being used throughout a workflow are
        # the same. The state of calibration of the vectors is noted
        # here, and then checked each time the vectors are used -
        # if they differ, an error message and instructions for
        # re-calibration are issued
        self.calstate = self.braggvectors.calstate
        assert self.calstate["center"], "braggvectors must be centered"
        if self.calstate["rotate"] is False:
            warnings.warn(
                ("Real to reciprocal space rotation not calibrated"),
                UserWarning,
            )

        # get the BVM
        # a new BVM using the current calstate is computed
        self.bvm = self.braggvectors.histogram(mode="cal")

    # braggvector properties

    @property
    def braggvectors(self):
        return self._braggvectors

    @braggvectors.setter
    def braggvectors(self, x):
        assert isinstance(
            x, BraggVectors
        ), f".braggvectors must be BraggVectors, not type {type(x)}"
        assert (
            x.calibration.origin is not None
        ), "braggvectors must have a calibrated origin"
        self._braggvectors = x
        self._braggvectors.tree(self, force=True)

    @property
    def rshape(self):
        return self._braggvectors.Rshape

    @property
    def qshape(self):
        return self._braggvectors.Qshape

    @property
    def origin(self):
        return self.calibration.get_origin_mean()

    def reset_calstate(self):
        """
        Resets the calibration state. This recomputes the BVM, and removes any computations
        this StrainMap instance has stored, which will need to be recomputed.
        """
        for attr in (
            "g0",
            "g1",
            "g2",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        self.calstate = self.braggvectors.calstate
        pass

    # Class methods

    def choose_basis_vectors(
        self,
        index_g1=None,
        index_g2=None,
        index_origin=None,
        subpixel="multicorr",
        upsample_factor=16,
        sigma=0,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=10,
        x0=None,
        y0=None,
        figsize=(14, 9),
        c_indices="lightblue",
        c0="g",
        c1="r",
        c2="r",
        c_vectors="r",
        c_vectorlabels="w",
        size_indices=15,
        width_vectors=1,
        size_vectorlabels=15,
        vis_params={},
        returncalc=False,
        returnfig=False,
    ):
        """
        Choose basis lattice vectors g1 and g2 for strain mapping.

        Overlays the bvm with the points detected via local 2D
        maxima detection, plus an index for each point. Three points
        are selected which correspond to the origin, and the basis
        reciprocal lattice vectors g1 and g2. By default these are
        automatically located; the user can override and select these
        manually using the `index_*` arguments.

        Parameters
        ----------
        index_g1 : int
            selected index for g1
        index_g2 :int
            selected index for g2
        index_origin : int
            selected index for the origin
        subpixel : str in ('pixel','poly','multicorr')
            See the docstring for py4DSTEM.preprocess.get_maxima_2D
        upsample_factor : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        sigma : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minAbsoluteIntensity : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minRelativeIntensity : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        relativeToPeak : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minSpacing : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        edgeBoundary : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        maxNumPeaks : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        figsize : 2-tuple
            the size of the figure
        c_indices : color
            color of the maxima
        c0 : color
            color of the origin
        c1 : color
            color of g1 point
        c2 : color
            color of g2 point
        c_vectors : color
            color of the g1/g2 vectors
        c_vectorlabels : color
            color of the vector labels
        size_indices : number
            size of the indices
        width_vectors : number
            width of the vectors
        size_vectorlabels : number
            size of the vector labels
        vis_params : dict
            additional visualization parameters passed to `show`
        returncalc : bool
            toggles returning the answer
        returnfig : bool
            toggles returning the figure

        Returns
        -------
        (optional) : None or (g0,g1,g2) or (fig,(ax1,ax2)) or the latter two
        """
        # validate inputs
        for i in (index_origin, index_g1, index_g2):
            assert isinstance(i, (int, np.integer)) or (
                i is None
            ), "indices must be integers!"
        # check the calstate
        assert (
            self.calstate == self.braggvectors.calstate
        ), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        # find the maxima

        g = get_maxima_2D(
            self.bvm.data,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            sigma=sigma,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            relativeToPeak=relativeToPeak,
            minSpacing=minSpacing,
            edgeBoundary=edgeBoundary,
            maxNumPeaks=maxNumPeaks,
        )

        # guess the origin and g1 g2 vectors if indices aren't provided
        if np.any([x is None for x in (index_g1, index_g2, index_origin)]):
            # get distances and angles from calibrated origin
            g_dists = np.hypot(g["x"] - self.origin[0], g["y"] - self.origin[1])
            g_angles = np.angle(
                g["x"] - self.origin[0] + 1j * (g["y"] - self.origin[1])
            )

            # guess the origin
            if index_origin is None:
                index_origin = np.argmin(g_dists)
                g_dists[index_origin] = 2 * np.max(g_dists)

            # guess g1
            if index_g1 is None:
                index_g1 = np.argmin(g_dists)
                g_dists[index_g1] = 2 * np.max(g_dists)

            # guess g2
            if index_g2 is None:
                angle_scaling = np.cos(g_angles - g_angles[index_g1]) ** 2
                index_g2 = np.argmin(g_dists * (angle_scaling + 0.1))

        # get the lattice vectors
        gx, gy = g["x"], g["y"]
        g0 = gx[index_origin], gy[index_origin]
        g1x = gx[index_g1] - g0[0]
        g1y = gy[index_g1] - g0[1]
        g2x = gx[index_g2] - g0[0]
        g2y = gy[index_g2] - g0[1]
        g1, g2 = (g1x, g1y), (g2x, g2y)

        # index the lattice vectors
        _, _, braggdirections = index_bragg_directions(
            g0[0], g0[1], g["x"], g["y"], g1, g2
        )

        # make the figure
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        show(self.bvm.data, figax=(fig, ax[0]), **vis_params)
        show(self.bvm.data, figax=(fig, ax[1]), **vis_params)
        self.show_bragg_indexing(
            self.bvm.data,
            bragg_directions=braggdirections,
            points=True,
            figax=(fig, ax[2]),
            size=size_indices,
            **vis_params,
        )

        # Add indices to left panel
        d = {"x": gx, "y": gy, "size": size_indices, "color": c_indices}
        d0 = {
            "x": gx[index_origin],
            "y": gy[index_origin],
            "size": size_indices,
            "color": c0,
            "fontweight": "bold",
            "labels": [str(index_origin)],
        }
        d1 = {
            "x": gx[index_g1],
            "y": gy[index_g1],
            "size": size_indices,
            "color": c1,
            "fontweight": "bold",
            "labels": [str(index_g1)],
        }
        d2 = {
            "x": gx[index_g2],
            "y": gy[index_g2],
            "size": size_indices,
            "color": c2,
            "fontweight": "bold",
            "labels": [str(index_g2)],
        }
        add_pointlabels(ax[0], d)
        add_pointlabels(ax[0], d0)
        add_pointlabels(ax[0], d1)
        add_pointlabels(ax[0], d2)

        # Add vectors to right panel
        dg1 = {
            "x0": gx[index_origin],
            "y0": gy[index_origin],
            "vx": g1[0],
            "vy": g1[1],
            "width": width_vectors,
            "color": c_vectors,
            "label": r"$g_1$",
            "labelsize": size_vectorlabels,
            "labelcolor": c_vectorlabels,
        }
        dg2 = {
            "x0": gx[index_origin],
            "y0": gy[index_origin],
            "vx": g2[0],
            "vy": g2[1],
            "width": width_vectors,
            "color": c_vectors,
            "label": r"$g_2$",
            "labelsize": size_vectorlabels,
            "labelcolor": c_vectorlabels,
        }
        add_vector(ax[1], dg1)
        add_vector(ax[1], dg2)

        # store vectors
        self.g = g
        self.g0 = g0
        self.g1 = g1
        self.g2 = g2

        # center the bragg directions and store
        braggdirections.data["qx"] -= self.origin[0]
        braggdirections.data["qy"] -= self.origin[1]
        self.braggdirections = braggdirections

        # return
        if returncalc and returnfig:
            return (self.g0, self.g1, self.g2, self.braggdirections), (fig, ax)
        elif returncalc:
            return (self.g0, self.g1, self.g2, self.braggdirections)
        elif returnfig:
            return (fig, ax)
        else:
            return

    def set_hkl(
        self,
        g1_hkl: Union[List[int], Tuple[int, int, int], np.ndarray[np.int64]],
        g2_hkl: Union[List[int], Tuple[int, int, int], np.ndarray[np.int64]],
    ):
        """
        calculate the [h,k,l] reflections from the `g1_ind`,`g2_ind` from known 'g1_hkl` and 'g2_hkl' reflections.
        Creates 'bragg_vectors_indexed_hkl' attribute
        Args:
            g1_hkl (list[int] | tuple[int,int,int] | np.ndarray[int]): known [h,k,l] reflection for g1_vector
            g2_hkl (list[int] | tuple[int,int,int] | np.ndarray[int]): known [h,k,l] reflection for g1_vector
        """

        g1_hkl = np.array(g1_hkl)
        g2_hkl = np.array(g2_hkl)

        # Initialize a PLA
        bvs_hkl = PointListArray(
            shape=self.shape,
            dtype=[
                ("qx", float),
                ("qy", float),
                ("intensity", float),
                ("h", int),
                ("k", int),
                ("l", int),
            ],
        )
        # loop over the probe posistions
        for Rx, Ry in tqdmnd(
            self.shape[0],
            self.shape[1],
            desc="Converting (g1_ind,g2_ind) to (h,k,l)",
            unit="DP",
            unit_scale=True,
        ):
            # get a single indexed
            braggvectors_indexed_dp = self.bragg_vectors_indexed[Rx, Ry]

            # make a Pointlsit
            bvs_hkl_curr = PointList(
                data=np.empty(len(braggvectors_indexed_dp), dtype=bvs_hkl.dtype)
            )
            # populate qx, qy and intensity fields
            bvs_hkl_curr.data["qx"] = braggvectors_indexed_dp["qx"]
            bvs_hkl_curr.data["qy"] = braggvectors_indexed_dp["qy"]
            bvs_hkl_curr.data["intensity"] = braggvectors_indexed_dp["intensity"]

            # calcuate the hkl vectors
            vectors_hkl = (
                g1_hkl[:, np.newaxis] * braggvectors_indexed_dp["g1_ind"]
                + g2_hkl[:, np.newaxis] * braggvectors_indexed_dp["g2_ind"]
            )
            # self.vectors_hkl = vectors_hkl

            # populate h,k,l fields
            # print(vectors_hkl.shape)
            # bvs_hkl_curr.data['h'] = vectors_hkl[0,:]
            # bvs_hkl_curr.data['k'] = vectors_hkl[1,:]
            # bvs_hkl_curr.data['l'] = vectors_hkl[2,:]
            (
                bvs_hkl_curr.data["h"],
                bvs_hkl_curr.data["k"],
                bvs_hkl_curr.data["l"],
            ) = np.vsplit(vectors_hkl, 3)

            # add to the PLA
            bvs_hkl[Rx, Ry] += bvs_hkl_curr

        # add the PLA to the Strainmap object
        self.bragg_vectors_indexed_hkl = bvs_hkl

    def set_max_peak_spacing(
        self,
        max_peak_spacing,
        returnfig=False,
        **vis_params,
    ):
        """
        Set the size of the regions of diffraction space in which detected Bragg
        peaks will be indexed and included in subsequent fitting of basis
        vectors, and visualize those regions.

        Parameters
        ----------
        max_peak_spacing : number
            The maximum allowable distance in pixels between a detected Bragg peak and
            the indexed maxima found in `choose_basis_vectors` for the detected
            peak to be indexed
        returnfig : bool
            Toggles returning the figure
        vis_params : dict
            Any additional arguments are passed to the `show` function when
            visualization the BVM
        """
        # set the max peak spacing
        self.max_peak_spacing = max_peak_spacing

        # make the figure
        fig, ax = show(
            self.bvm.data,
            returnfig=True,
            **vis_params,
        )

        # make the circle patch collection
        patches = []
        qx = self.braggdirections["qx"]
        qy = self.braggdirections["qy"]
        origin = self.origin
        for idx in range(len(qx)):
            c = Circle(
                xy=(qy[idx] + origin[1], qx[idx] + origin[0]),
                radius=self.max_peak_spacing,
                edgecolor="r",
                fill=False,
            )
            patches.append(c)
        pc = PatchCollection(patches, match_original=True)

        # draw the circles
        ax.add_collection(pc)

        # return
        if returnfig:
            return fig, ax
        else:
            plt.show()

    def fit_basis_vectors(
        self, mask=None, max_peak_spacing=None, vis_params={}, returncalc=False
    ):
        """
        Fit the basis lattice vectors to the detected Bragg peaks at each
        scan position.

        First, the lattice vectors at each scan position are indexed using the
        basis vectors g1 and g2 specified previously with `choose_basis_vectors`
        Detected Bragg peaks which are farther from the set of lattice vectors
        found in `choose_basis vectors` than the maximum peak spacing are
        ignored; the maximum peak spacing can be set previously by calling
        `set_max_peak_spacing` or by specifying the `max_peak_spacing` argument
        here. A fit is then performed to refine the values of g1 and g2 at each
        scan position, fitting the basis vectors to all detected and indexed
        peaks, weighting the peaks according to their intensity.

        Parameters
        ----------
        mask : 2d boolean array
            A real space shaped Boolean mask indicating scan positions at which
            to fit the lattice vectors.
        max_peak_spacing : float
            Maximum distance from the ideal lattice points to include a peak
            for indexing
        vis_params : dict
            Visualization parameters for showing the max peak spacing; ignored
            if `max_peak_spacing` is not set
        returncalc : bool
            if True, returns bragg_directions, bragg_vectors_indexed, g1g2_map
        """
        # check the calstate
        assert (
            self.calstate == self.braggvectors.calstate
        ), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        # handle the max peak spacing
        if max_peak_spacing is not None:
            self.set_max_peak_spacing(max_peak_spacing, **vis_params)
        assert hasattr(self, "max_peak_spacing"), "Set the maximum peak spacing!"

        # index the bragg vectors

        # handle the mask
        if mask is None:
            mask = np.ones(self.braggvectors.Rshape, dtype=bool)
        assert (
            mask.shape == self.braggvectors.Rshape
        ), "mask must have same shape as pointlistarray"
        assert mask.dtype == bool, "mask must be boolean"
        self.mask = mask

        # set up new braggpeaks PLA
        indexed_braggpeaks = PointListArray(
            dtype=[
                ("qx", float),
                ("qy", float),
                ("intensity", float),
                ("g1_ind", int),
                ("g2_ind", int),
            ],
            shape=self.braggvectors.Rshape,
        )

        # loop over all the scan positions
        # and perform indexing, excluding peaks outside of max_peak_spacing
        calstate = self.braggvectors.calstate
        for Rx, Ry in tqdmnd(
            mask.shape[0],
            mask.shape[1],
            desc="Indexing Bragg scattering",
            unit="DP",
            unit_scale=True,
        ):
            if mask[Rx, Ry]:
                pl = self.braggvectors.get_vectors(
                    Rx,
                    Ry,
                    center=True,
                    ellipse=calstate["ellipse"],
                    rotate=calstate["rotate"],
                    pixel=False,
                )
                for i in range(pl.data.shape[0]):
                    r = np.hypot(
                        pl.data["qx"][i] - self.braggdirections.data["qx"],
                        pl.data["qy"][i] - self.braggdirections.data["qy"],
                    )
                    ind = np.argmin(r)
                    if r[ind] <= self.max_peak_spacing:
                        indexed_braggpeaks[Rx, Ry].add_data_by_field(
                            (
                                pl.data["qx"][i],
                                pl.data["qy"][i],
                                pl.data["intensity"][i],
                                self.braggdirections.data["g1_ind"][ind],
                                self.braggdirections.data["g2_ind"][ind],
                            )
                        )
        self.bragg_vectors_indexed = indexed_braggpeaks

        # fit bragg vectors
        g1g2_map = fit_lattice_vectors_all_DPs(self.bragg_vectors_indexed)
        self.g1g2_map = g1g2_map

        # update the mask
        g1g2_mask = self.g1g2_map["mask"].data.astype("bool")
        self.mask = np.logical_and(self.mask, g1g2_mask)

        # return
        if returncalc:
            return self.bragg_vectors_indexed, self.g1g2_map

    def get_strain(
        self, gvects=None, coordinate_rotation=0, returncalc=False, **kwargs
    ):
        """
        Compute the strain as the deviation of the basis reciprocal lattice
        vectors which have been fit at each scan position with respect to a
        pair of reference lattice vectors, determined by the argument `gvects`.

        Parameters
        ----------
        gvects : None or 2d-array or tuple
            Specifies how to select the reference lattice vectors. If None,
            use the median of the fit lattice vectors over the whole dataset.
            If a 2d array is passed, it should be real space shaped and boolean.
            In this case, uses the median of the fit lattice vectors in all scan
            positions where this array is True. Otherwise, should be a length 2
            tuple of length 2 array/list/tuples, which are used directly as
            g1 and g2.
        coordinate_rotation : number
            Rotate the reference coordinate system counterclockwise by this
            amount, in degrees
        returncal : bool
            It True, returns rotated map
        **kwargs: keywords passed to py4DSTEM show function
        """
        # confirm that the calstate hasn't changed
        assert (
            self.calstate == self.braggvectors.calstate
        ), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        # get the reference g-vectors
        if gvects is None:
            g1_ref, g2_ref = get_reference_g1g2(self.g1g2_map, self.mask)
        elif isinstance(gvects, np.ndarray):
            assert gvects.shape == self.rshape
            assert gvects.dtype == bool
            g1_ref, g2_ref = get_reference_g1g2(
                self.g1g2_map, np.logical_and(gvects, self.mask)
            )
        else:
            g1_ref = np.array(gvects[0])
            g2_ref = np.array(gvects[1])

        # find the strain
        strainmap_g1g2 = get_strain_from_reference_g1g2(self.g1g2_map, g1_ref, g2_ref)
        self.strainmap_g1g2 = strainmap_g1g2

        # get the reference coordinate system
        theta = np.radians(coordinate_rotation)
        xaxis_x = np.cos(theta)
        xaxis_y = np.sin(theta)
        self.coordinate_rotation_degrees = coordinate_rotation
        self.coordinate_rotation_radians = theta

        # get the strain in the reference coordinates
        strainmap_rotated = get_rotated_strain_map(
            self.strainmap_g1g2,
            xaxis_x=xaxis_x,
            xaxis_y=xaxis_y,
            flip_theta=False,
        )

        # store the data
        self.data[0] = strainmap_rotated["e_xx"].data
        self.data[1] = strainmap_rotated["e_yy"].data
        self.data[2] = strainmap_rotated["e_xy"].data
        self.data[3] = strainmap_rotated["theta"].data
        self.data[4] = strainmap_rotated["mask"].data

        # plot the results
        fig, ax = self.show_strain(
            **kwargs,
            returnfig=True,
        )

        # return
        if returncalc:
            return self.strainmap

    def get_reference_g1g2(self, ROI):
        """
        Get reference g1,g2 vectors by taking the median fit vectors
        in the specified ROI.

        Parameters
        ----------
        ROI : real space shaped 2d boolean ndarray
            Use scan positions where ROI is True

        Returns
        -------
        g1_ref,g2_ref : 2 tuple of length 2 ndarrays
        """
        g1_ref, g2_ref = get_reference_g1g2(self.g1g2_map, ROI)
        return g1_ref, g2_ref

    def show_strain(
        self,
        vrange=[-3, 3],
        vrange_theta=[-3, 3],
        vrange_exx=None,
        vrange_exy=None,
        vrange_eyy=None,
        show_cbars=None,
        bordercolor="k",
        borderwidth=1,
        titlesize=18,
        ticklabelsize=10,
        ticknumber=5,
        unitlabelsize=16,
        cmap="RdBu_r",
        cmap_theta="PRGn",
        mask_color="k",
        color_axes="k",
        show_legend=True,
        show_gvects=True,
        color_gvects="r",
        legend_camera_length=1.6,
        scale_gvects=0.6,
        layout="square",
        figsize=None,
        returnfig=False,
        **kwargs,
    ):
        """
        Display a strain map, showing the 4 strain components
        (e_xx,e_yy,e_xy,theta), and masking each image with
        strainmap.get_slice('mask')

        Parameters
        ----------
        vrange : length 2 list or tuple
            The colorbar intensity range for exx,eyy, and exy.
        vrange_theta : length 2 list or tuple
            The colorbar intensity range for theta.
        vrange_exx : length 2 list or tuple
            The colorbar intensity range for exx; overrides `vrange`
            for exx
        vrange_exy : length 2 list or tuple
            The colorbar intensity range for exy; overrides `vrange`
            for exy
        vrange_eyy : length 2 list or tuple
            The colorbar intensity range for eyy; overrides `vrange`
            for eyy
        bkgrd : bool
            Overlay a mask over background pixels
        show_cbars : None or a tuple of strings
            Show colorbars for the specified axes. Valid strings are
            'exx', 'eyy', 'exy', and 'theta'.
        bordercolor : color
            Color for the image borders
        borderwidth : number
            Width of the image borders
        titlesize : number
            Size of the image titles
        ticklabelsize : number
            Size of the colorbar ticks
        ticknumber : number
            Number of ticks on colorbars
        unitlabelsize : number
            Size of the units label on the colorbars
        cmap : colormap
            Colormap for exx, exy, and eyy
        cmap_theta : colormap
            Colormap for theta
        mask_color : color
            Color for the background mask
        color_axes : color
            Color for the legend coordinate axes
        show_gvects : bool
            Toggles displaying the g-vectors in the legend
        color_gvects : color
            Color for the legend g-vectors
        legend_camera_length : number
            The distance the legend is viewed from; a smaller number yields
            a larger legend
        scale_gvects : number
            Scaling for the legend g-vectors relative to the coordinate axes
        layout : int
            Determines the layout of the grid which the strain components
            will be plotted in.  Options are "square", "horizontal", "vertical."
        figsize : length 2 tuple of numbers
            Size of the figure
        returnfig : bool
            Toggles returning the figure
        **kwargs: keywords passed to py4DSTEM show function
        """

        from py4DSTEM.visualize import show_strain

        fig, ax = show_strain(
            self,
            vrange=vrange,
            vrange_theta=vrange_theta,
            vrange_exx=vrange_exx,
            vrange_exy=vrange_exy,
            vrange_eyy=vrange_eyy,
            show_cbars=show_cbars,
            bordercolor=bordercolor,
            borderwidth=borderwidth,
            titlesize=titlesize,
            ticklabelsize=ticklabelsize,
            ticknumber=ticknumber,
            unitlabelsize=unitlabelsize,
            cmap=cmap,
            cmap_theta=cmap_theta,
            mask_color=mask_color,
            color_axes=color_axes,
            show_legend=show_legend,
            rotation_deg=np.rad2deg(self.coordinate_rotation_radians),
            show_gvects=show_gvects,
            g1=self.g1,
            g2=self.g2,
            color_gvects=color_gvects,
            legend_camera_length=legend_camera_length,
            scale_gvects=scale_gvects,
            layout=layout,
            figsize=figsize,
            returnfig=True,
            **kwargs,
        )

        # show/return
        if not returnfig:
            plt.show()
            return
        else:
            return fig, ax

    def show_reference_directions(
        self,
        im_uncal=None,
        im_cal=None,
        color_axes="linen",
        color_gvects="r",
        origin_uncal=None,
        origin_cal=None,
        camera_length=1.8,
        visp_uncal={"scaling": "log"},
        visp_cal={"scaling": "log"},
        layout="horizontal",
        titlesize=16,
        size_labels=14,
        figsize=None,
        returnfig=False,
    ):
        """
        Show the reference coordinate system used to compute the strain
        overlaid over calibrated and uncalibrated diffraction space images.

        The diffraction images used can be specificied with the `im_uncal`
        and `im_cal` arguments, and default to the uncalibrated and calibrated
        Bragg vector maps.  The `rotate_cal` argument causes the `im_cal` array
        to be rotated by -QR rotation from the calibration metadata, so that an
        uncalibrated image (like a raw diffraction image or mean or max
        diffraction pattern) can be passed to the `im_cal` argument.

        Parameters
        ----------
        im_uncal : 2d array or None
            Uncalibrated diffraction space image to dispay; defaults to
            the maximal diffraction image.
        im_cal : 2d array or None
            Calibrated diffraction space image to display; defaults to
            the calibrated Bragg vector map.
        color_axes : color
            The color of the overlaid coordinate axes
        color_gvects : color
            The color of the g-vectors
        origin_uncal : 2-tuple or None
            Where to place the origin of the coordinate system overlaid on
            the uncalibrated diffraction image. Defaults to the mean origin
            from the calibration metadata.
        origin_cal : 2-tuple or None
            Where to place the origin of the coordinate system overlaid on
            the calibrated diffraction image. Defaults to the mean origin
            from the calibration metadata.
        camera_length : number
            Determines the length of the overlaid coordinate axes; a smaller
            number yields larger axes.
        visp_uncal : dict
            Visualization parameters for the uncalibrated diffraction image.
        visp_cal : dict
            Visualization parameters for the calibrated diffraction image.
        layout : str; either "horizontal" or "vertical"
            Determines the layout of the visualization.
        titlesize : number
            The size of the plot titles
        size_labels : number
            The size of the axis labels
        figsize : length 2 tuple of numbers or None
            Size of the figure
        returnfig : bool
            Toggles returning the figure
        """
        # Set up the figure
        assert layout in ("horizontal", "vertical")

        # Set the figsize
        if figsize is None:
            ratio = np.sqrt(self.rshape[1] / self.rshape[0])
            if layout == "horizontal":
                figsize = (10 * ratio, 8 / ratio)
            else:
                figsize = (8 * ratio, 12 / ratio)

        # Create the figure
        if layout == "horizontal":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # prepare images
        if im_uncal is None:
            im_uncal = self.braggvectors.histogram(mode="raw")
        if im_cal is None:
            im_cal = self.braggvectors.histogram(mode="cal")

        # display images
        show(im_cal, figax=(fig, ax1), **visp_cal)
        show(im_uncal, figax=(fig, ax2), **visp_uncal)
        ax1.set_title("Calibrated", size=titlesize)
        ax2.set_title("Uncalibrated", size=titlesize)

        # Get the coordinate axes

        # get the directions

        # calibrated
        rotation = self.coordinate_rotation_radians
        xaxis_cal = np.array([np.cos(rotation), np.sin(rotation)])
        yaxis_cal = np.array(
            [np.cos(rotation + np.pi / 2), np.sin(rotation + np.pi / 2)]
        )

        # uncalibrated
        QRrot = self.calibration.get_QR_rotation()
        rotation = np.sum([self.coordinate_rotation_radians, -QRrot])
        xaxis_uncal = np.array([np.cos(rotation), np.sin(rotation)])
        yaxis_uncal = np.array(
            [np.cos(rotation + np.pi / 2), np.sin(rotation + np.pi / 2)]
        )
        # inversion
        if self.calibration.get_QR_flip():
            xaxis_uncal = np.array([xaxis_uncal[1], xaxis_uncal[0]])
            yaxis_uncal = np.array([yaxis_uncal[1], yaxis_uncal[0]])

        # set the lengths
        Lmean = np.mean([im_cal.shape[0], im_cal.shape[1]]) / 2
        xaxis_cal *= Lmean / camera_length
        yaxis_cal *= Lmean / camera_length
        xaxis_uncal *= Lmean / camera_length
        yaxis_uncal *= Lmean / camera_length

        # Get the g-vectors

        # calibrated
        g1_cal = np.array(self.g1)
        g2_cal = np.array(self.g2)

        # uncalibrated
        R = np.array([[np.cos(QRrot), -np.sin(QRrot)], [np.sin(QRrot), np.cos(QRrot)]])
        g1_uncal = np.matmul(g1_cal, R)
        g2_uncal = np.matmul(g2_cal, R)
        # inversion
        if self.calibration.get_QR_flip():
            g1_uncal = np.array([g1_uncal[1], g1_uncal[0]])
            g2_uncal = np.array([g2_uncal[1], g2_uncal[0]])

        # Set origin positions
        if origin_uncal is None:
            origin_uncal = self.calibration.get_origin_mean()
        if origin_cal is None:
            origin_cal = self.calibration.get_origin_mean()

        # Draw calibrated coordinate axes
        coordax_width = Lmean * 2 / 100
        ax1.arrow(
            x=origin_cal[1],
            y=origin_cal[0],
            dx=xaxis_cal[1],
            dy=xaxis_cal[0],
            color=color_axes,
            length_includes_head=True,
            width=coordax_width,
            head_width=coordax_width * 5,
        )
        ax1.arrow(
            x=origin_cal[1],
            y=origin_cal[0],
            dx=yaxis_cal[1],
            dy=yaxis_cal[0],
            color=color_axes,
            length_includes_head=True,
            width=coordax_width,
            head_width=coordax_width * 5,
        )
        ax1.text(
            x=origin_cal[1] + xaxis_cal[1] * 1.16,
            y=origin_cal[0] + xaxis_cal[0] * 1.16,
            s="x",
            fontsize=size_labels,
            color=color_axes,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax1.text(
            x=origin_cal[1] + yaxis_cal[1] * 1.16,
            y=origin_cal[0] + yaxis_cal[0] * 1.16,
            s="y",
            fontsize=size_labels,
            color=color_axes,
            horizontalalignment="center",
            verticalalignment="center",
        )

        # Draw uncalibrated coordinate axes
        ax2.arrow(
            x=origin_uncal[1],
            y=origin_uncal[0],
            dx=xaxis_uncal[1],
            dy=xaxis_uncal[0],
            color=color_axes,
            length_includes_head=True,
            width=coordax_width,
            head_width=coordax_width * 5,
        )
        ax2.arrow(
            x=origin_uncal[1],
            y=origin_uncal[0],
            dx=yaxis_uncal[1],
            dy=yaxis_uncal[0],
            color=color_axes,
            length_includes_head=True,
            width=coordax_width,
            head_width=coordax_width * 5,
        )
        ax2.text(
            x=origin_uncal[1] + xaxis_uncal[1] * 1.16,
            y=origin_uncal[0] + xaxis_uncal[0] * 1.16,
            s="x",
            fontsize=size_labels,
            color=color_axes,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax2.text(
            x=origin_uncal[1] + yaxis_uncal[1] * 1.16,
            y=origin_uncal[0] + yaxis_uncal[0] * 1.16,
            s="y",
            fontsize=size_labels,
            color=color_axes,
            horizontalalignment="center",
            verticalalignment="center",
        )

        # Draw the calibrated g-vectors

        # draw the g vectors
        ax1.arrow(
            x=origin_cal[1],
            y=origin_cal[0],
            dx=g1_cal[1],
            dy=g1_cal[0],
            color=color_gvects,
            length_includes_head=True,
            width=coordax_width * 0.5,
            head_width=coordax_width * 2.5,
        )
        ax1.arrow(
            x=origin_cal[1],
            y=origin_cal[0],
            dx=g2_cal[1],
            dy=g2_cal[0],
            color=color_gvects,
            length_includes_head=True,
            width=coordax_width * 0.5,
            head_width=coordax_width * 2.5,
        )
        ax1.text(
            x=origin_cal[1] + g1_cal[1] * 1.16,
            y=origin_cal[0] + g1_cal[0] * 1.16,
            s=r"$g_1$",
            fontsize=size_labels * 0.88,
            color=color_gvects,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax1.text(
            x=origin_cal[1] + g2_cal[1] * 1.16,
            y=origin_cal[0] + g2_cal[0] * 1.16,
            s=r"$g_2$",
            fontsize=size_labels * 0.88,
            color=color_gvects,
            horizontalalignment="center",
            verticalalignment="center",
        )

        # Draw the uncalibrated g-vectors

        # draw the g vectors
        ax2.arrow(
            x=origin_uncal[1],
            y=origin_uncal[0],
            dx=g1_uncal[1],
            dy=g1_uncal[0],
            color=color_gvects,
            length_includes_head=True,
            width=coordax_width * 0.5,
            head_width=coordax_width * 2.5,
        )
        ax2.arrow(
            x=origin_uncal[1],
            y=origin_uncal[0],
            dx=g2_uncal[1],
            dy=g2_uncal[0],
            color=color_gvects,
            length_includes_head=True,
            width=coordax_width * 0.5,
            head_width=coordax_width * 2.5,
        )
        ax2.text(
            x=origin_uncal[1] + g1_uncal[1] * 1.16,
            y=origin_uncal[0] + g1_uncal[0] * 1.16,
            s=r"$g_1$",
            fontsize=size_labels * 0.88,
            color=color_gvects,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax2.text(
            x=origin_uncal[1] + g2_uncal[1] * 1.16,
            y=origin_uncal[0] + g2_uncal[0] * 1.16,
            s=r"$g_2$",
            fontsize=size_labels * 0.88,
            color=color_gvects,
            horizontalalignment="center",
            verticalalignment="center",
        )

        # show/return
        if not returnfig:
            plt.show()
            return
        else:
            return fig, (ax1, ax2)

    def show_lattice_vectors(
        ar,
        x0,
        y0,
        g1,
        g2,
        color="r",
        width=1,
        labelsize=20,
        labelcolor="w",
        returnfig=False,
        **kwargs,
    ):
        """
        Adds the vectors g1,g2 to an image, with tail positions at (x0,y0).
        g1 and g2 are 2-tuples (gx,gy).
        """
        fig, ax = show(ar, returnfig=True, **kwargs)

        # Add vectors
        dg1 = {
            "x0": x0,
            "y0": y0,
            "vx": g1[0],
            "vy": g1[1],
            "width": width,
            "color": color,
            "label": r"$g_1$",
            "labelsize": labelsize,
            "labelcolor": labelcolor,
        }
        dg2 = {
            "x0": x0,
            "y0": y0,
            "vx": g2[0],
            "vy": g2[1],
            "width": width,
            "color": color,
            "label": r"$g_2$",
            "labelsize": labelsize,
            "labelcolor": labelcolor,
        }
        add_vector(ax, dg1)
        add_vector(ax, dg2)

        if returnfig:
            return fig, ax
        else:
            plt.show()
            return

    def show_bragg_indexing(
        self,
        ar,
        bragg_directions,
        voffset=5,
        hoffset=0,
        color="w",
        size=20,
        points=True,
        pointcolor="r",
        pointsize=50,
        figax=None,
        returnfig=False,
        **kwargs,
    ):
        """
        Shows an array with an overlay describing the Bragg directions

        Parameters
        ----------
        ar : np.ndarray
            The display image
        bragg_directions : PointList
            The Bragg scattering directions.  Must have coordinates
            ('qx','qy','h', and 'k') or ('qx','qy','g1_ind', and 'g2_ind'. Optionally may also have 'l'.
        """
        assert isinstance(bragg_directions, PointList)
        # checking if it has h, k or g1_ind, g2_ind
        assert all(
            key in bragg_directions.data.dtype.names for key in ("qx", "qy", "h", "k")
        ) or all(
            key in bragg_directions.data.dtype.names
            for key in ("qx", "qy", "g1_ind", "g2_ind")
        ), 'pointlist must contain ("qx", "qy", "h", "k") or ("qx", "qy", "g1_ind", "g2_ind") fields'

        if figax is None:
            fig, ax = show(ar, returnfig=True, **kwargs)
        else:
            fig = figax[0]
            ax = figax[1]
            show(ar, figax=figax, **kwargs)

        d = {
            "bragg_directions": bragg_directions,
            "voffset": voffset,
            "hoffset": hoffset,
            "color": color,
            "size": size,
            "points": points,
            "pointsize": pointsize,
            "pointcolor": pointcolor,
        }
        # this can take ("qx", "qy", "h", "k") or ("qx", "qy", "g1_ind", "g2_ind") fields
        add_bragg_index_labels(ax, d)

        if returnfig:
            return fig, ax
        else:
            return

    def copy(self, name=None):
        name = name if name is not None else self.name + "_copy"
        strainmap_copy = StrainMap(self.braggvectors)
        for attr in (
            "g",
            "g0",
            "g1",
            "g2",
            "calstate",
            "bragg_directions",
            "bragg_vectors_indexed",
            "g1g2_map",
            "strainmap_g1g2",
            "strainmap_rotated",
            "mask",
        ):
            if hasattr(self, attr):
                setattr(strainmap_copy, attr, getattr(self, attr))

        for k in self.metadata.keys():
            strainmap_copy.metadata = self.metadata[k].copy()
        return strainmap_copy

    # TODO IO methods

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
