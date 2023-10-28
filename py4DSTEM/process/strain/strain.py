# Defines the Strain class

import warnings
from typing import Optional

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from py4DSTEM import PointList, PointListArray, tqdmnd
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.data import Data, RealSlice
from py4DSTEM.preprocess.utils import get_maxima_2D
from py4DSTEM.process.strain.latticevectors import (
    add_indices_to_braggvectors,
    fit_lattice_vectors_all_DPs,
    get_reference_g1g2,
    get_rotated_strain_map,
    get_strain_from_reference_g1g2,
    index_bragg_directions,
)
from py4DSTEM.visualize import add_bragg_index_labels, add_pointlabels, add_vector, show
from py4DSTEM.visualize import ax_addaxes, ax_addaxes_QtoR

warnings.simplefilter(action="always", category=UserWarning)


class StrainMap(RealSlice, Data):
    """
    Storage and processing methods for 4D-STEM datasets.

    """

    def __init__(self, braggvectors: BraggVectors, name: Optional[str] = "strainmap"):
        """
        Accepts:
            braggvectors (BraggVectors): BraggVectors for Strain Map
            name (str): the name of the strainmap
        Returns:
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
            slicelabels=["exx", "eyy", "exy", "theta", "mask", "error"],
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
        if self.calstate["rotate"] == False:
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

    @property
    def mask(self):
        try:
            return self.g1g2_map["mask"].data.astype("bool")
        except:
            return np.ones(self.rshape, dtype=bool)

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

    def choose_lattice_vectors(
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
        Choose which lattice vectors to use for strain mapping.

        Overlays the bvm with the points detected via local 2D
        maxima detection, plus an index for each point. User selects
        3 points using the overlaid indices, which are identified as
        the origin and the termini of the lattice vectors g1 and g2.

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
        (optional) : None or (g0,g1,g2) or (fig,(ax1,ax2)) or both of the latter
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

    def fit_lattice_vectors(
        self,
        max_peak_spacing=2,
        mask=None,
        returncalc=False,
    ):
        """
        From an origin (x0,y0), a set of reciprocal lattice vectors gx,gy, and an pair of
        lattice vectors g1=(g1x,g1y), g2=(g2x,g2y), find the indices (h,k) of all the
        reciprocal lattice directions.

        Args:
            max_peak_spacing: float
                Maximum distance from the ideal lattice points
                to include a peak for indexing
            mask: bool
                Boolean mask, same shape as the pointlistarray, indicating which
                locations should be indexed. This can be used to index different regions of
                the scan with different lattices
            returncalc : bool
                if True, returns bragg_directions, bragg_vectors_indexed, g1g2_map
        """
        # check the calstate
        assert (
            self.calstate == self.braggvectors.calstate
        ), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        ### add indices to the bragg vectors

        # validate mask
        if mask is None:
            mask = np.ones(self.braggvectors.Rshape, dtype=bool)
        assert (
            mask.shape == self.braggvectors.Rshape
        ), "mask must have same shape as pointlistarray"
        assert mask.dtype == bool, "mask must be boolean"

        # set up new braggpeaks PLA
        indexed_braggpeaks = PointListArray(
            dtype=[
                ("qx", float),
                ("qy", float),
                ("intensity", float),
                ("h", int),
                ("k", int),
            ],
            shape=self.braggvectors.Rshape,
        )
        calstate = self.braggvectors.calstate

        # loop over all the scan positions
        for Rx, Ry in tqdmnd(mask.shape[0], mask.shape[1]):
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
                    if r[ind] <= max_peak_spacing:
                        indexed_braggpeaks[Rx, Ry].add_data_by_field(
                            (
                                pl.data["qx"][i],
                                pl.data["qy"][i],
                                pl.data["intensity"][i],
                                self.braggdirections.data["h"][ind],
                                self.braggdirections.data["k"][ind],
                            )
                        )
        self.bragg_vectors_indexed = indexed_braggpeaks

        ### fit bragg vectors
        g1g2_map = fit_lattice_vectors_all_DPs(self.bragg_vectors_indexed)
        self.g1g2_map = g1g2_map

        # return
        if returncalc:
            return self.bragg_vectors_indexed, self.g1g2_map

    def get_strain(
        self, mask=None, coordinate_rotation=0, returncalc=False, **kwargs
    ):
        """
        Parameters
        ----------
        mask : nd.array (bool)
            Use lattice vectors from g1g2_map scan positions
            wherever mask==True. If mask is None gets median strain
            map from entire field of view. If mask is not None, gets
            reference g1 and g2 from region and then calculates strain.
        coordinate_rotation : number
            Rotate the reference coordinate system counterclockwise by this
            amount, in degrees
        returncal : bool
            It True, returns rotated map
        """
        # confirm that the calstate hasn't changed
        assert (
            self.calstate == self.braggvectors.calstate
        ), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        # get the mask
        if mask is None:
            mask = self.mask
            # mask = np.ones(self.g1g2_map.shape, dtype="bool")
            # strainmap_g1g2 = get_strain_from_reference_region(
            #     self.g1g2_map,
            #     mask=mask,
            # )

        #     g1_ref, g2_ref = get_reference_g1g2(self.g1g2_map, mask)
        #     strain_map = get_strain_from_reference_g1g2(self.g1g2_map, g1_ref, g2_ref)
        # else:

        # get the reference g1/g2 vectors
        g1_ref, g2_ref = get_reference_g1g2(self.g1g2_map, mask)

        # find the strain
        strainmap_g1g2 = get_strain_from_reference_g1g2(self.g1g2_map, g1_ref, g2_ref)
        self.strainmap_g1g2 = strainmap_g1g2

        # get the reference coordinate system
        theta = np.radians(coordinate_rotation)
        xaxis_x = np.cos(theta)
        xaxis_y = np.sin(theta)

        # get the strain in the reference coordinates
        strainmap_rotated = get_rotated_strain_map(
            self.strainmap_g1g2,
            xaxis_x = xaxis_x,
            xaxis_y = xaxis_y,
            flip_theta = False,
        )

        # store the data
        self.data[0] = strainmap_rotated["e_xx"].data
        self.data[1] = strainmap_rotated["e_yy"].data
        self.data[2] = strainmap_rotated["e_xy"].data
        self.data[3] = strainmap_rotated["theta"].data
        self.data[4] = strainmap_rotated["mask"].data
        self.coordinate_rotation = coordinate_rotation

        # plot the results
        fig, ax = self.show_strain(
            **kwargs,
            returnfig=True,
        )

        # modify masking
        if not np.all(mask == True):
            ax[0][0].imshow(mask, alpha=0.2, cmap="binary")
            ax[0][1].imshow(mask, alpha=0.2, cmap="binary")
            ax[1][0].imshow(mask, alpha=0.2, cmap="binary")
            ax[1][1].imshow(mask, alpha=0.2, cmap="binary")

        # return
        if returncalc:
            return self.strainmap


    def show_strain(
        self,
        vrange = [-3,3],
        vrange_theta = [-3,3],
        vrange_exx=None,
        vrange_exy=None,
        vrange_eyy=None,
        bkgrd=True,
        show_cbars=("eyy", "theta"),
        bordercolor="k",
        borderwidth=1,
        titlesize=24,
        ticklabelsize=16,
        ticknumber=5,
        unitlabelsize=24,
        cmap="RdBu_r",
        cmap_theta="PRGn",
        mask_color="k",
        color_axes="k",
        show_gvects=False,
        color_gvects="r",
        legend_camera_length = 1.6,
        layout=0,
        figsize=None,
        returnfig=False,
    ):
        """
        Display a strain map, showing the 4 strain components
        (e_xx,e_yy,e_xy,theta), and masking each image with
        strainmap.get_slice('mask')

        Parameters
        ----------
        vrange : length 2 list or tuple
        vrange_theta : length 2 list or tuple
        vrange_exx : length 2 list or tuple
        vrange_exy : length 2 list or tuple
        vrange_eyy :length 2 list or tuple
        bkgrd : bool
        show_cbars :tuple of strings
            Show colorbars for the specified axes. Must be a tuple
            containing any, all, or none of ('exx','eyy','exy','theta')
        bordercolor : color
        borderwidth : number
        titlesize : number
        ticklabelsize : number
        ticknumber : number
            number of ticks on colorbars
        unitlabelsize : number
        cmap : colormap
        cmap_theta : colormap
        mask_color : color
        color_axes : color
        show_gvects : bool
            Toggles displaying the g-vectors in the legend
        color_gvects : color
        legend_camera_length : number
            The distance the legend is viewed from; a smaller number yields
            a larger legend
        layout : int
            determines the layout of the grid which the strain components
            will be plotted in.  Must be in (0,1,2).  0=(2x2), 1=(1x4), 2=(4x1).
        figsize : length 2 tuple of numbers
        returnfig : bool
        """
        # Lookup table for different layouts
        assert layout in (0, 1, 2)
        layout_lookup = {
            0: ["left", "right", "left", "right"],
            1: ["bottom", "bottom", "bottom", "bottom"],
            2: ["right", "right", "right", "right"],
        }
        layout_p = layout_lookup[layout]

        # Contrast limits
        if vrange_exx is None:
            vrange_exx = vrange
        if vrange_exy is None:
            vrange_exy = vrange
        if vrange_eyy is None:
            vrange_eyy = vrange
        for vrange in (vrange_exx, vrange_eyy, vrange_exy, vrange_theta):
            assert len(vrange) == 2, "vranges must have length 2"
        vmin_exx, vmax_exx = vrange_exx[0] / 100.0, vrange_exx[1] / 100.0
        vmin_eyy, vmax_eyy = vrange_eyy[0] / 100.0, vrange_eyy[1] / 100.0
        vmin_exy, vmax_exy = vrange_exy[0] / 100.0, vrange_exy[1] / 100.0
        # theta is plotted in units of degrees
        vmin_theta, vmax_theta = vrange_theta[0] / (180.0 / np.pi), vrange_theta[1] / (
            180.0 / np.pi
        )

        # Get images
        e_xx = np.ma.array(
            self.get_slice("exx").data, mask=self.get_slice("mask").data == False
        )
        e_yy = np.ma.array(
            self.get_slice("eyy").data, mask=self.get_slice("mask").data == False
        )
        e_xy = np.ma.array(
            self.get_slice("exy").data, mask=self.get_slice("mask").data == False
        )
        theta = np.ma.array(
            self.get_slice("theta").data,
            mask=self.get_slice("mask").data == False,
        )

        ## Plot

        # if figsize hasn't been set, set it based on the
        # chosen layout and the image shape
        if figsize is None:
            ratio = np.sqrt(self.rshape[1]/self.rshape[0])
            if layout == 0:
                figsize = (13*ratio,8/ratio)
            elif layout == 1:
                figsize = (10*ratio,4/ratio)
            else:
                figsize = (4*ratio,10/ratio)


        # set up layout
        if layout == 0:
            fig, ((ax11, ax12, ax_legend1), (ax21, ax22, ax_legend2)) =\
                plt.subplots(2, 3, figsize=figsize)
        elif layout == 1:
            figsize = (figsize[0] * np.sqrt(2), figsize[1] / np.sqrt(2))
            fig, (ax11, ax12, ax21, ax22, ax_legend) =\
                plt.subplots(1, 5, figsize=figsize)
        else:
            figsize = (figsize[0] / np.sqrt(2), figsize[1] * np.sqrt(2))
            fig, (ax11, ax12, ax21, ax22, ax_legend) =\
                plt.subplots(5, 1, figsize=figsize)

        # display images, returning cbar axis references
        cax11 = show(
            e_xx,
            figax=(fig, ax11),
            vmin=vmin_exx,
            vmax=vmax_exx,
            intensity_range="absolute",
            cmap=cmap,
            mask=self.mask,
            mask_color=mask_color,
            returncax=True,
        )
        cax12 = show(
            e_yy,
            figax=(fig, ax12),
            vmin=vmin_eyy,
            vmax=vmax_eyy,
            intensity_range="absolute",
            cmap=cmap,
            mask=self.mask,
            mask_color=mask_color,
            returncax=True,
        )
        cax21 = show(
            e_xy,
            figax=(fig, ax21),
            vmin=vmin_exy,
            vmax=vmax_exy,
            intensity_range="absolute",
            cmap=cmap,
            mask=self.mask,
            mask_color=mask_color,
            returncax=True,
        )
        cax22 = show(
            theta,
            figax=(fig, ax22),
            vmin=vmin_theta,
            vmax=vmax_theta,
            intensity_range="absolute",
            cmap=cmap_theta,
            mask=self.mask,
            mask_color=mask_color,
            returncax=True,
        )
        ax11.set_title(r"$\epsilon_{xx}$", size=titlesize)
        ax12.set_title(r"$\epsilon_{yy}$", size=titlesize)
        ax21.set_title(r"$\epsilon_{xy}$", size=titlesize)
        ax22.set_title(r"$\theta$", size=titlesize)

        # Add black background
        if bkgrd:
            mask = np.ma.masked_where(
                self.get_slice("mask").data.astype(bool),
                np.zeros_like(self.get_slice("mask").data),
            )
            ax11.matshow(mask, cmap="gray")
            ax12.matshow(mask, cmap="gray")
            ax21.matshow(mask, cmap="gray")
            ax22.matshow(mask, cmap="gray")

        # add colorbars
        show_cbars = np.array(
            [
                "exx" in show_cbars,
                "eyy" in show_cbars,
                "exy" in show_cbars,
                "theta" in show_cbars,
            ]
        )
        if np.any(show_cbars):
            divider11 = make_axes_locatable(ax11)
            divider12 = make_axes_locatable(ax12)
            divider21 = make_axes_locatable(ax21)
            divider22 = make_axes_locatable(ax22)
            cbax11 = divider11.append_axes(layout_p[0], size="4%", pad=0.15)
            cbax12 = divider12.append_axes(layout_p[1], size="4%", pad=0.15)
            cbax21 = divider21.append_axes(layout_p[2], size="4%", pad=0.15)
            cbax22 = divider22.append_axes(layout_p[3], size="4%", pad=0.15)
            for ind, show_cbar, cax, cbax, vmin, vmax, tickside, tickunits in zip(
                range(4),
                show_cbars,
                (cax11, cax12, cax21, cax22),
                (cbax11, cbax12, cbax21, cbax22),
                (vmin_exx, vmin_eyy, vmin_exy, vmin_theta),
                (vmax_exx, vmax_eyy, vmax_exy, vmax_theta),
                (layout_p[0], layout_p[1], layout_p[2], layout_p[3]),
                ("% ", " %", "% ", r" $^\circ$"),
            ):
                if show_cbar:
                    ticks = np.linspace(vmin, vmax, ticknumber, endpoint=True)
                    if ind < 3:
                        ticklabels = np.round(
                            np.linspace(
                                100 * vmin, 100 * vmax, ticknumber, endpoint=True
                            ),
                            decimals=2,
                        ).astype(str)
                    else:
                        ticklabels = np.round(
                            np.linspace(
                                (180 / np.pi) * vmin,
                                (180 / np.pi) * vmax,
                                ticknumber,
                                endpoint=True,
                            ),
                            decimals=2,
                        ).astype(str)

                    if tickside in ("left", "right"):
                        cb = plt.colorbar(
                            cax, cax=cbax, ticks=ticks, orientation="vertical"
                        )
                        cb.ax.set_yticklabels(ticklabels, size=ticklabelsize)
                        cbax.yaxis.set_ticks_position(tickside)
                        cbax.set_ylabel(tickunits, size=unitlabelsize, rotation=0)
                        cbax.yaxis.set_label_position(tickside)
                    else:
                        cb = plt.colorbar(
                            cax, cax=cbax, ticks=ticks, orientation="horizontal"
                        )
                        cb.ax.set_xticklabels(ticklabels, size=ticklabelsize)
                        cbax.xaxis.set_ticks_position(tickside)
                        cbax.set_xlabel(tickunits, size=unitlabelsize, rotation=0)
                        cbax.xaxis.set_label_position(tickside)
                else:
                    cbax.axis("off")

        # Add borders
        if bordercolor is not None:
            for ax in (ax11, ax12, ax21, ax22):
                for s in ["bottom", "top", "left", "right"]:
                    ax.spines[s].set_color(bordercolor)
                    ax.spines[s].set_linewidth(borderwidth)
                ax.set_xticks([])
                ax.set_yticks([])

        # Legend

        # for layout 0, combine vertical plots on the right end
        if layout == 0:
            # get gridspec object
            gs = ax_legend1.get_gridspec()
            # remove last two axes
            ax_legend1.remove()
            ax_legend2.remove()
            # make new axis
            ax_legend = fig.add_subplot(gs[:,-1])

        # get the coordinate axes' directions
        QRrot = self.calibration.get_QR_rotation()
        rotation = np.sum([
            np.radians(self.coordinate_rotation),
            QRrot
        ])
        xaxis_vectx = np.cos(rotation)
        xaxis_vecty = np.sin(rotation)
        yaxis_vectx = np.cos(rotation+np.pi/2)
        yaxis_vecty = np.sin(rotation+np.pi/2)

        # make the coordinate axes
        ax_legend.arrow(
            x = 0,
            y = 0,
            dx = xaxis_vecty,
            dy = xaxis_vectx,
            color = color_axes,
            length_includes_head = True,
            width = 0.01,
            head_width = 0.1,
        )
        ax_legend.arrow(
            x = 0,
            y = 0,
            dx = yaxis_vecty,
            dy = yaxis_vectx,
            color = color_axes,
            length_includes_head = True,
            width = 0.01,
            head_width = 0.1,
        )
        ax_legend.text(
            x = xaxis_vecty*1.12,
            y = xaxis_vectx*1.12,
            s = 'x',
            fontsize = 14,
            color = color_axes,
            horizontalalignment = 'center',
            verticalalignment = 'center',
        )
        ax_legend.text(
            x = yaxis_vecty*1.12,
            y = yaxis_vectx*1.12,
            s = 'y',
            fontsize = 14,
            color = color_axes,
            horizontalalignment = 'center',
            verticalalignment = 'center',
        )

        # make the g-vectors
        if show_gvects:

            # get the g-vectors directions
            g1q = np.array(self.g1)
            g2q = np.array(self.g2)
            g1norm = np.linalg.norm(g1q)
            g2norm = np.linalg.norm(g2q)
            g1q /= np.linalg.norm(g1norm)
            g2q /= np.linalg.norm(g2norm)
            # set the lengths
            g_ratio = g2norm/g1norm
            if g_ratio > 1:
                g1q /= g_ratio
            else:
                g2q *= g_ratio
            # rotate
            R = np.array(
                [
                    [ np.cos(QRrot), np.sin(QRrot)],
                    [-np.sin(QRrot), np.cos(QRrot)]
                ]
            )
            g1_x,g1_y = np.matmul(g1q,R)
            g2_x,g2_y = np.matmul(g2q,R)

            # draw the g vectors
            ax_legend.arrow(
                x = 0,
                y = 0,
                dx = g1_y*0.8,
                dy = g1_x*0.8,
                color = color_gvects,
                length_includes_head = True,
                width = 0.005,
                head_width = 0.05,
            )
            ax_legend.arrow(
                x = 0,
                y = 0,
                dx = g2_y*0.8,
                dy = g2_x*0.8,
                color = color_gvects,
                length_includes_head = True,
                width = 0.005,
                head_width = 0.05,
            )
            ax_legend.text(
                x = g1_y*0.96,
                y = g1_x*0.96,
                s = r'$g_1$',
                fontsize = 12,
                color = color_gvects,
                horizontalalignment = 'center',
                verticalalignment = 'center',
            )
            ax_legend.text(
                x = g2_y*0.96,
                y = g2_x*0.96,
                s = r'$g_2$',
                fontsize = 12,
                color = color_gvects,
                horizontalalignment = 'center',
                verticalalignment = 'center',
            )

        # find center and extent
        xmin = np.min([0,0,xaxis_vectx,yaxis_vectx])
        xmax = np.max([0,0,xaxis_vectx,yaxis_vectx])
        ymin = np.min([0,0,xaxis_vecty,yaxis_vecty])
        ymax = np.max([0,0,xaxis_vecty,yaxis_vecty])
        if show_gvects:
            xmin = np.min([xmin,g1_x,g2_x])
            xmax = np.max([xmax,g1_x,g2_x])
            ymin = np.min([ymin,g1_y,g2_y])
            ymax = np.max([ymax,g1_y,g2_y])
        x0 = np.mean([xmin,xmax])
        y0 = np.mean([ymin,ymax])
        xL = (xmax-x0) * legend_camera_length
        yL = (ymax-y0) * legend_camera_length

        # set the extent and aspect
        ax_legend.set_xlim([y0-yL,y0+yL])
        ax_legend.set_ylim([x0-xL,x0+xL])
        ax_legend.invert_yaxis()
        ax_legend.set_aspect("equal")
        ax_legend.axis('off')

        # show/return
        if not returnfig:
            plt.show()
            return
        else:
            axs = ((ax11, ax12), (ax21, ax22))
            return fig, axs

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
        """Adds the vectors g1,g2 to an image, with tail positions at (x0,y0).  g1 and g2 are 2-tuples (gx,gy)."""
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

        Accepts:
            ar                  (arrray) the image
            bragg_directions    (PointList) the bragg scattering directions; must have coordinates
                                'qx','qy','h', and 'k'. Optionally may also have 'l'.
        """
        assert isinstance(bragg_directions, PointList)
        for k in ("qx", "qy", "h", "k"):
            assert k in bragg_directions.data.dtype.fields

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