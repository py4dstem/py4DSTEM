import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import nnls
import matplotlib as mpl
import matplotlib.pyplot as plt

from emdfile import tqdmnd, PointListArray
from py4DSTEM.visualize import show, show_image_grid
from py4DSTEM.process.diffraction.crystal_viz import plot_diffraction_pattern


class Crystal_Phase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray???

    """

    def __init__(
        self,
        crystals,
        orientation_maps,
        name,
    ):
        """
        Args:
            crystals (list):            List of crystal instances
            orientation_maps (list):    List of orientation maps
            name (str):                 Name of Crystal_Phase instance
        """
        if isinstance(crystals, list):
            self.crystals = crystals
            self.num_crystals = len(crystals)
        else:
            raise TypeError("crystals must be a list of crystal instances.")
        if isinstance(orientation_maps, list):
            if len(self.crystals) != len(orientation_maps):
                raise ValueError(
                    "Orientation maps must have the same number of entries as crystals."
                )
            self.orientation_maps = orientation_maps
        else:
            raise TypeError("orientation_maps must be a list of orientation maps.")
        self.name = name
        return

    def plot_all_phase_maps(self, map_scale_values=None, index=0):
        """
        Visualize phase maps of dataset.

        Args:
            map_scale_values (float):   Value to scale correlations by
        """
        phase_maps = []
        if map_scale_values is None:
            map_scale_values = [1] * len(self.orientation_maps)
        corr_sum = np.sum(
            [
                (self.orientation_maps[m].corr[:, :, index] * map_scale_values[m])
                for m in range(len(self.orientation_maps))
            ]
        )
        for m in range(len(self.orientation_maps)):
            phase_maps.append(self.orientation_maps[m].corr[:, :, index] / corr_sum)
        show_image_grid(lambda i: phase_maps[i], 1, len(phase_maps), cmap="inferno")
        return

    def plot_phase_map(self, index=0, cmap=None):
        corr_array = np.dstack(
            [maps.corr[:, :, index] for maps in self.orientation_maps]
        )
        best_corr_score = np.max(corr_array, axis=2)
        best_match_phase = [
            np.where(corr_array[:, :, p] == best_corr_score, True, False)
            for p in range(len(self.orientation_maps))
        ]

        if cmap is None:
            cm = plt.get_cmap("rainbow")
            cmap = [
                cm(1.0 * i / len(self.orientation_maps))
                for i in range(len(self.orientation_maps))
            ]

        fig, (ax) = plt.subplots(figsize=(6, 6))
        ax.matshow(
            np.zeros((self.orientation_maps[0].num_x, self.orientation_maps[0].num_y)),
            cmap="gray",
        )
        ax.axis("off")

        for m in range(len(self.orientation_maps)):
            c0, c1 = (cmap[m][0] * 0.35, cmap[m][1] * 0.35, cmap[m][2] * 0.35, 1), cmap[
                m
            ]
            cm = mpl.colors.LinearSegmentedColormap.from_list("cmap", [c0, c1], N=10)
            ax.matshow(
                np.ma.array(
                    self.orientation_maps[m].corr[:, :, index], mask=best_match_phase[m]
                ),
                cmap=cm,
            )
        plt.show()

        return

    # Potentially introduce a way to check best match out of all orientations in phase plan and plug into model
    # to quantify phase

    # def phase_plan(
    #     self,
    #     method,
    #     zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
    #     angle_step_zone_axis: float = 2.0,
    #     angle_coarse_zone_axis: float = None,
    #     angle_refine_range: float = None,
    #     angle_step_in_plane: float = 2.0,
    #     accel_voltage: float = 300e3,
    #     intensity_power: float = 0.25,
    #     tol_peak_delete=None,
    #     tol_distance: float = 0.01,
    #     fiber_axis = None,
    #     fiber_angles = None,
    # ):
    #     return

    def quantify_phase(
        self,
        pointlistarray,
        tolerance_distance=0.08,
        method="nnls",
        intensity_power=0,
        mask_peaks=None,
    ):
        """
        Quantification of the phase of a crystal based on the crystal instances and the pointlistarray.

        Args:
            pointlisarray (pointlistarray):                 Pointlistarray to quantify phase of
            tolerance_distance (float):                     Distance allowed between a peak and match
            method (str):                                   Numerical method used to quantify phase
            intensity_power (float):                        ...
            mask_peaks (list, optional):                    A pointer of which positions to mask peaks from

        Details:
        """
        if isinstance(pointlistarray, PointListArray):
            phase_weights = np.zeros(
                (
                    pointlistarray.shape[0],
                    pointlistarray.shape[1],
                    np.sum([map.num_matches for map in self.orientation_maps]),
                )
            )
            phase_residuals = np.zeros(pointlistarray.shape)
            for Rx, Ry in tqdmnd(pointlistarray.shape[0], pointlistarray.shape[1]):
                (
                    _,
                    phase_weight,
                    phase_residual,
                    crystal_identity,
                ) = self.quantify_phase_pointlist(
                    pointlistarray,
                    position=[Rx, Ry],
                    tolerance_distance=tolerance_distance,
                    method=method,
                    intensity_power=intensity_power,
                    mask_peaks=mask_peaks,
                )
                phase_weights[Rx, Ry, :] = phase_weight
                phase_residuals[Rx, Ry] = phase_residual
            self.phase_weights = phase_weights
            self.phase_residuals = phase_residuals
            self.crystal_identity = crystal_identity
            return
        else:
            return TypeError("pointlistarray must be of type pointlistarray.")
        return

    def quantify_phase_pointlist(
        self,
        pointlistarray,
        position,
        method="nnls",
        tolerance_distance=0.08,
        intensity_power=0,
        mask_peaks=None,
    ):
        """
        Args:
            pointlisarray (pointlistarray):                 Pointlistarray to quantify phase of
            position (tuple/list):                          Position of pointlist in pointlistarray
            tolerance_distance (float):                     Distance allowed between a peak and match
            method (str):                                   Numerical method used to quantify phase
            intensity_power (float):                        ...
            mask_peaks (list, optional):                    A pointer of which positions to mask peaks from

        Returns:
            pointlist_peak_intensity_matches (np.ndarray):  Peak matches in the rows of array and the crystals in the columns
            phase_weights (np.ndarray):                     Weights of each phase
            phase_residuals (np.ndarray):                   Residuals
            crystal_identity (list):                        List of lists, where the each entry represents the position in the
                                                                crystal and orientation match that is associated with the phase
                                                                weights. for example, if the output was [[0,0], [0,1], [1,0], [0,1]],
                                                                the first entry [0,0] in phase weights is associated with the first crystal
                                                                the first match within that crystal. [0,1] is the first crystal and the
                                                                second match within that crystal.
        """
        # Things to add:
        #     1. Better cost for distance from peaks in pointlists
        #     2. Iterate through multiple tolerance_distance values to find best value. Cost function residuals, or something else?

        pointlist = pointlistarray.get_pointlist(position[0], position[1])
        pl_mask = np.where((pointlist["qx"] == 0) & (pointlist["qy"] == 0), 1, 0)
        pointlist.remove(pl_mask)
        # False Negatives (exp peak with no match in crystal instances) will appear here, already coded in

        if intensity_power == 0:
            pl_intensities = np.ones(pointlist["intensity"].shape)
        else:
            pl_intensities = pointlist["intensity"] ** intensity_power
        # Prepare matches for modeling
        pointlist_peak_matches = []
        crystal_identity = []

        for c in range(len(self.crystals)):
            for m in range(self.orientation_maps[c].num_matches):
                crystal_identity.append([c, m])
                phase_peak_match_intensities = np.zeros((pointlist["intensity"].shape))
                bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
                    self.orientation_maps[c].get_orientation(position[0], position[1]),
                    ind_orientation=m,
                )
                # Find the best match peak within tolerance_distance and add value in the right position
                for d in range(pointlist["qx"].shape[0]):
                    distances = []
                    for p in range(bragg_peaks_fit["qx"].shape[0]):
                        distances.append(
                            np.sqrt(
                                (pointlist["qx"][d] - bragg_peaks_fit["qx"][p]) ** 2
                                + (pointlist["qy"][d] - bragg_peaks_fit["qy"][p]) ** 2
                            )
                        )
                    ind = np.where(distances == np.min(distances))[0][0]

                    # Potentially for-loop over multiple values for 'tolerance_distance' to find best tolerance_distance value
                    if distances[ind] <= tolerance_distance:
                        ## Somewhere in this if statement is probably where better distances from the peak should be coded in
                        if (
                            intensity_power == 0
                        ):  # This could potentially be a different intensity_power arg
                            phase_peak_match_intensities[d] = 1 ** (
                                (tolerance_distance - distances[ind])
                                / tolerance_distance
                            )
                        else:
                            phase_peak_match_intensities[d] = bragg_peaks_fit[
                                "intensity"
                            ][ind] ** (
                                (tolerance_distance - distances[ind])
                                / tolerance_distance
                            )
                    else:
                        ## This is probably where the false positives (peaks in crystal but not in experiment) should be handled
                        continue

                pointlist_peak_matches.append(phase_peak_match_intensities)
                pointlist_peak_intensity_matches = np.dstack(pointlist_peak_matches)
                pointlist_peak_intensity_matches = (
                    pointlist_peak_intensity_matches.reshape(
                        pl_intensities.shape[0],
                        pointlist_peak_intensity_matches.shape[-1],
                    )
                )

        if len(pointlist["qx"]) > 0:
            if mask_peaks is not None:
                for i in range(len(mask_peaks)):
                    if mask_peaks[i] == None:  # noqa: E711
                        continue
                    inds_mask = np.where(
                        pointlist_peak_intensity_matches[:, mask_peaks[i]] != 0
                    )[0]
                    for mask in range(len(inds_mask)):
                        pointlist_peak_intensity_matches[inds_mask[mask], i] = 0

            if method == "nnls":
                phase_weights, phase_residuals = nnls(
                    pointlist_peak_intensity_matches, pl_intensities
                )

            elif method == "lstsq":
                phase_weights, phase_residuals, rank, singluar_vals = lstsq(
                    pointlist_peak_intensity_matches, pl_intensities, rcond=-1
                )
                phase_residuals = np.sum(phase_residuals)
            else:
                raise ValueError(method + " Not yet implemented. Try nnls or lstsq.")
        else:
            phase_weights = np.zeros((pointlist_peak_intensity_matches.shape[1],))
            phase_residuals = np.NaN
        return (
            pointlist_peak_intensity_matches,
            phase_weights,
            phase_residuals,
            crystal_identity,
        )

    # def plot_peak_matches(
    #     self,
    #     pointlistarray,
    #     position,
    #     tolerance_distance,
    #     ind_orientation,
    #     pointlist_peak_intensity_matches,
    # ):
    #     """
    #     A method to view how the tolerance distance impacts the peak matches associated with
    #     the quantify_phase_pointlist method.

    #     Args:
    #         pointlistarray,
    #         position,
    #         tolerance_distance
    #         pointlist_peak_intensity_matches
    #     """
    #     pointlist = pointlistarray.get_pointlist(position[0],position[1])

    #     for m in range(pointlist_peak_intensity_matches.shape[1]):
    #         bragg_peaks_fit = self.crystals[m].generate_diffraction_pattern(
    #                 self.orientation_maps[m].get_orientation(position[0], position[1]),
    #                 ind_orientation = ind_orientation
    #                 )
    #         peak_inds = np.where(bragg_peaks_fit.data['intensity'] == pointlist_peak_intensity_matches[:,m])

    #         fig, (ax1, ax2) = plt.subplots(2,1,figsize = figsize)
    #         ax1 = plot_diffraction_pattern(pointlist,)
    #     return
