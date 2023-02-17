import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import nnls
import matplotlib as mpl
import matplotlib.pyplot as plt

from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.visualize import show, show_image_grid
# from py4DSTEM.io.datastructure.emd.pointlistarray import PointListArray
# from py4DSTEM.process.diffraction.crystal_viz import plot_diffraction_pattern
from py4DSTEM.io.datastructure import PointList, PointListArray

from dataclasses import dataclass, field

@dataclass
class CrystalPhase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray???

    """

    name: str
    num_crystals: int

    def __init__(
        self,
        crystals,
        names = None,
    ):
        """
        Args:
            crystals (list):            List of crystal instances
            name (str):                 Name of CrystalPhase instance
        """
        # validate inputs
        assert(isinstance(crystals,list)), '`crystals` must be a list of crystal instances'
        for xtal in crystals:
            assert(hasattr(xtal,'orientation_map')), '`crystals` elements must be Crystal instances with a .orientation_map - try running .match_orientations'

        # assign variables
        self.num_crystals = len(crystals)
        self.crystals = crystals
        # self.orientation_maps = [xtal.orientation_map for xtal in crystals]

        # Get some attributes from crystals
        self.k_max = np.zeros(self.num_crystals, dtype='int')
        self.num_matches = np.zeros(self.num_crystals, dtype='int')
        self.crystal_identity = np.zeros((0,2), dtype='int')
        for a0 in range(self.num_crystals):
            self.k_max[a0] = self.crystals[a0].k_max
            self.num_matches[a0] = self.crystals[a0].orientation_map.num_matches
            for a1 in range(self.num_matches[a0]):
                self.crystal_identity = np.append(self.crystal_identity,np.array((a0,a1),dtype='int')[None,:], axis=0)

        self.num_fits = np.sum(self.num_matches)
        # for a0 in range(self.crystals):


        if names is not None:
            self.names = names
        else:
            self.names = ['crystal'] * self.num_crystals



    def quantify_single_pattern(
        self,
        pointlistarray: PointListArray,
        xy_position = (0,0),
        corr_kernel_size = 0.04,
        include_false_positives = True,
        sigma_excitation_error = 0.02,
        power_experiment = 0.5,
        power_calculated = 0.5,
        plot_result = True,
        scale_markers_experiment = 10,
        scale_markers_calculated = 4000,
        crystal_inds_plot = None,
        figsize = (12,8),
        returnfig = False,
        ):
        """
        Quantify the phase for a single diffraction pattern.
        """

        # tolerance
        tol2 = 4e-4

        # Experimental values
        bragg_peaks = pointlistarray.get_pointlist(xy_position[0],xy_position[1]).copy()
        keep = bragg_peaks.data["qx"]**2 + bragg_peaks.data["qy"]**2 > tol2
        # ind_center_beam = np.argmin(
        #     bragg_peaks.data["qx"]**2 + bragg_peaks.data["qy"]**2)
        # mask = np.ones_like(bragg_peaks.data["qx"], dtype='bool')
        # mask[ind_center_beam] = False
        # bragg_peaks.remove(ind_center_beam)
        qx = bragg_peaks.data["qx"][keep]
        qy = bragg_peaks.data["qy"][keep]
        qx0 = bragg_peaks.data["qx"][np.logical_not(keep)]
        qy0 = bragg_peaks.data["qy"][np.logical_not(keep)]
        if power_experiment == 0:
            intensity = np.ones_like(qx)
            intensity0 = np.ones_like(qx0)
        else:
            intensity = bragg_peaks.data["intensity"][keep]**power_experiment
            intensity0 = bragg_peaks.data["intensity"][np.logical_not(keep)]**power_experiment

        # init basis array
        if include_false_positives:
            basis = np.zeros((intensity.shape[0], self.num_fits))
            unpaired_peaks = []
        else:
            basis = np.zeros((intensity.shape[0], self.num_fits))

        # kernel radius squared
        radius_max_2 = corr_kernel_size**2

        # init for plotting
        if plot_result:
            library_peaks = []
            library_int = []
            library_matches = []

        # Generate point list data, match to experimental peaks
        for a0 in range(self.num_fits):
            c = self.crystal_identity[a0,0]
            m = self.crystal_identity[a0,1]
            # for c in range(self.num_crystals):
            #     for m in range(self.num_matches[c]):
            #         ind_match += 1

            # Generate simulated peaks
            bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
                self.crystals[c].orientation_map.get_orientation(
                    xy_position[0], xy_position[1]
                ),
                ind_orientation = m,
                sigma_excitation_error = sigma_excitation_error,
            )
            del_peak = bragg_peaks_fit.data["qx"]**2 \
                +      bragg_peaks_fit.data["qy"]**2 < tol2
            bragg_peaks_fit.remove(del_peak)

            # peak intensities
            if power_calculated == 0:
                int_fit = np.ones_like(bragg_peaks_fit.data["qx"])
            else:
                int_fit = bragg_peaks_fit.data['intensity']**power_calculated
            
            # Pair peaks to experiment
            if plot_result:
                matches = np.zeros((bragg_peaks_fit.data.shape[0]),dtype='bool')

            for a1 in range(bragg_peaks_fit.data.shape[0]):
                dist2 = (bragg_peaks_fit.data['qx'][a1] - qx)**2 \
                    +   (bragg_peaks_fit.data['qy'][a1] - qy)**2
                ind_min = np.argmin(dist2)
                val_min = dist2[ind_min]

                if val_min < radius_max_2:
                    weight = 1 - np.sqrt(dist2[ind_min]) / corr_kernel_size
                    basis[ind_min,a0] = weight * int_fit[a1]
                    if plot_result:
                        matches[a1] = True
                elif include_false_positives:
                    unpaired_peaks.append([a0,int_fit[a1]])

            if plot_result:
                library_peaks.append(bragg_peaks_fit)                
                library_int.append(int_fit)
                library_matches.append(matches)

        # If needed, augment basis and observations with false positives
        if include_false_positives:
            basis_aug = np.zeros((len(unpaired_peaks),self.num_fits))
            for a0 in range(len(unpaired_peaks)):
                basis_aug[a0,unpaired_peaks[a0][0]] = unpaired_peaks[a0][1]

            basis = np.vstack((basis, basis_aug))
            obs = np.hstack((intensity, np.zeros(len(unpaired_peaks))))
        else:
            obs = intensity
        
        # Solve for phase coefficients
        phase_weights, phase_residual = nnls(
            basis,
            obs,
        )

        print(np.round(phase_weights,decimals=2))
        # print()
        # print(np.array(unpaired_peaks))
        # print()
        
        # initialize matching array


                # phase_peak_match_intensities = np.zeros((pointlist['intensity'].shape))
                # bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
                #     self.orientation_maps[c].get_orientation(position[0], position[1]),
                #     ind_orientation = m
                # )


        # Plotting
        if plot_result:
            # fig, ax = plt.subplots(figsize=figsize)
            fig = plt.figure(figsize=figsize)
            # if plot_layout == 0:
            #     ax_x = fig.add_axes(
            #         [0.0+figbound[0], 0.0, 0.4-2*+figbound[0], 1.0])
            ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
            ax_leg = fig.add_axes([0.68, 0.0, 0.3, 1.0])

            # plot the experimental radii
            t = np.linspace(0,2*np.pi,91,endpoint=True)
            ct = np.cos(t) * corr_kernel_size
            st = np.sin(t) * corr_kernel_size
            for a0 in range(qx.shape[0]):
                ax.plot(
                    qy[a0] + st,
                    qx[a0] + ct,
                    color = 'k',
                    linewidth = 1,
                    )


            # plot the experimental peaks
            ax.scatter(
                qy0,
                qx0,
                s = scale_markers_experiment * intensity0,
                marker = "o",
                facecolor = [0.0, 0.0, 0.0],
                )
            ax.scatter(
                qy,
                qx,
                s = scale_markers_experiment * intensity,
                marker = "o",
                facecolor = [0.0, 0.0, 0.0],
                )
            # legend
            k_max = np.max(self.k_max)
            dx_leg =  -0.05*k_max
            dy_leg =   0.04*k_max
            text_params = {
                "va": "center",
                "ha": "left",
                "family": "sans-serif",
                "fontweight": "normal",
                "color": "k",
                "size": 14,
            }
            ax_leg.plot(
                    0 + st*0.5,
                    -dx_leg + ct*0.5,
                    color = 'k',
                    linewidth = 1,
                    )
            ax_leg.scatter(
                0,
                0,
                s = 200,
                marker = "o",
                facecolor = [0.0, 0.0, 0.0],
                )
            ax_leg.text(
                dy_leg,
                0,
                'Experimental peaks',
                **text_params)
            ax_leg.text(
                dy_leg,
                -dx_leg,
                'Correlation radius',
                **text_params)



            # plot calculated diffraction patterns
            # Currently just hardcoded for 6 max phases
            cvals = np.array((
                (1.0,0.0,0.0,1.0),
                (0.0,0.8,1.0,1.0),
                (0.0,0.6,0.0,1.0),
                (1.0,0.0,1.0,1.0),
                (0.0,0.2,1.0,1.0),
                (1.0,0.8,0.0,1.0),
            ))
            uvals = np.array((
                (1.0,0.0,0.0,0.2),
                (0.0,0.8,1.0,0.2),
                (0.0,0.6,0.0,0.2),
                (1.0,0.0,1.0,0.2),
                (0.0,0.2,1.0,0.2),
                (1.0,0.8,0.0,0.2),
            ))
            mvals = ['v','^','<','>','d','s',]

            for a0 in range(self.num_fits):
                c = self.crystal_identity[a0,0]
                m = self.crystal_identity[a0,1]

                if crystal_inds_plot == None or np.min(np.abs(c - crystal_inds_plot)) == 0:

                    qx_fit = library_peaks[a0].data['qx']
                    qy_fit = library_peaks[a0].data['qy']
                    int_fit = library_int[a0]
                    matches_fit = library_matches[a0]

                    if np.mod(m,2) == 0:
                        ax.scatter(
                            qy_fit[matches_fit],
                            qx_fit[matches_fit],
                            s = scale_markers_calculated * int_fit[matches_fit],
                            marker = mvals[c],
                            facecolor = cvals[c,:],
                            )
                        ax.scatter(
                            qy_fit[np.logical_not(matches_fit)],
                            qx_fit[np.logical_not(matches_fit)],
                            s = scale_markers_calculated * int_fit[np.logical_not(matches_fit)],
                            marker = mvals[c],
                            facecolor = uvals[c,:],
                            )

                        # legend
                        ax_leg.scatter(
                            0,
                            dx_leg*(a0+1),
                            s = 200,
                            marker = mvals[c],
                            facecolor = cvals[c,:],
                            )
                    else:
                        ax.scatter(
                            qy_fit[matches_fit],
                            qx_fit[matches_fit],
                            s = scale_markers_calculated * int_fit[matches_fit],
                            marker = mvals[c],
                            edgecolors = cvals[c,:],
                            facecolors = (1,1,1,0.5),
                            linewidth = 2,
                            )
                        ax.scatter(
                            qy_fit[np.logical_not(matches_fit)],
                            qx_fit[np.logical_not(matches_fit)],
                            s = scale_markers_calculated * int_fit[np.logical_not(matches_fit)],
                            marker = mvals[c],
                            edgecolors = uvals[c,:],
                            facecolors = (1,1,1,0.5),
                            linewidth = 2,
                            )

                        # legend
                        ax_leg.scatter(
                            0,
                            dx_leg*(a0+1),
                            s = 200,
                            marker = mvals[c],
                            edgecolors = cvals[c,:],
                            facecolors = (1,1,1,0.5),
                            )

                    # legend text
                    ax_leg.text(
                        dy_leg,
                        (a0+1)*dx_leg,
                        self.names[c],
                        **text_params)


            # appearance
            ax.set_xlim((-k_max, k_max))
            ax.set_ylim((-k_max, k_max))

            ax_leg.set_xlim((-0.1*k_max, 0.4*k_max))
            ax_leg.set_ylim((-0.5*k_max, 0.5*k_max))
            ax_leg.set_axis_off()


        if returnfig:
            return phase_weights, phase_residual, fig, ax
        else:
            return phase_weights, phase_residual

    def quantify_phase(


        ):





    def plot_all_phase_maps(
        self,
        map_scale_values = None,
        index = 0,
        layout = 0,
    ):
        """
        Visualize phase maps of dataset.

        Args:
            map_scale_values (float):   Value to scale correlations by
        """
        phase_maps = []
        if map_scale_values == None:
            map_scale_values = [1] * len(self.orientation_maps)
        corr_sum = np.sum([(self.orientation_maps[m].corr[:,:,index] * map_scale_values[m]) for m in range(len(self.orientation_maps))])
        for m in range(len(self.orientation_maps)):
            phase_maps.append(self.orientation_maps[m].corr[:,:,index] / corr_sum)
        if layout == 0:
            show_image_grid(lambda i:phase_maps[i], 1, len(phase_maps), cmap = 'inferno')
        elif layout == 1:
            show_image_grid(lambda i:phase_maps[i], len(phase_maps), 1, cmap = 'inferno')
        return

    def plot_phase_map(
        self,
        index = 0,
        cmap = None
        
    ):
        corr_array = np.dstack([maps.corr[:,:,index] for maps in self.orientation_maps])
        best_corr_score = np.max(corr_array,axis=2)
        best_match_phase = [np.where(corr_array[:,:,p] == best_corr_score, True,False)
                            for p in range(len(self.orientation_maps))
                            ]

        if cmap == None:
            cm = plt.get_cmap('rainbow')
            cmap = [cm(1.*i/len(self.orientation_maps)) for i in range(len(self.orientation_maps))]
        
        fig, (ax) = plt.subplots(figsize = (6,6))
        ax.matshow(np.zeros((self.orientation_maps[0].num_x, self.orientation_maps[0].num_y)), cmap = 'gray')
        ax.axis('off')
        
        for m in range(len(self.orientation_maps)):
            c0, c1 = (cmap[m][0]*0.35,cmap[m][1]*0.35,cmap[m][2]*0.35,1), cmap[m]
            cm = mpl.colors.LinearSegmentedColormap.from_list('cmap', [c0,c1], N = 10)
            ax.matshow(
                np.ma.array(
                    self.orientation_maps[m].corr[:,:,index], 
                    mask = best_match_phase[m]), 
                cmap = cm)
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
    
    # def quantify_phase(
    #     self,
    #     pointlistarray,
    #     tolerance_distance = 0.08,
    #     method = 'nnls',
    #     intensity_power = 0,
    #     mask_peaks = None
    # ):
    #     """
    #     Quantification of the phase of a crystal based on the crystal instances and the pointlistarray.

    #     Args:
    #         pointlisarray (pointlistarray):                 Pointlistarray to quantify phase of
    #         tolerance_distance (float):                     Distance allowed between a peak and match
    #         method (str):                                   Numerical method used to quantify phase
    #         intensity_power (float):                        ...
    #         mask_peaks (list, optional):                    A pointer of which positions to mask peaks from
        
    #     Details:
    #     """
    #     if isinstance(pointlistarray, PointListArray):

    #         phase_weights = np.zeros((
    #             pointlistarray.shape[0],
    #             pointlistarray.shape[1],
    #             np.sum([map.num_matches for map in self.orientation_maps])
    #             ))
    #         phase_residuals = np.zeros(pointlistarray.shape)
    #         for Rx, Ry in tqdmnd(pointlistarray.shape[0], pointlistarray.shape[1]):
    #             _, phase_weight, phase_residual, crystal_identity = self.quantify_phase_pointlist(
    #                 pointlistarray,
    #                 position = [Rx, Ry],
    #                 tolerance_distance=tolerance_distance,
    #                 method = method,
    #                 intensity_power = intensity_power,
    #                 mask_peaks = mask_peaks
    #             )
    #             phase_weights[Rx,Ry,:] = phase_weight
    #             phase_residuals[Rx,Ry] = phase_residual
    #         self.phase_weights = phase_weights
    #         self.phase_residuals = phase_residuals
    #         self.crystal_identity = crystal_identity
    #         return
    #     else:
    #         return TypeError('pointlistarray must be of type pointlistarray.')
    #     return
    
    # def quantify_phase_pointlist(
    #     self,
    #     pointlistarray,
    #     position,
    #     method = 'nnls', 
    #     tolerance_distance = 0.08,
    #     intensity_power = 0,
    #     mask_peaks = None
    # ):
    #     """
    #     Args:
    #         pointlisarray (pointlistarray):                 Pointlistarray to quantify phase of
    #         position (tuple/list):                          Position of pointlist in pointlistarray
    #         tolerance_distance (float):                     Distance allowed between a peak and match
    #         method (str):                                   Numerical method used to quantify phase
    #         intensity_power (float):                        ...
    #         mask_peaks (list, optional):                    A pointer of which positions to mask peaks from

    #     Returns:
    #         pointlist_peak_intensity_matches (np.ndarray):  Peak matches in the rows of array and the crystals in the columns
    #         phase_weights (np.ndarray):                     Weights of each phase
    #         phase_residuals (np.ndarray):                   Residuals
    #         crystal_identity (list):                        List of lists, where the each entry represents the position in the
    #                                                             crystal and orientation match that is associated with the phase
    #                                                             weights. for example, if the output was [[0,0], [0,1], [1,0], [0,1]],
    #                                                             the first entry [0,0] in phase weights is associated with the first crystal
    #                                                             the first match within that crystal. [0,1] is the first crystal and the
    #                                                             second match within that crystal.
    #     """
    #     # Things to add:
    #     #     1. Better cost for distance from peaks in pointlists
    #     #     2. Iterate through multiple tolerance_distance values to find best value. Cost function residuals, or something else?
        
    #     pointlist = pointlistarray.get_pointlist(position[0], position[1]) 
    #     pl_mask = np.where((pointlist['qx'] == 0) & (pointlist['qy'] == 0), 1, 0)
    #     pointlist.remove(pl_mask)
    #     # False Negatives (exp peak with no match in crystal instances) will appear here, already coded in
    
    #     if intensity_power == 0:
    #         pl_intensities = np.ones(pointlist['intensity'].shape)
    #     else:
    #         pl_intensities = pointlist['intensity']**intensity_power
    #     #Prepare matches for modeling
    #     pointlist_peak_matches = []
    #     crystal_identity = []
        
    #     for c in range(len(self.crystals)):
    #         for m in range(self.orientation_maps[c].num_matches):
    #             crystal_identity.append([c,m])
    #             phase_peak_match_intensities = np.zeros((pointlist['intensity'].shape))
    #             bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
    #                 self.orientation_maps[c].get_orientation(position[0], position[1]),
    #                 ind_orientation = m
    #             )
    #             #Find the best match peak within tolerance_distance and add value in the right position
    #             for d in range(pointlist['qx'].shape[0]):
    #                 distances = []
    #                 for p in range(bragg_peaks_fit['qx'].shape[0]):
    #                     distances.append(
    #                         np.sqrt((pointlist['qx'][d] - bragg_peaks_fit['qx'][p])**2 + 
    #                                 (pointlist['qy'][d]-bragg_peaks_fit['qy'][p])**2)
    #                     )
    #                 ind = np.where(distances == np.min(distances))[0][0]
                    
    #                 #Potentially for-loop over multiple values for 'tolerance_distance' to find best tolerance_distance value
    #                 if distances[ind] <= tolerance_distance:
    #                     ## Somewhere in this if statement is probably where better distances from the peak should be coded in
    #                     if intensity_power == 0: #This could potentially be a different intensity_power arg
    #                         phase_peak_match_intensities[d] = 1**((tolerance_distance-distances[ind])/tolerance_distance)  
    #                     else:
    #                         phase_peak_match_intensities[d] = bragg_peaks_fit['intensity'][ind]**((tolerance_distance-distances[ind])/tolerance_distance)
    #                 else:
    #                     ## This is probably where the false positives (peaks in crystal but not in experiment) should be handled
    #                     continue   
                
    #             pointlist_peak_matches.append(phase_peak_match_intensities)
    #             pointlist_peak_intensity_matches = np.dstack(pointlist_peak_matches)
    #             pointlist_peak_intensity_matches = pointlist_peak_intensity_matches.reshape(
    #                 pl_intensities.shape[0],
    #                 pointlist_peak_intensity_matches.shape[-1]
    #                 )
        
    #     if len(pointlist['qx']) > 0:    
    #         if mask_peaks is not None:
    #             for i in range(len(mask_peaks)):
    #                 if mask_peaks[i] == None:
    #                     continue
    #                 inds_mask = np.where(pointlist_peak_intensity_matches[:,mask_peaks[i]] != 0)[0]
    #                 for mask in range(len(inds_mask)):
    #                     pointlist_peak_intensity_matches[inds_mask[mask],i] = 0

    #         if method == 'nnls':    
    #             phase_weights, phase_residuals = nnls(
    #                 pointlist_peak_intensity_matches,
    #                 pl_intensities
    #             )
            
    #         elif method == 'lstsq':
    #             phase_weights, phase_residuals, rank, singluar_vals = lstsq(
    #                 pointlist_peak_intensity_matches,
    #                 pl_intensities,
    #                 rcond = -1
    #             )
    #             phase_residuals = np.sum(phase_residuals)
    #         else:
    #             raise ValueError(method + ' Not yet implemented. Try nnls or lstsq.')   
    #     else:
    #         phase_weights = np.zeros((pointlist_peak_intensity_matches.shape[1],))
    #         phase_residuals = np.NaN
    #     return pointlist_peak_intensity_matches, phase_weights, phase_residuals, crystal_identity
    
    # # def plot_peak_matches(
    # #     self,
    # #     pointlistarray,
    # #     position,
    # #     tolerance_distance, 
    # #     ind_orientation,
    # #     pointlist_peak_intensity_matches,
    # # ):
    # #     """
    # #     A method to view how the tolerance distance impacts the peak matches associated with
    # #     the quantify_phase_pointlist method.
        
    # #     Args:
    # #         pointlistarray,
    # #         position,
    # #         tolerance_distance
    # #         pointlist_peak_intensity_matches
    # #     """
    # #     pointlist = pointlistarray.get_pointlist(position[0],position[1])
        
    # #     for m in range(pointlist_peak_intensity_matches.shape[1]):
    # #         bragg_peaks_fit = self.crystals[m].generate_diffraction_pattern(
    # #                 self.orientation_maps[m].get_orientation(position[0], position[1]),
    # #                 ind_orientation = ind_orientation
    # #                 )
    # #         peak_inds = np.where(bragg_peaks_fit.data['intensity'] == pointlist_peak_intensity_matches[:,m])
        
    # #         fig, (ax1, ax2) = plt.subplots(2,1,figsize = figsize)
    # #         ax1 = plot_diffraction_pattern(pointlist,)
    # #     return
    
