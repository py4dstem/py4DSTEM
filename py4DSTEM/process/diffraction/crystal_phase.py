import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import nnls
import matplotlib as mpl
import matplotlib.pyplot as plt

<<<<<<< HEAD
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.visualize import show, show_image_grid
# from py4DSTEM.io.datastructure.emd.pointlistarray import PointListArray
# from py4DSTEM.process.diffraction.crystal_viz import plot_diffraction_pattern
from py4DSTEM.io.datastructure import PointList, PointListArray

from dataclasses import dataclass, field
=======
from emdfile import tqdmnd, PointListArray
from py4DSTEM.visualize import show, show_image_grid
from py4DSTEM.process.diffraction.crystal_viz import plot_diffraction_pattern
>>>>>>> dev

@dataclass
class CrystalPhase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray???

    """

    names: str
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
        self.k_max = np.zeros(self.num_crystals)
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
<<<<<<< HEAD
            self.names = ['crystal'] * self.num_crystals



    def quantify_single_pattern(
=======
            raise TypeError('orientation_maps must be a list of orientation maps.')
        self.name = name
        return

    def plot_all_phase_maps(
>>>>>>> dev
        self,
        pointlistarray: PointListArray,
        xy_position = (0,0),
        corr_kernel_size = 0.04,
        include_false_positives = True,
        sigma_excitation_error = 0.02,
        power_experiment = 0.5,
        power_calculated = 0.5,
        plot_result = True,
        scale_markers_experiment = 4,
        scale_markers_calculated = 4000,
        crystal_inds_plot = None,
        phase_colors = np.array((
            (1.0,0.0,0.0,1.0),
            (0.0,0.8,1.0,1.0),
            (0.0,0.6,0.0,1.0),
            (1.0,0.0,1.0,1.0),
            (0.0,0.2,1.0,1.0),
            (1.0,0.8,0.0,1.0),
        )),
        figsize = (12,8),
        verbose = True,
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
        int_total = np.sum(intensity) 


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

            # Loop over all people
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
        try:
            phase_weights, phase_residual = nnls(
                basis,
                obs,
            )
        except:
            phase_weights = np.zeros(self.num_fits)
            phase_residual = np.sqrt(np.sum(intensity**2))

        if verbose:
            ind_max = np.argmax(phase_weights)
            # print()
            print('\033[1m' + 'phase_weight   or_ind   name' + '\033[0m')
            # print()
            for a0 in range(self.num_fits):
                c = self.crystal_identity[a0,0]
                m = self.crystal_identity[a0,1]
                line = '{:>12} {:>8}   {:<12}'.format(
                    np.round(phase_weights[a0],decimals=2), 
                    m,
                    self.names[c]
                    )
                if a0 == ind_max:
                    print('\033[1m' + line + '\033[0m')
                else:
                    print(line)
            # print()

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
            uvals = phase_colors.copy()
            uvals[:,3] = 0.3
            # uvals = np.array((
            #     (1.0,0.0,0.0,0.2),
            #     (0.0,0.8,1.0,0.2),
            #     (0.0,0.6,0.0,0.2),
            #     (1.0,0.0,1.0,0.2),
            #     (0.0,0.2,1.0,0.2),
            #     (1.0,0.8,0.0,0.2),
            # ))
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
                            facecolor = phase_colors[c,:],
                            )
                        ax.scatter(
                            qy_fit[np.logical_not(matches_fit)],
                            qx_fit[np.logical_not(matches_fit)],
                            s = scale_markers_calculated * int_fit[np.logical_not(matches_fit)],
                            marker = mvals[c],
                            facecolor = phase_colors[c,:],
                            )

                        # legend
                        ax_leg.scatter(
                            0,
                            dx_leg*(a0+1),
                            s = 200,
                            marker = mvals[c],
                            facecolor = phase_colors[c,:],
                            )
                    else:
                        ax.scatter(
                            qy_fit[matches_fit],
                            qx_fit[matches_fit],
                            s = scale_markers_calculated * int_fit[matches_fit],
                            marker = mvals[c],
                            edgecolors = uvals[c,:],
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
                            edgecolors = uvals[c,:],
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
            return phase_weights, phase_residual, int_total, fig, ax
        else:
            return phase_weights, phase_residual, int_total

    def quantify_phase(
        self,
        pointlistarray: PointListArray,
        corr_kernel_size = 0.04,
        include_false_positives = True,
        sigma_excitation_error = 0.02,
        power_experiment = 0.5,
        power_calculated = 0.5,
        progress_bar = True,
        ):
        """
        Quantify phase of all diffraction patterns.
        """

        # init results arrays
        self.phase_weights = np.zeros((
            pointlistarray.shape[0],
            pointlistarray.shape[1],
            self.num_fits,
        ))
        self.phase_residuals = np.zeros((
            pointlistarray.shape[0],
            pointlistarray.shape[1],
        ))
        self.int_total = np.zeros((
            pointlistarray.shape[0],
            pointlistarray.shape[1],
        ))
        
        for rx, ry in tqdmnd(
                *pointlistarray.shape,
                desc="Matching Orientations",
                unit=" PointList",
                disable=not progress_bar,
            ):
            # calculate phase weights
            phase_weights, phase_residual, int_peaks = self.quantify_single_pattern(
                pointlistarray = pointlistarray,
                xy_position = (rx,ry),
                corr_kernel_size = corr_kernel_size,
                include_false_positives = include_false_positives,
                sigma_excitation_error = sigma_excitation_error,
                power_experiment = power_experiment,
                power_calculated = power_calculated,
                plot_result = False,
                verbose = False,
                returnfig = False,
                )
            self.phase_weights[rx,ry] = phase_weights
            self.phase_residuals[rx,ry] = phase_residual
            self.int_total[rx,ry] = int_peaks


    def plot_phase_weights(
        self,
        weight_range = (0.5,1,0),
        weight_normalize = False,
        total_intensity_normalize = True,
        cmap = 'gray',
        show_ticks = False,
        show_axes = True,
        layout = 0,
        figsize = (6,6),
        returnfig = False,
        ):
        """
        Plot the individual phase weight maps and residuals.
        """

        # Normalization if required to total DF peak intensity
        phase_weights = self.phase_weights.copy()
        phase_residuals = self.phase_residuals.copy()
        if total_intensity_normalize:
            sub = self.int_total > 0.0
            for a0 in range(self.num_fits):
                phase_weights[:,:,a0][sub] /= self.int_total[sub]
            phase_residuals[sub] /= self.int_total[sub]

        # intensity range for plotting
        if weight_normalize:
            scale = np.median(np.max(phase_weights,axis=2))
        else:
            scale = 1
        weight_range = np.array(weight_range) * scale

        # plotting
        if layout == 0:
            fig,ax = plt.subplots(
                1,
                self.num_crystals + 1,
                figsize=(figsize[0],(self.num_fits+1)*figsize[1]))
        elif layout == 1:
            fig,ax = plt.subplots(
                self.num_crystals + 1,
                1,
                figsize=(figsize[0],(self.num_fits+1)*figsize[1]))

        for a0 in range(self.num_crystals):
            sub = self.crystal_identity[:,0] == a0
            im = np.sum(phase_weights[:,:,sub],axis=2)
            im = np.clip(
                (im - weight_range[0]) / (weight_range[1] - weight_range[0]),
                0,1)
            ax[a0].imshow(
                im,
                vmin = 0,
                vmax = 1,
                cmap = cmap,
            )
            ax[a0].set_title(
                self.names[a0],
                fontsize = 16,
            )
            if not show_ticks:
                ax[a0].set_xticks([])
                ax[a0].set_yticks([])
            if not show_axes:
                ax[a0].set_axis_off()

        # plot residuals
        im = np.clip(
                (phase_residuals - weight_range[0]) \
                / (weight_range[1] - weight_range[0]),
                0,1)
        ax[self.num_crystals].imshow(
            im,
            vmin = 0,
            vmax = 1,
            cmap = cmap,
        )
        ax[self.num_crystals].set_title(
            'Residuals',
            fontsize = 16,
        )
        if not show_ticks:
            ax[self.num_crystals].set_xticks([])
            ax[self.num_crystals].set_yticks([])
        if not show_axes:
            ax[self.num_crystals].set_axis_off()

        if returnfig:
            return fig, ax


    def plot_phase_maps(
        self,
        weight_threshold = 0.5,
        weight_normalize = True,
        total_intensity_normalize = True,

        plot_combine = False,
        crystal_inds_plot = None,
        phase_colors = np.array((
            (1.0,0.0,0.0),
            (0.0,0.8,1.0),
            (0.0,0.6,0.0),
            (1.0,0.0,1.0),
            (0.0,0.2,1.0),
            (1.0,0.8,0.0),
        )),
        show_ticks = False,
        show_axes = True,
        layout = 0,
        figsize = (6,6),
        return_phase_estimate = False,
        return_rgb_images = False,
        returnfig = False,
        ):
        """
        Plot the individual phase weight maps and residuals.
        """

        phase_weights = self.phase_weights.copy()
        if total_intensity_normalize:
            sub = self.int_total > 0.0
            for a0 in range(self.num_fits):
                phase_weights[:,:,a0][sub] /= self.int_total[sub]

        # intensity range for plotting
        if weight_normalize:
            scale = np.median(np.max(phase_weights,axis=2))
        else:
            scale = 1
        weight_threshold = weight_threshold * scale

        # init
        im_all = np.zeros((
            self.num_crystals,
            self.phase_weights.shape[0],
            self.phase_weights.shape[1]))
        im_rgb_all = np.zeros((
            self.num_crystals,
            self.phase_weights.shape[0],
            self.phase_weights.shape[1],
            3))

        # phase weights over threshold
        for a0 in range(self.num_crystals):
            sub = self.crystal_identity[:,0] == a0
            im = np.sum(phase_weights[:,:,sub],axis=2)
            im_all[a0] = np.maximum(im - weight_threshold, 0)

        # estimate compositions
        im_sum = np.sum(im_all, axis = 0)
        sub = im_sum > 0.0
        for a0 in range(self.num_crystals):
            im_all[a0][sub] /= im_sum[sub]

            for a1 in range(3):
                im_rgb_all[a0,:,:,a1] = im_all[a0] * phase_colors[a0,a1]

        if plot_combine:
            if crystal_inds_plot is None:
                im_rgb = np.sum(im_rgb_all, axis = 0)
            else:
                im_rgb = np.sum(im_rgb_all[np.array(crystal_inds_plot)], axis = 0)

            im_rgb = np.clip(im_rgb,0,1)

            fig,ax = plt.subplots(1,1,figsize=figsize)
            ax.imshow(
                im_rgb,
            )
            ax.set_title(
                'Phase Maps',
                fontsize = 16,
            )
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            if not show_axes:
                ax.set_axis_off()

        else:
            # plotting
            if layout == 0:
                fig,ax = plt.subplots(
                    1,
                    self.num_crystals,
                    figsize=(figsize[0],(self.num_fits+1)*figsize[1]))
            elif layout == 1:
                fig,ax = plt.subplots(
                    self.num_crystals,
                    1,
                    figsize=(figsize[0],(self.num_fits+1)*figsize[1]))
                
            for a0 in range(self.num_crystals):
                
                ax[a0].imshow(
                    im_rgb_all[a0],
                )
                ax[a0].set_title(
                    self.names[a0],
                    fontsize = 16,
                )
                if not show_ticks:
                    ax[a0].set_xticks([])
                    ax[a0].set_yticks([])
                if not show_axes:
                    ax[a0].set_axis_off()

        # All possible returns
        if return_phase_estimate:
            if returnfig:
                return im_all, fig, ax
            else:
                return im_all
        elif return_rgb_images:
            if plot_combine:
                if returnfig:
                    return im_rgb, fig, ax
                else:
                    return im_rgb
            else:
                if returnfig:
                    return im_rgb_all, fig, ax
                else:
                    return im_rgb_all
        else:
            if returnfig:
                return fig, ax


        
<<<<<<< HEAD
=======
    #         fig, (ax1, ax2) = plt.subplots(2,1,figsize = figsize)
    #         ax1 = plot_diffraction_pattern(pointlist,)
    #     return
    
>>>>>>> dev
