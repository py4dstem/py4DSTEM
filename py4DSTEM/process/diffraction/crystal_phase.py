import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import nnls
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
        crystal_names = None,
        orientation_maps = None,
        name = None,
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

        # List of orientation maps
        if orientation_maps is None:
            self.orientation_maps = [crystals[ind].orientation_map for ind in range(self.num_crystals)]
        else:
            if len(self.crystals) != len(orientation_maps):
                raise ValueError(
                    "Orientation maps must have the same number of entries as crystals."
                )
            self.orientation_maps = orientation_maps

        # Names of all crystal phases
        if crystal_names is None:
            self.crystal_names = ['crystal' + str(ind) for ind in range(self.num_crystals)]
        else:
            self.crystal_names = crystal_names

        # Name of the phase map
        if name is None:
            self.name = 'phase map'
        else:
            self.name = name

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



    def quantify_single_pattern(
        self,
        pointlistarray: PointListArray,
        xy_position = (0,0),
        corr_kernel_size = 0.04,
        sigma_excitation_error: float = 0.02,
        precession_angle_degrees = None,
        power_intensity: float = 0.25, 
        power_intensity_experiment: float = 0.25,  
        k_max = None,
        max_number_patterns = 2,
        single_phase = False,
        allow_strain = False,
        strain_iterations = 3,
        strain_max = 0.02,
        include_false_positives = True,
        weight_false_positives = 1.0,
        weight_unmatched_peaks = 1.0,
        plot_result = True,
        plot_only_nonzero_phases = True,
        plot_unmatched_peaks = False,
        plot_correlation_radius = False,
        scale_markers_experiment = 40,
        scale_markers_calculated = 200,
        crystal_inds_plot = None,
        phase_colors = None,
        figsize = (10,7),
        verbose = True,
        returnfig = False,
        ):
        """
        Quantify the phase for a single diffraction pattern.

        TODO - determine the difference between false positive peaks and unmatched peaks (if any).

        Parameters
        ----------
        
        pointlistarray: (PointListArray)
            Full array of all calibrated experimental bragg peaks, with shape = (num_x,num_y)
        xy_position: (int,int)
            The (x,y) or (row,column) position to be quantified.
        corr_kernel_size: (float)     
            Correlation kernel size length. The size of the overlap kernel between the 
            measured Bragg peaks and diffraction library Bragg peaks. [1/Angstroms]
        sigma_excitation_error: (float)
            The out of plane excitation error tolerance. [1/Angstroms]
        precession_angle_degrees: (float)
            Tilt angle of illuminaiton cone in degrees for precession electron diffraction (PED).
        power_intensity: (float)       
            Power for scaling the correlation intensity as a function of simulated peak intensity.
        power_intensity_experiment: (float):      
            Power for scaling the correlation intensity as a function of experimental peak intensity.
        k_max: (float)
            Max k values included in fits, for both x and y directions.
        max_number_patterns: int
            Max number of orientations which can be included in a match.
        single_phase: bool
            Set to true to force result to output only the best-fit phase (minimum intensity residual).
        allow_strain: bool,
            Allow the simulated diffraction patterns to be distorted to improve the matches.
        strain_iterations: int
            Number of pattern position refinement iterations.
        strain_max: float
            Maximum strain fraction allowed - this value should be low, typically a few percent (~0.02).
        include_false_positives: bool
            Penalize patterns which generate false positive peaks.
        weight_false_positives: float
            Weight strength of false positive peaks.
        weight_unmatched_peaks: float
            Penalize unmatched peaks.
        plot_result: bool
            Plot the resulting fit.
        plot_only_nonzero_phases: bool
            Only plot phases with phase weights > 0.
        plot_unmatched_peaks: bool
            Plot the false postive peaks.
        plot_correlation_radius: bool
            In the visualization, draw the correlation radius.
        scale_markers_experiment: float
            Size of experimental diffraction peak markers.
        scale_markers_calculated: float
            Size of the calculate diffraction peak markers.
        crystal_inds_plot: tuple of ints
            Which crystal index / indices to plot.
        phase_colors: np.array
            Color of each phase, should have shape = (num_phases, 3)
        figsize: (float,float)
            Size of the output figure.
        verbose: bool
            Print the resulting fit weights to console.
        returnfig: bool
            Return the figure and axis handles for the plot.


        Returns
        -------
        phase_weights: (np.array)
            Estimated relative fraction of each phase for all probe positions. 
            shape = (num_x, num_y, num_orientations)
            where num_orientations is the total number of all orientations for all phases.
        phase_residual: (np.array)
            Residual intensity not represented by the best fit phase weighting for all probe positions.
            shape = (num_x, num_y)
        phase_reliability: (np.array)
            Estimated reliability of match(es) for all probe positions.
            Typically calculated as the best fit score minus the second best fit.
            shape = (num_x, num_y)
        int_total: (np.array)
            Sum of experimental peak intensities for all probe positions.
            shape = (num_x, num_y)
        fig,ax: (optional)
            matplotlib figure and axis handles

        """

        # tolerance
        tol2 = 1e-6

        # calibrations
        center  = pointlistarray.calstate['center']
        ellipse = pointlistarray.calstate['ellipse']
        pixel   = pointlistarray.calstate['pixel']
        rotate  = pointlistarray.calstate['rotate']
        if center is False:
            raise ValueError('Bragg peaks must be center calibration')
        if pixel is False:
            raise ValueError('Bragg peaks must have pixel size calibration')
        # TODO - potentially warn the user if ellipse / rotate calibration not available

        if phase_colors is None:
            phase_colors = np.array((
                (1.0,0.0,0.0,1.0),
                (0.0,0.8,1.0,1.0),
                (0.0,0.6,0.0,1.0),
                (1.0,0.0,1.0,1.0),
                (0.0,0.2,1.0,1.0),
                (1.0,0.8,0.0,1.0),
            ))

        # Experimental values
        bragg_peaks = pointlistarray.get_vectors(
            xy_position[0],
            xy_position[1],
            center = center,
            ellipse = ellipse,
            pixel = pixel,
            rotate = rotate)
        # bragg_peaks = pointlistarray.get_pointlist(xy_position[0],xy_position[1]).copy()
        if k_max is None:
            keep = bragg_peaks.data["qx"]**2 + bragg_peaks.data["qy"]**2 > tol2
        else:
            keep = np.logical_and.reduce((
            bragg_peaks.data["qx"]**2 + bragg_peaks.data["qy"]**2 > tol2,
            np.abs(bragg_peaks.data["qx"]) < k_max,
            np.abs(bragg_peaks.data["qy"]) < k_max,
        ))

            
        # ind_center_beam = np.argmin(
        #     bragg_peaks.data["qx"]**2 + bragg_peaks.data["qy"]**2)
        # mask = np.ones_like(bragg_peaks.data["qx"], dtype='bool')
        # mask[ind_center_beam] = False
        # bragg_peaks.remove(ind_center_beam)
        qx = bragg_peaks.data["qx"][keep]
        qy = bragg_peaks.data["qy"][keep]
        qx0 = bragg_peaks.data["qx"][np.logical_not(keep)]
        qy0 = bragg_peaks.data["qy"][np.logical_not(keep)]
        if power_intensity_experiment == 0:
            intensity = np.ones_like(qx)
            intensity0 = np.ones_like(qx0)
        else:
            intensity = bragg_peaks.data["intensity"][keep]**power_intensity_experiment
            intensity0 = bragg_peaks.data["intensity"][np.logical_not(keep)]**power_intensity_experiment
        int_total = np.sum(intensity) 

        # init basis array
        if include_false_positives:
            basis = np.zeros((intensity.shape[0], self.num_fits))
            unpaired_peaks = []
        else:
            basis = np.zeros((intensity.shape[0], self.num_fits))
        if allow_strain:
            m_strains = np.zeros((self.num_fits,2,2))
            m_strains[:,0,0] = 1.0
            m_strains[:,1,1] = 1.0

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
                precession_angle_degrees = precession_angle_degrees,
            )
            if k_max is None:
                del_peak = bragg_peaks_fit.data["qx"]**2 \
                    +      bragg_peaks_fit.data["qy"]**2 < tol2
            else:
                del_peak = np.logical_or.reduce((
                    bragg_peaks_fit.data["qx"]**2 \
                    +      bragg_peaks_fit.data["qy"]**2 < tol2,
                    np.abs(bragg_peaks_fit.data["qx"]) > k_max,
                    np.abs(bragg_peaks_fit.data["qy"]) > k_max,
                ))
            bragg_peaks_fit.remove(del_peak)

            # peak intensities
            if power_intensity == 0:
                int_fit = np.ones_like(bragg_peaks_fit.data["qx"])
            else:
                int_fit = bragg_peaks_fit.data['intensity']**power_intensity
            
            # Pair peaks to experiment
            if plot_result:
                matches = np.zeros((bragg_peaks_fit.data.shape[0]),dtype='bool')

            if allow_strain:
                for a1 in range(strain_iterations):
                    # Initial peak pairing to find best-fit strain distortion
                    pair_sub = np.zeros(bragg_peaks_fit.data.shape[0],dtype='bool')
                    pair_inds = np.zeros(bragg_peaks_fit.data.shape[0],dtype='int')
                    for a1 in range(bragg_peaks_fit.data.shape[0]):
                        dist2 = (bragg_peaks_fit.data['qx'][a1] - qx)**2 \
                            +   (bragg_peaks_fit.data['qy'][a1] - qy)**2
                        ind_min = np.argmin(dist2)
                        val_min = dist2[ind_min]

                        if val_min < radius_max_2:
                            pair_sub[a1] = True
                            pair_inds[a1] = ind_min

                    # calculate best-fit strain tensor, weighted by the intensities.
                    # requires at least 4 peak pairs
                    if np.sum(pair_sub) >= 4:
                        # pair_obs = bragg_peaks_fit.data[['qx','qy']][pair_sub]
                        pair_basis = np.vstack((
                            bragg_peaks_fit.data['qx'][pair_sub],
                            bragg_peaks_fit.data['qy'][pair_sub],
                        )).T
                        pair_obs = np.vstack((
                            qx[pair_inds[pair_sub]],
                            qy[pair_inds[pair_sub]],
                        )).T

                        # weights
                        dists = np.sqrt(
                            (bragg_peaks_fit.data['qx'][pair_sub] - qx[pair_inds[pair_sub]])**2 + \
                            (bragg_peaks_fit.data['qx'][pair_sub] - qx[pair_inds[pair_sub]])**2)
                        weights = np.sqrt(
                            int_fit[pair_sub] * intensity[pair_inds[pair_sub]]
                        ) * (1 - dists / corr_kernel_size)
                        # weights = 1 - dists / corr_kernel_size

                        # strain tensor
                        m_strain = np.linalg.lstsq(
                            pair_basis * weights[:,None],
                            pair_obs * weights[:,None],
                            rcond = None,
                        )[0]

                        # Clamp strains to be within the user-specified limit
                        m_strain = np.clip(
                            m_strain,
                            np.eye(2) - strain_max,
                            np.eye(2) + strain_max,
                        )
                        m_strains[a0] *= m_strain

                        # Transformed peak positions
                        qx_copy = bragg_peaks_fit.data['qx']
                        qy_copy = bragg_peaks_fit.data['qy']
                        bragg_peaks_fit.data['qx'] = qx_copy*m_strain[0,0] + qy_copy*m_strain[1,0]
                        bragg_peaks_fit.data['qy'] = qx_copy*m_strain[0,1] + qy_copy*m_strain[1,1]                        

            # Loop over all peaks, pair experiment to library
            for a1 in range(bragg_peaks_fit.data.shape[0]):
                dist2 = (bragg_peaks_fit.data['qx'][a1] - qx)**2 \
                    +   (bragg_peaks_fit.data['qy'][a1] - qy)**2
                ind_min = np.argmin(dist2)
                val_min = dist2[ind_min]

                if include_false_positives:
                    weight = np.clip(1 - np.sqrt(dist2[ind_min]) / corr_kernel_size,0,1)
                    basis[ind_min,a0] = int_fit[a1] * weight
                    unpaired_peaks.append([
                        a0,
                        int_fit[a1] * (1 - weight),
                    ])
                    if weight > 1e-8 and plot_result:
                        matches[a1] = True
                else:
                    if val_min < radius_max_2:
                        basis[ind_min,a0] = int_fit[a1]
                        if plot_result:
                            matches[a1] = True


                # if val_min < radius_max_2:
                #     # weight = 1 - np.sqrt(dist2[ind_min]) / corr_kernel_size
                #     # weight = 1 + corr_distance_scale * \
                #     #     np.sqrt(dist2[ind_min]) / corr_kernel_size
                #     # basis[ind_min,a0] = weight * int_fit[a1]
                #     basis[ind_min,a0] = int_fit[a1]
                #     if plot_result:
                #         matches[a1] = True
                # elif include_false_positives:
                #     # unpaired_peaks.append([a0,int_fit[a1]*(1 + corr_distance_scale)])
                #     unpaired_peaks.append([a0,int_fit[a1]])

            if plot_result:
                library_peaks.append(bragg_peaks_fit)                
                library_int.append(int_fit)
                library_matches.append(matches)

        # If needed, augment basis and observations with false positives
        if include_false_positives:
            basis_aug = np.zeros((len(unpaired_peaks),self.num_fits))
            for a0 in range(len(unpaired_peaks)):
                basis_aug[a0,unpaired_peaks[a0][0]] = unpaired_peaks[a0][1]

            basis = np.vstack((basis, basis_aug * weight_false_positives))
            obs = np.hstack((intensity, np.zeros(len(unpaired_peaks))))

        else:
            obs = intensity
        
        # Solve for phase weight coefficients
        try:
            phase_weights = np.zeros(self.num_fits)

            if single_phase:
                # loop through each crystal structure and determine the best fit structure,
                # which can contain multiple orientations up to max_number_patterns
                crystal_res = np.zeros(self.num_crystals)

                for a0 in range(self.num_crystals):
                    inds_solve = self.crystal_identity[:,0] == a0
                    search = True

                    while search is True:

                        basis_solve = basis[:,inds_solve]
                        obs_solve = obs.copy()

                        if weight_unmatched_peaks > 1.0:
                            sub_unmatched = np.sum(basis_solve,axis=1)<1e-8
                            obs_solve[sub_unmatched] *= weight_unmatched_peaks

                        phase_weights_cand, phase_residual_cand = nnls(
                            basis_solve,
                            obs_solve,
                        )

                        if np.count_nonzero(phase_weights_cand > 0.0) <= max_number_patterns:
                            phase_weights[inds_solve] = phase_weights_cand
                            crystal_res[a0] = phase_residual_cand
                            search = False
                        else:
                            inds = np.where(inds_solve)[0]
                            inds_solve[inds[np.argmin(phase_weights_cand)]] = False

                ind_best_fit = np.argmin(crystal_res)
                # ind_best_fit = np.argmax(phase_weights)

                phase_residual = crystal_res[ind_best_fit]
                sub = np.logical_not(
                    self.crystal_identity[:,0] == ind_best_fit
                )
                phase_weights[sub] = 0.0

                # Estimate reliability as difference between best fit and 2nd best fit
                crystal_res = np.sort(crystal_res)
                phase_reliability = crystal_res[1] - crystal_res[0]

            else:
                # Allow all crystals and orientation matches in the pattern
                inds_solve = np.ones(self.num_fits,dtype='bool')
                search = True
                while search is True:
                    phase_weights_cand, phase_residual_cand = nnls(
                        basis[:,inds_solve],
                        obs,
                    )

                    if np.count_nonzero(phase_weights_cand > 0.0) <= max_number_patterns:
                        phase_weights[inds_solve] = phase_weights_cand
                        phase_residual = phase_residual_cand
                        search = False
                    else:
                        inds = np.where(inds_solve)[0]
                        inds_solve[inds[np.argmin(phase_weights_cand)]] = False


                # Estimate the phase reliability
                inds_solve = np.ones(self.num_fits,dtype='bool')
                inds_solve[phase_weights > 1e-8] = False

                if np.all(inds_solve == False):
                    phase_reliability = 0.0
                else:
                    search = True
                    while search is True:
                        phase_weights_cand, phase_residual_cand = nnls(
                            basis[:,inds_solve],
                            obs,
                        )
                        if np.count_nonzero(phase_weights_cand > 0.0) <= max_number_patterns:
                            phase_residual_2nd = phase_residual_cand
                            search = False
                        else:
                            inds = np.where(inds_solve)[0]
                            inds_solve[inds[np.argmin(phase_weights_cand)]] = False

                    phase_weights_cand, phase_residual_cand = nnls(
                        basis[:,inds_solve],
                        obs,
                    )
                    phase_reliability = phase_residual_2nd - phase_residual

        except:
            phase_weights = np.zeros(self.num_fits)
            phase_residual = np.sqrt(np.sum(intensity**2))
            phase_reliability = 0.0


        if verbose:
            ind_max = np.argmax(phase_weights)
            # print()
            print('\033[1m' + 'phase_weight   or_ind   name' + '\033[0m')
            # print()
            for a0 in range(self.num_fits):
                c = self.crystal_identity[a0,0]
                m = self.crystal_identity[a0,1]
                line = '{:>12} {:>8}   {:<12}'.format(
                    f'{phase_weights[a0]:.2f}',
                    m,
                    self.crystal_names[c]
                    )
                if a0 == ind_max:
                    print('\033[1m' + line + '\033[0m')
                else:
                    print(line)
            print('----------------------------')
            line = '{:>12} {:>15}'.format(
                f'{sum(phase_weights):.2f}',
                'fit total'
                )
            print('\033[1m' + line + '\033[0m')
            line = '{:>12} {:>15}'.format(
                f'{phase_residual:.2f}',
                'fit residual'
                )
            print(line)

        # Plotting
        if plot_result:
            # fig, ax = plt.subplots(figsize=figsize)
            fig = plt.figure(figsize=figsize)
            # if plot_layout == 0:
            #     ax_x = fig.add_axes(
            #         [0.0+figbound[0], 0.0, 0.4-2*+figbound[0], 1.0])
            ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
            ax_leg = fig.add_axes([0.68, 0.0, 0.3, 1.0])

            if plot_correlation_radius:
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
                # s = scale_markers_experiment * intensity0,
                s = scale_markers_experiment * bragg_peaks.data["intensity"][np.logical_not(keep)],
                marker = "o",
                facecolor = [0.7, 0.7, 0.7],
                )
            ax.scatter(
                qy,
                qx,
                # s = scale_markers_experiment * intensity,
                s = scale_markers_experiment * bragg_peaks.data["intensity"][keep],
                marker = "o",
                facecolor = [0.7, 0.7, 0.7],
                )
            # legend
            if k_max is None:
                k_max = np.max(self.k_max)
            dx_leg =  -0.05*k_max
            dy_leg =   0.04*k_max
            text_params = {
                "va": "center",
                "ha": "left",
                "family": "sans-serif",
                "fontweight": "normal",
                "color": "k",
                "size": 12,
            }
            if plot_correlation_radius:
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
                facecolor = [0.7, 0.7, 0.7],
                )
            ax_leg.text(
                dy_leg,
                0,
                'Experimental peaks',
                **text_params)
            if plot_correlation_radius:
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

            count_leg = 0
            for a0 in range(self.num_fits):
                c = self.crystal_identity[a0,0]
                m = self.crystal_identity[a0,1]

                if crystal_inds_plot == None or np.min(np.abs(c - crystal_inds_plot)) == 0:

                    qx_fit = library_peaks[a0].data['qx']
                    qy_fit = library_peaks[a0].data['qy']

                    if allow_strain:
                        m_strain = m_strains[a0]
                        # Transformed peak positions
                        qx_copy = qx_fit.copy()
                        qy_copy = qy_fit.copy()
                        qx_fit = qx_copy*m_strain[0,0] + qy_copy*m_strain[1,0]
                        qy_fit = qx_copy*m_strain[0,1] + qy_copy*m_strain[1,1]

                    int_fit = library_int[a0]
                    matches_fit = library_matches[a0]

                    if plot_only_nonzero_phases is False or phase_weights[a0] > 0:

                        # if np.mod(m,2) == 0:
                        ax.scatter(
                            qy_fit[matches_fit],
                            qx_fit[matches_fit],
                            s = scale_markers_calculated * int_fit[matches_fit],
                            marker = mvals[c],
                            facecolor = phase_colors[c,:],
                            )
                        if plot_unmatched_peaks:
                            ax.scatter(
                                qy_fit[np.logical_not(matches_fit)],
                                qx_fit[np.logical_not(matches_fit)],
                                s = scale_markers_calculated * int_fit[np.logical_not(matches_fit)],
                                marker = mvals[c],
                                facecolor = phase_colors[c,:],
                                )

                    # legend
                    if m == 0:
                        ax_leg.text(
                            dy_leg,
                            (count_leg+1)*dx_leg,
                            self.crystal_names[c],
                            **text_params)
                        ax_leg.scatter(
                            0,
                            (count_leg+1) * dx_leg,
                            s = 200,
                            marker = mvals[c],
                            facecolor = phase_colors[c,:],
                            )
                        count_leg += 1
                        # else:
                        #     ax.scatter(
                        #         qy_fit[matches_fit],
                        #         qx_fit[matches_fit],
                        #         s = scale_markers_calculated * int_fit[matches_fit],
                        #         marker = mvals[c],
                        #         edgecolors = uvals[c,:],
                        #         facecolors = (uvals[c,0],uvals[c,1],uvals[c,2],0.3),
                        #         # facecolors = (1,1,1,0.5),
                        #         linewidth = 2,
                        #         )
                        #     if plot_unmatched_peaks:
                        #         ax.scatter(
                        #             qy_fit[np.logical_not(matches_fit)],
                        #             qx_fit[np.logical_not(matches_fit)],
                        #             s = scale_markers_calculated * int_fit[np.logical_not(matches_fit)],
                        #             marker = mvals[c],
                        #             edgecolors = uvals[c,:],
                        #             facecolors = (1,1,1,0.5),
                        #             linewidth = 2,
                        #             )

                        #     # legend
                        #     ax_leg.scatter(
                        #         0,
                        #         dx_leg*(a0+1),
                        #         s = 200,
                        #         marker = mvals[c],
                        #         edgecolors = uvals[c,:],
                        #         facecolors = (uvals[c,0],uvals[c,1],uvals[c,2],0.3),
                        #         # facecolors = (1,1,1,0.5),
                        #         )



            # appearance
            ax.set_xlim((-k_max, k_max))
            ax.set_ylim((k_max, -k_max))

            ax_leg.set_xlim((-0.1*k_max, 0.4*k_max))
            ax_leg.set_ylim((-0.5*k_max, 0.5*k_max))
            ax_leg.set_axis_off()

        if returnfig:
            return phase_weights, phase_residual, phase_reliability, int_total, fig, ax
        else:
            return phase_weights, phase_residual, phase_reliability, int_total

    def quantify_phase(
        self,
        pointlistarray: PointListArray,
        corr_kernel_size = 0.04,
        sigma_excitation_error = 0.02,

        precession_angle_degrees = None,
        power_intensity: float = 0.25, 
        power_intensity_experiment: float = 0.25,  
        k_max = None,
        # power_experiment = 0.25,
        # power_calculated = 0.25,
        max_number_patterns = 2,
        single_phase = False,
        allow_strain = True,
        strain_iterations = 3,
        strain_max = 0.02,
        include_false_positives = True,
        weight_false_positives = 1.0,
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
        self.phase_reliability = np.zeros((
            pointlistarray.shape[0],
            pointlistarray.shape[1],
        ))
        self.int_total = np.zeros((
            pointlistarray.shape[0],
            pointlistarray.shape[1],
        ))
        self.single_phase = single_phase
        
        for rx, ry in tqdmnd(
                *pointlistarray.shape,
                desc="Quantifying Phase",
                unit=" PointList",
                disable=not progress_bar,
            ):
            # calculate phase weights
            phase_weights, phase_residual, phase_reliability, int_peaks = self.quantify_single_pattern(
                pointlistarray = pointlistarray,
                xy_position = (rx,ry),
                corr_kernel_size = corr_kernel_size,
                sigma_excitation_error = sigma_excitation_error,

                precession_angle_degrees = precession_angle_degrees,
                power_intensity = power_intensity, 
                power_intensity_experiment = power_intensity_experiment,  
                k_max = k_max,

                max_number_patterns = max_number_patterns,
                single_phase = single_phase,
                allow_strain = allow_strain,
                strain_iterations = strain_iterations,
                strain_max = strain_max,
                include_false_positives = include_false_positives,
                weight_false_positives = weight_false_positives,
                plot_result = False,
                verbose = False,
                returnfig = False,
                )
            self.phase_weights[rx,ry] = phase_weights
            self.phase_residuals[rx,ry] = phase_residual
            self.phase_reliability[rx,ry] = phase_reliability
            self.int_total[rx,ry] = int_peaks


    def plot_phase_weights(
        self,
        weight_range = (0.0,1.0),
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
                self.crystal_names[a0],
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
        phase_colors = None,
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

        if phase_colors is None:
            phase_colors = np.array((
                (1.0,0.0,0.0),
                (0.0,0.8,1.0),
                (0.0,0.8,0.0),
                (1.0,0.0,1.0),
                (0.0,0.4,1.0),
                (1.0,0.8,0.0),
            ))

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
                    self.crystal_names[a0],
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


    def plot_dominant_phase(
        self,
        use_correlation_scores = False,
        reliability_range = (0.0,1.0),
        sigma = 0.0,
        phase_colors = None,
        ticks = True,
        figsize = (6,6),
        legend_add = True,
        legend_fraction = 0.2,
        print_fractions = False,
        ):
        """
        Plot a combined figure showing the primary phase at each probe position.
        Mask by the reliability index (best match minus 2nd best match).
        """
        
        if phase_colors is None:
            phase_colors = np.array([
                [1.0,0.9,0.6],        
                [1,0,0],        
                [0,0.7,0],        
                [0,0.7,1],        
                [1,0,1],        
            ])
        

        # init arrays
        scan_shape = self.phase_weights.shape[:2]
        phase_map = np.zeros(scan_shape)
        phase_corr = np.zeros(scan_shape)
        phase_corr_2nd = np.zeros(scan_shape)
        phase_sig = np.zeros((self.num_crystals,scan_shape[0],scan_shape[1]))

        if use_correlation_scores:
            # Calculate scores from highest correlation match
            for a0 in range(self.num_crystals):
                phase_sig[a0] = np.maximum(
                    phase_sig[a0],
                    np.max(self.crystals[a0].orientation_map.corr,axis=2),
                )
        else:
            # sum up phase weights by crystal type
            for a0 in range(self.num_fits):
                ind = self.crystal_identity[a0,0]
                phase_sig[ind] += self.phase_weights[:,:,a0]

        # smoothing of the outputs
        if sigma > 0.0:
            for a0 in range(self.num_crystals):
                phase_sig[a0] = gaussian_filter(
                    phase_sig[a0],
                    sigma = sigma,
                    mode = 'nearest',
                    )

        # find highest correlation score for each crystal and match index
        for a0 in range(self.num_crystals):
            sub = phase_sig[a0] > phase_corr 
            phase_map[sub] = a0
            phase_corr[sub] = phase_sig[a0][sub]
        
        if self.single_phase:
            phase_scale = np.clip(
                (self.phase_reliability - reliability_range[0]) / (reliability_range[1] - reliability_range[0]),
                0,
                1)

        else:

            # find the second correlation score for each crystal and match index
            for a0 in range(self.num_crystals):
                corr = phase_sig[a0].copy()
                corr[phase_map==a0] = 0.0
                sub = corr > phase_corr_2nd
                phase_corr_2nd[sub] = corr[sub]
                
            # Estimate the reliability
            phase_rel = phase_corr - phase_corr_2nd
            phase_scale = np.clip(
                (phase_rel - reliability_range[0]) / (reliability_range[1] - reliability_range[0]),
                0,
                1)

        # Print the total area of fraction of each phase
        if print_fractions:
            phase_mask = phase_scale >= 0.5
            phase_total = np.sum(phase_mask)

            print('Phase Fractions')
            print('---------------')
            for a0 in range(self.num_crystals):
                phase_frac = np.sum((phase_map == a0) * phase_mask) / phase_total

                print(self.crystal_names[a0] + ' - ' + f'{phase_frac*100:.4f}' + '%')


        self.phase_rgb = np.zeros((scan_shape[0],scan_shape[1],3))
        for a0 in range(self.num_crystals):
            sub = phase_map==a0
            for a1 in range(3):
                self.phase_rgb[:,:,a1][sub] = phase_colors[a0,a1] * phase_scale[sub]
        # normalize
        # self.phase_rgb = np.clip(
        #     (self.phase_rgb - rel_range[0]) / (rel_range[1] - rel_range[0]),
        #     0,1)
        
            
        
        fig = plt.figure(figsize=figsize)
        if legend_add:
            width = 1

            ax = fig.add_axes((0,legend_fraction,1,1-legend_fraction))
            ax_leg = fig.add_axes((0,0,1,legend_fraction))

            for a0 in range(self.num_crystals):
                ax_leg.scatter(
                    a0*width,
                    0,
                    s = 200,
                    marker = 's',
                    edgecolor = (0,0,0,1),
                    facecolor = phase_colors[a0],
                )
                ax_leg.text(
                    a0*width+0.1,
                    0,
                    self.crystal_names[a0],
                    fontsize = 16,
                    verticalalignment = 'center',
                )
            ax_leg.axis('off')
            ax_leg.set_xlim((
                width * -0.5,
                width * (self.num_crystals+0.5),
            ))

        else:
            ax = fig.add_axes((0,0,1,1))

        ax.imshow(
            self.phase_rgb,
            # vmin = 0,
            # vmax = 5,
            # self.phase_rgb,
            # phase_corr - phase_corr_2nd,
            # cmap = 'turbo',
            # vmin = 0,
            # vmax = 3,
            # cmap = 'gray',
        )

        if ticks is False:
            ax.set_xticks([])
            ax.set_yticks([])



        return fig,ax


    # def plot_all_phase_maps(self, map_scale_values=None, index=0):
    #     """
    #     Visualize phase maps of dataset.

    #     Args:
    #         map_scale_values (float):   Value to scale correlations by
    #     """
    #     phase_maps = []
    #     if map_scale_values is None:
    #         map_scale_values = [1] * len(self.orientation_maps)
    #     corr_sum = np.sum(
    #         [
    #             (self.orientation_maps[m].corr[:, :, index] * map_scale_values[m])
    #             for m in range(len(self.orientation_maps))
    #         ]
    #     )
    #     for m in range(len(self.orientation_maps)):
    #         phase_maps.append(self.orientation_maps[m].corr[:, :, index] / corr_sum)
    #     show_image_grid(lambda i: phase_maps[i], 1, len(phase_maps), cmap="inferno")
    #     return

    # def plot_phase_map(self, index=0, cmap=None):
    #     corr_array = np.dstack(
    #         [maps.corr[:, :, index] for maps in self.orientation_maps]
    #     )
    #     best_corr_score = np.max(corr_array, axis=2)
    #     best_match_phase = [
    #         np.where(corr_array[:, :, p] == best_corr_score, True, False)
    #         for p in range(len(self.orientation_maps))
    #     ]

    #     if cmap is None:
    #         cm = plt.get_cmap("rainbow")
    #         cmap = [
    #             cm(1.0 * i / len(self.orientation_maps))
    #             for i in range(len(self.orientation_maps))
    #         ]

    #     fig, (ax) = plt.subplots(figsize=(6, 6))
    #     ax.matshow(
    #         np.zeros((self.orientation_maps[0].num_x, self.orientation_maps[0].num_y)),
    #         cmap="gray",
    #     )
    #     ax.axis("off")

    #     for m in range(len(self.orientation_maps)):
    #         c0, c1 = (cmap[m][0] * 0.35, cmap[m][1] * 0.35, cmap[m][2] * 0.35, 1), cmap[
    #             m
    #         ]
    #         cm = mpl.colors.LinearSegmentedColormap.from_list("cmap", [c0, c1], N=10)
    #         ax.matshow(
    #             np.ma.array(
    #                 self.orientation_maps[m].corr[:, :, index], mask=best_match_phase[m]
    #             ),
    #             cmap=cm,
    #         )
    #     plt.show()

    #     return

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
    #     tolerance_distance=0.08,
    #     method="nnls",
    #     intensity_power=0,
    #     mask_peaks=None,
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
    #         phase_weights = np.zeros(
    #             (
    #                 pointlistarray.shape[0],
    #                 pointlistarray.shape[1],
    #                 np.sum([map.num_matches for map in self.orientation_maps]),
    #             )
    #         )
    #         phase_residuals = np.zeros(pointlistarray.shape)
    #         for Rx, Ry in tqdmnd(pointlistarray.shape[0], pointlistarray.shape[1]):
    #             (
    #                 _,
    #                 phase_weight,
    #                 phase_residual,
    #                 crystal_identity,
    #             ) = self.quantify_phase_pointlist(
    #                 pointlistarray,
    #                 position=[Rx, Ry],
    #                 tolerance_distance=tolerance_distance,
    #                 method=method,
    #                 intensity_power=intensity_power,
    #                 mask_peaks=mask_peaks,
    #             )
    #             phase_weights[Rx, Ry, :] = phase_weight
    #             phase_residuals[Rx, Ry] = phase_residual
    #         self.phase_weights = phase_weights
    #         self.phase_residuals = phase_residuals
    #         self.crystal_identity = crystal_identity
    #         return
    #     else:
    #         return TypeError("pointlistarray must be of type pointlistarray.")
    #     return

    # def quantify_phase_pointlist(
    #     self,
    #     pointlistarray,
    #     position,
    #     method="nnls",
    #     tolerance_distance=0.08,
    #     intensity_power=0,
    #     mask_peaks=None,
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
    #     pl_mask = np.where((pointlist["qx"] == 0) & (pointlist["qy"] == 0), 1, 0)
    #     pointlist.remove(pl_mask)
    #     # False Negatives (exp peak with no match in crystal instances) will appear here, already coded in

    #     if intensity_power == 0:
    #         pl_intensities = np.ones(pointlist["intensity"].shape)
    #     else:
    #         pl_intensities = pointlist["intensity"] ** intensity_power
    #     # Prepare matches for modeling
    #     pointlist_peak_matches = []
    #     crystal_identity = []

    #     for c in range(len(self.crystals)):
    #         for m in range(self.orientation_maps[c].num_matches):
    #             crystal_identity.append([c, m])
    #             phase_peak_match_intensities = np.zeros((pointlist["intensity"].shape))
    #             bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
    #                 self.orientation_maps[c].get_orientation(position[0], position[1]),
    #                 ind_orientation=m,
    #             )
    #             # Find the best match peak within tolerance_distance and add value in the right position
    #             for d in range(pointlist["qx"].shape[0]):
    #                 distances = []
    #                 for p in range(bragg_peaks_fit["qx"].shape[0]):
    #                     distances.append(
    #                         np.sqrt(
    #                             (pointlist["qx"][d] - bragg_peaks_fit["qx"][p]) ** 2
    #                             + (pointlist["qy"][d] - bragg_peaks_fit["qy"][p]) ** 2
    #                         )
    #                     )
    #                 ind = np.where(distances == np.min(distances))[0][0]

    #                 # Potentially for-loop over multiple values for 'tolerance_distance' to find best tolerance_distance value
    #                 if distances[ind] <= tolerance_distance:
    #                     ## Somewhere in this if statement is probably where better distances from the peak should be coded in
    #                     if (
    #                         intensity_power == 0
    #                     ):  # This could potentially be a different intensity_power arg
    #                         phase_peak_match_intensities[d] = 1 ** (
    #                             (tolerance_distance - distances[ind])
    #                             / tolerance_distance
    #                         )
    #                     else:
    #                         phase_peak_match_intensities[d] = bragg_peaks_fit[
    #                             "intensity"
    #                         ][ind] ** (
    #                             (tolerance_distance - distances[ind])
    #                             / tolerance_distance
    #                         )
    #                 else:
    #                     ## This is probably where the false positives (peaks in crystal but not in experiment) should be handled
    #                     continue

    #             pointlist_peak_matches.append(phase_peak_match_intensities)
    #             pointlist_peak_intensity_matches = np.dstack(pointlist_peak_matches)
    #             pointlist_peak_intensity_matches = (
    #                 pointlist_peak_intensity_matches.reshape(
    #                     pl_intensities.shape[0],
    #                     pointlist_peak_intensity_matches.shape[-1],
    #                 )
    #             )

    #     if len(pointlist["qx"]) > 0:
    #         if mask_peaks is not None:
    #             for i in range(len(mask_peaks)):
    #                 if mask_peaks[i] == None:  # noqa: E711
    #                     continue
    #                 inds_mask = np.where(
    #                     pointlist_peak_intensity_matches[:, mask_peaks[i]] != 0
    #                 )[0]
    #                 for mask in range(len(inds_mask)):
    #                     pointlist_peak_intensity_matches[inds_mask[mask], i] = 0

    #         if method == "nnls":
    #             phase_weights, phase_residuals = nnls(
    #                 pointlist_peak_intensity_matches, pl_intensities
    #             )

    #         elif method == "lstsq":
    #             phase_weights, phase_residuals, rank, singluar_vals = lstsq(
    #                 pointlist_peak_intensity_matches, pl_intensities, rcond=-1
    #             )
    #             phase_residuals = np.sum(phase_residuals)
    #         else:
    #             raise ValueError(method + " Not yet implemented. Try nnls or lstsq.")
    #     else:
    #         phase_weights = np.zeros((pointlist_peak_intensity_matches.shape[1],))
    #         phase_residuals = np.NaN
    #     return (
    #         pointlist_peak_intensity_matches,
    #         phase_weights,
    #         phase_residuals,
    #         crystal_identity,
    #     )

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
