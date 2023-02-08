import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.visualize import show, show_image_grid
from py4DSTEM.io.datastructure.emd.pointlistarray import PointListArray
from numpy.linalg import lstsq
from scipy.optimize import nnls
from py4DSTEM.process.diffraction.crystal_viz import plot_diffraction_pattern

class CrystalPhase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray???

    """
    def __init__(
        self,
        crystals,
        name,
    ):
        """
        Args:
            crystals (list):            List of crystal instances
            name (str):                 Name of Crystal_Phase instance
        """
        # validate inputs
        assert(isinstance(crystals,list)), '`crystals` must be a list of crystal instances'
        for xtal in crystals:
            assert(hasattr(xtal,'orientation_map')), '`crystals` elements must be Crystal instances with a .orientation_map - try running .match_orientations'

        # assign variables
        self.num_crystals = len(crystals)
        self.crystals = crystals
        self.orientation_maps = [xtal.orientation_map for xtal in crystals]


    def plot_all_phase_maps(
        self,
        map_scale_values = None,
        index = 0
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
        show_image_grid(lambda i:phase_maps[i], 1, len(phase_maps), cmap = 'inferno')
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
    
    def quantify_phase(
        self,
        pointlistarray,
        tolerance_distance = 0.08,
        method = 'nnls',
        intensity_power = 0,
        mask_peaks = None
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

            phase_weights = np.zeros((
                pointlistarray.shape[0],
                pointlistarray.shape[1],
                np.sum([map.num_matches for map in self.orientation_maps])
                ))
            phase_residuals = np.zeros(pointlistarray.shape)
            for Rx, Ry in tqdmnd(pointlistarray.shape[0], pointlistarray.shape[1]):
                _, phase_weight, phase_residual, crystal_identity = self.quantify_phase_pointlist(
                    pointlistarray,
                    position = [Rx, Ry],
                    tolerance_distance=tolerance_distance,
                    method = method,
                    intensity_power = intensity_power,
                    mask_peaks = mask_peaks
                )
                phase_weights[Rx,Ry,:] = phase_weight
                phase_residuals[Rx,Ry] = phase_residual
            self.phase_weights = phase_weights
            self.phase_residuals = phase_residuals
            self.crystal_identity = crystal_identity
            return
        else:
            return TypeError('pointlistarray must be of type pointlistarray.')
        return
    
    def compare_intensitylists(
        self,
        masterpointlist,
        masterintensitylist,
        bragg_peaks_fit,
        tolerance_distance,
        intensity_power
    ):
        """
        Function to compare the exisiting point list enteries with the array. 
        """
        # Add a column of zeros in the master intensity list to make way for the new fitted intensity list
        zeros = np.zeros((masterintensitylist.shape[0], 1))
        masterintensitylist = np.concatenate((masterintensitylist,zeros),axis=1)

        # Compare with the exisiting bragg_peaks_fit with the masterpointlist.
        # Make a temporary intensity list to store the intensities of the the bragg_peaks_fit. 
        
        if intensity_power == 0:
            temporary_pl_intensities = np.ones(bragg_peaks_fit['intensity'].shape)
        else:
            temporary_pl_intensities = bragg_peaks_fit['intensity']**intensity_power
        
    
        # Go through the bragg_peaks_fit to find if the master list has an entry or not. 
        for d in range(bragg_peaks_fit['qx'].shape[0]):
            distances = []
            # Making a numpy array of the fitted bragg peak
            bragg_peak_point=np.array([bragg_peaks_fit['qx'][d],bragg_peaks_fit['qy'][d]])
            for p in range(masterpointlist.shape[0]):
                        distances.append(np.linalg.norm(bragg_peak_point-masterpointlist[p])
                        )
            ind = np.where(distances == np.min(distances))[0][0]
            # Potentially loop over to find the best tolerance distance.
            if distances[ind] <= tolerance_distance:
                columns_masterintensitylist = len(masterintensitylist[0])
                masterintensitylist[ind][columns_masterintensitylist-1]=temporary_pl_intensities[d]

            else:
                ## The point list is not in the mega list of point list so the point list last row of masterpointlist
                masterpointlist = np.vstack((masterpointlist,bragg_peak_point))
                ## Add a row to the intensity list such that all the remaining intensity lists should be 0 but only the new bragg intensity list is non zero but intensity power
                new_intensity_list_row = np.zeros((1, masterintensitylist.shape[1]-1))
                new_intensity_list_row = np.append(new_intensity_list_row, [temporary_pl_intensities[d]])
                new_intensity_list_row = new_intensity_list_row.reshape((1,-1))
                masterintensitylist = np.concatenate((masterintensitylist,new_intensity_list_row),axis=0)



        
        return masterpointlist,masterintensitylist
        



    def quantify_phase_pointlist(
        self,
        pointlistarray,
        position,
        method = 'nnls', 
        tolerance_distance = 0.08,
        intensity_power = 0,
        mask_peaks = None
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
        #     3. Make a flag variable for the experimental dataset which turns 1 if it is encountered in the simulated dataset.
        
        pointlist = pointlistarray.get_pointlist(position[0], position[1]) 
        ## Remove the central beam
        pl_mask = np.where((pointlist['qx'] == 0) & (pointlist['qy'] == 0), 1, 0)
        pointlist.remove(pl_mask)
        # False Negatives (exp peak with no match in crystal instances) will appear here, already coded in
    
        if intensity_power == 0:
            pl_intensities = np.ones(pointlist['intensity'].shape)
        else:
            pl_intensities = pointlist['intensity']**intensity_power
        
        #Prepare matches for modeling
        pointlist_peak_matches = []
        crystal_identity = []
        ## Initialize the megapointlist and master intensity list with the experimental intensity
        masterpointlist = np.column_stack((pointlist['qx'],pointlist['qy']))
        masterintensitylist = pl_intensities
        ## Convert masterintensitylist to a 2D array
        masterintensitylist = np.array(masterintensitylist, ndmin=2).T
        ## Loop over the number of crystals.
        for c in range(len(self.crystals)):
            ## Loop over the number of num matches which is the number of orientation candidates. 
            # This value of num matches was supplied when the orientation map was created. 
            for m in range(self.orientation_maps[c].num_matches):
                # Set crystal identity
                crystal_identity.append([c,m])
                # For a given crystal class generate a diffraction pattern given a orientation crystal and given num match
                bragg_peaks_fit = self.crystals[c].generate_diffraction_pattern(
                    self.orientation_maps[c].get_orientation(position[0], position[1]),
                    ind_orientation = m
                )
                # Check if there are any experimental intensity observed at all.
                if len(masterpointlist !=0):
                # Send this bragg_peaks_fit to the compare function to be compared with mega point list and master intensity list. 
                    masterpointlist,masterintensitylist=self.compare_intensitylists(masterpointlist,masterintensitylist,bragg_peaks_fit,tolerance_distance,intensity_power)
                else:
                    continue


        ### The intensity and point lists are accumulated in the masterintensitylist and masterpointlist.
        # The first column of the intensity lists are the observed experimental intensities.
        observed_intensities = masterintensitylist[:,0]
        expected_intensities = masterintensitylist[:,1:]

        if len(observed_intensities) > 0:   
            if mask_peaks is not None:
                for i in range(len(mask_peaks)):
                    if mask_peaks[i] == None:
                        continue
                    inds_mask = np.where(expected_intensities[:,mask_peaks[i]] != 0)[0]
                    for mask in range(len(inds_mask)):
                        expected_intensities[inds_mask[mask],i] = 0
            if method == 'nnls':    
                phase_weights, phase_residuals = nnls(
                    expected_intensities,
                    observed_intensities
                )
            
            elif method == 'lstsq':
                phase_weights, phase_residuals, rank, singluar_vals = lstsq(
                    expected_intensities,
                    observed_intensities,
                    rcond = -1
                )
                phase_residuals = np.sum(phase_residuals)
            else:
                raise ValueError(method + ' Not yet implemented. Try nnls or lstsq.')   
        else:
            # Find the number of expected phases
            number_expected_phases=0
            for c in range(len(self.crystals)):
                for m in range(self.orientation_maps[c].num_matches):
                    number_expected_phases+=1

            # If there are no diffraction patterns 
            phase_weights = np.zeros(number_expected_phases)
            phase_residuals = np.NaN
        
        return expected_intensities, phase_weights, phase_residuals, crystal_identity

    
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
    
