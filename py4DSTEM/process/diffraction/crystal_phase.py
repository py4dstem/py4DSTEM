import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.visualize import show, show_image_grid
from py4DSTEM.io.datastructure.emd.pointlistarray import PointListArray
from numpy.linalg import lstsq
from scipy.optimize import nnls

class Crystal_Phase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray???

    """
    def __init__(
        self,
        crystals,
        name,
        orientation_maps = None,
    ):
        """
        Args:
            crystals (list):            List of crystal instances
            name (str):                 Name of Crystal_Phase instance
            orientation_maps (list):    (Optional) List of orientation maps
        """
        if isinstance(crystals, list):
            self.crystals = crystals
            self.num_crystals = len(crystals)
        else:
            raise TypeError('crystals must be a list of crystal instances.')
       
        if orientation_maps is not None:
            if isinstance(orientation_maps, list):
                if len(self.crystals) != len(orientation_maps):
                    raise ValueError('Orientation maps must have the same number of entries as crystals.')
                self.orientation_maps = orientation_maps
            else:
                    raise TypeError('orientation_maps must be a list of orientation maps.')    
        self.name = name
        return
        
    def add_orientation_maps(
        self,
        orientation_maps
    ):
        """
        _summary_

        Args:
            orientation_maps (list): _description_
        """
        if isinstance(orientation_maps, list):
            if len(self.crystals) != len(orientation_maps):
                raise ValueError('Orientation maps must have the same number of entries as crystals.')
            self.orientation_maps = orientation_maps
        else:
            raise TypeError('orientation_maps must be a list of orientation maps.')
        
        return
    
    def show_all_phase_maps(
        self,
        method,
        map_scale_values = None,
        index = 0
    ):
        """
        Visualize phase maps of dataset.

        Args:
            method (_type_, optional): _description_. Defaults to None.
            map_scale_values (_type_, optional): _description_. Defaults to None.
        """
        
        if method == 'orientation_maps':
            phase_maps = []
            if map_scale_values == None:
                map_scale_values = [1] * len(self.orientation_maps)
            corr_sum = np.sum([(self.orientation_maps[m].corr[:,:,index] * map_scale_values[m]) for m in range(len(self.orientation_maps))])
            for m in range(len(self.orientation_maps)):
                phase_maps.append(self.orientation_maps[m].corr[:,:,index] / corr_sum)
            show_image_grid(lambda i:phase_maps[i], 1, len(phase_maps), cmap = 'inferno')
        else:
            print('Method not yet implemented.')
        return
    
    def show_phase_map(
        self,
        method,
        index = 0,
        
    ):
        if method == 'orientation_maps':
            corr_array = np.dstack([maps.corr[:,:,index] for maps in self.orientation_maps])
            best_corr_score = np.max(corr_array,axis=2)
            best_match_phase = [np.where(corr_array[:,:,p] == best_corr_score, True,False)
                                for p in range(len(self.orientation_maps))
                                ]

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
                        self.orientation_maps[m].corr, 
                        mask = best_match_phase[m]), 
                    cmap = cm)
            plt.show()
        else:
            print('Method not yet implemented.')
        return
    
    def quantify_phase_map(
        self,
        pointlistarray,
        method = 'nnls',
    ):
        """
        Quantification of the phase of a crystal based on the crystal instances and the pointlistarray.

        Args:
            pointlistarray (PointListArray):    Pointlistarray
            method (str)                        Computational method to compute phase with
        
        Details:
        """
        if isinstance(pointlistarray, PointListArray):
            for Rx, Ry in tqdmnd(pointlistarray.shape[0], pointlistarray.shape[1]):
                pointlist = pointlistarray.get_pointlist(Rx, Ry)
                #results = quantify_phase_pointlist(pointlist, )
                continue
            return
        else:
            return TypeError('pointlistarray must be of type pointlistarray.')
        return
    
    def quantify_phase_pointlist(
        self,
        pointlist,
        method = 'nnls'
    ):
        results = pointlist
        return results