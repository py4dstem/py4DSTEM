import numpy as np
import matplotlib.pyplot as plt

from py4DSTEM.utils.tqdmnd import tqdmnd

from numpy.linalg import lstsq, nnls

class Crystal_Phase:
    """
    A class storing multiple crystal structures, and associated diffraction data.
    Must be initialized after matching orientations to a pointlistarray.

    """
    def __init__(
        self,
        crystals
    ):
        """
        Args:
            crystals (list): list of crystal objects
        """
        self.crystals = crystals
        
    def plot_phase_map():
        # To do - crafty way to visualize phases
        return
    
    def quantify_phase(pointlistarray):
        # To do - method to quantify phase from PLA-
        # implement nonnegative least squares
        return