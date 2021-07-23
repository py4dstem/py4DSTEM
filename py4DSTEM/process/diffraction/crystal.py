# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd




class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    Args:
        positions ((n,3) numpy array):      fractional atomic coordinates 
        numbers ((n,) numpy array or scalar): atomic numbers 
        cell (3 or 6 element numpy array):    [a, b, c, alpha, beta, gamma], where angles (alpha,beta,gamma) are in degrees.
                                              If only [a, b, c] are provided, we assume 90 degree cell edges.
                                              If only [a] is provided, we assume b = c = a.
    """

    def __init__(self, positions, numbers, cell, **kwargs):
        """
        Instantiate a Crystal object. 
        Calculate lattice vectors.
        """
        
        # Initialize Crystal
        self.positions = positions   #: fractional atomic coordinates

        #: atomic numbers - if only one value is provided, assume all atoms are same species
        numbers = np.asarray(numbers, dtype='intp')
        if np.size(numbers) == 1:
            self.numbers = np.ones(positions.shape[0], dtype='intp') * numbers
        elif np.size(numbers) == positions.shape[0]:
            self.numbers = numbers
        else:
            raise Exception('Number of positions and atomic numbers do not match')


        # unit cell, as either [a a a 90 90 90], [a b c 90 90 90], or [a b c alpha beta gamma] 
        cell = np.asarray(cell, dtype='float_')      
        if np.size(cell) == 1:
            self.cell = np.hstack([cell, cell, cell, 90, 90, 90])
        elif np.size(cell) == 3:
            self.cell = np.hstack([cell, 90, 90, 90])
        elif np.size(cell) == 6:  
            self.cell = cell   			 
        else:
            raise Exception('Cell cannot contain ' + np.size(cell) + ' elements')

        # calculate unit cell lattice vectors
        a = self.cell[0]
        b = self.cell[1]
        c = self.cell[2]
        alpha = self.cell[3] * np.pi/180
        beta  = self.cell[4] * np.pi/180
        gamma = self.cell[5] * np.pi/180
        t = (np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        self.lat_real = np.array([ \
            [a, 0, 0],
            [b*np.cos(gamma), b*np.sin(gamma), 0],
            [c*np.cos(beta), c*t, c*np.sqrt(1-np.cos(beta)**2-t**2)]])



    def calculate_structure_factors(self, k_max):
        # Inverse lattice vectors
        lat_inv = np.linalg.inv(self.lat_real)

        # Find shortest lattice vector direction
        k_test = np.vstack([
            lat_inv[0,:],
            lat_inv[1,:],
            lat_inv[2,:],
            lat_inv[0,:] + lat_inv[1,:],
            lat_inv[0,:] + lat_inv[2,:],
            lat_inv[1,:] + lat_inv[2,:],
            lat_inv[0,:] + lat_inv[1,:] + lat_inv[2,:],
            lat_inv[0,:] - lat_inv[1,:] + lat_inv[2,:],
            lat_inv[0,:] + lat_inv[1,:] - lat_inv[2,:],
            lat_inv[0,:] - lat_inv[1,:] - lat_inv[2,:],
            ])
        k_leng_min = np.min(np.linalg.norm(k_test, axis=1))

        # Tile lattice vectors
        num_tile = np.ceil(k_max / k_leng_min)
        ya,xa,za = np.meshgrid(
            np.arange(-num_tile, num_tile+1),
            np.arange(-num_tile, num_tile+1),
            np.arange(-num_tile, num_tile+1))
        hkl = np.vstack([xa.ravel(), ya.ravel(), za.ravel()])
        k_vec_all = np.matmul(lat_inv, hkl) 

        # Delete lattice vectors outside of k_max
        keep = np.linalg.norm(k_vec_all, axis=0) <= k_max
        self.hkl = hkl[:,keep]
        self.k_vec_all = k_vec_all[:,keep]
        self.k_vec_leng = np.linalg.norm(self.k_vec_all, axis=0)

        # Calculate single atom scattering factors
        f_all = np.zeros((np.size(self.k_vec_leng, 0), self.positions.shape[0]), dtype='complex64')
        for a0 in range(self.positions.shape[0]):
            f = single_atom_scatter()


        # Calculate structure factors
        self.struct_factors = np.zeros(np.size(self.k_vec_leng, 0), dtype='complex64')
        for a0 in range(self.struct_factors.shape[0]):
            print(a0)


        # print(self.k_vec_leng.shape)



