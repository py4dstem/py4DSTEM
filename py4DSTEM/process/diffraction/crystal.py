# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .tdesign import tdesign

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd
from ..utils import single_atom_scatter
from ..utils import electron_wavelength_angstrom




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
        self.positions = np.asarray(positions)   #: fractional atomic coordinates

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


    def calculate_structure_factors(self, k_max=2, tol_structure_factor=1e-2):
        """
        Calculate structure factors for all hkl indices up to max scattering vector k_max
        
        Args:
            k_max (numpy float):                max scattering vector to include (1/Angstroms)
            tol_structure_factor (numpy float): tolerance for removing low-valued structure factors
        """

        # Store k_max
        self.k_max = np.asarray(k_max)

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
        num_tile = np.ceil(self.k_max / k_leng_min)
        ya,xa,za = np.meshgrid(
            np.arange(-num_tile, num_tile+1),
            np.arange(-num_tile, num_tile+1),
            np.arange(-num_tile, num_tile+1))
        hkl = np.vstack([xa.ravel(), ya.ravel(), za.ravel()])
        g_vec_all = np.matmul(lat_inv, hkl) 

        # Delete lattice vectors outside of k_max
        keep = np.linalg.norm(g_vec_all, axis=0) <= self.k_max
        self.hkl = hkl[:,keep]
        self.g_vec_all = g_vec_all[:,keep]
        self.g_vec_leng = np.linalg.norm(self.g_vec_all, axis=0)

        # Calculate single atom scattering factors
        # Note this can be sped up a ton, but we may want to generalize to allow non-1.0 occupancy in the future.
        f_all = np.zeros((np.size(self.g_vec_leng, 0), self.positions.shape[0]), dtype='float_')
        for a0 in range(self.positions.shape[0]):
            atom_sf = single_atom_scatter(
                [self.numbers[a0]],
                [1],
                self.g_vec_leng,
                'A')
            atom_sf.get_scattering_factor(
                [self.numbers[a0]],
                [1],
                self.g_vec_leng,
                'A')
            f_all[:,a0] = atom_sf.fe

        # Calculate structure factors
        self.struct_factors = np.zeros(np.size(self.g_vec_leng, 0), dtype='complex64')
        for a0 in range(self.positions.shape[0]):
            self.struct_factors += f_all[:,a0] * \
                np.exp((2j * np.pi) * \
                np.sum(self.hkl * np.expand_dims(self.positions[a0,:],axis=1),axis=0))

        # Remove structure factors below tolerance level
        keep = np.abs(self.struct_factors) > tol_structure_factor
        self.hkl = self.hkl[:,keep]
        self.g_vec_all = self.g_vec_all[:,keep]
        self.g_vec_leng = self.g_vec_leng[keep]
        self.struct_factors = self.struct_factors[keep]

        # Structure factor intensities
        self.struct_factors_int = np.abs(self.struct_factors)**2 


    def plot_structure_factors(
        self,
        proj_dir=[10,30],
        scale_markers=1,
        figsize=(8,8),
        returnfig=False):
        """
        3D scatter plot of the structure factors using magnitude^2, i.e. intensity.

        Args:
            dir_proj (2 or 3 element numpy array):    projection direction, either [azim elev] or normal vector
            scale_markers (float):  size scaling for markers
            figsize (2 element float):  size scaling of figure axes
            returnfig (bool):   set to True to return figure and axes handles
        """

        if np.size(proj_dir) == 2:
            el = proj_dir[0]
            az = proj_dir[1]
        elif np.size(proj_dir) == 3:
            if proj_dir[0] == 0 and proj_dir[1] == 0:
                el = 90 * np.sign(proj_dir[2])
            else:
                el = np.arctan(proj_dir[2]/np.sqrt(proj_dir[0]**2 + proj_dir[1]**2)) * 180/np.pi
            az = np.arctan2(proj_dir[1],proj_dir[0]) * 180/np.pi
        else:
            raise Exception('Projection direction cannot contain ' + np.size(proj_dir) + ' elements')


        # 3D plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(
            projection='3d',
            elev=el, 
            azim=az)
        # ax.set_box_aspect((np.ptp(
        #     self.g_vec_all[0,:]), 
        #     self.g_vec_all[1,:], 
        #     self.g_vec_all[2,:]))

        ax.scatter(
            xs=self.g_vec_all[0,:], 
            ys=self.g_vec_all[1,:], 
            zs=self.g_vec_all[2,:],
            s=scale_markers*self.struct_factors_int)

        # axes limits
        r = self.k_max * 1.05
        ax.axes.set_xlim3d(left=-r, right=r) 
        ax.axes.set_ylim3d(bottom=-r, top=r) 
        ax.axes.set_zlim3d(bottom=-r, top=r) 
        ax.set_box_aspect((1,1,1))

        plt.show()

        if returnfig:
            return fig, ax


    def spherical_harmonic_transform(
        self, 
        SHT_order=8,
        tol_distance=1e-3):
        """
        Calculate the (nested) spherical harmonic for a set of 3D structure factors
        
        Args:
            SHT_order (numpy float):  order of the spherical harmonic transform
            tol_distance (numpy float): tolerance for point distance tests (1/Angstroms)
        ect
        """

        # Determine the spherical shell radiis required
        radii = np.unique(np.round(
            self.g_vec_leng / tol_distance) * tol_distance)
        # Remove zero beam
        self.SHT_shell_radii = np.delete(radii,0)

        # Get sampling points on spherical surface
        _,_,self.SHT_verts = tdesign(SHT_order)

        # Compute spherical interpolations for all SF peaks
        self.SHT_shell_values = np.zeros((
            np.size(self.SHT_verts),
            np.size(self.SHT_shell_radii)))
        for a0 in range(np.size(self.SHT_shell_radii)):
            sub = np.abs(self.g_vec_leng - self.SHT_shell_radii[a0]) < tol_distance
            k = self.k_vec_all[sub,:]

        print(k)




        # print(self.SHT_shell_radii)


        #return 1






    def generate_diffraction_pattern(
        self, 
        accel_voltage = 300e3, 
        zone_axis = [0,0,1],
        proj_x_axis = [1,0,0],
        sigma_excitation_error = 0.02,
        tol_excitation_error_mult = 3,
        ):
        """
        Generate a single diffraction pattern, return all peaks as a pointlist.

        Args:
            accel_voltage (numpy float):        kinetic energy of electrons specificed in volts
        """

        accel_voltage = np.asarray(accel_voltage)
        zone_axis = np.asarray(zone_axis)

        # Calculate wavelenth
        wavelength = electron_wavelength_angstrom(accel_voltage)

        # wavevectors
        K0 = zone_axis / np.linalg.norm(zone_axis) / wavelength;
        # K0_plus_g = self.g_vec_all + K0[:,None]
        # cos_alpha = np.sum(K0[:,None] * self.g_vec_all, axis=0) \
        #     / np.linalg.norm(self.g_vec_all, axis=0) \
        #     / np.linalg.norm(K0)
        cos_alpha = np.sum((K0[:,None] + self.g_vec_all) * zone_axis[:,None], axis=0) \
            / np.linalg.norm(K0[:,None] + self.g_vec_all) \
            / np.linalg.norm(zone_axis)


        # Excitation errors
        # sg = (-0.5 / wavelength) \
        #     * np.sum((self.g_vec_all - 2*K0[:,None]) * self.g_vec_all, axis=0) \
        #     / np.sum((self.g_vec_all + K0[:,None]) * K0[:,None], axis=0)
        # sg = np.sum((self.g_vec_all - 2*K0[:,None]) * self.g_vec_all, axis=0) \
        #     / np.linalg.norm(K0)**2
        sg = (-0.5) * np.sum((2*K0[:,None] + self.g_vec_all) * self.g_vec_all, axis=0) \
            / (np.linalg.norm(K0[:,None] + self.g_vec_all)) / cos_alpha

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = (sg <= sg_max)
        g_diff = self.g_vec_all[:,keep]
        
        # Diffracted peak intensities
        g_int = self.struct_factors_int[keep] \
            * np.exp(sg[keep]**2/(-2*sigma_excitation_error**2))

        # Diffracted peak locations
        ky_proj = np.cross(zone_axis, proj_x_axis)
        kx_proj = np.cross(ky_proj, zone_axis)
        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)
        gx_proj = np.sum(g_diff * kx_proj[:,None], axis=0)
        gy_proj = np.sum(g_diff * ky_proj[:,None], axis=0)

        # Output as PointList
        bragg_peaks = PointList([('qx','float64'),('qy','float64'),('intensity','float64')])
        bragg_peaks.add_pointarray(np.vstack((gx_proj, gy_proj, g_int)).T)

        return bragg_peaks




    def new_function(self, args):
        """
        Description
        
        Args:
            k_max (numpy float):                max scattering vector to include
        """
        ect






def plot_diffraction_pattern(
    bragg_peaks,
    scale_markers=20,
    power_markers=1,
    figsize=(8,8),
    returnfig=False):
    """
    2D scatter plot of the Bragg peaks

    Args:
        scale_markers (float):  size scaling for markers
        power_markers (float):  power law scaling for marks (default is 1, i.e. amplitude)
        figsize (2 element float):  size scaling of figure axes
        returnfig (bool):   set to True to return figure and axes handles
    """

    # 2D plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    # ax = fig.add_subplot(
    #     projection='3d',
    #     elev=el, 
    #     azim=az)

    if power_markers == 2:
        marker_size = scale_markers*bragg_peaks.data['intensity']
    else:
        marker_size = scale_markers*(bragg_peaks.data['intensity']**(power_markers/2))

    ax.scatter(
        bragg_peaks.data['qy'], 
        bragg_peaks.data['qx'], 
        s=marker_size)

    ax.invert_yaxis()
    ax.set_box_aspect(1)

    # # axes limits
    # r = self.k_max * 1.05
    # ax.axes.set_xlim3d(left=-r, right=r) 
    # ax.axes.set_ylim3d(bottom=-r, top=r) 
    # ax.axes.set_zlim3d(bottom=-r, top=r) 

    plt.show()

    if returnfig:
        return fig, ax
