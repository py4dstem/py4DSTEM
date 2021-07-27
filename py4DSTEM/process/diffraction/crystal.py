# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# from scipy.interpolate import griddata
from scipy.special import sph_harm

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
        SHT_degree_max=10,
        SHT_shell_interp_dist=0.20,
        tol_distance=1e-3):
        """
        Calculate the (nested) spherical harmonic for a set of 3D structure factors
        
        Args:
            SHT_degree_max (numpy int):  degree of the spherical harmonic transform
            SHT_shell_interp_dist (numpy float):  distance value for interpolation on shells
            tol_distance (numpy float): tolerance for point distance tests (1/Angstroms)
        """

        self.SHT_degree_max = int(SHT_degree_max)
        self.SHT_shell_interp_dist = np.array(SHT_shell_interp_dist)

        # Determine the spherical shell radiis required
        radii = np.unique(np.round(
            self.g_vec_leng / tol_distance) * tol_distance)
        # Remove zero beam
        self.SHT_shell_radii = np.delete(radii,0)

        # Get sampling points on spherical surface - note we use t-design of 2 * max degree
        self.SHT_azim, self.SHT_elev, self.SHT_verts = tdesign(2*self.SHT_degree_max)
        # Degree and order of all SHT terms
        v = np.arange(-self.SHT_degree_max,self.SHT_degree_max+1)
        # m, n = np.meshgrid(
        #     np.arange(-self.SHT_degree_max, self.SHT_degree_max+1),
        #     np.arange(0, self.SHT_degree_max+1))
        m, n = np.meshgrid(
            np.arange(0, self.SHT_degree_max+1),
            np.arange(0, self.SHT_degree_max+1))
        keep = np.abs(m) <= n
        self.SHT_degree_order = np.vstack((
            n[keep], m[keep])).T
        # num_terms = (self.SHT_degree_max + 1)**2
        num_terms = int(self.SHT_degree_max*(self.SHT_degree_max + 1)/2)

        # compute Gaussian denominator prefactor for Gaussian KDE on shells
        # pre = -1/(2*SHT_shell_sigma**2)

        # initialize arrays
        self.SHT_shell_values = np.zeros((
            self.SHT_verts.shape[0],
            np.size(self.SHT_shell_radii)))
        self.SHT_basis = np.zeros((
            num_terms,
            self.SHT_verts.shape[0]),
            dtype='complex64')
        self.SHT_values = np.zeros((
            num_terms,
            np.size(self.SHT_shell_radii)),
            dtype='complex64')

        # Calculate spherical harmonic basis of all orders and degrees
        for a0 in range(num_terms):
            self.SHT_basis[a0,:] = sph_harm( \
                self.SHT_degree_order[a0,1],
                self.SHT_degree_order[a0,0],
                self.SHT_azim,
                self.SHT_elev)

        # Compute spherical interpolations for all SF peaks, and SHTs
        for a0 in range(np.size(self.SHT_shell_radii)):
            sub = np.abs(self.g_vec_leng - self.SHT_shell_radii[a0]) < tol_distance
            g = self.g_vec_all[:,sub]
            g = g / np.linalg.norm(g, axis=0)
            intensity = self.struct_factors_int[sub]
            
            # interpolate intenties on this shell
            for a1 in range(g.shape[1]):
                self.SHT_shell_values[:,a0] += intensity[a1] * \
                    np.maximum(self.SHT_shell_interp_dist - \
                    np.sqrt(np.sum((self.SHT_verts - g[:,a1])**2, axis=1)), 0)

            # Calculate SHT for this shell.
            # Note we take the complex conjugate to use as a reference SHT.
            self.SHT_values[:,a0] = np.conjugate(np.matmul(
                self.SHT_basis, self.SHT_shell_values[:,a0]))
        
        # for testing
        plt.figure(figsize=(16,4))
        plt.imshow(np.abs(self.SHT_values).T,cmap='gray')
        plt.show()

    def spherical_harmonic_correlation_plan(
        self, 
        zone_axis_range=np.array([[1,0,1],[1,1,1]]),
        angle_step_zone_axis=3.0,
        angle_step_in_plane=6.0):
        """
        Calculate the spherical harmonic basis arrays for an SO(3) rotation correlogram.
        
        Args:
            zone_axis_range (3x3 numpy float):  Row vectors give the range for zone axis orientations.
                                                Note that we always start at [0,0,1] to make z-x-z rotation work.
            angle_step_zone_axis (numpy float):  approximate angular step size for zone axis [degrees]
            angle_step_zone_axis (numpy float):  approximate angular step size for in-plane rotation [degrees]
        """

        self.SHT_zone_axis_range = np.vstack((np.array([0,0,1]),np.array(zone_axis_range)))

        # Solve for number of angular steps in zone axis (rads)
        z_angle_1 = np.arccos(np.sum(self.SHT_zone_axis_range[0,:]*self.SHT_zone_axis_range[1,:]) / \
            np.linalg.norm(self.SHT_zone_axis_range[0,:])/np.linalg.norm(self.SHT_zone_axis_range[1,:]))
        z_angle_2 = np.arccos(np.sum(self.SHT_zone_axis_range[0,:]*self.SHT_zone_axis_range[2,:]) / \
            np.linalg.norm(self.SHT_zone_axis_range[0,:])/np.linalg.norm(self.SHT_zone_axis_range[2,:]))
        self.SHT_zone_axis_steps = np.round(np.maximum( 
            (180/np.pi) * z_angle_1 / angle_step_zone_axis,
            (180/np.pi) * z_angle_2 / angle_step_zone_axis)).astype(np.int)

        # Solve for number of angular steps along in-plane rotation direction
        self.SHT_in_plane_steps = np.round(360/angle_step_in_plane).astype(np.int)

        # Calculate z-x angular range (rads)
        x_angle_1 = np.arctan2(self.SHT_zone_axis_range[1,1],self.SHT_zone_axis_range[1,0])
        x_angle_2 = np.arctan2(self.SHT_zone_axis_range[2,1],self.SHT_zone_axis_range[2,0])

        # Calculate z-x angles
        self.SHT_num_zones = ((self.SHT_zone_axis_steps+1)*(self.SHT_zone_axis_steps+2)/2).astype(np.int)
        elev = np.zeros(self.SHT_num_zones)
        azim = np.zeros(self.SHT_num_zones)
        for a0 in np.arange(1,self.SHT_zone_axis_steps+1):
            inds = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)
            w_elev = a0 / self.SHT_zone_axis_steps
            w_azim = np.linspace(0,1,a0+1)
            elev[inds] =  w_elev*((1-w_azim)*z_angle_1 + w_azim*z_angle_2)
            azim[inds] = (1-w_azim)*x_angle_1 + w_azim*x_angle_2

        # Calculate -z angles
        # gamma = 

        # Calculate rotation matrices
        self.SHT_corr_rotation_matrices = np.zeros((3,3,self.SHT_num_zones,self.SHT_in_plane_steps))


        x = np.cos(azim)*np.sin(elev)
        y = np.sin(azim)*np.sin(elev)
        z = np.cos(elev)

        # 3D plotting
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(
            projection='3d',
            elev=54.7, 
            azim=45)

        ax.scatter(
            xs=x, 
            ys=y, 
            zs=z,
            s=30)

        # axes limits
        ax.axes.set_xlim3d(left=-0.05, right=1) 
        ax.axes.set_ylim3d(bottom=-0.05, top=1) 
        ax.axes.set_zlim3d(bottom=-0.05, top=1) 
        ax.set_box_aspect((1,1,1))


        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot()

        # ax.scatter(
        #     dx_angle*180/np.pi, 
        #     dz_angle*180/np.pi, 
        #     s=100)
        # ax.invert_yaxis()

        # print(v)
        # print(z_angle_1 * 180/np.pi)
        # print(z_angle_2 * 180/np.pi)



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
