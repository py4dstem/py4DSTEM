# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, single_atom_scatter, electron_wavelength_angstrom


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

    def __init__(
        self, 
        positions, 
        numbers, 
        cell, 
        **kwargs):
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


    def calculate_structure_factors(
        self, 
        k_max=2, 
        tol_structure_factor=1e-2):
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
        g_vec_all = lat_inv @ hkl

        # Delete lattice vectors outside of k_max
        keep = np.linalg.norm(g_vec_all, axis=0) <= self.k_max
        self.hkl = hkl[:,keep]
        self.g_vec_all = g_vec_all[:,keep]
        self.g_vec_leng = np.linalg.norm(self.g_vec_all, axis=0)

        # Calculate single atom scattering factors
        # Note this can be sped up a lot, but we may want to generalize to allow non-1.0 occupancy in the future.
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


    def orientation_plan(
        self, 
        zone_axis_range=np.array([[0,1,1],[1,1,1]]),
        angle_step_zone_axis=3.0,
        angle_step_in_plane=6.0,
        tol_distance = 0.01,
        ):
        """
        Calculate the rotation basis arrays for an SO(3) rotation correlogram.
        
        Args:
            zone_axis_range (3x3 numpy float):  Row vectors give the range for zone axis orientations.
                                                Note that we always start at [0,0,1] to make z-x-z rotation work.
            angle_step_zone_axis (numpy float): Approximate angular step size for zone axis [degrees]
            angle_step_in_plane (numpy float):  Approximate angular step size for in-plane rotation [degrees]
            tol_distance (numpy float):         Distance tolerance for radial shell assignment [1/Angstroms]
        """

        # Define 3 vectors which span zone axis orientation range, normalize
        self.orientation_zone_axis_range = np.vstack((np.array([0,0,1]),np.array(zone_axis_range))).astype('float')
        self.orientation_zone_axis_range[1,:] /= np.linalg.norm(self.orientation_zone_axis_range[1,:])
        self.orientation_zone_axis_range[2,:] /= np.linalg.norm(self.orientation_zone_axis_range[2,:])

        # Solve for number of angular steps in zone axis (rads)
        angle_u_v = np.arccos(np.sum(self.orientation_zone_axis_range[0,:] * self.orientation_zone_axis_range[1,:]))
        angle_u_w = np.arccos(np.sum(self.orientation_zone_axis_range[0,:] * self.orientation_zone_axis_range[2,:]))
        self.orientation_zone_axis_steps = np.round(np.maximum( 
            (180/np.pi) * angle_u_v / angle_step_zone_axis,
            (180/np.pi) * angle_u_w / angle_step_zone_axis)).astype(np.int)

        # Calculate points along u and v using the SLERP formula
        # https://en.wikipedia.org/wiki/Slerp
        weights = np.linspace(0,1,self.orientation_zone_axis_steps+1)
        pv = self.orientation_zone_axis_range[0,:] * np.sin((1-weights[:,None])*angle_u_v)/np.sin(angle_u_v) + \
             self.orientation_zone_axis_range[1,:] * np.sin(   weights[:,None] *angle_u_v)/np.sin(angle_u_v) 

        # Calculate points along u and w using the SLERP formula
        pw = self.orientation_zone_axis_range[0,:] * np.sin((1-weights[:,None])*angle_u_w)/np.sin(angle_u_w) + \
             self.orientation_zone_axis_range[2,:] * np.sin(   weights[:,None] *angle_u_w)/np.sin(angle_u_w) 

        # Init array to hold all points
        self.orientation_num_zones = ((self.orientation_zone_axis_steps+1)*(self.orientation_zone_axis_steps+2)/2).astype(np.int)
        vecs = np.zeros((self.orientation_num_zones,3))
        vecs[0,:] = self.orientation_zone_axis_range[0,:]

        # Calculate zone axis points on the unit sphere with another application of SLERP
        for a0 in np.arange(1,self.orientation_zone_axis_steps+1):
            inds = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)

            p0 = pv[a0,:]
            p1 = pw[a0,:]
            angle_p = np.arccos(np.sum(p0 * p1))

            weights = np.linspace(0,1,a0+1)
            vecs[inds,:] = \
                p0[None,:] * np.sin((1-weights[:,None])*angle_p)/np.sin(angle_p) + \
                p1[None,:] * np.sin(   weights[:,None] *angle_p)/np.sin(angle_p) 

        # Convert to spherical coordinates
        azim = np.arctan2(vecs[:,1],vecs[:,0])
        elev = np.arctan2(np.hypot(vecs[:,1], vecs[:,0]), vecs[:,2])

        # Solve for number of angular steps along in-plane rotation direction
        self.orientation_in_plane_steps = np.round(360/angle_step_in_plane).astype(np.int)

        # Calculate -z angles (Euler angle 3)
        gamma = np.linspace(0,2*np.pi,self.orientation_in_plane_steps, endpoint=False)

        # init storage arrays
        self.orientation_rotation_angles = np.zeros((
            self.orientation_num_zones,
            self.orientation_in_plane_steps,3))
        self.orientation_rotation_matrices = np.zeros((
            self.orientation_num_zones,
            self.orientation_in_plane_steps,
            3,3))

        # Calculate rotation matrices
        for a0 in tqdmnd(np.arange(self.orientation_num_zones),desc='Computing orientation basis',unit=' terms',unit_scale=True):
            m1z = np.array([
                [ np.cos(azim[a0]), np.sin(azim[a0]), 0],
                [-np.sin(azim[a0]), np.cos(azim[a0]), 0],
                [ 0,                0,                1]])
            m2x = np.array([
                [1,  0,                0],
                [0,  np.cos(elev[a0]), np.sin(elev[a0])],
                [0, -np.sin(elev[a0]), np.cos(elev[a0])]])
            # m12 = np.matmul(m2x, m1z)
            m12 = m1z @ m2x 

            for a1 in np.arange(self.orientation_in_plane_steps):
                self.orientation_rotation_angles[a0,a1,:] = [elev[a0], azim[a0], gamma[a1]]
                
                # orientation matrix
                m3z = np.array([
                    [ np.cos(gamma[a1]), np.sin(gamma[a1]), 0],
                    [-np.sin(gamma[a1]), np.cos(gamma[a1]), 0],
                    [ 0,                 0,                 1]])
                self.orientation_rotation_matrices[a0,a1,:,:] = m12 @ m3z     

        # Determine the radii of all spherical shells
        radii = np.unique(np.round(
            self.g_vec_leng / tol_distance) * tol_distance)
        # Remove zero beam
        keep = np.abs(radii) > tol_distance 
        self.orientation_shell_radii = radii[keep]

        # Assign each structure factor point to a radial shell
        self.orientation_shell_index = -1*np.ones(self.g_vec_all.shape[1])
        for a0 in range(self.orientation_shell_radii.size):
            self.orientation_shell_index[ \
                np.abs(self.orientation_shell_radii[a0] - \
                    self.g_vec_leng) < tol_distance] = a0




    def orientation_match(
        self,
        bragg_peaks,
        accel_voltage = 300e3, 
        subpixel_tilt=True,
        plot_corr=False,
        figsize=(12,6),
        corr_kernel_size=0.05,
        tol_structure_factor=0.2,
        ):
        """
        Solve for the best fit orientation of a single diffraction pattern.

        Args:
            bragg_peaks (PointList):            numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
            accel_voltage (np float):           kinetic energy of electrons specificed in volts
            subpixel_tilt (bool):               set to false for faster matching, returning the nearest corr point
            plot_corr (bool):                   set to true to plot the resulting correlogram
            corr_kernel_size (np float):    correlation kernel size length in Angstroms
            tol_structure_factor (np float):    tolerance of structure factor intensity to include on shell

        """

        accel_voltage = np.asarray(accel_voltage)

        # Calculate wavelenth
        wavelength = electron_wavelength_angstrom(accel_voltage)

        # Calculate z direction offset for peaks projected onto Ewald sphere
        k0 = 1 / wavelength;
        gz = k0 - np.sqrt(k0**2 - bragg_peaks.data['qx']**2 - bragg_peaks.data['qy']**2)

        # 3D Bragg peak data
        g_vec_all = np.vstack((
            bragg_peaks.data['qx'],
            bragg_peaks.data['qy'],
            gz))
        intensity_all = bragg_peaks.data['intensity']
        # Vector lengths
        g_vec_leng = np.linalg.norm(g_vec_all, axis=0)

        # init arrays
        shell_index = np.zeros(g_vec_all.shape[1])
        shell_ref_check = np.zeros(self.orientation_shell_radii.size,dtype=bool)

        # Assign each Bragg peak to nearest shell from reference
        for a0 in range(self.orientation_shell_radii.size):
            sub = np.abs(self.orientation_shell_radii[a0] - g_vec_leng) < corr_kernel_size
            shell_index[sub] = a0
            if np.sum(sub) > 0:
                shell_ref_check[a0] = True
        shell_ref_inds = np.where(shell_ref_check)

        # init correlogram
        corr = np.zeros((self.orientation_num_zones,self.orientation_in_plane_steps))

        # compute correlogram
        for ind_shell in np.nditer(shell_ref_inds):
            g_ref = self.g_vec_all[:,self.orientation_shell_index == ind_shell]

            sub = shell_index == ind_shell
            g_test = g_vec_all[:,sub]
            intensity_test = intensity_all[sub]

            # for a0 in range(g_test.shape[1]):
            #     # corr += intensity_test[a0] * np.maximum(corr_kernel_size 
            #     #     - np.sqrt(np.min(np.sum((
            #     #     np.tensordot(g_test[:,a0],
            #     #     self.orientation_rotation_matrices,axes=1)[:,:,:,None] 
            #     #     - g_ref[:,None,None,:])**2, axis=0), axis=2)), 0)
                # corr += intensity_test[a0] * np.maximum(corr_kernel_size 
                #     - np.sqrt(np.min(np.sum((np.tensordot( \
                #     self.orientation_rotation_matrices,
                #     g_test[:,a0], axes=1)[:,:,:,None] - g_ref[None,None,:,:])**2, axis=2), axis=2)), 0)
                # corr += intensity_test[a0] * np.maximum(corr_kernel_size 
                #     - np.sqrt(np.min(np.sum(((
                #     self.orientation_rotation_matrices @ g_test[:,a0])[:,:,:,None]
                #     - g_ref[None,None,:,:])**2, axis=2), axis=2)), 0)

            corr += np.sum(
                intensity_test * np.maximum(corr_kernel_size 
                - np.sqrt(np.min(np.sum(((
                self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

        # print(corr)
        # print(corr_test.shape)

        # Determine the best fit orientation
        inds = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        

        if subpixel_tilt is False:
            elev_azim_gamma = self.orientation_rotation_angles[inds[0],inds[1],:]
            print(elev_azim_gamma)

        else:
            # Sub pixel refinement of zone axis orientation
            if inds[0] == 0:
                # Zone axis is (0,0,1)
                zone_axis_fit = self.orientation_zone_axis_range[0,:]

            elif inds[0] == self.orientation_num_zones - self.orientation_zone_axis_steps - 1:
                # Zone axis is 1st user provided direction
                zone_axis_fit = self.orientation_zone_axis_range[1,:]

            elif inds[0] == self.orientation_num_zones - 1:
                # Zone axis is the 2nd user-provided direction
                zone_axis_fit = self.orientation_zone_axis_range[2,:]

            else:
                # Subpixel refinement
                elev = self.orientation_rotation_angles[inds[0],inds[1],0]
                azim = self.orientation_rotation_angles[inds[0],inds[1],1]
                zone_axis_fit = np.array((
                    np.cos(azim)*np.sin(elev),
                    np.sin(azim)*np.sin(elev),
                    np.cos(elev)))        

        temp = zone_axis_fit / np.linalg.norm(zone_axis_fit)
        temp = np.round(temp * 1e3) / 1e3
        temp /= np.min(np.abs(temp[np.abs(temp)>0]))
        print('Highest corr point @ (' + str(temp) + ')')


        # plotting
        if plot_corr is True:

            # 2D correlation slice
            sig_zone_axis = np.max(corr,axis=1)
            im_corr_zone_axis = np.zeros((self.orientation_zone_axis_steps+1, self.orientation_zone_axis_steps+1))
            for a0 in np.arange(self.orientation_zone_axis_steps+1):
                inds_val = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)
                im_corr_zone_axis[a0,range(a0+1)] = sig_zone_axis[inds_val]

            # Zone axis
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            cmin = np.min(sig_zone_axis)
            cmax = np.max(sig_zone_axis)

            im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)

            ax[0].imshow(
                im_plot,
                cmap='viridis',
                vmin=0.0,
                vmax=1.0)

            inds_plot = np.unravel_index(np.argmax(im_plot, axis=None), im_plot.shape)
            ax[0].scatter(inds_plot[1],inds_plot[0], s=120, linewidth = 2, facecolors='none', edgecolors='r')

            # im_handle = 
            # fig.colorbar(im_handle, ax=axs[0])


            label_0 = self.orientation_zone_axis_range[0,:]
            label_0 = np.round(label_0 * 1e3) * 1e-3
            label_0 /= np.min(np.abs(label_0[np.abs(label_0)>0]))

            label_1 = self.orientation_zone_axis_range[1,:]
            label_1 = np.round(label_1 * 1e3) * 1e-3
            label_1 /= np.min(np.abs(label_1[np.abs(label_1)>0]))

            label_2 = self.orientation_zone_axis_range[2,:]
            label_2 = np.round(label_2 * 1e3) * 1e-3
            label_2 /= np.min(np.abs(label_2[np.abs(label_2)>0]))

            ax[0].set_yticks([0])
            ax[0].set_yticklabels([
                str(label_0)])

            ax[0].set_xticks([0, self.orientation_zone_axis_steps])
            ax[0].set_xticklabels([
                str(label_1),
                str(label_2)])

            # In-plane rotation
            ax[1].plot(
                self.orientation_rotation_angles[inds[0],:,2] * 180/np.pi, 
                (corr[inds[0],:] - cmin)/(cmax - cmin));
            ax[1].set_xlabel('In-plane rotation angle [deg]')
            ax[1].set_ylabel('Correlation Signal for maximum zone axis')


            plt.show()

        return corr




    def generate_diffraction_pattern(
        self, 
        accel_voltage = 300e3, 
        zone_axis = [0,0,1],
        proj_x_axis = None,
        sigma_excitation_error = 0.02,
        tol_excitation_error_mult = 3,
        tol_intensity = 0.5
        ):
        """
        Generate a single diffraction pattern, return all peaks as a pointlist.

        Args:
            accel_voltage (np float):        kinetic energy of electrons specificed in volts
            zone_axis (np float vector):     3 element projection direction for sim pattern
            proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)
            sigma_excitation_error (np float): sigma value for Gaussian envelope applied to s_g (excitation errors) in units of Angstroms
            tol_excitation_error_mult (np float): tolerance in units of sigma for s_g inclusion
            tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots
        """

        accel_voltage = np.asarray(accel_voltage)
        zone_axis = np.asarray(zone_axis)

        if proj_x_axis is None:
            if (zone_axis == np.array([1,0,0])).all:
                proj_x_axis = np.array([0,1,0])
            else:
                proj_x_axis = np.array([1,0,0])



        # Calculate wavelenth
        wavelength = electron_wavelength_angstrom(accel_voltage)

        # wavevectors
        k0 = zone_axis / np.linalg.norm(zone_axis) / wavelength;
        # k0_plus_g = self.g_vec_all + k0[:,None]
        # cos_alpha = np.sum(k0[:,None] * self.g_vec_all, axis=0) \
        #     / np.linalg.norm(self.g_vec_all, axis=0) \
        #     / np.linalg.norm(k0)
        cos_alpha = np.sum((k0[:,None] + self.g_vec_all) * zone_axis[:,None], axis=0) \
            / np.linalg.norm(k0[:,None] + self.g_vec_all) \
            / np.linalg.norm(zone_axis)

        # Excitation errors
        sg = (-0.5) * np.sum((2*k0[:,None] + self.g_vec_all) * self.g_vec_all, axis=0) \
            / (np.linalg.norm(k0[:,None] + self.g_vec_all)) / cos_alpha

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = (sg <= sg_max)
        g_diff = self.g_vec_all[:,keep]
        
        # Diffracted peak intensities
        g_int = self.struct_factors_int[keep] \
            * np.exp(sg[keep]**2/(-2*sigma_excitation_error**2))

        # Intensity tolerance
        keep_int = g_int > tol_intensity

        # Diffracted peak locations
        ky_proj = np.cross(zone_axis, proj_x_axis)
        kx_proj = np.cross(ky_proj, zone_axis)
        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)
        gx_proj = np.sum(g_diff[:,keep_int] * kx_proj[:,None], axis=0)
        gy_proj = np.sum(g_diff[:,keep_int] * ky_proj[:,None], axis=0)

        # Output as PointList
        bragg_peaks = PointList([('qx','float64'),('qy','float64'),('intensity','float64')])
        bragg_peaks.add_pointarray(np.vstack((gx_proj, gy_proj, g_int[keep_int])).T)

        return bragg_peaks





def plot_diffraction_pattern(
    bragg_peaks,
    scale_markers=20,
    power_markers=1,
    figsize=(8,8),
    returnfig=False):
    """
    2D scatter plot of the Bragg peaks

    Args:
        bragg_peaks (PointList): numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
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





def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)





# def cdesign(degree):
#     """
#     Returns the spherical coordinates of Colin-design.

#     Args:
#         degree: int designating the maximum order

#     Returns:
#         azim: Nx1, azimuth of each point in the t-design
#         elev: Nx1, elevation of each point in the t-design
#         vecs: Nx3, array of cartesian coordinates for each point

#     """

#     degree = np.asarray(degree).astype(np.int)
#     steps = (degree // 4) + 1

#     u = np.array((0,0,1))
#     v = np.array((0,1,1)) / np.sqrt(2)
#     w = np.array((1,1,1)) / np.sqrt(3)

#     # Calculate points along u and v using the SLERP formula
#     # https://en.wikipedia.org/wiki/Slerp
#     weights = np.linspace(0,1,steps+1)
#     angle_u_v = np.arccos(np.sum(u * v))
#     pv = u[None,:] * np.sin((1-weights[:,None])*angle_u_v)/np.sin(angle_u_v) + \
#          v[None,:] * np.sin(   weights[:,None] *angle_u_v)/np.sin(angle_u_v) 

#     # Calculate points along u and w using the SLERP formula
#     angle_u_w = np.arccos(np.sum(u * w))
#     pw = u[None,:] * np.sin((1-weights[:,None])*angle_u_w)/np.sin(angle_u_w) + \
#          w[None,:] * np.sin(   weights[:,None] *angle_u_w)/np.sin(angle_u_w) 


#     # Init array to hold all points
#     num_points = ((steps+1)*(steps+2)/2).astype(np.int)
#     vecs = np.zeros((num_points,3))
#     vecs[0,:] = u

#     # Calculate points on 1/48th of the unit sphere with another application of SLERP
#     for a0 in np.arange(1,steps+1):
#         inds = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)

#         p0 = pv[a0,:]
#         p1 = pw[a0,:]
#         angle_p = np.arccos(np.sum(p0 * p1))

#         weights = np.linspace(0,1,a0+1)
#         vecs[inds,:] = \
#             p0[None,:] * np.sin((1-weights[:,None])*angle_p)/np.sin(angle_p) + \
#             p1[None,:] * np.sin(   weights[:,None] *angle_p)/np.sin(angle_p) 

#     # Expand to 1/8 of the sphere
#     vecs = np.vstack((
#         vecs[:,[0,1,2]],
#         vecs[:,[0,2,1]],
#         vecs[:,[1,0,2]],
#         vecs[:,[1,2,0]],
#         vecs[:,[2,0,1]],
#         vecs[:,[2,1,0]],
#         ))
#     # Remove duplicate points
#     vecs = np.unique(vecs, axis=0)

#     # Expand to full the sphere
#     vecs = np.vstack((
#         vecs*np.array(( 1, 1, 1)),
#         vecs*np.array((-1, 1, 1)),
#         vecs*np.array(( 1,-1, 1)),
#         vecs*np.array((-1,-1, 1)),
#         vecs*np.array(( 1, 1,-1)),
#         vecs*np.array((-1, 1,-1)),
#         vecs*np.array(( 1,-1,-1)),
#         vecs*np.array((-1,-1,-1)),
#         ))
#     # Remove duplicate points
#     vecs = np.unique(vecs, axis=0)

#     # Spherical coordinates
#     azim = np.arctan2(vecs[:,1],vecs[:,0])
#     elev = np.arctan2(np.hypot(vecs[:,1], vecs[:,0]), vecs[:,2])

#     return azim, elev, vecs
