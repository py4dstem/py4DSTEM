# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    import pymatgen as mg
    from pymatgen.ext.matproj import MPRester
except Exception:
    print(r"pymatgen not found... kinematic module won't work ¯\_(ツ)_/¯")

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, single_atom_scatter, electron_wavelength_angstrom


class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    Args:
        structure       a pymatgen Structure object for the material
                        or a string containing the Materials Project ID for the 
                        structure (requires API key in config file, see:
                        https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file
    """

    def __init__(
        self, 
        structure,
        conventional_standard_structure=True,
        **kwargs):
        """
        Instantiate a Crystal object. 
        Calculate lattice vectors.
        """
        
        if isinstance(structure, str):
            with MPRester() as m:
                structure = m.get_structure_by_material_id(structure)

        assert isinstance(
            structure, mg.core.Structure
        ), "structure must be pymatgen Structure object"


        self.structure = (
            mg.symmetry.analyzer.SpacegroupAnalyzer(
                structure
            ).get_conventional_standard_structure()
            if conventional_standard_structure
            else structure
        )
        self.struc_dict = self.structure.as_dict()
        self.lat_inv = self.structure.lattice.reciprocal_lattice_crystallographic.matrix.T
        self.lat_real = self.structure.lattice.matrix.T

        # Initialize Crystal
        self.positions = self.structure.frac_coords   #: fractional atomic coordinates

        #: atomic numbers - if only one value is provided, assume all atoms are same species
        self.numbers = np.array([s.Z for s in self.structure.species], dtype=np.intp)

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
        ax.set_box_aspect((1,1,1));

        plt.show();

        if returnfig:
            return fig, ax


    def orientation_plan(
        self, 
        zone_axis_range = np.array([[0,1,1],[1,1,1]]),
        angle_step_zone_axis = 2.0,
        angle_step_in_plane = 2.0,
        accel_voltage = 300e3, 
        corr_kernel_size = 0.05,
        tol_distance = 0.01,
        plot_corr_norm = False,
        figsize = (6,6),
        ):
        """
        Calculate the rotation basis arrays for an SO(3) rotation correlogram.
        
        Args:
            zone_axis_range (3x3 numpy float):  Row vectors give the range for zone axis orientations.
                                                Note that we always start at [0,0,1] to make z-x-z rotation work.
                                                Setting this to 'full' as a string will use a hemispherical range.
            angle_step_zone_axis (numpy float): Approximate angular step size for zone axis [degrees]
            angle_step_in_plane (numpy float):  Approximate angular step size for in-plane rotation [degrees]
            accel_voltage (numpy float):        Accelerating voltage for electrons [Volts]
            corr_kernel_size (np float):        Correlation kernel size length in Angstroms
            tol_distance (numpy float):         Distance tolerance for radial shell assignment [1/Angstroms]
        """

        # Store inputs
        self.accel_voltage = np.asarray(accel_voltage)
        self.orientation_kernel_size = np.asarray(corr_kernel_size)

        # Calculate wavelenth
        self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

        if isinstance(zone_axis_range, str):
            self.orientation_zone_axis_range = np.array([
                [0,0,1],
                [0,1,0],
                [1,0,0]])

            if zone_axis_range == 'full':
                self.orientation_full = True
                self.orientation_half = False
            elif zone_axis_range == 'half':
                self.orientation_full = False
                self.orientation_half = True

        else:
            # Define 3 vectors which span zone axis orientation range, normalize
            self.orientation_zone_axis_range = np.vstack((np.array([0,0,1]),np.array(zone_axis_range))).astype('float')
            self.orientation_zone_axis_range[1,:] /= np.linalg.norm(self.orientation_zone_axis_range[1,:])
            self.orientation_zone_axis_range[2,:] /= np.linalg.norm(self.orientation_zone_axis_range[2,:])

            self.orientation_full = False
            self.orientation_half = False


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
        self.orientation_vecs = np.zeros((self.orientation_num_zones,3))
        self.orientation_vecs[0,:] = self.orientation_zone_axis_range[0,:]
        self.orientation_inds = np.zeros((self.orientation_num_zones,3), dtype='int')


        # Calculate zone axis points on the unit sphere with another application of SLERP
        for a0 in np.arange(1,self.orientation_zone_axis_steps+1):
            inds = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)

            p0 = pv[a0,:]
            p1 = pw[a0,:]
            angle_p = np.arccos(np.sum(p0 * p1))

            weights = np.linspace(0,1,a0+1)
            self.orientation_vecs[inds,:] = \
                p0[None,:] * np.sin((1-weights[:,None])*angle_p)/np.sin(angle_p) + \
                p1[None,:] * np.sin(   weights[:,None] *angle_p)/np.sin(angle_p)

            self.orientation_inds[inds,0] = a0
            self.orientation_inds[inds,1] = np.arange(a0+1)


        # expand to quarter sphere if needed
        if self.orientation_half or self.orientation_full:
            vec_new = np.copy(self.orientation_vecs) * np.array([-1,1,1])
            orientation_sector = np.zeros(vec_new.shape[0], dtype='int')

            keep = np.zeros(vec_new.shape[0],dtype='bool')
            for a0 in range(keep.size):
                if np.sqrt(np.min(np.sum((self.orientation_vecs - vec_new[a0,:])**2,axis=1))) > tol_distance:
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep,:]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]
            # self.orientation_sector = np.hstack((
            #     orientation_sector,
            #     np.ones(np.sum(keep), dtype='int')))
            self.orientation_inds = np.vstack((
                self.orientation_inds, 
                self.orientation_inds[keep,:])).astype('int')
            self.orientation_inds[:,2] = np.hstack((
                orientation_sector,
                np.ones(np.sum(keep), dtype='int')))


        # expand to hemisphere if needed
        if self.orientation_full:
            vec_new = np.copy(self.orientation_vecs) * np.array([1,-1,1])

            keep = np.zeros(vec_new.shape[0],dtype='bool')
            for a0 in range(keep.size):
                if np.sqrt(np.min(np.sum((self.orientation_vecs - vec_new[a0,:])**2,axis=1))) > tol_distance:
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep,:]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]


            # self.orientation_sector = np.hstack((
            #     self.orientation_sector,
            #     self.orientation_sector[keep]+2))
            orientation_sector = np.hstack((
                self.orientation_inds[:,2],
                self.orientation_inds[keep,2] + 2))
            self.orientation_inds = np.vstack((
                self.orientation_inds, 
                self.orientation_inds[keep,:])).astype('int')
            self.orientation_inds[:,2] = orientation_sector


        # Convert to spherical coordinates
        # azim = np.arctan2(
        #     self.orientation_vecs[:,1],
        #     self.orientation_vecs[:,0])
        elev = np.arctan2(np.hypot(
            self.orientation_vecs[:,0], 
            self.orientation_vecs[:,1]), 
            self.orientation_vecs[:,2])
        azim = -np.pi/2 + np.arctan2(
            self.orientation_vecs[:,1],
            self.orientation_vecs[:,0])


        # Solve for number of angular steps along in-plane rotation direction
        self.orientation_in_plane_steps = np.round(360/angle_step_in_plane).astype(np.int)

        # Calculate -z angles (Euler angle 3)
        self.orientation_gamma = np.linspace(0,2*np.pi,self.orientation_in_plane_steps, endpoint=False)

        # Determine the radii of all spherical shells
        radii_test = np.round(self.g_vec_leng / tol_distance) * tol_distance
        radii = np.unique(radii_test)
        # Remove zero beam
        keep = np.abs(radii) > tol_distance 
        self.orientation_shell_radii = radii[keep]

        # init
        self.orientation_shell_index = -1*np.ones(self.g_vec_all.shape[1], dtype='int')
        self.orientation_shell_count = np.zeros(self.orientation_shell_radii.size)

        # Assign each structure factor point to a radial shell
        for a0 in range(self.orientation_shell_radii.size):
            sub = np.abs(self.orientation_shell_radii[a0] - radii_test) <= tol_distance / 2

            self.orientation_shell_index[sub] = a0
            self.orientation_shell_count[a0] = np.sum(sub)
            self.orientation_shell_radii[a0] = np.mean(self.g_vec_leng[sub])

        # init storage arrays
        self.orientation_rotation_angles = np.zeros((self.orientation_num_zones,2))
        self.orientation_rotation_matrices = np.zeros((self.orientation_num_zones,3,3))
        self.orientation_ref = np.zeros((
            self.orientation_num_zones,
            np.size(self.orientation_shell_radii),
            self.orientation_in_plane_steps),
            dtype='complex64')

        # Calculate rotation matrices for zone axes
        # for a0 in tqdmnd(np.arange(self.orientation_num_zones),desc='Computing orientation basis',unit=' terms',unit_scale=True):
        for a0 in np.arange(self.orientation_num_zones):
            m1z = np.array([
                [ np.cos(azim[a0]), -np.sin(azim[a0]), 0],
                [ np.sin(azim[a0]),  np.cos(azim[a0]), 0],
                [ 0,                 0,                1]])
            m2x = np.array([
                [1,  0,                0],
                [0,  np.cos(elev[a0]), np.sin(elev[a0])],
                [0, -np.sin(elev[a0]),  np.cos(elev[a0])]])
            self.orientation_rotation_matrices[a0,:,:] = m1z @ m2x
            # m2y = np.array([
            #     [np.cos(elev[a0]),     0,  np.sin(elev[a0])],
            #     [0,                    1,   0],
            #     [-np.sin(elev[a0]),     0,   np.cos(elev[a0])]])
            # self.orientation_rotation_matrices[a0,:,:] = m1z @ m2y
            # self.orientation_rotation_matrices[a0,:,:] = m2x @ m1z
            # print(np.round(self.orientation_rotation_matrices[a0,:,:]*100)/100)
            self.orientation_rotation_angles[a0,:] = [azim[a0], elev[a0]]

        # init
        k0 = np.array([0, 0, 1]) / self.wavelength
        dphi = self.orientation_gamma[1] - self.orientation_gamma[0]

        # Calculate reference arrays for all orientations
        for a0 in tqdmnd(np.arange(self.orientation_num_zones), desc="Orientation plan", unit=" zone axes"):
            p = np.linalg.inv(self.orientation_rotation_matrices[a0,:,:]) @ self.g_vec_all
            # p = self.orientation_rotation_matrices[a0,:,:] @ self.g_vec_all

            # Excitation errors
            cos_alpha = (k0[2,None] + p[2,:]) \
                / np.linalg.norm(k0[:,None] + p, axis=0)
            sg = (-0.5) * np.sum((2*k0[:,None] + p) * p, axis=0) \
                / (np.linalg.norm(k0[:,None] + p, axis=0)) / cos_alpha

            # in-plane rotation angle
            phi = np.arctan2(p[1,:],p[0,:])

            for a1 in np.arange(self.g_vec_all.shape[1]):
                ind_radial = self.orientation_shell_index[a1]

                if ind_radial >= 0:
                    self.orientation_ref[a0,ind_radial,:] += \
                        self.orientation_shell_radii[ind_radial] * np.sqrt(self.struct_factors_int[a1]) * \
                        np.maximum(1 - np.sqrt(sg[a1]**2 + \
                        ((np.mod(self.orientation_gamma - phi[a1] + np.pi, 2*np.pi) - np.pi) * \
                        self.orientation_shell_radii[ind_radial])**2) / self.orientation_kernel_size, 0)

            # Normalization
            self.orientation_ref[a0,:,:] /= \
                np.sqrt(np.sum(self.orientation_ref[a0,:,:]**2))
            # self.orientation_ref[a0,:,:] /= \
            #     np.sqrt(np.sum(np.abs(np.fft.fft(self.orientation_ref[a0,:,:]))**2))



        # s = np.hstack((
        #     np.round(self.orientation_rotation_angles * 180/np.pi*100)/100,
        #     np.round(self.orientation_vecs*100)/100))
        # print(s)
        # print(np.round(self.orientation_vecs*100)/100)

        # ind = 36
        # print(np.round(self.orientation_vecs[ind,:]*100)/100)


        # fig, ax = plt.subplots(figsize=(32,8))
        # im_plot = np.real(self.orientation_ref[ind,:,:]).astype('float')
        # cmax = np.max(im_plot)
        # im_plot = im_plot / cmax 

        # im = ax.imshow(
        #     im_plot,
        #     cmap='viridis',
        #     vmin=0.0,
        #     vmax=1.0)
        # fig.colorbar(im)


        # Fourier domain along angular axis
        # self.orientation_ref = np.fft.fft(self.orientation_ref)
        self.orientation_ref = np.conj(np.fft.fft(self.orientation_ref))


        # plot the correlation normalization
        if plot_corr_norm is True:
            # 2D correlation slice
            im_corr_zone_axis = np.zeros((self.orientation_zone_axis_steps+1, self.orientation_zone_axis_steps+1))
            for a0 in np.arange(self.orientation_zone_axis_steps+1):
                inds_val = np.arange(a0*(a0+1)/2, a0*(a0+1)/2 + a0 + 1).astype(np.int)
                im_corr_zone_axis[a0,range(a0+1)] = self.orientation_corr_norm[inds_val]

            # Zone axis
            fig, ax = plt.subplots(figsize=figsize)
            # cmin = np.min(self.orientation_corr_norm)
            cmax = np.max(self.orientation_corr_norm)

            # im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
            im_plot = im_corr_zone_axis / cmax 

            im = ax.imshow(
                im_plot,
                cmap='viridis',
                vmin=0.0,
                vmax=1.0)
            fig.colorbar(im)

            label_0 = self.orientation_zone_axis_range[0,:]
            label_0 = np.round(label_0 * 1e3) * 1e-3
            label_0 /= np.min(np.abs(label_0[np.abs(label_0)>0]))

            label_1 = self.orientation_zone_axis_range[1,:]
            label_1 = np.round(label_1 * 1e3) * 1e-3
            label_1 /= np.min(np.abs(label_1[np.abs(label_1)>0]))

            label_2 = self.orientation_zone_axis_range[2,:]
            label_2 = np.round(label_2 * 1e3) * 1e-3
            label_2 /= np.min(np.abs(label_2[np.abs(label_2)>0]))

            ax.set_yticks([0])
            ax.set_yticklabels([
                str(label_0)])

            ax.set_xticks([0, self.orientation_zone_axis_steps])
            ax.set_xticklabels([
                str(label_1),
                str(label_2)])

            plt.show()  


        # ind = 1;
        # x = g_proj[ind,0,:]
        # y = g_proj[ind,1,:]
        # z = g_proj[ind,2,:]

        # fig = plt.figure(figsize=(8,8))
        # ax = fig.add_subplot(
        #     projection='3d',
        #     elev=0, 
        #     azim=0)
        # ax.scatter(
        #     xs=x, 
        #     ys=y, 
        #     zs=z,
        #     s=30,
        #     edgecolors=None)
        # r = 2.1
        # ax.axes.set_xlim3d(left=-r, right=r) 
        # ax.axes.set_ylim3d(bottom=-r, top=r) 
        # ax.axes.set_zlim3d(bottom=-r, top=r) 
        # axisEqual3D(ax)
        # plt.show()


        # # Test plotting
        # fig = plt.figure(figsize=(16,8))
        # ax = fig.add_subplot()
        # ax.scatter(
        #     self.g_vec_leng, 
        #     self.orientation_shell_index, 
        #     s=5)
        # plt.show()

    def match_orientations(
                           self,
                           bragg_peaks_array,
                           num_matches_return = 1,
                           return_corr=False,
                           subpixel_tilt=False,
                           ):

        if num_matches_return == 1:
            orientation_matrices = np.zeros((*bragg_peaks_array.shape, 3, 3),dtype=np.float64)
            if return_corr:
                corr_all = np.zeros(bragg_peaks_array.shape,dtype=np.float64)
        else:
            orientation_matrices = np.zeros((*bragg_peaks_array.shape, 3, 3, num_matches_return),dtype=np.float64)
            if return_corr:
                corr_all = np.zeros((*bragg_peaks_array.shape, num_matches_return),dtype=np.float64)


        for rx,ry in tqdmnd(*bragg_peaks_array.shape, desc="Matching Orientations", unit=" PointList"):
            bragg_peaks = bragg_peaks_array.get_pointlist(rx,ry)

            if num_matches_return == 1:
                if return_corr:
                    orientation_matrices[rx,ry,:,:], corr_all[rx,ry] = self.match_single_pattern(
                        bragg_peaks,
                        subpixel_tilt=subpixel_tilt,
                        plot_corr=False,
                        plot_corr_3D=False,
                        return_corr=True,
                        verbose=False,
                        )
                else:
                    orientation_matrices[rx,ry,:,:] = self.match_single_pattern(
                        bragg_peaks,
                        subpixel_tilt=subpixel_tilt,
                        plot_corr=False,
                        plot_corr_3D=False,
                        return_corr=False,
                        verbose=False,
                        )
            else:
                if return_corr:
                    orientation_matrices[rx,ry,:,:,:], corr_all[rx,ry,:] = self.match_single_pattern(
                        bragg_peaks,
                        num_matches_return = num_matches_return,
                        subpixel_tilt=subpixel_tilt,
                        plot_corr=False,
                        plot_corr_3D=False,
                        return_corr=True,
                        verbose=False,
                        )
                else:
                    orientation_matrices[rx,ry,:,:,:] = self.match_single_pattern(
                        bragg_peaks,
                        num_matches_return = num_matches_return,
                        subpixel_tilt=subpixel_tilt,
                        plot_corr=False,
                        plot_corr_3D=False,
                        return_corr=False,
                        verbose=False,
                        )

        if return_corr:
            return orientation_matrices, corr_all
        else:
            return orientation_matrices

    def match_single_pattern(
        self,
        bragg_peaks,
        return_corr=False,
        num_matches_return = 1,
        tol_peak_delete = 0.1,
        subpixel_tilt=False,
        plot_corr=False,
        plot_corr_3D=False,
        figsize=(12,6),
        verbose=False,
        ):
        """
        Solve for the best fit orientation of a single diffraction pattern.

        Args:
            bragg_peaks (PointList):            numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
            num_matches_return (int):           return these many matches as 3th dim of orient (matrix)
            subpixel_tilt (bool):               set to false for faster matching, returning the nearest corr point
            plot_corr (bool):                   set to true to plot the resulting correlogram

        Returns:
            orientation_output (3x3xN float)    orienation matrix where zone axis is the 3rd column, 3rd dim for multiple matches
            corr_value (float):                 (optional) return correlation values
        """

        # get bragg peak data
        qx = bragg_peaks.data['qx']
        qy = bragg_peaks.data['qy']
        intensity = bragg_peaks.data['intensity']


        # init orientation output, delete distance threshold squared
        if num_matches_return == 1:
            orientation_output = np.zeros((3,3))
        else:
            orientation_output = np.zeros((3,3,num_matches_return))
            r_del_2 = tol_peak_delete**2
            corr_output = np.zeros((num_matches_return))

        # loop over the number of matches to return
        for match_ind in range(num_matches_return):
            # Convert Bragg peaks to polar coordinates
            qr = np.sqrt(qx**2 + qy**2)
            qphi = np.arctan2(qy, qx)

            # Calculate polar Bragg peak image
            im_polar = np.zeros((
                np.size(self.orientation_shell_radii),
                self.orientation_in_plane_steps),
                dtype='float')

            for ind_radial, radius in enumerate(self.orientation_shell_radii):
                dqr = np.abs(qr - radius)
                sub = dqr < self.orientation_kernel_size

                if np.sum(sub) > 0:
                    im_polar[ind_radial,:] = np.sum(
                        radius * np.sqrt(intensity[sub,None]) 
                        * np.maximum(1 - np.sqrt(dqr[sub,None]**2 + \
                        ((np.mod(self.orientation_gamma[None,:] - qphi[sub,None] + np.pi, 2*np.pi) - np.pi) * \
                        radius)**2) / self.orientation_kernel_size, 0), axis=0)

            # Calculate orientation correlogram
            corr_full = np.sum(np.real(np.fft.ifft(self.orientation_ref * np.fft.fft(im_polar[None,:,:]))), axis=1)
            # Find best match for each zone axis
            ind_phi = np.argmax(corr_full, axis=1)
            corr_value = np.zeros(self.orientation_num_zones)
            corr_in_plane_angle = np.zeros(self.orientation_num_zones)
            dphi = self.orientation_gamma[1] - self.orientation_gamma[0]

            for a0 in range(self.orientation_num_zones):
                inds = np.mod(ind_phi[a0] + np.arange(-1,2), self.orientation_gamma.size).astype('int')
                c = corr_full[a0,inds]

                if np.max(c) > 0:
                    corr_value[a0] = c[1] + (c[0]-c[2])**2 / (4*(2*c[1]-c[0]-c[2])**2)
                    dc = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])
                    corr_in_plane_angle[a0] = self.orientation_gamma[ind_phi[a0]] + dc*dphi

            # Determine the best fit orientation
            ind_best_fit = np.unravel_index(np.argmax(corr_value), corr_value.shape)[0]

            # Get orientation matrix
            if subpixel_tilt is False:
                orientation_matrix = np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])

            else:
                def ind_to_sub(ind):
                    ind_x = np.floor(0.5*np.sqrt(8.0*ind+1) - 0.5).astype('int')
                    ind_y = ind - np.floor(ind_x*(ind_x+1)/2).astype('int')
                    return ind_x, ind_y
                def sub_to_ind(ind_x, ind_y):
                    return (np.floor(ind_x*(ind_x+1)/2) + ind_y).astype('int')

                # Sub pixel refinement of zone axis orientation
                if ind_best_fit == 0:
                    # Zone axis is (0,0,1)
                    orientation_matrix = np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])

                elif ind_best_fit == self.orientation_num_zones - self.orientation_zone_axis_steps - 1:
                    # Zone axis is 1st user provided direction
                    orientation_matrix = np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])

                elif ind_best_fit == self.orientation_num_zones - 1:
                    # Zone axis is the 2nd user-provided direction
                    orientation_matrix = np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])

                else:
                    ind_x, ind_y = ind_to_sub(ind_best_fit)
                    max_x, max_y = ind_to_sub(self.orientation_num_zones-1)
                    # print(self.orientation_num_zones)
                    # print(ind_x, ind_y)
                    # print(max_x, max_y)

                    if ind_y == 0:
                        ind_x_prev = sub_to_ind(ind_x-1, 0)
                        ind_x_post = sub_to_ind(ind_x+1, 0)

                        c = np.array([corr_value[ind_x_prev], corr_value[ind_best_fit], corr_value[ind_x_post]])
                        dc = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])

                        if dc > 0:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1-dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_post,:,:])*dc
                        else:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1+dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_prev,:,:])*-dc

                    elif ind_x == max_x:
                        ind_x_prev = sub_to_ind(max_x, ind_y-1)
                        ind_x_post = sub_to_ind(max_x, ind_y+1)

                        c = np.array([corr_value[ind_x_prev], corr_value[ind_best_fit], corr_value[ind_x_post]])
                        dc = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])
                        
                        if dc > 0:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1-dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_post,:,:])*dc
                        else:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1+dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_prev,:,:])*-dc


                    elif ind_x == ind_y:
                        ind_x_prev = sub_to_ind(ind_x-1, ind_y-1)
                        ind_x_post = sub_to_ind(ind_x+1, ind_y+1)

                        c = np.array([corr_value[ind_x_prev], corr_value[ind_best_fit], corr_value[ind_x_post]])
                        dc = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])

                        if dc > 0:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1-dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_post,:,:])*dc
                        else:
                            orientation_matrix = \
                                np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1+dc) + \
                                np.squeeze(self.orientation_rotation_matrices[ind_x_prev,:,:])*-dc

                    else:
                        # # best fit point is not on any of the corners or edges
                        ind_1 = sub_to_ind(ind_x-1, ind_y-1)
                        ind_2 = sub_to_ind(ind_x-1, ind_y  )
                        ind_3 = sub_to_ind(ind_x  , ind_y-1)
                        ind_4 = sub_to_ind(ind_x  , ind_y+1)
                        ind_5 = sub_to_ind(ind_x+1, ind_y  )
                        ind_6 = sub_to_ind(ind_x+1, ind_y+1)

                        c = np.array([ \
                            (corr_value[ind_1]+corr_value[ind_2])/2, 
                            corr_value[ind_best_fit], 
                            (corr_value[ind_5]+corr_value[ind_6])/2])
                        dx = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])

                        c = np.array([corr_value[ind_3], corr_value[ind_best_fit], corr_value[ind_4]])
                        dy = (c[2]-c[0]) / (4*c[1] - 2*c[0] - 2*c[2])

                        if dx > 0:
                            if dy > 0:
                                orientation_matrix = \
                                    np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1-dx)*(1-dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_4,:,:])       *(1-dx)*(  dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_6,:,:])       *dx
                            else:
                                orientation_matrix = \
                                    np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1-dx)*(1+dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_3,:,:])       *(1-dx)*( -dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_5,:,:])       *dx
                        else:
                            if dy > 0:
                                orientation_matrix = \
                                    np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1+dx)*(1-dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_4,:,:])       *(1+dx)*(  dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_2,:,:])       *-dx
                            else:
                                orientation_matrix = \
                                    np.squeeze(self.orientation_rotation_matrices[ind_best_fit,:,:])*(1+dx)*(1+dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_3,:,:])       *(1+dx)*( -dy) + \
                                    np.squeeze(self.orientation_rotation_matrices[ind_1,:,:])       *-dx

            # apply in-plane rotation
            phi = corr_in_plane_angle[ind_best_fit] # + np.pi
            m3z = np.array([
                    [ np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi), np.cos(phi), 0],
                    [ 0,           0,           1]])
            orientation_matrix = orientation_matrix @ m3z

            # Output the orientation matrix
            if num_matches_return == 1:
                orientation_output = orientation_matrix
                corr_output = corr_value[ind_best_fit]

            else:
                orientation_output[:,:,match_ind] = orientation_matrix
                corr_output[match_ind] = corr_value[ind_best_fit]

            if verbose:
                zone_axis_fit = orientation_matrix[:,2]
                temp = zone_axis_fit / np.linalg.norm(zone_axis_fit)
                temp = np.round(temp * 1e3) / 1e3
                print('Best fit zone axis = (' 
                    + str(temp) + ')' 
                    + ' for corr value = ' 
                    + str(np.round(corr_value[ind_best_fit] * 1e3) / 1e3))

            # if needed, delete peaks for next iteration
            if num_matches_return > 1:
                bragg_peaks_fit = self.generate_diffraction_pattern(
                    orientation_matrix,
                    sigma_excitation_error=self.orientation_kernel_size)

                # qr_fit = np.sqrt(
                #     bragg_peaks_fit.data['qx']**2 +
                #     bragg_peaks_fit.data['qy']**2)
                # qphi_fit = np.arctan2(
                #     bragg_peaks_fit.data['qy'],
                #     bragg_peaks_fit.data['qx'])
                # qx = bragg_peaks.data['qx']
                # qy = bragg_peaks.data['qy']
                # qr = np.sqrt(qx**2 + qy**2)
                # qphi = np.arctan2(qy, qx)

                remove = np.zeros_like(qx,dtype='bool')
                for a0 in np.arange(qx.size):
                    d_2 = (bragg_peaks_fit.data['qx'] - qx[a0])**2 \
                        + (bragg_peaks_fit.data['qy'] - qy[a0])**2
                    if np.min(d_2) < r_del_2:
                        remove[a0] = True

                # qx = qx[~np.isin(qx, remove)]
                # qy = qy[~np.isin(qy, remove)]
                # qr = qr[~np.isin(qr, remove)]
                # qphi = qphi[~np.isin(qphi, remove)]
                qx = qx[~remove]
                qy = qy[~remove]
                intensity = intensity[~remove]


        # plotting correlation image
        if plot_corr is True:


            if self.orientation_full:
                fig, ax = plt.subplots(1, 2, figsize=figsize*np.array([2,2]))
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros((
                    2*self.orientation_zone_axis_steps+1, 
                    2*self.orientation_zone_axis_steps+1))
                
                sub = self.orientation_inds[:,2] == 0
                x_inds = (self.orientation_inds[sub,0] - self.orientation_inds[sub,1]).astype('int') \
                    + self.orientation_zone_axis_steps
                y_inds = self.orientation_inds[sub,1].astype('int') \
                    + self.orientation_zone_axis_steps
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:,2] == 1
                x_inds = (self.orientation_inds[sub,0] - self.orientation_inds[sub,1]).astype('int') \
                    + self.orientation_zone_axis_steps
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[sub,1].astype('int') 
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:,2] == 2
                x_inds = (self.orientation_inds[sub,1] - self.orientation_inds[sub,0]).astype('int') \
                    + self.orientation_zone_axis_steps
                y_inds = self.orientation_inds[sub,1].astype('int') \
                    + self.orientation_zone_axis_steps
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:,2] == 3
                x_inds = (self.orientation_inds[sub,1] - self.orientation_inds[sub,0]).astype('int') \
                    + self.orientation_zone_axis_steps
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[sub,1].astype('int') 
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]


                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(
                    im_plot,
                    cmap='viridis',
                    vmin=0.0,
                    vmax=1.0)

            elif self.orientation_half:
                fig, ax = plt.subplots(1, 2, figsize=figsize*np.array([2,1]))
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros((
                    self.orientation_zone_axis_steps+1, 
                    self.orientation_zone_axis_steps*2+1))
                
                sub = self.orientation_inds[:,2] == 0
                x_inds = (self.orientation_inds[sub,0] - self.orientation_inds[sub,1]).astype('int')
                y_inds = self.orientation_inds[sub,1].astype('int') + self.orientation_zone_axis_steps
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:,2] == 1
                x_inds = (self.orientation_inds[sub,0] - self.orientation_inds[sub,1]).astype('int')
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[sub,1].astype('int') 
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(
                    im_plot,
                    cmap='viridis',
                    vmin=0.0,
                    vmax=1.0)


            else:
                fig, ax = plt.subplots(1, 2, figsize=figsize)
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros((self.orientation_zone_axis_steps+1, self.orientation_zone_axis_steps+1))
                x_inds = (self.orientation_inds[:,0] - self.orientation_inds[:,1]).astype('int')
                y_inds = self.orientation_inds[:,1].astype('int')
                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_value

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(
                    im_plot,
                    cmap='viridis',
                    vmin=0.0,
                    vmax=1.0)

                inds_plot = np.unravel_index(np.argmax(im_plot, axis=None), im_plot.shape)
                ax[0].scatter(inds_plot[1],inds_plot[0], s=120, linewidth = 2, facecolors='none', edgecolors='r')

                label_0 = self.orientation_zone_axis_range[0,:]
                label_0 = np.round(label_0 * 1e3) * 1e-3
                label_0 /= np.min(np.abs(label_0[np.abs(label_0)>0]))

                label_1 = self.orientation_zone_axis_range[1,:]
                label_1 = np.round(label_1 * 1e3) * 1e-3
                label_1 /= np.min(np.abs(label_1[np.abs(label_1)>0]))

                label_2 = self.orientation_zone_axis_range[2,:]
                label_2 = np.round(label_2 * 1e3) * 1e-3
                label_2 /= np.min(np.abs(label_2[np.abs(label_2)>0]))

                ax[0].set_xticks([0, self.orientation_zone_axis_steps])
                ax[0].set_xticklabels([
                    str(label_0),
                    str(label_2)])
                ax[0].xaxis.tick_top()

                ax[0].set_yticks([self.orientation_zone_axis_steps])
                ax[0].set_yticklabels([
                    str(label_1)])

            # In-plane rotation
            ax[1].plot(
                self.orientation_gamma * 180/np.pi, 
                (np.squeeze(corr_full[ind_best_fit,:]) - cmin)/(cmax - cmin));
            ax[1].set_xlabel('In-plane rotation angle [deg]')
            ax[1].set_ylabel('Correlation Signal for maximum zone axis')
            ax[1].set_ylim([0,1.01])
            # ax[1].set_ylim([0.99,1.01])

            plt.show()

        if plot_corr_3D is True:
                    # 3D plotting

            fig = plt.figure(figsize=[figsize[0],figsize[0]])
            ax = fig.add_subplot(
                projection='3d',
                elev=90, 
                azim=0)

            sig_zone_axis = np.max(corr,axis=1)

            el = self.orientation_rotation_angles[:,0,0]
            az = self.orientation_rotation_angles[:,0,1]
            x = np.cos(az)*np.sin(el)
            y = np.sin(az)*np.sin(el)
            z =            np.cos(el)

            v = np.vstack((x.ravel(),y.ravel(),z.ravel()))

            v_order = np.array([
                [0,1,2],
                [0,2,1],
                [1,0,2],
                [1,2,0],
                [2,0,1],
                [2,1,0],
                ])
            d_sign = np.array([
                [ 1, 1, 1],
                [-1, 1, 1],
                [ 1,-1, 1],
                [-1,-1, 1],
                # [ 1, 1,-1],
                # [-1, 1,-1],
                # [ 1,-1,-1],
                # [-1,-1,-1],
                ])

            for a1 in range(d_sign.shape[0]):
                for a0 in range(v_order.shape[0]):
                    ax.scatter(
                        xs=v[v_order[a0,0]] * d_sign[a1,0], 
                        ys=v[v_order[a0,1]] * d_sign[a1,1], 
                        zs=v[v_order[a0,2]] * d_sign[a1,2],
                        s=30,
                        c=sig_zone_axis.ravel(),
                        edgecolors=None)


            # v = np.array([])


            # ax.scatter(
            #     xs=x, 
            #     ys=y, 
            #     zs=z,
            #     s=10)
            # # axes limits
            r = 1.05
            ax.axes.set_xlim3d(left=-r, right=r) 
            ax.axes.set_ylim3d(bottom=-r, top=r) 
            ax.axes.set_zlim3d(bottom=-r, top=r) 
            axisEqual3D(ax)


            plt.show()


        return (orientation_output, corr_output) if return_corr else orientation_output




    def generate_diffraction_pattern(
        self, 
        zone_axis = [0,0,1],
        foil_normal = None,
        proj_x_axis = None,
        sigma_excitation_error = 0.02,
        tol_excitation_error_mult = 3,
        tol_intensity = 0.1
        ):
        """
        Generate a single diffraction pattern, return all peaks as a pointlist.

        Args:
            zone_axis (np float vector):     3 element projection direction for sim pattern
                                             Can also be a 3x3 orientation matrix (zone axis 3rd column)
            foil_normal:                     3 element foil normal - set to None to use zone_axis
            proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)
            sigma_excitation_error (np float): sigma value for envelope applied to s_g (excitation errors) in units of Angstroms
            tol_excitation_error_mult (np float): tolerance in units of sigma for s_g inclusion
            tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots

        Returns:
            bragg_peaks (PointList):         list of all Bragg peaks with fields [qx, qy, intensity, h, k, l]
        """

        if zone_axis.ndim == 1:
            zone_axis = np.asarray(zone_axis)
        elif zone_axis.shape == (3,3):
            proj_x_axis = zone_axis[:,0]
            zone_axis = zone_axis[:,2]
        else:
            proj_x_axis = zone_axis[:,0,0]
            zone_axis = zone_axis[:,2,0]

        # Foil normal
        if foil_normal is None:
            foil_normal = zone_axis
        else:
            foil_normal = np.asarray(foil_normal)
        foil_normal = foil_normal / np.linalg.norm(foil_normal)

        # Logic to set x axis for projected images
        if proj_x_axis is None:
            # if (zone_axis == np.array([-1,0,0])).all:
            #     proj_x_axis = np.array([0,-1,0])
            # else:
            #     proj_x_axis = np.array([-1,0,0])

            if np.all(zone_axis == np.array([-1,0,0])):
                proj_x_axis = np.array([0,-1,0])
            elif np.all(zone_axis == np.array([1,0,0])):
                proj_x_axis = np.array([0,1,0])
            else:
                proj_x_axis = np.array([-1,0,0])


        # wavevector
        zone_axis_norm = zone_axis / np.linalg.norm(zone_axis)
        k0 = zone_axis_norm / self.wavelength

        # Excitation errors
        cos_alpha = np.sum((k0[:,None] + self.g_vec_all) * foil_normal[:,None], axis=0) \
            / np.linalg.norm(k0[:,None] + self.g_vec_all, axis=0)
        sg = (-0.5) * np.sum((2*k0[:,None] + self.g_vec_all) * self.g_vec_all, axis=0) \
            / (np.linalg.norm(k0[:,None] + self.g_vec_all, axis=0)) / cos_alpha

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = np.abs(sg) <= sg_max
        g_diff = self.g_vec_all[:,keep]
        
        # Diffracted peak intensities and labels
        g_int = self.struct_factors_int[keep] \
            * np.exp(sg[keep]**2/(-2*sigma_excitation_error**2))
        hkl = self.hkl[:, keep]

        # Intensity tolerance
        keep_int = g_int > tol_intensity

        # Diffracted peak locations
        ky_proj = np.cross(zone_axis, proj_x_axis)
        kx_proj = np.cross(ky_proj, zone_axis)

        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)
        gx_proj = np.sum(g_diff[:,keep_int] * kx_proj[:,None], axis=0)
        gy_proj = np.sum(g_diff[:,keep_int] * ky_proj[:,None], axis=0)

        # Diffracted peak labels
        h = hkl[0, keep_int]
        k = hkl[1, keep_int]
        l = hkl[2, keep_int]

        # Output as PointList
        bragg_peaks = PointList([
            ('qx','float64'),
            ('qy','float64'),
            ('intensity','float64'),
            ('h','int'),
            ('k','int'),
            ('l','int')])
        bragg_peaks.add_pointarray(np.vstack((
            gx_proj, 
            gy_proj, 
            g_int[keep_int],
            h,
            k,
            l)).T)

        return bragg_peaks



    def plot_orientation_maps(
        self,
        orientation_matrices,
        corr_all=None,
        corr_range=np.array([0, 5]),
        orientation_index_plot = 0,
        scale_legend = None,
        corr_normalize=True,
        figsize=(20,5),
        returnfig=False):
        """
        Generate and plot the orientation maps

        Args:
            orientation_zone_axis_range(float):     numpy array (3,3) where the 3 rows are the basis vectors for the orientation triangle
            orientation_matrices (float):   numpy array containing orientations, with size (Rx, Ry, 3, 3) or (Rx, Ry, 3, 3, num_matches)
            corr_all(float):                numpy array containing the correlation values to use as a mask
            orientation_index_plot (int):   index of orientations to plot
            scale_legend (float):           2 elements, x and y scaling of legend panel
            returnfig (bool):               set to True to return figure and axes handles

        Returns:
            images_orientation (int):       RGB images 
            fig, axs (handles):             Figure and axes handes for the 
        
        NOTE:
            Currently, no symmetry reduction.  Therefore the x and y orientations
            are going to be correct only for [001][011][111] orientation triangle.

        """


        # Inputs
        # Legend size
        leg_size = np.array([300,300],dtype='int')

        # Color of the 3 corners
        color_basis = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.3, 1.0],
            ])

        # Basis for fitting
        A = self.orientation_zone_axis_range.T

        # initalize image arrays
        images_orientation = np.zeros((
            orientation_matrices.shape[0],
            orientation_matrices.shape[1],
            3,3))


        # loop over all pixels and calculate weights
        for ax in range(orientation_matrices.shape[0]):
            for ay in range(orientation_matrices.shape[1]):
                if orientation_matrices.ndim == 4:
                    orient = orientation_matrices[ax,ay,:,:]
                else:
                    orient = orientation_matrices[ax,ay,:,:,orientation_index_plot]


                for a0 in range(3):
                    # w = np.linalg.solve(A,orient[:,a0])
                    w = np.linalg.solve(
                        A,
                        np.sort(np.abs(orient[:,a0])))
                    w = w / (1 - np.exp(-np.max(w)))
                    # np.max(w)

                    rgb = color_basis[0,:] * w[0] \
                        + color_basis[1,:] * w[1] \
                        + color_basis[2,:] * w[2]

                    images_orientation[ax,ay,:,a0] = rgb

        # clip range
        images_orientation = np.clip(images_orientation,0,1)


        # Masking
        if corr_all is not None:
            if orientation_matrices.ndim == 4:
                if corr_normalize:
                    mask = corr_all / np.mean(corr_all)
                else:
                    mask = corr_all
            else:
                if corr_normalize:
                    mask = corr_all[:,:,orientation_index_plot] / np.mean(corr_all[:,:,orientation_index_plot])
                else:
                    mask = corr_all[:,:,orientation_index_plot]
                

            mask = (mask - corr_range[0]) / (corr_range[1] - corr_range[0])
            mask = np.clip(mask,0,1)

            for a0 in range(3):
                for a1 in range(3):
                    images_orientation[:,:,a0,a1] *=  mask

            # images_orientation *= mask
                # vec = np.array([1,-1,1])
                # vec = vec / np.linalg.norm(vec)
                # wz = np.linalg.solve(A, np.sort(np.abs(vec)))
                # print(np.round(wz*1e3)/1e3)

        # print(self.orientation_zone_axis_range)




        # Draw legend
        x = np.linspace(0,1,leg_size[0])
        y = np.linspace(0,1,leg_size[1])
        ya,xa=np.meshgrid(y,x)
        mask_legend = np.logical_and(2*xa > ya, 2*xa < 2-ya) 
        w0 = 1-xa - 0.5*ya
        w1 = xa - 0.5*ya
        w2 = ya

        w_scale = np.maximum(np.maximum(w0,w1),w2)
        # w_scale = w0 + w1 + w2
        # w_scale = (w0**4 + w1**4 + w2**4)**0.25
        w_scale = 1 - np.exp(-w_scale)
        w0 = w0 / w_scale # * mask_legend
        w1 = w1 / w_scale # * mask_legend
        w2 = w2 / w_scale # * mask_legend
        
        im_legend = np.zeros((
            leg_size[0],
            leg_size[1],
            3))
        for a0 in range(3):
            im_legend[:,:,a0] = \
                w0*color_basis[0,a0] + \
                w1*color_basis[1,a0] + \
                w2*color_basis[2,a0]
            im_legend[:,:,a0] *= mask_legend
            im_legend[:,:,a0] += 1-mask_legend
        im_legend = np.clip(im_legend,0,1)

        # plotting
        fig, ax = plt.subplots(1, 4, figsize=figsize)

        ax[0].imshow(images_orientation[:,:,:,0])
        ax[1].imshow(images_orientation[:,:,:,1])
        ax[2].imshow(images_orientation[:,:,:,2])

        ax[0].set_title(
            'Orientation of x-axis',
            size=20)
        ax[1].set_title(
            'Orientation of y-axis',
            size=20)
        ax[2].set_title(
            'Zone Axis',
            size=20) 
        ax[0].xaxis.tick_top()
        ax[1].xaxis.tick_top()
        ax[2].xaxis.tick_top()

        # Legend
        ax[3].imshow(im_legend,
            aspect='auto')

        label_0 = self.orientation_zone_axis_range[0,:]
        label_0 = np.round(label_0 * 1e3) * 1e-3
        label_0 /= np.min(np.abs(label_0[np.abs(label_0)>0]))

        label_1 = self.orientation_zone_axis_range[1,:]
        label_1 = np.round(label_1 * 1e3) * 1e-3
        label_1 /= np.min(np.abs(label_1[np.abs(label_1)>0]))

        label_2 = self.orientation_zone_axis_range[2,:]
        label_2 = np.round(label_2 * 1e3) * 1e-3
        label_2 /= np.min(np.abs(label_2[np.abs(label_2)>0]))
        

        # ax[3].set_xticks([0])
        # ax[3].set_xticklabels([
        #     str(label_0)])
        # ax[3].xaxis.tick_top()

        # ax[3].axis.set_label_position("right")
        ax[3].yaxis.tick_right()
        ax[3].set_yticks([(leg_size[0]-1)/2])
        ax[3].set_yticklabels([
            str(label_2)])

        ax3a = ax[3].twiny()
        ax3b = ax[3].twiny()

        ax3a.set_xticks([0])
        ax3a.set_xticklabels([
            str(label_0)])
        ax3a.xaxis.tick_top()
        ax3b.set_xticks([0])
        ax3b.set_xticklabels([
            str(label_1)])
        ax3b.xaxis.tick_bottom()
        ax[3].set_xticks([])

        # ax[3].xaxis.label.set_color('none')
        ax[3].spines['left'].set_color('none')
        ax[3].spines['right'].set_color('none')
        ax[3].spines['top'].set_color('none')
        ax[3].spines['bottom'].set_color('none')
        
        ax3a.spines['left'].set_color('none')
        ax3a.spines['right'].set_color('none')
        ax3a.spines['top'].set_color('none')
        ax3a.spines['bottom'].set_color('none')
        
        ax3b.spines['left'].set_color('none')
        ax3b.spines['right'].set_color('none')
        ax3b.spines['top'].set_color('none')
        ax3b.spines['bottom'].set_color('none')
        
        ax[3].tick_params(labelsize=16)
        ax3a.tick_params(labelsize=16)
        ax3b.tick_params(labelsize=16)


        if scale_legend is not None:
            pos = ax[3].get_position()
            pos_new = [
                pos.x0, 
                pos.y0 + pos.height*(1 - scale_legend[1])/2,
                pos.width*scale_legend[0], 
                pos.height*scale_legend[1],
                ] 
            ax[3].set_position(pos_new) 
        #     if scale_legend[0] != 1:
        #         x_range = ax[3].get_xlim()
        #         x_range_new = np.array([
        #             x_range[0],
        #             x_range[1] / scale_legend[0],
        #             ])                
        #         ax[3].set_xlim(x_range_new)
        #         ax3a.set_xlim(x_range_new)
        #         ax3b.set_xlim(x_range_new)

        #         # y_range = ax[3].get_xlim()

        #         # print(y_range)            

        if returnfig:
            return images_orientation, fig, ax
        else:
            return images_orientation


def plot_diffraction_pattern(
    bragg_peaks,
    bragg_peaks_compare=None,
    scale_markers=10,
    scale_markers_compare=None,
    power_markers=1,
    plot_range_kx_ky=None,
    add_labels=True,
    shift_labels=0.08,
    shift_marker = 0.005,
    min_marker_size = 1e-6,
    figsize=(8,8),
    returnfig=False):
    """
    2D scatter plot of the Bragg peaks

    Args:
        bragg_peaks (PointList):        numpy array containing ('qx', 'qy', 'intensity', 'h', 'k', 'l')
        bragg_peaks_compare(PointList): numpy array containing ('qx', 'qy', 'intensity')
        scale_markers (float):          size scaling for markers
        scale_markers_compare (float):  size scaling for markers of comparison
        power_markers (float):          power law scaling for marks (default is 1, i.e. amplitude)
        plot_range_kx_ky (float):       2 element numpy vector giving the plot range
        add_labels (bool):              flag to add hkl labels to peaks
        min_marker_size (float):        minimum marker size for the comparison peaks
        figsize (2 element float):      size scaling of figure axes
        returnfig (bool):               set to True to return figure and axes handles
    """

    # 2D plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    if power_markers == 2:
        marker_size = scale_markers*bragg_peaks.data['intensity']
    else:
        marker_size = scale_markers*(bragg_peaks.data['intensity']**(power_markers/2))

    if bragg_peaks_compare is None:
        ax.scatter(
            bragg_peaks.data['qy'], 
            bragg_peaks.data['qx'], 
            s=marker_size,
            facecolor='k')
    else:
        if scale_markers_compare is None:
            scale_markers_compare = scale_markers

        if power_markers == 2:
            marker_size_compare = np.maximum(
                scale_markers_compare*bragg_peaks_compare.data['intensity'], min_marker_size)
        else:
            marker_size_compare = np.maximum(
                scale_markers_compare*(bragg_peaks_compare.data['intensity']**(power_markers/2)), min_marker_size)

        ax.scatter(
            bragg_peaks_compare.data['qy'], 
            bragg_peaks_compare.data['qx'], 
            s=marker_size_compare,
            marker='o',
            facecolor=[0.0,0.7,1.0])
        ax.scatter(
            bragg_peaks.data['qy'], 
            bragg_peaks.data['qx'], 
            s=marker_size,
            marker='+',
            facecolor='k')


    if plot_range_kx_ky is not None:
        ax.set_xlim((-plot_range_kx_ky[0],plot_range_kx_ky[0]))
        ax.set_ylim((-plot_range_kx_ky[1],plot_range_kx_ky[1]))

    ax.invert_yaxis()
    ax.set_box_aspect(1)

    # Labels for all peaks
    if add_labels is True:
        # shift_labels = 0.08
        # shift_marker = 0.005
        text_params = {
            'ha': 'center',
            'va': 'center',
            'family': 'sans-serif',
            'fontweight': 'normal',
            'color': 'r',
            'size': 10}

        # def overline(x):
        #     return str(x) if np.abs(x) >= 0 else '$\overline{" + str(np.abs(x)) + "}$'
        # def overline(x):
        #     if np.abs(x) >= 0:
        #         return str(x) 
        #     else:
        #         return '$\overline{" + str(np.abs(x)) + "}$'

        for a0 in np.arange(bragg_peaks.data.shape[0]):
            h = bragg_peaks.data['h'][a0]
            k = bragg_peaks.data['k'][a0]
            l = bragg_peaks.data['l'][a0]

            # plt.text( \
                # bragg_peaks.data['qy'][a0],
                # bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                # f'{overline(h)}{overline(k)}{overline(l)}',
                # **text_params)  


            if h >= 0:
                if k >= 0:
                    if l >= 0:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            str(h) + ' ' + str(k) + ' ' + str(l),
                            **text_params)  
                    else:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            str(h) + ' ' + str(k) + ' ' + '$\overline{' + str(np.abs(l)) + '}$',
                            **text_params)  
                else:
                    if l >= 0:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            str(h) + ' ' + '$\overline{' + str(np.abs(k)) + '}$' + ' ' + str(l),
                            **text_params)  
                    else:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            str(h) + ' ' + '$\overline{' + str(np.abs(k)) + '}$' + ' ' + '$\overline{' + str(np.abs(l)) + '}$',
                            **text_params)  
            else:
                if k >= 0:
                    if l >= 0:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            '$\overline{' + str(np.abs(h)) + '}$' + ' ' + str(k) + ' ' + str(l),
                            **text_params)  
                    else:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            '$\overline{' + str(np.abs(h)) + '}$' + ' ' + str(k) + ' ' + '$\overline{' + str(np.abs(l)) + '}$',
                            **text_params)  
                else:
                    if l >= 0:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            '$\overline{' + str(np.abs(h)) + '}$' + ' ' + '$\overline{' + str(np.abs(k)) + '}$' + ' ' + str(l),
                            **text_params)  
                    else:
                        plt.text( \
                            bragg_peaks.data['qy'][a0],
                            bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
                            '$\overline{' + str(np.abs(h)) + '}$' + ' ' + '$\overline{' + str(np.abs(k)) + '}$' + ' ' + '$\overline{' + str(np.abs(l)) + '}$',
                            **text_params) 

            # if bragg_peaks.data['h'][a0] >= 0:
            #     t1 = str(bragg_peaks.data['h'][a0])
            # # else:
            # #     t1 = u'str(bragg_peaks.data['h'][a0])\u0305'

            # t = t1 + ' ' + \
            #     str(bragg_peaks.data['k'][a0]) + ' ' + \
            #     str(bragg_peaks.data['l'][a0])


            # plt.text( \
            #     bragg_peaks.data['qy'][a0],
            #     bragg_peaks.data['qx'][a0] - shift_labels - shift_marker*np.sqrt(marker_size[a0]),
            #     t,
            #     **text_params)      

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
