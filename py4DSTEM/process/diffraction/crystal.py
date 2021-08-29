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
        self.lat_real = self.structure.lattice.matrix

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
        angle_step_zone_axis = 3.0,
        angle_step_in_plane = 6.0,
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
            angle_step_zone_axis (numpy float): Approximate angular step size for zone axis [degrees]
            angle_step_in_plane (numpy float):  Approximate angular step size for in-plane rotation [degrees]
            accel_voltage (numpy float):        Accelerating voltage for electrons [Volts]
            corr_kernel_size (np float):        Correlation kernel size length in Angstroms
            tol_distance (numpy float):         Distance tolerance for radial shell assignment [1/Angstroms]
        """

        # Accelerating voltage
        self.accel_voltage = np.asarray(accel_voltage)

        # Calculate wavelenth
        self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

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
        # Keep
        self.vecs = vecs

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
        radii_test = np.round(self.g_vec_leng / tol_distance) * tol_distance
        radii = np.unique(radii_test)
        # Remove zero beam
        keep = np.abs(radii) > tol_distance 
        self.orientation_shell_radii = radii[keep]

        # init
        self.orientation_shell_index = -1*np.ones(self.g_vec_all.shape[1], dtype='int')
        self.orientation_shell_count = np.zeros(self.orientation_shell_radii.size)
        # self.orientation_shell_weight = np.zeros(self.orientation_shell_radii.size)

        # Assign each structure factor point to a radial shell
        for a0 in range(self.orientation_shell_radii.size):
            sub = np.abs(self.orientation_shell_radii[a0] - radii_test) <= tol_distance / 2

            self.orientation_shell_index[sub] = a0
            self.orientation_shell_count[a0] = np.sum(sub)
            self.orientation_shell_radii[a0] = np.mean(self.g_vec_leng[sub])

        # normalization
        self.orientation_corr_kernel_size = np.array(corr_kernel_size)
        self.orientation_corr_norm = np.zeros((self.orientation_num_zones))

        # # # Testing
        # vec = np.arange(
        #     -self.k_max,
        #     self.k_max+corr_kernel_size,
        #     corr_kernel_size)
        # ya, xa = np.meshgrid(vec, vec)
        # keep = xa**2 + ya**2 <= (self.k_max+corr_kernel_size/2)**2

        # bragg_peaks_test = PointList([('qx','float64'),('qy','float64'),('intensity','float64')])
        # bragg_peaks_test.add_pointarray(np.vstack((
        #     xa[keep], 
        #     ya[keep], 
        #     np.ones(np.sum(keep)))).T)

        # orient = self.match_single_pattern(
        #     bragg_peaks_test,
        #     plot_corr=True,
        # )

        mat = np.zeros((self.orientation_num_zones,3,3))
        for a0 in range(self.orientation_num_zones):
            # mat[a0,:,:] = np.transpose(self.orientation_rotation_matrices[a0,0,:,:],(1,0))
            mat[a0,:,:] = np.linalg.inv(self.orientation_rotation_matrices[a0,0,:,:])

        knorm = np.array([0,0,1])
        k0 = knorm / self.wavelength

        for a0 in range(self.orientation_shell_radii.size):
            k_shell = self.orientation_shell_radii[a0]

            sub = self.orientation_shell_index == a0
            g_proj = mat @ self.g_vec_all[:,sub]
            # intensity_ref = np.mean(self.struct_factors_int[sub])
            # amplitude_ref = np.sqrt(np.mean(self.struct_factors_int[sub]))
            # amplitude_ref = np.sqrt(self.struct_factors_int[sub])

            # Calculate s_g
            cos_alpha = np.sum((k0[None,:,None] + g_proj) * knorm[None,:,None], axis=1) \
                / np.linalg.norm(k0[None,:,None] + g_proj, axis=1)
            sg = (-0.5) * np.sum((2*k0[None,:,None] + g_proj) * g_proj, axis=1) \
                / (np.linalg.norm(k0[None,:,None] + g_proj, axis=1)) / cos_alpha

            # Add into normalization output
            # self.orientation_corr_norm += np.sum(np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)
            # self.orientation_corr_norm += np.sum(amplitude_ref * np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)
            # self.orientation_corr_norm += intensity_ref * np.sum(np.maximum(1 - np.abs(sg) / corr_kernel_size, 0), axis=1)
            # self.orientation_corr_norm += intensity_ref * np.sum(np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)
            # self.orientation_corr_norm += amplitude_ref * np.sum(np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)
            self.orientation_corr_norm += k_shell * np.sum(
                self.struct_factors_int[sub] * 
                np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)
            # self.orientation_corr_norm += k_shell * np.sum(
            #     np.sqrt(self.struct_factors_int[sub]) * 
            #     np.maximum(1 - sg**2 / (4*corr_kernel_size**2), 0), axis=1)

        # self.orientation_corr_norm = np.sqrt(self.orientation_corr_norm)
        # self.orientation_corr_norm =self.orientation_corr_norm**2

            # sub_ref = self.orientation_shell_index == ind_shell
            # g_ref = self.g_vec_all[:,sub_ref]
            # # intensity_ref = self.struct_factors_int[sub_ref]            
            # intensity_ref = np.mean(self.struct_factors_int[sub_ref])

            # self.orientation_corr_norm += np.sum(self.struct_factors_int[None,sub] \
            #     * np.maximum(1 - 0.25*np.abs(sg) / corr_kernel_size, 0), axis=1)
            # self.orientation_corr_norm += np.sum(np.maximum(1 - np.abs(sg) / corr_kernel_size, 0), axis=1)

            # self.orientation_corr_norm += np.sum(self.struct_factors_int[None,sub] \
            #     * np.maximum(1 - 0.5 * np.abs(sg) / corr_kernel_size, 0)**2, axis=1)

        # self.orientation_corr_norm = np.sqrt(self.orientation_corr_norm)

            # self.orientation_corr_norm += np.sum(self.struct_factors_int[None,sub] \
            #     * np.maximum(corr_kernel_size - np.abs(sg), 0), axis=1)
            # self.orientation_corr_norm += np.sum(self.struct_factors_int[None,sub] \
            #     * (np.maximum(1 - np.abs(sg) / corr_kernel_size, 0))**2, axis=1)
            # self.orientation_corr_norm += np.sum(np.maximum(1 - np.abs(sg) / corr_kernel_size, 0), axis=1)**2

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
                           subpixel_tilt=True,
                           ):

        orientations = np.zeros((*bragg_peaks_array.shape, 3),dtype=np.float64)

        for rx,ry in tqdmnd(*bragg_peaks_array.shape, desc="Matching Orientations", unit="PointList"):
            bragg_peaks = bragg_peaks_array.get_pointlist(rx,ry)
            orientations[rx,ry,:] = self.match_single_pattern(bragg_peaks,
                                                              subpixel_tilt=subpixel_tilt,
                                                              plot_corr=False,
                                                              plot_corr_3D=False,
                                                              return_corr=False,
                                                              verbose=False,
                                                              )

        return orientations

    def match_single_pattern(
        self,
        bragg_peaks,
        subpixel_tilt=True,
        normalize_corr=True,
        plot_corr=False,
        plot_corr_3D=False,
        figsize=(12,6),
        return_corr=False,
        verbose=True,
        ):
        """
        Solve for the best fit orientation of a single diffraction pattern.

        Args:
            bragg_peaks (PointList):            numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
            subpixel_tilt (bool):               set to false for faster matching, returning the nearest corr point
            normalize_corr (bool):              set to true to use normalization
            plot_corr (bool):                   set to true to plot the resulting correlogram

        """

        # Calculate z direction offset for peaks projected onto Ewald sphere (downwards direction)
        k0 = 1 / self.wavelength
        gz = k0 - np.sqrt(k0**2 - bragg_peaks.data['qx']**2 - bragg_peaks.data['qy']**2)

        # 3D Bragg peak data
        g_vec_all = np.vstack((
            bragg_peaks.data['qx'],
            bragg_peaks.data['qy'],
            gz))
        intensity_all = bragg_peaks.data['intensity']
        # Vector lengths
        g_vec_leng = np.linalg.norm(g_vec_all, axis=0)

        # # init arrays
        # shell_index = np.zeros(g_vec_all.shape[1])
        # shell_ref_check = np.zeros(self.orientation_shell_radii.size,dtype=bool)

        # # Assign each Bragg peak to nearest shell from reference
        # for a0 in range(self.orientation_shell_radii.size):
        #     sub = np.abs(self.orientation_shell_radii[a0] - g_vec_leng) < self.orientation_corr_kernel_size
        #     shell_index[sub] = a0
        #     if np.sum(sub) > 0:
        #         shell_ref_check[a0] = True
        # shell_ref_inds = np.where(shell_ref_check)

        # init correlogram
        corr = np.zeros((self.orientation_num_zones,self.orientation_in_plane_steps))

        # compute correlogram
        # for ind_shell in np.nditer(shell_ref_inds):

        for ind_shell in range(self.orientation_shell_radii.size):
            sub_ref = self.orientation_shell_index == ind_shell
            g_ref = self.g_vec_all[:,sub_ref]
            # intensity_ref = self.struct_factors_int[sub_ref]            
            # intensity_ref = np.mean(self.struct_factors_int[sub_ref])
            amplitude_ref = np.sqrt(np.mean(self.struct_factors_int[sub_ref]))

            # Determine with experimental points may fall on this shell
            sub_test = np.abs(g_vec_leng - self.orientation_shell_radii[ind_shell]) \
                < self.orientation_corr_kernel_size

            # sub_test = shell_index == ind_shell
            if np.sum(sub_test) > 0:
                g_test = g_vec_all[:,sub_test]
                # intensity_test = intensity_all[sub_test]
                amplitude_test = np.sqrt(intensity_all[sub_test])


                # corr += np.sum(np.maximum(
                #     self.orientation_corr_kernel_size - np.sqrt(np.min(np.sum(((
                #         self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                #     - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

                # corr += np.sum(intensity_test * np.maximum(
                #     self.orientation_corr_kernel_size - np.sqrt(np.min(np.sum(((
                #         self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                #     - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

                # corr += intensity_ref * np.mean(intensity_test * np.maximum(
                #     self.orientation_corr_kernel_size - np.sqrt(np.min(np.sum(((
                #         self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                #     - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

                corr += amplitude_ref * np.sum(amplitude_test * np.maximum(
                    self.orientation_corr_kernel_size - np.sqrt(np.min(np.sum(((
                        self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                    - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

        # normalization
        if normalize_corr is True:
            corr = corr / self.orientation_corr_norm[:,None]


                # corr += intensity_ref * np.sum(intensity_test * np.maximum(
                #     self.orientation_corr_kernel_size  - np.sqrt(np.min(np.sum(((
                #     self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                #     - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

                # amplitude_test = np.sqrt(intensity_all[sub_test])

                # for a0 in range(g_test.shape[1]):
                #     corr +=(self.orientation_shell_radii[ind_shell] *  intensity_test[a0]) \
                #         * np.maximum(corr_kernel_size - np.sqrt(np.min(np.sum(((
                #         self.orientation_rotation_matrices @ g_test[:,a0])[:,:,:,None]
                #         - g_ref[None,None,:,:])**2, axis=2), axis=2)), 0)

                # corr += (self.orientation_shell_weight[ind_shell] * intensity_ref) \
                #     * np.mean(intensity_test * np.maximum(
                #     self.corr_kernel_size  - np.sqrt(np.min(np.sum(((
                #     self.orientation_rotation_matrices @ g_test)[:,:,:,:,None] 
                #     - g_ref[None,None,:,None,:])**2, axis=2), axis=3)), 0), axis=2)

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
        # temp /= np.min(np.abs(temp[np.abs(temp)>0.11]))
        temp = np.round(temp * 1e3) / 1e3
        if verbose:
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


        return (zone_axis_fit, corr) if return_corr else zone_axis_fit




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
            foil_normal:                     3 element foil normal - set to None to use zone_axis
            proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)
            sigma_excitation_error (np float): sigma value for envelope applied to s_g (excitation errors) in units of Angstroms
            tol_excitation_error_mult (np float): tolerance in units of sigma for s_g inclusion
            tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots
        """

        zone_axis = np.asarray(zone_axis)

        # Foil normal
        if foil_normal is None:
            foil_normal = zone_axis
        else:
            foil_normal = np.asarray(foil_normal)
        foil_normal = foil_normal / np.linalg.norm(foil_normal)

        # Logic to set x axis for projected images
        if proj_x_axis is None:
            if (zone_axis == np.array([1,0,0])).all:
                proj_x_axis = np.array([0,1,0])
            else:
                proj_x_axis = np.array([1,0,0])

        # wavevector
        zone_axis_norm = zone_axis / np.linalg.norm(zone_axis)
        k0 = zone_axis_norm / self.wavelength

        # Excitation errors
        # cos_alpha = np.sum((k0[:,None] + self.g_vec_all) * zone_axis_norm[:,None], axis=0) \
        #     / np.linalg.norm(k0[:,None] + self.g_vec_all, axis=0)
        cos_alpha = np.sum((k0[:,None] + self.g_vec_all) * foil_normal[:,None], axis=0) \
            / np.linalg.norm(k0[:,None] + self.g_vec_all, axis=0)
        sg = (-0.5) * np.sum((2*k0[:,None] + self.g_vec_all) * self.g_vec_all, axis=0) \
            / (np.linalg.norm(k0[:,None] + self.g_vec_all, axis=0)) / cos_alpha

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = np.abs(sg) <= sg_max
        g_diff = self.g_vec_all[:,keep]
        
        # Diffracted peak intensities
        g_int = self.struct_factors_int[keep] \
            * np.exp(sg[keep]**2/(-2*sigma_excitation_error**2))

        # Scale location of output peaks by diffraction angle
        # angle_ideal = np.arccos(np.minimum(np.sum( \
        #     (k0[:,None] + g_diff) * zone_axis_norm[:,None], axis=0) \
        #     / np.linalg.norm(k0[:,None] + g_diff, axis=0), 1))
        # angle_real = np.arccos(np.minimum(np.sum( \
        #     (k0[:,None] + g_diff + sg[keep]*zone_axis_norm[:,None]) * zone_axis_norm[:,None], axis=0)
        #     / np.linalg.norm(k0[:,None] + g_diff + sg[keep]*zone_axis_norm[:,None], axis=0), 1))
        # angle_nonzero = angle_real > 0
        # g_diff[:,angle_nonzero] = g_diff[:,angle_nonzero] * angle_real[angle_nonzero] / angle_ideal[angle_nonzero]

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
    scale_markers=10,
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
