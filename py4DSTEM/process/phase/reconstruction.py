# Functions for phase reconstruction methods and classes.

import numpy as np
import matplotlib.pyplot as plt
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.process.utils import get_shifted_ar
from py4DSTEM.process.calibration import fit_origin

class Reconstruction:
    """
    A class which stores phase/complex reconstructions of object and probe waves.
    This includes differential phase contrast components including center-of-mass.
    """

    def __init__(
        self,
        dataset,
        fitfunction='plane',
        plot_center_of_mass = True,
        figsize = (12,12),
        progress_bar = True,
        ):
        """
        Args:
            dataset: (DataCube)     Raw 4D datacube


        """

        # coordinates
        kx = np.arange(dataset.data.shape[2])
        ky = np.arange(dataset.data.shape[3])
        kya,kxa = np.meshgrid(ky,kx)

        # init
        self.shape_xy = np.array(dataset.data.shape[0:2])
        self.com_meas_x = np.zeros(self.shape_xy)
        self.com_meas_y = np.zeros(self.shape_xy)
        self.int_total = np.zeros(self.shape_xy)

        # calculate the center of mass for all probe positions
        for rx, ry in tqdmnd(
            dataset.data.shape[0],
            dataset.data.shape[1],
            desc="Fitting center of mass",
            unit=" positions",
            disable=not progress_bar,
            ):
            self.int_total[rx,ry] = np.sum(dataset.data[rx,ry])
            self.com_meas_x[rx,ry] = np.sum(dataset.data[rx,ry] * kxa) / self.int_total[rx,ry]
            self.com_meas_y[rx,ry] = np.sum(dataset.data[rx,ry] * kya) / self.int_total[rx,ry]

        # Fit function to center of mass
        or_fits = fit_origin(
            (self.com_meas_x,self.com_meas_y),
            fitfunction='plane',
        )
        self.com_fit_x = or_fits[0]
        self.com_fit_y = or_fits[1]
        self.com_norm_x = self.com_meas_x - self.com_fit_x
        self.com_norm_y = self.com_meas_y - self.com_fit_y

        if plot_center_of_mass is True:
            fig,ax = plt.subplots(2,2,figsize=figsize)

            ax[0,0].imshow(
                self.com_meas_x,
                cmap='RdBu_r',
            )

            ax[0,1].imshow(
                self.com_meas_y,
                cmap='RdBu_r',
            )
            ax[1,0].imshow(
                self.com_norm_x,
                cmap='RdBu_r',
            )

            ax[1,1].imshow(
                self.com_norm_y,
                cmap='RdBu_r',
            )

    def solve_rotation(
        self,
        rotation_deg = np.arange(-90.0,90.0,1.0),
        plot_result = True,
        print_result = True,
        figsize = (12,4),
        progress_bar = True,
        ):
        """
        Solve for the relative rotation between real and reciprocal space.
        We do this by minimizing the curl of the vector field.
        Alternative - maximize the vector field divergence - less sharp though.
        """

        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.rotation_curl = np.zeros_like(rotation_deg)
        self.rotation_curl_transpose = np.zeros_like(rotation_deg)
        # self.rotation_div = np.zeros_like(rotation_deg)
        # self.rotation_div_transpose = np.zeros_like(rotation_deg)

        for a0 in tqdmnd(
            rotation_deg.shape[0],
            desc="Fitting rotation",
            unit=" angles",
            disable=not progress_bar,
            ):
            com_meas_x = np.cos(self.rotation_rad[a0]) * self.com_norm_x  \
                - np.sin(self.rotation_rad[a0]) * self.com_norm_y
            com_meas_y = np.sin(self.rotation_rad[a0]) * self.com_norm_x  \
                + np.cos(self.rotation_rad[a0]) * self.com_norm_y
            
            grad_x_y = com_meas_x[1:-1,2:] - com_meas_x[1:-1,:-2]
            grad_y_x = com_meas_y[2:,1:-1] - com_meas_y[:-2,1:-1]
            self.rotation_curl[a0] = np.mean(np.abs(grad_y_x - grad_x_y))

            # grad_x_x = com_meas_x[2:,1:-1] - com_meas_x[:-2,1:-1]
            # grad_y_y = com_meas_y[1:-1,2:] - com_meas_y[1:-1,:-2]
            # self.rotation_div[a0] = np.mean(np.abs(grad_x_x + grad_y_y))

            com_meas_x = np.cos(self.rotation_rad[a0]) * self.com_norm_y  \
                - np.sin(self.rotation_rad[a0]) * self.com_norm_x
            com_meas_y = np.sin(self.rotation_rad[a0]) * self.com_norm_y  \
                + np.cos(self.rotation_rad[a0]) * self.com_norm_x
            
            grad_x_y = com_meas_x[1:-1,2:] - com_meas_x[1:-1,:-2]
            grad_y_x = com_meas_y[2:,1:-1] - com_meas_y[:-2,1:-1]
            self.rotation_curl_transpose[a0] = np.mean(np.abs(grad_y_x - grad_x_y))

            # grad_x_x = com_meas_x[2:,1:-1] - com_meas_x[:-2,1:-1]
            # grad_y_y = com_meas_y[1:-1,2:] - com_meas_y[1:-1,:-2]
            # self.rotation_div_transpose[a0] = np.mean(np.abs(grad_x_x + grad_y_y))

        # Find lowest curl value
        ind_min = np.argmin(self.rotation_curl)
        ind_trans_min = np.argmin(self.rotation_curl_transpose)
        if self.rotation_curl[ind_min] <= self.rotation_curl_transpose[ind_trans_min]:
            self.rotation_best_deg = self.rotation_deg[ind_min]
            self.rotation_best_rad = self.rotation_rad[ind_min]
            self.rotation_best_tranpose = False
        else:
            self.rotation_best_deg = self.rotation_deg[ind_trans_min]
            self.rotation_best_rad = self.rotation_rad[ind_trans_min]
            self.rotation_best_tranpose = True

        # calculate corrected CoM
        if self.rotation_best_tranpose is False:
            self.com_x = np.cos(self.rotation_best_rad) * self.com_norm_x  \
                - np.sin(self.rotation_best_rad) * self.com_norm_y
            self.com_y = np.sin(self.rotation_best_rad) * self.com_norm_x  \
                + np.cos(self.rotation_best_rad) * self.com_norm_y
        else:
            self.com_x = np.cos(self.rotation_best_rad) * self.com_norm_y  \
                - np.sin(self.rotation_best_rad) * self.com_norm_x
            self.com_y = np.sin(self.rotation_best_rad) * self.com_norm_y  \
                + np.cos(self.rotation_best_rad) * self.com_norm_x

        if plot_result:
            fig,ax = plt.subplots(figsize=figsize)
            ax.plot(
                self.rotation_deg,
                self.rotation_curl,
                label='CoM',
                )
            ax.plot(
                self.rotation_deg,
                self.rotation_curl_transpose,
                label='CoM after transpose',
                )
            y_r = ax.get_ylim()
            ax.plot(
                np.ones(2)*self.rotation_best_deg,
                y_r,
                color=(0,0,0,1),
                )


            ax.set_xlabel(
                'Rotation [degrees]',
                fontsize=16,
                )
            ax.set_ylabel(
                'Mean Absolute Curl',
                fontsize=16,
                )
            ax.legend(
                loc='best',
                fontsize=12,
                )
            plt.show()

        # Display results
        if print_result:
            print('Best fit rotation = ' + str(np.round(self.rotation_best_deg)) + ' degrees')
            if self.rotation_best_tranpose:
                print('Diffraction space should be transposed')
            else:
                print('No diffraction transposed needed')

    def dpc_recon(
        self,
        padding_factor = 2,
        num_iterations = 64,
        step_size = 0.9,
        plot_result = True,
        figsize = (8,8),
        progress_bar = True,
        ):
        """
        This function performs iterative DPC reconstruction.
        """

        # init
        shape_xy_pad = np.round(self.shape_xy * padding_factor).astype('int')
        dpc_phase_pad = np.zeros(shape_xy_pad)
        mask = np.zeros(shape_xy_pad, dtype='bool')
        mask[:self.shape_xy[0],:self.shape_xy[1]] = True
        mask_inv = np.logical_not(mask)

        # Fourier coordinates and operators
        kx = np.fft.fftfreq(shape_xy_pad[0],d=1.0)
        ky = np.fft.fftfreq(shape_xy_pad[1],d=1.0)
        kya,kxa = np.meshgrid(ky,kx)
        k_den = kxa**2 + kya**2
        k_den[0,0] = np.inf
        k_den = 1 / k_den
        kx_op = -1j*0.25*kxa*k_den
        ky_op = -1j*0.25*kya*k_den

        # main loop
        for a0 in tqdmnd(
            num_iterations,
            desc="Reconstructing phase",
            unit=" iter",
            disable=not progress_bar,
            ):
            # forward projection
            dx = 0.5 * (np.roll(dpc_phase_pad,1,axis=0) - np.roll(dpc_phase_pad,-1,axis=0))
            dy = 0.5 * (np.roll(dpc_phase_pad,1,axis=1) - np.roll(dpc_phase_pad,-1,axis=1))


            # difference from measurement
            dx[mask] += self.com_x.ravel()
            dy[mask] += self.com_y.ravel()
            dx[mask_inv] = 0
            dy[mask_inv] = 0

            # back projection
            phase_update = np.real(
                np.fft.ifft2(
                    np.fft.fft2(dx)*kx_op + np.fft.fft2(dy)*ky_op
                )
            )

            # update
            dpc_phase_pad += step_size * phase_update


        # crop result
        self.dpc_phase = dpc_phase_pad[:self.shape_xy[0],:self.shape_xy[1]]


        # plotting
        if plot_result:
            fig,ax = plt.subplots(figsize=figsize)
            ax.imshow(
                self.dpc_phase,
                )
            plt.show()

    def ptychography_init(
        self,
        dataset,
        probe,
        pixel_size_inv_Ang,
        accel_voltage,
        recon_size_Ang,
        probe_defocus_Ang = 0,
        probe_step_Ang = None,
        probe_position_list_Ang = None,
        rotation_deg = None,
        bilinear = True,
        normalization_min = 0.1,
        plot_result = True,
        figsize = (8,8),
        progress_bar = True,
        ):
        """
        This function initializes a ptychographic reconstruction. Currently it
        assumes the rotation has been previously found using DPC curl minimization,
        but can also be manually specified. The result plotted is the field of view,
        showing the (padded) reconstruction space and probe positions.
        
        Args:
            dataset: (DataCube)       Raw intensity measurements.

        """

        # init
        self.probe_defocus_Ang = probe_defocus_Ang
        self.probe_C1_Ang = -1.0*probe_defocus_Ang


        # Conversion of accel voltage to wavelength
        self.accel_voltage = accel_voltage
        m = 9.109383*1e-31
        e = 1.602177*1e-19
        c = 2.99792458*1e8
        h = 6.62607*1e-34
        self.wavelength = h / np.sqrt(2*m*e*accel_voltage) \
            / np.sqrt(1 + e*accel_voltage/2/m/c**2) * 10**10 # wavelength in A

        # override rotation with user-specified value
        if rotation_deg is not None:
            self.rotation_best_deg = rotation_deg
            self.rotation_best_rad = np.deg2rad(rotation_deg)

        # pixel sizes
        self.pixel_size_inv_Ang = np.array(pixel_size_inv_Ang) * np.ones((2))
        self.k_max = self.pixel_size_inv_Ang * dataset.shape[2:4] / 2.0
        self.pixel_size_Ang = 1 / (2.0 * self.k_max)

        # init field of view
        self.recon_size_pixels = (np.round(np.array(recon_size_Ang) \
            / self.pixel_size_Ang / 2.0)*2).astype('int')
        self.recon_size_Ang = self.recon_size_pixels * self.pixel_size_Ang
        self.recon_object = np.ones(self.recon_size_pixels, dtype='complex')
        self.recon_update = np.ones(self.recon_size_pixels, dtype='complex')
        # self.recon_int = np.ones(self.recon_size_pixels)

        # initialize probe positions
        x = np.arange(dataset.data.shape[0]) * probe_step_Ang
        y = np.arange(dataset.data.shape[1]) * probe_step_Ang
        x -= np.mean(x)
        y -= np.mean(y)
        ya0,xa0 = np.meshgrid(y,x)
        xa =  np.cos(self.rotation_best_rad)*xa0 + np.sin(self.rotation_best_rad)*ya0
        ya = -np.sin(self.rotation_best_rad)*xa0 + np.cos(self.rotation_best_rad)*ya0
        self.ptycho_probe_x_Ang = xa.ravel() + self.recon_size_Ang[0]/2.0
        self.ptycho_probe_y_Ang = ya.ravel() + self.recon_size_Ang[1]/2.0
        self.ptycho_probe_x_pixels = self.ptycho_probe_x_Ang / self.pixel_size_Ang[0]
        self.ptycho_probe_y_pixels = self.ptycho_probe_y_Ang / self.pixel_size_Ang[1]

        # measurement and probe coordinate system in pixels
        self.probe_size_pixels = probe.data.shape
        self.x_ind = np.round(np.arange(self.probe_size_pixels[0]) 
            - self.probe_size_pixels[0]/2).astype('int')
        self.y_ind = np.round(np.arange(self.probe_size_pixels[1]) 
            - self.probe_size_pixels[1]/2).astype('int')

        # probe coordinate system in 1/pixels, operators
        kx = np.fft.fftfreq(self.probe_size_pixels[0], d=1.0)
        ky = np.fft.fftfreq(self.probe_size_pixels[1], d=1.0)
        self.kya_pixels,self.kxa_pixels = np.meshgrid(ky,kx)
        self.kx_shift = -2j*np.pi*self.kxa_pixels
        self.ky_shift = -2j*np.pi*self.kya_pixels

        # probe coordinate system in 1/Angstroms, operators
        kx = np.fft.fftfreq(self.probe_size_pixels[0], d=self.pixel_size_Ang[0])
        ky = np.fft.fftfreq(self.probe_size_pixels[1], d=self.pixel_size_Ang[1])
        self.kya,self.kxa = np.meshgrid(ky,kx)
        self.ka2 = self.kxa**2 + self.kya**2

        # initial probe amplitude and aberrations
        ya,xa = np.meshgrid(
            np.arange(self.probe_size_pixels[1]),
            np.arange(self.probe_size_pixels[0]),
            )
        int_total = np.sum(probe.data)
        probe_x0 = np.sum(xa * probe.data) / int_total
        probe_y0 = np.sum(ya * probe.data) / int_total
        probe_shift = get_shifted_ar(
            probe.data,
            -probe_x0,
            -probe_y0,
            bilinear = bilinear,
            )
        self.probe_amp = np.sqrt(np.maximum(probe_shift,0))
        self.probe = self.probe_amp * \
            np.exp((-1j*np.pi*self.wavelength*self.ka2)*self.probe_C1_Ang)

        # measured amplitude for all probe positions
        self.amp_meas = np.zeros((
            dataset.data.shape[0]*dataset.data.shape[1],
            dataset.data.shape[2],
            dataset.data.shape[3],
            ))
        for rx,ry in tqdmnd(
            dataset.data.shape[0],
            dataset.data.shape[1],
            desc="initializing measurements",
            unit=" probes",
            disable=not progress_bar,
            ):
            meas = get_shifted_ar(
                dataset.data[rx,ry],
                -self.com_fit_x[rx,ry],
                -self.com_fit_y[rx,ry],
                bilinear = bilinear,
            )
            ind = np.ravel_multi_index((rx,ry), dataset.data.shape[0:2])
            self.amp_meas[ind] = np.sqrt(np.maximum(meas,0))

        # compute inital intensity normalization
        # initial probes
        x0 = np.round(self.ptycho_probe_x_pixels).astype('int')
        y0 = np.round(self.ptycho_probe_y_pixels).astype('int')
        dx = self.ptycho_probe_x_pixels - x0 + self.recon_size_pixels[0]/2
        dy = self.ptycho_probe_y_pixels - y0 + self.recon_size_pixels[0]/2
        psi_0 = np.fft.ifft2(
            self.probe[None,:,:] * np.exp(
            self.kx_shift[None,:,:] * dx[:,None,None] + \
            self.ky_shift[None,:,:] * dy[:,None,None]))
        # Write intensities into normalization array
        self.recon_int = np.reshape(np.bincount(
            (((y0[:,None,None] + self.y_ind[None,None,:]) %  self.recon_size_pixels[1]) + \
            ((x0[:,None,None] + self.x_ind[None,:,None]) % self.recon_size_pixels[0]) * \
            self.recon_size_pixels[1]).ravel(),
            weights = np.abs(psi_0.ravel())**2,
            minlength = np.prod(self.recon_size_pixels),
        ),self.recon_size_pixels)
        self.recon_int_norm = 1 / np.sqrt(self.recon_int**2 + \
            (normalization_min*np.max(self.recon_int))**2)

        # self.recon_int = np.reshape(np.bincount(
        #     np.ravel_multi_index((
        #         x,
        #         y),
        #         self.recon_size_pixels,
        #     ),
        #     # (((x0[:,None,None] + self.x_ind[None,:,None]) % self.recon_size_pixels[0]) + \
        #     # (((y0[:,None,None] + self.y_ind[None,None,:])) %  self.recon_size_pixels[1]) * \
        #     # self.recon_size_pixels[0]).ravel(),
        #     weights = np.abs(psi_0.ravel())**2,
        #     minlength = np.prod(self.recon_size_pixels),
        # ),self.recon_size_pixels)
        

        # test = np.ravel_multi_index(
        #     (
        #         x0[:,None,None] + self.x_ind[None,:,None],
        #         y0[:,None,None] + self.x_ind[None,:,None],
        #     ),
        #     self.amp_meas.shape)

        # x0[:,None,None] + \
        #     self.x_ind[None,:,None] + \
        #     self.y_ind[None,None,:]
        


        # plotting
        if plot_result:
            fig,ax = plt.subplots(figsize=figsize)
            ax.imshow(
                self.recon_int,
                # np.angle(self.recon_object),
                cmap='gray',
                )
            ax.scatter(
                self.ptycho_probe_y_pixels,
                self.ptycho_probe_x_pixels,
                s=5,
                color=(1,0,0,1),
                )
            plt.show()



    def ptychography_recon(
        self,
        num_iterations = 1,
        step_size = 0.95,
        plot_result = True,
        figsize = (8,8),
        progress_bar = True,
        ):
        """
        This function performs ptychographic reconstruction.
        
        Args:
            num_iterations: (int)       Number of full iterations to perform.

        """

        # init
        # probes_grad = np.zeros(self.amp_meas.shape,dtype='complex')

        # main loop
        for iter in tqdmnd(
            num_iterations,
            desc="reconstructing... ",
            unit=" iterations",
            disable=not progress_bar,
            ):
            # indexing
            # ind_2D = 
            # ind_2D = np.reshape(np.bincount(
            #     (((y0[:,None,None] + self.y_ind[None,None,:]) %  self.recon_size_pixels[1]) + \
            #     ((x0[:,None,None] + self.x_ind[None,:,None]) % self.recon_size_pixels[0]) * \
            #     self.recon_size_pixels[1]).ravel(),


            # forward projection
            x0 = np.round(self.ptycho_probe_x_pixels).astype('int')
            y0 = np.round(self.ptycho_probe_y_pixels).astype('int')
            dx = self.ptycho_probe_x_pixels - x0 + self.recon_size_pixels[0]/2
            dy = self.ptycho_probe_y_pixels - y0 + self.recon_size_pixels[0]/2
            psi_0 = np.fft.ifft2(
                self.probe[None,:,:] * np.exp(
                self.kx_shift[None,:,:] * dx[:,None,None] + \
                self.ky_shift[None,:,:] * dy[:,None,None]))
            Psi = np.fft.fft2(psi_0 * self.recon_object[
                (x0[:,None,None] + self.x_ind[None,:,None]) %  self.recon_size_pixels[0],
                (y0[:,None,None] + self.y_ind[None,None,:]) %  self.recon_size_pixels[1],
            ])

            # gradient from difference
            Psi = np.conj(psi_0) * np.fft.ifft2((
                self.amp_meas - np.abs(Psi)) * np.exp(1j*np.angle(Psi)))

            # back projection
            grad = np.reshape(np.bincount(
                (((y0[:,None,None] + self.y_ind[None,None,:]) %  self.recon_size_pixels[1]) + \
                ((x0[:,None,None] + self.x_ind[None,:,None]) % self.recon_size_pixels[0]) * \
                self.recon_size_pixels[1]).ravel(),
                weights = Psi.ravel(),
                minlength = np.prod(self.recon_size_pixels),
            ),self.recon_size_pixels) * self.recon_int_norm.astype('complex')



            # update




        # plotting
        if plot_result:
            fig,ax = plt.subplots(figsize=figsize)
            ax.imshow(
                np.angle(grad),
                # np.angle(self.recon_object),
                cmap='gray',
                )
            plt.show()





