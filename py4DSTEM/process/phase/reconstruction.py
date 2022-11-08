# Functions for phase reconstruction methods and classes.

import numpy as np
import matplotlib.pyplot as plt
from py4DSTEM.utils.tqdmnd import tqdmnd
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
        self.com_meas_x = np.zeros(dataset.data.shape[0:2])
        self.com_meas_y = np.zeros(dataset.data.shape[0:2])
        self.int_total = np.zeros(dataset.data.shape[0:2])

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



            # fig,ax = plt.subplots(figsize=figsize)
            # ax.plot(
            #     self.rotation_deg,
            #     self.rotation_div,
            #     )
            # ax.plot(
            #     self.rotation_deg,
            #     self.rotation_div_transpose,
            #     )







# class Ptychography:
#     """
#     A class which stores phase/complex reconstructions of object and probe waves.

#     """


#     def __init__(
#         self,
        
#     ):
#         """
#         Args:
            

#         """
        
#         # Initialize Crystal
#         self.positions = np.asarray(positions)  #: fractional atomic coordinates


#     # def get_amplitude(
#     #     dataset,
#     #     ):
#     #     """
#     #     Create a Crystal object from a CIF file, using pymatgen to import the CIF

#     #     Note that pymatgen typically prefers to return primitive unit cells,
#     #     which can be overridden by setting conventional_standard_structure=True.

#     #     Args:
#     #         dataset: (DataCube)     A py4DSTEM DataCube

#     #     """

#     #     1+1


# def 