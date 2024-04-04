import numpy as np
from typing import Union, Optional
from scipy.optimize import curve_fit

from py4DSTEM.process.diffraction.utils import Orientation, calc_1D_profile

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core.structure import Structure
except ImportError:
    pass


def calibrate_pixel_size(
    self,
    bragg_peaks,
    scale_pixel_size=1.0,
    bragg_k_power=1.0,
    bragg_intensity_power=1.0,
    k_min=0.0,
    k_max=None,
    k_step=0.002,
    k_broadening=0.002,
    fit_all_intensities=False,
    set_calibration_in_place=False,
    verbose=True,
    plot_result=False,
    figsize: Union[list, tuple, np.ndarray] = (12, 6),
    returnfig=False,
):
    """
    Use the calculated structure factor scattering lengths to compute 1D
    diffraction patterns, and solve the best-fit relative scaling between them.
    Returns the fit pixel size in Å^-1.

    Args:
        bragg_peaks (BraggVectors): Input Bragg vectors.
        scale_pixel_size (float): Initial guess for scaling of the existing
            pixel size If the pixel size is currently uncalibrated, this is a
            guess of the pixel size in Å^-1. If the pixel size is already
            (approximately) calibrated, this is the scaling factor to
            correct that existing calibration.
        bragg_k_power (float): Input Bragg peak intensities are multiplied by
            k**bragg_k_power to change the weighting of longer scattering vectors
        bragg_intensity_power (float): Input Bragg peak intensities are raised
            power **bragg_intensity_power.
        k_min (float): min k value for fitting range (Å^-1)
        k_max (float): max k value for fitting range (Å^-1)
        k_step (float) step size of k in fitting range (Å^-1)
        k_broadening (float): Initial guess for Gaussian broadening of simulated
            pattern (Å^-1)
        fit_all_intensities (bool): Set to true to allow all peak intensities to
            change independently. False forces a single intensity scaling for all peaks.
        set_calibration (bool): if True, set the fit pixel size to the calibration
            metadata, and calibrate bragg_peaks
        verbose (bool): Output the calibrated pixel size.
        plot_result (bool): Plot the resulting fit.
        figsize (list, tuple, np.ndarray): Figure size of the plot.
        returnfig (bool): Return handles figure and axis

    Returns
    _______



    fig, ax: handles, optional
        Figure and axis handles, if returnfig=True.

    """

    assert hasattr(self, "struct_factors"), "Compute structure factors first..."

    # Prepare experimental data
    k, int_exp = self.calculate_bragg_peak_histogram(
        bragg_peaks, bragg_k_power, bragg_intensity_power, k_min, k_max, k_step
    )

    # Perform fitting
    def fit_profile(k, *coefs):
        scale_pixel_size = coefs[0]
        k_broadening = coefs[1]
        int_scale = coefs[2:]

        int_sf = calc_1D_profile(
            k,
            self.g_vec_leng * scale_pixel_size,
            self.struct_factors_int,
            k_broadening=k_broadening,
            int_scale=int_scale,
            normalize_intensity=False,
        )
        return int_sf

    if fit_all_intensities:
        coefs = (
            scale_pixel_size,
            k_broadening,
            *tuple(np.ones(self.g_vec_leng.shape[0])),
        )
        bounds = (0.0, np.inf)
        popt, pcov = curve_fit(fit_profile, k, int_exp, p0=coefs, bounds=bounds)
    else:
        coefs = (scale_pixel_size, k_broadening, 1.0)
        bounds = (0.0, np.inf)
        popt, pcov = curve_fit(fit_profile, k, int_exp, p0=coefs, bounds=bounds)

    scale_pixel_size = popt[0]
    k_broadening = popt[1]
    int_scale = np.array(popt[2:])

    # Get the answer
    pix_size_prev = bragg_peaks.calibration.get_Q_pixel_size()
    pixel_size_new = pix_size_prev / scale_pixel_size

    # if requested, apply calibrations in place
    if set_calibration_in_place:
        bragg_peaks.calibration.set_Q_pixel_size(pixel_size_new)
        bragg_peaks.calibration.set_Q_pixel_units("A^-1")

    # Output calibrated Bragg peaks
    bragg_peaks_cali = bragg_peaks.copy()
    bragg_peaks_cali.calibration.set_Q_pixel_size(pixel_size_new)
    bragg_peaks_cali.calibration.set_Q_pixel_units("A^-1")

    # Output pixel size
    if verbose:
        print(f"Calibrated pixel size = {np.round(pixel_size_new, decimals=8)} A^-1")

    # Plotting
    if plot_result:
        if int_scale.shape[0] < self.g_vec_leng.shape[0]:
            int_scale = np.hstack(
                (int_scale, np.ones(self.g_vec_leng.shape[0] - int_scale.shape[0]))
            )
        elif int_scale.shape[0] > self.g_vec_leng.shape[0]:
            print(int_scale.shape[0])
            int_scale = int_scale[: self.g_vec_leng.shape[0]]

        if returnfig:
            fig, ax = self.plot_scattering_intensity(
                bragg_peaks=bragg_peaks_cali,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
                returnfig=True,
            )
        else:
            self.plot_scattering_intensity(
                bragg_peaks=bragg_peaks_cali,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
            )

    # return
    if returnfig and plot_result:
        return bragg_peaks_cali, (fig, ax)
    else:
        return bragg_peaks_cali


def calibrate_unit_cell(
    self,
    bragg_peaks,
    coef_index=None,
    coef_update=None,
    bragg_k_power=1.0,
    bragg_intensity_power=1.0,
    k_min=0.0,
    k_max=None,
    k_step=0.005,
    k_broadening=0.02,
    fit_all_intensities=True,
    verbose=True,
    plot_result=False,
    figsize: Union[list, tuple, np.ndarray] = (12, 6),
    returnfig=False,
):
    """
    Solve for the best fit scaling between the computed structure factors and bragg_peaks.

    Args:
        bragg_peaks (BraggVectors):         Input Bragg vectors.
        coef_index (list of ints):          List of ints that act as pointers to unit cell parameters and angles to update.
        coef_update (list of bool):         List of booleans to indicate whether or not to update the cell at
                                            that position
        bragg_k_power (float):              Input Bragg peak intensities are multiplied by k**bragg_k_power
                                            to change the weighting of longer scattering vectors
        bragg_intensity_power (float):      Input Bragg peak intensities are raised power **bragg_intensity_power.
        k_min (float):                      min k value for fitting range (Å^-1)
        k_max (float):                      max k value for fitting range (Å^-1)
        k_step (float):                     step size of k in fitting range (Å^-1)
        k_broadening (float):               Initial guess for Gaussian broadening of simulated pattern (Å^-1)
        fit_all_intensities (bool):         Set to true to allow all peak intensities to change independently
                                            False forces a single intensity scaling.
        verbose (bool):                     Output the calibrated pixel size.
        plot_result (bool):                 Plot the resulting fit.
        figsize (list, tuple, np.ndarray)   Figure size of the plot.
        returnfig (bool):                   Return handles figure and axis

    Returns:
        fig, ax (handles):                  Optional figure and axis handles, if returnfig=True.

    Details:
    User has the option to define what is allowed to update in the unit cell using the arguments
    coef_index and coef_update. Each has 6 entries, corresponding to the a, b, c, alpha, beta, gamma
    parameters of the unit cell, in this order. The coef_update argument is a list of bools specifying
    whether or not the unit cell value will be allowed to change (True) or must maintain the original
    value (False) upon fitting. The coef_index argument provides a pointer to the index in which the
    code will update to.

    For example, to update a, b, c, alpha, beta, gamma all independently of eachother, the following
    arguments should be used:
        coef_index = [0, 1, 2, 3, 4, 5]
        coef_update = [True, True, True, True, True, True,]

    The default is set to automatically define what can update in a unit cell based on the
    point group constraints. When either 'coef_index' or 'coef_update' are None, these constraints
    will be automatically pulled from the pointgroup.

    For example, the default for cubic unit cells is:
        coef_index = [0, 0, 0, 3, 3, 3]
        coef_update = [True, True, True, False, False, False]
    Which allows a, b, and c to update (True in first 3 indices of coef_update)
    but b and c update based on the value of a (0 in the 1 and 2 list entries in coef_index) such
    that a = b = c. While coef_update is False for alpha, beta, and gamma (entries 3, 4, 5), no
    updates will be made to the angles.

    The user has the option to predefine coef_index or coef_update to override defaults.  In the
    coef_update list, there must be 6 entries and each are boolean. In the coef_index list, there
    must be 6 entries, with the first 3 entries being between 0 - 2 and the last 3 entries between
    3 - 5. These act as pointers to pull the updated parameter from.

    """
    # initialize structure
    if coef_index is None or coef_update is None:
        structure = Structure(
            self.lat_real, self.numbers, self.positions, coords_are_cartesian=False
        )
        self.pointgroup = SpacegroupAnalyzer(structure)
        assert (
            self.pointgroup.get_point_group_symbol() in parameter_updates
        ), "Unrecognized pointgroup returned by pymatgen!"
        coef_index, coef_update = parameter_updates[
            self.pointgroup.get_point_group_symbol()
        ]

    # Prepare experimental data
    k, int_exp = self.calculate_bragg_peak_histogram(
        bragg_peaks, bragg_k_power, bragg_intensity_power, k_min, k_max, k_step
    )

    # Define Fitting Class
    class FitCrystal:
        def __init__(
            self,
            crystal,
            coef_index,
            coef_update,
            fit_all_intensities,
        ):
            self.coefs_init = crystal.cell
            self.hkl = crystal.hkl
            self.struct_factors_int = crystal.struct_factors_int
            self.coef_index = coef_index
            self.coef_update = coef_update

        def get_coefs(
            self,
            coefs_fit,
        ):
            coefs = np.zeros_like(coefs_fit)
            for a0 in range(6):
                if self.coef_update[a0]:
                    coefs[a0] = coefs_fit[self.coef_index[a0]]
                else:
                    coefs[a0] = self.coefs_init[a0]
            coefs[6:] = coefs_fit[6:]

            return coefs

        def fitfun(self, k, *coefs_fit):
            coefs = self.get_coefs(coefs_fit=coefs_fit)

            # Update g vector positions
            a, b, c = coefs[:3]
            alpha = np.deg2rad(coefs[3])
            beta = np.deg2rad(coefs[4])
            gamma = np.deg2rad(coefs[5])
            f = np.cos(beta) * np.cos(gamma) - np.cos(alpha)
            vol = (
                a
                * b
                * c
                * np.sqrt(
                    1
                    + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                    - np.cos(alpha) ** 2
                    - np.cos(beta) ** 2
                    - np.cos(gamma) ** 2
                )
            )
            lat_real = np.array(
                [
                    [a, 0, 0],
                    [b * np.cos(gamma), b * np.sin(gamma), 0],
                    [
                        c * np.cos(beta),
                        -c * f / np.sin(gamma),
                        vol / (a * b * np.sin(gamma)),
                    ],
                ]
            )
            # Inverse lattice, metric tensors
            metric_real = lat_real @ lat_real.T
            metric_inv = np.linalg.inv(metric_real)
            lat_inv = metric_inv @ lat_real
            g_vec_all = (self.hkl.T @ lat_inv).T
            g_vec_leng = np.linalg.norm(g_vec_all, axis=0)

            # Calculate fitted intensity profile
            k_broadening = coefs[6]
            int_scale = coefs[7:]
            int_sf = calc_1D_profile(
                k,
                g_vec_leng,
                self.struct_factors_int,
                k_broadening=k_broadening,
                int_scale=int_scale,
                normalize_intensity=False,
            )

            return int_sf

    fit_crystal = FitCrystal(
        self,
        coef_index=coef_index,
        coef_update=coef_update,
        fit_all_intensities=fit_all_intensities,
    )

    if fit_all_intensities:
        coefs = (
            *tuple(self.cell),
            k_broadening,
            *tuple(np.ones(self.g_vec_leng.shape[0])),
        )
        bounds = (0.0, np.inf)
        popt, pcov = curve_fit(
            fit_crystal.fitfun,
            k,
            int_exp,
            p0=coefs,
            bounds=bounds,
        )
    else:
        coefs = (
            *tuple(self.cell),
            k_broadening,
            1.0,
        )
        bounds = (0.0, np.inf)
        popt, pcov = curve_fit(
            fit_crystal.fitfun,
            k,
            int_exp,
            p0=coefs,
            bounds=bounds,
        )

    if verbose:
        cell_init = self.cell
    # Update crystal with new lattice parameters
    self.cell = fit_crystal.get_coefs(popt[:6])
    self.calculate_lattice()
    self.calculate_structure_factors(self.k_max)

    # Output
    if verbose:
        # Print unit cell parameters
        print("Original unit cell = " + str(cell_init))
        print("Calibrated unit cell = " + str(self.cell))

    # Plotting
    if plot_result:
        k_broadening = popt[6]
        int_scale = popt[7:]
        if int_scale.shape[0] < self.g_vec_leng.shape[0]:
            int_scale = np.hstack(
                (int_scale, np.ones(self.g_vec_leng.shape[0] - int_scale.shape[0]))
            )
        elif int_scale.shape[0] > self.g_vec_leng.shape[0]:
            print(int_scale.shape[0])
            int_scale = int_scale[: self.g_vec_leng.shape[0]]

        if returnfig:
            fig, ax = self.plot_scattering_intensity(
                bragg_peaks=bragg_peaks,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
                returnfig=True,
            )
        else:
            self.plot_scattering_intensity(
                bragg_peaks=bragg_peaks,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
            )

    if returnfig and plot_result:
        return fig, ax
    else:
        return


# coef_index and coef_update sets for the fit_unit_cell function, in the order:
#   [coef_index, coef_update]
parameter_updates = {
    "1": [[0, 1, 2, 3, 4, 5], [True, True, True, True, True, True]],  # Triclinic
    "-1": [[0, 1, 2, 3, 4, 5], [True, True, True, True, True, True]],  # Triclinic
    "2": [[0, 1, 2, 3, 4, 3], [True, True, True, False, True, False]],  # Monoclinic
    "m": [[0, 1, 2, 3, 4, 3], [True, True, True, False, True, False]],  # Monoclinic
    "2/m": [[0, 1, 2, 3, 4, 3], [True, True, True, False, True, False]],  # Monoclinic
    "222": [
        [0, 1, 2, 3, 3, 3],
        [True, True, True, False, False, False],
    ],  # Orthorhombic
    "mm2": [
        [0, 1, 2, 3, 3, 3],
        [True, True, True, False, False, False],
    ],  # Orthorhombic
    "mmm": [
        [0, 1, 2, 3, 3, 3],
        [True, True, True, False, False, False],
    ],  # Orthorhombic
    "4": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "-4": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "4/m": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "422": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "4mm": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "-42m": [[0, 0, 2, 3, 3, 3], [True, True, True, False, False, False]],  # Tetragonal
    "4/mmm": [
        [0, 0, 2, 3, 3, 3],
        [True, True, True, False, False, False],
    ],  # Tetragonal
    "3": [[0, 0, 0, 3, 3, 3], [True, True, True, True, True, True]],  # Trigonal
    "-3": [[0, 0, 0, 3, 3, 3], [True, True, True, True, True, True]],  # Trigonal
    "32": [[0, 0, 0, 3, 3, 3], [True, True, True, True, True, True]],  # Trigonal
    "3m": [[0, 0, 0, 3, 3, 3], [True, True, True, True, True, True]],  # Trigonal
    "-3m": [[0, 0, 0, 3, 3, 3], [True, True, True, True, True, True]],  # Trigonal
    "6": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "-6": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "6/m": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "622": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "6mm": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "-6m2": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "6/mmm": [[0, 0, 2, 3, 3, 5], [True, True, True, False, False, True]],  # Hexagonal
    "23": [[0, 0, 0, 3, 3, 3], [True, True, True, False, False, False]],  # Cubic
    "m-3": [[0, 0, 0, 3, 3, 3], [True, True, True, False, False, False]],  # Cubic
    "432": [[0, 0, 0, 3, 3, 3], [True, True, True, False, False, False]],  # Cubic
    "-43m": [[0, 0, 0, 3, 3, 3], [True, True, True, False, False, False]],  # Cubic
    "m-3m": [[0, 0, 0, 3, 3, 3], [True, True, True, False, False, False]],  # Cubic
}
