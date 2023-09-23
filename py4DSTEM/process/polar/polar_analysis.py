# Analysis scripts for amorphous 4D-STEM data using polar transformations.

import numpy as np
import matplotlib.pyplot as plt

from emdfile import tqdmnd


def calculate_FEM_global(
    self,
    use_median_local=False,
    use_median_global=False,
    plot_results=False,
    figsize=(8, 4),
    returnval=False,
    returnfig=False,
    progress_bar=True,
):
    """
    Calculate fluctuation electron microscopy (FEM) statistics, including radial mean,
    variance, and normalized variance. This function uses the original FEM definitions,
    where the signal is computed pattern-by-pattern.

    TODO - finish docstrings, add median statistics.

    Parameters
    --------
    self: PolarDatacube
        Polar datacube used for measuring FEM properties.

    Returns
    --------
    radial_avg: np.array
        Average radial intensity
    radial_var: np.array
        Variance in the radial dimension


    """

    # Get the dimensioned radial bins
    self.scattering_vector = (
        self.radial_bins * self.qstep * self.calibration.get_Q_pixel_size()
    )
    self.scattering_vector_units = self.calibration.get_Q_pixel_units()

    # init radial data array
    self.radial_all = np.zeros(
        (
            self._datacube.shape[0],
            self._datacube.shape[1],
            self.polar_shape[1],
        )
    )

    # Compute the radial mean for each probe position
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Global FEM",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        self.radial_all[rx, ry] = np.mean(self.data[rx, ry], axis=0)

    self.radial_avg = np.mean(self.radial_all, axis=(0, 1))
    self.radial_var = np.mean(
        (self.radial_all - self.radial_avg[None, None]) ** 2, axis=(0, 1)
    )
    self.radial_var_norm = self.radial_var / self.radial_avg**2

    # plot results
    if plot_results:
        if returnfig:
            fig, ax = plot_FEM_global(
                self,
                figsize=figsize,
                returnfig=True,
            )
        else:
            plot_FEM_global(
                self,
                figsize=figsize,
            )

    # Return values
    if returnval:
        if returnfig:
            return self.radial_avg, self.radial_var, fig, ax
        else:
            return self.radial_avg, self.radial_var
    else:
        if returnfig:
            return fig, ax
        else:
            pass


def plot_FEM_global(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    Plotting function for the global FEM.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.scattering_vector,
        self.radial_var_norm,
    )

    ax.set_xlabel("Scattering Vector (" + self.scattering_vector_units + ")")
    ax.set_ylabel("Normalized Variance")
    ax.set_xlim((self.scattering_vector[0], self.scattering_vector[-1]))

    if returnfig:
        return fig, ax


def calculate_FEM_local(
    self,
    figsize=(8, 6),
    returnfig=False,
):
    """
    Calculate fluctuation electron microscopy (FEM) statistics, including radial mean,
    variance, and normalized variance. This function computes the radial average and variance
    for each individual probe position, which can then be mapped over the field-of-view.

    Parameters
    --------
    self: PolarDatacube
        Polar datacube used for measuring FEM properties.

    Returns
    --------
    radial_avg: np.array
        Average radial intensity
    radial_var: np.array
        Variance in the radial dimension


    """

    1 + 1
