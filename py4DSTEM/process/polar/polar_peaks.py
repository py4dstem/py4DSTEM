
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.signal import peak_prominences

# from emdfile import tqdmnd, PointList, PointListArray
from py4DSTEM import tqdmnd, PointList, PointListArray

def find_peaks_single_pattern(
    self,
    x,
    y,
    mask = None,
    bragg_peaks = None,
    bragg_mask_radius = None,
    sigma_annular_deg = 10.0,
    sigma_radial_px = 3.0,
    radial_background_subtract = True,
    radial_background_thresh = 0.25,
    num_peaks_max = 100,
    threshold_abs = 1.0,
    threshold_prom_annular = None,
    threshold_prom_radial = None,
    remove_masked_peaks = False,
    scale_sigma = 0.25,
    return_background = False,
    plot_result = True,
    plot_power_scale = 1.0,
    plot_scale_size = 100.0,
    returnfig = False,
    ):
    """
    Peak detection function for polar transformations.

    Parameters
    --------
    x: int
        x index of diffraction pattern
    y: int
        y index of diffraction pattern
    mask: np.array
        Boolean mask in Cartesian space, to filter detected peaks.
    bragg_peaks: py4DSTEM.BraggVectors
        Set of Bragg peaks used to generated a mask in Cartesian space, to filter detected peaks
    sigma_annular_deg: float
        smoothing along the annular direction in degrees, periodic
    sigma_radial_px: float
        smoothing along the radial direction in pixels, not periodic
    radial_background_subtract: bool
        If true, subtract radial background estimate
    radial_background_thresh: float
        Relative order of sorted values to use as background estimate.
        Setting to 0.5 is equivalent to median, 0.0 is min value.
    
    Returns
    --------

    """

    # if needed, generate mask from Bragg peaks
    if bragg_peaks is not None:
        mask_bragg = self._datacube.get_braggmask(
            bragg_peaks,
            x,
            y,
            radius = bragg_mask_radius,
        )
        if mask is None:
            mask = mask_bragg
        else:
            mask = np.logical_or(mask, mask_bragg)


    # Convert sigma values into units of bins
    sigma_annular = np.deg2rad(sigma_annular_deg) / self.annular_step
    sigma_radial = sigma_radial_px / self.qstep
    
    # Get transformed image and normalization array
    im_polar, im_polar_norm, norm_array, mask_bool = self.transform(
            self._datacube.data[x,y],
            mask = mask,
            returnval = 'all_zeros', 
        )
    # Change sign convention of mask
    mask_bool = np.logical_not(mask_bool)


    # Background subtraction
    if radial_background_subtract:
        sig_bg = np.zeros(im_polar.shape[1])
        for a0 in range(im_polar.shape[1]):
            if np.any(mask_bool[:,a0]):
                vals = np.sort(im_polar[mask_bool[:,a0],a0])
                ind = np.round(radial_background_thresh * (vals.shape[0]-1)).astype('int')
                sig_bg[a0] = vals[ind]
        im_polar = np.maximum(im_polar - sig_bg[None,:], 0)

    # apply smoothing and normalization
    im_polar_sm = gaussian_filter(
        im_polar * norm_array,
        sigma = (sigma_annular, sigma_radial),
        mode = ('wrap', 'nearest'),
        )
    im_mask = gaussian_filter(
        norm_array,
        sigma = (sigma_annular, sigma_radial),
        mode = ('wrap', 'nearest'),
        )
    sub = im_mask > 0.001 * np.max(im_mask)
    im_polar_sm[sub] /= im_mask[sub]

    # Find local maxima
    peaks = peak_local_max(
        im_polar_sm,
        num_peaks = num_peaks_max,
        threshold_abs = threshold_abs,
        )

    # check if peaks should be removed from the polar transformation mask
    if remove_masked_peaks:
        peaks = np.delete(
                peaks,
                mask_bool[peaks[:,0],peaks[:,1]] == False,
                axis = 0,
            )

    # peak intensity
    peaks_int = im_polar_sm[peaks[:,0],peaks[:,1]]

    # Estimate prominance of peaks, and their size in units of pixels
    peaks_prom = np.zeros((peaks.shape[0],4))
    annular_ind_center = np.atleast_1d(np.array(im_polar_sm.shape[0]//2).astype('int'))
    for a0 in range(peaks.shape[0]):

        # annular
        trace_annular = np.roll(
            np.squeeze(im_polar_sm[:,peaks[a0,1]]),
            annular_ind_center - peaks[a0,0])
        p_annular = peak_prominences(
            trace_annular, 
            annular_ind_center,
            )
        sigma_annular = scale_sigma * np.maximum(
            annular_ind_center - p_annular[1],
            p_annular[2] - annular_ind_center)

        # radial
        trace_radial  = im_polar_sm[peaks[a0,0],:]
        p_radial = peak_prominences(
            trace_radial, 
            np.atleast_1d(peaks[a0,1]),
            )
        sigma_radial = scale_sigma * np.maximum(
            peaks[a0,1] - p_radial[1],
            p_radial[2] - peaks[a0,1])

        # output
        peaks_prom[a0,0] = p_annular[0]
        peaks_prom[a0,1] = sigma_annular[0]
        peaks_prom[a0,2] = p_radial[0]
        peaks_prom[a0,3] = sigma_radial[0]

    # if needed, remove peaks using prominance criteria
    if threshold_prom_annular is not None:
        remove = peaks_prom[:,0] < threshold_prom_annular
        peaks = np.delete(
            peaks,
            remove,
            axis = 0,
            )
        peaks_int = np.delete(
            peaks_int,
            remove,
            )
        peaks_prom = np.delete(
            peaks_prom,
            remove,
            axis = 0,
            )
    if threshold_prom_radial is not None:
        remove = peaks_prom[:,2] < threshold_prom_radial
        peaks = np.delete(
            peaks,
            remove,
            axis = 0,
            )
        peaks_int = np.delete(
            peaks_int,
            remove,
            )
        peaks_prom = np.delete(
            peaks_prom,
            remove,
            axis = 0,
            )

    # Output data as a pointlist
    peaks_polar = PointList(
            np.column_stack((peaks, peaks_int, peaks_prom)).ravel().view([
            ('qt', np.float),
            ('qr', np.float),
            ('intensity', np.float),
            ('prom_annular', np.float),
            ('sigma_annular', np.float),
            ('prom_radial', np.float),
            ('sigma_radial', np.float),
        ]),
        name = 'peaks_polar')
    

    if plot_result:
        # init
        im_plot = im_polar.copy()
        im_plot = np.maximum(im_plot, 0) ** plot_power_scale

        t = np.linspace(0,2*np.pi,180+1)
        ct = np.cos(t)
        st = np.sin(t)


        fig,ax = plt.subplots(figsize=(12,6))

        ax.imshow(
            im_plot,
            cmap = 'gray',
            )

        # peaks
        ax.scatter(
            peaks_polar['qr'],
            peaks_polar['qt'],
            s = peaks_polar['intensity'] * plot_scale_size,
            marker='o',
            color = (1,0,0),
            )
        for a0 in range(peaks_polar.data.shape[0]):
            ax.plot(
                peaks_polar['qr'][a0] + st * peaks_polar['sigma_radial'][a0],
                peaks_polar['qt'][a0] + ct * peaks_polar['sigma_annular'][a0],
                linewidth = 1,
                color = 'r',
                )
            if peaks_polar['qt'][a0] - peaks_polar['sigma_annular'][a0] < 0:
                ax.plot(
                    peaks_polar['qr'][a0] + st * peaks_polar['sigma_radial'][a0],
                    peaks_polar['qt'][a0] + ct * peaks_polar['sigma_annular'][a0] + im_plot.shape[0],
                    linewidth = 1,
                    color = 'r',
                    )
            if peaks_polar['qt'][a0] + peaks_polar['sigma_annular'][a0] > im_plot.shape[0]:
                ax.plot(
                    peaks_polar['qr'][a0] + st * peaks_polar['sigma_radial'][a0],
                    peaks_polar['qt'][a0] + ct * peaks_polar['sigma_annular'][a0] - im_plot.shape[0],
                    linewidth = 1,
                    color = 'r',
                    )

        # plot appearance
        ax.set_xlim((0,im_plot.shape[1]-1))
        ax.set_ylim((im_plot.shape[0]-1,0))

    if returnfig and plot_result:
        if return_background:
            return peaks_polar, sig_bg, fig, ax
        else:
            return peaks_polar, fig, ax            
    else:
        if return_background:
            return peaks_polar, sig_bg
        else:
            return peaks_polar


def find_peaks(
    self,
    mask = None,
    bragg_peaks = None,
    bragg_mask_radius = None,
    sigma_annular_deg = 10.0,
    sigma_radial_px = 3.0,
    radial_background_subtract = True,
    radial_background_thresh = 0.25,
    num_peaks_max = 100,
    threshold_abs = 1.0,
    threshold_prom_annular = None,
    threshold_prom_radial = None,
    remove_masked_peaks = False,
    scale_sigma = 0.25,
    progress_bar = True,
    ):
    """
    Peak detection function for polar transformations. Loop through all probe positions,
    find peaks.  Store the peak positions and background signals.

    Parameters
    --------
    sigma_annular_deg: float
        smoothing along the annular direction in degrees, periodic
    sigma_radial_px: float
        smoothing along the radial direction in pixels, not periodic
    
    Returns
    --------

    """

    # init
    self.bragg_peaks = bragg_peaks
    self.peaks = PointListArray(
        dtype = [
        ('qt', '<f8'), 
        ('qr', '<f8'), 
        ('intensity', '<f8'), 
        ('prom_annular', '<f8'), 
        ('sigma_annular', '<f8'), 
        ('prom_radial', '<f8'), 
        ('sigma_radial', '<f8')],
        shape = self._datacube.Rshape,
        name = 'peaks_polardata',
        )
    self.background_radial = np.zeros((
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        self.radial_bins.shape[0],
        ))

    # Loop over probe positions
    for rx, ry in tqdmnd(
        *bragg_peaks.shape,
        desc="Finding peaks",
        unit=" images",
        disable=not progress_bar,
        ):

        polar_peaks, sig_bg = self.find_peaks_single_pattern(
            rx,
            ry,
            mask = mask,
            bragg_peaks = bragg_peaks,
            bragg_mask_radius = bragg_mask_radius,
            sigma_annular_deg = sigma_annular_deg,
            sigma_radial_px = sigma_radial_px,
            radial_background_subtract = radial_background_subtract,
            radial_background_thresh = radial_background_thresh,
            num_peaks_max = num_peaks_max,
            threshold_abs = threshold_abs,
            threshold_prom_annular = threshold_prom_annular,
            threshold_prom_radial = threshold_prom_radial,
            remove_masked_peaks = remove_masked_peaks,
            scale_sigma = scale_sigma,
            return_background = True,
            plot_result = False,
            )

        self.peaks[rx,ry] = polar_peaks
        self.background_radial[rx,ry] = sig_bg



def refine_peaks(
    self,
    radial_background_subtract = True,
    ):
    """
    Use local 2D elliptic gaussian fitting to refine the peak locations. 
    Optionally include background offset of the peaks.

    """
    pass


def plot_radial_peaks(
    self,
    returnfig = True,
    ):
    """
    Calculate and plot the total peak signal as a function of the radial coordinate.

    """
    pass


def plot_radial_background(
    self,
    returnfig = True,
    ):
    """
    Calculate and plot the mean background signal, background standard deviation.

    """
    pass


def make_orientation_histogram(
    self,
    ):
    """
    Make an orientation histogram, in order to use flowline visualization of orientation maps.
    """
    pass