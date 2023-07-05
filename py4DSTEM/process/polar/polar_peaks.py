
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import peak_prominences
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

# from emdfile import tqdmnd, PointList, PointListArray
from py4DSTEM import tqdmnd, PointList, PointListArray
from py4DSTEM.process.fit import polar_twofold_gaussian_2D, polar_twofold_gaussian_2D_background

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
        sig_bg_mask = np.sum(mask_bool, axis=0) >= (im_polar.shape[0]//2)
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
            return peaks_polar, sig_bg, sig_bg_mask, fig, ax
        else:
            return peaks_polar, fig, ax            
    else:
        if return_background:
            return peaks_polar, sig_bg, sig_bg_mask
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
    self.bragg_mask_radius = bragg_mask_radius
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
    self.background_radial_mask = np.zeros((
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        self.radial_bins.shape[0],
        ), dtype='bool')

    # Loop over probe positions
    for rx, ry in tqdmnd(
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        desc="Finding peaks ",
        unit=" images",
        disable=not progress_bar,
        ):

        polar_peaks, sig_bg, sig_bg_mask = self.find_peaks_single_pattern(
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
        self.background_radial_mask[rx,ry] = sig_bg_mask


def refine_peaks(
    self,
    mask = None,
    radial_background_subtract = True,
    reset_fits_to_init_positions = False,
    fit_range_sigma_annular = 2.0,
    fit_range_sigma_radial = 2.0,
    min_num_pixels_fit = 10,
    progress_bar = True,
    ):
    """
    Use local 2D elliptic gaussian fitting to refine the peak locations. 
    Optionally include background offset of the peaks.

    Parameters
    --------
    mask: np.array
        Mask image to apply to all images
    radial_background_subtract: bool
        Subtract radial background before fitting
    reset_init_positions: bool
        Use the initial peak parameters for fitting
    fit_range_sigma_annular: float
        Fit range in annular direction, in terms of the current sigma_annular
    fit_range_sigma_radial: float
        Fit range in radial direction, in terms of the current sigma_radial
    min_num_pixels_fit: int
        Minimum number of pixels to perform fitting
    progress_bar: bool
        Enable progress bar

    Returns
    --------

    """
    
    # See if the intial detected peaks have been saved yet, copy them if needed
    if not hasattr(self, "peaks_init"):
        self.peaks_init = self.peaks.copy()

    # coordinate scaling
    t_step = self._annular_step
    q_step = self._radial_step

    # Coordinate array for the fit
    qq,tt = np.meshgrid(
        self.qq,
        self.tt,
        )
    tq = np.vstack([
        tt.ravel(),
        qq.ravel(),
    ])

    # Loop over probe positions
    for rx, ry in tqdmnd(
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        desc="Refining peaks ",
        unit=" images",
        disable=not progress_bar,
        ):
        # mask
        # if needed, generate mask from Bragg peaks
        if self.bragg_peaks is not None:
            mask_bragg = self._datacube.get_braggmask(
                self.bragg_peaks,
                rx,
                ry,
                radius = self.bragg_mask_radius,
            )
            if mask is None:
                mask_fit = mask_bragg
            else:
                mask_fit = np.logical_or(mask, mask_bragg)


        # Get polar image
        im_polar, im_polar_norm, norm_array, mask_bool = self.transform(
                self._datacube.data[rx,ry],
                mask = mask_fit,
                returnval = 'all_zeros', 
            )
        # Change sign convention of mask
        mask_bool = np.logical_not(mask_bool)
        # Background subtraction
        if radial_background_subtract:
            sig_bg = self.background_radial[rx,ry]
            im_polar = np.maximum(im_polar - sig_bg[None,:], 0)

        # initial peak positions
        if reset_fits_to_init_positions:
            p = self.peaks_init[rx,ry]
        else:
            p = self.peaks[rx,ry]

        # loop over peaks
        for a0 in range(p.data.shape[0]):

            if radial_background_subtract:
                # initial parameters
                p0 = [
                    p['intensity'][a0],
                    p['qt'][a0] * t_step,
                    p['qr'][a0] * q_step,
                    p['sigma_annular'][a0] * t_step,
                    p['sigma_radial'][a0] * q_step,
                    ]

                # Mask around peak for fitting
                dt = np.mod(tt - p0[1] + np.pi/2, np.pi) - np.pi/2
                mask_peak = np.logical_and(mask_bool,
                    dt**2/(fit_range_sigma_annular*p0[3])**2 \
                    + (qq-p0[2])**2/(fit_range_sigma_radial*p0[4])**2 <= 1)

                if np.sum(mask_peak) > min_num_pixels_fit:
                    try:
                        # perform fitting
                        p0, pcov = curve_fit(
                            polar_twofold_gaussian_2D, 
                            tq[:,mask_peak.ravel()], 
                            im_polar[mask_peak], 
                            p0 = p0,
                            # bounds = bounds,
                            )

                        # Output parameters
                        self.peaks[rx,ry]['intensity'][a0] = p0[0]
                        self.peaks[rx,ry]['qt'][a0] = p0[1] / t_step
                        self.peaks[rx,ry]['qr'][a0] = p0[2] / q_step
                        self.peaks[rx,ry]['sigma_annular'][a0] = p0[3] / t_step
                        self.peaks[rx,ry]['sigma_radial'][a0] = p0[4] / q_step
                    
                    except:
                        pass

            else:
                # initial parameters
                p0 = [
                    p['intensity'][a0],
                    p['qt'][a0] * t_step,
                    p['qr'][a0] * q_step,
                    p['sigma_annular'][a0] * t_step,
                    p['sigma_radial'][a0] * q_step,
                    0,
                    ]

                # Mask around peak for fitting
                dt = np.mod(tt - p0[1] + np.pi/2, np.pi) - np.pi/2
                mask_peak = np.logical_and(mask_bool,
                    dt**2/(fit_range_sigma_annular*p0[3])**2 \
                    + (qq-p0[2])**2/(fit_range_sigma_radial*p0[4])**2 <= 1)

                if np.sum(mask_peak) > min_num_pixels_fit:
                    try:
                        # perform fitting
                        p0, pcov = curve_fit(
                            polar_twofold_gaussian_2D_background, 
                            tq[:,mask_peak.ravel()], 
                            im_polar[mask_peak], 
                            p0 = p0,
                            # bounds = bounds,
                            )

                        # Output parameters
                        self.peaks[rx,ry]['intensity'][a0] = p0[0]
                        self.peaks[rx,ry]['qt'][a0] = p0[1] / t_step
                        self.peaks[rx,ry]['qr'][a0] = p0[2] / q_step
                        self.peaks[rx,ry]['sigma_annular'][a0] = p0[3] / t_step
                        self.peaks[rx,ry]['sigma_radial'][a0] = p0[4] / q_step

                    except:
                        pass


def plot_radial_peaks(
    self,
    qmin = None,
    qmax = None,
    qstep = None,
    figsize = (8,4),
    returnfig = False,
    ):
    """
    Calculate and plot the total peak signal as a function of the radial coordinate.

    """
    
    # Get all peak data
    vects = np.concatenate(
        [self.peaks[i,j].data for i in range(self._datacube.Rshape[0]) for j in range(self._datacube.Rshape[1])])
    qr = vects['qr'] * self._radial_step 
    intensity = vects['intensity']

    # bins
    if qmin is None:
        qmin = self.qq[0]
    if qmax is None:
        qmax = self.qq[-1]
    if qstep is None:
        qstep = self.qq[1] - self.qq[0]
    q_bins = np.arange(qmin,qmax,qstep)
    q_num = q_bins.shape[0]

    # histogram
    q_ind = (qr - qmin) / qstep
    qf = np.floor(q_ind).astype("int")
    dq = q_ind - qf

    sub = np.logical_and(qf >= 0, qf < q_num)
    int_peaks = np.bincount(
        np.floor(q_ind[sub]).astype("int"),
        weights=(1 - dq[sub]) * intensity[sub],
        minlength=q_num,
    )
    sub = np.logical_and(q_ind >= -1, q_ind < q_num - 1)
    int_peaks += np.bincount(
        np.floor(q_ind[sub] + 1).astype("int"),
        weights=dq[sub] * intensity[sub],
        minlength=q_num,
    )


    # plotting
    fig,ax = plt.subplots(figsize = figsize)
    ax.plot(
        q_bins,
        int_peaks,
        color = 'r',
        linewidth = 2,
        )
    ax.set_xlim((q_bins[0],q_bins[-1]))
    ax.set_xlabel(
        'Scattering Angle (' + self.calibration.get_Q_pixel_units() +')',
        fontsize = 14,
        )
    ax.set_ylabel(
        'Total Peak Signal',
        fontsize = 14,
        )

    if returnfig:
        return fig,ax




def plot_radial_background(
    self,
    figsize = (8,4),
    returnfig = False,
    ):
    """
    Calculate and plot the mean background signal, background standard deviation.

    """
    
    # mean
    self.background_radial_mean = np.sum(
        self.background_radial * self.background_radial_mask,
        axis=(0,1))
    background_radial_mean_norm = np.sum(
        self.background_radial_mask,
        axis=(0,1))
    self.background_mask = \
        background_radial_mean_norm > (np.max(background_radial_mean_norm)*0.05)
    self.background_radial_mean[self.background_mask] \
        /= background_radial_mean_norm[self.background_mask]
    self.background_radial_mean[np.logical_not(self.background_mask)] = 0

    # variance and standard deviation
    self.background_radial_var = np.sum(
        (self.background_radial - self.background_radial_mean[None,None,:])**2 \
        * self.background_radial_mask,
        axis=(0,1))
    self.background_radial_var[self.background_mask] \
        /= self.background_radial_var[self.background_mask]
    self.background_radial_var[np.logical_not(self.background_mask)] = 0
    self.background_radial_std = np.sqrt(self.background_radial_var)


    fig,ax = plt.subplots(figsize = figsize)
    ax.fill_between(
        self.qq[self.background_mask], 
        self.background_radial_mean[self.background_mask] \
        - self.background_radial_std[self.background_mask], 
        self.background_radial_mean[self.background_mask] \
        + self.background_radial_std[self.background_mask], 
        color = 'r',
        alpha=0.2,
        )
    ax.plot(
        self.qq[self.background_mask],
        self.background_radial_mean[self.background_mask],
        color = 'r',
        linewidth = 2,
        )
    ax.set_xlim((
        self.qq[0],
        self.qq[-1]))
    ax.set_xlabel(
        'Scattering Angle (' + self.calibration.get_Q_pixel_units() +')',
        fontsize = 14,
        )
    ax.set_ylabel(
        'Background Signal',
        fontsize = 14,
        )

    if returnfig:
        return fig,ax


def make_orientation_histogram(
    self,
    radial_ranges: np.ndarray = None,
    orientation_flip_sign: bool = False,
    orientation_offset_degrees: float = 0.0,
    orientation_separate_bins: bool = False,
    upsample_factor: float = 4.0,
    theta_step_deg: float = None,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    sigma_theta: float = 3.0,
    normalize_intensity_image: bool = False,
    normalize_intensity_stack: bool = True,
    progress_bar: bool = True,
    ):
    """
    Make an orientation histogram, in order to use flowline visualization of orientation maps.
    Use peaks attached to polardatacube.

    NOTE - currently assumes two fold rotation symmetry
    TODO - add support for non two fold symmetry polardatacube

    Args:
        radial_ranges (np array):           Size (N x 2) array for N radial bins, or (2,) for a single bin.
        orientation_flip_sign (bool):       Flip the direction of theta
        orientation_offset_degrees (float): Offset for orientation angles
        orientation_separate_bins (bool):   whether to place multiple angles into multiple radial bins.
        upsample_factor (float):            Upsample factor
        theta_step_deg (float):             Step size along annular direction in degrees
        sigma_x (float):                    Smoothing in x direction before upsample
        sigma_y (float):                    Smoothing in x direction before upsample
        sigma_theta (float):                Smoothing in annular direction (units of bins, periodic)
        normalize_intensity_image (bool):   Normalize to max peak intensity = 1, per image
        normalize_intensity_stack (bool):   Normalize to max peak intensity = 1, all images
        progress_bar (bool):                Enable progress bar

    Returns:
        orient_hist (array):                4D array containing Bragg peak intensity histogram 
                                            [radial_bin x_probe y_probe theta]
    """

    # coordinates
    if theta_step_deg is None:
        # Get angles from polardatacube
        theta = self.tt
    else:
        theta = np.arange(0,180,theta_step_deg) * np.pi / 180.0
    dtheta = theta[1] - theta[0]
    dtheta_deg = dtheta * 180 / np.pi
    num_theta_bins = np.size(theta)

    # Input bins
    radial_ranges = np.array(radial_ranges)
    if radial_ranges.ndim == 1:
        radial_ranges = radial_ranges[None,:]
    radial_ranges_2 = radial_ranges**2
    num_radii = radial_ranges.shape[0]
    size_input = self._datacube.shape[0:2]

    # Output size
    size_output = np.round(np.array(size_input).astype('float') * upsample_factor).astype('int')

    # output init
    orient_hist = np.zeros([
        num_radii,
        size_output[0],
        size_output[1],
        num_theta_bins])

    # Loop over all probe positions
    for a0 in range(num_radii):
        t = "Generating histogram " + str(a0)
        # for rx, ry in tqdmnd(
        #         *bragg_peaks.shape, desc=t,unit=" probe positions", disable=not progress_bar
        #     ):
        for rx, ry in tqdmnd(
                *size_input, 
                desc=t,
                unit=" probe positions", 
                disable=not progress_bar
            ):
            x = (rx + 0.5)*upsample_factor - 0.5
            y = (ry + 0.5)*upsample_factor - 0.5
            x = np.clip(x,0,size_output[0]-2)
            y = np.clip(y,0,size_output[1]-2)

            xF = np.floor(x).astype('int')
            yF = np.floor(y).astype('int')
            dx = x - xF
            dy = y - yF

            add_data = False

            q = self.peaks[rx,ry]['qr'] * self._radial_step
            r2 = q**2
            sub = np.logical_and(r2 >= radial_ranges_2[a0,0], r2 < radial_ranges_2[a0,1])                
            if np.any(sub):
                add_data = True
                intensity = self.peaks[rx,ry]['intensity'][sub]

                # Angles of all peaks
                theta = self.peaks[rx,ry]['qt'][sub] * self._annular_step
                if orientation_flip_sign:
                    theta *= -1
                theta += orientation_offset_degrees

                t = theta / dtheta

            if add_data:
                tF = np.floor(t).astype('int')
                dt = t - tF

                orient_hist[a0,xF  ,yF  ,:] = orient_hist[a0,xF  ,yF  ,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(1-dx)*(1-dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[a0,xF  ,yF  ,:] = orient_hist[a0,xF  ,yF  ,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(1-dx)*(1-dy)*(  dt)*intensity,minlength=num_theta_bins)

                orient_hist[a0,xF+1,yF  ,:] = orient_hist[a0,xF+1,yF  ,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(  dx)*(1-dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[a0,xF+1,yF  ,:] = orient_hist[a0,xF+1,yF  ,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(  dx)*(1-dy)*(  dt)*intensity,minlength=num_theta_bins)
 
                orient_hist[a0,xF  ,yF+1,:] = orient_hist[a0,xF  ,yF+1,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(1-dx)*(  dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[a0,xF  ,yF+1,:] = orient_hist[a0,xF  ,yF+1,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(1-dx)*(  dy)*(  dt)*intensity,minlength=num_theta_bins)

                orient_hist[a0,xF+1,yF+1,:] = orient_hist[a0,xF+1,yF+1,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(  dx)*(  dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[a0,xF+1,yF+1,:] = orient_hist[a0,xF+1,yF+1,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(  dx)*(  dy)*(  dt)*intensity,minlength=num_theta_bins)           

    # smoothing / interpolation
    if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
        if num_radii > 1:
            print('Interpolating orientation matrices ...', end='')
        else:
            print('Interpolating orientation matrix ...', end='')            
        if sigma_x is not None and sigma_x > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,sigma_x*upsample_factor,
                mode='nearest',
                axis=1,
                truncate=3.0)
        if sigma_y is not None and sigma_y > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,sigma_y*upsample_factor,
                mode='nearest',
                axis=2,
                truncate=3.0)
        if sigma_theta is not None and sigma_theta > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,sigma_theta/dtheta_deg,
                mode='wrap',
                axis=3,
                truncate=2.0)
        print(' done.')

    # normalization
    if normalize_intensity_stack is True:
            orient_hist = orient_hist / np.max(orient_hist)
    elif normalize_intensity_image is True:
        for a0 in range(num_radii):
            orient_hist[a0,:,:,:] = orient_hist[a0,:,:,:] / np.max(orient_hist[a0,:,:,:])

    return orient_hist