"""
Functions for finding Braggdisks using cupy

"""

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import cupyx.scipy.fft as cufft
from time import time
import numba

from .kernels import kernels
from ...io import PointList, PointListArray
from ...utils.tqdmnd import tqdmnd


def find_Bragg_disks_CUDA(
    datacube,
    probe,
    corrPower=1,
    sigma=2,
    edgeBoundary=20,
    minRelativeIntensity=0.005,
    minAbsoluteIntensity=0.0,
    relativeToPeak=0,
    minPeakSpacing=60,
    maxNumPeaks=70,
    subpixel="multicorr",
    upsample_factor=16,
    filter_function=None,
    name="braggpeaks_raw",
    _qt_progress_bar=None,
    batching=True,
):
    """
    Finds the Bragg disks in all diffraction patterns of datacube by cross, hybrid, or
    phase correlation with probe. When hist = True, returns histogram of intensities in
    the entire datacube.

    Args:
        DP (ndarray): a diffraction pattern
        probe (ndarray): the vacuum probe template, in real space.
        corrPower (float between 0 and 1, inclusive): the cross correlation power. A
             value of 1 corresponds to a cross correaltion, and 0 corresponds to a
             phase correlation, with intermediate values giving various hybrids.
        sigma (float): the standard deviation for the gaussian smoothing applied to
             the cross correlation
        edgeBoundary (int): minimum acceptable distance from the DP edge, in pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak intensity,
            relative to the intensity of the brightest peak
        relativeToPeak (int): specifies the peak against which the minimum relative
            intensity is measured -- 0=brightest maximum. 1=next brightest, etc.
        minPeakSpacing (float): the minimum acceptable spacing between detected peaks
        maxNumPeaks (int): the maximum number of peaks to return
        subpixel (str): Whether to use subpixel fitting, and which algorithm to use.
            Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor (int): upsampling factor for subpixel fitting (only used when
             subpixel='multicorr')
        global_threshold (bool): if True, applies global threshold based on
            minGlobalIntensity and metric
        minGlobalThreshold (float): the minimum allowed peak intensity, relative to the
             selected metric (0-1), except in the case of 'manual' metric, in which the
             threshold value based on the minimum intensity that you want thresholder
             out should be set.
        metric (string): the metric used to compare intensities. 'average' compares peak
            intensity relative to the average of the maximum intensity in each
            diffraction pattern. 'max' compares peak intensity relative to the maximum
            intensity value out of all the diffraction patterns.  'median' compares peak
            intensity relative to the median of the maximum intensity peaks in each
            diffraction pattern. 'manual' Allows the user to threshold based on a
            predetermined intensity value manually determined. In this case,
            minIntensity should be an int.
        name (str): name for the returned PointListArray
        filter_function (callable): filtering function to apply to each diffraction
            pattern before peakfinding. Must be a function of only one argument (the
            diffraction pattern) and return the filtered diffraction pattern. The
            shape of the returned DP must match the shape of the probe kernel (but does
            not need to match the shape of the input diffraction pattern, e.g. the filter
            can be used to bin the diffraction pattern). If using distributed disk
            detection, the function must be able to be pickled with by dill.
        _qt_progress_bar (QProgressBar instance): used only by the GUI.
        batching (bool): Whether to batch the FFT cross correlation steps. 

    Returns:
        (PointListArray): the Bragg peak positions and correlation intensities
    """

    # Make the peaks PointListArray
    coords = [("qx", float), ("qy", float), ("intensity", float)]
    peaks = PointListArray(dtype=coords, shape=(datacube.R_Nx, datacube.R_Ny))

    # check that the filtered DP is the right size for the probe kernel:
    if filter_function:
        assert callable(filter_function), "filter_function must be callable"
    DP = (
        datacube.data[0, 0, :, :]
        if filter_function is None
        else filter_function(datacube.data[0, 0, :, :])
    )
    assert np.all(
        DP.shape == probe.shape
    ), "Probe kernel shape must match filtered DP shape"

    # Get the probe kernel FT as a cupy array
    probe_kernel_FT = cp.conj(cp.fft.fft2(cp.array(probe))).astype(cp.complex64)
    bytes_per_pattern = probe_kernel_FT.nbytes

    # get the maximal array kernel
    # if probe_kernel_FT.dtype == 'float64':
    #    get_maximal_points = kernels['maximal_pts_float64']
    # elif probe_kernel_FT.dtype == 'float32':
    #    get_maximal_points = kernels['maximal_pts_float32']
    # else:
    #    raise TypeError("Maximal kernel only valid for float32 and float64 types...")
    get_maximal_points = kernels["maximal_pts_float32"]

    if get_maximal_points.max_threads_per_block < DP.shape[1]:
        # naive blocks/threads will not work, figure out an OK distribution
        blocks = ((np.prod(DP.shape) // get_maximal_points.max_threads_per_block + 1),)
        threads = (get_maximal_points.max_threads_per_block,)
    else:
        blocks = (DP.shape[0],)
        threads = (DP.shape[1],)

    if _qt_progress_bar is not None:
        from PyQt5.QtWidgets import QApplication

    t0 = time()
    if batching:
        # compute the batch size based on available VRAM:
        max_num_bytes = cp.cuda.Device().mem_info[0]
        # use a fudge factor to leave room for the fourier transformed data
        # I have set this at 10, which results in underutilization of 
        # VRAM, because this yielded better performance in my testing
        batch_size = max_num_bytes // (bytes_per_pattern * 10)
        num_batches = datacube.R_N // batch_size + 1

        print(f"Using {num_batches} batches of {batch_size} patterns each...")

        # allocate array for batch of DPs
        batched_subcube = cp.zeros(
            (batch_size, datacube.Q_Nx, datacube.Q_Ny), dtype=cp.float32
        )

        for batch_idx in tqdmnd(
            range(num_batches), desc="Finding Bragg disks in batches", unit="batch"
        ):
            # the final batch may be smaller than the other ones:
            probes_remaining = datacube.R_N - (batch_idx * batch_size)
            this_batch_size = (
                probes_remaining if probes_remaining < batch_size else batch_size
            )

            # fill in diffraction patterns, with filtering
            for subbatch_idx in range(this_batch_size):
                patt_idx = batch_idx * batch_size + subbatch_idx
                rx, ry = np.unravel_index(patt_idx, (datacube.R_Nx, datacube.R_Ny))
                batched_subcube[subbatch_idx, :, :] = cp.array(
                    datacube.data[rx, ry, :, :]
                    if filter_function is None
                    else filter_function(datacube.data[rx, ry, :, :]),
                    dtype=cp.float32,
                )

            # Perform the FFT and multiplication by probe_kernel on the batched array
            batched_crosscorr = (
                cufft.fft2(batched_subcube, overwrite_x=True)
                * probe_kernel_FT[None, :, :]
            )

            # Iterate over the patterns in the batch and do the Bragg disk stuff
            for subbatch_idx in range(this_batch_size):
                patt_idx = batch_idx * batch_size + subbatch_idx
                rx, ry = np.unravel_index(patt_idx, (datacube.R_Nx, datacube.R_Ny))

                subFFT = batched_crosscorr[subbatch_idx]
                ccc = cp.abs(subFFT) ** corrPower * cp.exp(1j * cp.angle(subFFT))
                cc = cp.maximum(cp.real(cp.fft.ifft2(ccc)), 0)

                _find_Bragg_disks_single_DP_FK_CUDA(
                    None,
                    None,
                    ccc=ccc,
                    cc=cc,
                    corrPower=corrPower,
                    sigma=sigma,
                    edgeBoundary=edgeBoundary,
                    minRelativeIntensity=minRelativeIntensity,
                    minAbsoluteIntensity=minAbsoluteIntensity,
                    relativeToPeak=relativeToPeak,
                    minPeakSpacing=minPeakSpacing,
                    maxNumPeaks=maxNumPeaks,
                    subpixel=subpixel,
                    upsample_factor=upsample_factor,
                    filter_function=filter_function,
                    peaks=peaks.get_pointlist(rx, ry),
                    get_maximal_points=get_maximal_points,
                    blocks=blocks,
                    threads=threads,
                )

    else:
        # Loop over all diffraction patterns
        for (Rx, Ry) in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            desc="Finding Bragg Disks",
            unit="DP",
            unit_scale=True,
        ):
            if _qt_progress_bar is not None:
                _qt_progress_bar.setValue(Rx * datacube.R_Ny + Ry + 1)
                QApplication.processEvents()
            DP = datacube.data[Rx, Ry, :, :]
            _find_Bragg_disks_single_DP_FK_CUDA(
                DP,
                probe_kernel_FT,
                corrPower=corrPower,
                sigma=sigma,
                edgeBoundary=edgeBoundary,
                minRelativeIntensity=minRelativeIntensity,
                minAbsoluteIntensity=minAbsoluteIntensity,
                relativeToPeak=relativeToPeak,
                minPeakSpacing=minPeakSpacing,
                maxNumPeaks=maxNumPeaks,
                subpixel=subpixel,
                upsample_factor=upsample_factor,
                filter_function=filter_function,
                peaks=peaks.get_pointlist(Rx, Ry),
                get_maximal_points=get_maximal_points,
                blocks=blocks,
                threads=threads,
            )
    t = time() - t0
    print(
        f"Analyzed {datacube.R_N} diffraction patterns in {t//3600}h {t % 3600 // 60}m {t % 60:.2f}s\n(avg. speed {datacube.R_N/t:0.4f} patterns per second)".format()
    )
    peaks.name = name
    return peaks


def _find_Bragg_disks_single_DP_FK_CUDA(
    DP,
    probe_kernel_FT,
    corrPower=1,
    sigma=2,
    edgeBoundary=20,
    minRelativeIntensity=0.005,
    minAbsoluteIntensity=0.0,
    relativeToPeak=0,
    minPeakSpacing=60,
    maxNumPeaks=70,
    subpixel="multicorr",
    upsample_factor=16,
    filter_function=None,
    return_cc=False,
    peaks=None,
    get_maximal_points=None,
    blocks=None,
    threads=None,
    ccc=None,
    cc=None,
):
    """
    Finds the Bragg disks in DP by cross, hybrid, or phase correlation with probe_kernel_FT.

    After taking the cross/hybrid/phase correlation, a gaussian smoothing is applied
    with standard deviation sigma, and all local maxima are found. Detected peaks within
    edgeBoundary pixels of the diffraction plane edges are then discarded. Next, peaks with
    intensities less than minRelativeIntensity of the brightest peak in the correaltion are
    discarded. Then peaks which are within a distance of minPeakSpacing of their nearest neighbor
    peak are found, and in each such pair the peak with the lesser correlation intensities is
    removed. Finally, if the number of peaks remaining exceeds maxNumPeaks, only the maxNumPeaks
    peaks with the highest correlation intensity are retained.

    IMPORTANT NOTE: the argument probe_kernel_FT is related to the probe kernels generated by
    functions like get_probe_kernel() by:

            probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))

    if this function is simply passed a probe kernel, the results will not be meaningful! To run
    on a single DP while passing the real space probe kernel as an argument, use
    find_Bragg_disks_single_DP().

    Accepts:
        DP                   (ndarray) a diffraction pattern
        probe_kernel_FT      (cparray) the vacuum probe template, in Fourier space. Related to the
                             real space probe kernel by probe_kernel_FT = F(probe_kernel)*, where F
                             indicates a Fourier Transform and * indicates complex conjugation.
        corrPower            (float between 0 and 1, inclusive) the cross correlation power. A
                             value of 1 corresponds to a cross correaltion, and 0 corresponds to a
                             phase correlation, with intermediate values giving various hybrids.
        sigma                (float) the standard deviation for the gaussian smoothing applied to
                             the cross correlation
        edgeBoundary         (int) minimum acceptable distance from the DP edge, in pixels
        minRelativeIntensity (float) the minimum acceptable correlation peak intensity, relative to
                             the intensity of the relativeToPeak'th peak
        relativeToPeak       (int) specifies the peak against which the minimum relative intensity
                             is measured -- 0=brightest maximum. 1=next brightest, etc.
        minPeakSpacing       (float) the minimum acceptable spacing between detected peaks
        maxNumPeaks          (int) the maximum number of peaks to return
        subpixel             (str)          'none': no subpixel fitting
                                  (default) 'poly': polynomial interpolation of correlogram peaks
                                                    (fairly fast but not very accurate)
                                            'multicorr': uses the multicorr algorithm with
                                                        DFT upsampling
        upsample_factor      (int) upsampling factor for subpixel fitting (only used when subpixel='multicorr')
        filter_function      (callable) filtering function to apply to each diffraction pattern before peakfinding.
                             Must be a function of only one argument (the diffraction pattern) and return
                             the filtered diffraction pattern.
                             The shape of the returned DP must match the shape of the probe kernel (but does
                             not need to match the shape of the input diffraction pattern, e.g. the filter
                             can be used to bin the diffraction pattern). If using distributed disk detection,
                             the function must be able to be pickled with by dill.
        return_cc            (bool) if True, return the cross correlation
        peaks                (PointList) For internal use.
                             If peaks is None, the PointList of peak positions is created here.
                             If peaks is not None, it is the PointList that detected peaks are added
                             to, and must have the appropriate coords ('qx','qy','intensity').
        ccc and cc:         Precomputed complex and real-IFFT cross correlations. Used when called
                            in batched mode only, causing local calculation of those to be skipped

    Returns:
        peaks                (PointList) the Bragg peak positions and correlation intensities
    """

    # if we are in batching mode, cc and ccc will be provided. else, compute it
    if ccc is None:
        # Perform any prefiltering
        DP = cp.array(
            DP if filter_function is None else filter_function(DP), dtype=cp.float32
        )

        # Get the cross correlation
        if subpixel in ("none", "poly"):
            cc = get_cross_correlation_fk(DP, probe_kernel_FT, corrPower)
            ccc = None
        # for multicorr subpixel fitting, we need both the real and complex cross correlation
        else:
            ccc = get_cross_correlation_fk(
                DP, probe_kernel_FT, corrPower, returnval="fourier"
            )
            cc = cp.maximum(cp.real(cp.fft.ifft2(ccc)), 0)

    # Find the maxima
    maxima_x, maxima_y, maxima_int = get_maxima_2D(
        cc,
        sigma=sigma,
        edgeBoundary=edgeBoundary,
        minRelativeIntensity=minRelativeIntensity,
        minAbsoluteIntensity=minAbsoluteIntensity,
        relativeToPeak=relativeToPeak,
        minSpacing=minPeakSpacing,
        maxNumPeaks=maxNumPeaks,
        subpixel=subpixel,
        ar_FT=ccc,
        upsample_factor=upsample_factor,
        get_maximal_points=get_maximal_points,
        blocks=blocks,
        threads=threads,
    )

    # Make peaks PointList
    if peaks is None:
        coords = [("qx", float), ("qy", float), ("intensity", float)]
        peaks = PointList(coordinates=coords)
    peaks.add_data_by_field((maxima_x, maxima_y, maxima_int))

    if return_cc:
        return peaks, gaussian_filter(cc, sigma)
    else:
        return peaks


def get_cross_correlation_fk(ar, fourierkernel, corrPower=1, returnval="cc"):
    """
    Calculates the cross correlation of ar with fourierkernel.
    Here, fourierkernel = np.conj(np.fft.fft2(kernel)); speeds up computation when the same
    kernel is to be used for multiple cross correlations.
    corrPower specifies the correlation type, where 1 is a cross correlation, 0 is a phase
    correlation, and values in between are hybrids.

    The return value depends on the argument `returnval`:
        if return=='cc' (default), returns the real part of the cross correlation in real
        space.
        if return=='fourier', returns the output in Fourier space, before taking the
        inverse transform.
    """
    m = cp.fft.fft2(ar) * fourierkernel
    ccc = cp.abs(m) ** (corrPower) * cp.exp(1j * cp.angle(m))
    if returnval == "fourier":
        return ccc
    else:
        return cp.real(cp.fft.ifft2(ccc))


def get_maxima_2D(
    ar,
    sigma=0,
    edgeBoundary=0,
    minSpacing=0,
    minRelativeIntensity=0,
    minAbsoluteIntensity=0,
    relativeToPeak=0,
    maxNumPeaks=0,
    subpixel="poly",
    ar_FT=None,
    upsample_factor=16,
    get_maximal_points=None,
    blocks=None,
    threads=None,
):
    """
    Finds the indices where the 2D array ar is a local maximum.
    Optional parameters allow blurring of the array and filtering of the output;
    setting each of these to 0 (default) turns off these functions.

    Accepts:
        ar                      (ndarray) a 2D array
        sigma                   (float) guassian blur std to applyu to ar before finding the maxima
        edgeBoundary            (int) ignore maxima within edgeBoundary of the array edge
        minSpacing              (float) if two maxima are found within minSpacing, the dimmer one
                                is removed
        minRelativeIntensity    (float) maxima dimmer than minRelativeIntensity compared to the
                                relativeToPeak'th brightest maximum are removed
        relativeToPeak          (int) 0=brightest maximum. 1=next brightest, etc.
        maxNumPeaks             (int) return only the first maxNumPeaks maxima
        subpixel                (str)          'none': no subpixel fitting
                                     (default) 'poly': polynomial interpolation of correlogram peaks
                                                    (fairly fast but not very accurate)
                                               'multicorr': uses the multicorr algorithm with
                                                        DFT upsampling
        ar_FT                   (None or complex array) if subpixel=='multicorr' the
                                fourier transform of the image is required.  It may be
                                passed here as a complex array.  Otherwise, if ar_FT is None,
                                it is computed
        upsample_factor         (int) required iff subpixel=='multicorr'

    Returns
        maxima_x                (ndarray) x-coords of the local maximum, sorted by intensity.
        maxima_y                (ndarray) y-coords of the local maximum, sorted by intensity.
        maxima_intensity        (ndarray) intensity of the local maxima
    """

    # Get maxima
    ar = gaussian_filter(ar, sigma)
    maxima_bool = cp.zeros_like(ar, dtype=bool)
    sizex = ar.shape[0]
    sizey = ar.shape[1]
    N = sizex * sizey
    get_maximal_points(blocks, threads, (ar, maxima_bool, minAbsoluteIntensity, sizex, sizey, N))

    # Remove edges
    if edgeBoundary > 0:
        maxima_bool[:edgeBoundary, :] = False
        maxima_bool[-edgeBoundary:, :] = False
        maxima_bool[:, :edgeBoundary] = False
        maxima_bool[:, -edgeBoundary:] = False
    elif subpixel is True:
        maxima_bool[:1, :] = False
        maxima_bool[-1:, :] = False
        maxima_bool[:, :1] = False
        maxima_bool[:, -1:] = False

    # Get indices, sorted by intensity
    maxima_x, maxima_y = cp.nonzero(maxima_bool)
    maxima_x = maxima_x.get()
    maxima_y = maxima_y.get()
    dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
    maxima = np.zeros(len(maxima_x), dtype=dtype)
    maxima["x"] = maxima_x
    maxima["y"] = maxima_y

    ar = ar.get()
    maxima["intensity"] = ar[maxima_x, maxima_y]
    maxima = np.sort(maxima, order="intensity")[::-1]

    if len(maxima) > 0:
        # Remove maxima which are too close
        if minSpacing > 0:
            deletemask = np.zeros(len(maxima), dtype=bool)
            for i in range(len(maxima)):
                if deletemask[i] == False:
                    tooClose = (
                        (maxima["x"] - maxima["x"][i]) ** 2
                        + (maxima["y"] - maxima["y"][i]) ** 2
                    ) < minSpacing ** 2
                    tooClose[: i + 1] = False
                    deletemask[tooClose] = True
            maxima = np.delete(maxima, np.nonzero(deletemask)[0])

        # Remove maxima which are too dim
        if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
            deletemask = (
                maxima["intensity"] / maxima["intensity"][relativeToPeak]
                < minRelativeIntensity
            )
            maxima = np.delete(maxima, np.nonzero(deletemask)[0])

        # Remove maxima which are too dim, absolute scale
        if minAbsoluteIntensity > 0:
            deletemask = maxima["intensity"] < minAbsoluteIntensity
            maxima = np.delete(maxima, np.nonzero(deletemask)[0])

        # Remove maxima in excess of maxNumPeaks
        if maxNumPeaks is not None and maxNumPeaks > 0:
            if len(maxima) > maxNumPeaks:
                maxima = maxima[:maxNumPeaks]

        # Subpixel fitting
        # For all subpixel fitting, first fit 1D parabolas in x and y to 3 points (maximum, +/- 1 pixel)
        if subpixel != "none":
            for i in range(len(maxima)):
                Ix1_ = ar[int(maxima["x"][i]) - 1, int(maxima["y"][i])]
                Ix0 = ar[int(maxima["x"][i]), int(maxima["y"][i])]
                Ix1 = ar[int(maxima["x"][i]) + 1, int(maxima["y"][i])]
                Iy1_ = ar[int(maxima["x"][i]), int(maxima["y"][i]) - 1]
                Iy0 = ar[int(maxima["x"][i]), int(maxima["y"][i])]
                Iy1 = ar[int(maxima["x"][i]), int(maxima["y"][i]) + 1]
                deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
                deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
                maxima["x"][i] += deltax if np.abs(deltax) <= 1. else 0.
                maxima["y"][i] += deltay if np.abs(deltay) <= 1. else 0.
                maxima["intensity"][i] = linear_interpolation_2D(
                    ar, maxima["x"][i], maxima["y"][i]
                )
        # Further refinement with fourier upsampling
        if subpixel == "multicorr":
            ar_FT = cp.conj(ar_FT)

            xyShift = np.vstack((maxima["x"], maxima["y"])).T
            # we actually have to lose some precision and go down to half-pixel
            # accuracy. this could also be done by a single upsampling at factor 2
            # instead of get_maxima_2D.
            xyShift = cp.array(np.round(xyShift * 2.) / 2)

            subShift = upsampled_correlation(ar_FT, upsample_factor, xyShift).get()
            maxima["x"] = subShift[:,0]
            maxima["y"] = subShift[:,1]

    return maxima["x"], maxima["y"], maxima["intensity"]


def upsampled_correlation(imageCorr, upsampleFactor, xyShift):
    """
    Refine the correlation peak of imageCorr around xyShift by DFT upsampling.

    There are two approaches to Fourier upsampling for subpixel refinement: (a) one
    can pad an (appropriately shifted) FFT with zeros and take the inverse transform,
    or (b) one can compute the DFT by matrix multiplication using modified
    transformation matrices. The former approach is straightforward but requires
    performing the FFT algorithm (which is fast) on very large data. The latter method
    trades one speedup for a slowdown elsewhere: the matrix multiply steps are expensive
    but we operate on smaller matrices. Since we are only interested in a very small
    region of the FT around a peak of interest, we use the latter method to get
    a substantial speedup and enormous decrease in memory requirement. This
    "DFT upsampling" approach computes the transformation matrices for the matrix-
    multiply DFT around a small 1.5px wide region in the original `imageCorr`.

    Following the matrix multiply DFT we use parabolic subpixel fitting to
    get even more precision! (below 1/upsampleFactor pixels)

    NOTE: previous versions of multiCorr operated in two steps: using the zero-
    padding upsample method for a first-pass factor-2 upsampling, followed by the
    DFT upsampling (at whatever user-specified factor). I have implemented it
    differently, to better support iterating over multiple peaks. **The DFT is always
    upsampled around xyShift, which MUST be specified to HALF-PIXEL precision
    (no more, no less) to replicate the behavior of the factor-2 step.**
    (It is possible to refactor this so that peak detection is done on a Fourier
    upsampled image rather than using the parabolic subpixel and rounding as now...
    I like keeping it this way because all of the parameters and logic will be identical
    to the other subpixel methods.)


    Args:
        imageCorr (complex valued ndarray):
            Complex product of the FFTs of the two images to be registered
            i.e. m = np.fft.fft2(DP) * probe_kernel_FT;
            imageCorr = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
        upsampleFactor (int):
            Upsampling factor. Must be greater than 2. (To do upsampling
            with factor 2, use upsampleFFT, which is faster.)
        xyShift:
            Array of points around which to upsample, with shape [N-points, 2]

    Returns:
        (N_points, 2) cupy ndarray: Refined locations of the peaks in image coordinates.
    """

    xyShift = (cp.round(xyShift * upsampleFactor) / upsampleFactor).astype(cp.float32)

    globalShift = np.fix(np.ceil(upsampleFactor * 1.5) / 2)

    upsampleCenter = globalShift - upsampleFactor * xyShift

    imageCorrUpsample = dftUpsample(imageCorr, upsampleFactor, upsampleCenter).get()

    xSubShift, ySubShift = np.unravel_index(imageCorrUpsample.reshape(imageCorrUpsample.shape[0],-1).argmax(axis=1), imageCorrUpsample.shape[1:3])

    # add a subpixel shift via parabolic fitting, serially for each peak 
    for idx in range(xSubShift.shape[0]):
        try:
            icc = np.real(
                imageCorrUpsample[
                    idx,
                    xSubShift[idx] - 1 : xSubShift[idx] + 2,
                    ySubShift[idx] - 1 : ySubShift[idx] + 2,
                ]
            )
            dx = (icc[2, 1] - icc[0, 1]) / (4 * icc[1, 1] - 2 * icc[2, 1] - 2 * icc[0, 1])
            dy = (icc[1, 2] - icc[1, 0]) / (4 * icc[1, 1] - 2 * icc[1, 2] - 2 * icc[1, 0])
        except:
            dx, dy = (
                0,
                0,
            )  # this is the case when the peak is near the edge and one of the above values does not exist

        xyShift[idx] = xyShift[idx] + (cp.array([xSubShift[idx] + dx, ySubShift[idx] + dy]) - globalShift) / upsampleFactor

    return xyShift


def dftUpsample(imageCorr, upsampleFactor, xyShift):
    """
    This performs a matrix multiply DFT around a small neighboring region of the inital
    correlation peak. By using the matrix multiply DFT to do the Fourier upsampling, the
    efficiency is greatly improved. This is adapted from the subfuction dftups found in
    the dftregistration function on the Matlab File Exchange.

    https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

    The matrix multiplication DFT is from:

    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel
    image registration algorithms," Opt. Lett. 33, 156-158 (2008).
    http://www.sciencedirect.com/science/article/pii/S0045790612000778

    Args:
        imageCorr (complex valued ndarray):
            Correlation image between two images in Fourier space.
        upsampleFactor (int):
            Scalar integer of how much to upsample.
        xyShift (N_points,2) cp.ndarray, locations to upsample around:
            Coordinates in the UPSAMPLED GRID around which to upsample.
            These must be single-pixel IN THE UPSAMPLED GRID

    Returns:
        (ndarray):
            Stack of upsampled images from region around correlation peak.
    """
    N_pts = xyShift.shape[0]
    imageSize = imageCorr.shape
    pixelRadius = 1.5
    kernel_size = int(np.ceil(pixelRadius * upsampleFactor))

    colKern = cp.zeros((N_pts, imageSize[1], kernel_size),dtype=cp.complex64) # N_pts * image_size[1] * kernel_size
    rowKern = cp.zeros((N_pts, kernel_size, imageSize[0]),dtype=cp.complex64) # N_pts * kernel_size * image_size[0]

    # Fill in the DFT arrays using the CUDA kernels
    multicorr_col_kernel = kernels["multicorr_col_kernel"]
    blocks = ((np.prod(colKern.shape) // multicorr_col_kernel.max_threads_per_block + 1),)
    threads = (multicorr_col_kernel.max_threads_per_block,)
    multicorr_col_kernel(blocks,threads,(colKern, xyShift, N_pts, *imageSize, upsampleFactor))

    multicorr_row_kernel = kernels["multicorr_row_kernel"]
    blocks = ((np.prod(rowKern.shape) // multicorr_row_kernel.max_threads_per_block + 1),)
    threads = (multicorr_row_kernel.max_threads_per_block,)
    multicorr_row_kernel(blocks,threads,(rowKern, xyShift, N_pts, *imageSize, upsampleFactor))

    # Apply the DFT arrays to the correlation image
    imageUpsample = cp.real(rowKern @ imageCorr @ colKern)
    return imageUpsample

@numba.jit(nopython=True)
def linear_interpolation_2D(ar, x, y):
    """
    Calculates the 2D linear interpolation of array ar at position x,y using the four
    nearest array elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    dx = x - x0
    dy = y - y0
    return (
        (1 - dx) * (1 - dy) * ar[x0, y0]
        + (1 - dx) * dy * ar[x0, y1]
        + dx * (1 - dy) * ar[x1, y0]
        + dx * dy * ar[x1, y1]
    )
