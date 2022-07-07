# Functions for finding Bragg scattering by cross correlative template matching
# with a vacuum probe.

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ...io.datastructure.py4dstem import DataCube, QPoints, BraggVectors
from ..utils.multicorr import upsampled_correlation
from ...tqdmnd import tqdmnd





def find_Bragg_disks(
    data,
    template,

    filter_function = None,

    corrPower = 1,
    sigma = 2,
    subpixel = 'multicorr',
    upsample_factor = 16,

    minAbsoluteIntensity = 0,
    minRelativeIntensity = 0.005,
    relativeToPeak = 0,
    minPeakSpacing = 60,
    edgeBoundary = 20,
    maxNumPeaks = 70,

    CUDA = False,
    CUDA_batched = True,
    distributed = None,

    _qt_progress_bar = None,
    ):
    """
    Finds the Bragg disks in the diffraction patterns represented by `data` by
    cross/phase correlatin with `template`.

    Behavior depends on `data`. If it is

        - a DataCube: runs on all its diffraction patterns, and returns a
            BraggVectors instance
        - a 2D array: runs on this array, and returns a QPoints instance
        - a 3D array: runs slice the ar[i,:,:] slices of this array, and returns
            a len(ar.shape[0]) list of QPoints instances.
        - a 3-tuple (DataCube, rx, ry), for numbers or length-N arrays (rx,ry):
            runs on the diffraction patterns in DataCube at positions (rx,ry),
            and returns a instance or length N list of instances of QPoints

    For disk detection on a full DataCube, the calculation can be performed
    on the CPU, GPU or a cluster. By default the CPU is used.  If `CUDA` is set
    to True, tries to use the GPU.  If `CUDA_batched` is also set to True,
    batches the FFT/IFFT computations on the GPU. For distribution to a cluster,
    distributed must be set to a dictionary, with contents describing how
    distributed processing should be performed - see below for details.


    For each diffraction pattern, the algorithm works in 4 steps:

    (1) any pre-processing is performed to the diffraction image. This is
        accomplished by passing a callable function to the argument
        `filter_function`. If `filter_function` is None, this is skipped.
    (2) the diffraction image is cross correlated with the template.
        Phase/hybrid correlations can be used instead by setting the
        `corrPower` argument. Cross correlation can be skipped entirely,
        and the subsequent steps performed directly on the diffraction
        image instead of the cross correlation, by passing None to
        `template`.
    (3) the maxima of the cross correlation are located and their
        positions and intensities stored. The cross correlation may be
        passed through a gaussian filter first by passing the `sigma`
        argument. The method for maximum detection can be set with
        the `subpixel` parameter. Options, from something like fastest/least
        precise to slowest/most precise are 'pixel', 'poly', and 'multicorr'.
    (4) filtering is applied to remove untrusted or undesired positive counts,
        based on their intensity (`minRelativeIntensity`,`relativeToPeak`,
        `minAbsoluteIntensity`) their proximity to one another or the
        image edge (`minPeakSpacing`, `edgeBoundary`), and the total
        number of peaks per pattern (`maxNumPeaks`).


    Args:
        data (variable): see above
        template (2D array): the vacuum probe template, in real space. For
            Probe instances, this is `probe.kernel`.  If None, does not perform
            a cross correlation.
        filter_function (callable): filtering function to apply to each
            diffraction pattern before peakfinding. Must be a function of only
            one argument (the diffraction pattern) and return the filtered
            diffraction pattern. The shape of the returned DP must match the
            shape of the probe kernel (but does not need to match the shape of
            the input diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        corrPower (float between 0 and 1, inclusive): the cross correlation
            power. A value of 1 corresponds to a cross correaltion, 0
            corresponds to a phase correlation, and intermediate values giving
            hybrid correlations.
        sigma (float): if >0, a gaussian smoothing filter with this standard
            deviation is applied to the cross correlation before maxima are
            detected
        subpixel (str): Whether to use subpixel fitting, and which algorithm to
            use. Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor (int): upsampling factor for subpixel fitting (only used
            when subpixel='multicorr')
        minAbsoluteIntensity (float): the minimum acceptable correlation peak
            intensity, on an absolute scale
        minRelativeIntensity (float): the minimum acceptable correlation peak
            intensity, relative to the intensity of the brightest peak
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (float): the minimum acceptable spacing between detected
            peaks
        edgeBoundary (int): minimum acceptable distance for detected peaks from
            the diffraction image edge, in pixels.
        maxNumPeaks (int): the maximum number of peaks to return
        CUDA (bool): If True, import cupy and use an NVIDIA GPU to perform disk
            detection
        CUDA_batched (bool): If True, and CUDA is selected, the FFT and IFFT
            steps of disk detection are performed in batches to better utilize
            GPU resources.
        distributed (dict): contains information for parallel processing using an
            IPyParallel or Dask distributed cluster.  Valid keys are:
                * ipyparallel (dict):
                * client_file (str): path to client json for connecting to your
                    existing IPyParallel cluster
                * dask (dict): client (object): a dask client that connects to
                    your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): defaults to the working directory during
                    processing
            if distributed is None, which is the default, processing will be in
            serial
        _qt_progress_bar (QProgressBar instance): used only by the GUI for serial
            execution

    Returns:
        (variable): the Bragg peak positions and correlation intensities. If
            `data` is:
                - a DataCube, returns a BraggVectors instance
                - a 2D array, returns a QPoints instance
                - a 3D array, returns a list of QPoints instances
                - a (DataCube,rx,ry) 3-tuple, returns a list of QPoints
                    instances
    """

    # parse args

    # `data` type
    if isinstance(data, DataCube):
        mode = 'datacube'
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            mode = 'dp'
        elif data.ndim == 3:
            mode = 'dp_stack'
        else:
            er = f"if `data` is an array, must be 2- or 3-D, not {data.ndim}-D"
            raise Exception(er)
    else:
        try:
            dc,rx,ry = data[0],data[1],data[2]
            data = dc.data[rx,ry,:,:]
            mode = 'dp_stack'
        except:
            er = f"entry {data} for `data` could not be parsed"
            raise Exception(er)

    # CPU/GPU/cluster
    if mode == 'datacube':
        if distributed is None and CUDA == False:
            mode = 'dc_CPU'
        elif distributed is None and CUDA == True:
            if CUDA_batched == False:
                mode = 'dc_GPU'
            else:
                mode = 'dc_GPU_batched'
        else:
            x = _parse_distributed(distributed)
            connect, data_file, cluster_path, distributed_mode = x
            if distributed_mode == 'dask':
                mode = 'dc_dask'
            elif distributed_mode == 'ipyparallel':
                mode = 'dc_ipyparallel'
            else:
                er = f"unrecognized distributed mode {distributed_mode}"
                raise Exception(er)


    # select a function
    fns = _get_function_dictionary()
    fn = fns[mode]


    # prepare kwargs
    kws = {} if distributed is None else {'distributed':distributed}
    if _qt_progress_bar is not None:
        kws['_qt_progress_bar'] = _qt_progress_bar


    # run and return
    ans = fn(
        data,
        template,
        filter_function = filter_function,
        corrPower = corrPower,
        sigma = sigma,
        subpixel = subpixel,
        upsample_factor = upsample_factor,
        minAbsoluteIntensity = minAbsoluteIntensity,
        minRelativeIntensity = minRelativeIntensity,
        relativeToPeak = relativeToPeak,
        minPeakSpacing = minPeakSpacing,
        edgeBoundary = edgeBoundary,
        maxNumPeaks = maxNumPeaks,
        **kws
    )
    return ans



def _get_function_dictionary():

    d = {
        "dp" : _find_Bragg_disks_single,
        "dp_stack" : _find_Bragg_disks_stack,
        "dc_CPU" : _find_Bragg_disks_CPU,
        "dc_GPU" : _find_Bragg_disks_CUDA_unbatched,
        "dc_GPU_batched" : _find_Bragg_disks_CUDA_batched,
        "dc_dask" : _find_Bragg_disks_dask,
        "dc_ipyparallel" : _find_Bragg_disks_ipp,
    }

    return d





# Single diffraction pattern


def _find_Bragg_disks_single(
    DP,
    template,
    filter_function = None,
    corrPower = 1,
    sigma = 2,
    subpixel = 'poly',
    upsample_factor = 16,
    minAbsoluteIntensity = 0,
    minRelativeIntensity = 0,
    relativeToPeak = 0,
    minPeakSpacing = 0,
    edgeBoundary = 1,
    maxNumPeaks = 100,
    _return_cc = False,
    _template_space = 'real'
    ):


   # apply filter function
    er = "filter_function must be callable"
    if filter_function: assert callable(filter_function), er
    DP = DP if filter_function is None else filter_function(DP)

    # check for a template
    if template is None:
        cc = DP
    else:


        # fourier transform the template
        assert _template_space in ('real','fourier')
        if _template_space == 'real':
            template_FT = np.conj(np.fft.fft2(template))
        else:
            template_FT = template


        # Compute cross correlation
        _returnval = 'fourier' if subpixel == 'multicorr' else 'real'
        cc = _get_cross_correlation_FT(
            DP,
            template_FT,
            corrPower,
            _returnval
        )


    # Get maxima
    maxima = _get_maxima_2D(
        cc,
        subpixel = subpixel,
        upsample_factor = upsample_factor,
        minAbsoluteIntensity = minAbsoluteIntensity,
        minRelativeIntensity = minRelativeIntensity,
        relativeToPeak = relativeToPeak,
        minSpacing = minPeakSpacing,
        edgeBoundary = edgeBoundary,
        maxNumPeaks = maxNumPeaks,
    )

    # Wrap as QPoints instance
    maxima = QPoints( maxima )


    # Return
    if _return_cc is True:
        return maxima, cc
    return maxima





def _get_cross_correlation_FT(
    DP,
    template_FT,
    corrPower = 1,
    _returnval = 'real'
    ):
    """
    if _returnval is 'real', returns the real-valued cross-correlation.
    otherwise, returns the complex valued result.
    """

    m = np.fft.fft2(DP) * template_FT
    cc = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
    if _returnval == 'real':
        cc = np.maximum(np.real(np.fft.ifft2(cc)),0)
    return cc



def _get_maxima_2D(
    ar,
    subpixel = 'poly',
    upsample_factor = 16,
    sigma=0,
    minAbsoluteIntensity=0,
    minRelativeIntensity=0,
    relativeToPeak=0,
    minSpacing=0,
    edgeBoundary=1,
    maxNumPeaks=1,
    _ar_FT=None,
    ):
    """
    Finds the maximal points of a 2D array.

    Args:
        ar (array) the 2D array
        subpixel (str): specifies the subpixel resolution algorithm to use.
            must be in ('pixel','poly','multicorr'), which correspond
            to pixel resolution, subpixel resolution by fitting a
            parabola, and subpixel resultion by Fourier upsampling.
        upsample_factor: the upsampling factor for the 'multicorr'
            algorithm
        sigma: if >0, applies a gaussian filter
        maxNumPeaks: the maximum number of maxima to return
        minAbsoluteIntensity, minRelativeIntensity, relativeToPeak,
            minSpacing, edgeBoundary, maxNumPeaks: filtering applied
            after maximum detection and before subpixel refinement
        _ar_FT (complex array) if 'multicorr' is used and this is not
            None, uses this argument as the Fourier transform of `ar`,
            instead of recomputing it

    Returns:
        a structured array with fields 'x','y','intensity'
    """
    subpixel_modes = (
        'pixel',
        'poly',
        'multicorr'
    )
    er = f"Unrecognized subpixel option {subpixel}. Must be in {subpixel_modes}"
    assert subpixel in subpixel_modes, er

    # gaussian filtering
    ar = ar if sigma<=0 else gaussian_filter(ar, sigma)

    # local pixelwise maxima
    maxima_bool = \
        (ar >= np.roll(ar, (-1, 0), axis=(0, 1))) & (ar > np.roll(ar, (1, 0), axis=(0, 1))) & \
        (ar >= np.roll(ar, (0, -1), axis=(0, 1))) & (ar > np.roll(ar, (0, 1), axis=(0, 1))) & \
        (ar >= np.roll(ar, (-1, -1), axis=(0, 1))) & (ar > np.roll(ar, (-1, 1), axis=(0, 1))) & \
        (ar >= np.roll(ar, (1, -1), axis=(0, 1))) & (ar > np.roll(ar, (1, 1), axis=(0, 1)))

    # remove edges
    assert isinstance(edgeBoundary, (int, np.integer))
    if edgeBoundary < 1: edgeBoundary = 1
    maxima_bool[:edgeBoundary, :] = False
    maxima_bool[-edgeBoundary:, :] = False
    maxima_bool[:, :edgeBoundary] = False
    maxima_bool[:, -edgeBoundary:] = False

    # get indices
    # sort by intensity
    maxima_x, maxima_y = np.nonzero(maxima_bool)
    dtype = np.dtype([('x', float), ('y', float), ('intensity', float)])
    maxima = np.zeros(len(maxima_x), dtype=dtype)
    maxima['x'] = maxima_x
    maxima['y'] = maxima_y
    maxima['intensity'] = ar[maxima_x, maxima_y]
    maxima = np.sort(maxima, order='intensity')[::-1]

    if len(maxima) == 0:
            return maxima


    # filter
    maxima = _filter_2D_maxima(
        maxima,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minSpacing=minSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
    )

    if subpixel == 'pixel':
        return maxima


    # Parabolic subpixel refinement
    for i in range(len(maxima)):
        Ix1_ = ar[int(maxima['x'][i]) - 1, int(maxima['y'][i])].astype(np.float)
        Ix0 = ar[int(maxima['x'][i]), int(maxima['y'][i])].astype(np.float)
        Ix1 = ar[int(maxima['x'][i]) + 1, int(maxima['y'][i])].astype(np.float)
        Iy1_ = ar[int(maxima['x'][i]), int(maxima['y'][i]) - 1].astype(np.float)
        Iy0 = ar[int(maxima['x'][i]), int(maxima['y'][i])].astype(np.float)
        Iy1 = ar[int(maxima['x'][i]), int(maxima['y'][i]) + 1].astype(np.float)
        deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
        deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
        maxima['x'][i] += deltax
        maxima['y'][i] += deltay
        maxima['intensity'][i] = _linear_interpolation_2D(ar, maxima['x'][i], maxima['y'][i])

    if subpixel == 'poly':
        return maxima


    # Fourier upsampling
    if _ar_FT is None:
        _ar_FT = np.fft.fft2(ar)
    for ipeak in range(len(maxima['x'])):
        xyShift = np.array((maxima['x'][ipeak],maxima['y'][ipeak]))
        # we actually have to lose some precision and go down to half-pixel
        # accuracy. this could also be done by a single upsampling at factor 2
        # instead of get_maxima_2D.
        xyShift[0] = np.round(xyShift[0] * 2) / 2
        xyShift[1] = np.round(xyShift[1] * 2) / 2

        subShift = upsampled_correlation(_ar_FT,upsample_factor,xyShift)
        maxima['x'][ipeak]=subShift[0]
        maxima['y'][ipeak]=subShift[1]

    return maxima



def _filter_2D_maxima(
    maxima,
    minAbsoluteIntensity=0,
    minRelativeIntensity=0,
    relativeToPeak=0,
    minSpacing=0,
    edgeBoundary=1,
    maxNumPeaks=1,
    ):
    """
    Args:
        maxima : a numpy structured array with fields 'x', 'y', 'intensity'
        minAbsoluteIntensity : delete counts with intensity below this value
        minRelativeIntensity : delete counts with intensity below this value times
            the intensity of the i'th peak, where i is given by `relativeToPeak`
        relativeToPeak : see above
        minSpacing : if two peaks are within this euclidean distance from one
            another, delete the less intense of the two
        edgeBoundary : delete peaks within this distance of the image edge
        maxNumPeaks : an integer. defaults to 1

    Returns:
        a numpy structured array
    """

    # Remove maxima which are too dim
    if (minAbsoluteIntensity > 0):
        deletemask = maxima['intensity'] < minAbsoluteIntensity
        maxima = np.delete(maxima, np.nonzero(deletemask)[0])

    # Remove maxima which are too dim, compared to the n-th brightest
    if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
        assert isinstance(relativeToPeak, (int, np.integer))
        deletemask = maxima['intensity'] / maxima['intensity'][relativeToPeak] < minRelativeIntensity
        maxima = np.delete(maxima, np.nonzero(deletemask)[0])

    # Remove maxima which are too close
    if minSpacing > 0:
        deletemask = np.zeros(len(maxima), dtype=bool)
        for i in range(len(maxima)):
            if deletemask[i] == False:
                tooClose = ((maxima['x'] - maxima['x'][i]) ** 2 + \
                            (maxima['y'] - maxima['y'][i]) ** 2) < minSpacing ** 2
                tooClose[:i + 1] = False
                deletemask[tooClose] = True
        maxima = np.delete(maxima, np.nonzero(deletemask)[0])

    # Remove maxima in excess of maxNumPeaks
    if maxNumPeaks is not None:
        if len(maxima) > maxNumPeaks:
            maxima = maxima[:maxNumPeaks]

    return maxima


def _linear_interpolation_2D(ar, x, y):
    """
    Calculates the 2D linear interpolation of array ar at position x,y using the four
    nearest array elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    dx = x - x0
    dy = y - y0
    return (1 - dx) * (1 - dy) * ar[x0, y0] + (1 - dx) * dy * ar[x0, y1] + dx * (1 - dy) * ar[x1, y0] + dx * dy * ar[
        x1, y1]





# 3D stack of DPs


def _find_Bragg_disks_stack(
    dp_stack,
    template,
    filter_function = None,
    corrPower = 1,
    sigma = 2,
    subpixel = 'poly',
    upsample_factor = 16,
    minAbsoluteIntensity = 0,
    minRelativeIntensity = 0,
    relativeToPeak = 0,
    minPeakSpacing = 0,
    edgeBoundary = 1,
    maxNumPeaks = 100,
    _return_cc = False,
    _template_space = 'real'
    ):

    ans = []

    for idx in range(dp_stack.shape[0]):

        dp = dp_stack[idx,:,:]
        peaks =_find_Bragg_disks_single(
            dp,
            template,
            filter_function = None,
            corrPower = 1,
            sigma = 2,
            subpixel = 'poly',
            upsample_factor = 16,
            minAbsoluteIntensity = 0,
            minRelativeIntensity = 0,
            relativeToPeak = 0,
            minPeakSpacing = 0,
            edgeBoundary = 1,
            maxNumPeaks = 100,
            _return_cc = False,
            _template_space = 'real'
        )
        ans.append(peaks)

    return ans





# Whole datacube, CPU


def _find_Bragg_disks_CPU(
    datacube,
    probe,
    filter_function = None,
    corrPower = 1,
    sigma = 2,
    subpixel = 'multicorr',
    upsample_factor = 16,
    minAbsoluteIntensity = 0,
    minRelativeIntensity = 0.005,
    relativeToPeak = 0,
    minPeakSpacing = 60,
    edgeBoundary = 20,
    maxNumPeaks = 70,
    _qt_progress_bar = None,
    ):

    if _qt_progress_bar is not None:
        from PyQt5.QtWidgets import QApplication


    # Make the BraggVectors instance
    braggvectors = BraggVectors( datacube.Rshape )


    # Get the template's Fourier Transform
    probe_kernel_FT = np.conj(np.fft.fft2(probe)) if probe is not None else None


    # Loop over all diffraction patterns
    # Compute and populate BraggVectors data
    for (rx,ry) in tqdmnd(
        datacube.R_Nx,
        datacube.R_Ny,
        desc='Finding Bragg Disks',
        unit='DP',
        unit_scale=True
        ):
        if _qt_progress_bar is not None:
            _qt_progress_bar.setValue(Rx*datacube.R_Ny+Ry+1)
            QApplication.processEvents()

        # Get a diffraction pattern
        dp = datacube.data[rx,ry,:,:]

        # Compute
        peaks =_find_Bragg_disks_single(
            dp,
            template = probe_kernel_FT,
            filter_function = filter_function,
            corrPower = corrPower,
            sigma = sigma,
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minPeakSpacing = minPeakSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
            _return_cc = False,
            _template_space = 'fourier'
        )

        # Populate data
        braggvectors._v_uncal[rx,ry] = peaks


    # Return
    return braggvectors




















def _find_Bragg_disks_CUDA_unbatched(
    **kwargs
    ):
    pass


def _find_Bragg_disks_CUDA_batched(
    **kwargs
    ):
    pass

def _find_Bragg_disks_dask(
    **kwargs
    ):
    pass

def _find_Bragg_disks_ipp(
    **kwargs
    ):
    pass














def _parse_distributed(distributed):
    """
    Parse the `distributed` dict argument to determine distribution behavior
    """
    import os

    # parse mode (ipyparallel or dask)
    if "ipyparallel" in distributed:
        mode = 'ipyparallel'
        if "client_file" in distributed["ipyparallel"]:
            connect = distributed["ipyparallel"]["client_file"]
        else:
            er = "Within distributed[\"ipyparallel\"], "
            er += "missing key for \"client_file\""
            raise KeyError(er)

        try:
            import ipyparallel as ipp
            c = ipp.Client(url_file=connect, timeout=30)

            if len(c.ids) == 0:
                er = "No IPyParallel engines attached to cluster!"
                raise RuntimeError(er)
        except ImportError:
            raise ImportError("Unable to import module ipyparallel!")

    elif "dask" in distributed:
        mode = 'dask'
        if "client" in distributed["dask"]:
            connect = distributed["dask"]["client"]
        else:
            er = "Within distributed[\"dask\"], missing key for \"client\""
            raise KeyError(er)

    else:
        er = "Within distributed, you must specify 'ipyparallel' or 'dask'!"
        raise KeyError(er)


    # parse data file
    if "data_file" not in distributed:
        er = "Missing input data file path to distributed! "
        er += "Required key 'data_file'"
        raise KeyError(er)

    data_file = distributed["data_file"]

    if not isinstance(data_file, str):
        er = f"Expected string for distributed key 'data_file', "
        er += f"received {type(data_file)}"
        raise TypeError(er)
    if len(data_file.strip()) == 0:
        er = "Empty data file path from distributed key 'data_file'"
        raise ValueError(er)
    elif not os.path.exists(data_file):
        raise FileNotFoundError("File not found")


    # parse cluster path
    if "cluster_path" in distributed:
        cluster_path = distributed["cluster_path"]

        if not isinstance(cluster_path, str):
            er = f"distributed key 'cluster_path' must be of type str, "
            er += f"received {type(cluster_path)}"
            raise TypeError(er)

        if len(cluster_path.strip()) == 0:
            er = "distributed key 'cluster_path' cannot be an empty string!"
            raise ValueError(er)
        elif not os.path.exists(cluster_path):
            er = f"distributed key 'cluster_path' does not exist: {cluster_path}"
            raise FileNotFoundError(er)
        elif not os.path.isdir(cluster_path):
            er = f"distributed key 'cluster_path' is not a directory: "
            er += f"{cluster_path}"
            raise NotADirectoryError(er)
    else:
        cluster_path = None


    # return
    return connect, data_file, cluster_path, mode










#from .diskdetection_cuda import find_Bragg_disks_CUDA
#find_Bragg_disks_CUDA(
#    datacube,
#    probe,
#    corrPower=corrPower,
#    sigma=sigma,
#    edgeBoundary=edgeBoundary,
#    minRelativeIntensity=minRelativeIntensity,
#    minAbsoluteIntensity=minAbsoluteIntensity,
#    relativeToPeak=relativeToPeak,
#    minPeakSpacing=minPeakSpacing,
#    maxNumPeaks=maxNumPeaks,
#    subpixel=subpixel,
#    upsample_factor=upsample_factor,
#    name=name,
#    filter_function=filter_function,
#    _qt_progress_bar=_qt_progress_bar,
#    batching=CUDA_batched
#)
#
#
#from .diskdetection_parallel import find_Bragg_disks_ipp
#find_Bragg_disks_ipp(
#    datacube,
#    probe,
#    corrPower=corrPower,
#    sigma=sigma,
#    edgeBoundary=edgeBoundary,
#    minRelativeIntensity=minRelativeIntensity,
#    minAbsoluteIntensity=minAbsoluteIntensity,
#    relativeToPeak=relativeToPeak,
#    minPeakSpacing=minPeakSpacing,
#    maxNumPeaks=maxNumPeaks,
#    subpixel=subpixel,
#    upsample_factor=upsample_factor,
#    filter_function=filter_function,
#    ipyparallel_client_file=connect,
#    data_file=data_file,
#    cluster_path=cluster_path
#)
#
#from .diskdetection_parallel import find_Bragg_disks_dask
#find_Bragg_disks_dask(
#    datacube,
#    probe,
#    corrPower=corrPower,
#    sigma=sigma,
#    edgeBoundary=edgeBoundary,
#    minRelativeIntensity=minRelativeIntensity,
#    minAbsoluteIntensity=minAbsoluteIntensity,
#    relativeToPeak=relativeToPeak,
#    minPeakSpacing=minPeakSpacing,
#    maxNumPeaks=maxNumPeaks,
#    subpixel=subpixel,
#    upsample_factor=upsample_factor,
#    filter_function=filter_function,
#    dask_client=connect,
#    data_file=data_file,
#    cluster_path=cluster_path
#    )































# Old code







def find_Bragg_disks_single_DP(
    DP,
    probe_kernel,
    corrPower = 1,
    sigma = 2,
    edgeBoundary = 20,
    minRelativeIntensity = 0.005,
    minAbsoluteIntensity = 0,
    relativeToPeak = 0,
    minPeakSpacing = 60,
    maxNumPeaks = 70,
    subpixel = 'multicorr',
    upsample_factor = 16,
    filter_function = None,
    return_cc = False
    ):
    """
    Identical to _find_Bragg_disks_single_DP_FK, accept that this function
    accepts a probe_kernel in real space, rather than Fourier space. For more
    info, see the _find_Bragg_disks_single_DP_FK documentation.

    Args:
        DP (ndarray): a diffraction pattern
        probe_kernel (ndarray): the vacuum probe template, in real space. If
            None, no correlation is performed.
        corrPower (float between 0 and 1, inclusive): the cross correlation
            power. A value of 1 corresponds to a cross correaltion, and 0
            corresponds to a phase correlation, with intermediate values giving
            various hybrids.
        sigma (float): the standard deviation for the gaussian smoothing applied
            to the cross correlation
        edgeBoundary (int): minimum acceptable distance from the DP edge, in
            pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak
            intensity, relative to the intensity of the brightest peak
        minAbsoluteIntensity (float): the minimum acceptable correlation peak
            intensity, on an absolute scale
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (float): the minimum acceptable spacing between detected
            peaks
        maxNumPeaks (int): the maximum number of peaks to return
        subpixel (str): Whether to use subpixel fitting, and which algorithm to
            use. Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor (int): upsampling factor for subpixel fitting (only used
            when subpixel='multicorr')
        filter_function (callable): filtering function to apply to each
            diffraction pattern before peakfinding. Must be a function of only
            one argument (the diffraction pattern) and return the filtered
            diffraction pattern. The shape of the returned DP must match the
            shape of the probe kernel (but does not need to match the shape of
            the input diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        return_cc (bool): if True, return the cross correlation

    Returns:
        (PointList): the Bragg peak positions and correlation intensities
    """
    er = "filter_function must be callable"
    if filter_function: assert callable(filter_function), er
    if probe_kernel is not None:
        probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))
    else:
        probe_kernel_FT = None
    return _find_Bragg_disks_single_DP_FK(
        DP,
        probe_kernel_FT,
        corrPower = corrPower,
        sigma = sigma,
        edgeBoundary = edgeBoundary,
        minRelativeIntensity = minRelativeIntensity,
        minAbsoluteIntensity = minAbsoluteIntensity,
        relativeToPeak = relativeToPeak,
        minPeakSpacing = minPeakSpacing,
        maxNumPeaks = maxNumPeaks,
        subpixel = subpixel,
        upsample_factor = upsample_factor,
        filter_function = filter_function,
        return_cc = return_cc
    )


def find_Bragg_disks_selected(
    datacube,
    probe,
    Rx,
    Ry,
    corrPower = 1,
    sigma = 2,
    edgeBoundary = 20,
    minRelativeIntensity = 0.005,
    minAbsoluteIntensity = 0,
    relativeToPeak = 0,
    minPeakSpacing = 60,
    maxNumPeaks = 70,
    subpixel = 'multicorr',
    upsample_factor = 16,
    filter_function = None,
    return_ccs = False
    ):
    """
    Finds the Bragg disks in the diffraction patterns of datacube at scan
    positions (Rx,Ry) by cross, hybrid, or phase correlation with probe.

    Args:
        DP (ndarray): a diffraction pattern
        probe (ndarray): the vacuum probe template, in real space. If None, no
            correlation is performed.
        Rx (int or tuple/list of ints): scan position x-coords of DPs of interest
        Ry (int or tuple/list of ints): scan position y-coords of DPs of interest
        corrPower (float between 0 and 1, inclusive): the cross correlation
            power. A value of 1 corresponds to a cross correaltion, and 0
            corresponds to a phase correlation, with intermediate values giving
            various hybrids.
        sigma (float): the standard deviation for the gaussian smoothing applied
            to the cross correlation
        edgeBoundary (int): minimum acceptable distance from the DP edge, in
            pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak
            intensity, relative to the intensity of the brightest peak
        minAbsoluteIntensity (float): the minimum acceptable correlation peak
            intensity, on an absolute scale
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (float:) the minimum acceptable spacing between detected
            peaks
        maxNumPeaks (int): the maximum number of peaks to return
        subpixel (str): Whether to use subpixel fitting, and which algorithm to
            use. Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor (int): upsampling factor for subpixel fitting (only used
            when subpixel='multicorr')
        filter_function (callable): filtering function to apply to each
            diffraction pattern before peakfinding. Must be a function of only
            one argument (the diffraction pattern) and return the filtered
            diffraction pattern. The shape of the returned DP must match the
            shape of the probe kernel (but does not need to match the shape of
            the input diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        return_ccs (bool): if True, return the cross correlations

    Returns:
        (n-tuple of PointLists, n=len(Rx)): the Bragg peak positions and
            correlation intensities at each scan position (Rx,Ry).  If
            return_ccs=True, returns (peaks,ccs), where peaks is the n-tuple of
            PointLists, and ccs is a (QNx,QNy,n) shaped array of the correlograms.
    """
    assert(len(Rx)==len(Ry))
    er = "filter_function must be callable"
    if filter_function: assert callable(filter_function), er
    peaks = []

    # Get probe kernel in Fourier space
    if probe is not None:
        probe_kernel_FT = np.conj(np.fft.fft2(probe))
    else:
        probe_kernel = None

    n = len(Rx)
    if return_ccs:
        ccs = np.zeros((datacube.Q_Nx,datacube.Q_Ny,n))

    # Loop over selected diffraction patterns
    for i in range(len(Rx)):
        DP = datacube.data[Rx[i],Ry[i],:,:]
        _peaks =  _find_Bragg_disks_single_DP_FK(
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
            return_cc=return_ccs
        )
        if return_ccs:
            _peaks,ccs[:,:,i] = _peaks
        peaks.append(_peaks)

    peaks = tuple(peaks)
    if return_ccs:
        return peaks,ccs
    return peaks







# Thresholding fns

def threshold_Braggpeaks(
    pointlistarray,
    minRelativeIntensity,
    relativeToPeak,
    minPeakSpacing,
    maxNumPeaks
    ):
    """
    Takes a PointListArray of detected Bragg peaks and applies additional
    thresholding, returning the thresholded PointListArray. To skip a threshold,
    set that parameter to False.

    Args:
        pointlistarray (PointListArray): The Bragg peaks. Must have
            coords=('qx','qy','intensity')
        minRelativeIntensity  (float): the minimum allowed peak intensity,
            relative to the brightest peak in each diffraction pattern
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (int): the minimum allowed spacing between adjacent peaks
        maxNumPeaks (int): maximum number of allowed peaks per diffraction
            pattern
    """
    assert all([item in pointlistarray.dtype.fields for item in ['qx','qy','intensity']]), (
                "pointlistarray must include the coordinates 'qx', 'qy', and 'intensity'.")
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        pointlist.sort(coordinate='intensity', order='descending')

        # Remove peaks below minRelativeIntensity threshold
        if minRelativeIntensity is not False:
            deletemask = pointlist.data['intensity']/pointlist.data['intensity'][relativeToPeak] < \
                                                                           minRelativeIntensity
            pointlist.remove_points(deletemask)

        # Remove peaks that are too close together
        if maxNumPeaks is not False:
            r2 = minPeakSpacing**2
            deletemask = np.zeros(pointlist.length, dtype=bool)
            for i in range(pointlist.length):
                if deletemask[i] == False:
                    tooClose = ( (pointlist.data['qx']-pointlist.data['qx'][i])**2 + \
                                 (pointlist.data['qy']-pointlist.data['qy'][i])**2 ) < r2
                    tooClose[:i+1] = False
                    deletemask[tooClose] = True
            pointlist.remove_points(deletemask)

        # Keep only up to maxNumPeaks
        if maxNumPeaks is not False:
            if maxNumPeaks < pointlist.length:
                deletemask = np.zeros(pointlist.length, dtype=bool)
                deletemask[maxNumPeaks:] = True
                pointlist.remove_points(deletemask)

    return pointlistarray


def universal_threshold(
    pointlistarray,
    thresh,
    metric='maximum',
    minPeakSpacing=False,
    maxNumPeaks=False,
    name=None
    ):
    """
    Takes a PointListArray of detected Bragg peaks and applies universal
    thresholding, returning the thresholded PointListArray. To skip a threshold,
    set that parameter to False.

    Args:
        pointlistarray (PointListArray): The Bragg peaks. Must have
            coords=('qx','qy','intensity')
        thresh (float): the minimum allowed peak intensity. The meaning of this
            threshold value is determined by the value of the 'metric' argument,
            below
        metric (string): the metric used to compare intensities. Must be in
            ('maximum','average','median','manual'). In each case aside from
            'manual', the intensity threshold is set to Val*thresh, where Val is
            given by
                * 'maximum' - the maximum intensity in the entire pointlistarray
                * 'average' - the average of the maximum intensities of each
                  scan position in the pointlistarray
                * 'median' - the medain of the maximum intensities of each
                  scan position in the entire pointlistarray
            If metric is 'manual', the threshold is exactly minIntensity
        minPeakSpacing (int): the minimum allowed spacing between adjacent peaks.
            optional, default is false
        maxNumPeaks (int): maximum number of allowed peaks per diffraction pattern.
            optional, default is false
        name (str, optional): a name for the returned PointListArray. If
            unspecified, takes the old PLA name and appends '_unithresh'.

    Returns:
        (PointListArray): Bragg peaks thresholded by intensity.
    """
    assert isinstance(pointlistarray,PointListArray)
    assert metric in ('maximum','average','median','manual')
    assert all([item in pointlistarray.dtype.fields for item in ['qx','qy','intensity']]), (
                "pointlistarray must include the coordinates 'qx', 'qy', and 'intensity'.")
    _pointlistarray = pointlistarray.copy()
    if name is None:
        _pointlistarray.name = pointlistarray.name+"_unithresh"

    HI_array = np.zeros( (_pointlistarray.shape[0], _pointlistarray.shape[1]) )
    for (Rx, Ry) in tqdmnd(_pointlistarray.shape[0],_pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
            pointlist = _pointlistarray.get_pointlist(Rx,Ry)
            if pointlist.data.shape[0] == 0:
                top_value = np.nan
            else:
                HI_array[Rx, Ry] = np.max(pointlist.data['intensity'])

    if metric=='maximum':
        _thresh = np.max(HI_array)*thresh
    elif metric=='average':
        _thresh = np.nanmean(HI_array)*thresh
    elif metric=='median':
        _thresh = np.median(HI_array)*thresh
    else:
        _thresh = thresh

    for (Rx, Ry) in tqdmnd(_pointlistarray.shape[0],_pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
            pointlist = _pointlistarray.get_pointlist(Rx,Ry)

            # Remove peaks below minRelativeIntensity threshold
            deletemask = pointlist.data['intensity'] < _thresh
            pointlist.remove_points(deletemask)

            # Remove peaks that are too close together
            if maxNumPeaks is not False:
                r2 = minPeakSpacing**2
                deletemask = np.zeros(pointlist.length, dtype=bool)
                for i in range(pointlist.length):
                    if deletemask[i] == False:
                        tooClose = ( (pointlist.data['qx']-pointlist.data['qx'][i])**2 + \
                                     (pointlist.data['qy']-pointlist.data['qy'][i])**2 ) < r2
                        tooClose[:i+1] = False
                        deletemask[tooClose] = True
                pointlist.remove_points(deletemask)

            # Keep only up to maxNumPeaks
            if maxNumPeaks is not False:
                if maxNumPeaks < pointlist.length:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    deletemask[maxNumPeaks:] = True
                    pointlist.remove_points(deletemask)
    return _pointlistarray


def get_pointlistarray_intensities(pointlistarray):
    """
    Concatecates the Bragg peak intensities from a PointListArray of Bragg peak
    positions into one array and returns the intensities. This output can be used
    for understanding the distribution of intensities in your dataset for
    universal thresholding.

    Args:
        pointlistarray (PointListArray):

    Returns:
        (ndarray): all detected peak intensities
    """
    assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']]), (
        "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'.")
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"

    first_pass = True
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1],desc='Getting disk intensities',unit='DP',unit_scale=True):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        for i in range(pointlist.length):
            if first_pass:
                peak_intensities = np.array(pointlist.data[i][2])
                peak_intensities = np.reshape(peak_intensities, 1)
                first_pass = False
            else:
                temp_array = np.array(pointlist.data[i][2])
                temp_array = np.reshape(temp_array, 1)
                peak_intensities = np.append(peak_intensities, temp_array)
    return peak_intensities



















def find_Bragg_disks_DEPR(
    datacube,
    probe,
    corrPower = 1,
    sigma = 2,
    edgeBoundary = 20,
    minRelativeIntensity = 0.005,
    minAbsoluteIntensity = 0,
    relativeToPeak = 0,
    minPeakSpacing = 60,
    maxNumPeaks = 70,
    subpixel = 'multicorr',
    upsample_factor = 16,
    name = 'braggpeaks_raw',
    filter_function = None,
    _qt_progress_bar = None,
    distributed = None,
    CUDA = False,
    CUDA_batched = True
    ):
    """
    Finds the Bragg disks in all of datacube's diffraction patterns by
    cross correlative template matching against probe.

    Args:
        DP (ndarray): a diffraction pattern
        probe (ndarray): the vacuum probe template, in real space. If None, does
            not perform a cross correlation.
        corrPower (float between 0 and 1, inclusive): the cross correlation
            power. A value of 1 corresponds to a cross correaltion, and 0
            corresponds to a phase correlation, with intermediate values giving
            various hybrids.
        sigma (float): the standard deviation for the gaussian smoothing applied
            to the cross correlation
        edgeBoundary (int): minimum acceptable distance from the DP edge, in
            pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak
            intensity, relative to the intensity of the brightest peak
        minAbsoluteIntensity (float): the minimum acceptable correlation peak
            intensity, on an absolute scale
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (float): the minimum acceptable spacing between detected
            peaks
        maxNumPeaks (int): the maximum number of peaks to return
        subpixel (str): Whether to use subpixel fitting, and which algorithm to
            use. Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor (int): upsampling factor for subpixel fitting (only used
            when subpixel='multicorr')
        name (str): name for the returned PointListArray
        filter_function (callable): filtering function to apply to each
            diffraction pattern before peakfinding. Must be a function of only
            one argument (the diffraction pattern) and return the filtered
            diffraction pattern. The shape of the returned DP must match the
            shape of the probe kernel (but does not need to match the shape of
            the input diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        _qt_progress_bar (QProgressBar instance): used only by the GUI for serial
            execution
        distributed (dict): contains information for parallelprocessing using an
            IPyParallel or Dask distributed cluster.  Valid keys are:
                * ipyparallel (dict):
                * client_file (str): path to client json for connecting to your
                    existing IPyParallel cluster
                * dask (dict): client (object): a dask client that connects to
                    your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): defaults to the working directory during
                    processing
            if distributed is None, which is the default, processing will be in
            serial
        CUDA (bool): If True, import cupy and use an NVIDIA GPU to perform disk
            detection
        CUDA_batched (bool): If True, and CUDA is selected, the FFT and IFFT
            steps of disk detection are performed in batches to better utilize
            GPU resources.

    Returns:
        (PointListArray): the Bragg peak positions and correlation intensities
    """

    if distributed is None:
        if not CUDA:
            return find_Bragg_disks_serial(
                datacube,
                probe,
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
                name=name,
                filter_function=filter_function,
                _qt_progress_bar=_qt_progress_bar)
        else:
            from .diskdetection_cuda import find_Bragg_disks_CUDA
            return find_Bragg_disks_CUDA(
                datacube,
                probe,
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
                name=name,
                filter_function=filter_function,
                _qt_progress_bar=_qt_progress_bar,
                batching=CUDA_batched)

    elif isinstance(distributed, dict):
        connect, data_file, cluster_path = _parse_distributed(distributed)

        if "ipyparallel" in distributed:
            from .diskdetection_parallel import find_Bragg_disks_ipp

            return find_Bragg_disks_ipp(
                datacube,
                probe,
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
                ipyparallel_client_file=connect,
                data_file=data_file,
                cluster_path=cluster_path
                )
        elif "dask" in distributed:
            from .diskdetection_parallel import find_Bragg_disks_dask

            return find_Bragg_disks_dask(
                datacube,
                probe,
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
                dask_client=connect,
                data_file=data_file,
                cluster_path=cluster_path
                )
    else:
        raise ValueError("Expected type dict or None for distributed, instead found : {}".format(type(distributed)))















