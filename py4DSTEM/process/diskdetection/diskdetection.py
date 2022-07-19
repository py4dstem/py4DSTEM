# Functions for finding Bragg scattering by cross correlative template matching
# with a vacuum probe.

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ...io.datastructure.py4dstem import DataCube, QPoints, BraggVectors
from ..utils.get_maxima_2D import get_maxima_2D
from ..utils.cross_correlate import get_cross_correlation_FT
from ...utils.tqdmnd import tqdmnd





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

            # h5py datasets have different rules for slicing than
            # numpy arrays, so we have to do this manually
            if "h5py" in str(type(dc.data)):
                data = np.zeros((len(rx),dc.Q_Nx,dc.Q_Ny))
                for i,(x,y) in enumerate(zip(rx,ry)):
                    data[i] = dc.data[x,y]
            else:
                data = dc.data[np.array(rx),np.array(ry),:,:]
            if data.ndim == 2:
                mode = 'dp'
            elif data.ndim == 3:
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
    kws = {}
    if _qt_progress_bar is not None:
        kws['_qt_progress_bar'] = _qt_progress_bar
    # distributed kwargs
    if distributed is not None:
        kws['connect'] = connect
        kws['data_file'] = data_file
        kws['cluster_path'] = cluster_path

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
        # _returnval = 'fourier' if subpixel == 'multicorr' else 'real'
        cc = get_cross_correlation_FT(
            DP,
            template_FT,
            corrPower,
            'fourier',
        )


    # Get maxima
    maxima = get_maxima_2D(
        np.maximum(np.real(np.fft.ifft2(cc)),0),
        subpixel = subpixel,
        upsample_factor = upsample_factor,
        minAbsoluteIntensity = minAbsoluteIntensity,
        minRelativeIntensity = minRelativeIntensity,
        relativeToPeak = relativeToPeak,
        minSpacing = minPeakSpacing,
        edgeBoundary = edgeBoundary,
        maxNumPeaks = maxNumPeaks,
        _ar_FT = cc
    )

    # Wrap as QPoints instance
    maxima = QPoints( maxima )


    # Return
    if _return_cc is True:
        return maxima, cc
    return maxima





#def _get_cross_correlation_FT(
#    DP,
#    template_FT,
#    corrPower = 1,
#    _returnval = 'real'
#    ):
#    """
#    if _returnval is 'real', returns the real-valued cross-correlation.
#    otherwise, returns the complex valued result.
#    """
#
#    m = np.fft.fft2(DP) * template_FT
#    cc = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
#    if _returnval == 'real':
#        cc = np.maximum(np.real(np.fft.ifft2(cc)),0)
#    return cc






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
    _template_space = 'real'
    ):

    ans = []

    for idx in range(dp_stack.shape[0]):

        dp = dp_stack[idx,:,:]
        peaks =_find_Bragg_disks_single(
            dp,
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
            _template_space = _template_space,
            _return_cc = False
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
    braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )


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





# CUDA - unbatched


def _find_Bragg_disks_CUDA_unbatched(
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

    # compute
    from .diskdetection_cuda import find_Bragg_disks_CUDA
    peaks = find_Bragg_disks_CUDA(
        datacube,
        probe,
        filter_function=filter_function,
        corrPower=corrPower,
        sigma=sigma,
        subpixel=subpixel,
        upsample_factor=upsample_factor,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minPeakSpacing=minPeakSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
        _qt_progress_bar=_qt_progress_bar,
        batching=False)

    # Populate a BraggVectors instance and return
    braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )
    braggvectors._v_uncal = peaks
    return braggvectors




# CUDA - batched


def _find_Bragg_disks_CUDA_batched(
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

    # compute
    from .diskdetection_cuda import find_Bragg_disks_CUDA
    peaks = find_Bragg_disks_CUDA(
        datacube,
        probe,
        filter_function=filter_function,
        corrPower=corrPower,
        sigma=sigma,
        subpixel=subpixel,
        upsample_factor=upsample_factor,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minPeakSpacing=minPeakSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
        _qt_progress_bar=_qt_progress_bar,
        batching=True)

    # Populate a BraggVectors instance and return
    braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )
    braggvectors._v_uncal = peaks
    return braggvectors





# Distributed - ipyparallel


def _find_Bragg_disks_ipp(
    datacube,
    probe,
    connect,
    data_file,
    cluster_path,
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

    # compute
    from .diskdetection_parallel import find_Bragg_disks_ipp
    peaks = find_Bragg_disks_ipp(
        datacube,
        probe,
        filter_function=filter_function,
        corrPower=corrPower,
        sigma=sigma,
        subpixel=subpixel,
        upsample_factor=upsample_factor,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minPeakSpacing=minPeakSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
        ipyparallel_client_file=connect,
        data_file=data_file,
        cluster_path=cluster_path
        )

    # Populate a BraggVectors instance and return
    braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )
    braggvectors._v_uncal[rx,ry] = peaks
    return braggvectors





# Distributed - dask

def _find_Bragg_disks_dask(
    datacube,
    probe,
    connect,
    data_file,
    cluster_path,
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

    # compute
    from .diskdetection_parallel import find_Bragg_disks_dask
    peaks = find_Bragg_disks_dask(
        datacube,
        probe,
        filter_function=filter_function,
        corrPower=corrPower,
        sigma=sigma,
        subpixel=subpixel,
        upsample_factor=upsample_factor,
        minAbsoluteIntensity=minAbsoluteIntensity,
        minRelativeIntensity=minRelativeIntensity,
        relativeToPeak=relativeToPeak,
        minPeakSpacing=minPeakSpacing,
        edgeBoundary=edgeBoundary,
        maxNumPeaks=maxNumPeaks,
        dask_client_file=connect,
        data_file=data_file,
        cluster_path=cluster_path
        )

    # Populate a BraggVectors instance and return
    braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )
    braggvectors._v_uncal[rx,ry] = peaks
    return braggvectors








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







