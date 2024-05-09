import numpy as np
from scipy.ndimage import gaussian_filter
import inspect

from emdfile import tqdmnd, Metadata
from py4DSTEM.utils import get_maxima_2D, get_cross_correlation_FT
from py4DSTEM.data import QPoints
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.datacube.diskdetection import Probe


class BraggFinder(object):
    """
    Handles disk detection.
    """

    def __init__(
        self,
    ):
        self.bragg_detection_defaults = {
            'template' : None,
            'data' : 'all',
            'preprocess' : False,
            'corr_power' : 1,
            'corr_sigma' : 0,
            'sigma' : 0,
            'min_intensity' : 0,
            'min_spacing' : 5,
            'subpixel' : 'poly',
            'upsample_factor' : 16,
            'edge_filter' : 1,
            'n_peaks_max' : 10000,
            'min_prominence' : 0,
            'prominence_kernel_size' : 3,
            'min_rel_intensity' : 0,
            'min_rel_ref_peak' : 0,
            'device' : False,
            'ML' : False,
            'show_peaks' : True,
            'show_peaks_params' : {
                's_peaks' : 1,
            },
        }

    def update_defaults(
        self,
        template=None,
        data=None,
        preprocess=None,
        corr_power=None,
        corr_sigma=None,
        sigma=None,
        subpixel=None,
        upsample_factor=None,
        min_intensity=None,
        min_prominence=None,
        prominence_kernel_size=None,
        min_rel_intensity=None,
        min_rel_ref_peak=None,
        min_spacing=None,
        edge_filter=None,
        n_peaks_max=None,
        device=None,
        ML=None,
        show_peaks=None,
        show_peaks_params=None,
    ):
        # add new defaults to a dict
        new_defaults = {}
        if template is not None: new_defaults['template'] = template
        if data is not None: new_defaults['data'] = data
        if preprocess is not None: new_defaults['preprocess'] = preprocess
        if corr_power is not None: new_defaults['corr_power'] = corr_power
        if corr_sigma is not None: new_defaults['corr_sigma'] = corr_sigma
        if sigma is not None: new_defaults['sigma'] = sigma
        if subpixel is not None: new_defaults['subpixel'] = subpixel
        if upsample_factor is not None: new_defaults['upsample_factor'] = upsample_factor
        if min_intensity is not None: new_defaults['min_intensity'] = min_intensity
        if min_prominence is not None: new_defaults['min_prominence'] = min_prominence
        if prominence_kernel_size is not None: new_defaults['prominence_kernel_size'] = prominence_kernel_size
        if min_rel_intensity is not None: new_defaults['min_rel_intensity'] = min_rel_intensity
        if min_rel_ref_peak is not None: new_defaults['min_rel_ref_peak'] = min_rel_ref_peak
        if min_spacing is not None: new_defaults['min_spacing'] = min_spacing
        if edge_filter is not None: new_defaults['edge_filter'] = edge_filter
        if n_peaks_max is not None: new_defaults['n_peaks_max'] = n_peaks_max
        if device is not None: new_defaults['device'] = device
        if ML is not None: new_defaults['ML'] = ML
        if show_peaks is not None: new_defaults['show_peaks'] = show_peaks
        if show_peaks_params is not None: new_defaults['show_peaks_params'] = show_peaks_params

        # update
        self.bragg_detection_defaults = self.bragg_detection_defaults | new_defaults


    def find_bragg_vectors(
        self,
        template=None,
        data=None,
        preprocess=None,
        corr_power=None,
        corr_sigma=None,
        sigma=None,
        n_peaks_max=None,
        min_spacing=None,
        min_intensity=None,
        min_prominence=None,
        prominence_kernel_size=None,
        min_rel_intensity=None,
        min_rel_ref_peak=None,
        edge_filter=None,
        subpixel=None,
        upsample_factor=None,
        device=None,
        ML=None,
        show_peaks=None,
        show_peaks_params=None,
        return_cc=False,
        name = 'braggvectors',
        returncalc = True,
        _return_cc = False,
        **kwargs,
    ):
        """
        Finds Bragg scattering vectors.

        In normal operation, localizes Bragg scattering using template matching,
        by (1) optional preprocessing, (2) cross-correlating with the template,
        and (3) finding local maxima, thresholding and returning. Acceleration is
        handled with `device`.

        Input parameters are stored internally once they're set, and will be
        reused in subsequent function calls unless new values are provided, which
        will replace the set prior values.  Currently set values are available at
        .bragg_detection_defaults. BraggVectors instances returned by this method
        (i.e. return values when the full datacube is used) also store the input
        parameters used by this method to create them as a Metadata dictionary in
        vectors.metadata['gen_params'].

        Invoking `ML` makes use of a custom neural network called FCU-net
        instead of template matching. If you use FCU-net in your work,
        please reference "Munshi, Joydeep, et al. npj Computational Materials
        8.1 (2022): 254".


        Basic Usage (CPU + cross-correlation)
        -------------------------------------

        >>> datacube.get_bragg_vectors( template )

        will find bragg scattering for the entire datacube using cross-
        correlative template matching on the CPU using the currently set
        detection parameters or, if no parameters have been set, with the
        default parameters.

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     corr_power = 1,
        >>>     sigma = 2,
        >>>     subpixel = 'multicorr',
        >>>     upsample_factor = 16,
        >>> )

        will perform the same computation now using a cross-correlation power
        of 1 (i.e. a normal cross correlatin), a guassian blur after cross
        cross correlation but before peak detection of 2 pixels, Fourier
        upsampling for subpixel refinement with an upsampling factor of 16,
        and the currently set or defaults for the remaining paramters.

        Subsequently calling

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>         'minAboluteIntensity' : 100,
        >>>         'minPeakSpacing' : 18,
        >>>         'edgeBoundary' : 10,
        >>>         'maxNumPeaks' : 100,
        >>>     },
        >>> )

        will perform the same computation with the parameters set previously,
        now using thresholds for the absolute peak intensity, the spacing
        between the peaks, the distance of the peaks from the image edge,
        and the number of peaks as specified.

        Each of the calls above performs disk detection on the full datacube
        and returns a BraggVectors instance, which has the input parameters
        stored in its .metadata['gen_params']  attribute.


        Running on Subsets of Data (e.g. Testing)
        -----------------------------------------

        Using

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (5,6)
        >>> )

        will perform template matching against the diffraction image at
        scan position (5,6) only, returning a single set of detected
        peaks as a QPoints instance, and

        >>> datacube.select_patterns(
        >>>     (np.array([4,5,6]),
        >>>      np.array([10,11,12]))
        >>> )
        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = 's')
        >>> )

        will perform template matching against the 3 diffraction images at
        scan positions (4,10), (5,11), (6,12), returning a list of 3
        QPoints instances.  Note that in these two examples, the inputs
        describing the function call *are* stored internally in the calling
        datacube instance its new disk detection defaults, however they are
        *not* stored in any metadata associated with the output object, as
        they would be with a run on the full datacube.


        Running without Cross Correlation (i.e. simple peak finding)
        ------------------------------------------------------------

        Data with sharp peaks instead of disks at the Bragg points, either
        due to a very small convergence angle or a parallel beam, may not
        require a cross correlation; simply locating the maxima may suffice.
        Using

        >>> datacube.fing_bragg_vectors(
        >>>     template = False,
        >>> )

        will skip the cross-correlation step, and will instead perform maximum
        detection on the raw data of the entire datacube, applying whichever
        other filtering or preprocessing arguments that have been specified.


        Preprocessing - Pre-set Methods
        -------------------------------

        Several types of preprocessing - methods run on each diffraction
        pattern before the cross-correlation step - are possible, including
        pre-set methods and custom methods.

        Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = 'local_average'
        >>> )

        will use the local 3x3 gaussian averaged pattern at each scan position,
        i.e. the weighted average diffraction image with a 3x3 gaussian
        footprint in real space

            [[ 1, 2, 1 ],
             [ 2, 4, 2 ],  *  (1/16)
             [ 1, 2, 1 ]]

        is used in lieu of the raw diffraction pattern.

        Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = 'radial_background_subtraction'
        >>> )

        will calculate and subtract the radial median from each diffraction
        image; using this option requires the the origin has been located
        and set (e.g. with the .fit_origin method) and that a PolarDatacube
        as been initialized, e.g. using

        >>> polarcube = py4DSTEM.PolarDatacube(
        >>>     datacube,
        >>>     n_annular = 180,
        >>>     qstep = 1,
        >>> )

        The aliases 'bs' and 'la' can be used, respectively, instead of
        'radial_background_subtraction' and 'local_average'.

        Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = ['la','bs']
        >>> )

        Will perform a local average then a background subtraction in
        succession.


        Preprocessing - Custom Methods
        ------------------------------

        Custom preprocessing methods may be specified as well.  Using

        >>> def preprocess_fn(data):
        >>>     return py4DSTEM.utils.bin2D(data,2)
        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = preprocess_fn
        >>> )

        will bin each diffraction image by 2 before cross-correlating.
        Note that the template shape must match the final, preprocessed
        data shape.  Note also that in this case the `preprocess_fn`
        function takes only a single input, which must be a 2D array
        representing a diffraction pattern.

        To use a custom preprocessing method which includes arguments,
        use

        >>> def preprocess_fn_with_args(data,N):
        >>>     return py4DSTEM.utils.bin2D(data,N)
        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'f' : preprocess_fn_with_args,
        >>>         'N' : 2
        >>> )

        where the dictionary element 'f' must point to the preprocessing
        function to be used, and all other function inputs must be
        represented by a dictionary keyword-argument pair.

        It can be useful to write preprocessing methods which make use
        of the scan position values, e.g. to retrieve calibration values
        which vary by scan position. In this case, use

        >>> def preprocess_fn_with_args_and_scan_pos(data,N,x,y):
        >>>     return py4DSTEM.utils.bin2D(data,N)
        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'f' : preprocess_fn_with_args_and_scan_pos,
        >>>         'N' : 2,
        >>>         'x' : None,
        >>>         'y' : None,
        >>> )

        The values passed for 'x' and 'y' in the dictionary are ignored,
        and will be replaced by the scan position integers for each
        pattern.


        Examples (GPU acceleration, cluster acceleration, and ML)
        -------------------------------------------------------
        # TODO!

        device=None,
        ML=None,
        return_cc=False,
        name = 'braggvectors',
        returncalc = True,
        _return_cc = False,


        Parameters
        ----------
        template : 2d np.ndarray (Qshape) or Probe or False
            The matching template. If an ndarray is passed, must be centered
            about the origin in the FFT sense (i.e. the corner). If a Probe is
            passed, probe.kernel is used (and must be populated). If False is
            passed, cross correlation is skipped and the maxima are taken
            directly from the (possibly preprocessed) diffraction data
        data : None or string or (int,int) or Qshaped 2D array
            Specifies the input data and return value.
            If None, use the current default
            If a string, must be in ('s', 'selected', 'a', 'all').
            Either 's' or 'selected' runs the scan positions selected using
            .select_patterns and return a list of QPoints instance.
            Either 'a' or 'all' runs on the full dataset and returns a
            BraggVectors instance.
            If a 2-tuple (int,int), use the diffraction pattern at this scan
            position and return a QPoints instance.
            If a 2D boolean numpy array (Rshaped), runs on the diffraction
            images where True and return a list of QPoints instances at scan
            positions np.nonzero(ar)
            If a 2D numpy array of floats (Qshaped), runs on the input array
            and returns a QPoints instance
        preprocess : (False,string,list,callable,dict)
            If False, no preprocessing is performed.
            Strings must be in ('radial_background_subtraction','bs',
            'local_average','la'), where 'bs' and 'la' are aliases for their
            preceding entries.  Lists indicate preprocessing steps to take in
            succession, and may include the strings above and callables.
            lists must contain only strings in the above list and callables.
            Callables (i.e. functions) should accept a single input
            corresponding to the diffraction array.  Preprocessing functions
            of additional variables are invoked by passing a dictionary, which
            must include the keyword 'f' with the callable function as its value
            as well as keyword-argument pairs for each additional function input.
            The keywords 'x' and 'y' are reserved; if these are included in the
            function, at runtime their value will be replaced by the scan postion
            (x,y) of ints of the pattern being analyzed
        corr_power : number in [0-1]
            type of correlation to perform, where 1 is a cross correlation, 0 is
            a phase correlation, values in between are hybrid correlations given
            by abs(CC)^corr_power * exp(arg(CC))
        corr_sigma : number
            if >0, apply a gaussian blur to the diffraction image before cross
            correlating
        sigma : number
            if >0, apply a gaussian blur to the correlogram before finding maxima
        n_peaks_max : int
            only the brightest n_peaks_max peaks are returned
        min_spacing=None : number
            if two maxima are closer together than this value, the dimmer
            maximum is removed
        min_intensity : number
            maxima with intensities below this value are removed. Ignored if set
            to 0.
        min_prominence : number
            maxima with intensity differences relative to their background less
            than this value are removed. Ignored if set to 0.
        prominence_kernel_size : odd integer
            window size (footprint radius in pixels) used to determine the
            background value used in `min_prominence` calculation
        min_rel_intensity : number
            maxima with intensities below a reference maximum * (this value) are
            removed. The reference maximum is selected for each diffraction image
            according to the `min_rel_ref_peak` argument
        min_rel_ref_peak : int
            specifies the reference maximum used in `min_rel_intensity`
            calculation. 0 = brightest maximum, 1 = second brightest maximum, etc
        edge_filter : number
            maxima closer to the edge of the image than this value are removed
        subpixel: False, 'none', 'poly', or 'multicorr'
            determines if and how the outputs are refined to better precision
            than the pixel grid.  If False or 'none' is passed subpixel refinement
            is skipped.  If 'poly' is used a polynomial (2D parabolic) numerical
            fit is performed and if 'multicorr' is used Fourier upsampling (more
            precise but slower) is used, with a precision determined by the value
            of `upsample_factor`
        upsample_factor : int
            the upsampling factor used for 'multicorr' subpixel refinement.
            Determines the precision of the refinement. Ignored if `subpixel` is
            not 'multicorr'
        device : None or dict
            If False, uses the CPU.  Otherwise, should be a dictionary with
            valid keys:
                * CUDA (bool): enable GPU acceleration
                * CUDA_batched (bool): enable batched GPU computation
                * ipyparallel (bool): enable multinode distribution using
                  ipyparallel. Must also specify the "client" and "data_file",
                  and optionally "cluster_path" keywords
                * dask (bool): enable multinode distribution using
                  dask. Must also specify the "client" and "data_file",
                  and optionally "cluster_path" keywords
                * client (str or obj): when used with ipyparralel, must be a
                  path to a client json for connecting to your existing
                  IPyParallel cluster. When used with dask, must be a dask
                  client that connects to your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): working directory for cluster processing,
                  defaults to current directory
            Note that only one of 'CUDA' or 'ipyparallel' or 'dask' may be set
            to True, and that 'client' and 'data_file' and optionally
            'cluster_path' must be specified if either 'ipyparallel' or 'dask'
            computation is selected. Note also that preprocessing is currently
            not performed for any device accelerated computations.
        ML : False or dict
            If False, does cross correlative template matching.  Otherwise, should
            be a dictionary with valid keys:
                * num_attempts (int): Number of attempts to predict the Bragg
                  disks. More attempts (ideally) results in a more confident
                  prediction, as FCU-net uses Monte Carlo dropout to estimate
                  the model uncertainty.  Note that increasing num_attempts will
                  increase the compute time and it is adviced to use GPU (CUDA)
                  acceleration for num_attempts > 1.
                  # TODO: @alex-rakowski - you had "Recommended: 5" but also
                  # had the default set to 1, and this comment that using >1
                  # is not recommended without CUDA.  Can we clarify?
                * batch_size (int): number of diffraction images to send to
                  the model at once for prediction. For CPU a batch size of 1
                  is recommended.  For GPU the batch size may be selected based
                  on the available GPU RAM and the size of the diffraction
                  images, with larger batch sizes accelerating the computation
                  and increasing the required memory.
                * model_path (None or str): if None, py4DSTEM will check if a
                  model is available locally, then download and update the model
                  if one is not available or the local model is not up-to-date.
                  Otherwise, must be a filepath to a Tensorflow model of weights
            Note that to use GPU / batched GPU computation, the "CUDA" and
            "CUDA_batched" flags should be set to True in the `device` arguement.
        return_cc : bool
            If True, returns the cross correlation in addition to the detected
            peaks.
        name : str
            Name for the output BraggVectors instance
        returncalc : bool
            If True, return the answer
        """
        # handle deprecated inputs
        if 'minAbsoluteIntensity' in kwargs:
            FutureWarning("'minAbsoluteIntensity' is deprecated and will be removed in a future version; use 'min_intensity' instead")
        if 'minRelativeIntensity' in kwargs:
            FutureWarning("'minRelativeIntensity' is deprecated and will be removed in a future version; use 'min_rel_intensity' instead")
        if 'relativeToPeak' in kwargs:
            FutureWarning("'relativeToPeak' is deprecated and will be removed in a future version; use 'min_rel_ref_peak' instead")
        if 'minPeakSpacing' in kwargs:
            FutureWarning("'minPeakSpacing' is deprecated and will be removed in a future version; use 'min_spacing' instead")
        if 'edgeBoundary' in kwargs:
            FutureWarning("'edgeBoundary' is deprecated and will be removed in a future version; use 'edge_filter' instead")
        if 'maxNumPeaks' in kwargs:
            FutureWarning("'maxNumPeaks' is deprecated and will be removed in a future version; use 'n_peaks_max' instead")

        self.update_defaults(
            template=template,
            data=data,
            preprocess=preprocess,
            corr_power=corr_power,
            corr_sigma=corr_sigma,
            sigma=sigma,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            min_intensity=min_intensity,
            min_prominence=min_prominence,
            prominence_kernel_size=prominence_kernel_size,
            min_rel_intensity=min_rel_intensity,
            min_rel_ref_peak=min_rel_ref_peak,
            min_spacing=min_spacing,
            edge_filter=edge_filter,
            n_peaks_max=n_peaks_max,
            device=device,
            ML=ML,
            show_peaks=show_peaks,
            show_peaks_params=show_peaks_params,
        )
        params = self.bragg_detection_defaults
        template = params['template']
        data = params['data']
        preprocess = params['preprocess']
        corr_power = params['corr_power']
        corr_sigma = params['corr_sigma']
        sigma = params['sigma']
        subpixel = params['subpixel']
        upsample_factor = params['upsample_factor']
        min_intensity = params['min_intensity']
        min_prominence = params['min_prominence']
        prominence_kernel_size = params['prominence_kernel_size']
        min_rel_intensity = params['min_rel_intensity']
        min_rel_ref_peak = params['min_rel_ref_peak']
        min_spacing = params['min_spacing']
        edge_filter = params['edge_filter']
        n_peaks_max = params['n_peaks_max']
        device = params['device']
        ML = params['ML']
        show_peaks = params['show_peaks']
        show_peaks_params = params['show_peaks_params']

        # ensure there is a template or no cross-correlation has been selected
        if template is None:
            raise Exception('Please set the cross-correlation template with the `template` input. To skip cross-correlation, set it to `False`.')
        # use device?
        if device:
            raise Exception("Hardware acceleration isn't implemented here yet, please use find_Bragg_disks")
        # use ML?
        if ML:
            raise Exception("ML isn't implemented here yet, please use find_Bragg_disks")


        ## Set up methods (preprocess, crosscorr, threshold)

        # preprocess
        preprocess_options_rbs = [
            'radial_background_subtraction',
            'background_subtraction',
            'rbs',
            'bs'
        ]
        preprocess_options_la = [
            'local_averaging',
            'local_average',
            'la'
        ]
        preprocess_options = \
            preprocess_options_rbs + \
            preprocess_options_la
        # validate inputs
        if isinstance(preprocess,list):
            for el in preprocess:
                assert(isinstance(el,str) or callable(el))
                if isinstance(el,str):
                    assert(el in preprocess_options), f"preprocessing options must be functions or strings from the options {preprocess_options}; received input {el}"
        # define the method
        f = None
        def _preprocess(dp,x,y):
            """ dp = _preprocess_pattern(datacube.data[x,y])
            """
            if preprocess is False:
                return dp
            elif callable(preprocess):
                return preprocess(dp)
            # for dicts, element 'f' is a callable and
            # all others are arguments to pass to it
            elif isinstance(preprocess,dict):
                nonlocal f
                if f is None:
                    assert(np.logical_xor(
                        'f' in preprocess.keys(),
                        'function' in preprocess.keys()
                        )), "`preprocess` dict must contain exactly one of the keys 'f', 'function'"
                    if 'f' in preprocess.keys():
                        f = preprocess.pop('f')
                    else:
                        f = preprocess.pop('function')
                # if x+y are keys in preprocess, the callable f should
                # recieve scan positions 'x' and 'y' as inputs
                if 'x' in preprocess.keys() and 'y' in preprocess.keys():
                    preprocess['x'] = x
                    preprocess['y'] = y
                return f(dp,**preprocess)
            else:
                for el in preprocess:
                    if callable(el):
                        dp = el(dp)
                    else:
                        if el in preprocess_options_rbs:
                            dp = self.get_radial_bksb_dp(x,y,sigma=0)
                            pass
                        elif el in preprocess_options_la:
                            dp = self.get_local_ave_dp(x,y)
                        else:
                            raise Exception(f"Unrecognized preprocess option {el}")
                return dp

        # cross correlate
        def _cross_correlate(dp):
            """ cc = _cross_correlate(dp)
            """
            dp = dp.astype(np.float32)
            if corr_sigma > 0:
                dp = gaussian_filter(dp, corr_sigma)
            if template is not False:
                cc = get_cross_correlation_FT(
                    dp,
                    template_FT,
                    corr_power,
                    "fourier",
                )
            else:
                cc = dp
            return cc

        # threshold
        def _threshold(cc):
            """ vec = _threshold(cc)
            """
            cc_real = np.maximum(np.real(np.fft.ifft2(cc)),0)
            if subpixel == 'multicorr':
                return get_maxima_2D(
                    cc_real,
                    subpixel='multicorr',
                    upsample_factor=upsample_factor,
                    sigma=sigma,
                    minAbsoluteIntensity=min_intensity,
                    minProminence=min_prominence,
                    prominenceKernelSize=prominence_kernel_size,
                    minRelativeIntensity=min_rel_intensity,
                    relativeToPeak=min_rel_ref_peak,
                    minSpacing=min_spacing,
                    edgeBoundary=edge_filter,
                    maxNumPeaks=n_peaks_max,
                    _ar_FT=cc,
                )
            else:
                return get_maxima_2D(
                    cc_real,
                    subpixel=subpixel,
                    sigma=sigma,
                    minAbsoluteIntensity=min_intensity,
                    minProminence=min_prominence,
                    prominenceKernelSize=prominence_kernel_size,
                    minRelativeIntensity=min_rel_intensity,
                    relativeToPeak=min_rel_ref_peak,
                    minSpacing=min_spacing,
                    edgeBoundary=edge_filter,
                    maxNumPeaks=n_peaks_max,
                )

        # prepare the template
        if template is not False:
            if isinstance(template,Probe):
                assert(np.any(template.kernel)), "template.kernel is not populated - try running template.get_kernel"
                template_FT = np.conj(np.fft.fft2(template.kernel)).astype(np.complex64)
            else:
                template_FT = np.conj(np.fft.fft2(template)).astype(np.complex64)


        # use device?
        if device:
            #raise Exception("Hardware acceleration isn't implemented here yet, please use find_Bragg_disks")
            # goto cuda
            return

        # prepare the data and output container for...
        data_options_selected = ['s','selected']
        data_options_all = ['a','all']
        data_options = data_options_selected + data_options_all
        if data in data_options_all:
            # ...all indices
            rxs = np.tile(np.arange(self.R_Nx),self.R_Ny)
            rys = np.tile(np.arange(self.R_Ny),(self.R_Nx,1)).T.reshape(self.R_N)
            N = len(rxs)
            vectors = BraggVectors(
                self.Rshape,
                self.Qshape,
                calibration=self.calibration
            )
        elif data in data_options_selected:
            assert(self.selected_patterns.n > 0), "no diffraction patterns selected - use .select_patterns"
            data_pos = self.selected_patterns.pos
            rxs,rys = data_pos[0,:],data_pos[1,:]
            N = len(rxs)
            vectors = []
            if _return_cc:
                ccs = []
        else:
            raise Exception(f"`data` must be in {data_options}; recived value {data}")


        # Compute
        for idx in tqdmnd(
            N,
            desc="Finding Bragg Disks",
            unit="DP",
            unit_scale=True,
        ):
            # get a diffraction pattern
            rx,ry = rxs[idx],rys[idx]
            dp = self.data[rx,ry]

            # preprocess
            dp = _preprocess(dp,rx,ry)

            # cross correlate
            cc = _cross_correlate(dp)

            # threshold
            peaks = _threshold(cc)

            # store results
            peaks = QPoints(peaks)
            if data in data_options_all:
                vectors._v_uncal[rx,ry] = peaks
            else:
                vectors.append(peaks)
                if _return_cc:
                    ccs.append(cc)

        # Attach metadata, link datacube
        if data in data_options_all:
            vectors.metadata = Metadata(
                name="gen_params",
                data={
                    "_calling_method": inspect.stack()[0][3],
                    "_calling_class": __class__.__name__,
                } | params,
            )
            self.braggvectors = vectors

        # Show
        if data not in data_options_all and show_peaks:
            self.show_selected_patterns(
                peaks = vectors,
                **show_peaks_params,
            )

        # Return
        if _return_cc is True:
            return vectors, ccs
        else:
            return vectors




