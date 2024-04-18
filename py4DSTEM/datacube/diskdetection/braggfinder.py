import numpy as np
from scipy.ndimage import gaussian_filter
import inspect

from emdfile import tqdmnd, Metadata
from py4DSTEM.utils import get_maxima_2D, get_cross_correlation_FT
from py4DSTEM.data import QPoints
from py4DSTEM.braggvectors import BraggVectors


class BraggFinder(object):
    """
    Handles disk detection.
    """

    def __init__(
        self,
    ):
        self.bragg_detection_defaults = {
            'template' : None,
            'preprocess' : False,
            'corr_power' : 1,
            'corr_sigma' : 0,
            'sigma' : 2,
            'min_intensity' : 0,
            'min_spacing' : 5,
            'subpixel' : 'poly',
            'upsample_factor' : 16,
            'edge_boundary' : 1,
            'n_peaks_max' : 10000,
            'min_prominence' : 0,
            'prominence_kernel_size' : 3,
            'min_rel_intensity' : 0,
            'ref_peak' : 0,
            'device' : False,
            'ML' : False,
        }

    def update_defaults(
        self,
        template=None,
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
        ref_peak=None,
        min_spacing=None,
        edge_boundary=None,
        n_peaks_max=None,
        device=None,
        ML=None,
    ):
        # add new defaults to a dict
        new_defaults = {}
        if template is not None: new_defaults['template'] = template
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
        if ref_peak is not None: new_defaults['ref_peak'] = ref_peak
        if min_spacing is not None: new_defaults['min_spacing'] = min_spacing
        if edge_boundary is not None: new_defaults['edge_boundary'] = edge_boundary
        if n_peaks_max is not None: new_defaults['n_peaks_max'] = n_peaks_max
        if device is not None: new_defaults['device'] = device
        if ML is not None: new_defaults['ML'] = ML

        # update
        self.bragg_detection_defaults = self.bragg_detection_defaults | new_defaults


    def find_bragg_vectors(
        self,
        template,
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
        ref_peak=None,
        min_spacing=None,
        edge_boundary=None,
        n_peaks_max=None,
        device=None,
        ML=None,
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

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (np.array([4,5,6]),np.array([10,11,12]))
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
        data : None or 2-tuple or 2D numpy ndarray
            Specifies the input data and return value.
            If None, uses the full datacube and return a BraggVectors instance.
            If a 2-tuple (int,int), uses the diffraction pattern at this scan
            position and returns a QPoints instance.
            If a 2-tuple of arrays of ints, uses the diffraction patterns
            at scan positions (rxs,rys) and return a list of QPoints instance
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
        min_intensity : number
            maxima with intensities below this value are removed. Ignored if set
            to 0.
        minAbsoluteIntensity : number (Deprecated)
            alias for `min_intensity`
        min_prominence : number
            maxima with intensity differences relative to their background less
            than this value are removed. Ignored if set to 0.
        prominence_kernel_size : odd integer
            window size (footprint radius in pixels) used to determine the
            background value used in `min_prominence` calculation
        min_rel_intensity : number
            maxima with intensities below a reference maximum * (this value) are
            removed. The reference maximum is selected for each diffraction image
            according to the `ref_peak` argument
        minRelativeIntensity : number (Deprecated)
            alias for `min_rel_intensity`
        ref_peak : int
            specifies the reference maximum used in `min_rel_intensity`
            calculation. 0 = brightest maximum, 1 = second brightest maximum, etc
        relativeToPeak : int (Deprecated)
            alias for `ref_peak`
        min_spacing=None : number
            if two maxima are closer together than this value, the dimmer
            maximum is removed
        minPeakSpacing : number (Deprecated)
            alias for `min_spacing`
        edge_boundary : number
            maxima closer to the edge of the image than this value are removed
        edgeBoundary : number (Deprecated)
            alias for edge_boundary
        n_peaks_max : int
            only the brightest n_peaks_max peaks are returned
        maxNumPeaks : int (Deprecated)
            alias for `n_peaks_max`
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
            FutureWarning("'relativeToPeak' is deprecated and will be removed in a future version; use 'ref_peak' instead")
        if 'minPeakSpacing' in kwargs:
            FutureWarning("'minPeakSpacing' is deprecated and will be removed in a future version; use 'min_spacing' instead")
        if 'edgeBoundary' in kwargs:
            FutureWarning("'edgeBoundary' is deprecated and will be removed in a future version; use 'edge_boundary' instead")
        if 'maxNumPeaks' in kwargs:
            FutureWarning("'maxNumPeaks' is deprecated and will be removed in a future version; use 'n_peaks_max' instead")

        self.update_defaults(
            template=template,
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
            ref_peak=ref_peak,
            min_spacing=min_spacing,
            edge_boundary=edge_boundary,
            n_peaks_max=n_peaks_max,
            device=device,
            ML=ML,
        )
        params = self.bragg_detection_defaults

        # ensure there is a template or no cross-correlation has been selected
        if params['template'] is None:
            raise Exception('Please set the cross-correlation template with the `template` input. To skip cross-correlation, set it to `False`.')
        # use device?
        if device:
            raise Exception("Hardware acceleration isn't implemented here yet, please use find_Bragg_disks")
        # use ML?
        if ML:
            raise Exception("ML isn't implemented here yet, please use find_Bragg_disks")


        ## Set up metamethods (preprocess, crosscorr, threshold)

        # preprocess
        preprocess_options = [
            "bs",
            "radial_background_subtraction",
            "la",
            "local_averaging"
        ]
        # validate inputs
        preprocess = params['preprocess']
        if isinstance(preprocess,list):
            for el in preprocess:
                assert(isinstance(el,str) or callable(el))
                if isinstance(el,str):
                    assert(el in preprocess_options)
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
                    f = preprocess.pop('f')
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
                        if el in ('bs','radial_background_subtraction'):
                            dp = self.get_radial_bksb_dp(x,y,sigma=0)
                            pass
                        elif el in ('la','local_averaging'):
                            dp = self.get_local_ave_dp(x,y)
                        else:
                            raise Exception(f"Unrecognized preprocess option {el}")
                return dp

        # cross correlate
        def _cross_correlate(dp):
            """ cc = _cross_correlate(dp)
            """
            if params['corr_sigma'] > 0:
                dp = gaussian_filter(dp, params['corr_sigma'])
            if template is not False:
                cc = get_cross_correlation_FT(
                    dp,
                    template_FT,
                    params['corr_power'],
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
            if params['subpixel'] == 'multicorr':
                return get_maxima_2D(
                    cc_real,
                    subpixel='multicorr',
                    upsample_factor=params['upsample_factor'],
                    sigma=params['sigma'],
                    minAbsoluteIntensity=params['min_intensity'],
                    minProminence=params['min_prominence'],
                    prominenceKernelSize=params['prominence_kernel_size'],
                    minRelativeIntensity=params['min_rel_intensity'],
                    relativeToPeak=params['ref_peak'],
                    minSpacing=params['min_spacing'],
                    edgeBoundary=params['edge_boundary'],
                    maxNumPeaks=params['n_peaks_max'],
                    _ar_FT=cc,
                )
            else:
                return get_maxima_2D(
                    cc_real,
                    subpixel=params['subpixel'],
                    sigma=params['sigma'],
                    minAbsoluteIntensity=params['min_intensity'],
                    minProminence=params['min_prominence'],
                    prominenceKernelSize=params['prominence_kernel_size'],
                    minRelativeIntensity=params['min_rel_intensity'],
                    relativeToPeak=params['ref_peak'],
                    minSpacing=params['min_spacing'],
                    edgeBoundary=params['edge_boundary'],
                    maxNumPeaks=params['n_peaks_max'],
                )

        # prepare the template
        if template is not False:
            template_FT = np.conj(np.fft.fft2(template))

        # prepare the data and output container for...
        if data is None:
            # ...all indices
            rxs = np.tile(np.arange(self.R_Nx),self.R_Ny)
            rys = np.tile(np.arange(self.R_Ny),(self.R_Nx,1)).T.reshape(self.R_N)
            N = len(rxs)
            vectors = BraggVectors(
                self.Rshape,
                self.Qshape,
                calibration=self.calibration
            )
        elif isinstance(data,(tuple,list)):
            # ...specified indices
            rxs,rys = data
            N = len(rxs)
            vectors = []
            if _return_cc:
                ccs = []
        else:
            raise Exception(f"Invalid specification of data, {data}")


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
            if data is None:
                vectors._v_uncal[rx,ry] = peaks
            else:
                vectors.append(peaks)
                if _return_cc:
                    ccs.append(cc)

        # Attach metadata
        if data is None:
            vectors.metadata = Metadata(
                name="gen_params",
                data={
                    "_calling_method": inspect.stack()[0][3],
                    "_calling_class": __class__.__name__,
                } | params,
            )

        # Return
        if _return_cc is True:
            return vectors, ccs
        else:
            return vectors





    def get_beamstop_mask(
        self,
        threshold=0.25,
        distance_edge=2.0,
        include_edges=True,
        sigma=0,
        use_max_dp=False,
        scale_radial=None,
        name="mask_beamstop",
        returncalc=True,
    ):
        """
        This function uses the mean diffraction pattern plus a threshold to
        create a beamstop mask.

        Args:
            threshold (float):  Value from 0 to 1 defining initial threshold for
                beamstop mask, taken from the sorted intensity values - 0 is the
                dimmest pixel, while 1 uses the brighted pixels.
            distance_edge (float): How many pixels to expand the mask.
            include_edges (bool): If set to True, edge pixels will be included
                in the mask.
            sigma (float):
                Gaussain blur std to apply to image before thresholding.
            use_max_dp (bool):
                Use the max DP instead of the mean DP.
            scale_radial (float):
                Scale from center of image by this factor (can help with edge)
            name (string): Name of the output array.
            returncalc (bool): Set to true to return the result.

        Returns:
            (Optional): if returncalc is True, returns the beamstop mask

        """

        if scale_radial is not None:
            x = np.arange(self.data.shape[2]) * 2.0 / self.data.shape[2]
            y = np.arange(self.data.shape[3]) * 2.0 / self.data.shape[3]
            ya, xa = np.meshgrid(y - np.mean(y), x - np.mean(x))
            im_scale = 1.0 + np.sqrt(xa**2 + ya**2) * scale_radial

        # Get image for beamstop mask
        if use_max_dp:
            # if not "dp_mean" in self.tree.keys():
            #     self.get_dp_max();
            # im = self.tree["dp_max"].data.astype('float')
            if not "dp_max" in self._branch.keys():
                self.get_dp_max()
            im = self.tree("dp_max").data.copy().astype("float")
        else:
            if not "dp_mean" in self._branch.keys():
                self.get_dp_mean()
            im = self.tree("dp_mean").data.copy()

            # if not "dp_mean" in self.tree.keys():
            #     self.get_dp_mean();
            # im = self.tree["dp_mean"].data.astype('float')

        # smooth and scale if needed
        if sigma > 0.0:
            im = gaussian_filter(im, sigma, mode="nearest")
        if scale_radial is not None:
            im *= im_scale

        # Calculate beamstop mask
        int_sort = np.sort(im.ravel())
        ind = np.round(
            np.clip(int_sort.shape[0] * threshold, 0, int_sort.shape[0])
        ).astype("int")
        intensity_threshold = int_sort[ind]
        mask_beamstop = im >= intensity_threshold

        # clean up mask
        mask_beamstop = np.logical_not(binary_fill_holes(np.logical_not(mask_beamstop)))
        mask_beamstop = binary_fill_holes(mask_beamstop)

        # Edges
        if include_edges:
            mask_beamstop[0, :] = False
            mask_beamstop[:, 0] = False
            mask_beamstop[-1, :] = False
            mask_beamstop[:, -1] = False

        # Expand mask
        mask_beamstop = distance_transform_edt(mask_beamstop) < distance_edge

        # Wrap beamstop mask in a class
        x = Array(data=mask_beamstop, name=name)

        # Add metadata
        x.metadata = Metadata(
            name="gen_params",
            data={
                #'gen_func' :
                "threshold": threshold,
                "distance_edge": distance_edge,
                "include_edges": include_edges,
                "name": "mask_beamstop",
                "returncalc": returncalc,
            },
        )

        # Add to tree
        self.tree(x)

        # return
        if returncalc:
            return mask_beamstop










#        ### OLD CODE
#
#        elif mode == "datacube":
#            if distributed is None and CUDA == False:
#                mode = "dc_CPU"
#            elif distributed is None and CUDA == True:
#                if CUDA_batched == False:
#                    mode = "dc_GPU"
#                else:
#                    mode = "dc_GPU_batched"
#            else:
#                x = _parse_distributed(distributed)
#                connect, data_file, cluster_path, distributed_mode = x
#                if distributed_mode == "dask":
#                    mode = "dc_dask"
#                elif distributed_mode == "ipyparallel":
#                    mode = "dc_ipyparallel"
#                else:
#                    er = f"unrecognized distributed mode {distributed_mode}"
#                    raise Exception(er)
#        # overwrite if ML selected
#
#        # select a function
#        fn_dict = {
#            "dp": _find_Bragg_disks_single,
#            "dp_stack": _find_Bragg_disks_stack,
#            "dc_CPU": _find_Bragg_disks_CPU,
#            "dc_GPU": _find_Bragg_disks_CUDA_unbatched,
#            "dc_GPU_batched": _find_Bragg_disks_CUDA_batched,
#            "dc_dask": _find_Bragg_disks_dask,
#            "dc_ipyparallel": _find_Bragg_disks_ipp,
#            "dc_ml": find_Bragg_disks_aiml,
#        }
#        fn = fn_dict[mode]
#
#        # prepare kwargs
#        kws = {}
#        # distributed kwargs
#        if distributed is not None:
#            kws["connect"] = connect
#            kws["data_file"] = data_file
#            kws["cluster_path"] = cluster_path
#        # ML arguments
#        if ML == True:
#            kws["CUDA"] = CUDA
#            kws["model_path"] = ml_model_path
#            kws["num_attempts"] = ml_num_attempts
#            kws["batch_size"] = ml_batch_size
#
#        # if radial background subtraction is requested, add to args
#        if radial_bksb and mode == "dc_CPU":
#            kws["radial_bksb"] = radial_bksb
#
#        # run and return
#        ans = fn(
#            data,
#            template,
#            filter_function=filter_function,
#            corrPower=corrPower,
#            sigma_dp=sigma_dp,
#            sigma_cc=sigma_cc,
#            subpixel=subpixel,
#            upsample_factor=upsample_factor,
#            minAbsoluteIntensity=minAbsoluteIntensity,
#            minRelativeIntensity=minRelativeIntensity,
#            relativeToPeak=relativeToPeak,
#            minPeakSpacing=minPeakSpacing,
#            edgeBoundary=edgeBoundary,
#            maxNumPeaks=maxNumPeaks,
#            **kws,
#        )
#        return ans
#
#
#
#
#
#        # parse args
#        if data is None:
#            x = self
#        elif isinstance(data, tuple):
#            x = self, data[0], data[1]
#        elif isinstance(data, np.ndarray):
#            assert data.dtype == bool, "array must be boolean"
#            assert data.shape == self.Rshape, "array must be Rspace shaped"
#            x = self.data[data, :, :]
#        else:
#            raise Exception(f"unexpected type for `data` {type(data)}")
#
#
#
#
#        # compute
#        peaks = find_Bragg_disks(
#            data=x,
#            template=template,
#            radial_bksb=radial_bksb,
#            filter_function=filter_function,
#            corrPower=corrPower,
#            sigma_dp=sigma_dp,
#            sigma_cc=sigma_cc,
#            subpixel=subpixel,
#            upsample_factor=upsample_factor,
#            minAbsoluteIntensity=minAbsoluteIntensity,
#            minRelativeIntensity=minRelativeIntensity,
#            relativeToPeak=relativeToPeak,
#            minPeakSpacing=minPeakSpacing,
#            edgeBoundary=edgeBoundary,
#            maxNumPeaks=maxNumPeaks,
#            CUDA=CUDA,
#            CUDA_batched=CUDA_batched,
#            distributed=distributed,
#            ML=ML,
#            ml_model_path=ml_model_path,
#            ml_num_attempts=ml_num_attempts,
#            ml_batch_size=ml_batch_size,
#        )
#
#        if isinstance(peaks, Node):
#            # add metadata
#            peaks.name = name
#            peaks.metadata = Metadata(
#                name="gen_params",
#                data={
#                    #'gen_func' :
#                    "template": template,
#                    "filter_function": filter_function,
#                    "corrPower": corrPower,
#                    "sigma_dp": sigma_dp,
#                    "sigma_cc": sigma_cc,
#                    "subpixel": subpixel,
#                    "upsample_factor": upsample_factor,
#                    "minAbsoluteIntensity": minAbsoluteIntensity,
#                    "minRelativeIntensity": minRelativeIntensity,
#                    "relativeToPeak": relativeToPeak,
#                    "minPeakSpacing": minPeakSpacing,
#                    "edgeBoundary": edgeBoundary,
#                    "maxNumPeaks": maxNumPeaks,
#                    "CUDA": CUDA,
#                    "CUDA_batched": CUDA_batched,
#                    "distributed": distributed,
#                    "ML": ML,
#                    "ml_model_path": ml_model_path,
#                    "ml_num_attempts": ml_num_attempts,
#                    "ml_batch_size": ml_batch_size,
#                },
#            )
#
#            # add to tree
#            if data is None:
#                self.attach(peaks)
#
#        # return
#        if returncalc:
#            return peaks
#
#    # aliases
#    find_disks = find_bragg = find_bragg_disks = find_bragg_scattering = find_bragg_vectors
#
#
