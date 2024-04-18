import numpy as np
from typing import Optional
from scipy.ndimage import gaussian_filter

from emdfile import tqdmnd
from emdfile import Metadata
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
        pass


    def find_bragg_vectors(
        self,
        template,
        data=None,
        preprocess=None,
        corr=None,
        thresh=None,
        device=None,
        ML=None,
        return_cc=False,
        name = 'braggvectors',
        returncalc = True,
        _return_cc = False,
    ):

        """
        Finds Bragg scattering vectors.

        In normal operation, localizes Bragg scattering using template matching,
        by (1) optional preprocessing, (2) cross-correlating with the template,
        and (3) finding local maxima, thresholding and returning. See
        `preprocess`, `corr`, and `thresh` below. Accelration is handle with
        `device`.

        Invoking `ML` makes use of a custom neural network called FCU-net
        instead of template matching. If you use FCU-net in your work,
        please reference "Munshi, Joydeep, et al. npj Computational Materials
        8.1 (2022): 254".


        Examples (CPU + cross-correlation)
        ----------------------------------

        >>> datacube.get_bragg_vectors( template )

        will find bragg scattering for the entire datacube using cross-
        correlative template matching on the CPU with a correlation power of 1,
        gaussian blurring on each correlagram of 2 pixels, polynomial subpixel
        refinement, and the default thresholding parameters.

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     corr = {
        >>>         'corrPower' : 1,
        >>>         'sigma' : 2,
        >>>         'subpixel' : 'multicorr',
        >>>         'upsample_factor' : 16
        >>>     },
        >>> )

        will perform the same computation but use Fourier upsampling for
        subpixel refinement, and

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     thresh = {
        >>>         'minAboluteIntensity' : 100,
        >>>         'minPeakSpacing' : 18,
        >>>         'edgeBoundary' : 10,
        >>>         'maxNumPeaks' : 100,
        >>>     },
        >>> )

        will perform the same computation but threshold the detected
        maxima using an absolute rather than relative intensity threshhold,
        and modify the other threshold params as above.

        Using

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (5,6)
        >>> )

        will perform template matching against the diffraction image at
        scan position (5,6), and using

        >>> datacube.get_bragg_vectors(
        >>>     template,
        >>>     data = (np.array([4,5,6]),np.array([10,11,12]))
        >>> )

        will perform template matching against the 3 diffraction images at
        scan positions (4,10), (5,11), (6,12).

        Using

        >>> datacube.fing_bragg_vectors(
        >>>     template = None,
        >>>     corr = {
        >>>         'sigma' : 5,
        >>>         'subpixel' : 'poly'
        >>>     },
        >>> )

        will not cross-correlate at all, and will instead perform maximum
        detection on the raw data of the entire datacube, after applying
        a gaussian blur to each diffraction image of 5 pixels, and using
        polynomial subpixel refinement.

        Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'sigma' : 2,
        >>>     },
        >>>     corr = {
        >>>         'sigma' : 4,
        >>>     }
        >>> )

        will apply a 2-pixel gaussian blur to the diffraction image, then
        cross correlate, then apply a 4-pixel gaussian blur to the cross-
        correlation before finding maxima. Using

        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'radial_bkgrd' : True,
        >>>         'localave' : True,
        >>>     },
        >>> )

        will subtract the radial median from each diffraction image, then
        obtain the weighted average diffraction image with a 3x3 gaussian
        footprint in real space (i.e.

            [[ 1, 2, 1 ],
             [ 2, 4, 2 ],  *  (1/16)
             [ 1, 2, 1 ]]

        ) and perform template matching against the resulting images.
        Using

        >>> def preprocess_fn(data):
        >>>     return py4DSTEM.utils.bin2D(data,2)
        >>> datacube.find_bragg_vectors(
        >>>     template,
        >>>     preprocess = {
        >>>         'filter_function' : preprocess_fn
        >>>     },
        >>> )

        will bin each diffraction image by 2 before cross-correlating.
        Note that the template shape must match the final, preprocessed
        data shape.


        Examples (GPU acceleration, cluster acceleration, and ML)
        -------------------------------------------------------
        # TODO!


        Parameters
        ----------
        template : qshape'd 2d np.ndarray or Probe or None
            The matching template. If an ndarray is passed, must be centered
            about the origin. If a Probe is passed, probe.kernel must be
            populated. If None is passed, cross correlation is skipped and
            the maxima are taken directly from the (possibly preprocessed)
            diffraction data
        data : None or 2-tuple or 2D numpy ndarray
            Specifies the data in which to find the Bragg scattering. Valid
            entries and their behavoirs are:
                * None: use the entire DataCube, and return a BraggVectors
                  instance
                * 2-tuple of ints: use the diffraction pattern at scan
                  position (rx,ry), and return a QPoints instance
                * 2-tuple of arrays of ints: use the diffraction patterns
                  at scan positions (rxs,rys), and return a list of QPoints
                  instances
                * 2D numpy array, real-space shaped, boolean: run on the
                  diffraction images specified by the True pixels in the
                  input array, and return a list of QPoints instances
                * 2D numpy array, diffraction-space shaped: run on the
                  input array, and return a QPoints instance
        preprocess : None or dict
            If None, no preprocessing is performed.  Otherwise, should be a
            dictionary with the following valid keys:
                * radial_bkgrd (bool): if True, finds and subtracts the local
                  median of each diffraction image.  Origin must be calibrated.
                * localave (bool): if True, takes the local 3x3 gaussian
                  average of each diffraction image
                * sigma (number): if >0, applies a gaussian blur to the data
                  before cross correlating
                * filter_function (callable): function applied to each
                  diffraction image before peak finding. Must be a function of
                  only one argument (the diffr image) which returns the pre-
                  processed image. The shape of the returned DP must match
                  the shape of the probe. If using distributed disk detection,
                  the function must be able to be pickled with dill.
            If more than one key is passed then all requested preprocessing
            steps are performed, in the order they're listed here.
        corr : None or dict
            If None, no cross correlation is performed, and maximum detection
            is performed on the (possibly preprocessed) diffraction data with
            no subpixel refinement applied. Otherwise, should be a dictionary
            with valid keys:
                * corrPower (number, 0-1): type of correlation to perform,
                  where 1 is a cross correlation, 0 is a phase correlation,
                  and values in between are hybrid correlations. Pure cross
                  correlation is recommend to minimize noise
                * sigma (number): if >0, apply a gaussian blur to the cross
                  correlation before detecting maxima
                * subpixel ('none' or 'poly' or 'multicorr'): controls
                  subpixel refinement of maxima. 'none' returns the values
                  to pixel precision. 'poly' performs polynomial (2D
                  parabolic) numerical refinement. 'multicorr' performs
                  Fourier upsampling subpixel refinement, and requires
                  the `upsample_factor` keyword also be specified. 'poly'
                  is fast; 'multicorr' is much slower but allows greater
                  precision.
                * upsample_factor (int): the upsampling factor used for
                  'multicorr' subpixel refinement and defining the precision
                  of the refinement. Ignored if `subpixel` is not 'multicorr'
            Note that passing `template=None` skips cross-correlation but not
            maxmimum detection - in this case, `corrPower` is ignored, but all
            parameters in this dictionary are used.
            Note also that if this dictionary is specified (i.e. is not None)
            but corrPower or sigma or subpixel or upsample_factor are not
            specified, their default values (corrPower=1, sigma=2,
            subpixel='poly', upsample_factor=16) are used.
        thresh : None or dict
            If None, no thresholding is performed (not recommended!).  Otherwise,
            should be a dictionary with valid keys:
                * minAbsoluteIntensity (number): maxima with intensities below
                  `this value are removed. Ignored if set to 0.
                * minRelativeIntensity (number): maxima with intensities below a
                  reference maximum * (this value) are removed. The refernce
                  maximum is selected for each diffraction image according to
                  the `relativeToPeak` argument: 0 specifies the brightest
                  maximum, 1 specifies the second brightest, etc.
                * relativeToPeak (int): specifies the reference maximum used
                  int the `minRelativeIntensity` threshold
                * minPeakSpacing (number): if two maxima are closer together than
                  this number of pixels, the dimmer maximum is removed
                * edgeBoundary (number): maxima closer to the edge of the
                  diffraction image than this value are removed
                * maxNumPeaks (int): only the brightest `maxNumPeaks` maxima
                  are kept
        device : None or dict
            If None, uses the CPU.  Otherwise, should be a dictionary with
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
        ML : None or dict
            If None, does cross correlative template matching.  Otherwise, should
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

        # use ML?
        if ML:
            raise Exception("ML isn't implemented here yet, please use find_Bragg_disks")
        # use device?
        if device:
            raise Exception("Hardware acceleration isn't implemented here yet, please use find_Bragg_disks")

        # parse inputs
        corr_defaults = {
            'sigma' : 0,
            'corr_power' : 1,
        }
        thresh_defaults = {
            'min_intensity' : 0,
            'min_spacing' : 5,
            'subpixel' : 'poly',
            'upsample_factor' : 16,
            'edge' : 0,
            'sigma' : 0,
            'n_peaks_max' : 10000,
            'min_prominence' : 0,
            'prominence_kernel_size' : 3,
            'min_rel_intensity' : 0,
            'ref_peak' : 0,
        }
        corr = corr_defaults if corr is None else corr_defaults | corr
        thresh = thresh_defaults if thresh is None else thresh_defaults | thresh

        ## Set up metamethods (preprocess, crosscorr, thresholding)

        # preprocess
        preprocess_options = [
            "bs",
            "radial_background_subtraction",
            "la",
            "local_averaging"
        ]
        f = None
        # validate inputs
        if isinstance(preprocess,list):
            for el in preprocess:
                assert(isinstance(el,str) or callable(el))
                if isinstance(el,str):
                    assert(el in preprocess_options)
        def _preprocess(dp,x,y):
            """ dp = _preprocess_pattern(datacube.data[x,y])
            """
            if preprocess is None:
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
                            raise Exception("How did you get here?? A preprocess option may have been added incorrectly")
                return dp

        # cross correlate
        def _cross_correlate(dp):
            """ cc = _cross_correlate(dp)
            """
            if corr['sigma'] > 0:
                dp = gaussian_filter(dp, corr['sigma'])
            if template is not None:
                cc = get_cross_correlation_FT(
                    dp,
                    template_FT,
                    corr['corr_power'],
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
            if thresh['subpixel'] == 'multicorr':
                return get_maxima_2D(
                    cc_real,
                    subpixel='multicorr',
                    upsample_factor=thresh['upsample_factor'],
                    sigma=thresh['sigma'],
                    minAbsoluteIntensity=thresh['min_intensity'],
                    minProminence=thresh['min_prominence'],
                    prominenceKernelSize=thresh['prominence_kernel_size'],
                    minRelativeIntensity=thresh['min_rel_intensity'],
                    relativeToPeak=thresh['ref_peak'],
                    minSpacing=thresh['min_spacing'],
                    edgeBoundary=thresh['edge'],
                    maxNumPeaks=thresh['n_peaks_max'],
                    _ar_FT=cc,
                )
            else:
                return get_maxima_2D(
                    cc_real,
                    subpixel=thresh['subpixel'],
                    sigma=thresh['sigma'],
                    minAbsoluteIntensity=thresh['min_intensity'],
                    minProminence=thresh['min_prominence'],
                    prominenceKernelSize=thresh['prominence_kernel_size'],
                    minRelativeIntensity=thresh['min_rel_intensity'],
                    relativeToPeak=thresh['ref_peak'],
                    minSpacing=thresh['min_spacing'],
                    edgeBoundary=thresh['edge'],
                    maxNumPeaks=thresh['n_peaks_max'],
                )


        # prepare the template
        if template is not None:
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
                calibration=self.calibration)
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
