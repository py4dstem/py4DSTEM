# Functions for finding Bragg disks using AI/ML pipeline (CUDA version)
# Joydeep Munshi
'''

Functions for finding Braggdisks (AI/ML) using cupy and tensorflow-gpu

'''

import numpy as np
from time import time

try:
    from numba import cuda
except ImportError:
    raise ImportError("Import Error: Please install numba before proceeding")

try:
    import cupy as cp
except:
    raise ImportError("Import Error: Please install cupy before proceeding")

try:
    import tensorflow as tf
except:
    raise ImportError("Please install tensorflow before proceeding - please check " + "https://www.tensorflow.org/install" + "for more information")

from cupyx.scipy.ndimage import gaussian_filter

from .diskdetection_aiml import _get_latest_model

from ...io import PointList, PointListArray
from ..utils import tqdmnd
from .kernels import kernels
from .diskdetection import universal_threshold

def find_Bragg_disks_aiml_CUDA(datacube, probe,
                          num_attmpts = 5,
                          int_window_radius = 1,
                          predict = True,
                          batch_size = 8,
                          sigma = 0,
                          edgeBoundary = 20,
                          minRelativeIntensity = 0.005,
                          minAbsoluteIntensity = 0,
                          relativeToPeak = 0,
                          minPeakSpacing = 60,
                          maxNumPeaks = 70,
                          subpixel = 'multicorr',
                          upsample_factor = 16,
                          global_threshold = False,
                          minGlobalIntensity = 0.005,
                          metric = 'mean',
                          filter_function = None,
                          name = 'braggpeaks_raw',
                          _qt_progress_bar = None,
                          model_path=None):
    """
    Finds the Bragg disks in all diffraction patterns of datacube by AI/ML method (CUDA version)
    This method utilizes FCU-Net to predict Bragg disks from diffraction images.
    
     Args:
        datacube (datacube): a diffraction datacube
        probe (ndarray): the vacuum probe template
        num_attmpts (int): Number of attempts to predict the Bragg disks. Recommended: 5.
            Ideally, the more num_attmpts the better (confident) the prediction will be
            as the ML prediction utilizes Monte Carlo Dropout technique to estimate model
            uncertainty using Bayesian approach. Note: increasing num_attmpts will increase
            the compute time significantly and it is advised to use GPU (CUDA) enabled environment
            for fast prediction with num_attmpts > 1
        int_window_radius (int): window radius (in pixels) for disk intensity integration over the 
            predicted atomic potentials array
        predict (bool): Flag to determine if ML prediction is opted.
        batch_size (int): batch size for Tensorflow model.predict() function, by default batch_size = 2,
            Note: if you are using CPU for model.predict(), please use batch_size < 2. Future version 
            will implement Dask parrlelization implementation of the serial function to boost up the 
            performance of Tensorflow CPU predictions. Keep in mind that this funciton will take
            significant amount of time to predict for all the DPs in a datacube.
        edgeBoundary (int): minimum acceptable distance from the DP edge, in pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak intensity,
            relative to the intensity of the relativeToPeak'th peak
        minAbsoluteIntensity (float): the minimum acceptable correlation peak intensity,
            on an absolute scale
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
        model_path (str): filepath for the model weights (Tensorflow model) to load from.
            By default, if the model_path is not provided, py4DSTEM will search for the 
            latest model stored on cloud using metadata json file. It is not recommended to
            keep track of the model path and advised to keep this argument unchanged (None)
            to always search for the latest updated training model weights.

    Returns:
        (PointListArray): the Bragg peak positions and correlation intensities
    """

    # Make the peaks PointListArray
    coords = [('qx',float),('qy',float),('intensity',float)]
    peaks = PointListArray(coordinates=coords, shape=(datacube.R_Nx, datacube.R_Ny))

    # check that the filtered DP is the right size for the probe kernel:
    if filter_function: assert callable(filter_function), "filter_function must be callable"
    DP = datacube.data[0,0,:,:] if filter_function is None else filter_function(datacube.data[0,0,:,:])
    assert np.all(DP.shape == probe.shape), 'Probe kernel shape must match filtered DP shape'

    get_maximal_points = kernels['maximal_pts_float64']

    if get_maximal_points.max_threads_per_block < DP.shape[1]:
        blocks = ((np.prod(DP.shape)//get_maximal_points.max_threads_per_block + 1),)
        threads = ((get_maximal_points.max_threads_per_block))
    else:
        blocks = (DP.shape[0],)
        threads = (DP.shape[1],)
        
    if predict:
        t0 = time()
        model = _get_latest_model(model_path = model_path)
        probe = tf.expand_dims(tf.repeat(tf.expand_dims(probe, axis=0), 
                                             datacube.R_Nx*datacube.R_Ny, axis=0), axis=-1)
        DP = tf.expand_dims(tf.reshape(datacube.data,
                                      (datacube.R_Nx*datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny)), axis = -1)
            
        prediction = np.zeros(shape = (datacube.R_Nx*datacube.R_Ny, datacube.Q_Nx, datacube.Q_Ny, 1))
        
        image_num = datacube.R_Nx*datacube.R_Ny
        batch_num = int(image_num//batch_size)

        for att in tqdmnd(num_attmpts, desc='Neural network is predicting structure factors', unit='ATTEMPTS',unit_scale=True):
            for i in range(batch_num):
                prediction[i*batch_size:(i+1)*batch_size] += model.predict([DP[i*batch_size:(i+1)*batch_size],probe[i*batch_size:(i+1)*batch_size]])
            if (i+1)*batch_size < image_num:
                prediction[(i+1)*batch_size:] += model.predict([DP[(i+1)*batch_size:],probe[(i+1)*batch_size:]])
        
        print('Averaging over {} attempts \n'.format(num_attmpts))
        prediction = prediction/num_attmpts
    
        prediction = np.reshape(np.transpose(prediction, (0,3,1,2)),
                                (datacube.R_Nx, datacube.R_Ny, datacube.Q_Nx, datacube.Q_Ny))


    # Loop over all diffraction patterns
    for (Rx,Ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding Bragg Disks using AI/ML CUDA',unit='DP',unit_scale=True):
        DP = prediction[Rx,Ry,:,:]
        _find_Bragg_disks_aiml_single_DP_CUDA(DP, probe,
                                      num_attmpts = num_attmpts,
                                      int_window_radius = int_window_radius,
                                      predict = False,
                                      sigma = sigma,
                                      edgeBoundary = edgeBoundary,
                                      minRelativeIntensity = minRelativeIntensity,
                                      minAbsoluteIntensity=minAbsoluteIntensity,
                                      relativeToPeak = relativeToPeak,
                                      minPeakSpacing = minPeakSpacing,
                                      maxNumPeaks = maxNumPeaks,
                                      subpixel = subpixel,
                                      upsample_factor = upsample_factor,
                                      filter_function = filter_function,
                                      peaks = peaks.get_pointlist(Rx,Ry),
                                      get_maximal_points = get_maximal_points,
                                      blocks = blocks,
                                      threads = threads)
    t2 = time()-t0
    print("Analyzed {} diffraction patterns in {}h {}m {}s".format(datacube.R_N, int(t2/3600),
                                                                   int(t2/60), int(t2%60)))
    if global_threshold == True:
        peaks = universal_threshold(peaks, minGlobalIntensity, metric, minPeakSpacing,
                                    maxNumPeaks)
    peaks.name = name
    return peaks

def _find_Bragg_disks_aiml_single_DP_CUDA(DP, probe,
                                  num_attmpts = 5,
                                  int_window_radius = 1,
                                  predict = True,
                                  sigma = 0,
                                  edgeBoundary = 20,
                                  minRelativeIntensity = 0.005,
                                  minAbsoluteIntensity = 0,
                                  relativeToPeak = 0,
                                  minPeakSpacing = 60,
                                  maxNumPeaks = 70,
                                  subpixel = 'multicorr',
                                  upsample_factor = 16,
                                  filter_function = None,
                                  return_cc = False,
                                  peaks = None,
                                  get_maximal_points = None,
                                  blocks = None,
                                  threads = None):
    """
    Finds the Bragg disks in single DP by AI/ML method. This method utilizes FCU-Net
    to predict Bragg disks from diffraction images.
    
    The input DP and Probes need to be aligned before the prediction. Detected peaks within 
    edgeBoundary pixels of the diffraction plane edges are then discarded. Next, peaks
    with intensities less than minRelativeIntensity of the brightest peak in the
    correlation are discarded. Then peaks which are within a distance of minPeakSpacing
    of their nearest neighbor peak are found, and in each such pair the peak with the
    lesser correlation intensities is removed. Finally, if the number of peaks remaining
    exceeds maxNumPeaks, only the maxNumPeaks peaks with the highest correlation
    intensity are retained.
    
    Args:
        DP (ndarray): a diffraction pattern
        probe (ndarray): the vacuum probe template
        num_attmpts (int): Number of attempts to predict the Bragg disks. Recommended: 5
            Ideally, the more num_attmpts the better (confident) the prediction will be
            as the ML prediction utilizes Monte Carlo Dropout technique to estimate model
            uncertainty using Bayesian approach. Note: increasing num_attmpts will increase
            the compute time significantly and it is advised to use GPU (CUDA) enabled environment
            for fast prediction with num_attmpts > 1
        int_window_radius (int): window radius (in pixels) for disk intensity integration over the 
            predicted atomic potentials array
        predict (bool): Flag to determine if ML prediction is opted.
        edgeBoundary (int): minimum acceptable distance from the DP edge, in pixels
        minRelativeIntensity (float): the minimum acceptable correlation peak intensity,
            relative to the intensity of the relativeToPeak'th peak
        minAbsoluteIntensity (float): the minimum acceptable correlation peak intensity,
            on an absolute scale
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
        filter_function (callable): filtering function to apply to each diffraction
            pattern before peakfinding. Must be a function of only one argument (the
            diffraction pattern) and return the filtered diffraction pattern. The shape
            of the returned DP must match the shape of the probe kernel (but does not
            need to match the shape of the input diffraction pattern, e.g. the filter
            can be used to bin the diffraction pattern). If using distributed disk
            detection, the function must be able to be pickled with by dill.
        peaks (PointList): For internal use. If peaks is None, the PointList of peak
            positions is created here. If peaks is not None, it is the PointList that
            detected peaks are added to, and must have the appropriate coords
            ('qx','qy','intensity').
        model_path (str): filepath for the model weights (Tensorflow model) to load from.
            By default, if the model_path is not provided, py4DSTEM will search for the 
            latest model stored on cloud using metadata json file. It is not recommeded to
            keep track of the model path and advised to keep this argument unchanged (None)
            to always search for the latest updated training model weights.

     Returns:
         peaks                (PointList) the Bragg peak positions and correlation intensities
     """
    assert subpixel in [ 'none', 'poly', 'multicorr' ], "Unrecognized subpixel option {}, subpixel must be 'none', 'poly', or 'multicorr'".format(subpixel)

    if predict:
        assert(len(DP.shape)==2), "Dimension of single diffraction should be 2 (Qx, Qy)"
        assert(len(probe.shape)==2), "Dimension of Probe should be 2 (Qx, Qy)"
        
        model = _get_latest_model(model_path = model_path)
        DP = tf.expand_dims(tf.expand_dims(DP, axis=0), axis=-1)
        probe = tf.expand_dims(tf.expand_dims(probe, axis=0), axis=-1)
        prediction = np.zeros(shape = (1, DP.shape[1],DP.shape[2],1))
        
        for att in tqdmnd(num_attmpts, desc='Neural network is predicting structure factors', unit='ATTEMPTS',unit_scale=True):
            print('attempt {} \n'.format(att+1))
            prediction += model.predict([DP,probe])
        print('Averaging over {} attempts \n'.format(num_attmpts))
        pred = cp.array(prediction[0,:,:,0]/num_attmpts,dtype='float64')
    else:
        assert(len(DP.shape)==2), "Dimension of single diffraction should be 2 (Qx, Qy)"
        pred = cp.array(DP if filter_function is None else filter_function(DP),dtype='float64')

    # Find the maxima
    maxima_x,maxima_y,maxima_int = get_maxima_2D_cp(pred,
                                                    sigma=sigma,
                                                    edgeBoundary=edgeBoundary,
                                                    minRelativeIntensity=minRelativeIntensity,
                                                    minAbsoluteIntensity=minAbsoluteIntensity,
                                                    relativeToPeak=relativeToPeak,
                                                    minSpacing=minPeakSpacing,
                                                    maxNumPeaks=maxNumPeaks,
                                                    subpixel=subpixel,
                                                    upsample_factor = upsample_factor,
                                                    get_maximal_points=get_maximal_points,
                                                    blocks=blocks, threads=threads)
    
    maxima_x, maxima_y, maxima_int = _integrate_disks_cp(pred, maxima_x,maxima_y,maxima_int,int_window_radius=int_window_radius)

    # Make peaks PointList
    if peaks is None:
        coords = [('qx',float),('qy',float),('intensity',float)]
        peaks = PointList(coordinates=coords)
    else:
        assert(isinstance(peaks,PointList))
    peaks.add_tuple_of_nparrays((maxima_x,maxima_y,maxima_int))

    return peaks


def get_maxima_2D_cp(ar, 
                     sigma=0, 
                     edgeBoundary=0, 
                     minSpacing=0, 
                     minRelativeIntensity=0,
                     minAbsoluteIntensity=0,
                     relativeToPeak=0, 
                     maxNumPeaks=0, 
                     subpixel='poly', 
                     ar_FT = None,
                     upsample_factor=16,
                     get_maximal_points=None,
                     blocks=None,
                     threads=None):
    """
    Finds the indices where the 2D array ar is a local maximum.
    Optional parameters allow blurring of the array and filtering of the output;
    setting each of these to 0 (default) turns off these functions.

    Accepts:
        ar                      (ndarray) a 2D array
        sigma                   (float) guassian blur std to apply to ar before finding the maxima
        edgeBoundary            (int) ignore maxima within edgeBoundary of the array edge
        minSpacing              (float) if two maxima are found within minSpacing, the dimmer one
                                is removed
        minRelativeIntensity    (float) maxima dimmer than minRelativeIntensity compared to the
                                relativeToPeak'th brightest maximum are removed
        minAbsoluteIntensity   (float) the minimum acceptable correlation peak intensity,
                                on an absolute scale
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
    assert subpixel in [ 'none', 'poly', 'multicorr' ], "Unrecognized subpixel option {}, subpixel must be 'none', 'poly', or 'multicorr'".format(subpixel)

    # Get maxima
    ar = gaussian_filter(ar, sigma)
    maxima_bool = cp.zeros_like(ar,dtype=bool)
    sizex = ar.shape[0]
    sizey = ar.shape[1]
    N = sizex*sizey
    get_maximal_points(blocks,threads,(ar,maxima_bool,sizex,sizey,N))

    # Remove edges
    if edgeBoundary > 0:
        assert isinstance(edgeBoundary, (int, np.integer))
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
    dtype = np.dtype([('x', float), ('y', float), ('intensity', float)])
    maxima = np.zeros(len(maxima_x), dtype=dtype)
    maxima['x'] = maxima_x
    maxima['y'] = maxima_y

    ar = ar.get()
    maxima['intensity'] = ar[maxima_x, maxima_y]
    maxima = np.sort(maxima, order='intensity')[::-1]

    if len(maxima) > 0:
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

        # Remove maxima which are too dim
        if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
            assert isinstance(relativeToPeak, (int, np.integer))
            deletemask = maxima['intensity'] / maxima['intensity'][relativeToPeak] < minRelativeIntensity
            maxima = np.delete(maxima, np.nonzero(deletemask)[0])
            
        # Remove maxima which are too dim, absolute scale
        if (minAbsoluteIntensity > 0):
            deletemask = maxima['intensity'] < minAbsoluteIntensity
            maxima = np.delete(maxima, np.nonzero(deletemask)[0])

        # Remove maxima in excess of maxNumPeaks
        if maxNumPeaks > 0:
            assert isinstance(maxNumPeaks, (int, np.integer))
            if len(maxima) > maxNumPeaks:
                maxima = maxima[:maxNumPeaks]
                

        # Subpixel fitting 
        # For all subpixel fitting, first fit 1D parabolas in x and y to 3 points (maximum, +/- 1 pixel)
        if subpixel != 'none':
            for i in range(len(maxima)):
                Ix1_ = ar[int(maxima['x'][i]) - 1, int(maxima['y'][i])]
                Ix0 = ar[int(maxima['x'][i]), int(maxima['y'][i])]
                Ix1 = ar[int(maxima['x'][i]) + 1, int(maxima['y'][i])]
                Iy1_ = ar[int(maxima['x'][i]), int(maxima['y'][i]) - 1]
                Iy0 = ar[int(maxima['x'][i]), int(maxima['y'][i])]
                Iy1 = ar[int(maxima['x'][i]), int(maxima['y'][i]) + 1]
                deltax = (Ix1 - Ix1_) / (4 * Ix0 - 2 * Ix1 - 2 * Ix1_)
                deltay = (Iy1 - Iy1_) / (4 * Iy0 - 2 * Iy1 - 2 * Iy1_)
                maxima['x'][i] += deltax
                maxima['y'][i] += deltay
                maxima['intensity'][i] = linear_interpolation_2D_cp(ar, maxima['x'][i], maxima['y'][i])
        # Further refinement with fourier upsampling
        if subpixel == 'multicorr':
            if ar_FT is None:
                ar_FT = cp.conj(cp.fft.fft2(cp.array(ar)))
            else:
                ar_FT = cp.conj(ar_FT)
            for ipeak in range(len(maxima['x'])):
                xyShift = np.array((maxima['x'][ipeak],maxima['y'][ipeak]))
                # we actually have to lose some precision and go down to half-pixel
                # accuracy. this could also be done by a single upsampling at factor 2
                # instead of get_maxima_2D_cp.
                xyShift[0] = np.round(xyShift[0] * 2) / 2
                xyShift[1] = np.round(xyShift[1] * 2) / 2

                subShift = upsampled_correlation_cp(ar_FT,upsample_factor,xyShift)
                maxima['x'][ipeak]=subShift[0]
                maxima['y'][ipeak]=subShift[1]

    return maxima['x'], maxima['y'], maxima['intensity']


def upsampled_correlation_cp(imageCorr, upsampleFactor, xyShift):
    '''
    Refine the correlation peak of imageCorr around xyShift by DFT upsampling using cupy.

    Args:
        imageCorr (complex valued ndarray):
            Complex product of the FFTs of the two images to be registered
            i.e. m = np.fft.fft2(DP) * probe_kernel_FT;
            imageCorr = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
        upsampleFactor (int):
            Upsampling factor. Must be greater than 2. (To do upsampling
            with factor 2, use upsampleFFT, which is faster.)
        xyShift:
            Location in original image coordinates around which to upsample the
            FT. This should be given to exactly half-pixel precision to
            replicate the initial FFT step that this implementation skips

    Returns:
        (2-element np array): Refined location of the peak in image coordinates.
    '''
    
    #-------------------------------------------------------------------------------------
    #There are two approaches to Fourier upsampling for subpixel refinement: (a) one
    #can pad an (appropriately shifted) FFT with zeros and take the inverse transform,
    #or (b) one can compute the DFT by matrix multiplication using modified
    #transformation matrices. The former approach is straightforward but requires
    #performing the FFT algorithm (which is fast) on very large data. The latter method
    #trades one speedup for a slowdown elsewhere: the matrix multiply steps are expensive
    #but we operate on smaller matrices. Since we are only interested in a very small
    #region of the FT around a peak of interest, we use the latter method to get
    #a substantial speedup and enormous decrease in memory requirement. This
    #"DFT upsampling" approach computes the transformation matrices for the matrix-
    #multiply DFT around a small 1.5px wide region in the original `imageCorr`.

    #Following the matrix multiply DFT we use parabolic subpixel fitting to
    #get even more precision! (below 1/upsampleFactor pixels)

    #NOTE: previous versions of multiCorr operated in two steps: using the zero-
    #padding upsample method for a first-pass factor-2 upsampling, followed by the
    #DFT upsampling (at whatever user-specified factor). I have implemented it
    #differently, to better support iterating over multiple peaks. **The DFT is always
    #upsampled around xyShift, which MUST be specified to HALF-PIXEL precision
    #(no more, no less) to replicate the behavior of the factor-2 step.**
    #(It is possible to refactor this so that peak detection is done on a Fourier
    #upsampled image rather than using the parabolic subpixel and rounding as now...
    #I like keeping it this way because all of the parameters and logic will be identical
    #to the other subpixel methods.)
    #-------------------------------------------------------------------------------------

    assert upsampleFactor > 2

    xyShift[0] = np.round(xyShift[0] * upsampleFactor) / upsampleFactor
    xyShift[1] = np.round(xyShift[1] * upsampleFactor) / upsampleFactor

    globalShift = np.fix(np.ceil(upsampleFactor * 1.5)/2)

    upsampleCenter = globalShift - upsampleFactor*xyShift

    imageCorrUpsample = cp.conj(dftUpsample_cp(imageCorr, upsampleFactor, upsampleCenter )).get()

    xySubShift = np.unravel_index(imageCorrUpsample.argmax(), imageCorrUpsample.shape)

    # add a subpixel shift via parabolic fitting
    try:
        icc = np.real(imageCorrUpsample[xySubShift[0] - 1 : xySubShift[0] + 2, xySubShift[1] - 1 : xySubShift[1] + 2])
        dx = (icc[2,1] - icc[0,1]) / (4 * icc[1,1] - 2 * icc[2,1] - 2 * icc[0,1])
        dy = (icc[1,2] - icc[1,0]) / (4 * icc[1,1] - 2 * icc[1,2] - 2 * icc[1,0])
    except:
        dx, dy = 0, 0 # this is the case when the peak is near the edge and one of the above values does not exist

    xySubShift = xySubShift - globalShift

    xyShift = xyShift + (xySubShift + np.array([dx, dy])) / upsampleFactor

    return xyShift


def dftUpsample_cp(imageCorr, upsampleFactor, xyShift):
    '''
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
        xyShift (list of 2 floats):
            Coordinates in the UPSAMPLED GRID around which to upsample.
            These must be single-pixel IN THE UPSAMPLED GRID

    Returns:
        (ndarray):
            Upsampled image from region around correlation peak.
    '''
    imageSize = imageCorr.shape
    pixelRadius = 1.5
    numRow = np.ceil(pixelRadius * upsampleFactor)
    numCol = numRow

    colKern = cp.exp(
    (-1j * 2 * cp.pi / (imageSize[1] * upsampleFactor))
    * cp.outer( (cp.fft.ifftshift( (cp.arange(imageSize[1])) ) - cp.floor(imageSize[1]/2)),  (cp.arange(numCol) - xyShift[1]))
    )

    rowKern = cp.exp(
    (-1j * 2 * cp.pi / (imageSize[0] * upsampleFactor))
    * cp.outer( (cp.arange(numRow) - xyShift[0]), (cp.fft.ifftshift(cp.arange(imageSize[0])) - cp.floor(imageSize[0]/2)))
    )

    imageUpsample = cp.real(rowKern @ imageCorr @ colKern)
    return imageUpsample

def linear_interpolation_2D_cp(ar, x, y):
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

def _integrate_disks_cp(DP, maxima_x,maxima_y,maxima_int,int_window_radius=1):    
    disks = []
    DP = cp.asnumpy(DP)
    img_size = DP.shape[0]
    for x,y,i in zip(maxima_x,maxima_y,maxima_int):
        r1,r2 = np.ogrid[-x:img_size-x, -y:img_size-y]
        mask = r1**2 + r2**2 <= int_window_radius**2
        mask_arr = np.zeros((img_size, img_size))
        mask_arr[mask] = 1
        disk = DP*mask_arr
        disks.append(np.average(disk))
    try:
        disks = disks/max(disks)
    except:
        pass
    return (maxima_x,maxima_y,disks)
