# Functions for finding Bragg disks using AI/ML pipeline
# Joydeep Munshi

''' 
Functions for finding Braggdisks using AI/ML method using tensorflow 
'''

import os
import glob
import json
import shutil
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from time import time
from numbers import Number

from ...io import PointList, PointListArray
from ..utils import get_cross_correlation_fk, get_maxima_2D, tqdmnd

def find_Bragg_disks_aiml_single_DP(DP, probe,
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
                                     peaks = None,
                                     model_path = None):
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
        (PointList): the Bragg peak positions and correlation intensities
    """
    
    try:
        import crystal4D
    except:
        raise ImportError("Import Error: Please install crystal4D before proceeding")
        
    try:
        import tensorflow as tf
    except:
        raise ImportError("Please install tensorflow before proceeding - please check " + "https://www.tensorflow.org/install" + "for more information")

    
    assert subpixel in [ 'none', 'poly', 'multicorr' ], "Unrecognized subpixel option {}, subpixel must be 'none', 'poly', or 'multicorr'".format(subpixel)
    
    # Perform any prefiltering
    if filter_function: assert callable(filter_function), "filter_function must be callable"
    DP = DP if filter_function is None else filter_function(DP)

    if predict:
        assert(len(DP.shape)==2), "Dimension of single diffraction should be 2 (Qx, Qy)"
        assert(len(probe.shape)==2), "Dimension of probe should be 2 (Qx, Qy)"
        model = _get_latest_model(model_path = model_path)
        DP = tf.expand_dims(tf.expand_dims(DP, axis=0), axis=-1)
        probe = tf.expand_dims(tf.expand_dims(probe, axis=0), axis=-1)
        prediction = np.zeros(shape = (1, DP.shape[1],DP.shape[2],1))
        
        for i in tqdmnd(num_attmpts, desc='Neural network is predicting atomic potential', unit='ATTEMPTS',unit_scale=True):
            prediction += model.predict([DP,probe])
        print('Averaging over {} attempts \n'.format(num_attmpts))
        pred = prediction[0,:,:,0]/num_attmpts
    else:
        assert(len(DP.shape)==2), "Dimension of single diffraction should be 2 (Qx, Qy)"
        pred = DP
    
    maxima_x,maxima_y,maxima_int = get_maxima_2D(pred, 
                                                 sigma = sigma,
                                                 minRelativeIntensity=minRelativeIntensity,
                                                 minAbsoluteIntensity=minAbsoluteIntensity,
                                                 edgeBoundary=edgeBoundary,
                                                 relativeToPeak=relativeToPeak,
                                                 maxNumPeaks=maxNumPeaks,
                                                 minSpacing = minPeakSpacing,
                                                 subpixel=subpixel,
                                                 upsample_factor=upsample_factor)
            
    
    maxima_x, maxima_y, maxima_int = _integrate_disks(pred, maxima_x,maxima_y,maxima_int,int_window_radius=int_window_radius)

    # Make peaks PointList
    if peaks is None:
        coords = [('qx',float),('qy',float),('intensity',float)]
        peaks = PointList(coordinates=coords)
    else:
        assert(isinstance(peaks,PointList))
    peaks.add_tuple_of_nparrays((maxima_x,maxima_y,maxima_int))

    return peaks


def find_Bragg_disks_aiml_selected(datacube, probe, Rx, Ry,
                                   num_attmpts = 5,
                                   int_window_radius = 1,
                                   batch_size = 1,
                                   predict =True,
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
                                   model_path = None):
    """
    Finds the Bragg disks in the diffraction patterns of datacube at scan positions
    (Rx,Ry) by AI/ML method. This method utilizes FCU-Net to predict Bragg 
    disks from diffraction images.
    
    Args:
        datacube (datacube): a diffraction datacube
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
            latest model stored on cloud using metadata json file. It is not recommended to
            keep track of the model path and advised to keep this argument unchanged (None)
            to always search for the latest updated training model weights.

    Returns:
        (n-tuple of PointLists, n=len(Rx)): the Bragg peak positions and
        correlation intensities at each scan position (Rx,Ry).
    """
    
    try:
        import crystal4D
    except:
        raise ImportError("Import Error: Please install crystal4D before proceeding")
        
    try:
        import tensorflow as tf
    except:
        raise ImportError("Please install tensorflow before proceeding - please check " + "https://www.tensorflow.org/install" + "for more information")
    
    assert(len(Rx)==len(Ry))
    peaks = []
    
    if predict:
        model = _get_latest_model(model_path = model_path)
        t0= time()
        probe = np.expand_dims(np.repeat(np.expand_dims(probe, axis=0), 
                                             len(Rx), axis=0), axis=-1)
        DP = np.expand_dims(np.expand_dims(datacube.data[Rx[0],Ry[0],:,:], axis=0), axis=-1)
        total_DP = len(Rx)
        for i in range(1,len(Rx)):
            DP_ = np.expand_dims(np.expand_dims(datacube.data[Rx[i],Ry[i],:,:], axis=0), axis=-1)
            DP = np.concatenate([DP,DP_], axis=0)
            
        prediction = np.zeros(shape = (total_DP, datacube.Q_Nx, datacube.Q_Ny, 1))
        
        image_num = len(Rx)
        batch_num = int(image_num//batch_size)
        
        for att in tqdmnd(num_attmpts, desc='Neural network is predicting structure factors', unit='ATTEMPTS',unit_scale=True):
            for i in range(batch_num):
                prediction[i*batch_size:(i+1)*batch_size] += model.predict([DP[i*batch_size:(i+1)*batch_size],probe[i*batch_size:(i+1)*batch_size]], verbose=0)
            if (i+1)*batch_size < image_num:
                prediction[(i+1)*batch_size:] += model.predict([DP[(i+1)*batch_size:],probe[(i+1)*batch_size:]], verbose=0)
        
        prediction = prediction/num_attmpts
        
    # Loop over selected diffraction patterns
    for Rx in tqdmnd(image_num,desc='Finding Bragg Disks using AI/ML',unit='DP',unit_scale=True):
        DP = prediction[Rx,:,:,0]
        _peaks =  find_Bragg_disks_aiml_single_DP(DP, probe,
                                                   int_window_radius = int_window_radius,
                                                   predict = False,
                                                   sigma = sigma,
                                                   edgeBoundary=edgeBoundary,
                                                   minRelativeIntensity=minRelativeIntensity,
                                                   minAbsoluteIntensity=minAbsoluteIntensity,
                                                   relativeToPeak=relativeToPeak,
                                                   minPeakSpacing=minPeakSpacing,
                                                   maxNumPeaks=maxNumPeaks,
                                                   subpixel=subpixel,
                                                   upsample_factor=upsample_factor,
                                                   filter_function=filter_function,
                                                   model_path=model_path)
        peaks.append(_peaks)
    t2 = time()-t0
    print("Analyzed {} diffraction patterns in {}h {}m {}s".format(image_num, int(t2/3600),
                                                                   int(t2/60), int(t2%60)))

    peaks = tuple(peaks)
    return peaks

def find_Bragg_disks_aiml_serial(datacube, probe,
                                 num_attmpts = 5,
                                 int_window_radius = 1,
                                 predict =True,
                                 batch_size = 2,
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
                                 model_path = None,):
    """
    Finds the Bragg disks in all diffraction patterns of datacube from AI/ML method. 
    When hist = True, returns histogram of intensities in the entire datacube.

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
    
    try:
        import crystal4D
    except:
        raise ImportError("Import Error: Please install crystal4D before proceeding")
        
    try:
        import tensorflow as tf
    except:
        raise ImportError("Please install tensorflow before proceeding - please check " + "https://www.tensorflow.org/install" + "for more information")

    # Make the peaks PointListArray
    coords = [('qx',float),('qy',float),('intensity',float)]
    peaks = PointListArray(coordinates=coords, shape=(datacube.R_Nx, datacube.R_Ny))

    # check that the filtered DP is the right size for the probe kernel:
    if filter_function: assert callable(filter_function), "filter_function must be callable"
    DP = datacube.data[0,0,:,:] if filter_function is None else filter_function(datacube.data[0,0,:,:])
    #assert np.all(DP.shape == probe.shape), 'Probe kernel shape must match filtered DP shape'
    
    if predict:
        t0=time()
        model = _get_latest_model(model_path = model_path)
        probe = np.expand_dims(np.repeat(np.expand_dims(probe, axis=0), 
                                             datacube.R_N, axis=0), axis=-1)
        DP = np.expand_dims(np.reshape(datacube.data,
                                      (datacube.R_N,datacube.Q_Nx,datacube.Q_Ny)), axis = -1)
            
        prediction = np.zeros(shape = (datacube.R_N, datacube.Q_Nx, datacube.Q_Ny, 1))
        
        image_num = datacube.R_N
        batch_num = int(image_num//batch_size)

        for att in tqdmnd(num_attmpts, desc='Neural network is predicting structure factors', unit='ATTEMPTS',unit_scale=True):
            for i in range(batch_num):
                prediction[i*batch_size:(i+1)*batch_size] += model.predict([DP[i*batch_size:(i+1)*batch_size],probe[i*batch_size:(i+1)*batch_size]], verbose =0)
            if (i+1)*batch_size < image_num:
                prediction[(i+1)*batch_size:] += model.predict([DP[(i+1)*batch_size:],probe[(i+1)*batch_size:]], verbose =0)

        prediction = prediction/num_attmpts
    
        prediction = np.reshape(np.transpose(prediction, (0,3,1,2)),
                                (datacube.R_Nx, datacube.R_Ny, datacube.Q_Nx, datacube.Q_Ny))

    # Loop over all diffraction patterns
    for (Rx,Ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding Bragg Disks using AI/ML',unit='DP',unit_scale=True):
        DP_ = prediction[Rx,Ry,:,:]
        find_Bragg_disks_aiml_single_DP(DP_, probe,
                                         num_attmpts = num_attmpts,
                                         int_window_radius = int_window_radius,
                                         predict = False,
                                         sigma = sigma,
                                         edgeBoundary=edgeBoundary,
                                         minRelativeIntensity=minRelativeIntensity,
                                         minAbsoluteIntensity=minAbsoluteIntensity,
                                         relativeToPeak=relativeToPeak,
                                         minPeakSpacing=minPeakSpacing,
                                         maxNumPeaks=maxNumPeaks,
                                         subpixel=subpixel,
                                         upsample_factor=upsample_factor,
                                         filter_function=filter_function,
                                         peaks = peaks.get_pointlist(Rx,Ry),
                                         model_path=model_path)
    t2 = time()-t0
    print("Analyzed {} diffraction patterns in {}h {}m {}s".format(datacube.R_N, int(t2/3600),
                                                                   int(t2/60), int(t2%60)))
        
    if global_threshold == True:
        peaks = universal_threshold(peaks, minGlobalIntensity, metric, minPeakSpacing,
                                    maxNumPeaks)
    peaks.name = name
    return peaks

def find_Bragg_disks_aiml(datacube, probe,
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
                          name = 'braggpeaks_raw',
                          filter_function = None,
                          _qt_progress_bar = None,
                          model_path = None,
                          distributed = None,
                          CUDA = True):
    """
    Finds the Bragg disks in all diffraction patterns of datacube by AI/ML method. This method 
    utilizes FCU-Net to predict Bragg disks from diffraction images.

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
        distributed (dict): contains information for parallelprocessing using an
            IPyParallel or Dask distributed cluster.  Valid keys are:
                * ipyparallel (dict):
                * client_file (str): path to client json for connecting to your
                  existing IPyParallel cluster
                * dask (dict):
                  client (object): a dask client that connects to your
                                      existing Dask cluster
                * data_file (str): the absolute path to your original data
                  file containing the datacube
                * cluster_path (str): defaults to the working directory during processing
            if distributed is None, which is the default, processing will be in serial
        CUDA (bool): When True, py4DSTEM will use CUDA-enabled disk_detection_aiml function
        
    Returns:
        (PointListArray): the Bragg peak positions and correlation intensities
    """
    
    try:
        import crystal4D
    except:
        raise ImportError("Please install crystal4D before proceeding")
        
    try:
        import tensorflow as tf
    except:
        raise ImportError("Please install tensorflow before proceeding - please check " + "https://www.tensorflow.org/install" + "for more information")

    def _parse_distributed(distributed):
        import os

        if "ipyparallel" in distributed:
            if "client_file" in distributed["ipyparallel"]:
                connect = distributed["ipyparallel"]["client_file"]
            else:
                raise KeyError("Within distributed[\"ipyparallel\"], missing key for \"client_file\"")

            try:
                import ipyparallel as ipp
                c = ipp.Client(url_file=connect, timeout=30)

                if len(c.ids) == 0:
                    raise RuntimeError("No IPyParallel engines attached to cluster!")
            except ImportError:
                raise ImportError("Unable to import module ipyparallel!")
        elif "dask" in distributed:
            if "client" in distributed["dask"]:
                connect = distributed["dask"]["client"]
            else:
                raise KeyError("Within distributed[\"dask\"], missing key for \"client\"")
        else:
            raise KeyError(
                "Within distributed, you must specify 'ipyparallel' or 'dask'!")

        if "data_file" not in distributed:
            raise KeyError("Missing input data file path to distributed!  Required key 'data_file'")

        data_file = distributed["data_file"]

        if not isinstance(data_file, str):
            raise TypeError("Expected string for distributed key 'data_file', received {}".format(type(data_file)))
        if len(data_file.strip()) == 0:
            raise ValueError("Empty data file path from distributed key 'data_file'")
        elif not os.path.exists(data_file):
            raise FileNotFoundError("File not found")

        if "cluster_path" in distributed:
            cluster_path = distributed["cluster_path"]

            if not isinstance(cluster_path, str):
                raise TypeError(
                    "distributed key 'cluster_path' must be of type str, received {}".format(type(cluster_path)))

            if len(cluster_path.strip()) == 0:
                raise ValueError("distributed key 'cluster_path' cannot be an empty string!")
            elif not os.path.exists(cluster_path):
                raise FileNotFoundError("distributed key 'cluster_path' does not exist: {}".format(cluster_path))
            elif not os.path.isdir(cluster_path):
                raise NotADirectoryError("distributed key 'cluster_path' is not a directory: {}".format(cluster_path))
        else:
            cluster_path = None

        return connect, data_file, cluster_path
    
    if distributed is None:
        import warnings
        if not CUDA:
            if _check_cuda_device_available():
                warnings.warn('WARNING: CUDA = False is selected but py4DSTEM found available CUDA device to speed up. Going ahead anyway with non-CUDA mode (CPU only). You may want to abort and switch to CUDA = True to speed things up... \n')
            if num_attmpts > 1:
                warnings.warn('WARNING: num_attmpts > 1 will take significant amount of time with Non-CUDA mode ...')
            return find_Bragg_disks_aiml_serial(datacube,
                                                probe,
                                                num_attmpts = num_attmpts,
                                                int_window_radius = int_window_radius,
                                                predict = predict,
                                                batch_size = batch_size,
                                                sigma = sigma,
                                                edgeBoundary=edgeBoundary,
                                                minRelativeIntensity=minRelativeIntensity,
                                                minAbsoluteIntensity=minAbsoluteIntensity,
                                                relativeToPeak=relativeToPeak,
                                                minPeakSpacing=minPeakSpacing,
                                                maxNumPeaks=maxNumPeaks,
                                                subpixel=subpixel,
                                                upsample_factor=upsample_factor,
                                                model_path=model_path,
                                                name=name,
                                                filter_function=filter_function)
        elif _check_cuda_device_available():
            from .diskdetection_aiml_cuda import find_Bragg_disks_aiml_CUDA
            return find_Bragg_disks_aiml_CUDA(datacube,
                                              probe,
                                              num_attmpts = num_attmpts,
                                              int_window_radius = int_window_radius,
                                              predict = predict,
                                              batch_size = batch_size,
                                              sigma = sigma,
                                              edgeBoundary=edgeBoundary,
                                              minRelativeIntensity=minRelativeIntensity,
                                              minAbsoluteIntensity=minAbsoluteIntensity,
                                              relativeToPeak=relativeToPeak,
                                              minPeakSpacing=minPeakSpacing,
                                              maxNumPeaks=maxNumPeaks,
                                              subpixel=subpixel,
                                              upsample_factor=upsample_factor,
                                              model_path=model_path,
                                              name=name,
                                              filter_function=filter_function)
        else:
            import warnings
            warnings.warn('WARNING: py4DSTEM attempted to speed up the process using GPUs but no CUDA enabled devices are found. Switching back to Non-CUDA (CPU only) mode (Note it will take significant amount of time to get AIML predictions for disk detection using CPUs!!!!) \n')
            if num_attmpts > 1:
                warnings.warn('WARNING: num_attmpts > 1 will take significant amount of time with Non-CUDA mode ...')
            return find_Bragg_disks_aiml_serial(datacube,
                                                probe,
                                                num_attmpts = num_attmpts,
                                                int_window_radius = int_window_radius,
                                                predict = predict,
                                                batch_size = batch_size,
                                                sigma = sigma,
                                                edgeBoundary=edgeBoundary,
                                                minRelativeIntensity=minRelativeIntensity,
                                                minAbsoluteIntensity=minAbsoluteIntensity,
                                                relativeToPeak=relativeToPeak,
                                                minPeakSpacing=minPeakSpacing,
                                                maxNumPeaks=maxNumPeaks,
                                                subpixel=subpixel,
                                                upsample_factor=upsample_factor,
                                                model_path=model_path,
                                                name=name,
                                                filter_function=filter_function)

    elif isinstance(distributed, dict):
        raise Exception("{} is not yet implemented for aiml pipeline".format(type(distributed)))
    else:
        raise Exception("Expected type dict or None for distributed, instead found : {}".format(type(distributed)))

def _integrate_disks(DP, maxima_x,maxima_y,maxima_int,int_window_radius=1):
    """
    Integrate DP over the circular patch of pixel with radius
    """
    disks = []
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

def _check_cuda_device_available():
    """
    Check if GPU is available to use by python/tensorflow.
    """

    import tensorflow as tf
    
    tf_recog_gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if len(tf_recog_gpus) >0:
        return True
    else:
        return False
    
def _get_latest_model(model_path = None):
    """
    get the latest tensorflow model and model weights for disk detection
    
    Args:
        model_path (filepath string): File path for the tensorflow models stored in local system,
            if provided, disk detection will be performed loading the model provided by user.
            By default, there is no need to provide any file path unless specificly required for
            development/debug purpose. If None, _get_latest_model() will look up the latest model
            from cloud and download and load them. 

    Returns:
         model:    Trained tensorflow model for disk detection
    """
    
    import crystal4D
    import tensorflow as tf
    from ...io.google_drive_downloader import download_file_from_google_drive
    tf.keras.backend.clear_session()
    
    if model_path is None:
        try:
            os.mkdir('./tmp')
        except:
            pass
        # download the json file with the meta data
        download_file_from_google_drive('1uofpSGy7PDlpRiSnuvS5XemnpVbzpcle','./tmp/model_metadata.json')
        with open('./tmp/model_metadata.json') as f:
            metadata = json.load(f)
            file_id = metadata['file_id']
            file_path = metadata['file_path']
            file_type = metadata['file_type']
        
        try:
            with open('./tmp/model_metadata_old.json') as f_old:
                metaold = json.load(f_old)
                file_id_old = metaold['file_id']
        except:
            file_id_old = file_id
        
        if os.path.exists(file_path) and file_id == file_id_old:
            print('Latest model weight is already available in the local system. Loading the model... \n')
            model_path = file_path
            os.remove('./tmp/model_metadata_old.json')
            os.rename('./tmp/model_metadata.json', './tmp/model_metadata_old.json')
        else:
            print('Checking the latest model on the cloud... \n')
            filename = file_path + file_type
            download_file_from_google_drive(file_id,filename)
            shutil.unpack_archive(filename, './tmp' ,format="zip")
            model_path = file_path
            os.remove(filename)
            os.rename('./tmp/model_metadata.json', './tmp/model_metadata_old.json')
            print('Loading the model... \n')

        model = tf.keras.models.load_model(model_path,
                                           custom_objects={'lrScheduler': crystal4D.utils.utils.lrScheduler(128)})
    else:
        print('Loading the user provided model... \n')
        model = tf.keras.models.load_model(model_path,
                                           custom_objects={'lrScheduler': crystal4D.utils.utils.lrScheduler(128)})
    
    return model