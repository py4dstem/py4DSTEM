# Functions to become DataCube methods

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes


# Add to tree

from py4DSTEM.io.datastructure.emd import Array
def add(
    self,
    data,
    name = ''
    ):
    """
    Adds a block of data to the DataCube's tree. If `data` is an instance of
    an EMD/py4DSTEM class, add it to the tree.  If it's a numpy array,
    turn it into an Array instance, then save to the tree.
    """
    if isinstance(data, np.ndarray):
        data = Array(
            data = data,
            name = name
        )
    self.tree[data.name] = data



# Diffraction imaging

from py4DSTEM.io.datastructure.py4dstem.virtualdiffraction import VirtualDiffraction
def get_virtual_diffraction(
    self,
    method = 'max',
    mode = None,
    geometry = None,
    calibrated = False,
    shift_center = False,
    verbose = True,
    name = 'virtual_diffracton',
    returncalc = True,
    ):
    """
    Function to calculate virtual diffraction
            Args:
                datacube (Datacube) : datacube class object which stores 4D-dataset
                    needed for calculation
                
                method (str)        : defines method used for diffraction pattern, options are
                                      'mean', 'median', and 'max' 
                
                mode (str)          : defines mode for selecting area in real space to use for 
                                      virtual diffraction. The default is None, which means no
                                      geometry will be applied and the whole datacube will be used
                                      for the calculation.
                    Options:
                        - 'point' uses singular point as detector
                        - 'circle' or 'circular' uses round detector, like bright field
                        - 'annular' or 'annulus' uses annular detector, like dark field
                        - 'rectangle', 'square', 'rectangular', uses rectangular detector
                        - 'mask' flexible detector, any 2D array

                geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows. The default is None, which means no geometry will be applied
                and the whole datacube will be used for the calculation. If mode is None the geometry 
                will not be applied.
                        - 'point': 2-tuple, (rx,ry),
                           qx and qy are each single float or int to define center
                        - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                           qx, qy and radius, are each single float or int
                        - 'annular' or 'annulus': nested 2-tuple, ((rx,ry),(radius_i,radius_o)),
                           qx, qy, radius_i, and radius_o are each single float or integer
                        - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                        - `mask`: flexible detector, any boolean or floating point 2D array with
                            the same shape as datacube.Rshape
                            
                calibrated (bool)   : if True, geometry is specified in units of 'A' instead of pixels.
                The datacube's calibrations must have its `"R_pixel_units"` parameter set to "A". 
                If mode is None the geometry and calibration will not be applied.
    
                shift_center (bool) : if True, the difraction pattern is shifted to account for beam shift 
                or the changing of the origin through the scan. The datacube's calibration['origin'] 
                parameter must be set Only 'max' and 'mean' supported for this option.  
                
                verbose (bool)      : if True, show progress bar
            
            Returns:
                (2D array): the diffraction image
    """

    # perform computation
    from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
    dp = get_virtual_diffraction(
        self,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
        verbose = verbose,
    )

    # wrap with a py4dstem class
    dp = VirtualDiffraction(
        data = dp,
        name = name,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
    )

    # add to the tree
    self.tree[name] = dp

    # return
    if returncalc:
        return dp

def get_dp_max(
    self,
    method = 'max',
    mode = None,
    geometry = None,
    calibrated = False,
    shift_center = False,
    verbose = True,
    name = 'dp_max',
    returncalc = True,
    ):
    """
    Function to calculate maximum virtual diffraction. Default captures pattern across
    entire 4D-dataset.
            Args:
                datacube (Datacube) : datacube class object which stores 4D-dataset
                    needed for calculation
                
                mode (str)          : defines mode for selecting area in real space to use for 
                                      virtual diffraction. The default is None, which means no
                                      geometry will be applied and the whole datacube will be used
                                      for the calculation.
                    Options:
                        - 'point' uses singular point as detector
                        - 'circle' or 'circular' uses round detector, like bright field
                        - 'annular' or 'annulus' uses annular detector, like dark field
                        - 'rectangle', 'square', 'rectangular', uses rectangular detector
                        - 'mask' flexible detector, any 2D array

                geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows. The default is None, which means no geometry will be applied
                and the whole datacube will be used for the calculation. If mode is None the geometry 
                will not be applied.
                        - 'point': 2-tuple, (rx,ry),
                           qx and qy are each single float or int to define center
                        - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                           qx, qy and radius, are each single float or int
                        - 'annular' or 'annulus': nested 2-tuple, ((rx,ry),(radius_i,radius_o)),
                           qx, qy, radius_i, and radius_o are each single float or integer
                        - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                        - `mask`: flexible detector, any boolean or floating point 2D array with
                            the same shape as datacube.Rshape
                            
                calibrated (bool)   : if True, geometry is specified in units of 'A' instead of pixels.
                The datacube's calibrations must have its `"R_pixel_units"` parameter set to "A". 
                If mode is None the geometry and calibration will not be applied.
                
                shift_center (bool) : if True, the difraction pattern is shifted to account for beam shift 
                or the changing of the origin through the scan. The datacube's calibration['origin'] 
                parameter must be set Only 'max' and 'mean' supported for this option.  
                verbose (bool)      : if True, show progress bar
            
            Returns:
                (2D array): the diffraction image
    """

    # perform computation
    
    from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
    
    dp = get_virtual_diffraction(
        self,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
        verbose = verbose,
    )

    # wrap with a py4dstem class
    dp = VirtualDiffraction(
        data = dp,
        name = name,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
    )

    # add to the tree
    self.tree[name] = dp

    # return
    if returncalc:
        return dp

def get_dp_mean(
    self,
    method = 'max',
    mode = None,
    geometry = None,
    calibrated = False,
    shift_center = False,
    verbose = True,
    name = 'dp_mean',
    returncalc = True,
    ):
    """
    Function to calculate mean virtual diffraction. Default captures pattern across
    entire 4D-dataset.
            Args:
                datacube (Datacube) : datacube class object which stores 4D-dataset
                    needed for calculation
                
                mode (str)          : defines mode for selecting area in real space to use for 
                                      virtual diffraction. The default is None, which means no
                                      geometry will be applied and the whole datacube will be used
                                      for the calculation.
                    Options:
                        - 'point' uses singular point as detector
                        - 'circle' or 'circular' uses round detector, like bright field
                        - 'annular' or 'annulus' uses annular detector, like dark field
                        - 'rectangle', 'square', 'rectangular', uses rectangular detector
                        - 'mask' flexible detector, any 2D array

                geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows. The default is None, which means no geometry will be applied
                and the whole datacube will be used for the calculation. If mode is None the geometry 
                will not be applied.
                        - 'point': 2-tuple, (rx,ry),
                           qx and qy are each single float or int to define center
                        - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                           qx, qy and radius, are each single float or int
                        - 'annular' or 'annulus': nested 2-tuple, ((rx,ry),(radius_i,radius_o)),
                           qx, qy, radius_i, and radius_o are each single float or integer
                        - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                        - `mask`: flexible detector, any boolean or floating point 2D array with
                            the same shape as datacube.Rshape
                            
                calibrated (bool)   : if True, geometry is specified in units of 'A' instead of pixels.
                The datacube's calibrations must have its `"R_pixel_units"` parameter set to "A". 
                If mode is None the geometry and calibration will not be applied.
                
                shift_center (bool) : if True, the difraction pattern is shifted to account for beam shift 
                or the changing of the origin through the scan. The datacube's calibration['origin'] 
                parameter must be set Only 'max' and 'mean' supported for this option.  
                
                verbose (bool)      : if True, show progress bar
            
            Returns:
                (2D array): the diffraction image
    """

    # perform computation
    from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
    dp = get_virtual_diffraction(
        self,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
        verbose = verbose,
    )

    # wrap with a py4dstem class
    dp = VirtualDiffraction(
        data = dp,
        name = name,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
    )

    # add to the tree
    self.tree[name] = dp

    # return
    if returncalc:
        return dp

def get_dp_median(
    self,
    method = 'median',
    mode = None,
    geometry = None,
    calibrated = False,
    shift_center = False,
    verbose = True,
    name = 'dp_median',
    returncalc = True,
    ):
    """
    Function to calculate median virtual diffraction. Default captures pattern across
    entire 4D-dataset.
            Args:
                datacube (Datacube) : datacube class object which stores 4D-dataset
                    needed for calculation
                
                mode (str)          : defines mode for selecting area in real space to use for 
                                      virtual diffraction. The default is None, which means no
                                      geometry will be applied and the whole datacube will be used
                                      for the calculation.
                    Options:
                        - 'point' uses singular point as detector
                        - 'circle' or 'circular' uses round detector, like bright field
                        - 'annular' or 'annulus' uses annular detector, like dark field
                        - 'rectangle', 'square', 'rectangular', uses rectangular detector
                        - 'mask' flexible detector, any 2D array

                geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows. The default is None, which means no geometry will be applied
                and the whole datacube will be used for the calculation. If mode is None the geometry 
                will not be applied.
                        - 'point': 2-tuple, (rx,ry),
                           qx and qy are each single float or int to define center
                        - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                           qx, qy and radius, are each single float or int
                        - 'annular' or 'annulus': nested 2-tuple, ((rx,ry),(radius_i,radius_o)),
                           qx, qy, radius_i, and radius_o are each single float or integer
                        - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                        - `mask`: flexible detector, any boolean or floating point 2D array with
                            the same shape as datacube.Rshape
                            
                calibrated (bool)   : if True, geometry is specified in units of 'A' instead of pixels.
                The datacube's calibrations must have its `"R_pixel_units"` parameter set to "A". 
                If mode is None the geometry and calibration will not be applied.
    
                
                shift_center (bool) : if True, the difraction pattern is shifted to account for beam shift 
                or the changing of the origin through the scan. The datacube's calibration['origin'] 
                parameter must be set Only 'max' and 'mean' supported for this option.  
                
                verbose (bool)      : if True, show progress bar
            
            Returns:
                (2D array): the diffraction image
    """

    # perform computation
    from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
    dp = get_virtual_diffraction(
        self,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
        verbose = verbose,
    )

    # wrap with a py4dstem class
    dp = VirtualDiffraction(
        data = dp,
        name = name,
        method = method,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
        calibrated = calibrated,
    )

    # add to the tree
    self.tree[name] = dp

    # return
    if returncalc:
        return dp



# Virtual imaging

from py4DSTEM.io.datastructure.py4dstem.virtualimage import VirtualImage
def get_virtual_image(
    self,
    mode,
    geometry,
    centered = None,
    calibrated = None,
    shift_center = None,
    verbose = True,
    dask = False,
    return_mask = False,
    name = 'virtual_image',
    returncalc = True,
    ):
    """
    Get a virtual image and store it in `datacube`s tree under `name`.
    The kind of virtual image is specified by the `mode` argument.

    Args:
        mode (str)          : defines geometry mode for calculating virtual image
            options:
                - 'point' uses singular point as detector
                - 'circle' or 'circular' uses round detector, like bright field
                - 'annular' or 'annulus' uses annular detector, like dark field
                - 'rectangle', 'square', 'rectangular', uses rectangular detector
                - 'mask' flexible detector, any 2D array
        geometry (variable) : valid entries are determined by the `mode`, values in pixels
            argument, as follows:
                - 'point': 2-tuple, (qx,qy),
                   qx and qy are each single float or int to define center
                - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius),
                   qx, qy and radius, are each single float or int
                - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                   qx, qy, radius_i, and radius_o are each single float or integer
                - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                - `mask`: flexible detector, any boolean or floating point 2D array with the
                    same size as datacube.Qshape
        centered (bool)     : if False, the origin is in the upper left corner.
             If True, the mean measured origin in the datacube calibrations
             is set as center. The measured origin is set with datacube.calibration.set_origin()
             In this case, for example, a centered bright field image could be defined 
             by geometry = ((0,0), R). For `mode="mask"`, has no effect. Default is None and will 
             set to True if the origin has been set.
        calibrated (bool)   : if True, geometry is specified in units of 'A^-1' instead of pixels.
            The datacube's calibrations must have its `"Q_pixel_units"` parameter set to "A^-1".
            For `mode="mask"`, has no effect. Default is None and will set to True if the calibration 
            has been set.
        shift_center (bool) : if True, the mask is shifted at each real space position to
            account for any shifting of the origin of the diffraction images. The datacube's
            calibration['origin'] parameter must be set (centered = True). The shift applied to each 
            pattern is the difference between the local origin position and the mean origin position
            over all patterns, rounded to the nearest integer for speed. Default is None and will set 
            to True if centered == True. 
        verbose (bool)      : if True, show progress bar
        dask (bool)         : if True, use dask arrays
        return_mask (bool)  : if False (default) returns a virtual image as usual.  If True, does
            *not* generate or return a virtual image, instead returning the mask that would be
            used in virtual image computation for any call to this function where
            `shift_center = False`.  Otherwise, must be a 2-tuple of integers corresponding
            to a scan position (rx,ry); in this case, returns the mask that would be used for
            virtual image computation at this scan position with `shift_center` set to `True`.
            Setting return_mask to True does not add anything to the datacube's tree.
        name (str)          : the output object's name
        returncalc (bool)   : if True, returns the output

    Returns:
        (Optional): if returncalc is True, returns the VirtualImage
    """
    #check for calibration
    if calibrated is None:
        if self.calibration['Q_pixel_units'] == 'A^-1' and 'qx0' in self.calibration.keys: 
            calibrated = True
            print ('note: setting `calibrated` to True')
        else: 
            calibrated = False

    #check for centered 
    if centered is None:
        if self.calibration.get_origin(): 
            centered = True
            print ('note: setting `centered` to True')
        else: 
            centered = False

    # logic to determine shift_center
    if shift_center is None: 
        if centered: 
            shift_center = True
            print ('note: setting `shift_center` to True')
        else: 
            shift_center = False 

    # perform computation
    from py4DSTEM.process.virtualimage import get_virtual_image
    im = get_virtual_image(
        self,
        mode = mode,
        geometry = geometry,
        centered = centered,
        calibrated = calibrated,
        shift_center = shift_center,
        verbose = verbose,
        dask = dask,
        return_mask = return_mask
    )

    # if a mask is requested, skip the remaining i/o functionality
    if return_mask is not False:
        return im

    # wrap with a py4dstem class
    im = VirtualImage(
        data = im,
        name = name,
        mode = mode,
        geometry = geometry,
        shift_center = shift_center,
    )

    # add to the tree
    self.tree[name] = im

    # return
    if returncalc:
        return im

# Probe

from py4DSTEM.io.datastructure.py4dstem.probe import Probe
def get_vacuum_probe(
    self,
    name = 'probe',
    returncalc = True,
    **kwargs
    ):
    """

    """

    # perform computation
    from py4DSTEM.process.probe import get_vacuum_probe
    x = get_vacuum_probe(
        self,
        **kwargs
    )

    # wrap with a py4dstem class
    x = Probe(
        data = x,
        **kwargs
    )

    # add to the tree
    self.tree[name] = x

    # return
    if returncalc:
        return x


from py4DSTEM.io.datastructure.py4dstem.calibration import Calibration
def get_probe_size(
    self,
    mode = None,
    returncal = True,
    ** kwargs,
    ):
    """

    """
    #perform computation 
    from py4DSTEM.process.calibration import get_probe_size

    if mode is None:
        assert 'no mode speficied, using mean diffraciton pattern'
        assert 'dp_mean' in self.tree.keys(), "calculate .get_dp_mean()"
        DP = self.tree['dp_mean'].data
    elif type(mode) == str:
        assert mode in self.tree.keys(), "mode not found"
        DP = self.tree[mode].data
    elif type(mode) == np.ndarray:
        assert len(mode.shape) == 2, "must be a 2D array"
        DP = mode

    x = get_probe_size(DP
        ,
        **kwargs
    )

    # try to add to calibration
    try:
        self.calibration.set_probe_param(x)
    except AttributeError:
        # should a warning be raised?
        pass

    # return
    if returncal:
        return x


# Bragg disks

def find_Bragg_disks(
    self,

    template,
    data = None,

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

    name = 'braggvectors',
    returncalc = True,
    ):
    """
    Finds the Bragg disks by cross correlation with `template`.

    For each diffraction image, the algorithm works in 4 steps:

    (1) optional pre-processing by passing the image through some
        `filter_funtion`, which should accept and return 2D arrays
    (2) the image is cross correlated with the template.
        Phase/hybrid correlations can be used instead by setting the
        `corrPower` argument. Cross correlation can be skipped entirely,
        and steps 3 and 4 performed directly on the diffraction
        image itself rathar than a cross correlation, by passing None
        to `template`.
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


    Running on a subset of the data may be desireable, and is controlled
    by the `data` parameter. If None (default), runs on the whole DataCube,
    and stores the output in its tree. Otherwise, nothing is stored in tree,
    but some value is returned. Valid entries are:

        - a 2-tuple of numbers (rx,ry): run on this diffraction image,
            and return a QPoints instance
        - a 2-tuple of arrays (rx,ry): run on these diffraction images,
            and return a list of QPoints instances
        - an Rspace shapped 2D boolean array: run on the diffraction images
            specified by the True counts and return a list of QPoints instances


    Args:
        template (2D array): the vacuum probe template, in real space. For
            Probe instances, this is `probe.kernel`.  If None, does not perform
            a cross correlation.
        data (variable): see above.
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
        name (str): name for the output BraggVectors
        returncalc (bool): if True, returns the answer

    Returns:
        (BraggVectors or QPoints or list of QPoints)
    """
    from py4DSTEM.process.diskdetection import find_Bragg_disks

    # parse args
    if data is None:
        x = self
    elif isinstance(data, tuple):
        x = self, data[0], data[1]
    elif isinstance(data, np.ndarray):
        assert data.dtype == bool, 'array must be boolean'
        assert data.shape == self.Rshape, 'array must be Rspace shaped'
        x = self.data[data,:,:]
    else:
        raise Exception(f'unexpected type for `data` {type(data)}')


    # compute
    peaks = find_Bragg_disks(
        data = x,
        template = template,

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

        CUDA = CUDA,
        CUDA_batched = CUDA_batched,
        distributed = distributed,

        _qt_progress_bar = _qt_progress_bar,
    )


    # name
    try:
        peaks.name = name
    except AttributeError:
        pass

    # add to tree
    if data is None:
        self.tree[name] = peaks

    # return
    if returncalc:
        return peaks





def get_beamstop_mask(
    self,
    threshold = 0.25,
    distance_edge = 4.0,
    include_edges = True,
    name = "mask_beamstop",
    returncalc = True,
    ):
    """
    This function uses the mean diffraction pattern plus a threshold to create a beamstop mask.

    Args:
        threshold: (float)  Value from 0 to 1 defining initial threshold for beamstop mask,
            taken from the sorted intensity values - 0 is the dimmest
            pixel, while 1 uses the brighted pixels.
        distance_edge: (float)  How many pixels to expand the mask.
        include_edges: (bool)   If set to True, edge pixels will be included in the mask.
        name: (string)          Name of the output array.
        returncalc: (bool):     Set to true to return the result.

    Returns:
        (Optional): if returncalc is True, returns the beamstop mask

    """

    # Calculate dp_mean if needed
    if not "dp_mean" in self.tree.keys():
        self.get_dp_mean();

    # normalized dp_mean
    int_sort = np.sort(self.tree["dp_mean"].data.ravel())
    ind = np.round(np.clip(
            int_sort.shape[0]*threshold,
            0,int_sort.shape[0])).astype('int')
    intensity_threshold = int_sort[ind]

    # Use threshold to calculate initial mask
    mask_beamstop = self.tree["dp_mean"].data >= intensity_threshold

    # clean up mask
    mask_beamstop = np.logical_not(binary_fill_holes(np.logical_not(mask_beamstop)))
    mask_beamstop = binary_fill_holes(mask_beamstop)

    # Edges
    if include_edges:
        mask_beamstop[0,:] = False
        mask_beamstop[:,0] = False
        mask_beamstop[-1,:] = False
        mask_beamstop[:,-1] = False


    # Expand mask
    mask_beamstop = distance_transform_edt(mask_beamstop) < distance_edge

    # Output mask for beamstop
    self.name = name
    self.tree[name] = mask_beamstop

    # return
    if returncalc:
        return mask_beamstop

