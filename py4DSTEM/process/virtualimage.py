# Functions for generating virtual images
import numpy as np
import dask.array as da
from py4DSTEM.utils.tqdmnd import tqdmnd

def get_virtual_image(
    datacube,
    mode,
    geometry,
    centered = False,
    calibrated = False,
    shift_center = False,
    verbose = True,
    dask = False,
    return_mask = False
):
    '''
    Function to calculate virtual image

    Args:
        datacube (Datacube) : datacube class object which stores 4D-dataset
            needed for calculation
        mode (str)          : defines geometry mode for calculating virtual image.
            Options:
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
                - `mask`: flexible detector, any boolean or floating point 2D array with
                    the same shape as datacube.Qshape
        centered (bool)     : if False (default), the origin is in the upper left corner.
             If True, the mean measured origin in the datacube calibrations
             is set as center. The measured origin is set with datacube.calibration.set_origin()
             In this case, for example, a centered bright field image could be defined
             by geometry = ((0,0), R). For `mode="mask"`, has no effect.
        calibrated (bool)   : if True, geometry is specified in units of 'A^-1' instead of pixels.
            The datacube's calibrations must have its `"Q_pixel_units"` parameter set to "A^-1".
            For `mode="mask"`, has no effect.
        shift_center (bool) : if True, the mask is shifted at each real space position to
            account for any shifting of the origin of the diffraction images. The datacube's
            calibration['origin'] parameter must be set (centered = True). The shift applied to each
            pattern is the difference between the local origin position and the mean origin position
            over all patterns, rounded to the nearest integer for speed.
        verbose (bool)      : if True, show progress bar
        dask (bool)         : if True, use dask arrays
        return_mask (bool)  : if False (default) returns a virtual image as usual.  If True, does
            *not* generate or return a virtual image, instead returning the mask that would be
            used in virtual image computation for any call to this function where
            `shift_center = False`.  Otherwise, must be a 2-tuple of integers corresponding
            to a scan position (rx,ry); in this case, returns the mask that would be used for
            virtual image computation at this scan position with `shift_center` set to `True`.

    Returns:
        (2D array) virtual image
    '''

    assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
    'check doc strings for supported modes'
    g = geometry

    # Get calibration metadata
    if centered:
        assert datacube.calibration.get_origin(), "origin need to be calibrated"
        x0, y0 = datacube.calibration.get_origin()
        x0_mean = np.mean(x0)
        y0_mean = np.mean(y0)
    if calibrated:
        assert datacube.calibration['Q_pixel_units'] == 'A^-1', \
        'check datacube.calibration. datacube must be calibrated in A^-1 to use `calibrated=True`'

    # Convert units into detector pixels, if `centered` or `calibrated` are True
    if centered == True:
        if mode == 'point':
            g = (g[0] + x0_mean, g[1] + y0_mean)
        if mode in('circle', 'circular', 'annulus', 'annular'):
            g = ((g[0][0] + x0_mean, g[0][1] + y0_mean), g[1])
        if mode in('rectangle', 'square', 'rectangular') :
             g = (g[0] + x0_mean, g[1] + x0_mean, g[2] + y0_mean, g[3] + y0_mean)

    if calibrated == True:
        unit_conversion = datacube.calibration['Q_pixel_size']
        if mode == 'point':
            g = (g[0]/unit_conversion, g[1]/unit_conversion)
        if mode in('circle', 'circular'):
            g = ((g[0][0]/unit_conversion, g[0][1]/unit_conversion),
                (g[1]/unit_conversion))
        if mode in('annulus', 'annular'):
            g = ((g[0][0]/unit_conversion, g[0][1]/unit_conversion),
                (g[1][0]/unit_conversion, g[1][1]/unit_conversion))
        if mode in('rectangle', 'square', 'rectangular') :
            g = (g[0]/unit_conversion, g[1]/unit_conversion,
                 g[2]/unit_conversion, g[3]/unit_conversion)

    if shift_center == True:
        assert centered, "centered must be True"

    # Get mask
    mask = make_detector(datacube.Qshape, mode, g)
    # if return_mask is True, skip computation
    if return_mask == True and shift_center == False:
        return mask

    # Calculate images

    # no center shifting
    if shift_center == False:

        # dask 
        if dask == True:

            # set up a generalized universal function for dask distribution
            def _apply_mask_dask(datacube,mask):
                virtual_image = np.sum(np.multiply(datacube.data,mask), dtype=np.float64)
            apply_mask_dask = da.as_gufunc(
                _apply_mask_dask,signature='(i,j),(i,j)->()',
                output_dtypes=np.float64,
                axes=[(2,3),(0,1),()],
                vectorize=True
            )

            # compute
            virtual_image = apply_mask_dask(datacube.data, mask)

        # non-dask
        else:

            # compute
            if mask.dtype == 'complex':
                virtual_image = np.zeros(datacube.Rshape, dtype = 'complex')
            else:
                virtual_image = np.zeros(datacube.Rshape)
            for rx,ry in tqdmnd(
                datacube.R_Nx,
                datacube.R_Ny,
                disable = not verbose,
            ):
                virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)

    # with center shifting
    else:

        # get shifts
        qx_shift = (x0_mean-x0).round().astype(int)
        qy_shift = (y0_mean-y0).round().astype(int)

        # if return_mask is True, skip computation
        if return_mask is not False:
            try:
                rx,ry = return_mask
            except TypeError:
                raise Exception("when `shift_center` is True, return_mask must be a 2-tuple of ints or False")
            # get shifted mask
            _mask = np.roll(
                mask,
                (qx_shift[rx,ry], qy_shift[rx,ry]),
                axis=(0,1)
            )
            return _mask

        # compute
        if mask.dtype == 'complex':
            virtual_image = np.zeros(datacube.Rshape, dtype = 'complex')
        else:
            virtual_image = np.zeros(datacube.Rshape)
        
        for rx,ry in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            disable = not verbose,
        ):
            # get shifted mask
            _mask = np.roll(
                mask,
                (qx_shift[rx,ry], qy_shift[rx,ry]),
                axis=(0,1)
            )
            virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*_mask)

    return virtual_image

def make_detector(
    shape,
    mode,
    geometry,
):
    '''
    Function to return 2D mask

    Args:
        shape (tuple)      : defines shape of mask, for example (Q_Nx, Q_Ny) where Q_Nx and Q_Ny are mask sizes
        mode (str)         : defines geometry mode for calculating virtual image
            options:
                - 'point' uses singular point as detector
                - 'circle' or 'circular' uses round detector, like bright field
                - 'annular' or 'annulus' uses annular detector, like dark field
                - 'rectangle', 'square', 'rectangular', uses rectangular detector
                - 'mask' flexible detector, any boolean or floating point 2D array with
                    the same shape as datacube.Qshape or datacube.Rshape for virtual image
                    or diffraction image respectively
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
                    same shape as datacube.Qshape or datacube.Rshapefor virtual image
                    or diffraction image respectively

    Returns:
        virtual detector in the form of a 2D mask (array)
    '''
    g = geometry

    #point mask 
    if mode == 'point':
        assert(isinstance(g,tuple) and len(g)==2), 'specify qx and qy as tuple (qx, qy)'
        mask = np.zeros(shape, dtype=bool)

        qx = int(g[0])
        qy = int(g[1])

        mask[qx,qy] = 1

    #circular mask
    if mode in('circle', 'circular'):
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and isinstance(g[1],(float,int))), \
        'specify qx, qy, radius_i as ((qx, qy), radius)'

        qxa, qya = np.indices(shape)
        mask = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1] ** 2

    #annular mask 
    if mode in('annulus', 'annular'):
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and len(g[1])==2), \
        'specify qx, qy, radius_i, radius_0 as ((qx, qy), (radius_i, radius_o))'

        assert g[1][1] > g[1][0], "Inner radius must be smaller than outer radius"

        qxa, qya = np.indices(shape)
        mask1 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 > g[1][0] ** 2
        mask2 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1][1] ** 2
        mask = np.logical_and(mask1, mask2)

    #rectangle mask 
    if mode in('rectangle', 'square', 'rectangular') :
        assert(isinstance(g,tuple) and len(g)==4), \
       'specify x_min, x_max, y_min, y_max as (x_min, x_max, y_min, y_max)'
        mask = np.zeros(shape, dtype=bool)

        xmin = int(np.round(g[0]))
        xmax = int(np.round(g[1]))
        ymin = int(np.round(g[2]))
        ymax = int(np.round(g[3]))

        mask[xmin:xmax, ymin:ymax] = 1

    #flexible mask
    if mode == 'mask':
        assert type(g) == np.ndarray, '`geometry` type should be `np.ndarray`'
        assert (g.shape == shape), 'mask and diffraction pattern shapes do not match'
        mask = g
    return mask


def get_virtual_image_pointlistarray(
    peaks,
    mode = None,
    geometry = None,
    ):
    """
    Make a virtual image from a pointlist array.
    TODO - implement more virtual detectors.

    Args:
        peaks (PointListArray): List of all peaks and intensities.
        mode (str)          : defines geometry mode for calculating virtual image.
            Options:
                - 'circular' or 'circle' uses round detector, like bright field
                - 'annular' or 'annulus' uses annular detector, like dark field
        geometry (variable) : valid entries are determined by the `mode`, values in pixels
                              argument, as follows:
                - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius),
                   qx, qy and radius, are each single float or int
                - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                   qx, qy, radius_i, and radius_o are each single float or integer
                - Note that (qx,qy) can be skipped, which assumes peaks centered at (0,0)

    Returns:
        im_virtual (2D numpy array): the calculated virtual image
    """

    # Set geometry
    if mode is None:
        if geometry is None:
            center = None
            radial_range = np.array((0,np.inf))
        else:
            if len(geometry[0]) == 0:
                center = None
            else:
                center = np.array(geometry[0])
            if isinstance(geometry[1], int) or isinstance(geometry[1], float):
                radial_range = np.array((0,geometry[1]))
            elif len(geometry[1]) == 0:
                radial_range = None
            else:
                radial_range = np.array(geometry[1])
    elif mode == 'circular' or mode == 'circle':
        radial_range = np.array((0,geometry[1]))
        if len(geometry[0]) == 0:
                center = None
        else:
            center = np.array(geometry[0])
    elif mode == 'annular' or mode == 'annulus':
        radial_range = np.array(geometry[1])
        if len(geometry[0]) == 0:
                center = None
        else:
            center = np.array(geometry[0])



    # init
    im_virtual = np.zeros(peaks.shape)

    # Generate image
    for rx,ry in tqdmnd(peaks.shape[0],peaks.shape[1]):
        p = peaks.get_pointlist(rx,ry)
        if p.data.shape[0] > 0:
            if radial_range is None:
                im_virtual[rx,ry] = np.sum(p.data['intensity'])
            else:
                if center is None:
                    qr = np.hypot(p.data['qx'],p.data['qy'])
                else:
                    qr = np.hypot(p.data['qx'] - center[0],p.data['qy'] - center[1])
                sub = np.logical_and(
                    qr >= radial_range[0],
                    qr <  radial_range[1])
                if np.sum(sub) > 0:
                    im_virtual[rx,ry] = np.sum(p.data['intensity'][sub])

    return im_virtual


def get_virtual_image_braggvectors(
    bragg_peaks,
    mode = None,
    geometry = None,
    ):
    '''
    Function to calculate virtual images from braggvectors / pointlist arrays.
    TODO - implement these detectors for braggvectors

    Args:
        bragg_peaks (BraggVectors) : BraggVectors class object which stores bragg peaks
        mode (str)          : defines geometry mode for calculating virtual image.
            Options:
                - 'circular' or 'circle' uses round detector, like bright field
                - 'annular' or 'annulus' uses annular detector, like dark field
        geometry (variable) : valid entries are determined by the `mode`, values in pixels
                              argument, as follows:
                - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius),
                   qx, qy and radius, are each single float or int
                - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                   qx, qy, radius_i, and radius_o are each single float or integer
                - Note that (qx,qy) can be skipped, which assumes peaks centered at (0,0)
                
    Returns:
        im_virtual (2D numpy array): the calculated virtual image
    '''

    virtual_image = get_virtual_image_pointlistarray(
        bragg_peaks.vectors,    
        mode = mode,
        geometry = geometry,
        )

    return virtual_image
