# Functions for generating diffraction images

import numpy as np
from py4DSTEM.utils.tqdmnd import tqdmnd

def get_virtual_diffraction(
    datacube,
    method,
    mode = None,
    geometry = None,
    calibrated = False,
    shift_center = False,
    verbose = True,
    return_mask = False,
):
    '''
    Function to calculate virtual diffraction

    Args:
        datacube (Datacube) : datacube class object which stores 4D-dataset
            needed for calculation
        method (str) : defines method used for diffraction pattern, options are
            'mean', 'median', and 'max'
        mode (str)  : defines mode for selecting area in real space to use for
            virtual diffraction. The default is None, which means no
            geometry will be applied and the whole datacube will be used
            for the calculation. Options:
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
        verbose (bool) : if True, show progress bar
        return_mask (bool) : if False (default) returns a virtual image as usual.  If True, does
            *not* generate or return a virtual image, instead returning the mask that would be
            used in virtual diffraction computation.

    Returns:
        (2D array): the diffraction image
    '''

    assert method in ('max', 'median', 'mean'),\
        'check doc strings for supported types'

    #create mask
    if mode is not None:
        use_all_points = False

        from py4DSTEM.process.virtualimage import make_detector
        assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
        'check doc strings for supported modes'
        g = geometry

        if calibrated == True:
            assert datacube.calibration['R_pixel_units'] == 'A', \
                'check datacube.calibration. datacube must be calibrated in A to use `calibrated=True`'

            unit_conversion = datacube.calibration['R_pixel_size']
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

        # Get mask
        mask = make_detector(datacube.Rshape, mode, g)

        # Determine if mask is boolean and if so, vectorize
        if mask.dtype == bool:
            mask_is_boolean = True
            mask_indices = np.nonzero(mask)
        else: 
            mask_is_boolean = False

    #if no mask 
    else:
        mask = np.ones(datacube.Rshape, dtype=bool)
        mask_indices = np.nonzero(mask)
        use_all_points = True
        mask_is_boolean = False

    # if return_mask is True, skip computation
    if return_mask == True:
        return mask

    # Calculate diffracton pattern...

    # ...with no center shifting
    if shift_center == False:

        # ...for the whole pattern
        if use_all_points:
            if method == 'mean':
                virtual_diffraction = np.mean(datacube.data, axis=(0,1))
            elif method == 'max':
                virtual_diffraction = np.max(datacube.data, axis=(0,1))
            else:
                virtual_diffraction = np.median(datacube.data, axis=(0,1))

        # ...for boolean masks
        elif mask_is_boolean:
            if method == 'mean':
                virtual_diffraction = np.mean(datacube.data[mask_indices[0],mask_indices[1],:,:], axis=0)
            elif method == 'max':
                virtual_diffraction = np.max(datacube.data[mask_indices[0],mask_indices[1],:,:], axis=0)
            else:
                virtual_diffraction = np.median(datacube.data[mask_indices[0],mask_indices[1],:,:], axis=0)
            
        # ...for floating point masks
        else:
            virtual_diffraction = np.zeros(datacube.Qshape)
            for qx,qy in tqdmnd(
                datacube.Q_Nx,
                datacube.Q_Ny,
                disable = not verbose,
            ):
                if method == 'mean':
                    virtual_diffraction[qx,qy] = np.sum( np.squeeze(datacube.data[:,:,qx,qy])*mask )
                elif method == 'max':
                    virtual_diffraction[qx,qy] = np.max( np.squeeze(datacube.data[:,:,qx,qy])*mask )
                elif method == 'median':
                    virtual_diffraction[qx,qy] = np.median( np.squeeze(datacube.data[:,:,qx,qy])*mask )
            
        # norm by weighting term for means
        if method == 'mean':
            virtual_diffraction /= np.sum(mask)

    # ...with center shifting
    else:
        assert method in ('max', 'mean'),\
            "only 'mean' and 'max' are supported for center-shifted virtual diffraction"

        # Get calibration metadata
        assert datacube.calibration.get_origin(), "origin needs to be calibrated"
        x0, y0 = datacube.calibration.get_origin()
        x0_mean = np.mean(x0)
        y0_mean = np.mean(y0)

        # get shifts
        qx_shift = (x0_mean-x0).round().astype(int)
        qy_shift = (y0_mean-y0).round().astype(int)


        # compute...

        # ...for boolean masks / whole datacubes
        if mask_is_boolean or use_all_points:
            virtual_diffraction = np.zeros(datacube.Qshape)
            for rx,ry in zip(mask_indices[0],mask_indices[1]):
                # get shifted DP
                DP = np.roll(
                    datacube.data[rx,ry, :,:,],
                    (qx_shift[rx,ry], qy_shift[rx,ry]),
                    axis=(0,1),
                    )
                # compute
                if method == 'mean':
                    virtual_diffraction += DP
                elif method == 'max':
                    virtual_diffraction = np.maximum(virtual_diffraction, DP)
            if method == 'mean':
                virtual_diffraction /= len(mask_indices[0])

        # ...for floating point masks
        else:
            virtual_diffraction = np.zeros(datacube.Qshape)
            for rx,ry in tqdmnd(
                datacube.R_Nx,
                datacube.R_Ny,
                disable = not verbose,
            ):
                # get shifted DP
                DP = np.roll(
                    datacube.data[rx,ry, :,:,],
                    (qx_shift[rx,ry], qy_shift[rx,ry]),
                    axis=(0,1),
                    )
                # compute
                w = mask[rx,ry]
                if w > 0:
                    if method == 'mean':
                        virtual_diffraction += DP*w
                    elif method == 'max':
                        virtual_diffraction = np.maximum(virtual_diffraction, DP*w)
            if method == 'mean':
                virtual_diffraction /= np.sum(mask)

    # return
    return virtual_diffraction


