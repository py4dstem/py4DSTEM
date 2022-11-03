import numpy as np
from  py4DSTEM.visualize.show import show
from py4DSTEM.io.datastructure.py4dstem import (
    Calibration, DataCube, DiffractionSlice
)

def position_detector(
    data,
    mode,
    geometry,
    centered,
    calibrated,
    shift_center,
    color = 'r',
    alpha = 0.4
):
    """
    Display a diffraction space image with an overlaid mask representing
    a virtual detector.

    Args:
        data (DataCube, DiffractionSlice, array, tuple):
            behavoir depends on the argument type:
                DataCube - check to see if this datacube has a mean, max,
                    or median diffraction pattern, and if found, uses it
                    (order of preference as written here). If not found,
                    raises an exception.
                DiffractionSlice - use the first slice
                array - use this array. This mode only works when
                    centered, calibrated, and shift_center are False.
                    Otherwise, use the tuple entry (array, Calibration)
                tuple - must be either:
                    - (DataCube, rx, ry) for rx,ry integers.
                    Use the diffraction pattern at this scan position.
                    `shift_center` is auto set to True in this mode.
                    - (array, Calibration)
        mode: see py4DSTEM.process.get_virtual_image
        geometry: see py4DSTEM.process.get_virtual_image
        centered: see py4DSTEM.process.get_virtual_image
        calibrated: see py4DSTEM.process.get_virtual_image
        shift_center: see py4DSTEM.process.get_virtual_image; if True, `data`
            should be a 3-tuple (DataCube, rx, ry)
    """
    # Parse data
    if isinstance(data, DataCube):
        cal = data.calibration
        keys = ['dp_mean','dp_max','dp_median']
        for k in keys:
            try:
                image = data.tree[k]
                break
            except:
                pass
        else:
            raise Exception("No mean, max, or median diffraction image found; try calling datacube.get_mean_dp() first")
    elif isinstance(data, DiffractionSlice):
        cal = data.calibration
        try:
            image = data[:,:,0]
        except IndexError:
            image = data[:,:]
    elif isinstance(data, np.ndarray):
        er = "centered and calibrated must be False to pass an uncalibrated array; set these to False or try using `data = (array, Calibration)`"
        assert all([x is False for x in [centered,calibrated]]), er
        image = data
        cal = None
    elif isinstance(data, tuple):
        if len(data)==2:
            image,cal = data
            assert isinstance(image, np.ndarray)
            assert isinstance(cal, Calibration)
        elif len(data)==3:
            assert(shift_center is True), "If `data` is a 3-tuple, `shift_center` must be True"
            data,rx,ry = data
            image = data[rx,ry,:,:]
            cal = data.calibration
    else:
        raise Exception(f"Invalid entry {data} for argument `data`")


    # Get geometry
    from py4DSTEM.process.virtualimage import get_calibrated_geometry
    g = get_calibrated_geometry(
        calibration = cal,
        mode = mode,
        geometry = geometry,
        centered = centered,
        calibrated = calibrated
    )

    # Get mask
    from py4DSTEM.process.virtualimage import make_detector
    mask = make_detector(image.shape, mode, g)

    # Shift center
    if shift_center:
        try:
            rx,ry
        except NameError:
            raise Exception("if `shift_center` is True then `data` must be the 3-tuple (DataCube,rx,ry)")
        # get shifts
        assert cal.get_origin_shift(), "origin shifts need to be calibrated"
        qx_shift,qy_shift = cal.get_origin_shift()
        qx_shift = int(np.round(qx_shift[rx,ry]))
        qy_shift = int(np.round(qy_shift[rx,ry]))
        mask = np.roll(
            mask,
            (qx_shift, qy_shift),
            axis=(0,1)
        )

    # Display

    show(
        image,
        mask = mask,
        mask_color = color,
        mask_alpha = alpha, 
        intensity_range = 'absolute', 
        vmin = np.log(np.min(image.data)), 
        vmax = np.log(np.max(image.data)),
        scaling = 'log'
    )

    return


