# Transforms to calibrate Bragg vector data


def transform(
    data
    cal,
    scanxy,
    center,
    ellipse,
    pixel,
    ):
    """
    Return a transformed copy of stractured data `data` with fields
    with fields 'qx','qy','intensity', applying calibrating transforms
    according to the values of center, ellipse, pixel, using the
    measurements found in Calibration instance cal for scan position scanxy.
    """

    ans = data.copy
    x,y = scanxy

    # origin

    if origin:
        origin = cal.get_origin(x,y)
        ans['qx'] -= origin[0]
        ans['qy'] -= origin[1]


    # ellipse
    if ellipse is not None:
        braggpeaks = correct_braggpeak_elliptical_distortions(
            braggpeaks,
            p_ellipse = (0,0)+ellipse,
            centered = True
        )

    # pixel size
    if pixel is not None:
        braggpeaks = calibrate_Bragg_peaks_pixel_size(
            braggpeaks,
            q_pixel_size = pixel_size
        )

    # Q/R rotation
    if rotflip is not None:
        rot,flip = rotflip
        braggpeaks = calibrate_bragg_peaks_rotation(
            braggpeaks,
            theta = rot,
            flip = flip
        )


    # return
    return braggpeaks


    return t






### Correct Bragg peak positions, making a circular coordinate system

def correct_braggpeak_elliptical_distortions(
    braggpeaks,
    p_ellipse,
    centered=True
    ):
    """
    Correct the elliptical distortions in a BraggPeaks instance.

    Accepts:
        braggpeaks (PointListArray): the detected, unshifted bragg peaks
        p_ellipse (5-tuple): the ellipse parameters (x0,y0,a,b,theta)
        centered (bool): if True, assumes that the braggpeaks PointListArray has been
            centered, and uses (x0,y0)=(0,0). Otherwise, uses the (x0,y0) from
            `p_ellipse`

    Returns:
        (PointListArray): the corrected Bragg peaks
    """
    assert(isinstance(braggpeaks,PointListArray))

    # Unpack parameters
    x0,y0,a,b,theta = p_ellipse
    if centered:
        x0,y0 = 0,0

    # Get the transformation matrix
    e = b/a
    sint, cost = np.sin(theta-np.pi/2.), np.cos(theta-np.pi/2.)
    T = np.array(
            [
                [e*sint**2 + cost**2, sint*cost*(1-e)],
                [sint*cost*(1-e), sint**2 + e*cost**2]
            ]
        )

    # Correct distortions
    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            x, y = pointlist.data["qx"] - x0, pointlist.data["qy"] - y0
            xyar_i = np.vstack([x, y])
            xyar_f = np.matmul(T, xyar_i)
            pointlist.data["qx"] = xyar_f[0, :] + x0
            pointlist.data["qy"] = xyar_f[1, :] + y0
    return braggpeaks



def calibrate_Bragg_peaks_pixel_size(
    braggpeaks: PointListArray,
    q_pixel_size: Number,
    ):
    """
    Calibrate the reciprocal length of Bragg peak positions.

    Accepts:
        braggpeaks (PointListArray) the detected, unscaled bragg peaks
        q_pixel_size (float) Q pixel size in inverse Ångström

    Returns:
        (PointListArray)
    """
    assert isinstance(braggpeaks, PointListArray)
    assert isinstance(q_pixel_size, Number)

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            pointlist.data["qx"] *= q_pixel_size
            pointlist.data["qy"] *= q_pixel_size

    return braggpeaks



def calibrate_bragg_peaks_rotation(
    braggpeaks: PointListArray,
    theta: float,
    flip: bool,
    ) -> PointListArray:
    """
    Calibrate rotation of Bragg peak positions, using either the R/Q rotation `theta`
    or the `QR_rotation` value inside a Calibration object.

    Accepts:
        braggpeaks  (PointListArray) the CENTERED Bragg peaks
        theta       (float) the rotation between real and reciprocal space in radians
        flip        (bool) whether there is a flip between real and reciprocal space

    Returns:
        braggpeaks_rotated  (PointListArray) the rotated Bragg peaks
    """

    assert isinstance(braggpeaks, PointListArray)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)

            if flip:
                positions = R @ np.vstack((pointlist.data["qy"], pointlist.data["qx"]))
            else:
                positions = R @ np.vstack((pointlist.data["qx"], pointlist.data["qy"]))

            rotated_pointlist = braggpeaks.get_pointlist(Rx, Ry)
            rotated_pointlist.data["qx"] = positions[0, :]
            rotated_pointlist.data["qy"] = positions[1, :]

    return braggpeaks






