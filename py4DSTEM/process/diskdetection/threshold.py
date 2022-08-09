# Bragg peaks hresholding fns

import numpy as np

from ...utils.tqdmnd import tqdmnd





def threshold_Braggpeaks(
    pointlistarray,
    minRelativeIntensity,
    relativeToPeak,
    minPeakSpacing,
    maxNumPeaks
    ):
    """
    Takes a PointListArray of detected Bragg peaks and applies additional
    thresholding, returning the thresholded PointListArray. To skip a threshold,
    set that parameter to False.

    Args:
        pointlistarray (PointListArray): The Bragg peaks. Must have
            coords=('qx','qy','intensity')
        minRelativeIntensity  (float): the minimum allowed peak intensity,
            relative to the brightest peak in each diffraction pattern
        relativeToPeak (int): specifies the peak against which the minimum
            relative intensity is measured -- 0=brightest maximum. 1=next
            brightest, etc.
        minPeakSpacing (int): the minimum allowed spacing between adjacent peaks
        maxNumPeaks (int): maximum number of allowed peaks per diffraction
            pattern
    """
    assert all([item in pointlistarray.dtype.fields for item in ['qx','qy','intensity']]), (
                "pointlistarray must include the coordinates 'qx', 'qy', and 'intensity'.")
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        pointlist.sort(coordinate='intensity', order='descending')

        # Remove peaks below minRelativeIntensity threshold
        if minRelativeIntensity is not False:
            deletemask = pointlist.data['intensity']/pointlist.data['intensity'][relativeToPeak] < \
                                                                           minRelativeIntensity
            pointlist.remove_points(deletemask)

        # Remove peaks that are too close together
        if maxNumPeaks is not False:
            r2 = minPeakSpacing**2
            deletemask = np.zeros(pointlist.length, dtype=bool)
            for i in range(pointlist.length):
                if deletemask[i] == False:
                    tooClose = ( (pointlist.data['qx']-pointlist.data['qx'][i])**2 + \
                                 (pointlist.data['qy']-pointlist.data['qy'][i])**2 ) < r2
                    tooClose[:i+1] = False
                    deletemask[tooClose] = True
            pointlist.remove_points(deletemask)

        # Keep only up to maxNumPeaks
        if maxNumPeaks is not False:
            if maxNumPeaks < pointlist.length:
                deletemask = np.zeros(pointlist.length, dtype=bool)
                deletemask[maxNumPeaks:] = True
                pointlist.remove_points(deletemask)

    return pointlistarray


def universal_threshold(
    pointlistarray,
    thresh,
    metric='maximum',
    minPeakSpacing=False,
    maxNumPeaks=False,
    name=None
    ):
    """
    Takes a PointListArray of detected Bragg peaks and applies universal
    thresholding, returning the thresholded PointListArray. To skip a threshold,
    set that parameter to False.

    Args:
        pointlistarray (PointListArray): The Bragg peaks. Must have
            coords=('qx','qy','intensity')
        thresh (float): the minimum allowed peak intensity. The meaning of this
            threshold value is determined by the value of the 'metric' argument,
            below
        metric (string): the metric used to compare intensities. Must be in
            ('maximum','average','median','manual'). In each case aside from
            'manual', the intensity threshold is set to Val*thresh, where Val is
            given by
                * 'maximum' - the maximum intensity in the entire pointlistarray
                * 'average' - the average of the maximum intensities of each
                  scan position in the pointlistarray
                * 'median' - the medain of the maximum intensities of each
                  scan position in the entire pointlistarray
            If metric is 'manual', the threshold is exactly minIntensity
        minPeakSpacing (int): the minimum allowed spacing between adjacent peaks.
            optional, default is false
        maxNumPeaks (int): maximum number of allowed peaks per diffraction pattern.
            optional, default is false
        name (str, optional): a name for the returned PointListArray. If
            unspecified, takes the old PLA name and appends '_unithresh'.

    Returns:
        (PointListArray): Bragg peaks thresholded by intensity.
    """
    assert isinstance(pointlistarray,PointListArray)
    assert metric in ('maximum','average','median','manual')
    assert all([item in pointlistarray.dtype.fields for item in ['qx','qy','intensity']]), (
                "pointlistarray must include the coordinates 'qx', 'qy', and 'intensity'.")
    _pointlistarray = pointlistarray.copy()
    if name is None:
        _pointlistarray.name = pointlistarray.name+"_unithresh"

    HI_array = np.zeros( (_pointlistarray.shape[0], _pointlistarray.shape[1]) )
    for (Rx, Ry) in tqdmnd(_pointlistarray.shape[0],_pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
            pointlist = _pointlistarray.get_pointlist(Rx,Ry)
            if pointlist.data.shape[0] == 0:
                top_value = np.nan
            else:
                HI_array[Rx, Ry] = np.max(pointlist.data['intensity'])

    if metric=='maximum':
        _thresh = np.max(HI_array)*thresh
    elif metric=='average':
        _thresh = np.nanmean(HI_array)*thresh
    elif metric=='median':
        _thresh = np.median(HI_array)*thresh
    else:
        _thresh = thresh

    for (Rx, Ry) in tqdmnd(_pointlistarray.shape[0],_pointlistarray.shape[1],desc='Thresholding Bragg disks',unit='DP',unit_scale=True):
            pointlist = _pointlistarray.get_pointlist(Rx,Ry)

            # Remove peaks below minRelativeIntensity threshold
            deletemask = pointlist.data['intensity'] < _thresh
            pointlist.remove_points(deletemask)

            # Remove peaks that are too close together
            if maxNumPeaks is not False:
                r2 = minPeakSpacing**2
                deletemask = np.zeros(pointlist.length, dtype=bool)
                for i in range(pointlist.length):
                    if deletemask[i] == False:
                        tooClose = ( (pointlist.data['qx']-pointlist.data['qx'][i])**2 + \
                                     (pointlist.data['qy']-pointlist.data['qy'][i])**2 ) < r2
                        tooClose[:i+1] = False
                        deletemask[tooClose] = True
                pointlist.remove_points(deletemask)

            # Keep only up to maxNumPeaks
            if maxNumPeaks is not False:
                if maxNumPeaks < pointlist.length:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    deletemask[maxNumPeaks:] = True
                    pointlist.remove_points(deletemask)
    return _pointlistarray


def get_pointlistarray_intensities(pointlistarray):
    """
    Concatecates the Bragg peak intensities from a PointListArray of Bragg peak
    positions into one array and returns the intensities. This output can be used
    for understanding the distribution of intensities in your dataset for
    universal thresholding.

    Args:
        pointlistarray (PointListArray):

    Returns:
        (ndarray): all detected peak intensities
    """
    assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']]), (
        "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'.")
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"

    first_pass = True
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1],desc='Getting disk intensities',unit='DP',unit_scale=True):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        for i in range(pointlist.length):
            if first_pass:
                peak_intensities = np.array(pointlist.data[i][2])
                peak_intensities = np.reshape(peak_intensities, 1)
                first_pass = False
            else:
                temp_array = np.array(pointlist.data[i][2])
                temp_array = np.reshape(temp_array, 1)
                peak_intensities = np.append(peak_intensities, temp_array)
    return peak_intensities







