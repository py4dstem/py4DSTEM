import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np



def filter_2D_maxima(
    maxima,
    minAbsoluteIntensity=0,
    minRelativeIntensity=0,
    relativeToPeak=0,
    minSpacing=0,
    edgeBoundary=1,
    maxNumPeaks=1,
):
    """
    Parameters
    ----------
    maxima : a numpy structured array with fields 'x', 'y', 'intensity'
    minAbsoluteIntensity : number
        delete counts with intensity below this value
    minRelativeIntensity : number
        delete counts with intensity below this value times
        the intensity of the i'th peak, where i is given by `relativeToPeak`
    relativeToPeak : int
        0=brightest peak, 1=second brightest peak, etc
    minSpacing : number
        if two peaks are within this euclidean distance from one
        another, delete the less intense of the two
    edgeBoundary : number
        delete peaks within this distance of the image edge
    maxNumPeaks : int
        maximum number of peaks to return

    Returns
    -------
    a numpy structured array with fields 'x', 'y', 'intensity'
    """

    # Remove maxima which are too dim
    if minAbsoluteIntensity > 0:
        deletemask = maxima["intensity"] < minAbsoluteIntensity
        maxima = maxima[~deletemask]

    # Remove maxima which are too dim, compared to the n-th brightest
    if (minRelativeIntensity > 0) & (len(maxima) > relativeToPeak):
        assert isinstance(relativeToPeak, (int, np.integer))
        deletemask = (
            maxima["intensity"] / maxima["intensity"][relativeToPeak]
            < minRelativeIntensity
        )
        maxima = maxima[~deletemask]

    # Remove maxima which are too close
    if minSpacing > 0:
        deletemask = np.zeros(len(maxima), dtype=bool)
        for i in range(len(maxima)):
            if deletemask[i] == False:  # noqa: E712
                tooClose = (
                    (maxima["x"] - maxima["x"][i]) ** 2
                    + (maxima["y"] - maxima["y"][i]) ** 2
                ) < minSpacing**2
                tooClose[: i + 1] = False
                deletemask[tooClose] = True
        maxima = maxima[~deletemask]

    # Remove maxima in excess of maxNumPeaks
    if maxNumPeaks is not None:
        if len(maxima) > maxNumPeaks:
            maxima = maxima[:maxNumPeaks]

    return maxima




