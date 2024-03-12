import numpy as np



def radial_reduction(ar, x0, y0, binsize=1, fn=np.mean, coords=None):
    """
    Evaluate a reduction function on pixels within annular rings centered on (x0,y0),
    with a ring width of binsize.

    By default, returns the mean value of pixels within each annulus.
    Some other useful reductions include: np.sum, np.std, np.count, np.median, ...

    When running in a loop, pre-compute the pixel coordinates and pass them in
    for improved performance, like so:
        coords = np.mgrid[0:ar.shape[0],0:ar.shape[1]]
        radial_sums = radial_reduction(ar, x0,y0, coords=coords)
    """
    qx, qy = coords if coords else np.mgrid[0 : ar.shape[0], 0 : ar.shape[1]]

    r = (
        np.floor(np.hypot(qx - x0, qy - y0).ravel() / binsize).astype(np.int64)
        * binsize
    )
    edges = np.cumsum(np.bincount(r)[::binsize])
    slices = [slice(0, edges[0])] + [
        slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
    ]
    rargsort = np.argsort(r)
    sorted_ar = ar.ravel()[rargsort]
    reductions = np.array([fn(sorted_ar[s]) for s in slices])

    return reductions



