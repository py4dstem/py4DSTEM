import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

def show(ar,sig_min=0,sig_max=3,power=1,figsize=(12,12),**kwargs):
    """
    General visualization function for a single 2D array while setting the min/max
    values in terms of standard deviations from the median - often useful for
    diffraction data.  Data is optionally scaled by some power, handling any negative
    pixel values by setting them to -(-val)**power.  **kwargs is passed directly to
    ax.matshow(), and must be keywords this method accepts.

    Accepts:
        ar          (2D array) the array to plot
        sig_min     (number) minimum display value is set to (median - sig_min*std)
        sig_max     (number) maximum display value is set to (median + sig_max*std)
        power       (number) should be <=1; pixel values are set to val**power
        figsize     (2-tuple) size of the plot
        **kwargs    any keywords accepted by matplotlib's ax.matshow()
    """
    if np.min(ar)<0 and power!=1:
        mask = ar<0
        _ar = np.abs(ar)**power
        _ar[mask] = -_ar[mask]
    else:
        _ar = ar**power

    med,std = np.median(_ar),np.std(_ar)
    vmin = med-sig_min*std
    vmax = med+sig_max*std

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.matshow(_ar,vmin=vmin,vmax=vmax,**kwargs)
    plt.show()

    return


