import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

def show(ar,sig_min=0,sig_max=3,power=1,figsize=(12,12),returnfig=False,**kwargs):
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

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
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
    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_rect(ar,sig_min=0,sig_max=3,power=1,figsize=(12,12),returnfig=False,
              lims=(0,1,0,1),color='r',fill=True,alpha=1,**kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed rectangles.
    To overlay one rectangle, lims must be a single 4-tuple.  To overlay N rectangles,
    lims must be a list of N 4-tuples.  color, fill, and alpha may each be single values,
    which are then applied to all the rectangles, or a length N list.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        lims        (4-tuple, or list of N 4-tuples) the rectangle bounds (x0,xf,y0,yf)
        color       (string of list of N strings)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    if isinstance(lims,tuple):
        assert(len(lims)==4)
        lims = [lims]
        N = 1
    else:
        assert(isinstance(lims,list))
        assert(all([isinstance(t,tuple) for t in lims]))
        assert(all([len(t)==4 for t in lims]))
        N = len(lims)
    if isinstance(color,str):
        color = [color for i in range(N)]
    else:
        assert(isinstance(color,list))
        assert(len(color)==N)
        assert(all([isinstance(c,str) for c in color]))
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(a,(float,int,np.float)) for a in alpha]))

    fig,ax = show(ar,sig_min,sig_max,power,figsize,returnfig=True,**kwargs)

    rects = []
    for i in range(N):
        l,c,f,a = lims[i],color[i],fill[i],alpha[i]
        rect = Rectangle((l[2],l[0]),l[3]-l[2],l[1]-l[0],color=c,fill=f,alpha=a)
        ax.add_patch(rect)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax
def show_circ(ar,sig_min=0,sig_max=3,power=1,figsize=(12,12),returnfig=False,
              center=(0,0),R=10,color='r',fill=True,alpha=1,**kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed circles.
    To overlay one circle, center must be a single 2-tuple.  To overlay N rectangles,
    center must be a list of N 2-tuples.  color, fill, and alpha may each be single values,
    which are then applied to all the circles, or a length N list.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        center      (2-tuple, or list of N 2-tuples) the center of the circle (x0,y0)
        R           (number of list of N numbers) the circles radius
        color       (string of list of N strings)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    if isinstance(center,tuple):
        assert(len(center)==2)
        center = [center]
        N = 1
    else:
        assert(isinstance(center,list))
        assert(all([isinstance(c,tuple) for c in center]))
        assert(all([len(c)==2 for c in center]))
        N = len(center)
    if isinstance(R,(float,int,np.float)):
        R = [R for i in range(N)]
    else:
        assert(isinstance(R,list))
        assert(len(R)==N)
        assert(all([isinstance(r,(float,int,np.float)) for r in R]))
    if isinstance(color,str):
        color = [color for i in range(N)]
    else:
        assert(isinstance(color,list))
        assert(len(color)==N)
        assert(all([isinstance(c,str) for c in color]))
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(a,(float,int,np.float)) for a in alpha]))

    fig,ax = show(ar,sig_min,sig_max,power,figsize,returnfig=True,**kwargs)

    rects = []
    for i in range(N):
        cent,r,col,f,a = center[i],R[i],color[i],fill[i],alpha[i]
        circ = Circle((cent[1],cent[0]),r,color=col,fill=f,alpha=a)
        ax.add_patch(circ)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

