import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like,ListedColormap
from .overlay import add_rectangles,add_circles,add_annuli,add_ellipses

def show(ar,figsize=(8,8),cmap='gray',scaling='none',clipvals='minmax',min=None,max=None,
         power=1,bordercolor=None,borderwidth=5,returnfig=False,figax=None,
         hist=False,n_bins=256,mask=None,mask_color='k',rectangle=None,
         circle=None,annulus=None,ellipse=None,points=None,grid=None,**kwargs):
    """
    General visualization function for 2D arrays.

    Accepts:
        ar          (2D array) the array to plot
        figsize     (2-tuple) size of the plot
        cmap        (colormap) any matplotlib cmap; default is gray
        scaling     (str) selects a scaling scheme for the intensity values.
                    Default is none. Accepted values:
                        'none'
                        'log'       values where ar<=0 are set to 0
                        'power'     requires the 'power' argument be set.
                                    values where ar<=0 are set to 0
        clipvals    (str) method for setting clipvalues.  Default is the array
                    min and max values. Accepted values:
                        'minmax'    The min/max values are np.min(ar)/np.max(r)

                        'manual'    The min/max values are set to the values of
                                    the min,max arguments received by this function
                        'std'       The min/max values are
                                            np.median(ar) -/+ N*np.std(ar),
                                    and N is this functions min,max vals.
                        'centered'  The min/max values are set to
                                            c -/+ m
                                    Where by default 'c' is zero and m is
                                    the max(abs(ar-c), or the two params
                                    can be user specified using the kwargs
                                    min/max -> c/m.
        min         (number) behavior depends on clipvals
        max         (number)
        power       (number)
        bordercolor (str or None) if not None, add a border of this color.
                    The color can be anything matplotlib recognizes as a color.
        borderwidth (number)
        returnfig   (bool) if True, the function returns the tuple (figure,axis)
        figax       (None or 2-tuple) controls which matplotlib Axes object draws
                    the image.  If None, generates a new figure with a single
                    Axes instance. Otherwise, ax must be a 2-tuple containing
                    the matplotlib class instances (Figure,Axes), with ar then
                    plotted in the specified Axes instance.
        hist        (bool) if True, instead of plotting a 2D image in ax, plots
                    a histogram of the intensity values of ar, after any scaling
                    this function has performed. Plots the clipvals as dashed
                    vertical lines
        n_bins      (int) number of hist bins
        mask        (None or boolean array) if not None, must have the same shape
                    as 'ar'. Wherever mask==True, pixel values are set to
                    mask_color. If hist==True, ignore these values in the histogram
        mask_color  (color) see 'mask'
        **kwargs    any keywords accepted by matplotlib's ax.matshow()

    Returns:
        if returnfig==False (default), the figure is plotted and nothing is returned.
        if returnfig==True, return the figure and the axis.
    """
    assert scaling in ('none','log','power','hist')
    assert clipvals in ('minmax','manual','std','centered')
    if mask is not None:
        assert mask.shape == ar.shape
        assert is_color_like(mask_color)

    # Perform any scaling
    if scaling == 'none':
        _ar = ar
    elif scaling == 'log':
        _mask = ar>0
        _ar = np.zeros_like(ar,dtype=float)
        _ar[_mask] = np.log(ar[_mask])
    elif scaling == 'power':
        _mask = ar>0
        _ar = np.zeros_like(ar,dtype=float)
        _ar[_mask] = np.power(ar[_mask],power)
    else:
        raise Exception

    # Set the clipvalues
    if clipvals == 'minmax':
        vmin,vmax = np.min(_ar),np.max(_ar)
    elif clipvals == 'manual':
        assert min is not None and max is not None
        vmin,vmax = min,max
    elif clipvals == 'std':
        assert min is not None and max is not None
        m,s = np.median(_ar),np.std(_ar)
        vmin = m - min*s
        vmax = m + max*s
    elif clipvals == 'centered':
        c = 0 if min is None else min
        m = np.max(np.abs(ar)) if max is None else max
        vmin = c-m
        vmax = c+m
    else:
        raise Exception

    # Create or attach to the appropriate Figure and Axis
    if figax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax = figax
        assert(isinstance(fig,Figure))
        assert(isinstance(ax,Axes))

    # Plot the image
    if not hist:
        ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)
        if mask is not None:
            mask_display = np.ma.array(data=mask,mask=mask==False)
            cmap = ListedColormap([mask_color,'w'])
            ax.matshow(mask_display,cmap=cmap)
    # ...or, plot its histogram
    else:
        if mask is None:
            hist,bin_edges = np.histogram(_ar,bins=np.linspace(np.min(_ar),np.max(_ar),
                                                               num=n_bins))
        else:
            hist,bin_edges = np.histogram(_ar[mask==False],
                                   bins=np.linspace(np.min(_ar),np.max(_ar),num=n_bins))
        w = bin_edges[1]-bin_edges[0]
        x = bin_edges[:-1]+w/2.
        ax.bar(x,hist,width=w)
        ax.vlines((vmin,vmax),0,ax.get_ylim()[1],color='k',ls='--')

    # Add any overlays
    if rectangle is not None:
        add_rectangles(ax,rectangle)
    if circle is not None:
        add_circles(ax,circle)
    if annulus is not None:
        add_annuli(ax,annulus)
    if ellipse is not None:
        add_ellipses(ax,ellipse)
    if grid is not None:
        add_grid(ax,grid)

    # Add a border
    if bordercolor is not None:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color(bordercolor)
            ax.spines[s].set_linewidth(borderwidth)
        ax.set_xticks([])
        ax.set_yticks([])

    # Show or return
    if returnfig:
        return fig,ax
    elif figax is not None:
        return
    else:
        plt.show()
        return

def show_hist(arr, bins=200, vlines=None, vlinecolor='k', vlinestyle='--',
                                        returnhist=False, returnfig=False):
    """
    Visualization function to show histogram from any ndarray (arr).

    Accepts:
        arr                 (ndarray) any array
        bins                (int) number of bins that the intensity values will be sorted
                            into for histogram
        returnhist          (bool) determines whether or not the histogram values are
                            returned (see Returns)
        returnfig           (bool) determines whether or not figure and its axis are
                            returned (see Returns)

    Returns:
        If
            returnhist==False and returnfig==False      returns nothing
            returnhist==True  and returnfig==True       returns (counts,bin_edges) the histogram
                                                        values and bin edge locations
            returnhist==False and returnfig==True       returns (fig,ax), the Figure and Axis
            returnhist==True  and returnfig==True       returns (hist,bin_edges),(fig,ax)
    """
    counts, bin_edges = np.histogram(arr, bins=bins, range=(np.min(arr), np.max(arr)))
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width/2

    fig, ax = plt.subplots(1,1)
    ax.bar(bin_centers, counts, width = bin_width, align = 'center')
    plt.ylabel('Counts')
    plt.xlabel('Intensity')
    if vlines is not None:
        ax.vlines(vlines,0,np.max(counts),color=vlinecolor,ls=vlinestyle)

    if not returnhist and not returnfig:
        plt.show()
        return
    elif returnhist and not returnfig:
        return counts,bins_edges
    elif not returnhist and returnfig:
        return fig, ax
    else:
        return (counts,bin_edges),(fig,ax)

# Show with overlays

def show_rectangles(ar,lims=(0,1,0,1),color='r',fill=True,alpha=0.25,linewidth=2,returnfig=False,
                    **kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed rectangles.
    lims is specified in the order (x0,xf,y0,yf). The rectangle bounds begin at the upper
    left corner of (x0,y0) and end at the upper left corner of (xf,yf) -- i.e inclusive
    in the lower bound, exclusive in the upper bound -- so that the boxed region encloses
    the area of array ar specified by ar[x0:xf,y0:yf].

    To overlay one rectangle, lims must be a single 4-tuple.  To overlay N rectangles,
    lims must be a list of N 4-tuples.  color, fill, and alpha may each be single values,
    which are then applied to all the rectangles, or a length N list.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        lims        (4-tuple, or list of N 4-tuples) the rectangle bounds (x0,xf,y0,yf)
        color       (valid matplotlib color, or list of N colors)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency
        linewidth   (number)

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'lims':lims,'color':color,'fill':fill,'alpha':alpha,'linewidth':linewidth}
    add_rectangles(ax,d)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_circles(ar,center,R,color='r',fill=True,alpha=0.3,linewidth=2,returnfig=False,**kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed circles.
    To overlay one circle, center must be a single 2-tuple.  To overlay N circles,
    center must be a list of N 2-tuples.  color, fill, and alpha may each be single values,
    which are then applied to all the circles, or a length N list.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        center      (2-tuple, or list of N 2-tuples) the center of the circle (x0,y0)
        R           (number of list of N numbers) the circles radius
        color       (valid matplotlib color, or list of N colors)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency
        linewidth   (number)

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'center':center,'R':R,'color':color,'fill':fill,'alpha':alpha,'linewidth':linewidth}
    add_circles(ax,d)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_ellipses(ar,center,a,e,theta,color='r',fill=True,alpha=0.3,linewidth=2,
                                                        returnfig=False,**kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed ellipses.
    To overlay one ellipse, center must be a single 2-tuple.  To overlay N circles,
    center must be a list of N 2-tuples.  Similarly, the remaining ellipse parameters -
    a, e, and theta - must each be a single number or a len-N list.  color, fill, and
    alpha may each be single values, which are then applied to all the circles, or
    length N lists.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        center      (2-tuple, or list of N 2-tuples) the center of the circle (x0,y0)
        a           (number or list of N numbers) the semimajor axis length
        e           (number or list of N numbers) ratio of semiminor/semimajor length
        theta       (number or list of N numbers) the tilt angle in radians
        color       (valid matplotlib color, or list of N colors)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency
        linewidth   (number)

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'center':center,'a':a,'e':e,'theta':theta,'color':color,'fill':fill,
         'alpha':alpha,'linewidth':linewidth}
    add_ellipses(ax,d)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_annuli(ar,center,Ri,Ro,color='r',fill=True,alpha=0.3,linewidth=2,returnfig=False,
                **kwargs):
    """
    Visualization function which plots a 2D array with one or more overlayed annuli.
    To overlay one annulus, center must be a single 2-tuple.  To overlay N annuli,
    center must be a list of N 2-tuples.  color, fill, and alpha may each be single values,
    which are then applied to all the circles, or a length N list.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        center      (2-tuple, or list of N 2-tuples) the center of the annulus (x0,y0)
        Ri,Ro       (number of list of N numbers) the inner and outer radii
        color       (string of list of N strings)
        fill        (bool or list of N bools) filled in or empty rectangles
        alpha       (number, 0 to 1) transparency
        linewidth   (number)

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'center':center,'Ri':Ri,'Ro':Ro,'color':color,'fill':fill,'alpha':alpha,
         'linewidth':linewidth}
    add_annuli(ax,d)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_points(ar,x,y,s=1,scale=50,alpha=1,point_color='r',
                figsize=(12,12),returnfig=False,**kwargs):
    """
    Visualization function which plots a 2D array with one or more points.
    x and y are the point centers and must have the same length, N.
    s and point_color are the size and color and must each have length 1 or N.

    See the docstring for py4DSTEM.visualize.show() for descriptions of all input
    parameters not listed below.

    Accepts:
        ar          (array) the image
        x,y         (number or iterable of numbers) the point positions
        s           (number or iterable of numbers) the point sizes
        point_color

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,figsize,returnfig=True,**kwargs)
    ax.scatter(y,x,s=s*scale/np.max(s),color=point_color)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax



