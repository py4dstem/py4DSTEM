import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like

def show(ar,figsize=(8,8),cmap='gray',scaling='none',clipvals='minmax',min=0,max=4,
         power=1,bordercolor=None,borderwidth=5,returnfig=False,figax=None,
         returncax=False,hist=False,n_bins=256,**kwargs):
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
        returncax   (bool) if True, returns the matplotlib color mappable object
                    returned by matshow, used e.g. for adding colorbars
        hist        (bool) if True, instead of plotting a 2D image in ax, plots
                    a histogram of the intensity values of ar, after any scaling
                    this function has performed. Plots the clipvals as dashed
                    vertical lines
        n_bins      (int) number of hist bins
        **kwargs    any keywords accepted by matplotlib's ax.matshow()

    Returns:
        if
            returnfig==False and returncax==False (default),    returns nothing
            returnfig==True  and returncax==False,              returns (fig,ax), the
                                                                matplotlib Figure and Axis
            returnfig==False and returncax==True,               returns cax, the matplotlib
                                                                color mappable object
            returnfig==True  and returncax==True                returns (fig,ax),cax
    """
    assert scaling in ('none','log','power','hist')
    assert clipvals in ('minmax','manual','std','hist')

    if scaling == 'none':
        _ar = ar
    elif scaling == 'log':
        mask = ar>0
        _ar = np.zeros_like(ar)
        _ar[mask] = np.log(ar[mask])
    elif scaling == 'power':
        mask = ar>0
        _ar = np.zeros_like(ar)
        _ar[mask] = np.power(ar[mask],power)
    else:
        raise Exception

    if clipvals == 'minmax':
        vmin,vmax = np.min(_ar),np.max(_ar)
    elif clipvals == 'manual':
        vmin,vmax = min,max
    elif clipvals == 'std':
        m,s = np.median(_ar),np.std(_ar)
        vmin = m - min*s
        vmax = m + max*s
    else:
        raise Exception

    if figax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax = figax
        assert(isinstance(fig,Figure))
        assert(isinstance(ax,Axes))

    if hist:
        hist,bin_edges = np.histogram(_ar,bins=np.linspace(np.min(_ar),np.max(_ar),
                                                           num=n_bins))
        w = bin_edges[1]-bin_edges[0]
        x = bin_edges[:-1]+w/2.
        ax.bar(x,hist,width=w)
        ax.vlines((vmin,vmax),0,ax.get_ylim()[1],color='k',ls='--')
    else:
        cax = ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)

    if bordercolor is not None:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color(bordercolor)
            ax.spines[s].set_linewidth(borderwidth)
        ax.set_xticks([])
        ax.set_yticks([])

    if not returnfig and not returncax:
        if figax is None:
            plt.show()
            return
        else:
            return
    elif returnfig and not returncax:
        return fig,ax
    elif not returnfig and returncax:
        return cax
    else:
        return (fig,ax),cax

def show_rect(ar,lims=(0,1,0,1),color='r',fill=True,alpha=0.25,linewidth=2,returnfig=False,
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
    if isinstance(lims,tuple):
        assert(len(lims)==4)
        lims = [lims]
        N = 1
    else:
        assert(isinstance(lims,list))
        assert(all([isinstance(t,tuple) for t in lims]))
        assert(all([len(t)==4 for t in lims]))
        N = len(lims)
    if isinstance(color,list):
        assert len(color)==N
        assert all([is_color_like(c) for c in color])
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))

    fig,ax = show(ar,returnfig=True,**kwargs)

    for i in range(N):
        l,c,f,a,lw = lims[i],color[i],fill[i],alpha[i],linewidth[i]
        rect = Rectangle((l[2]-0.5,l[0]-0.5),l[3]-l[2],l[1]-l[0],color=c,fill=f,
                          alpha=a,linewidth=lw)
        ax.add_patch(rect)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_circ(ar,center,R,color='r',fill=True,alpha=0.3,linewidth=2,returnfig=False,**kwargs):
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
    if isinstance(color,list):
        assert len(color)==N
        assert all([is_color_like(c) for c in color])
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))


    fig,ax = show(ar,returnfig=True,**kwargs)

    for i in range(N):
        cent,r,col,f,a,lw = center[i],R[i],color[i],fill[i],alpha[i],linewidth[i]
        circ = Circle((cent[1],cent[0]),r,color=col,fill=f,alpha=a,linewidth=lw)
        ax.add_patch(circ)

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
    if isinstance(center,tuple):
        assert(len(center)==2)
        center = [center]
        N = 1
    else:
        assert(isinstance(center,list))
        assert(all([isinstance(c,tuple) for c in center]))
        assert(all([len(c)==2 for c in center]))
        N = len(center)
    if isinstance(Ri,(float,int,np.float)):
        Ri = [Ri for i in range(N)]
    if isinstance(Ro,(float,int,np.float)):
        Ro = [Ro for i in range(N)]
    else:
        assert(isinstance(Ri,list))
        assert(len(Ri)==N)
        assert(all([isinstance(r,(float,int,np.float)) for r in Ri]))
        assert(isinstance(Ro,list))
        assert(len(Ro)==N)
        assert(all([isinstance(r,(float,int,np.float)) for r in Ro]))
    if isinstance(color,list):
        assert len(color)==N
        assert all([is_color_like(c) for c in color])
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))


    fig,ax = show(ar,returnfig=True,**kwargs)

    for i in range(N):
        cent,ri,ro,col,f,a,lw = center[i],Ri[i],Ro[i],color[i],fill[i],alpha[i],linewidth[i]
        annulus = Wedge((cent[1],cent[0]),ro,0,360,width=ro-ri,color=col,fill=f,alpha=a,linewidth=lw)
        ax.add_patch(annulus)

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

def show_hist(arr, bins = 200, returnfig = False):
    """
    Visualization function to show histogram from any ndarray (arr).
    
    Accepts:
        arr                 (ndarray) any array
        bins                (int) number of bins that the intensity values will be sorted into for histogram
        returnfig           (bool) determines whether or not figure and its axis are returned (see Returns)
    
    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    counts, bin_edges = np.histogram(arr, bins = bins, range = (np.min(arr), np.max(arr)) )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width/2
    
    fig, ax = plt.subplots(1,1)
    ax.bar(bin_centers, counts, width = bin_width, align = 'center')
    plt.ylabel('Counts')
    plt.xlabel('Intensity')
    
    if not returnfig:
        plt.show()
        return
    else:
        return fig, ax

