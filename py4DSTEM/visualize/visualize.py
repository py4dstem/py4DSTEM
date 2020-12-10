import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def show(ar,min=0,max=3,power=1,figsize=(12,12),contrast='std',ax=None,
         bordercolor=None,borderwidth=5,returnfig=False,cmap='gray',**kwargs):
    """
    General visualization function for 2D arrays.

    Accepts:
        ar          (2D array) the array to plot
        min         (number) minimum display value is set to (median - min*std)
        max         (number) maximum display value is set to (median + max*std)
        power       (number) should be <=1; pixel values are set to val**power
        figsize     (2-tuple) size of the plot
        contrast    (str) determines how image contrast is set. Accepted values
                    and their behaviors are as follows:
                        'std'       (default) The min/max values are
                                    median -/+ n*std, where the two values of n
                                    are given by the arguments min and max.
                        'minmax'    The min/max values are np.min(ar)/np.max(r)
                        'vrange'    The min/max values are set to min/max
                        'log'       Takes log_10(ar) and visualizes, clipping the
                                    histogram at the min/max vals. Values <=0 in
                                    ar are set to 0.
        ax          (None or 2-tuple) if None, generates a new figure with a single
                    Axes instance. Otherwise, ax must be a 2-tuple containing
                    the matplotlib class instances (Figure,Axes), and ar is
                    plotted in the specified Axes instance.
                    the passed Axes instance.
        bordercolor (str or None) if not None, add a border of this color
        borderwidth (number)
        **kwargs    any keywords accepted by matplotlib's ax.matshow()

    Returns:
        if returnfig==False (default), the figure is plotted and nothing is returned.
        if returnfig==True, return the figure and the axis.
    """
    if np.min(ar)<0 and power!=1:
        mask = ar<0
        _ar = np.abs(ar)**power
        _ar[mask] = -_ar[mask]
    elif contrast == 'log':
        _ar = np.where(ar>0,np.log(ar),0)
    else:
        _ar = ar**power

    if contrast == 'std':
        med,std = np.median(_ar),np.std(_ar)
        vmin = med-min*std
        vmax = med+max*std
    elif contrast == 'minmax':
        vmin = np.min(_ar)
        vmax = np.max(_ar)
    elif contrast == 'vrange':
        vmin = min
        vmax = max
    elif contrast == 'log':
        vmin=np.min(_ar)
        vmax=np.max(_ar)
    else:
        raise Exception("Parameter contrast must be 'std' or 'minmax' or 'vrange' or 'log'.")

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax = ax
        assert(isinstance(fig,Figure))
        assert(isinstance(ax,Axes))

    ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)
    if bordercolor is not None:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color(bordercolor)
            ax.spines[s].set_linewidth(borderwidth)
        ax.set_xticks([])
        ax.set_yticks([])
    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def ax_show(ar,ax,min=0,max=3,power=1,contrast='std',
            bordercolor=None,borderwidth=5,cmap='gray',returncax=False,**kwargs):
    """
    Visualization function which displays the 2D array `ar` in a matplotlib
    axis instance `ax`. `ax` must be generated and plotted elsewhere.

    Accepts:
        ar          (2D array) the array to plot
        ax          (matplotlib.axis) the axis the plot will be displayed in
        min         (number) minimum display value is set to (median - min*std)
        max         (number) maximum display value is set to (median + max*std)
        power       (number) should be <=1; pixel values are set to val**power
        contrast    (str) determines how image contrast is set. Accepted values
                    and their behaviors are as follows:
                        'std'       (default) The min/max values are
                                    median -/+ n*std, where the two values of n
                                    are given by the arguments min and max.
                        'minmax'    The min/max values are np.min(ar)/np.max(r)
                        'vrange'    The min/max values are set to min/max
        ax          (None or 2-tuple) if None, generates a new figure with a single
                    Axes instance. Otherwise, ax must be a 2-tuple containing
                    the matplotlib class instances (Figure,Axes), and ar is
                    plotted in the specified Axes instance.
                    the passed Axes instance.
        bordercolor (str or None) if not None, add a border of this color
        borderwidth (number)
        returncax   (bool) if True, returns the matplotlib.image.AxesImage,
                    i.e. the image attached to the axis; used to add colorbars
        **kwargs    any keywords accepted by matplotlib's ax.matshow()
    """
    assert isinstance(ax,Axes), "ax must be a matplotlib.axis.Axes instance"

    if np.min(ar)<0 and power!=1:
        mask = ar<0
        _ar = np.abs(ar)**power
        _ar[mask] = -_ar[mask]
    else:
        _ar = ar**power

    if contrast == 'std':
        med,std = np.median(_ar),np.std(_ar)
        vmin = med-min*std
        vmax = med+max*std
    elif contrast == 'minmax':
        vmin = np.min(_ar)
        vmax = np.max(_ar)
    elif contrast == 'vrange':
        vmin = min
        vmax = max
    else:
        raise Exception("Parameter contrast must be 'std' or 'minmax' or 'vrange'.")

    kwargs.pop('vmin',None),kwargs.pop('vmax',None)
    ax_image = ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)
    if bordercolor is not None:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color(bordercolor)
            ax.spines[s].set_linewidth(borderwidth)
        ax.set_xticks([])
        ax.set_yticks([])
    if returncax:
        return ax_image

def show_rect(ar,min=0,max=3,power=1,figsize=(12,12),returnfig=False,
              lims=(0,1,0,1),color='r',fill=True,alpha=1,linewidth=2,**kwargs):
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
        color       (string of list of N strings)
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))

    fig,ax = show(ar,min,max,power,figsize,returnfig=True,**kwargs)

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

def show_circ(ar,min=0,max=3,power=1,figsize=(12,12),returnfig=False,
              center=(0,0),R=10,color='r',fill=True,alpha=1,linewidth=2,**kwargs):
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))


    fig,ax = show(ar,min,max,power,figsize,returnfig=True,**kwargs)

    for i in range(N):
        cent,r,col,f,a,lw = center[i],R[i],color[i],fill[i],alpha[i],linewidth[i]
        circ = Circle((cent[1],cent[0]),r,color=col,fill=f,alpha=a,linewidth=lw)
        ax.add_patch(circ)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_annuli(ar,min=0,max=3,power=1,figsize=(12,12),returnfig=False,
              center=(0,0),Ri=10,Ro=20,color='r',fill=True,alpha=1,linewidth=2,**kwargs):
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
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))


    fig,ax = show(ar,min,max,power,figsize,returnfig=True,**kwargs)

    for i in range(N):
        cent,ri,ro,col,f,a,lw = center[i],Ri[i],Ro[i],color[i],fill[i],alpha[i],linewidth[i]
        annulus = Wedge((cent[1],cent[0]),ro,0,360,width=ro-ri,color=col,fill=f,alpha=a,linewidth=lw)
        ax.add_patch(annulus)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_points(ar,x,y,s=1,scale=500,alpha=1,point_color='r',
                min=0,max=3,power=1,figsize=(12,12),returnfig=False,**kwargs):
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
    fig,ax = show(ar,min,max,power,figsize,returnfig=True,**kwargs)
    ax.scatter(y,x,s=s*scale/np.max(s),color=point_color)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax



