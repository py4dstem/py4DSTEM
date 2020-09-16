import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .visualize import show,show_points

def show_DP_grid(datacube,x0,y0,xL,yL,axsize=(6,6),returnfig=False,**kwargs):
    """
    Shows a grid of diffraction patterns from DataCube datacube, starting from
    scan position (x0,y0) and extending xL,yL.

    Accepts:
        datacube        (DataCube) the 4D-STEM data
        (x0,y0)         the corner of the grid of DPs to display
        xL,yL           the extent of the grid
        axsize          the size of each diffraction pattern

    Returns:
        if returnfig==false (default), the figure is plotted and nothing is returned.
        if returnfig==false, the figure and its one axis are returned, and can be
        further edited.
    """
    yy,xx = np.meshgrid(np.arange(y0,y0+yL),np.arange(x0,x0+xL))

    fig,axs = plt.subplots(xL,yL,figsize=(yL*axsize[0],xL*axsize[1]))
    for xi in range(xL):
        for yi in range(yL):
            ax = axs[xi,yi]
            x,y = xx[xi,yi],yy[xi,yi]
            dp = datacube.data[x,y,:,:]
            _,_ = show(dp,ax=(fig,ax),returnfig=True,**kwargs)
    plt.tight_layout()

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_grid_overlay(ar,x0,y0,xL,yL,color='k',linewidth=1,alpha=1,
                                            returnfig=False,**kwargs):
    """
    Shows the image ar with an overlaid boxgrid outline about the pixels
    beginning at (x0,y0) and with extent xL,yL in the two directions.

    Accepts:
        ar          the image array
        x0,y0       the corner of the grid
        xL,xL       the extent of the grid
    """
    yy,xx = np.meshgrid(np.arange(y0,y0+yL),np.arange(x0,x0+xL))

    fig,ax = show(ar,returnfig=True,**kwargs)
    for xi in range(xL):
        for yi in range(yL):
            x,y = xx[xi,yi],yy[xi,yi]
            rect = Rectangle((y-0.5,x-0.5),1,1,lw=linewidth,color=color,
                             alpha=alpha,fill=False)
            ax.add_patch(rect)
    plt.tight_layout()

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_image_grid(get_ar,H,W,axsize=(6,6),returnfig=False,titlesize=0,
                    get_bc=None,get_x=None,get_y=None,get_pointcolors=None,
                    get_s=None,**kwargs):
    """
    Displays a set of images in a grid.

    The images are specified by some function get_ar(i), which returns an
    image for values of some integer index i.  The values of i passed to
    get_ar are 0 through HW-1.

    To display the first 4 two-dimensional slices of some 3D array ar
    some 3D array ar, you can do

        >>> show_image_grid(lambda i:ar[:,:,i], H=2, W=2)

    Its also possible to add colored borders, or overlaid points,
    using similar functions to get_ar, i.e. functions which return
    the color or set of points of interest as a function of index
    i, which must be defined in the range [0,HW-1].

    Accepts:
        get_ar      a function which returns a 2D array when passed
                    the integers 0 through HW-1
        H,W         integers, the dimensions of the grid
        axsize      the size of each image
        titlesize   if >0, prints the index i passed to get_ar over
                    each image
        get_bc      if not None, should be a function defined over
                    the same i as get_ar, and which returns a
                    valid matplotlib color for each i. Adds
                    a colored bounding box about each image. E.g.
                    if `colors` is an array of colors:

        >>> show_image_grid(lambda i:ar[:,:,i],H=2,W=2,
                            get_bc=lambda i:colors[i])

        get_x,get_y     functions which returns sets of x/y positions
                        as a function of index i
        get_s           function which returns a set of point sizes
                        as a function of index i
        get_pointcolors a function which returns a color or list of colors
                        as a function of index i

    Returns:
        if returnfig==false (default), the figure is plotted and nothing is returned.
        if returnfig==false, the figure and its one axis are returned, and can be
        further edited.
    """
    _get_bc = get_bc is not None
    _get_points = (get_x is not None) and (get_y is not None)
    _get_colors = get_pointcolors is not None
    _get_s = get_s is not None
    fig,axs = plt.subplots(H,W,figsize=(W*axsize[0],H*axsize[1]))
    if H==1:
        axs = axs[np.newaxis,:]
    elif W==1:
        axs = axs[:,np.newaxis]
    for i in range(H):
        for j in range(W):
            ax = axs[i,j]
            N = i*W+j
            try:
                ar = get_ar(N)
                if _get_bc and _get_points:
                    bc = get_bc(N)
                    x,y = get_x(N),get_y(N)
                    if _get_colors:
                        pointcolors = get_pointcolors(N)
                    else:
                        pointcolors='r'
                    if _get_s:
                        s = get_s(N)
                        _,_ = show_points(ar,ax=(fig,ax),returnfig=True,
                                          bordercolor=bc,x=x,y=y,s=s,
                                          point_color=pointcolors,**kwargs)
                    else:
                        _,_ = show_points(ar,ax=(fig,ax),returnfig=True,
                                          bordercolor=bc,x=x,y=y,
                                          point_color=pointcolors,**kwargs)
                elif _get_bc:
                    bc = get_bc(N)
                    _,_ = show(ar,ax=(fig,ax),returnfig=True,
                               bordercolor=bc,**kwargs)
                elif _get_points:
                    x,y = get_x(N),get_y(N)
                    if _get_colors:
                        pointcolors = get_pointcolors(N)
                    else:
                        pointcolors='r'
                    _,_ = show_points(ar,ax=(fig,ax),returnfig=True,
                                      point_color=pointcolors,**kwargs)
                else:
                    _,_ = show(ar,ax=(fig,ax),returnfig=True,**kwargs)
                if titlesize>0:
                    ax.set_title(N,fontsize=titlesize)
            except IndexError:
                ax.axis('off')
    plt.tight_layout()

    if not returnfig:
        plt.show()
        return
    else:
        return fig,axs


