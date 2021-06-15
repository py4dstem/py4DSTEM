import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like,ListedColormap
from numpy.ma import MaskedArray
from numbers import Number
from math import log
from .overlay import add_rectangles,add_circles,add_annuli,add_ellipses,add_points
from .overlay import add_cartesian_grid,add_polarelliptical_grid,add_rtheta_grid,add_scalebar
from ..io.datastructure import Coordinates

def show(ar,figsize=(8,8),cmap='gray',scaling='none',clipvals='minmax',
         min=None,max=None,power=1,bordercolor=None,borderwidth=5,
         returnclipvals=False,returncax=False,returnfig=False,figax=None,
         hist=False,n_bins=256,mask=None,mask_color='k',
         rectangle=None,circle=None,annulus=None,ellipse=None,points=None,grid_overlay=None,
         cartesian_grid=None,polarelliptical_grid=None,rtheta_grid=None,scalebar=None,
         coordinates=None,rx=None,ry=None,space='Q',
         pixelsize=None,pixelunits=None,x0=None,y0=None,e=None,theta=None,
         **kwargs):
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
                    as 'ar'. Wherever mask==True, plot the pixel normally, and where
                    mask==False, pixel values are set to mask_color. If hist==True,
                    ignore these values in the histogram
        mask_color  (color) see 'mask'
        **kwargs    any keywords accepted by matplotlib's ax.matshow()

        ---------------------------------Overlays-------------------------------

        The following arguments generate overlays, including points, shapes,
        scalebars, and coordinate grids. If they are not None,
        they should be a dictionary, which is passed to the corresponding
        add_* functions (e.g. add_rectangles(ax,dict), add_circles(), etc.)

                      ---------------Shapes and Points-------------

        The shape/point overlays are:

            rectangle   (dictionary)
            circle
            ellipse
            points
            grid_overlay

        See those functions' docstrings for acceptable dict keyword-value pairs.

               ---------------Coordinate grids and Scalebars-------------

        For coordinate systems and scalebars, as above, the argument passed to this
        function should be a dictionary.  The passed dict may be empty, in which case
        default formatting options will be used.

        The coordinate/scalebar overlays are:

            cartesian_grid
            polarelliptical_grid
            rtheta_grid
            scalebar

        There are two types of additional optional parameters which can be passed to
        specify how coordinates or scalebars should be plotted: (1) metadata specifying
        the scale, units, and other parameters of the coordinates, e.g. the position
        of the origin, and (2) visual formatting parameters.

        (1) scale/unit/coordinate parameter metadata should be passed as arguments to the
        show function. These may be passed manually, or in a Coordinates instance.

        To use a Coordinates instance, pass it to the 'coordinates' argument.  The
        function will find and use relevant metadata it finds in the instance.  Some
        metadata may vary by scan position - e.g. the origin of diffraction space.  In
        this case please also pass this function a scan position with the parameters
        'rx' and 'ry'.

        To pass metadata manually, just specify the values as function arguments.  The
        relevant arguments are: pixelsize, pixelunits, x0, y0, e, theta. Any arguments
        passed in this way will supercede the values found in a Coordinates instance.

        Use the argument 'space' to specify whether the data is in diffraction space,
        real space, or r-theta space, with values of ('Q','R','rtheta').

        (2) visual formatting parameters should be passed as keyword-value pairs inside
        the dictionary associated with the overlay being drawn.  E.g.:

            >>> show(im,cartesian_grid{'color':'black','alpha':0.25,'label':True}

        will plot an image with a cartesian grid overlay in black, with a transparency
        set to 0.25, and with labeled axes and tickmarks.  For the full set of
        formatting parameters which can be specified for each overlay, see the docstring
        for the corresponding add_* function.

    Returns:
        if returnfig==False (default), the figure is plotted and nothing is returned.
        if returnfig==True, return the figure and the axis.
    """
    assert scaling in ('none','log','power','hist')
    assert clipvals in ('minmax','manual','std','centered')
    if mask is not None:
        assert mask.shape == ar.shape
        assert is_color_like(mask_color)
        if isinstance(ar,np.ma.masked_array):
            ar = np.ma.array(data=ar.data,mask=mask==False)
        else:
            ar = np.ma.array(data=ar,mask=mask==False)
    elif isinstance(ar,np.ma.masked_array):
        pass
    else:
        mask = np.ones_like(ar,dtype=bool)
        ar = np.ma.array(data=ar,mask=mask==False)

    # Perform any scaling
    if scaling == 'none':
        _ar = ar.copy()
        _mask = np.ones_like(_ar.data,dtype=bool)
    elif scaling == 'log':
        _mask = ar>0
        _ar = np.zeros_like(ar.data,dtype=float)
        _ar[_mask] = np.log(ar[_mask])
        if min != None:
            if min > 0:
                min = np.log(min)
            else:
                min = np.min(_ar[_mask])
        if max != None:
            max = np.log(max)
    elif scaling == 'power':
        _mask = ar>0
        _ar = np.zeros_like(ar.data,dtype=float)
        _ar[_mask] = np.power(ar[_mask],power)
        if min != None:
            min = np.power(min,power)
        if max != None:
            max = np.power(max,power)
    else:
        raise Exception
    _ar = np.ma.array(data=_ar.data,mask=np.logical_and(ar.mask==False,_mask)==False)

    # Set the clipvalues
    if clipvals == 'minmax':
        vmin,vmax = np.ma.min(_ar),np.ma.max(_ar)
    elif clipvals == 'manual':
        assert min is not None and max is not None
        vmin,vmax = min,max
    elif clipvals == 'std':
        assert min is not None and max is not None
        m,s = np.ma.median(_ar),np.ma.std(_ar)
        vmin = m + min*s
        vmax = m + max*s
    elif clipvals == 'centered':
        c = np.ma.mean(_ar) if min is None else min
        m = np.ma.max(np.ma.abs(c-_ar)) if max is None else max
        vmin = c-m
        vmax = c+m
    else:
        raise Exception
    if returnclipvals:
        return vmin,vmax

    # Create or attach to the appropriate Figure and Axis
    if figax is None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax = figax
        assert(isinstance(fig,Figure))
        assert(isinstance(ax,Axes))

    # Plot the image
    if not hist:
        cax = ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)
        if np.any(_ar.mask==True):
            mask_display = np.ma.array(data=_ar.mask,mask=_ar.mask==False)
            cmap = ListedColormap([mask_color,'w'])
            ax.matshow(mask_display,cmap=cmap)
    # ...or, plot its histogram
    else:
        hist,bin_edges = np.histogram(_ar,bins=np.linspace(np.min(_ar),
                                                np.max(_ar),num=n_bins))
        w = bin_edges[1]-bin_edges[0]
        x = bin_edges[:-1]+w/2.
        ax.bar(x,hist,width=w)
        ax.vlines((vmin,vmax),0,ax.get_ylim()[1],color='k',ls='--')

    # Add a border
    if bordercolor is not None:
        for s in ['bottom','top','left','right']:
            ax.spines[s].set_color(bordercolor)
            ax.spines[s].set_linewidth(borderwidth)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add shape/point overlays
    if rectangle is not None:
        add_rectangles(ax,rectangle)
    if circle is not None:
        add_circles(ax,circle)
    if annulus is not None:
        add_annuli(ax,annulus)
    if ellipse is not None:
        add_ellipses(ax,ellipse)
    if points is not None:
        add_points(ax,points)
    if grid_overlay is not None:
        add_grid_overly(ax,grid)


    # Parse arguments for scale/coordinate overlays
    if coordinates is not None:
        assert isinstance(coordinates,Coordinates)
    assert space in ('Q','R')
    # pixel size/units
    if pixelsize is None and coordinates is None:
        pixelsize = 1
    if pixelsize is not None:
        pass
    else:
        if space == 'Q':
            pixelsize = coordinates.get_Q_pixel_size()
        else:
            pixelsize = coordinates.get_R_pixel_size()
    if pixelunits is None and coordinates is None:
        pixelunits = 'pixels'
    if pixelunits is not None:
        pass
    else:
        if space == 'Q':
            pixelunits = coordinates.get_Q_pixel_units()
        else:
            pixelunits = coordinates.get_R_pixel_units()
    # origin
    if space == 'Q':
        if x0 is not None:
            pass
        elif coordinates is not None:
            try:
                x0 = coordinates.get_qx0(rx,ry)
            except AttributeError:
                raise Exception('The Coordinates instance passed does not contain a value for qx0')
        else:
            x0 = 0
        if y0 is not None:
            pass
        elif coordinates is not None:
            try:
                y0 = coordinates.get_qy0(rx,ry)
            except AttributeError:
                raise Exception('The Coordinates instance passed does not contain a value for qy0')
        else:
            y0 = 0
    else:
        x0 = x0 if x0 is not None else 0
        y0 = y0 if y0 is not None else 0
    # ellipticity
    if space == 'Q':
        if e is not None:
            pass
        elif coordinates is not None:
            try:
                e = coordinates.get_e(rx,ry)
            except AttributeError:
                raise Exception('The Coordinates instance passed does not contain a value for e')
        else:
            e = 1
        if theta is not None:
            pass
        elif coordinates is not None:
            try:
                theta = coordinates.get_theta(rx,ry)
            except AttributeError:
                raise Exception('The Coordinates instance passed does not contain a value for theta')
        else:
            theta = 0
    else:
        e = e if e is not None else 1
        theta = theta if theta is not None else 0


    # Add a scalebar
    if scalebar is not None:
        # Add the grid
        scalebar['Nx'],scalebar['Ny']=ar.shape
        scalebar['pixelsize'] = pixelsize
        scalebar['pixelunits'] = pixelunits
        scalebar['space'] = space
        add_scalebar(ax,scalebar)


    # Add cartesian grid
    if cartesian_grid is not None:
        Nx,Ny = ar.shape
        assert isinstance(x0,Number), "Error: x0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(y0,Number), "Error: y0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        cartesian_grid['x0'],cartesian_grid['y0']=x0,y0
        cartesian_grid['Nx'],cartesian_grid['Ny']=Nx,Ny
        cartesian_grid['pixelsize'] = pixelsize
        cartesian_grid['pixelunits'] = pixelunits
        cartesian_grid['space'] = space
        add_cartesian_grid(ax,cartesian_grid)


    # Add polarelliptical grid
    if polarelliptical_grid is not None:
        Nx,Ny = ar.shape
        assert isinstance(x0,Number), "Error: x0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(y0,Number), "Error: y0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(e,Number), "Error: e must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(theta,Number), "Error: theta must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        polarelliptical_grid['x0'],polarelliptical_grid['y0']=x0,y0
        polarelliptical_grid['e'],polarelliptical_grid['theta']=e,theta
        polarelliptical_grid['Nx'],polarelliptical_grid['Ny']=Nx,Ny
        polarelliptical_grid['pixelsize'] = pixelsize
        polarelliptical_grid['pixelunits'] = pixelunits
        polarelliptical_grid['space'] = space
        add_polarelliptical_grid(ax,polarelliptical_grid)


    # Add r-theta grid
    if rtheta_grid is not None:
        add_rtheta_grid(ax,grid)


    # Show or return
    if returnfig:
        return fig,ax
    elif returncax:
        return cax
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

# Show functions with overlaid scalebars and/or coordinate system gridlines

def show_Q(ar,scalebar=True,grid=False,polargrid=False,
           Q_pixel_size=None,Q_pixel_units=None,
           coordinates=None,rx=None,ry=None,
           qx0=None,qy0=None,
           e=None,theta=None,
           scalebarloc=0,scalebarsize=None,scalebarwidth=None,
           scalebartext=None,scalebartextloc='above',scalebartextsize=12,
           gridspacing=None,gridcolor='w',
           majorgridlines=True,majorgridlw=1,majorgridls=':',
           minorgridlines=True,minorgridlw=0.5,minorgridls=':',
           gridlabels=False,gridlabelsize=12,gridlabelcolor='k',
           alpha=0.35,
           **kwargs):
    """
    Shows a diffraction space image with options for several overlays to define the scale,
    including a scalebar, a cartesian grid, or a polar / polar-elliptical grid.

    Regardless of which overlay is requested, the function must recieve either values
    for Q_pixel_size and Q_pixel_units, or a Coordinates instance containing these values.
    If both are passed, the manually passed values take precedence.
    If a cartesian grid is requested, (qx0,qy0) are required, either passed manually or
    passed as a Coordinates instance with the appropriate (rx,ry) value.
    If a polar grid is requested, (qx0,qy0,e,theta) are required, again either manually
    or via a Coordinates instance.

    Any arguments accepted by the show() function (e.g. image scaling, clipvalues, etc)
    may be passed to this function as kwargs.
    """
    # Check inputs
    assert(isinstance(ar,np.ndarray) and len(ar.shape)==2)
    if coordinates is not None:
        assert isinstance(coordinates,Coordinates)
    try:
        Q_pixel_size = Q_pixel_size if Q_pixel_size is not None else \
                       coordinates.get_Q_pixel_size()
    except AttributeError:
        raise Exception("Q_pixel_size must be specified, either in coordinates or manually")
    try:
        Q_pixel_units = Q_pixel_units if Q_pixel_units is not None else \
                       coordinates.get_Q_pixel_units()
    except AttributeError:
        raise Exception("Q_pixel_size must be specified, either in coordinates or manually")
    if grid or polargrid:
        try:
            qx0 = qx0 if qx0 is not None else coordinates.get_qx0(rx,ry)
        except AttributeError:
            raise Exception("qx0 must be specified, either in coordinates or manually")
        try:
            qy0 = qy0 if qy0 is not None else coordinates.get_qy0(rx,ry)
        except AttributeError:
            raise Exception("qy0 must be specified, either in coordinates or manually")
        assert isinstance(qx0,Number), "Error: qx0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(qy0,Number), "Error: qy0 must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
    if polargrid:
        e = e if e is not None else coordinates.get_e(rx,ry)
        theta = theta if theta is not None else coordinates.get_theta(rx,ry)
        assert isinstance(e,Number), "Error: e must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."
        assert isinstance(theta,Number), "Error: theta must be a number. If a Coordinate system was passed, try passing a position (rx,ry)."

    # Make the plot
    fig,ax = show(ar,returnfig=True,**kwargs)

    # Add a scalebar
    if scalebar:
        pass

    # Add a cartesian grid
    if grid:
        # parse arguments
        assert isinstance(majorgridlines,bool)
        majorgridlw = majorgridlw if majorgridlines else 0
        assert isinstance(majorgridlw,Number)
        assert isinstance(majorgridls,str)
        assert isinstance(minorgridlines,bool)
        minorgridlw = minorgridlw if minorgridlines else 0
        assert isinstance(minorgridlw,Number)
        assert isinstance(minorgridls,str)
        assert is_color_like(gridcolor)
        assert isinstance(gridlabels,bool)
        assert isinstance(gridlabelsize,Number)
        assert is_color_like(gridlabelcolor)
        if gridspacing is not None:
            assert isinstance(gridspacing,Number)

        Q_Nx,Q_Ny = ar.shape
        assert qx0<Q_Nx and qy0<Q_Ny

        # Get the major grid-square size
        if gridspacing is None:
            D = np.mean((Q_Nx*Q_pixel_size,Q_Ny*Q_pixel_size))/2.
            exp = int(log(D,10))
            if np.sign(log(D,10))<0:
                exp-=1
            base = D/(10**exp)
            if base>=1 and base<1.25:
                _gridspacing=0.4
            elif base>=1.25 and base<1.75:
                _gridspacing=0.5
            elif base>=1.75 and base<2.5:
                _gridspacing=0.75
            elif base>=2.5 and base<3.25:
                _gridspacing=1
            elif base>=3.25 and base<4.75:
                _gridspacing=1.5
            elif base>=4.75 and base<6:
                _gridspacing=2
            elif base>=6 and base<8:
                _gridspacing=2.5
            elif base>=8 and base<10:
                _gridspacing=3
            else:
                raise Exception("how did this happen?? base={}".format(base))
            gridspacing = _gridspacing * 10**exp

        # Get the positions and label for the major gridlines
        xmin = (-qx0)*Q_pixel_size
        xmax = (Q_Nx-1-qx0)*Q_pixel_size
        ymin = (-qy0)*Q_pixel_size
        ymax = (Q_Ny-1-qy0)*Q_pixel_size
        xticksmajor = np.concatenate((-1*np.arange(0,np.abs(xmin),gridspacing)[1:][::-1],
                                      np.arange(0,xmax,gridspacing)))
        yticksmajor = np.concatenate((-1*np.arange(0,np.abs(ymin),gridspacing)[1:][::-1],
                                      np.arange(0,ymax,gridspacing)))
        xticklabels = xticksmajor.copy()
        yticklabels = yticksmajor.copy()
        xticksmajor = (xticksmajor-xmin)/Q_pixel_size
        yticksmajor = (yticksmajor-ymin)/Q_pixel_size
        # Labels
        exp_spacing = int(np.round(log(gridspacing,10),6))
        if np.sign(log(gridspacing,10))<0:
            exp_spacing-=1
        base_spacing = gridspacing/(10**exp_spacing)
        xticklabels = xticklabels/(10**exp_spacing)
        yticklabels = yticklabels/(10**exp_spacing)
        if exp_spacing == 1:
            xticklabels *= 10
            yticklabels *= 10
        if _gridspacing in (0.4,0.75,1.5,2.5) and exp_spacing!=1:
            xticklabels = ["{:.1f}".format(n) for n in xticklabels]
            yticklabels = ["{:.1f}".format(n) for n in yticklabels]
        else:
            xticklabels = ["{:.0f}".format(n) for n in xticklabels]
            yticklabels = ["{:.0f}".format(n) for n in yticklabels]

        # Add the grid
        ax.set_xticks(yticksmajor)
        ax.set_yticks(xticksmajor)
        ax.xaxis.set_ticks_position('bottom')
        if gridlabels:
            ax.set_xticklabels(yticklabels,size=gridlabelsize,color=gridlabelcolor)
            ax.set_yticklabels(xticklabels,size=gridlabelsize,color=gridlabelcolor)
            if exp_spacing in (0,1):
                ax.set_xlabel(r"$q_y$ ("+Q_pixel_units+")")
                ax.set_ylabel(r"$q_x$ ("+Q_pixel_units+")")
            else:
                ax.set_xlabel(r"$q_y$ ("+Q_pixel_units+" e"+str(exp_spacing)+")")
                ax.set_ylabel(r"$q_x$ ("+Q_pixel_units+" e"+str(exp_spacing)+")")
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.grid(linestyle=majorgridls,linewidth=majorgridlw,color=gridcolor,alpha=alpha)



        # Add the grid
        if majorgridlines:
            add_cartesian_grid(ax,d={
                            'x0':qx0,'y0':qy0,
                            'spacing':gridspacing,
                            'majorlw':majorgridlw,
                            'majorls':majorgridls,
                            'minorlw':minorgridlw,
                            'minorls':minorgridls,
                            'color':gridcolor,
                            'label':gridlabels,
                            'labelsize':gridlabelsize,
                            'labelcolor':gridlabelcolor,
                            'alpha':alpha})
        if minorgridlines:
            add_cartesian_grid(ax,d={
                            'x0':qx0,'y0':qy0,
                            'spacing':gridspacing,
                            'majorlw':majorgridlw,
                            'majorls':majorgridls,
                            'minorlw':minorgridlw,
                            'minorls':minorgridls,
                            'color':gridcolor,
                            'label':gridlabels,
                            'labelsize':gridlabelsize,
                            'labelcolor':gridlabelcolor,
                            'alpha':alpha})


    # Add a polar-elliptical grid
    if polargrid:
        pass

    return


# Shape overlays

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

def show_points(ar,x,y,s=1,scale=50,alpha=1,pointcolor='r',open_circles=False,
                returnfig=False,**kwargs):
    """
    Plots a 2D array with one or more points.
    x and y are the point centers and must have the same length, N.
    s is the relative point sizes, and must have length 1 or N.
    scale is the size of the largest point.
    pointcolor have length 1 or N.

    Accepts:
        ar          (array) the image
        x,y         (number or iterable of numbers) the point positions
        s           (number or iterable of numbers) the relative point sizes
        scale       (number) the maximum point size
        pointcolor
        alpha

    Returns:
        If returnfig==False (default), the figure is plotted and nothing is returned.
        If returnfig==False, the figure and its one axis are returned, and can be
        further edited.
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'x':x,'y':y,'s':s,'scale':scale,'pointcolor':pointcolor,'alpha':alpha,
         'open_circles':open_circles}
    add_points(ax,d)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax



