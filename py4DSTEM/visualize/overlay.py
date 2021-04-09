import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge,Ellipse
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from numbers import Number
from math import log
from fractions import Fraction

def add_rectangles(ax,d):
    """
    Adds one or more rectangles to Axis ax using the parameters in dictionary d.
    """
    # Handle inputs
    assert isinstance(ax,Axes)
    # lims
    assert('lims' in d.keys())
    lims = d['lims']
    if isinstance(lims,tuple):
        assert(len(lims)==4)
        lims = [lims]
    assert(isinstance(lims,list))
    N = len(lims)
    assert(all([isinstance(t,tuple) for t in lims]))
    assert(all([len(t)==4 for t in lims]))
    # color
    color = d['color'] if 'color' in d.keys() else 'r'
    if isinstance(color,list):
        assert(len(color)==N)
        assert(all([is_color_like(c) for c in color]))
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
    # fill
    fill = d['fill'] if 'fill' in d.keys() else False
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(a,(float,int,np.float)) for a in alpha]))
    # linewidth
    linewidth = d['linewidth'] if 'linewidth' in d.keys() else 2
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('lims','color','fill','alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the rectangles
    for i in range(N):
        l,c,f,a,lw = lims[i],color[i],fill[i],alpha[i],linewidth[i]
        rect = Rectangle((l[2]-0.5,l[0]-0.5),l[3]-l[2],l[1]-l[0],color=c,fill=f,
                          alpha=a,linewidth=lw,**kwargs)
        ax.add_patch(rect)

    return

def add_circles(ax,d):
    """
    adds one or more circles to axis ax using the parameters in dictionary d.
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # center
    assert('center' in d.keys())
    center = d['center']
    if isinstance(center,tuple):
        assert(len(center)==2)
        center = [center]
    assert(isinstance(center,list))
    N = len(center)
    assert(all([isinstance(x,tuple) for x in center]))
    assert(all([len(x)==2 for x in center]))
    # radius
    assert('R' in d.keys())
    R = d['R']
    if isinstance(R,Number):
        R = [R for i in range(N)]
    assert(isinstance(R,list))
    assert(len(R)==N)
    assert(all([isinstance(i,Number) for i in R]))
    # color
    color = d['color'] if 'color' in d.keys() else 'r'
    if isinstance(color,list):
        assert(len(color)==N)
        assert(all([is_color_like(c) for c in color]))
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
    # fill
    fill = d['fill'] if 'fill' in d.keys() else False
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(a,(float,int,np.float)) for a in alpha]))
    # linewidth
    linewidth = d['linewidth'] if 'linewidth' in d.keys() else 2
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('center','R','color','fill','alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the circles
    for i in range(N):
        cent,r,col,f,a,lw = center[i],R[i],color[i],fill[i],alpha[i],linewidth[i]
        circ = Circle((cent[1],cent[0]),r,color=col,fill=f,alpha=a,linewidth=lw,**kwargs)
        ax.add_patch(circ)

    return

def add_annuli(ax,d):
    """
    Adds one or more annuli to Axis ax using the parameters in dictionary d.
    """
    # Handle inputs
    assert isinstance(ax,Axes)
    # center
    assert('center' in d.keys())
    center = d['center']
    if isinstance(center,tuple):
        assert(len(center)==2)
        center = [center]
    assert(isinstance(center,list))
    N = len(center)
    assert(all([isinstance(x,tuple) for x in center]))
    assert(all([len(x)==2 for x in center]))
    # radii
    assert('Ri' in d.keys())
    assert('Ro' in d.keys())
    Ri,Ro = d['Ri'],d['Ro']
    if isinstance(Ri,Number):
        Ri = [Ri for i in range(N)]
    if isinstance(Ro,Number):
        Ro = [Ro for i in range(N)]
    assert(isinstance(Ri,list))
    assert(isinstance(Ro,list))
    assert(len(Ri)==N)
    assert(len(Ro)==N)
    assert(all([isinstance(i,Number) for i in Ri]))
    assert(all([isinstance(i,Number) for i in Ro]))
    # color
    color = d['color'] if 'color' in d.keys() else 'r'
    if isinstance(color,list):
        assert(len(color)==N)
        assert(all([is_color_like(c) for c in color]))
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
    # fill
    fill = d['fill'] if 'fill' in d.keys() else False
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(a,(float,int,np.float)) for a in alpha]))
    # linewidth
    linewidth = d['linewidth'] if 'linewidth' in d.keys() else 2
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('center','Ri','Ro','color','fill','alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the annuli
    for i in range(N):
        cent,ri,ro,col,f,a,lw = center[i],Ri[i],Ro[i],color[i],fill[i],alpha[i],linewidth[i]
        annulus = Wedge((cent[1],cent[0]),ro,0,360,width=ro-ri,color=col,fill=f,alpha=a,
                        linewidth=lw,**kwargs)
        ax.add_patch(annulus)

    return

def add_ellipses(ax,d):
    """
    Adds one or more ellipses to axis ax using the parameters in dictionary d.

    Parameters:
        center
        r
        e
        theta
        color
        fill
        alpha
        linewidth
        linestyle
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # semimajor axis length
    assert('r' in d.keys())
    r = d['r']
    if isinstance(r,Number):
        r = [r]
    assert(isinstance(r,list))
    N = len(r)
    assert(all([isinstance(i,Number) for i in r]))
    # center
    assert('center' in d.keys())
    center = d['center']
    if isinstance(center,tuple):
        assert(len(center)==2)
        center = [center for i in range(N)]
    assert(isinstance(center,list))
    assert(len(center)==N)
    assert(all([isinstance(x,tuple) for x in center]))
    assert(all([len(x)==2 for x in center]))
    # ratio of axis lengths
    assert('e' in d.keys())
    e = d['e']
    if isinstance(e,Number):
        e = [e for i in range(N)]
    assert(isinstance(e,list))
    assert(len(e)==N)
    assert(all([isinstance(i,Number) for i in e]))
    # theta
    assert('theta' in d.keys())
    theta = d['theta']
    if isinstance(theta,Number):
        theta = [theta for i in range(N)]
    assert(isinstance(theta,list))
    assert(len(theta)==N)
    assert(all([isinstance(i,Number) for i in theta]))
    # color
    color = d['color'] if 'color' in d.keys() else 'r'
    if isinstance(color,list):
        assert(len(color)==N)
        assert(all([is_color_like(c) for c in color]))
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
    # fill
    fill = d['fill'] if 'fill' in d.keys() else False
    if isinstance(fill,bool):
        fill = [fill for i in range(N)]
    else:
        assert(isinstance(fill,list))
        assert(len(fill)==N)
        assert(all([isinstance(f,bool) for f in fill]))
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1
    if isinstance(alpha,(float,int,np.float)):
        alpha = [alpha for i in range(N)]
    else:
        assert(isinstance(alpha,list))
        assert(len(alpha)==N)
        assert(all([isinstance(alp,(float,int,np.float)) for alp in alpha]))
    # linewidth
    linewidth = d['linewidth'] if 'linewidth' in d.keys() else 2
    if isinstance(linewidth,(float,int,np.float)):
        linewidth = [linewidth for i in range(N)]
    else:
        assert(isinstance(linewidth,list))
        assert(len(linewidth)==N)
        assert(all([isinstance(lw,(float,int,np.float)) for lw in linewidth]))
    # linestyle
    linestyle = d['linestyle'] if 'linestyle' in d.keys() else '-'
    if isinstance(linestyle,(str)):
        linestyle = [linestyle for i in range(N)]
    else:
        assert(isinstance(linestyle,list))
        assert(len(linestyle)==N)
        assert(all([isinstance(lw,(str)) for lw in linestyle]))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('center','r','e','theta','color',
                                            'fill','alpha','linewidth','linestyle')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the ellipses
    for i in range(N):
        cent,_r,_e,_theta,col,f,_alpha,lw,ls = (center[i],r[i],e[i],theta[i],color[i],fill[i],
                                                alpha[i],linewidth[i],linestyle[i])
        ellipse = Ellipse((cent[1],cent[0]),2*_r*_e,2*_r,90-np.degrees(_theta),color=col,fill=f,
                        alpha=_alpha,linewidth=lw,linestyle=ls,**kwargs)
        ax.add_patch(ellipse)

    return

def add_points(ax,d):
    """
    adds one or more points to axis ax using the parameters in dictionary d.
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # x
    assert('x' in d.keys())
    x = d['x']
    if isinstance(x,Number):
        x = [x]
    x = np.array(x)
    N = len(x)
    # y
    assert('y' in d.keys())
    y = d['y']
    if isinstance(y,Number):
        y = [y]
    y = np.array(y)
    assert(len(y)==N)
    # s
    s = d['s'] if 's' in d.keys() else np.ones(N)
    if isinstance(s,Number):
        s = np.ones_like(x)*s
    assert(len(s)==N)
    s = np.where(s>0,s,0)
    # scale
    scale = d['scale'] if 'scale' in d.keys() else 25
    assert isinstance(scale,Number)
    # point color
    color = d['pointcolor'] if 'pointcolor' in d.keys() else 'r'
    if isinstance(color,(list,np.ndarray)):
        assert(len(color)==N)
        assert(all([is_color_like(c) for c in color]))
    else:
        assert is_color_like(color)
        color = [color for i in range(N)]
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1.
    assert isinstance(alpha,Number)
    # additional parameters
    kws = [k for k in d.keys() if k not in ('x','y','s','scale','pointcolor','alpha')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the points
    ax.scatter(y,x,s=s*scale/np.max(s),color=color,alpha=alpha)

    return


def add_grid_overlay(ax,d):
    """
    adds an overlaid grid over some subset of pixels in an image
    using the parameters in dictionary d.

    The dictionary d has required and optional parameters as follows:
        x0,y0       (req'd) (ints) the corner of the grid
        xL,xL       (req'd) (ints) the extent of the grid
        color       (color)
        linewidth   (number)
        alpha       (number)
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # corner, extent
    lims = [0,0,0,0]
    for i,k in enumerate(('x0','y0','xL','yL')):
        assert(k in d.keys()), "Error: add_grid_overlay expects keys 'x0','y0','xL','yL'"
        lims[i] = d[k]
    x0,y0,xL,yL = lims
    # color
    color = d['color'] if 'color' in d.keys() else 'k'
    assert is_color_like(color)
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 1
    assert isinstance(alpha,(Number))
    # linewidth
    linewidth = d['linewidth'] if 'linewidth' in d.keys() else 1
    assert isinstance(linewidth,(Number))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('x0','y0','xL','yL','color',
                                            'alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the grid
    yy,xx = np.meshgrid(np.arange(y0,y0+yL),np.arange(x0,x0+xL))
    for xi in range(xL):
        for yi in range(yL):
            x,y = xx[xi,yi],yy[xi,yi]
            rect = Rectangle((y-0.5,x-0.5),1,1,lw=linewidth,color=color,
                             alpha=alpha,fill=False)
            ax.add_patch(rect)
    return


def add_cartesian_grid(ax,d):
    """
    adds an overlaid cartesian coordinate grid over an image
    using the parameters in dictionary d.

    The dictionary d has required and optional parameters as follows:
        x0,y0           (req'd) the origin
        Nx,Ny           (req'd) the image extent
        space           (str) 'Q' or 'R'
        spacing         (number) spacing between gridlines
        pixelsize       (number)
        pixelunits      (str)
        lw              (number)
        ls              (str)
        color           (color)
        label           (bool)
        labelsize       (number)
        labelcolor      (color)
        alpha           (number)
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # origin
    assert('x0' in d.keys())
    assert('y0' in d.keys())
    x0,y0 = d['x0'],d['y0']
    # image extent
    assert('Nx' in d.keys())
    assert('Ny' in d.keys())
    Nx,Ny = d['Nx'],d['Ny']
    assert x0<Nx and y0<Ny
    # real or diffraction
    assert('space' in d.keys())
    space = d['space']
    assert(space in ('Q','R'))
    # spacing, pixelsize, pixelunits
    spacing = d['spacing'] if 'spacing' in d.keys() else None
    pixelsize = d['pixelsize'] if 'pixelsize' in d.keys() else 1
    pixelunits = d['pixelunits'] if 'pixelunits' in d.keys() else 'pixels'
    # gridlines
    lw = d['lw'] if 'lw' in d.keys() else 1
    ls = d['ls'] if 'ls' in d.keys() else ':'
    # color
    color = d['color'] if 'color' in d.keys() else 'w'
    assert is_color_like(color)
    # labels
    label = d['label'] if 'label' in d.keys() else False
    labelsize = d['labelsize'] if 'labelsize' in d.keys() else 12
    labelcolor = d['labelcolor'] if 'labelcolor' in d.keys() else 'k'
    assert isinstance(label,bool)
    assert isinstance(labelsize,Number)
    assert is_color_like(labelcolor)
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 0.35
    assert isinstance(alpha,(Number))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('x0','y0','spacing','lw','ls',
                        'color','label','labelsize','labelcolor','alpha')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # Get the major grid-square size
    if spacing is None:
        D = np.mean((Nx*pixelsize,Ny*pixelsize))/2.
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
    else:
        gridspacing = spacing
        _gridspacing = 1.5

    # Get positions for the major gridlines
    xmin = (-x0)*pixelsize
    xmax = (Nx-1-x0)*pixelsize
    ymin = (-y0)*pixelsize
    ymax = (Ny-1-y0)*pixelsize
    xticksmajor = np.concatenate((-1*np.arange(0,np.abs(xmin),gridspacing)[1:][::-1],
                                  np.arange(0,xmax,gridspacing)))
    yticksmajor = np.concatenate((-1*np.arange(0,np.abs(ymin),gridspacing)[1:][::-1],
                                  np.arange(0,ymax,gridspacing)))
    xticklabels = xticksmajor.copy()
    yticklabels = yticksmajor.copy()
    xticksmajor = (xticksmajor-xmin)/pixelsize
    yticksmajor = (yticksmajor-ymin)/pixelsize

    # Get labels
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
    if label:
        ax.set_xticklabels(yticklabels,size=labelsize,color=labelcolor)
        ax.set_yticklabels(xticklabels,size=labelsize,color=labelcolor)
        axislabel_x = r"$q_x$" if space=='Q' else r"$x$"
        axislabel_y = r"$q_y$" if space=='Q' else r"$y$"
        if exp_spacing in (0,1):
            ax.set_xlabel(axislabel_y+" ("+pixelunits+")",size=labelsize,color=labelcolor)
            ax.set_ylabel(axislabel_x+" ("+pixelunits+")",size=labelsize,color=labelcolor)
        else:
            ax.set_xlabel(axislabel_y+" ("+pixelunits+" e"+str(exp_spacing)+")",
                          size=labelsize,color=labelcolor)
            ax.set_ylabel(axislabel_x+" ("+pixelunits+" e"+str(exp_spacing)+")",
                          size=labelsize,color=labelcolor)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(linestyle=ls,linewidth=lw,color=color,alpha=alpha)

    return


def add_polarelliptical_grid(ax,d):
    """
    adds an overlaid polar-ellitpical coordinate grid over an image
    using the parameters in dictionary d.

    The dictionary d has required and optional parameters as follows:
        x0,y0           (req'd) the origin
        e,theta         (req'd) the ellipticity (a/b) and major axis angle (radians)
        Nx,Ny           (req'd) the image extent
        space           (str) 'Q' or 'R'
        spacing         (number) spacing between radial gridlines
        N_thetalines    (int) the number of theta gridlines
        pixelsize       (number)
        pixelunits      (str)
        lw              (number)
        ls              (str)
        color           (color)
        label           (bool)
        labelsize       (number)
        labelcolor      (color)
        alpha           (number)
    """
    # handle inputs
    assert isinstance(ax,Axes)
    # origin
    assert('x0' in d.keys())
    assert('y0' in d.keys())
    x0,y0 = d['x0'],d['y0']
    # ellipticity
    assert('e' in d.keys())
    assert('theta' in d.keys())
    e,theta = d['e'],d['theta']
    # image extent
    assert('Nx' in d.keys())
    assert('Ny' in d.keys())
    Nx,Ny = d['Nx'],d['Ny']
    assert x0<Nx and y0<Ny
    # real or diffraction
    assert('space' in d.keys())
    space = d['space']
    assert(space in ('Q','R'))
    # spacing, N_thetalines, pixelsize, pixelunits
    spacing = d['spacing'] if 'spacing' in d.keys() else None
    N_thetalines = d['N_thetalines'] if 'N_thetalines' in d.keys() else 8
    assert(N_thetalines%2==0), "N_thetalines must be even"
    N_thetalines = N_thetalines//2
    assert isinstance(N_thetalines,(int,np.integer))
    pixelsize = d['pixelsize'] if 'pixelsize' in d.keys() else 1
    pixelunits = d['pixelunits'] if 'pixelunits' in d.keys() else 'pixels'
    # gridlines
    lw = d['lw'] if 'lw' in d.keys() else 1
    ls = d['ls'] if 'ls' in d.keys() else ':'
    # color
    color = d['color'] if 'color' in d.keys() else 'w'
    assert is_color_like(color)
    # labels
    label = d['label'] if 'label' in d.keys() else False
    labelsize = d['labelsize'] if 'labelsize' in d.keys() else 8
    labelcolor = d['labelcolor'] if 'labelcolor' in d.keys() else color
    assert isinstance(label,bool)
    assert isinstance(labelsize,Number)
    assert is_color_like(labelcolor)
    # alpha
    alpha = d['alpha'] if 'alpha' in d.keys() else 0.5
    assert isinstance(alpha,(Number))
    # additional parameters
    kws = [k for k in d.keys() if k not in ('x0','y0','spacing','lw','ls',
                        'color','label','labelsize','labelcolor','alpha')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # Get the radial spacing
    if spacing is None:
        D = np.mean((Nx*pixelsize,Ny*pixelsize))/2.
        exp = int(log(D,10))
        if np.sign(log(D,10))<0:
            exp-=1
        base = D/(10**exp)
        if base>=1 and base<1.25:
            _spacing=0.4
        elif base>=1.25 and base<1.75:
            _spacing=0.5
        elif base>=1.75 and base<2.5:
            _spacing=0.75
        elif base>=2.5 and base<3.25:
            _spacing=1
        elif base>=3.25 and base<4.75:
            _spacing=1.5
        elif base>=4.75 and base<6:
            _spacing=2
        elif base>=6 and base<8:
            _spacing=2.5
        elif base>=8 and base<10:
            _spacing=3
        else:
            raise Exception("how did this happen?? base={}".format(base))
        spacing = _spacing * 10**exp
        spacing = spacing/2.
    else:
        _spacing = 1.5

    # Get positions for the radial gridlines
    xmin = (-x0)*pixelsize
    xmax = (Nx-1-x0)*pixelsize
    ymin = (-y0)*pixelsize
    ymax = (Ny-1-y0)*pixelsize
    rcorners = (np.hypot(xmin,ymin),
               np.hypot(xmin,ymax),
               np.hypot(xmax,ymin),
               np.hypot(xmax,ymax))
    rticks = np.arange(0,np.max(rcorners),spacing)[1:]
    rticklabels = rticks.copy()
    rticks = rticks/pixelsize

    # Add radial gridlines
    N = len(rticks)
    d_ellipses = {'r':list(rticks),
                  'center':(x0,y0),
                  'e':e,
                  'theta':theta,
                  'fill':False,
                  'color':color,
                  'linewidth':lw,
                  'linestyle':ls,
                  'alpha':alpha}
    add_ellipses(ax,d_ellipses)

    # Add radial gridline labels
    if label:
        # Get gridline label positions
        rlabelpos_scale = 1+ (e-1)*np.sin(np.pi/2-theta)**4
        rlabelpositions = x0 - rticks*rlabelpos_scale
        for i in range(len(rticklabels)):
            xpos = rlabelpositions[i]
            if xpos>labelsize/2:
                ax.text(y0,rlabelpositions[i],rticklabels[i],size=labelsize,
                color=labelcolor,alpha=alpha,ha='center',va='center')

    # Add theta gridlines
    def add_line(ax,x0,y0,theta,Nx,Ny):
        """ adds a line through (x0,y0) at an angle theta which terminates at the image edges
            returns the termination points (xi,yi),(xf,xy)
        """
        theta = np.mod(np.pi/2-theta,np.pi)
        if theta==0:
            xs,ys = [0,Nx-1],[y0,y0]
        elif theta==np.pi/2:
            xs,ys = [x0,x0],[0,Ny-1]
        else:
            # Get line params
            m = np.tan(theta)
            b = y0-m*x0
            # Get intersections with x=0,x=Nx-1,y=0,y=Ny-1
            x1,y1 = 0,b
            x2,y2 = Nx-1,m*(Nx-1)+b
            x3,y3 = -b/m,0
            x4,y4 = (Ny-1 -b)/m,Ny-1
            # Determine which points are on the image bounding box
            xs,ys = [],[]
            if 0<=y1<Ny-1: xs.append(x1),ys.append(y1)
            if 0<=y2<Ny-1: xs.append(x2),ys.append(y2)
            if 0<=x3<Nx-1: xs.append(x3),ys.append(y3)
            if 0<=x4<Nx-1: xs.append(x4),ys.append(y4)
            assert len(xs)==len(ys)==2

        ax.plot(xs,ys,color=color,ls=ls,alpha=alpha,lw=lw)
        return tuple([(xs[i],ys[i]) for i in range(2)])

    thetalabelpos = []
    for t in theta+np.linspace(0,np.pi,N_thetalines,endpoint=False):
        thetalabelpos.append(add_line(ax,x0,y0,t,Nx,Ny))
    thetalabelpos = [thetalabelpos[i][0] for i in range(len(thetalabelpos))] + \
                    [thetalabelpos[i][1] for i in range(len(thetalabelpos))]
    # Get angles for the theta gridlines
    thetaticklabels = [str(Fraction(i,N_thetalines))+r'$\pi$' for i in range(2*N_thetalines)]
    thetaticklabels[0] = '0'
    thetaticklabels[N_thetalines] = r'$\pi$'

    # Add theta gridline labels
    if label:
        for i in range(len(thetaticklabels)):
            x,y = thetalabelpos[i]
            if x==0: ha,va='left','center'
            elif x==Nx-1: ha,va='right','center'
            elif y==0: ha,va='center','top'
            else: ha,va='center','bottom'
            ax.text(x,y,thetaticklabels[i],size=labelsize,color=labelcolor,
                    alpha=alpha,ha=ha,va=va)
        pass

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0,Nx-1])
    ax.set_ylim([0,Ny-1])
    ax.invert_yaxis()
    return






def add_rtheta_grid(ar,d):
    return

def add_scalebar(ar,d):
    return





