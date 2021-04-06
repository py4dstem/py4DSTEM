import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge,Ellipse
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from numbers import Number
from math import log

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
    # semimajor axis length
    assert('a' in d.keys())
    a = d['a']
    if isinstance(a,Number):
        a = [a for i in range(N)]
    assert(isinstance(a,list))
    assert(len(a)==N)
    assert(all([isinstance(i,Number) for i in a]))
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
    # additional parameters
    kws = [k for k in d.keys() if k not in ('center','a','e','theta','color',
                                            'fill','alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the ellipses
    for i in range(N):
        cent,_a,_e,_theta,col,f,_alpha,lw = (center[i],a[i],e[i],theta[i],color[i],fill[i],
                                               alpha[i],linewidth[i])
        ellipse = Ellipse((cent[1],cent[0]),2*_a*_e,2*_a,90-np.degrees(_theta),color=col,fill=f,
                        alpha=_alpha,linewidth=lw,**kwargs)
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
    assert('s' in d.keys())
    s = d['s']
    if isinstance(s,Number):
        s = np.ones_like(x)*s
    assert(len(s)==N)
    s = np.where(s>0,s,0)
    # scale
    assert('scale' in d.keys())
    scale = d['scale']
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
        if exp_spacing in (0,1):
            ax.set_xlabel(r"$q_y$ ("+pixelunits+")")
            ax.set_ylabel(r"$q_x$ ("+pixelunits+")")
        else:
            ax.set_xlabel(r"$q_y$ ("+pixelunits+" e"+str(exp_spacing)+")")
            ax.set_ylabel(r"$q_x$ ("+pixelunits+" e"+str(exp_spacing)+")")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(linestyle=ls,linewidth=lw,color=color,alpha=alpha)

    return













def add_polarelliptical_grid(ar,d):
    return

def add_rtheta_grid(ar,d):
    return

def add_scalebar(ar,d):
    return





