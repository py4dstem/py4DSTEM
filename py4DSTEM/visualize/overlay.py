import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Wedge,Ellipse
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from numbers import Number

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
    # radius
    assert('R' in d.keys())
    R = d['R']
    if isinstance(R,Number):
        R = [R for i in range(N)]
    assert(isinstance(R,list))
    assert(len(R)==N)
    assert(all([isinstance(i,Number) for i in R]))
    # a
    assert('a' in d.keys())
    a = d['a']
    if isinstance(a,Number):
        a = [a for i in range(N)]
    assert(isinstance(a,list))
    assert(len(a)==N)
    assert(all([isinstance(i,Number) for i in a]))
    # b
    assert('b' in d.keys())
    b = d['b']
    if isinstance(b,Number):
        b = [b for i in range(N)]
    assert(isinstance(b,list))
    assert(len(b)==N)
    assert(all([isinstance(i,Number) for i in b]))
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
    kws = [k for k in d.keys() if k not in ('center','R','a','b','theta','color',
                                            'fill','alpha','linewidth')]
    kwargs = dict()
    for k in kws:
        kwargs[k] = d[k]

    # add the ellipses
    for i in range(N):
        cent,r,_a,_b,_theta,col,f,_alpha,lw = (center[i],R[i],a[i],b[i],theta[i],color[i],fill[i],
                                               alpha[i],linewidth[i])
        ellipse = Ellipse((cent[1],cent[0]),2*_a*r,2*_b*r,90-np.degrees(_theta),color=col,fill=f,
                        alpha=_alpha,linewidth=lw,**kwargs)
        ax.add_patch(ellipse)

    return


    #if grid is not None:
    #    add_grid(ax,grid)



