import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from . import show,show_circ,show_grid
from ..process.utils import get_voronoi_vertices

def show_kernel(kernel,R,L,W,figsize=(12,6)):
    """
    Plots, side by side, the probe kernel and its line profile.
    R is the kernel plot's window size.
    L and W are the lenth ad width of the lineprofile.
    """
    lineprofile = np.concatenate([np.sum(kernel[-L:,:W],axis=(1)),
                                  np.sum(kernel[:L,:W],axis=(1))])

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    ax1.matshow(kernel[:int(R),:int(R)],cmap='gray')
    ax2.plot(np.arange(len(lineprofile)),lineprofile)
    plt.show()
    return

def show_voronoi(ar,x,y,color_points='r',color_lines='w',max_dist=None,
                                                returnfig=False,**kwargs):
    """
    words
    """
    Nx,Ny = ar.shape
    points = np.vstack((x,y)).T
    voronoi = Voronoi(points)
    vertices = get_voronoi_vertices(voronoi,Nx,Ny)

    if max_dist is None:
        fig,ax = show(ar,returnfig=True,**kwargs)
    else:
        centers = [(x[i],y[i]) for i in range(len(x))]
        fig,ax = show_circ(ar,returnfig=True,**kwargs,
                           center=centers,R=max_dist,fill=False,color=color_points)

    ax.scatter(voronoi.points[:,1],voronoi.points[:,0],color=color_points)
    for region in range(len(vertices)):
        vertices_curr = vertices[region]
        for i in range(len(vertices_curr)):
            x0,y0 = vertices_curr[i,:]
            xf,yf = vertices_curr[(i+1)%len(vertices_curr),:]
            ax.plot((y0,yf),(x0,xf),color=color_lines)
    ax.set_xlim([0,Ny])
    ax.set_ylim([0,Nx])
    plt.gca().invert_yaxis()
    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_class_BPs(ar,x,y,s,s2,color='r',color2='y',**kwargs):
    """
    words
    """
    N = len(x)
    assert(N==len(y)==len(s))

    fig,ax = show(ar,returnfig=True,**kwargs)
    ax.scatter(y,x,s=s2,color=color2)
    ax.scatter(y,x,s=s,color=color)
    plt.show()
    return

def show_class_BPs_grid(ar,H,W,x,y,get_s,s2,color='r',color2='y',returnfig=False,
                        axsize=(6,6),titlesize=0,get_bc=None,**kwargs):
    """
    words
    """
    fig,axs = show_grid(lambda i:ar,H,W,axsize=axsize,titlesize=titlesize,
                       get_bc=get_bc,returnfig=True,**kwargs)
    for i in range(H):
        for j in range(W):
            ax = axs[i,j]
            N = i*W+j
            s = get_s(N)
            ax.scatter(y,x,s=s2,color=color2)
            ax.scatter(y,x,s=s,color=color)
    plt.gca().invert_yaxis()
    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax



