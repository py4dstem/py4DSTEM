import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from . import show,show_circ
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

def show_voronoi(ar,x,y,color_points='r',color_lines='w',max_dist=None,**kwargs):
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
    plt.show()
    return


