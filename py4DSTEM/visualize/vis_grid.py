import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .visualize import show

def show_DP_grid(datacube,rx0,ry0,xL,yL,axsize=(6,6),returnfig=False,**kwargs):
    """
    Words

    Accepts:

    Returns:
        if returnfig==false (default), the figure is plotted and nothing is returned.
        if returnfig==false, the figure and its one axis are returned, and can be
        further edited.
    """
    ryy,rxx = np.meshgrid(np.arange(ry0,ry0+yL),np.arange(rx0,rx0+xL))

    fig,axs = plt.subplots(xL,yL,figsize=(yL*axsize[0],xL*axsize[1]))
    for rxi in range(xL):
        for ryi in range(yL):
            ax = axs[rxi,ryi]
            rx,ry = rxx[rxi,ryi],ryy[rxi,ryi]
            dp = datacube.data[rx,ry,:,:]
            _,_ = show(dp,ax=(fig,ax),returnfig=True,**kwargs)
    plt.tight_layout()

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_grid_overlay(ar,rx0,ry0,xL,yL,color='k',linewidth=1,alpha=1,
                                            returnfig=False,**kwargs):
    """
    Words
    """
    ryy,rxx = np.meshgrid(np.arange(ry0,ry0+yL),np.arange(rx0,rx0+xL))

    fig,ax = show(ar,returnfig=True,**kwargs)
    for rxi in range(xL):
        for ryi in range(yL):
            rx,ry = rxx[rxi,ryi],ryy[rxi,ryi]
            rect = Rectangle((ry-0.5,rx-0.5),1,1,lw=linewidth,color=color,
                             alpha=alpha,fill=False)
            ax.add_patch(rect)
    plt.tight_layout()

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_grid(get_ar,H,W,axsize=(6,6),returnfig=False,titlesize=0,
                                      get_bc=None,**kwargs):
    """
    Words

    Accepts:

    Returns:
        if returnfig==false (default), the figure is plotted and nothing is returned.
        if returnfig==false, the figure and its one axis are returned, and can be
        further edited.
    """
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
                if get_bc is not None:
                    bc = get_bc(N)
                    _,_ = show(ar,ax=(fig,ax),returnfig=True,
                               bordercolor=bc,**kwargs)
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


