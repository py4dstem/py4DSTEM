import numpy as np
import matplotlib.pyplot as plt
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
    rxx,ryy = np.meshgrid(np.arange(ry0,ry0+yL),np.arange(rx0,rx0+xL))

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

def show_grid(get_im,get_im_args,returnfig=False,**kwargs):
    """
    Words

    Accepts:

    Returns:
        if returnfig==false (default), the figure is plotted and nothing is returned.
        if returnfig==false, the figure and its one axis are returned, and can be
        further edited.
    """

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax


