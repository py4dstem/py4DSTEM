# Defines utility functions used by other functions in the /process/ directory.

#import numpy as np
#from numpy.fft import fftfreq, fftshift
#from scipy.ndimage import gaussian_filter
#import math as ma
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#import matplotlib.font_manager as fm
#
#from emdfile import tqdmnd
#from py4DSTEM.process.utils.multicorr import upsampled_correlation
#from py4DSTEM.preprocess.utils import make_Fourier_coords2D
#
#try:
#    from IPython.display import clear_output
#except ImportError:
#
#    def clear_output(wait=True):
#        pass
#
#
#try:
#    import cupy as cp
#except (ModuleNotFoundError, ImportError):
#    cp = np




# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import matplotlib.font_manager as fm
#
#
# try:
#    from IPython.display import clear_output
# except ImportError:
#    def clear_output(wait=True):
#        pass
#
# def plot(img, title='Image', savePath=None, cmap='inferno', show=True, vmax=None,
#                                                        figsize=(10, 10), scale=None):
#    fig, ax = plt.subplots(figsize=figsize)
#    im = ax.imshow(img, interpolation='nearest', cmap=plt.get_cmap(cmap), vmax=vmax)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    ax.set_title(title)
#    fontprops = fm.FontProperties(size=18)
#    if scale is not None:
#        scalebar = AnchoredSizeBar(ax.transData,
#                                   scale[0], scale[1], 'lower right',
#                                   pad=0.1,
#                                   color='white',
#                                   frameon=False,
#                                   size_vertical=img.shape[0] / 40,
#                                   fontproperties=fontprops)
#
#        ax.add_artist(scalebar)
#    ax.grid(False)
#    if savePath is not None:
#        fig.savefig(savePath + '.png', dpi=600)
#        fig.savefig(savePath + '.eps', dpi=600)
#    if show:
#        plt.show()



#def plot(
#    img,
#    title="Image",
#    savePath=None,
#    cmap="inferno",
#    show=True,
#    vmax=None,
#    figsize=(10, 10),
#    scale=None,
#):
#    fig, ax = plt.subplots(figsize=figsize)
#    im = ax.imshow(img, interpolation="nearest", cmap=plt.get_cmap(cmap), vmax=vmax)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    ax.set_title(title)
#    fontprops = fm.FontProperties(size=18)
#    if scale is not None:
#        scalebar = AnchoredSizeBar(
#            ax.transData,
#            scale[0],
#            scale[1],
#            "lower right",
#            pad=0.1,
#            color="white",
#            frameon=False,
#            size_vertical=img.shape[0] / 40,
#            fontproperties=fontprops,
#        )
#
#        ax.add_artist(scalebar)
#    ax.grid(False)
#    if savePath is not None:
#        fig.savefig(savePath + ".png", dpi=600)
#        fig.savefig(savePath + ".eps", dpi=600)
#    if show:
#        plt.show()


