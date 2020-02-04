import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np


def plot(img, title='Image', savePath=None, cmap='inferno', show=True, vmax=None, figsize=(10, 10), scale=None):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(title)
    fontprops = fm.FontProperties(size=18)
    if scale is not None:
        scalebar = AnchoredSizeBar(ax.transData,
                                   scale[0], scale[1], 'lower right',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=img.shape[0] / 40,
                                   fontproperties=fontprops)

        ax.add_artist(scalebar)
    ax.grid(False)
    if savePath is not None:
        fig.savefig(savePath + '.png', dpi=600)
        fig.savefig(savePath + '.eps', dpi=600)
    if show:
        plt.show()


def plot_bragg_disks(peaks, datacube, Rx, Ry, image=None, scale=200, power=0.25, filter_function=None):
    from cycler import cycler
    gridsize = int(np.ceil(np.sqrt(len(Rx) + 1)))
    fig, ax = plt.subplots(int(np.ceil((len(Rx) + 1) / gridsize)), gridsize, figsize=(12, 12))

    ax[0, 0].matshow(image if image is not None else np.zeros((datacube.R_Nx, datacube.R_Ny)))

    colors = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                    '#bcbd22', '#17becf'])

    _ = filter_function or (lambda x: x)

    for i, clr in zip(range(len(Rx)), colors):
        c = clr['color']
        ax[0, 0].scatter(Ry[i], Rx[i], color=c)

        ax.ravel()[i + 1].matshow(_(datacube.data[Rx[i], Ry[i]])**power)

        if scale == 0:
            ax.ravel()[i + 1].scatter(peaks[i].data['qy'], peaks[i].data['qx'], color=c)
        else:
            ax.ravel()[i + 1].scatter(peaks[i].data['qy'], peaks[i].data['qx'], color=c,
                s=scale * peaks[i].data['intensity'] / np.max(peaks[i].data['intensity']))

    for a in ax.ravel():
        a.axis('off')

    plt.show()
