import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi
from . import show,show_circ,show_image_grid,ax_show,ax_addaxes,ax_addaxes_QtoR
from ..process.utils import get_voronoi_vertices

def show_qprofile(q,intensity,ymax,figsize=(12,4),returnfig=False,
                  color='k',xlabel='q (pixels)',ylabel='Intensity (A.U.)',
                  labelsize=16,ticklabelsize=14,grid='on',
                  **kwargs):
    """
    Plots a diffraction space radial profile.
    Params:
        q               (1D array) the diffraction coordinate / x-axis
        intensity       (1D array) the y-axis values
        ymax            (number) max value for the yaxis
        color           (matplotlib color) profile color
        xlabel          (str)
        ylabel
        labelsize       size of x and y labels
        ticklabelsize
        grid            'off' or 'on'
    """
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(q,intensity,color=color)
    ax.grid(grid)
    ax.set_ylim(0,ymax)
    ax.tick_params(axis='x',labelsize=ticklabelsize)
    ax.set_yticklabels([])
    ax.set_xlabel(xlabel,size=labelsize)
    ax.set_ylabel(ylabel,size=labelsize)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax

def show_kernel(kernel,R,L,W,figsize=(12,6),returnfig=False,**kwargs):
    """
    Plots, side by side, the probe kernel and its line profile.
    R is the kernel plot's window size.
    L and W are the lenth ad width of the lineprofile.
    """
    lineprofile = np.concatenate([np.sum(kernel[-L:,:W],axis=(1)),
                                  np.sum(kernel[:L,:W],axis=(1))])

    fig,axs = plt.subplots(1,2,figsize=figsize)
    axs[0].matshow(kernel[:int(R),:int(R)],cmap='gray')
    axs[1].plot(np.arange(len(lineprofile)),lineprofile)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,axs

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
    fig,axs = show_image_grid(lambda i:ar,H,W,axsize=axsize,titlesize=titlesize,
                       get_bc=get_bc,returnfig=True,**kwargs)
    for i in range(H):
        for j in range(W):
            ax = axs[i,j]
            N = i*W+j
            s = get_s(N)
            ax.scatter(y,x,s=s2,color=color2)
            ax.scatter(y,x,s=s,color=color)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,axs

def show_strain(strainmap,vrange_exx,vrange_theta,vrange_exy=None,vrange_eyy=None,
                bkgrd=True,show_cbars=('exx','eyy','exy','theta'),
                titlesize=24,ticklabelsize=16,unitlabelsize=24,
                show_axes=True,axes_x0=0,axes_y0=0,xaxis_x=1,xaxis_y=0,axes_length=10,
                axes_width=1,axes_color='r',xaxis_space='Q',labelaxes=True,QR_rotation=0,
                axes_labelsize=12,axes_labelcolor='r',axes_plots=('exx'),cmap='RdBu_r',
                figsize=(12,12),returnfig=False):
    """
    Words
    """
    # Contrast limits
    if vrange_exy is None:
        vrange_exy = vrange_exx
    if vrange_eyy is None:
        vrange_eyy = vrange_exx
    for vrange in (vrange_exx,vrange_eyy,vrange_exy,vrange_theta):
        assert(len(vrange)==2), 'vranges must have length 2'
    vmin_exx,vmax_exx = vrange_exx[0]/100.,vrange_exx[1]/100.
    vmin_eyy,vmax_eyy = vrange_eyy[0]/100.,vrange_eyy[1]/100.
    vmin_exy,vmax_exy = vrange_exy[0]/100.,vrange_exy[1]/100.
    vmin_theta,vmax_theta = vrange_theta[0]/100.,vrange_theta[1]/100.

    # Get images
    e_xx = np.ma.array(strainmap.slices['e_xx'],mask=strainmap.slices['mask']==False)
    e_yy = np.ma.array(strainmap.slices['e_yy'],mask=strainmap.slices['mask']==False)
    e_xy = np.ma.array(strainmap.slices['e_xy'],mask=strainmap.slices['mask']==False)
    theta = np.ma.array(strainmap.slices['theta'],mask=strainmap.slices['mask']==False)

    # Plot
    fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=figsize)
    cax11 = ax_show(e_xx,ax11,min=vmin_exx,max=vmax_exx,contrast='vrange',cmap=cmap,returncax=True)
    cax12 = ax_show(e_yy,ax12,min=vmin_eyy,max=vmax_eyy,contrast='vrange',cmap=cmap,returncax=True)
    cax21 = ax_show(e_xy,ax21,min=vmin_exy,max=vmax_exy,contrast='vrange',cmap=cmap,returncax=True)
    cax22 = ax_show(theta,ax22,min=vmin_theta,max=vmax_theta,contrast='vrange',cmap=cmap,returncax=True)
    ax11.set_title(r'$\epsilon_{xx}$',size=titlesize)
    ax12.set_title(r'$\epsilon_{yy}$',size=titlesize)
    ax21.set_title(r'$\epsilon_{xy}$',size=titlesize)
    ax22.set_title(r'$\theta$',size=titlesize)
    ax11.axis('off')
    ax12.axis('off')
    ax21.axis('off')
    ax22.axis('off')

    # Add black background
    if bkgrd:
        mask = np.ma.masked_where(strainmap.slices['mask'].astype(bool),
                                  np.zeros_like(strainmap.slices['mask']))
        ax11.matshow(mask,cmap='gray')
        ax12.matshow(mask,cmap='gray')
        ax21.matshow(mask,cmap='gray')
        ax22.matshow(mask,cmap='gray')

    # Colorbars
    show_cbars = np.array(['exx' in show_cbars,'eyy' in show_cbars,
                           'exy' in show_cbars,'theta' in show_cbars])
    if np.any(show_cbars):
        divider11 = make_axes_locatable(ax11)
        divider12 = make_axes_locatable(ax12)
        divider21 = make_axes_locatable(ax21)
        divider22 = make_axes_locatable(ax22)
        cbax11 = divider11.append_axes("left",size="4%",pad=0.15)
        cbax12 = divider12.append_axes("right",size="4%",pad=0.15)
        cbax21 = divider21.append_axes("left",size="4%",pad=0.15)
        cbax22 = divider22.append_axes("right",size="4%",pad=0.15)
        for (show_cbar,cax,cbax,vmin,vmax,tickside,tickunits) in zip(
                                    show_cbars,
                                    (cax11,cax12,cax21,cax22),
                                    (cbax11,cbax12,cbax21,cbax22),
                                    (vmin_exx,vmin_eyy,vmin_exy,vmin_theta),
                                    (vmax_exx,vmax_eyy,vmax_exy,vmax_theta),
                                    ('left','right','left','right'),
                                    ('% ',' %','% ',r' $^\circ$')):
            if show_cbar:
                cb = plt.colorbar(cax,cax=cbax,ticks=np.linspace(vmin,vmax,5,endpoint=True))
                cb.ax.set_yticklabels(['{}'.format(val) for val in np.linspace(100*vmin,100*vmax,5,endpoint=True)],size=ticklabelsize)
                cbax.yaxis.set_ticks_position(tickside)
                cbax.set_ylabel(tickunits,size=unitlabelsize,rotation=0)
                cbax.yaxis.set_label_position(tickside)
            else:
                cbax.axis('off')

    # Add coordinate axes
    if show_axes:
        assert(xaxis_space in ('R','Q')), "xaxis_space must be 'R' or 'Q'"
        show_which_axes = np.array(['exx' in axes_plots,'eyy' in axes_plots,
                                    'exy' in axes_plots,'theta' in axes_plots])
        for show,ax in zip(show_which_axes,(ax11,ax12,ax21,ax22)):
            if show:
                if xaxis_space=='R':
                    ax_addaxes(ax,xaxis_x,xaxis_y,axes_length,axes_x0,axes_y0,
                               width=axes_width,color=axes_color,labelaxes=labelaxes,
                               labelsize=axes_labelsize,labelcolor=axes_labelcolor)
                else:
                    ax_addaxes_QtoR(ax,xaxis_x,xaxis_y,axes_length,axes_x0,axes_y0,QR_rotation,
                                    width=axes_width,color=axes_color,labelaxes=labelaxes,
                                    labelsize=axes_labelsize,labelcolor=axes_labelcolor)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,axs


