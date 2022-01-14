import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi
from . import show
from .overlay import add_pointlabels,add_vector,add_bragg_index_labels,add_ellipses
from .vis_grid import show_image_grid
from .vis_RQ import ax_addaxes,ax_addaxes_QtoR
from ..io import PointList
from ..process.utils import get_voronoi_vertices,convert_ellipse_params
from ..process.calibration import double_sided_gaussian
from ..process.latticevectors import get_selected_lattice_vectors

def show_elliptical_fit(ar,fitradii,p_ellipse,fill=True,
                        color_ann='y',color_ell='r',alpha_ann=0.2,alpha_ell=0.7,
                        linewidth_ann=2,linewidth_ell=2,returnfig=False,**kwargs):
    """
    Plots an elliptical curve over its annular fit region.

    Args:
        center (2-tuple): the center
        fitradii (2-tuple of numbers): the annulus inner and outer fit radii
        p_ellipse (5-tuple): the parameters of the fit ellipse, (qx0,qy0,a,b,theta).
            See the module docstring for utils.elliptical_coords for more details.
        fill (bool): if True, fills in the annular fitting region,
          else shows only inner/outer edges
        color_ann (color): annulus color
        color_ell (color): ellipse color
        alpha_ann: transparency for the annulus
        alpha_ell: transparency forn the fit ellipse
        linewidth_ann:
        linewidth_ell:
    """
    Ri,Ro = fitradii
    qx0,qy0,a,b,theta = p_ellipse
    fig,ax = show(ar,
                  annulus={'center':(qx0,qy0),'Ri':Ri,'Ro':Ro,'fill':fill,
                           'color':color_ann,'alpha':alpha_ann,'linewidth':linewidth_ann},
                  ellipse={'center':(qx0,qy0),'a':a,'b':b,'theta':theta,
                           'color':color_ell,'alpha':alpha_ell,'linewidth':linewidth_ell},
                  returnfig=True,**kwargs)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax


def show_amorphous_ring_fit(dp,fitradii,p_dsg,N=12,cmap=('gray','gray'),
                            fitborder=True,fitbordercolor='k',fitborderlw=0.5,
                            scaling='log',ellipse=False,ellipse_color='r',
                            ellipse_alpha=0.7,ellipse_lw=2,returnfig=False,**kwargs):
    """
    Display a diffraction pattern with a fit to its amorphous ring, interleaving
    the data and the fit in a pinwheel pattern.

    Args:
        dp (array): the diffraction pattern
        fitradii (2-tuple of numbers): the min/max distances of the fitting annulus
        p_dsg (11-tuple): the fit parameters to the double-sided gaussian
            function returned by fit_ellipse_amorphous_ring
        N (int): the number of pinwheel sections
        cmap (colormap or 2-tuple of colormaps): if passed a single cmap, uses this
            colormap for both the data and the fit; if passed a 2-tuple of cmaps, uses
            the first for the data and the second for the fit
        fitborder (bool): if True, plots a border line around the fit data
        fitbordercolor (color): color of the fitborder
        fitborderlw (number): linewidth of the fitborder
        scaling (str): the normal scaling param -- see docstring for visualize.show
        ellipse (bool): if True, overlay an ellipse
        returnfig (bool): if True, returns the figure
    """
    assert(len(p_dsg)==11)
    assert(isinstance(N,(int,np.integer)))
    if isinstance(cmap,tuple):
        cmap_data,cmap_fit = cmap[0],cmap[1]
    else:
        cmap_data,cmap_fit = cmap,cmap
    Q_Nx,Q_Ny = dp.shape
    qmin,qmax = fitradii

    # Make coords
    qx0,qy0 = p_dsg[6],p_dsg[7]
    qyy,qxx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    qx,qy = qxx-qx0,qyy-qy0
    q = np.hypot(qx,qy)
    theta = np.arctan2(qy,qx)

    # Make mask
    thetas = np.linspace(-np.pi,np.pi,2*N+1)
    pinwheel = np.zeros((Q_Nx,Q_Ny),dtype=bool)
    for i in range(N):
        pinwheel += (theta>thetas[2*i]) * (theta<=thetas[2*i+1])
    mask = pinwheel * (q>qmin) * (q<=qmax)

    # Get fit data
    fit = double_sided_gaussian(p_dsg, qxx, qyy)

    # Show
    (fig,ax),(vmin,vmax) = show(dp,scaling=scaling,cmap=cmap_data,
                  mask=np.logical_not(mask),mask_color='empty',
                  returnfig=True,returnclipvals=True,**kwargs)
    show(fit,scaling=scaling,figax=(fig,ax),clipvals='manual',min=vmin,max=vmax,
         cmap=cmap_fit,mask=mask,mask_color='empty',**kwargs)
    if fitborder:
        if N%2==1: thetas += (thetas[1]-thetas[0])/2
        if (N//2%2)==0: thetas = np.roll(thetas,-1)
        for i in range(N):
            ax.add_patch(Wedge((qy0,qx0),qmax,np.degrees(thetas[2*i]),
                         np.degrees(thetas[2*i+1]),width=qmax-qmin,fill=None,
                         color=fitbordercolor,lw=fitborderlw))

    # Add ellipse overlay
    if ellipse:
        A,B,C = p_dsg[8],p_dsg[9],p_dsg[10]
        a,b,theta = convert_ellipse_params(A,B,C)
        ellipse={'center':(qx0,qy0),'a':a,'b':b,'theta':theta,
                 'color':ellipse_color,'alpha':ellipse_alpha,'linewidth':ellipse_lw}
        add_ellipses(ax,ellipse)

    if not returnfig:
        plt.show()
        return
    else:
        return fig,ax


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
        fig,ax = show(ar,returnfig=True,**kwargs,
                      circle={'center':centers,'R':max_dist,'fill':False,'color':color_points})

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
                        axsize=(6,6),titlesize=0,get_bordercolor=None,**kwargs):
    """
    words
    """
    fig,axs = show_image_grid(lambda i:ar,H,W,axsize=axsize,titlesize=titlesize,
                       get_bordercolor=get_bordercolor,returnfig=True,**kwargs)
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

def show_strain(strainmap,
                vrange_exx,
                vrange_theta,
                vrange_exy=None,
                vrange_eyy=None,
                bkgrd=True,
                show_cbars=('exx','eyy','exy','theta'),
                bordercolor='k',
                borderwidth=1,
                titlesize=24,
                ticklabelsize=16,
                ticknumber=5,
                unitlabelsize=24,
                show_axes=True,
                axes_x0=0,
                axes_y0=0,
                xaxis_x=1,
                xaxis_y=0,
                axes_length=10,
                axes_width=1,
                axes_color='r',
                xaxis_space='Q',
                labelaxes=True,
                QR_rotation=0,
                axes_labelsize=12,
                axes_labelcolor='r',
                axes_plots=('exx'),
                cmap='RdBu_r',
                layout=0,
                figsize=(12,12),
                returnfig=False):
    """
    Display a strain map, showing the 4 strain components (e_xx,e_yy,e_xy,theta), and
    masking each image with strainmap.slices['mask'].

    Args:
        strainmap (RealSlice):
        vrange_exx (length 2 list or tuple):
        vrange_theta (length 2 list or tuple):
        vrange_exy (length 2 list or tuple):
        vrange_eyy (length 2 list or tuple):
        bkgrd (bool):
        show_cbars (tuple of strings): Show colorbars for the specified axes. Must be a
            tuple containing any, all, or none of ('exx','eyy','exy','theta').
        bordercolor (color):
        borderwidth (number):
        titlesize (number):
        ticklabelsize (number):
        ticknumber (number): number of ticks on colorbars
        unitlabelsize (number):
        show_axes (bool):
        axes_x0 (number):
        axes_y0 (number):
        xaxis_x (number):
        xaxis_y (number):
        axes_length (number):
        axes_width (number):
        axes_color (color):
        xaxis_space (string): must be 'Q' or 'R'
        labelaxes (bool):
        QR_rotation (number):
        axes_labelsize (number):
        axes_labelcolor (color):
        axes_plots (tuple of strings): controls if coordinate axes showing the
            orientation of the strain matrices are overlaid over any of the plots.
            Must be a tuple of strings containing any, all, or none of
            ('exx','eyy','exy','theta').
        cmap (colormap):
        layout=0 (int): determines the layout of the grid which the strain components
            will be plotted in.  Must be in (0,1,2).  0=(2x2), 1=(1x4), 2=(4x1).
        figsize (length 2 tuple of numbers):
        returnfig (bool):
    """
    # Lookup table for different layouts
    assert(layout in (0,1,2))
    layout_lookup = {
        0:['left','right','left','right'],
        1:['bottom','bottom','bottom','bottom'],
        2:['right','right','right','right'],
    }
    layout_p = layout_lookup[layout]

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
    if layout==0:
        fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,figsize=figsize)
    elif layout==1:
        fig,(ax11,ax12,ax21,ax22) = plt.subplots(1,4,figsize=figsize)
    else:
        fig,(ax11,ax12,ax21,ax22) = plt.subplots(4,1,figsize=figsize)
    cax11 = show(e_xx,figax=(fig,ax11),min=vmin_exx,max=vmax_exx,clipvals='manual',
                 cmap=cmap,returncax=True)
    cax12 = show(e_yy,figax=(fig,ax12),min=vmin_eyy,max=vmax_eyy,clipvals='manual',
                 cmap=cmap,returncax=True)
    cax21 = show(e_xy,figax=(fig,ax21),min=vmin_exy,max=vmax_exy,clipvals='manual',
                 cmap=cmap,returncax=True)
    cax22 = show(theta,figax=(fig,ax22),min=vmin_theta,max=vmax_theta,clipvals='manual',
                 cmap=cmap,returncax=True)
    ax11.set_title(r'$\epsilon_{xx}$',size=titlesize)
    ax12.set_title(r'$\epsilon_{yy}$',size=titlesize)
    ax21.set_title(r'$\epsilon_{xy}$',size=titlesize)
    ax22.set_title(r'$\theta$',size=titlesize)

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
        cbax11 = divider11.append_axes(layout_p[0],size="4%",pad=0.15)
        cbax12 = divider12.append_axes(layout_p[1],size="4%",pad=0.15)
        cbax21 = divider21.append_axes(layout_p[2],size="4%",pad=0.15)
        cbax22 = divider22.append_axes(layout_p[3],size="4%",pad=0.15)
        for (show_cbar,cax,cbax,vmin,vmax,tickside,tickunits) in zip(
                                    show_cbars,
                                    (cax11,cax12,cax21,cax22),
                                    (cbax11,cbax12,cbax21,cbax22),
                                    (vmin_exx,vmin_eyy,vmin_exy,vmin_theta),
                                    (vmax_exx,vmax_eyy,vmax_exy,vmax_theta),
                                    (layout_p[0],layout_p[1],layout_p[2],layout_p[3]),
                                    ('% ',' %','% ',r' $^\circ$')):
            if show_cbar:
                ticks = np.linspace(vmin,vmax,ticknumber,endpoint=True)
                ticklabels = (np.round(np.linspace(100*vmin,100*vmax,ticknumber,endpoint=True)*100)/100).astype(str)
                if tickside in ('left','right'):
                    cb = plt.colorbar(cax,cax=cbax,ticks=ticks,orientation='vertical')
                    cb.ax.set_yticklabels(ticklabels,size=ticklabelsize)
                    cbax.yaxis.set_ticks_position(tickside)
                    cbax.set_ylabel(tickunits,size=unitlabelsize,rotation=0)
                    cbax.yaxis.set_label_position(tickside)
                else:
                    cb = plt.colorbar(cax,cax=cbax,ticks=ticks,orientation='horizontal')
                    cb.ax.set_xticklabels(ticklabels,size=ticklabelsize)
                    cbax.xaxis.set_ticks_position(tickside)
                    cbax.set_xlabel(tickunits,size=unitlabelsize,rotation=0)
                    cbax.xaxis.set_label_position(tickside)
            else:
                cbax.axis('off')

    # Add coordinate axes
    if show_axes:
        assert(xaxis_space in ('R','Q')), "xaxis_space must be 'R' or 'Q'"
        show_which_axes = np.array(['exx' in axes_plots,'eyy' in axes_plots,
                                    'exy' in axes_plots,'theta' in axes_plots])
        for _show,_ax in zip(show_which_axes,(ax11,ax12,ax21,ax22)):
            if _show:
                if xaxis_space=='R':
                    ax_addaxes(_ax,xaxis_x,xaxis_y,axes_length,axes_x0,axes_y0,
                               width=axes_width,color=axes_color,labelaxes=labelaxes,
                               labelsize=axes_labelsize,labelcolor=axes_labelcolor)
                else:
                    ax_addaxes_QtoR(_ax,xaxis_x,xaxis_y,axes_length,axes_x0,axes_y0,QR_rotation,
                                    width=axes_width,color=axes_color,labelaxes=labelaxes,
                                    labelsize=axes_labelsize,labelcolor=axes_labelcolor)

    # Add borders
    if bordercolor is not None:
        for ax in (ax11,ax12,ax21,ax22):
            for s in ['bottom','top','left','right']:
                ax.spines[s].set_color(bordercolor)
                ax.spines[s].set_linewidth(borderwidth)
            ax.set_xticks([])
            ax.set_yticks([])

    if not returnfig:
        plt.show()
        return
    else:
        axs = ((ax11,ax12),(ax21,ax22))
        return fig,axs


def show_pointlabels(ar,x,y,color='lightblue',size=20,alpha=1,returnfig=False,**kwargs):
    """
    Show enumerated index labels for a set of points
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'x':x,'y':y,'size':size,'color':color,'alpha':alpha}
    add_pointlabels(ax,d)

    if returnfig:
        return fig,ax
    else:
        plt.show()
        return


def select_point(ar,x,y,i,color='lightblue',color_selected='r',size=20,returnfig=False,**kwargs):
    """
    Show enumerated index labels for a set of points, with one selected point highlighted
    """
    fig,ax = show(ar,returnfig=True,**kwargs)
    d1 = {'x':x,'y':y,'size':size,'color':color}
    d2 = {'x':x[i],'y':y[i],'size':size,'color':color_selected,'fontweight':'bold'}
    add_pointlabels(ax,d1)
    add_pointlabels(ax,d2)

    if returnfig:
        return fig,ax
    else:
        plt.show()
        return


def select_lattice_vectors(ar,gx,gy,i0,i1,i2,
               c_indices='lightblue',c0='g',c1='r',c2='r',c_vectors='r',c_vectorlabels='w',
               size_indices=20,width_vectors=1,size_vectorlabels=20,
               figsize=(12,6),returnfig=False,**kwargs):
    """
    This function accepts a set of reciprocal lattice points (gx,gy) and three indices
    (i0,i1,i2).  Using those indices as, respectively, the origin, the endpoint of g1, and
    the endpoint of g2, this function computes the basis lattice vectors g1,g2, visualizes
    them, and returns them.  To compute these vectors without visualizing, use
    latticevectors.get_selected_lattice_vectors().

    Returns:
        if returnfig==False:    g1,g2
        if returnfig==True      g1,g2,fig,ax
    """
    # Make the figure
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    show(ar,figax=(fig,ax1),**kwargs)
    show(ar,figax=(fig,ax2),**kwargs)

    # Add indices to left panel
    d = {'x':gx,'y':gy,'size':size_indices,'color':c_indices}
    d0 = {'x':gx[i0],'y':gy[i0],'size':size_indices,'color':c0,'fontweight':'bold','labels':[str(i0)]}
    d1 = {'x':gx[i1],'y':gy[i1],'size':size_indices,'color':c1,'fontweight':'bold','labels':[str(i1)]}
    d2 = {'x':gx[i2],'y':gy[i2],'size':size_indices,'color':c2,'fontweight':'bold','labels':[str(i2)]}
    add_pointlabels(ax1,d)
    add_pointlabels(ax1,d0)
    add_pointlabels(ax1,d1)
    add_pointlabels(ax1,d2)

    # Compute vectors
    g1,g2 = get_selected_lattice_vectors(gx,gy,i0,i1,i2)

    # Add vectors to right panel
    dg1 = {'x0':gx[i0],'y0':gy[i0],'vx':g1[0],'vy':g1[1],'width':width_vectors,
           'color':c_vectors,'label':r'$g_1$','labelsize':size_vectorlabels,'labelcolor':c_vectorlabels}
    dg2 = {'x0':gx[i0],'y0':gy[i0],'vx':g2[0],'vy':g2[1],'width':width_vectors,
           'color':c_vectors,'label':r'$g_2$','labelsize':size_vectorlabels,'labelcolor':c_vectorlabels}
    add_vector(ax2,dg1)
    add_vector(ax2,dg2)

    if returnfig:
        return g1,g2,fig,(ax1,ax2)
    else:
        plt.show()
        return g1,g2


def show_lattice_vectors(ar,x0,y0,g1,g2,color='r',width=1,labelsize=20,labelcolor='w',returnfig=False,**kwargs):
    """ Adds the vectors g1,g2 to an image, with tail positions at (x0,y0).  g1 and g2 are 2-tuples (gx,gy).
    """
    fig,ax = show(ar,returnfig=True,**kwargs)

    # Add vectors
    dg1 = {'x0':x0,'y0':y0,'vx':g1[0],'vy':g1[1],'width':width,
           'color':color,'label':r'$g_1$','labelsize':labelsize,'labelcolor':labelcolor}
    dg2 = {'x0':x0,'y0':y0,'vx':g2[0],'vy':g2[1],'width':width,
           'color':color,'label':r'$g_2$','labelsize':labelsize,'labelcolor':labelcolor}
    add_vector(ax,dg1)
    add_vector(ax,dg2)

    if returnfig:
        return fig,ax
    else:
        plt.show()
        return


def show_bragg_indexing(ar,braggdirections,voffset=5,hoffset=0,color='w',size=20,
                        points=True,pointcolor='r',pointsize=50,returnfig=False,**kwargs):
    """
    Shows an array with an overlay describing the Bragg directions

    Accepts:
        ar                  (arrray) the image
        bragg_directions    (PointList) the bragg scattering directions; must have coordinates
                            'qx','qy','h', and 'k'. Optionally may also have 'l'.
    """
    assert isinstance(braggdirections,PointList)
    for k in ('qx','qy','h','k'):
        assert k in braggdirections.data.dtype.fields

    fig,ax = show(ar,returnfig=True,**kwargs)
    d = {'braggdirections':braggdirections,'voffset':voffset,'hoffset':hoffset,'color':color,
         'size':size,'points':points,'pointsize':pointsize,'pointcolor':pointcolor}
    add_bragg_index_labels(ax,d)

    if returnfig:
        return fig,ax
    else:
        plt.show()
        return


def show_max_peak_spacing(ar,spacing,braggdirections,color='g',lw=2,returnfig=False,**kwargs):
    """ Show a circle of radius `spacing` about each Bragg direction
    """
    centers = [(braggdirections.data['qx'][i],braggdirections.data['qy'][i]) for i in range(braggdirections.length)]
    fig,ax = show(ar,circle={'center':centers,'R':spacing,'color':color,'fill':False,'lw':lw},
                  returnfig=True,**kwargs)
    if returnfig:
        return fig,ax
    else:
        plt.show()
        return
