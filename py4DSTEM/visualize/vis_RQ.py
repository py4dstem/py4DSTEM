import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .show import show,show_points
from ..process.calibration.rotation import get_Qvector_from_Rvector,get_Rvector_from_Qvector

def show_selected_dp(datacube,image,rx,ry,figsize=(12,6),returnfig=False,
                     pointsize=50,pointcolor='r',scaling='log',**kwargs):
    """
    """
    dp = datacube.data[rx,ry,:,:]
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    _,_=show_points(image,rx,ry,scale=pointsize,pointcolor=pointcolor,figax=(fig,ax1),returnfig=True)
    _,_=show(dp,figax=(fig,ax2),scaling=scaling,returnfig=True)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,(ax1,ax2)

def show_RQ(realspace_image, diffractionspace_image,
            realspace_pdict={}, diffractionspace_pdict={'scaling':'log'},
            figsize=(12,6),returnfig=False):
    """
    Shows side-by-side real/reciprocal space images.

    Accepts:
        realspace_image             (2D array)
        diffractionspace_image      (2D array)
        realspace_pdict             (dictionary) arguments and values to pass
                                    to the show() fn for the real space image
        diffractionspace_pdict      (dictionary)
    """
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
    show(realspace_image,figax=(fig,ax1),**realspace_pdict)
    show(diffractionspace_image,figax=(fig,ax2),**diffractionspace_pdict)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,(ax1,ax2)

def ax_addvector(ax,vx,vy,vlength,x0,y0,width=1,color='r'):
    """
    Adds a vector to the subplot at ax.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the vector
                                    Only the orientation is used, vector is
                                    normalized and rescaled by
        vlength                     (number) the vector length
        x0,y0                       (numbers) the origin / vector tail position
    """
    vL = np.hypot(vx,vy)
    vx,vy = vlength*vx/vL,vlength*vy/vL
    ax.arrow(y0,x0,vy,vx,color=color,width=width,length_includes_head=True)

def ax_addvector_RtoQ(ax,vx,vy,vlength,x0,y0,QR_rotation,width=1,color='r'):
    """
    Adds a vector to the subplot at ax, where the vector (vx,vy) passed
    to the function is in real space and the plotted vector is transformed
    into and plotted in reciprocal space.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the vector,
                                    in *real* space. Only the orientation is used,
                                    vector is normalized and rescaled by
        vlength                     (number) the vector length, in *reciprocal*
                                    space
        x0,y0                       (numbers) the origin / vector tail position,
                                    in *reciprocal* space
        QR_rotation               (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
    """
    _,_,vx,vy = get_Qvector_from_Rvector(vx,vy,QR_rotation)
    vx,vy = vx*vlength,vy*vlength
    ax.arrow(y0,x0,vy,vx,color=color,width=width,length_includes_head=True)

def ax_addvector_QtoR(ax,vx,vy,vlength,x0,y0,QR_rotation,width=1,color='r'):
    """
    Adds a vector to the subplot at ax, where the vector (vx,vy) passed
    to the function is in reciprocal space and the plotted vector is
    transformed into and plotted in real space.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the vector,
                                    in *reciprocal* space. Only the orientation is
                                    used, vector is normalized and rescaled by
        vlength                     (number) the vector length, in *real* space
        x0,y0                       (numbers) the origin / vector tail position,
                                    in *real* space
        QR_rotation                 (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
    """
    vx,vy,_,_ = get_Rvector_from_Qvector(vx,vy,QR_rotation)
    vx,vy = vx*vlength,vy*vlength
    ax.arrow(y0,x0,vy,vx,color=color,width=width,length_includes_head=True)

def show_RQ_vector(realspace_image, diffractionspace_image,
                   realspace_pdict, diffractionspace_pdict,
                   vx,vy,vlength_R,vlength_Q,x0_R,y0_R,x0_Q,y0_Q,
                   QR_rotation,vector_space='R',
                   width_R=1,color_R='r',
                   width_Q=1,color_Q='r',
                   figsize=(12,6),returnfig=False):
    """
    Shows side-by-side real/reciprocal space images with a vector
    overlaid in each showing corresponding directions.

    Accepts:
        realspace_image             (2D array)
        diffractionspace_image      (2D array)
        realspace_pdict             (dictionary) arguments and values to pass
                                    to the show() fn for the real space image
        diffractionspace_pdict      (dictionary)
        vx,vy                       (numbers) x,y components of the vector
                                    in either real or diffraction space,
                                    depending on the value of vector_space.
                                    Note (vx,vy) is used for the orientation
                                    only - the two vectors are normalized
                                    and rescaled by
        vlength_R,vlength_Q         (number) the vector length in each
                                    space, in pixels
        x0_R,y0_R,x0_Q,y0_Q         (numbers) the origins / vector tail positions
        QR_rotation               (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
        vector_space                (string) must be 'R' or 'Q'. Specifies
                                    whether the (vx,vy) values passed to this
                                    function describes a real or diffracation
                                    space vector.
    """
    assert(vector_space in ('R','Q'))
    fig,(ax1,ax2) = show_RQ(realspace_image, diffractionspace_image,
                            realspace_pdict, diffractionspace_pdict,
                            figsize=figsize,returnfig=True)
    if vector_space=='R':
        ax_addvector(ax1,vx,vy,vlength_R,x0_R,y0_R,width=width_R,color=color_R)
        ax_addvector_RtoQ(ax2,vx,vy,vlength_Q,x0_Q,y0_Q,QR_rotation,width=width_Q,color=color_Q)
    else:
        ax_addvector(ax2,vx,vy,vlength_Q,x0_Q,y0_Q,width=width_Q,color=color_Q)
        ax_addvector_QtoR(ax1,vx,vy,vlength_R,x0_R,y0_R,QR_rotation,width=width_R,color=color_R)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,(ax1,ax2)

def show_RQ_vectors(realspace_image, diffractionspace_image,
                    realspace_pdict, diffractionspace_pdict,
                    vx,vy,vlength_R,vlength_Q,x0_R,y0_R,x0_Q,y0_Q,
                    QR_rotation,vector_space='R',
                    width_R=1,color_R='r',
                    width_Q=1,color_Q='r',
                    figsize=(12,6),returnfig=False):
    """
    Shows side-by-side real/reciprocal space images with several vectors
    overlaid in each showing corresponding directions.

    Accepts:
        realspace_image             (2D array)
        diffractionspace_image      (2D array)
        realspace_pdict             (dictionary) arguments and values to pass
                                    to the show() fn for the real space image
        diffractionspace_pdict      (dictionary)
        vx,vy                       (1D arrays) x,y components of the vectors
                                    in either real or diffraction space,
                                    depending on the value of vector_space.
                                    Note (vx,vy) is used for the orientation
                                    only - the two vectors are normalized
                                    and rescaled by
        vlength_R,vlenght_Q         (number) the vector length in each
                                    space, in pixels
        x0_R,y0_R,x0_Q,y0_Q         (numbers) the origins / vector tail positions
        QR_rotation               (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
        vector_space                (string) must be 'R' or 'Q'. Specifies
                                    whether the (vx,vy) values passed to this
                                    function describes a real or diffracation
                                    space vector.
    """
    assert(vector_space in ('R','Q'))
    assert(len(vx)==len(vy))
    if isinstance(color_R,tuple) or isinstance(color_R,list):
       assert(len(vx)==len(color_R))
    else:
        color_R =[color_R for i in range(len(vx))]
    if isinstance(color_Q,tuple) or isinstance(color_Q,list):
       assert(len(vx)==len(color_Q))
    else:
        color_Q =[color_Q for i in range(len(vx))]

    fig,(ax1,ax2) = show_RQ(realspace_image, diffractionspace_image,
                            realspace_pdict, diffractionspace_pdict,
                            figsize=figsize,returnfig=True)
    for x,y,cR,cQ in zip(vx,vy,color_R,color_Q):
        if vector_space=='R':
            ax_addvector(ax1,x,y,vlength_R,x0_R,y0_R,width=width_R,color=cR)
            ax_addvector_RtoQ(ax2,x,y,vlength_Q,x0_Q,y0_Q,QR_rotation,width=width_Q,color=cQ)
        else:
            ax_addvector(ax2,x,y,vlength_Q,x0_Q,y0_Q,width=width_Q,color=cQ)
            ax_addvector_QtoR(ax1,x,y,vlength_R,x0_R,y0_R,QR_rotation,width=width_R,color=cR)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,(ax1,ax2)

def ax_addaxes(ax,vx,vy,vlength,x0,y0,width=1,color='r',labelaxes=True,
               labelsize=12,labelcolor='r',righthandedcoords=True):
    """
    Adds a pair of x/y axes to the matplotlib subplot ax. The user supplies
    the x-axis direction with (vx,vy), and the y-axis is then chosen
    by rotating 90 degrees, in a direction set by righthandedcoords.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the x-axis,
                                    Only the orientation is used; the axis
                                    is normalized and rescaled by
        vlength                     (number) the axis length
        x0,y0                       (numbers) the origin of the axes
        labelaxes                   (bool) if True, label 'x' and 'y'
        righthandedcoords           (bool) if True, y-axis is counterclockwise
                                    with respect to x-axis
    """
    # Get the x-axis
    vL = np.hypot(vx,vy)
    xaxis_x,xaxis_y = vlength*vx/vL,vlength*vy/vL
    # Get the y-axes
    if righthandedcoords:
        yaxis_x,yaxis_y = -xaxis_y,xaxis_x
    else:
        yaxis_x,yaxis_y = xaxis_y,-xaxis_x
    ax_addvector(ax,xaxis_x,xaxis_y,vlength,x0,y0,width=width,color=color)
    ax_addvector(ax,yaxis_x,yaxis_y,vlength,x0,y0,width=width,color=color)
    # Label axes:
    if labelaxes:
        xaxislabel_x = x0 + 1.1*xaxis_x
        xaxislabel_y = y0 + xaxis_y
        yaxislabel_x = x0 + yaxis_x
        yaxislabel_y = y0 + 1.1*yaxis_y
        ax.text(xaxislabel_y,xaxislabel_x,'x',color=labelcolor,size=labelsize)
        ax.text(yaxislabel_y,yaxislabel_x,'y',color=labelcolor,size=labelsize)

def ax_addaxes_QtoR(ax,vx,vy,vlength,x0,y0,QR_rotation,width=1,color='r',
                    labelaxes=True,labelsize=12,labelcolor='r'):
    """
    Adds a pair of x/y axes to the matplotlib subplot ax. The user supplies
    the x-axis direction with (vx,vy) in reciprocal space coordinates, and
    the function transforms and displays the corresponding vector in real space.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the x-axis,
                                    in reciprocal space coordinates. Only
                                    the orientation is used; the axes
                                    are normalized and rescaled by
        vlength                     (number) the axis length, in real space
        x0,y0                       (numbers) the origin of the axes, in
                                    real space
        labelaxes                   (bool) if True, label 'x' and 'y'
        QR_rotation                 (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
    """
    vx,vy,_,_ = get_Rvector_from_Qvector(vx,vy,QR_rotation)
    ax_addaxes(ax,vx,vy,vlength,x0,y0,width=width,color=color,labelaxes=labelaxes,
               labelsize=labelsize,labelcolor=labelcolor,righthandedcoords=True)

def ax_addaxes_RtoQ(ax,vx,vy,vlength,x0,y0,QR_rotation,width=1,color='r',
                    labelaxes=True,labelsize=12,labelcolor='r'):
    """
    Adds a pair of x/y axes to the matplotlib subplot ax. The user supplies
    the x-axis direction with (vx,vy) in real space coordinates, and the function
    transforms and displays the corresponding vector in reciprocal space.

    Accepts:
        ax                          (matplotlib subplot)
        vx,vy                       (numbers) x,y components of the x-axis,
                                    in real space coordinates. Only
                                    the orientation is used; the axes
                                    are normalized and rescaled by
        vlength                     (number) the axis length, in reciprocal space
        x0,y0                       (numbers) the origin of the axes, in
                                    reciprocal space
        labelaxes                   (bool) if True, label 'x' and 'y'
        QR_rotation                 (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
    """
    _,_,vx,vy = get_Qvector_from_Rvector(vx,vy,QR_rotation)
    ax_addaxes(ax,vx,vy,vlength,x0,y0,width=width,color=color,labelaxes=labelaxes,
               labelsize=labelsize,labelcolor=labelcolor,righthandedcoords=True)

def show_RQ_axes(realspace_image,diffractionspace_image,
                 realspace_pdict, diffractionspace_pdict,
                 vx,vy,vlength_R,vlength_Q,x0_R,y0_R,x0_Q,y0_Q,
                 QR_rotation,vector_space='R',
                 width_R=1,color_R='r',width_Q=1,color_Q='r',
                 labelaxes=True,labelcolor_R='r',labelcolor_Q='r',
                 labelsize_R=12,labelsize_Q=12,figsize=(12,6),returnfig=False):
    """
    Shows side-by-side real/reciprocal space images with a set of corresponding
    coordinate axes overlaid in each.  (vx,vy) specifies the x-axis, and the y-axis
    is rotated 90 degrees counterclockwise in reciprocal space (relevant in case of
    an R/Q transposition).

    Accepts:
        realspace_image             (2D array)
        diffractionspace_image      (2D array)
        realspace_pdict             (dictionary) arguments and values to pass
                                    to the show() fn for the real space image
        diffractionspace_pdict      (dictionary)
        vx,vy                       (numbers) x,y components of the x-axis
                                    in either real or diffraction space,
                                    depending on the value of vector_space.
                                    Note (vx,vy) is used for the orientation
                                    only - the vectors are normalized
                                    and rescaled by
        vlength_R,vlength_Q         (number or 1D arrays) the vector length in each
                                    space, in pixels
        x0_R,y0_R,x0_Q,y0_Q         (number) the origins / vector tail positions
        QR_rotation               (number) the offset angle between real and
                                    diffraction space. Specifically, this is
                                    the counterclockwise rotation of real space
                                    with respect to diffraction space. In degrees.
        vector_space                (string) must be 'R' or 'Q'. Specifies
                                    whether the (vx,vy) values passed to this
                                    function describes a real or diffracation
                                    space vector.
    """
    assert(vector_space in ('R','Q'))
    fig,(ax1,ax2) = show_RQ(realspace_image, diffractionspace_image,
                            realspace_pdict, diffractionspace_pdict,
                            figsize=figsize,returnfig=True)
    if vector_space=='R':
        ax_addaxes(ax1,vx,vy,vlength_R,x0_R,y0_R,width=width_R,color=color_R,
                   labelaxes=labelaxes,labelsize=labelsize_R,labelcolor=labelcolor_R)
        ax_addaxes_RtoQ(ax2,vx,vy,vlength_Q,x0_Q,y0_Q,QR_rotation,width=width_Q,color=color_Q,
                        labelaxes=labelaxes,labelsize=labelsize_Q,labelcolor=labelcolor_Q)
    else:
        ax_addaxes(ax2,vx,vy,vlength_Q,x0_Q,y0_Q,width=width_Q,color=color_Q,
                   labelaxes=labelaxes,labelsize=labelsize_Q,labelcolor=labelcolor_Q)
        ax_addaxes_QtoR(ax1,vx,vy,vlength_R,x0_R,y0_R,QR_rotation,width=width_R,color=color_R,
                        labelaxes=labelaxes,labelsize=labelsize_R,labelcolor=labelcolor_R)
    if not returnfig:
        plt.show()
        return
    else:
        return fig,(ax1,ax2)


