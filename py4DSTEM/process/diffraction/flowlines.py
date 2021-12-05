# Functions for creating flowline maps from diffraction spots


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd


def make_orientation_histogram(
    bragg_peaks,
    radial_ranges,
    upsample_factor=4.0,
    theta_step_deg=1.0,
    sigma_x = 1.0,
    sigma_y = 1.0,
    sigma_theta = 3.0,
    normalize_intensity: bool = True,
    progress_bar: bool = True,
    ):
    """
    Create an 3D or 4D orientation histogram from a braggpeaks PointListArray,
    from user-specified radial ranges.
    
    Args:
        bragg_peaks (PointListArray):    2D of pointlists containing centered peak locations.
        radial_ranges (np array):       [N x 2] array for N radial bins.
        upsample_factor (float):          Upsample factor
        theta_step_deg (float):         Step size along annular direction in degrees
        sigma_x (float):                smoothing in x direction before upsample
        sigma_y (float):                smoothing in x direction before upsample
        sigma_theta (float):            smoothing in annular direction (units of bins, periodic)
        normalize_intensity (bool):   Normalize to max peak intensity = 1
        progress_bar (bool):          Enable progress bar

    Returns:
        orient_hist (array):          3D or 4D array containing Bragg peaks
    """

    # Input bins
    radial_ranges = np.array(radial_ranges)
    num_radii = radial_ranges.shape[0]
    
    # coordinates
    theta = np.arange(0,180,theta_step_deg) * np.pi / 180.0
    dtheta = theta[1] - theta[0]
    num_theta_bins = np.size(theta)
    size_input = bragg_peaks.shape
    size_output = np.round(np.array(size_input).astype('float') * upsample_factor).astype('int')

    # output init
    orient_hist = np.zeros([
        size_output[0],
        size_output[1],
        num_radii,num_theta_bins])

    # Loop over all probe positions
    for rx, ry in tqdmnd(
            *bragg_peaks.shape, desc="Generating histogram",unit=" probe positions", disable=not progress_bar
        ):

        x = (rx + 0.5)*upsample_factor - 0.5
        y = (ry + 0.5)*upsample_factor - 0.5
        x = np.clip(x,0,size_output[0]-2)
        y = np.clip(y,0,size_output[1]-2)

        xF = np.floor(x).astype('int')
        yF = np.floor(y).astype('int')
        dx = x - xF
        dy = y - yF

        p = bragg_peaks.get_pointlist(rx,ry)
        r2 = p.data['qx']**2 + p.data['qy']**2

        for a0 in range(num_radii):
            sub = np.logical_and(r2 >= radial_ranges[a0,0], r2 >= radial_ranges[a0,1])
            
            if np.any(sub):
                intensity = p.data['intensity'][sub]
                t = np.arctan2(p.data['qy'][sub],p.data['qx'][sub]) / dtheta
                tF = np.floor(t).astype('int')
                dt = t - tF

                orient_hist[    xF  ,yF  ,a0,:] = orient_hist[xF  ,yF  ,a0,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(1-dx)*(1-dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[    xF  ,yF  ,a0,:] = orient_hist[xF  ,yF  ,a0,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(1-dx)*(1-dy)*(  dt)*intensity,minlength=num_theta_bins)

                orient_hist[    xF+1,yF  ,a0,:] = orient_hist[xF+1,yF  ,a0,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(  dx)*(1-dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[    xF+1,yF  ,a0,:] = orient_hist[xF+1,yF  ,a0,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(  dx)*(1-dy)*(  dt)*intensity,minlength=num_theta_bins)
 
                orient_hist[    xF  ,yF+1,a0,:] = orient_hist[xF  ,yF+1,a0,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(1-dx)*(  dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[    xF  ,yF+1,a0,:] = orient_hist[xF  ,yF+1,a0,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(1-dx)*(  dy)*(  dt)*intensity,minlength=num_theta_bins)

                orient_hist[    xF+1,yF+1,a0,:] = orient_hist[xF+1,yF+1,a0,:] + \
                    np.bincount(np.mod(tF  ,num_theta_bins),
                        weights=(  dx)*(  dy)*(1-dt)*intensity,minlength=num_theta_bins)
                orient_hist[    xF+1,yF+1,a0,:] = orient_hist[xF+1,yF+1,a0,:] + \
                    np.bincount(np.mod(tF+1,num_theta_bins),
                        weights=(  dx)*(  dy)*(  dt)*intensity,minlength=num_theta_bins)               

    # smoothing
    if sigma_x is not None and sigma_x > 0:
        orient_hist = gaussian_filter1d(orient_hist,sigma_x*upsample_factor,mode='nearest',axis=0)
    if sigma_y is not None and sigma_y > 0:
        orient_hist = gaussian_filter1d(orient_hist,sigma_y*upsample_factor,mode='nearest',axis=1)
    if sigma_theta is not None and sigma_theta > 0:
        orient_hist = gaussian_filter1d(orient_hist,sigma_theta,mode='wrap',axis=3)



    # normalization
    if normalize_intensity is True:
        for a0 in range(num_radii):
            orient_hist[:,:,a0,:] = orient_hist[:,:,a0,:] / np.max(orient_hist[:,:,a0,:])



    return orient_hist





def make_flowline_map(
    orient_hist,
    thresh_seed = 0.2,
    thresh_grow = 0.05,
    thresh_collision = 0.01,
    sep_seeds = 3,
    sep_xy = 6.0,
    sep_theta = 5.0,
    linewidth = 2.0,
    step_size = 0.5,
    max_steps = 1000,
    sigma_x = 1.0,
    sigma_y = 1.0,
    sigma_theta = 2.0,
    progress_bar: bool = True,
    ):
    """
    Create an 3D or 4D orientation flowline map - essentially a pixelated "stream map" which represents diffraction data.
    
    Args:
        orient_hist (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                We assume theta bin ranges from 0 to 180 degrees and is periodic.
        sep (float):            Separation of the flowlines in pixels.
        progress_bar (bool):          Enable progress bar

    Returns:
        orient_flowlines (array):          3D or 4D array containing flowlines
    """

    # number of radial bins
    num_radii = orient_hist.shape[2]
    
    # coordinates
    theta = np.linspace(0,np.pi,orient_hist.shape[3],endpoint=False)
    dtheta = theta[1] - theta[0]
    size_3D = np.array([
        orient_hist.shape[0],
        orient_hist.shape[1],
        orient_hist.shape[3],
    ])

    # initialize weighting array
    vx = np.arange(-np.ceil(2*sigma_x),np.ceil(2*sigma_x)+1)
    vy = np.arange(-np.ceil(2*sigma_y),np.ceil(2*sigma_y)+1)
    vt = np.arange(-np.ceil(2*sigma_theta),np.ceil(2*sigma_theta)+1)
    ay,ax,at = np.meshgrid(vy,vx,vt)
    k = np.exp(ax**2/(-2*sigma_x**2)) * \
        np.exp(ay**2/(-2*sigma_y**2)) * \
        np.exp(at**2/(-2*sigma_theta**2))
    k = k / np.sum(k)  
    vx = vx[:,None,None].astype('int')
    vy = vy[None,:,None].astype('int')
    vt = vt[None,None,:].astype('int')

    # initialize collision check array
    cr = np.arange(-np.ceil(sep_xy),np.ceil(sep_xy)+1)
    ct = np.arange(-np.ceil(sep_theta),np.ceil(sep_theta)+1)
    ay,ax,at = np.meshgrid(cr,cr,ct)
    c_mask = (ax**2 + ay**2)/sep_xy**2 + at**2/sep_theta**2 <= (1 + 1/sep_xy)**2
    cx = cr[:,None,None].astype('int')
    cy = cr[None,:,None].astype('int')
    ct = ct[None,None,:].astype('int')

    # initalize flowline array
    orient_flowlines = np.zeros_like(orient_hist)

    # initialize output
    xy_t_int = np.zeros((max_steps+1,4))
    xy_t_int_rev = np.zeros((max_steps+1,4))

    # Loop over radial bins
    for a0 in np.arange(num_radii):
        # Find all seed locations
        orient = orient_hist[:,:,a0,:]
        sub_seeds = np.logical_and(np.logical_and(
            orient >= np.roll(orient,1,axis=2),
            orient >= np.roll(orient,-1,axis=2)),
            orient >= thresh_seed)

        # Separate seeds
        if sep_seeds > 0:
            for a1 in range(sep_seeds-1):
                sub_seeds[a1::sep_seeds,:,:] = False
                sub_seeds[:,a1::sep_seeds,:] = False

        # Index seeds
        x_inds,y_inds,t_inds = np.where(sub_seeds)
        inds_sort = np.argsort(orient[sub_seeds])[::-1]
        x_inds = x_inds[inds_sort]
        y_inds = y_inds[inds_sort]
        t_inds = t_inds[inds_sort]    

        # for a1 in tqdmnd(range(0,40), desc="Drawing flowlines",unit=" seeds", disable=not progress_bar):
        for a1 in tqdmnd(range(0,x_inds.shape[0]), desc="Drawing flowlines",unit=" seeds", disable=not progress_bar):
            # initial coordinate and intensity
            xy0 = np.array((x_inds[a1],y_inds[a1]))
            t0 = theta[t_inds[a1]]

            # init theta
            inds_theta = np.mod(np.round(t0/dtheta).astype('int')+vt,orient.shape[2])
            orient_crop = k * orient[
                np.clip(np.round(xy0[0]).astype('int')+vx,0,orient.shape[0]-1),
                np.clip(np.round(xy0[1]).astype('int')+vy,0,orient.shape[1]-1),
                inds_theta]
            theta_crop = theta[inds_theta]
            t0 = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)

            # forward direction
            t = t0
            v0 = np.array((-np.sin(t),np.cos(t)))
            v = v0 * step_size
            xy = xy0
            int_val = get_intensity(orient,xy0[0],xy0[1],t0/dtheta)
            xy_t_int[0,0:2] = xy0
            xy_t_int[0,2] = t/dtheta
            xy_t_int[0,3] = int_val
            # main loop
            grow = True
            count = 0
            while grow is True:
                count += 1

                # update position and intensity
                xy = xy + v 
                int_val = get_intensity(orient,xy[0],xy[1],t/dtheta)

                # check for collision
                flow_crop = orient_flowlines[
                    np.clip(np.round(xy[0]).astype('int')+cx,0,orient.shape[0]-1),
                    np.clip(np.round(xy[1]).astype('int')+cy,0,orient.shape[1]-1),
                    (a0).astype('int'),
                    np.mod(np.round(t/dtheta).astype('int')+ct,orient.shape[2])
                ]
                int_flow = np.max(flow_crop[c_mask])

                if  xy[0] < 0 or \
                    xy[1] < 0 or \
                    xy[0] > orient.shape[0] or \
                    xy[1] > orient.shape[1] or \
                    int_val < thresh_grow or \
                    int_flow > thresh_collision:
                    grow = False
                else:
                    # update direction
                    inds_theta = np.mod(np.round(t/dtheta).astype('int')+vt,orient.shape[2])
                    orient_crop = k * orient[
                        np.clip(np.round(xy[0]).astype('int')+vx,0,orient.shape[0]-1),
                        np.clip(np.round(xy[1]).astype('int')+vy,0,orient.shape[1]-1),
                        inds_theta]
                    theta_crop = theta[inds_theta]
                    t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)
                    v = np.array((-np.sin(t),np.cos(t))) * step_size

                    xy_t_int[count,0:2] = xy
                    xy_t_int[count,2] = t/dtheta
                    xy_t_int[count,3] = int_val

                    if count > max_steps-1:
                        grow=False

            # reverse direction
            t = t0 + np.pi
            v0 = np.array((-np.sin(t),np.cos(t)))
            v = v0 * step_size
            xy = xy0
            int_val = get_intensity(orient,xy0[0],xy0[1],t0/dtheta)
            xy_t_int_rev[0,0:2] = xy0
            xy_t_int_rev[0,2] = t/dtheta
            xy_t_int_rev[0,3] = int_val
            # main loop
            grow = True
            count_rev = 0
            while grow is True:
                count_rev += 1

                # update position and intensity
                xy = xy + v 
                int_val = get_intensity(orient,xy[0],xy[1],t/dtheta)

                # check for collision
                flow_crop = orient_flowlines[
                    np.clip(np.round(xy[0]).astype('int')+cx,0,orient.shape[0]-1),
                    np.clip(np.round(xy[1]).astype('int')+cy,0,orient.shape[1]-1),
                    (a0).astype('int'),
                    np.mod(np.round(t/dtheta).astype('int')+ct,orient.shape[2])
                ]
                int_flow = np.max(flow_crop[c_mask])

                if  xy[0] < 0 or \
                    xy[1] < 0 or \
                    xy[0] > orient.shape[0] or \
                    xy[1] > orient.shape[1] or \
                    int_val < thresh_grow or \
                    int_flow > thresh_collision:
                    grow = False
                else:
                    # update direction
                    inds_theta = np.mod(np.round(t/dtheta).astype('int')+vt,orient.shape[2])
                    orient_crop = k * orient[
                        np.clip(np.round(xy[0]).astype('int')+vx,0,orient.shape[0]-1),
                        np.clip(np.round(xy[1]).astype('int')+vy,0,orient.shape[1]-1),
                        inds_theta]
                    theta_crop = theta[inds_theta]
                    t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop) + np.pi
                    v = np.array((-np.sin(t),np.cos(t))) * step_size

                    xy_t_int_rev[count_rev,0:2] = xy
                    xy_t_int_rev[count_rev,2] = t/dtheta
                    xy_t_int_rev[count_rev,3] = int_val

                    if count_rev > max_steps-1:
                        grow=False

            # xy_t_int_rev[0:count_rev,2] = xy_t_int_rev[0:count_rev,2] - np.pi/dtheta

            # print(np.round(xy_t_int[0:100:10,:]))
            # print(np.round(xy_t_int_rev[1:1001:100,:]))
            # print(xy_t_int_rev[1:count_rev:17,3])
            # print(xy_t_int_rev[1:count_rev,0])


            # if count > 1 or count_rev > 2:

            #     fig,ax = plt.subplots(1,1,figsize=(8,4))
            #     ax.plot(
            #         xy_t_int[0:count,0],
            #         xy_t_int[0:count,3])
            #     ax.plot(
            #         xy_t_int_rev[0:count_rev,0],
            #         xy_t_int_rev[0:count_rev,3])

            #     plt.show()

            # write into output array
            if count > 0:
                orient_flowlines[:,:,a0,:] = set_intensity(
                    orient_flowlines[:,:,a0,:],
                    xy_t_int[1:count,:])
            if count_rev > 1:
                orient_flowlines[:,:,a0,:] = set_intensity(
                    orient_flowlines[:,:,a0,:],
                    xy_t_int_rev[1:count_rev,:])

    # normalize to step size
    orient_flowlines = orient_flowlines * step_size

    # linewidth
    if linewidth > 1.0:
        s = linewidth - 1
        
        orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=0)
        orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=1)
        orient_flowlines = orient_flowlines * (s**2)

    return orient_flowlines




def make_flowline_image(
    orient_flowlines,
    int_range = [0,0.5],
    sym_rotation_order = 2,
    greyscale = False,
    greyscale_max = True,
    white_background = False,
    power_scaling = 1,
    # cmap = 'hsv',
    # correct_luminosity = True,
    ):
    """
    Create an 3D or 4D orientation flowline map - essentially a pixelated "stream map" which represents diffraction data.
    
    Args:
        orient_flowline (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                    We assume theta bin ranges from 0 to 180 degrees and is periodic.
        int_range (float)           2 element array giving the intensity range
        sym_rotation_order (int):   rotational symmety for colouring

    Returns:
        im_flowline (array):          3D or 4D array containing flowline images
    """

    # init array
    size_input = orient_flowlines.shape
    size_output = np.array([size_input[0],size_input[1],3,size_input[2]])
    im_flowline = np.zeros(size_output)

    if greyscale is True:
        for a0 in np.arange(size_input[2]):
            if greyscale_max is True:
                im = np.max(orient_flowlines[:,:,a0,:],axis=2)
            else:
                im = np.mmean(orient_flowlines[:,:,a0,:],axis=2)

            sig = np.clip((im - int_range[0]) \
                / (int_range[1] - int_range[0]),0,1)

            if power_scaling != 1:
                sig = sig ** power_scaling

            if white_background is False:
                im_flowline[:,:,:,a0] = sig[:,:,None]
            else:
                im_flowline[:,:,:,a0] = 1-sig[:,:,None]


    else:
        # Color basis
        c0 = np.array([1.0, 0.0, 0.0])
        c1 = np.array([0.0, 0.7, 0.0])
        c2 = np.array([0.0, 0.3, 1.0])

        # angles
        theta = np.linspace(0,np.pi,size_input[3],endpoint=False)
        # dtheta = theta[1] - theta[0]
        theta_color = theta * sym_rotation_order
        # print(theta_color*180/np.pi)

        # color basis
        b0 =  np.maximum(1 - np.abs(np.mod(theta_color + np.pi, 2*np.pi) - np.pi)**2 / (np.pi*2/3)**2, 0)
        b1 =  np.maximum(1 - np.abs(np.mod(theta_color - np.pi*2/3 + np.pi, 2*np.pi) - np.pi)**2 / (np.pi*2/3)**2, 0)
        b2 =  np.maximum(1 - np.abs(np.mod(theta_color - np.pi*4/3 + np.pi, 2*np.pi) - np.pi)**2 / (np.pi*2/3)**2, 0)


        for a0 in np.arange(size_input[2]):
            for a1 in np.arange(size_input[3]):
                sig = np.clip(
                    (orient_flowlines[:,:,a0,a1] - int_range[0]) \
                    / (int_range[1] - int_range[0]),0,1)

                for a2 in np.arange(3):
                    im_flowline[:,:,:,a0] = im_flowline[:,:,:,a0] + \
                        sig[:,:,None]*(c0[None,None,:]*b0[a1]) + \
                        sig[:,:,None]*(c1[None,None,:]*b1[a1]) + \
                        sig[:,:,None]*(c2[None,None,:]*b2[a1])

            # clip limit
            im_flowline[:,:,:,a0] = np.clip(im_flowline[:,:,:,a0],0,1)

            # contrast flip
            if white_background is True:
                im = rgb_to_hsv(im_flowline[:,:,:,a0])
                # im_s = im[:,:,1]
                im_v = im[:,:,2]
                im[:,:,1] = im_v
                im[:,:,2] = 1
                im_flowline[:,:,:,a0] = hsv_to_rgb(im)

        # print(im_flowline.shape)
        # print((sig[:,:,None] * c1[None,None,:]).shape)

        # fig,ax = plt.subplots(1,1,figsize=(8,8))
        # ax.plot(theta,b0)
        # ax.plot(theta,b1)
        # ax.plot(theta,b2)
        # plt.show()


    return im_flowline




def get_intensity(orient,x,y,t):
    x = np.clip(x,0,orient.shape[0]-2)
    y = np.clip(y,0,orient.shape[1]-2)

    xF = np.floor(x).astype('int')
    yF = np.floor(y).astype('int')
    tF = np.floor(t).astype('int')
    dx = x - xF
    dy = y - yF
    dt = t - tF
    t1 = np.mod(tF  ,orient.shape[2])
    t2 = np.mod(tF+1,orient.shape[2])

    int_vals = \
        orient[xF  ,yF  ,t1]*((1-dx)*(1-dy)*(1-dt)) + \
        orient[xF  ,yF  ,t2]*((1-dx)*(1-dy)*(  dt)) + \
        orient[xF  ,yF+1,t1]*((1-dx)*(  dy)*(1-dt)) + \
        orient[xF  ,yF+1,t2]*((1-dx)*(  dy)*(  dt)) + \
        orient[xF+1,yF  ,t1]*((  dx)*(1-dy)*(1-dt)) + \
        orient[xF+1,yF  ,t2]*((  dx)*(1-dy)*(  dt)) + \
        orient[xF+1,yF+1,t1]*((  dx)*(  dy)*(1-dt)) + \
        orient[xF+1,yF+1,t2]*((  dx)*(  dy)*(  dt))

    return int_vals


def set_intensity(orient,xy_t_int):
    # x = np.clip(x,0,orient.shape[0]-1)
    # y = np.clip(y,0,orient.shape[0]-1)

    xF = np.floor(xy_t_int[:,0]).astype('int')
    yF = np.floor(xy_t_int[:,1]).astype('int')
    tF = np.floor(xy_t_int[:,2]).astype('int')
    dx = xy_t_int[:,0] - xF
    dy = xy_t_int[:,1] - yF
    dt = xy_t_int[:,2] - tF

    inds_1D = np.ravel_multi_index(
        [xF  ,yF  ,tF  ], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(1-dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF  ,yF  ,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(1-dy)*(  dt)
    inds_1D = np.ravel_multi_index(
        [xF  ,yF+1,tF  ], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(  dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF  ,yF+1,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(  dy)*(  dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF  ,tF  ], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(  dx)*(1-dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF  ,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(  dx)*(1-dy)*(  dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF+1,tF  ], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(  dx)*(  dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF+1,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(  dx)*(  dy)*(  dt)


    return orient
