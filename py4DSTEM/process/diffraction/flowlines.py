# Functions for creating flowline maps from diffraction spots


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

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
    thresh_grow = 0.1,
    thresh_stop = 0.1,
    line_separation = 4.0,
    step_size = 0.25,
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
        line_separation (float):   Separation of the flowlines in pixels. Can be a float or a vector of floats.
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

    # Expand line separation into a vector if needed
    if np.isscalar(line_separation):
        line_separation = np.ones(num_radii) * line_separation

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

    # initalize flowline array
    orient_flowlines = np.zeros_like(orient_hist)

    # initialize output
    xy_t_int = np.zeros((max_steps+1,4))

    # Loop over radial bins
    for a0 in np.arange(num_radii):
        # Find all seed locations
        orient = orient_hist[:,:,a0,:]
        sub_seeds = np.logical_and(np.logical_and(
            orient >= np.roll(orient,1,axis=2),
            orient >= np.roll(orient,-1,axis=2)),
            orient >= thresh_seed)
        x_inds,y_inds,t_inds = np.where(sub_seeds)

        inds_sort = np.argsort(orient[sub_seeds])[::-1]
        x_inds = x_inds[inds_sort]
        y_inds = y_inds[inds_sort]
        t_inds = t_inds[inds_sort]    

        for a1 in tqdmnd(range(0,21), desc="Drawing flowlines",unit=" seeds", disable=not progress_bar):
            # initial coordinate and intensity
            xy0 = np.array((x_inds[a1],y_inds[a1]))
            t = theta[t_inds[a1]]
            int_val = get_intensity(orient,xy0[0],xy0[1],t/dtheta)
            xy_t_int[0,:] = np.hstack((xy0,t/dtheta,int_val))[None,:]

            # forward direction
            v = np.array((-np.sin(t),np.cos(t))) * step_size
            xy = xy0
            # main loop
            grow = True
            count = 0
            while grow is True:
                count += 1

                # update position and intensity
                xy = xy + v 
                int_val = get_intensity(orient,xy[0],xy[1],t/dtheta)

                if  xy[0] < 0 or \
                    xy[1] <0 or \
                    xy[0] < 0 or \
                    xy[1] < 0 or \
                    int_val < thresh_grow:
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

                    if count > max_steps:
                        grow=False
                    #grow=False

            # write into output array
            # print(np.round(xy_t_int[0:count,:],decimals=0))
            orient_flowlines[:,:,a0,:] = set_intensity(orient_flowlines[:,:,a0,:],xy_t_int[0:count,:])

            # reverse direction






        # inds = np.unravel_index(np.where(sub_seeds),size_3D)
        # print(inds.shape)


        # Generate all streams


    # normalize to step size
    orient_flowlines = orient_flowlines * step_size

    return orient_flowlines



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
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(1-dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF  ,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(1-dy)*(  dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF+1,tF  ], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(  dy)*(1-dt)
    inds_1D = np.ravel_multi_index(
        [xF+1,yF+1,tF+1], 
        orient.shape[0:3], 
        mode=['clip','clip','wrap'])
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:,3]*(1-dx)*(  dy)*(  dt)


    return orient
