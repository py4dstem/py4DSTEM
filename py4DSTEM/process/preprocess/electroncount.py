# Electron counting
#
# Includes functions for electron counting on either the CPU (electron_count) or the GPU
# (electron_count_GPU).  For GPU electron counting, pytorch is used to interface between
# numpy and the GPU, and the datacube is expected in numpy.memmap (memory mapped) form.

import numpy as np
from scipy import optimize

from ..utils import get_maximal_points, print_progress_bar, bin2D
from ...io import PointListArray

def electron_count(datacube, darkreference, Nsamples=40,
                                            thresh_bkgrnd_Nsigma=4,
                                            thresh_xray_Nsigma=10,
                                            binfactor = 1,
                                            sub_pixel=True,
                                            output='pointlist'):
    """
    Performs electron counting.

    The algorithm is as follows:
    From a random sampling of frames, calculate an x-ray and background threshold value.
    In each frame, subtract the dark reference, then apply the two thresholds. Find all
    local maxima with respect to the nearest neighbor pixels. These are considered
    electron strike events.

    Thresholds are specified in units of standard deviations, either of a gaussian fit to
    the histogram background noise (for thresh_bkgrnd) or of the histogram itself (for
    thresh_xray). The background (lower) threshold is more important; we will always be
    missing some real electron counts and incorrectly counting some noise as electron
    strikes - this threshold controls their relative balance. The x-ray threshold may be
    set fairly high.

    Args:
        datacube: a 4D numpy.ndarray pointing to the datacube. Note: the R/Q axes are
            flipped with respect to py4DSTEM DataCube objects
        darkreference: a 2D numpy.ndarray with the dark reference
        Nsamples: the number of frames to use in dark reference and threshold
            calculation.
        thresh_bkgrnd_Nsigma: the background threshold is
            ``mean(guassian fit) + (this #)*std(gaussian fit)``
            where the gaussian fit is to the background noise.
        thresh_xray_Nsigma: the X-ray threshold is
            ``mean(hist) +/- (this #)*std(hist)``
            where hist is the histogram of all pixel values in the Nsamples random frames
        binfactor: the binnning factor
        sub_pixel (bool): controls whether subpixel refinement is performed
        output (str): controls output format; must be 'datacube' or 'pointlist'

    Returns:
        (variable) if output=='pointlist', returns a PointListArray of all electron
        counts in each frame. If output=='datacube', returns a 4D array of bools, with
        True indicating electron strikes
    """
    assert isinstance(output, str), "output must be a str"
    assert output in ['pointlist', 'datacube'], "output must be 'pointlist' or 'datacube'"

    # Get dimensions
    R_Nx,R_Ny,Q_Nx,Q_Ny = np.shape(datacube)

    # Get threshholds
    print('Calculating threshholds')
    thresh_bkgrnd, thresh_xray = calculate_thresholds(
                                      datacube,
                                      darkreference,
                                      Nsamples=Nsamples,
                                      thresh_bkgrnd_Nsigma=thresh_bkgrnd_Nsigma,
                                      thresh_xray_Nsigma=thresh_xray_Nsigma)

    # Save to a new datacube
    if output=='datacube':
        counted = np.ones((R_Nx,R_Ny,Q_Nx//binfactor,Q_Ny//binfactor))
        # Loop through frames
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                print_progress_bar(Rx*R_Ny+Ry+1, R_Ny*R_Nx, prefix=' Counting:', suffix='Complete',
                                                                                 length = 50)
                frame = datacube[Rx,Ry,:,:].astype(np.int16)    # Get frame from file
                workingarray = frame-darkreference              # Subtract dark ref from frame
                events = workingarray>thresh_bkgrnd             # Threshold electron events
                events *= thresh_xray>workingarray

                ## Keep events which are greater than all NN pixels ##
                events = get_maximal_points(workingarray*events)

                if(binfactor>1):
                    # Perform binning
                    counted[Rx,Ry,:,:]=bin2D(events, factor=binfactor)
                else:
                    counted[Rx,Ry,:,:]=events
        return counted

    # Save to a PointListArray
    else:
        coordinates = [('qx',int),('qy',int)]
        pointlistarray = PointListArray(coordinates=coordinates, shape=(R_Nx,R_Ny))
        # Loop through frames
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                print_progress_bar(Rx*R_Ny+Ry+1, R_Ny*R_Nx, prefix=' Counting:',
                                    suffix='Complete', length=50)
                frame = datacube[Rx,Ry,:,:].astype(np.int16)    # Get frame from file
                workingarray = frame-darkreference        # Subtract dark ref from frame
                events = workingarray>thresh_bkgrnd       # Threshold electron events
                events *= thresh_xray>workingarray

                ## Keep events which are greater than all NN pixels ##
                events = get_maximal_points(workingarray*events)

                # Perform binning
                if(binfactor>1):
                    events=bin2D(events, factor=binfactor)

                # Save to PointListArray
                x,y = np.nonzero(events)
                pointlist = pointlistarray.get_pointlist(Rx,Ry)
                pointlist.add_tuple_of_nparrays((x,y))

    return pointlistarray

def electron_count_GPU(datacube, darkreference, Nsamples=40,
                                            thresh_bkgrnd_Nsigma=4,
                                            thresh_xray_Nsigma=10,
                                            binfactor = 1,
                                            sub_pixel=True,
                                            output='pointlist'):
    """
    Performs electron counting on the GPU.

    Uses pytorch to interface between numpy and cuda.  Requires cuda and pytorch.
    This function expects datacube to be a np.memmap object.
    See electron_count() for additional documentation.
    """
    import torch
    import dm
    assert isinstance(output, str), "output must be a str"
    assert output in ['pointlist', 'datacube'], "output must be 'pointlist' or 'datacube'"

    # Get dimensions
    R_Nx,R_Ny,Q_Nx,Q_Ny = np.shape(datacube)

    # Get threshholds
    print('Calculating threshholds')
    thresh_bkgrnd, thresh_xray = calculate_thresholds(datacube,
                                                      darkreference,
                                                      Nsamples=Nsamples,
                                                      thresh_bkgrnd_Nsigma=thresh_bkgrnd_Nsigma,
                                                      thresh_xray_Nsigma=thresh_xray_Nsigma)

    # Make a torch device object, to interface numpy with the GPU
    # Put a few arrays on it - dark reference, counted image
    device = torch.device('cuda')
    darkref = torch.from_numpy(darkreference.astype(np.int16)).to(device)
    counted = torch.ones(R_Nx,R_Ny,Q_Nx//binfactor,Q_Ny//binfactor,dtype=torch.short).to(device)

    # Loop through frames
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            print_progress_bar(Rx*R_Ny+Ry+1, R_Ny*R_Nx, prefix=' Counting:', suffix='Complete',
                                                                             length = 50)
            frame = datacube[Rx,Ry,:,:].astype(np.int16)    # Get frame from file
            gframe = torch.from_numpy(frame).to(device)     # Move frame to GPU
            workingarray = gframe-darkref                   # Subtract dark ref from frame
            events = workingarray>thresh_bkgrnd             # Threshold electron events
            events = thresh_xray>workingarray

            ## Keep events which are greater than all NN pixels ##

            #Check pixel is greater than all adjacent pixels
            log = workingarray[1:-1,:]>workingarray[0:-2,:]
            events[1:-1,:] = events[1:-1,:] & log
            log = workingarray[0:-2,:]>workingarray[1:-1,:]
            events[0:-2,:] = events[0:-2,:] & log
            log = workingarray[:,1:-1]>workingarray[:,0:-2]
            events[:,1:-1] = events[:,1:-1] & log
            log = workingarray[:,0:-2]>workingarray[:,1:-1]
            events[:,0:-2] = events[:,0:-2] & log
            #Check pixel is greater than adjacent diagonal pixels
            log = workingarray[1:-1,1:-1]>workingarray[0:-2,0:-2]
            events[1:-1,1:-1] = events[1:-1,1:-1] & log
            log = workingarray[0:-2,1:-1]>workingarray[1:-1,0:-2]
            events[0:-2,1:-1] = events[0:-2,1:-1] & log
            log = workingarray[1:-1,0:-2]>workingarray[0:-2,1:-1]
            events[2:-1,0:-2] = events[1:-1,0:-2] & log
            log = workingarray[0:-2,0:-2]>workingarray[1:-1,1:-1]
            events[0:-2,0:-2] = events[0:-2,0:-2] & log

            if(binfactor>1):
                # Perform binning on GPU in torch_bin function
                counted[Rx,Ry,:,:]=torch.transpose(torch_bin(events.type(torch.cuda.ShortTensor),
                                            device,factor=binfactor),0,1).flip(0).flip(1)
            else:
                # I'm not sure I understand this - we're flipping coordinates to match what?
                # TODO: check array flipping - may vary by camera
                counted[Rx,Ry,:,:]=torch.transpose(events.type(torch.cuda.ShortTensor),0,1).flip(0).flip(1)

    if output=='datacube':
        return counted.cpu().numpy()
    else:
        return counted_datacube_to_pointlistarray(counted)

####### Support functions ########

def calculate_thresholds(datacube, darkreference,
                                   Nsamples=20,
                                   thresh_bkgrnd_Nsigma=4,
                                   thresh_xray_Nsigma=10,
                                   return_params=False):
    """
    Calculate the upper and lower thresholds for thresholding what to register as
    an electron count.

    Both thresholds are determined from the histogram of detector pixel values summed
    over Nsamples frames. The thresholds are set to::

        thresh_xray_Nsigma =    mean(histogram)    + thresh_upper * std(histogram)
        thresh_bkgrnd_N_sigma = mean(guassian fit) + thresh_lower * std(gaussian fit)

    For more info, see the electron_count docstring.

    Args:
        datacube: a 4D numpy.ndarrau pointing to the datacube
        darkreference: a 2D numpy.ndarray with the dark reference
        Nsamples: the number of frames to use in dark reference and threshold
            calculation.
        thresh_bkgrnd_Nsigma: the background threshold is
            ``mean(guassian fit) + (this #)*std(gaussian fit)``
            where the gaussian fit is to the background noise.
        thresh_xray_Nsigma: the X-ray threshold is
            ``mean(hist) + (this #)*std(hist)``
            where hist is the histogram of all pixel values in the Nsamples random frames
        return_params: bool, if True return n,hist of the histogram and popt of the
            gaussian fit

    Returns:
        (5-tuple): A 5-tuple containing:

            * **thresh_bkgrnd**: the background threshold
            * **thresh_xray**: the X-ray threshold
            * **n**: returned iff return_params==True. The histogram values
            * **hist**: returned iff return_params==True. The histogram bin edges
            * **popt**: returned iff return_params==True. The fit gaussian parameters,
              (A, mu, sigma).
    """
    R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.shape

    # Select random set of frames
    nframes = R_Nx*R_Ny
    samples = np.arange(nframes)
    np.random.shuffle(samples)
    samples = samples[:Nsamples]

    # Get frames and subtract dark references
    sample = np.zeros((Q_Nx,Q_Ny,Nsamples),dtype=np.int16)
    for i in range(Nsamples):
        sample[:,:,i] = datacube[samples[i]//R_Nx,samples[i]%R_Ny,:,:]
        sample[:,:,i] -= darkreference
    sample = np.ravel(sample)  # Flatten array

    # Get upper (X-ray) threshold
    mean = np.mean(sample)
    stddev = np.std(sample)
    thresh_xray = mean+thresh_xray_Nsigma*stddev

    # Make a histogram
    binmax = min(int(np.ceil(np.amax(sample))),int(mean+thresh_xray*stddev))
    binmin = max(int(np.ceil(np.amin(sample))),int(mean-thresh_xray*stddev))
    step = max(1,(binmax-binmin)//1000)
    bins = np.arange(binmin,binmax,step=step,dtype=np.int)
    n, bins = np.histogram(sample,bins=bins)

    # Define Guassian to fit to, with parameters p:
    #   p[0] is amplitude
    #   p[1] is the mean
    #   p[2] is std deviation
    fitfunc = lambda p, x: p[0]*np.exp(-0.5*np.square((x-p[1])/p[2]))
    errfunc = lambda p, x, y: fitfunc(p, x) - y          # Error for scipy's optimize routine

    # Get initial guess
    p0 = [n.max(),(bins[n.argmax()+1]-bins[n.argmax()])/2,np.std(sample)]
    p1,success = optimize.leastsq(errfunc,p0[:],args=(bins[:-1],n))  # Use the scipy optimize routine
    p1[1] += 0.5                            #Add a half to account for integer bin width

    # Set lower threshhold for electron counts to count
    thresh_bkgrnd = p1[1]+p1[2]*thresh_bkgrnd_Nsigma

    if return_params:
        return thresh_bkgrnd, thresh_xray, n, bins, p1
    else:
        return thresh_bkgrnd, thresh_xray


def torch_bin(array,device,factor=2):
    """
    Bin data on the GPU using torch.

    Args:
        array: a 2D numpy array
        device: a torch device class instance
        factor (int): the binning factor

    Returns:
        (array): the binned array
    """

    import torch

    x,y =  array.shape
    binx,biny = x//factor,y//factor
    xx,yy = binx*factor,biny*factor

    # Make a binned array on the device
    binned_ar = torch.zeros(biny,binx,device=device,dtype = array.dtype)

    # Collect pixel sums into new bins
    for ix in range(factor):
        for iy in range(factor):
            binned_ar += array[0+ix:xx+ix:factor,0+iy:yy+iy:factor]
    return binned_ar

def counted_datacube_to_pointlistarray(counted_datacube, subpixel=False):
    """
    Converts an electron counted datacube to PointListArray.

    Args:
        counted_datacube: a 4D array of bools, with true indicating an electron strike.
        subpixel (bool): controls if subpixel electron strike positions are expected

    Returns:
        (PointListArray): a PointListArray of electron strike events
    """
    # Get shape, initialize PointListArray
    R_Nx,R_Ny,Q_Nx,Q_Ny = counted_datacube.shape
    if subpixel:
        coordinates = [('qx',float),('qy',float)]
    else:
        coordinates = [('qx',int),('qy',int)]
    pointlistarray = PointListArray(coordinates=coordinates, shape=(R_Nx,R_Ny))

    # Loop through frames, adding electron counts to the PointListArray for each.
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            frame = counted_datacube[Rx,Ry,:,:]
            x,y = np.nonzero(frame)
            pointlist = pointlistarray.get_pointlist(Rx,Ry)
            pointlist.add_tuple_of_nparrays((x,y))

    return pointlistarray

def counted_pointlistarray_to_datacube(counted_pointlistarray, shape, subpixel=False):
    """
    Converts an electron counted PointListArray to a datacube.

    Args:
        counted_pointlistarray (PointListArray): a PointListArray of electron strike
            events
        shape (4-tuple): a length 4 tuple of ints containing (R_Nx,R_Ny,Q_Nx,Q_Ny)
        subpixel (bool): controls if subpixel electron strike positions are expected

    Returns:
        (4D array of bools): a 4D array of bools, with true indicating an electron strike.
    """
    assert len(shape)==4
    assert subpixel==False, "subpixel mode not presently supported."
    R_Nx,R_Ny,Q_Nx,Q_Ny = shape
    counted_datacube = np.zeros((R_Nx,R_Nx,Q_Nx,Q_Ny),dtype=bool)

    # Loop through frames, adding electron counts to the datacube for each.
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            pointlist = counted_pointlistarray.get_pointlist(Rx,Ry)
            counted_datacube[Rx,Ry,pointlist.data['qx'],pointlist.data['qy']] = True

    return counted_datacube



if __name__=="__main__":

    from py4DSTEM.process.preprocess import get_darkreference
    from py4DSTEM.io import DataCube, save
    from ncempy.io import dm

    dm4_filepath = 'Capture25.dm4'

    # Parameters for dark reference determination
    drwidth = 100

    # Parameters for electron counting
    Nsamples = 40
    thresh_bkgrnd_Nsigma = 4
    thresh_xray_Nsigma = 30
    binfactor = 1
    subpixel = False
    output = 'pointlist'


    # Get memory mapped 4D datacube from dm file
    datacube = dm.dmReader(dm4_filepath,dSetNum=0,verbose=False)['data']
    datacube = np.moveaxis(datacube,(0,1),(2,3))

    # Get dark reference
    darkreference = 1 # TODO: get_darkreference(datacube = ...!

    electron_counted_data = electron_count(datacube, darkreference, Nsamples=Nsamples,
                                           thresh_bkgrnd_Nsigma=thresh_bkgrnd_Nsigma,
                                           thresh_xray_Nsigma=thresh_xray_Nsigma,
                                           binfactor=binfactor,
                                           sub_pixel=True,
                                           output='pointlist')

    # For outputting datacubes, wrap counted into a py4DSTEM DataCube
    if output=='datacube':
        electron_counted_data = DataCube(data=electron_counted_data)

    output_path = dm4_filepath.replace('.dm4','.h5')
    save(electron_counted_data, output_path)



