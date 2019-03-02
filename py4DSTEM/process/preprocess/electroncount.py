# Electron counting
#
# Electron counting is performed on the GPU, using torch to interface between numpy and the GPU.
# Expects the datacube in numpy.memmap (memory mapped) form.

import numpy as np
from scipy import optimize
import h5py
import torch
import dm

from ...file.datastructure import PointListArray

def electron_count(datacube, darkreference, Nsamples=40,
                                            thresh_bkgrnd_Nsigma=4,
                                            thresh_xray_Nsigma=10
                                            binfactor = 1,
                                            sub_pixel=True,
                                            output='pointlist'):
    """
    Performs electron counting.

    The algorithm is as follows:
    From a random sampling of frames, calculate a dark reference image, and an x-ray and
    baclground threshold value. In each frame, subtract the background, then apply the two
    thresholds. Find the local maxima with respect to the nearest neighbor pixels. These
    are considered electron strike events.

    Thresholds are specified in units of standard deviations, either of a gaussian fit to the
    histogram background noise (for thresh_bkgrnd) or of the histogram itself (for thresh_xray).
    The background (lower) threshold is more important; we will always be missing some real
    electron counts, and will always be incorrectly counting some noise as events, and this
    threshold controls their relative balance. The x-ray threshold may be set fairly high.

    Accepts:
        datacube               a 4D numpy.memmap pointing to the datacube
                               note: the R/Q axes are flipped with respect to py4DSTEM DataCubes
        darkreference          a 2D numpy.ndarray with the dark reference
        Nsamples               the number of frames to use in dark reference
                               and threshold calculation.
        thresh_bkgrnd_Nsigma   background threshold is
                                    mean(guassian fit) + (this #)*std(gaussian fit)
                               where the gaussian fit is to the background noise.
        thresh_xray_Nsigma     the X-ray threshold is
                                    mean(hist) +/- (this #)*std(hist)
                               where hist is the histogram of all pixel values in the
                               Nsamples random frames
        binfactor              the binnning factor
        sub_pixel              (bool) controls whether subpixel refinement is performed
        output                 (str) controls output format; must be 'datacube' or 'pointlist'

    Returns:
        pointlist       if output=='pointlist', returns a PointListArray of all electron counts
                        in each frame
           - OR -
        datacube        if output=='datacube', output a 4D array of bools, with True indicating
                        electron strikes
    """
    assert isinstance(output, str), "output must be a str"
    assert output in ['pointlist', 'datacube'], "output must be 'pointlist' or 'datacube'"

    # Get dimensions
    Q_Nx,Q_Ny,R_Nx,R_Ny = np.shape(datacube)

    # Get threshholds
    print('Calculating threshholds')
    sigma_l, sigma_u = calculate_thresholds(datacube,
                                            darkreference,
                                            Nsamples=Nsamples,
                                            thresh_bkgrd_Nsigma=thresh_bkgrnd_Nsigma,
                                            thresh_xray_Nsigma=thresh_xray_Nsigma)

    # Make a torch device object, to interface numpy with the GPU
    # Put a few arrays on it - dark reference, counted image
    device = torch.device('cuda')
    darkref = torch.from_numpy(darkreference.astype(np.int16)).to(device)
    counted = torch.ones(Q_Nx//binfactor,Q_Ny//binfactor,R_Nx,R_Ny,dtype=torch.short).to(device)

    # Loop through frames
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            printProgressBar(R_Nx*R_Ny+Ry+1, R_Ny*R_Nx, prefix=' Counting:', suffix='Complete',
                                                                             length = 50)
            frame = datacube[:,:,Rx,Ry].astype(np.int16)    # Get frame from file
            gframe = torch.from_numpy(frame).to(device)     # Move frame to GPU
            workingarray = gframe-darkref                   # Subtract dark ref from frame
            events = workingarray>thresh_l                  # Threshold electron events
            events = thresh_u>workingarray

            ## Keep events which are greater than all NN pixels ##

            #Check pixel is greater than all adjacent pixels
            log = workingarray[1:-1,:]>workingarray[0:-2,:]
            events[1:-1,:] = events[1:-1,:] & log
            log = workingarray[0:-1,:]>workingarray[1:-1,:]
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
                counted[:,:,Rx,Ry]=torch.transpose(torch_bin(events.type(torch.cuda.ShortTensor),
                                            device,factor=binfactor),0,1).flip(0).flip(1)
            else:
                # I'm not sure I understand this - we're flipping coordinates to match what?
                counted[:,:,Rx,Ry]=torch.transpose(events.type(torch_.cuda.ShortTensor),0,1).flip(0).flip(1)

    if output=='datacube':
        return counted.cpu().numpy()
    else:
        return counted_datacube_to_pointlistarray(counted)

####### Support functions ########

def calculate_thresholds(datacube, darkreference, Nsamples=20,
                                   thresh_lower=4,thresh_upper=10):
    """
    Calculate the upper and lower thresholds for thresholding what to register as
    an electron count.

    Both thresholds are determined from the histogram of detector pixel values summed over
    Nsamples frames. The thresholds are set to
        thresh_u = mean(histogram)    + thresh_upper * std(histogram)
        thresh_l = mean(guassian fit) + thresh_lower * std(gaussian fit)
    For more info, see the count_datacube docstring.

    Accepts:
        datacube        a 4D numpy.memmap of the data
        darkreference   a 2D numpy.ndarray of the camera dark reference
        Nsamples        number of random frames to sample
        thresh_lower    stds above the mean of a gaussian fit which becomes the upper threshold
        thresh_upper    stds above the mean of the histogram which becomes the upper threshold

    Returns:
        thresh_l        the lower threshold
        thresh_u        the upper threshold
    """
    Q_Nx,Q_Ny,R_Nx,R_Ny = datacube.shape

    # Select random set of frames
    nframes = R_Nx*R_Ny
    samples = np.arange(nframes)
    np.random.shuffle(samples)
    samples = samples[:Nsamples]

    # Get frames and subtract dark references
    sample = np.zeros((Q_Nx,Q_Ny,Nsamples),dtype=np.int16)
    for i in range(Nsamples):
        sample[:,:,i] = datacube[:,:,samples[i]//R_Nx,samples[i]%R_Ny]
        sample[:,:,i] -= darkreference
    sample = np.ravel(sample)  # Flatten array

    # Get upper (X-ray) threshold
    mean = np.mean(sample)
    stddev = np.std(sample)
    thresh_u = mean+thresh_upper*stddev

    # Make a histogram
    binmax = min(int(np.ceil(np.amax(sample))),int(mean+sigmathresh*stddev))
    binmin = max(int(np.ceil(np.amin(sample))),int(mean-sigmathresh*stddev))
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
    p1,success = optimize.leastsq(errfunc,p0[:],args=(bins[:-1],n))   # Use the scipy optimize routine
    p1[1] += 0.5                            #Add a half to account for integer bin width

    # Set lower threshhold for electron counts to count
    thresh_l = p1[1]+p1[2]*thresh_lower

    return thresh_l, thresh_u

def torch_bin(array,device,factor=2):
    """
    Bin data on the GPU using torch.

    Accepts:
        array       a 2D numpy array
        device      a torch device class instance
        factor      (int) the binning factor

    Returns:
        binned_ar   the binned array
    """
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

    Accepts:
        counted_datacube    a 4D array of bools, with true indicating an electron strike.
                            should work with array in a torch GPU device, or as a standard
                            numpy array
        subpixel            (bool) controls if subpixel electron strike positions are expected

    Returns:
        pointlistarray      a PointListArray of electron strike events
    """
    # Get shape, initialize PointListArray
    Q_Nx,Q_Ny,R_Nx,R_Ny = counted_datacube.shape
    if subpixel:
        coordinates = [('qx',float),('qy',float)]
    else:
        coordinates = [('qx',int),('qy',int)]
    pointlistarray = PointListArray(coordinates=coordinates, shape=(R_Nx,R_Ny))

    # Loop through frames, adding electron counts to the PointListArray for each.
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            frame = counted_datacube[:,:,Rx,Ry]
            x,y = np.nonzero(frame)
            pointlist = pointlistarray.get_pointlist(Rx,Ry)
            pointlist.add_tuple_of_nparrays((x,y))

    return return pointlistarray

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1,
                                                     length = 100, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()



if __name__=="__main__":

    from py4DSTEM.process.preprocess import get_darkreference

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
    datacube = dm.dmReader(dm4filename,dSetNum=0,verbose=False)['data']
    datacube = np.moveaxis(datacube,(0,1),(2,3))

    # Get dark reference
    darkreference = 1 # TODO: get_darkreference(datacube = ...!
                      # What we really need now is to tie the memory mapping in seamlessly with
                      # DataCube.data4D, so that whether our data is memory mapped, we just
                      # reference this as a numpy array.

    counted = count_datacube(datacube, darkreference, Nsamples=Nsamples,
                                       thresh_bkgrnd_Nsigma=thresh_bkgrnd_Nsigma,
                                       thresh_xray_Nsigma=thresh_xray_Nsigma,
                                       binfactor=binfactor,
                                       sub_pixel=True,
                                       output='pointlist'):

    # For outputting datacubes, wrap counted into a py4DSTEM DataCube
    if output=='datacube':
        counted = DataCube(data=counted)

    output_path = dm4filename.replace('.dm4','.h5')
    save(counted, output_path)



