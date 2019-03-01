import numpy as np, matplotlib.pyplot as plt,copy,sys,os
import IPython
import torch
from scipy import optimize
from PIL import Image
from scipy.ndimage.measurements import label,center_of_mass,find_objects
from scipy.ndimage import maximum_filter,watershed_ift
import scipy.signal
import hyperspy.api as hs
import h5py
import dm
import matplotlib as mpl

# def cshift(x: torch.Tensor, shift: int, dim: int = -1):
# 
#     if 0 == shift:
#         return x
# 
#     elif shift < 0:
#         shift = -shift
#         gap = x.index_select(dim, torch.arange(shift))
#         return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)
# 
#     else:
#         shift = x.size(dim) - shift
#         gap = x.index_select(dim, torch.arange(shift, x.size(dim)))
#         return torch.cat([gap, x.index_select(dim, torch.arange(shift))], dim=dim)
# 
# def convolve(arr1,arr2):
# 	'''Find the cross correlation of two arrays using the Fourier transform.
#     Assumes that both arrays are periodic so can be circularly shifted.'''
# 	return np.abs(ifft2(fft2(arr1,norm='ortho')*fft2(arr2,norm='ortho')
#                                                          ,norm='ortho'))
# 
# def roll_zeropad(a, shift, axis=None):
#     """
#     Roll array elements along a given axis.
# 
#     Elements off the end of the array are treated as zeros.
# 
#     Parameters
#     ----------
#     a : array_like
#         Input array.
#     shift : int
#         The number of places by which elements are shifted.
#     axis : int, optional
#         The axis along which elements are shifted.  By default, the array
#         is flattened before shifting, after which the original
#         shape is restored.
# 
#     Returns
#     -------
#     res : ndarray
#         Output array, with the same shape as `a`.
# 
#     See Also
#     --------
#     roll     : Elements that roll off one end come back on the other.
#     rollaxis : Roll the specified axis backwards, until it lies in a
#                given position.
# 
#     Examples
#     --------
#     >>> x = np.arange(10)
#     >>> roll_zeropad(x, 2)
#     array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
#     >>> roll_zeropad(x, -2)
#     array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])
# 
#     >>> x2 = np.reshape(x, (2,5))
#     >>> x2
#     array([[0, 1, 2, 3, 4],
#            [5, 6, 7, 8, 9]])
#     >>> roll_zeropad(x2, 1)
#     array([[0, 0, 1, 2, 3],
#            [4, 5, 6, 7, 8]])
#     >>> roll_zeropad(x2, -2)
#     array([[2, 3, 4, 5, 6],
#            [7, 8, 9, 0, 0]])
#     >>> roll_zeropad(x2, 1, axis=0)
#     array([[0, 0, 0, 0, 0],
#            [0, 1, 2, 3, 4]])
#     >>> roll_zeropad(x2, -1, axis=0)
#     array([[5, 6, 7, 8, 9],
#            [0, 0, 0, 0, 0]])
#     >>> roll_zeropad(x2, 1, axis=1)
#     array([[0, 0, 1, 2, 3],
#            [0, 5, 6, 7, 8]])
#     >>> roll_zeropad(x2, -2, axis=1)
#     array([[2, 3, 4, 0, 0],
#            [7, 8, 9, 0, 0]])
# 
#     >>> roll_zeropad(x2, 50)
#     array([[0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0]])
#     >>> roll_zeropad(x2, -50)
#     array([[0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0]])
#     >>> roll_zeropad(x2, 0)
#     array([[0, 1, 2, 3, 4],
#            [5, 6, 7, 8, 9]])
# 
#     """
#     a = np.asanyarray(a)
#     if shift == 0: return a
#     if axis is None:
#         n = a.size
#         reshape = True
#     else:
#         n = a.shape[axis]
#         reshape = False
#     if np.abs(shift) > n:
#         res = np.zeros_like(a)
#     elif shift < 0:
#         shift += n
#         zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
#         res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
#     else:
#         zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
#         res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
#     if reshape:
#         return res.reshape(a.shape)
#     else:
#         return res

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


def get_dark_reference(dm4filenametemplate,ndrsamples=20,upperlimit=5,drwidth=100):
    '''
    For a 4D-STEM datacube subtract the dark reference

    Note that output should be cast to fit detector/array shape, e.g. dr[:,np.newaxis]
    '''
    #Get dimensions
    y,x,z,t = np.shape(datacube)
    nframes = z*t

    #Sum a set of random frames from the data to create
    #the background reference
    rand = np.random.randint(0,nframes-1,min(nframes,ndrsamples))
    refframe = np.zeros((y,x),dtype=np.int16)
    mean = 0
    var=0
    for i in range(min(nframes,ndrsamples)):
        frame = datacube[:,:,rand[i]//z,rand[i]%t]
        refframe += frame
        mean += np.mean(frame)
        var += np.var(frame)
    mean/=min(nframes,ndrsamples)
    var/=min(nframes,ndrsamples)
    #TODO: make GUI to choose region to take dark reference from
    #Create dark reference image by choosing a strip of the reference
    #image near the boundary, projecting this horizontally (axis 1)
    #and summing.
    #Normalisation by the width of the reference region and the
    #number of sample images necessary.
    dr = np.sum(refframe[:,:drwidth],axis=1)//drwidth//min(nframes,ndrsamples)
    # dr = np.median(refframe[:,:drwidth],axis=1)//min(z,ndrsamples)
    # print(np.shape(dr),drwidth)
    # plt.show(plt.plot(dr))

    return dr,mean,np.sqrt(var)

# def electron_list_to_array(list,array_size):
#     '''For a list of electron positions as a fractional
#        if the original pixel dimensions of the camera
#        create a 2D image of the diffraction pattern for
#        arbitrary array size array_size.'''
#     nelectrons = np.shape(list)[0]
#     array_out = np.zeros(array_size,dtype=np.int)
#     sze = np.asarray(array_size)
#     for i in range(nelectrons):
#         y,x = [int(x) for x in list[i,:2]*sze]
#         array_out[y,x] += 1
#     return array_out

def calculate_counting_threshhold(datacube,darkreference,sigmathresh=4,nsamples=20,
                                  upper_limit=10,plot_histogram=False):
    """For a datacube calculate the threshhold for an electron count to be
       registered."""
    y,x,z,t = datacube.shape
    nframes = z*t
    samples = np.random.randint(0,nframes,size=nsamples)

    #This flattens the array, it shouldn't make a copy of it
    sample = np.zeros((y,x,nsamples),dtype=np.int16)

    for i in range(nsamples):
        sample[:,:,i] = datacube[:,:,samples[i]//z,samples[i]%t]
        sample[:,:,i] -= darkreference[:,np.newaxis]
        # sample[:,:,i] = scipy.signal.convolve(sample[:,:,i],np.ones((3,3),dtype=np.int16),'same')

    sample  = np.ravel(sample)

    #Remove X-rays
    mean = np.mean(sample)
    stddev = np.std(sample)

    #zero all X-ray counts
    sample[sample>(mean+upper_limit*stddev)]=mean
    sample[sample<(mean-upper_limit*stddev)]=mean

    print(np.floor(np.amin(sample)),np.ceil(np.amax(sample)))
    binmax = min(int(np.ceil(np.amax(sample))),int(mean+sigmathresh*stddev))
    binmin = max(int(np.ceil(np.amin(sample))),int(mean-sigmathresh*stddev))
    step = max(1,(binmax-binmin)//1000)
    #Since the data is 16 bit integer choosing bins is easy
    bins = np.arange(binmin,binmax,step =step,dtype=np.int)
    #Fit background
    n, bins = np.histogram(sample, bins=bins)

    #Now fit Gaussian function to the histogram
    #First define Gaussian function, list p contains parameters
    #p[0] is amplitude of Gaussian
    #p[1] is centre of Gaussian
    #p[2] is std deviation of Gaussian
    fitfunc = lambda p, x: p[0]*np.exp(-0.5*np.square((x-p[1])/p[2]))
    #Define the error routine for scipy's optimize routine
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    #Initial guess of Gaussian function parameters
    p0 = [n.max(),(bins[n.argmax()+1]-bins[n.argmax()])/2,np.std(sample)]
    #Use the scipy optimize routine
    p1,success = optimize.leastsq(errfunc,p0[:],args=(bins[:-1],n))
    #Add a half to account for integer bin width
    p1[1] += 0.5

    #Threshhold for an electron count to be considered an electron count
    thresh = p1[1]+p1[2]*sigmathresh

    #If desired, plot histogram with line of best fit
    if(plot_histogram):
        n, bins, patches = plt.hist(x=sample, bins=bins,
        color='#0504aa',
        alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of K2 intensity readout')
        maxfreq = n.max()
        stddev =np.std(sample)
        (plt.xlim( xmin = bins[n.argmax()] - 5*stddev,
        xmax = bins[n.argmax()] + 5*stddev))
        ymax = np.ceil(maxfreq / 10) * 10
        plt.ylim(ymax=ymax)
        plt.plot(bins[:-1],fitfunc(p1,bins[:-1]),color='red')
        plt.plot([thresh,thresh],[0.0,ymax],'k--')
        plt.text(thresh-0.5,ymax/2,'e$^{-}$ count\nthreshhold',ha='right')
        plt.savefig('Intensity_histogram.pdf')
        plt.show()


    return thresh

# def indices(n): return np.arange(-n//2+n%2,n//2+n%2)

def torch_bin(array,device,factor=2):
    y,x =  array.shape
    biny,binx = [y//factor,x//factor]
    yy,xx = [biny*factor,binx*factor]
    result = torch.zeros(biny,binx,device=device,dtype = array.dtype)

    for iy in range(factor):
        for ix in range(factor):
            result += array[0+iy:yy+iy:factor,0+ix:xx+ix:factor]
    return result

def count_dm4(datacube,binfactor = 1,sigmathresh=4,
              nsamples=40,upperlimit=10,drwidth=100,sub_pixel=True,
              plot_histogram=False,plot_electrons=False,thresh=None):

    #Get dimensions
    y,x,z,t = np.shape(datacube)

    integrated = np.zeros((y,x),dtype=np.int)
    print('Getting dark current reference')
    #Remove dark background
    dr,mean,stddev = get_dark_reference(datacube,ndrsamples=nsamples,
                                        upperlimit=upper_limit,drwidth=drwidth)

    print('Calculating threshhold')
    #Get threshhold
    if(thresh is None): thresh = calculate_counting_threshhold(datacube,dr
                                    ,sigmathresh=sigmathresh
                                    ,nsamples=nsamples,upper_limit=upper_limit
                                    ,plot_histogram=plot_histogram)


    #
    device = torch.device('cuda')

    total_electrons =0
    #Make dark refernce array on device
    darkref = torch.from_numpy(np.zeros((y,x),dtype=np.int16)+dr[:,np.newaxis].
                                              astype(np.int16)).to(device)

    #Initialise integrated image
    #integrated = torch.from_numpy(np.zeros((y,x),dtype=np.int16)).to(device)

    #Initialised counted image
    counted = torch.ones(x//binfactor,y//binfactor,z,t,dtype=torch.short).to(device)
    for zz in range(z):
        for tt in range(t):
            printProgressBar(zz*t+tt+1, t*z, prefix = ' Counting:', suffix = 'Complete'
                                                                    , length = 50)
            #Get frame from file
            frame = datacube[:,:,zz,tt].astype(np.int16)

            #Move frame to GPU
            gframe = torch.from_numpy(frame).to(device)

            #Subtract dark reference from frame
            workingarray = gframe-darkref

            #Accumulate "integrated" intensity (ie not counted)
            integrated = integrated + workingarray

            #Threshhold electron events
            events = workingarray>thresh

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
            events[1:-1,0:-2] = events[1:-1,0:-2] & log
            log = workingarray[0:-2,0:-2]>workingarray[1:-1,1:-1]
            events[0:-2,0:-2] = events[0:-2,0:-2] & log

            if(plot_electrons):
                xwind = [900,1000]
                ywind = [900,1000]
                figsize = max(np.shape(frame)[:2])//200
                fig = plt.figure(figsize = (figsize*2,figsize*2))
                ax  = fig.add_subplot(221)
                points = np.argwhere(events.cpu().numpy())
                ax.imshow(frame,origin='lower',vmax=2*thresh
                ,cmap=plt.get_cmap('viridis'))
                ax.plot(points[:,1],points[:,0],'bo')
                ax.set_xlim(xwind)
                ax.set_ylim(ywind)
                plt.show()
            # events_ = torch.transpose(events,0,1)
            # print(events.shape,events_.shape)
            if(binfactor>1):
                counted[:,:,zz,tt] = torch.transpose(torch_bin(events.type(torch.cuda.ShortTensor),
                                            device,factor=binfactor),0,1).flip(0).flip(1)
            else:
                counted[:,:,zz,tt] =torch.transpose(events.type(torch_.cuda.ShortTensor),0,1).flip(0).flip(1)

    return counted.cpu().numpy(),integrated.cpu().numpy()

if __name__=="__main__":

    camera_lengths = np.asarray([37,46,58,73,91,115,145,185,230,285,360],
                                                            dtype=np.int)

    #filename template
    dm4filenames = ['Capture25.dm4','Capture26.dm4','Capture27.dm4','Capture28.dm4']

    #Directions for plotting -useful for enquiry
    plot_histogram = True
    plot_electrons = True

    Do_counting=True
    #If dark_subtract=True then the original
    #dm4 file will be loaded and dark subtraction
    #performed
    # dark_subtract = False

    #Perform segmentation of labelled regions according
    #to local maxima
    local_max_segmentation = True

    #Width of region from which to take Dark reference
    drwidth = 100
    #Number of samples of the dataset from which
    #to construct dark reference image and ca;culate
    #threshhold
    nsamples = 40

    #threshold for electron coutning
    #is mean + sigma * this value
    sigmathresh = 4

    #Threshhold (in units of standard deviation) with
    #which to remove x-ray counts
    upper_limit = 30


    # thresh = 10.8
    thresh = None
    # thresh = 11.4429
    #Shape of array into which the counted electrons will be downsampled



    # frames = 400
    #Load data using hyperspy
    # shape = get_frame(dm4filename,0).shape
    # datacube = np.zeros(shape+(frames,),dtype=np.int16)
    for dm4filename in dm4filenames:
        datacube = dm.dmReader(dm4filename,dSetNum=0,verbose=False)['data']
        datacube = np.moveaxis(datacube,(0,1),(2,3))

        counted,integrated = count_dm4(datacube,sigmathresh=sigmathresh,
                                      nsamples=nsamples,binfactor=6,
                                      upperlimit=upper_limit,drwidth=drwidth,
                                      sub_pixel=False,plot_histogram=plot_histogram,
                                      plot_electrons=plot_electrons,thresh=thresh)
        # Image.fromarray(np.asarray(integrated,dtype=np.float)).save('{0}_integrated.tif'.format(dm4filename.replace('.dm4','')))
        # Image.fromarray(np.asarray(np.sum(counted,axis=(2,3)),dtype=np.float)).save('{0}_counted.tif'.format(dm4filename.replace('.dm4','')))

        f = h5py.File(dm4filename.replace('.dm4','')+'.hdf5','w')
        f.attrs['version_major']=0
        f.attrs['version_minor']=1
        grp = f.create_group('/4D-STEM_data/datacube')
        grp.attrs['emd_group_type']=1
        grp.create_dataset('datacube',data=np.moveaxis(counted,(0,1),(2,3)))
        gpr = f.create_group('4D-STEM_data/metadata/original/shortlist')
        gpr = f.create_group('4D-STEM_data/metadata/original/all')
        gpr = f.create_group('4D-STEM_data/metadata/microscope')
        gpr = f.create_group('4D-STEM_data/metadata/sample')
        gpr = f.create_group('4D-STEM_data/metadata/user')
        gpr = f.create_group('4D-STEM_data/metadata/processing')
        gpr = f.create_group('4D-STEM_data/metadata/calibration')
        gpr = f.create_group('4D-STEM_data/metadata/comments')
        f.close()
# |--attr: version_major=0
# |--attr: version_minor=2
# |--grp: 4D-STEM_data
#             |
#             |--grp: datacube
#             |         |--attr: emd_group_type=1
#             |         |--data: datacube
#             |         |--data: dim1
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim2
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim3
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim4
#             |               |--attr: name="R_y"
#             |               |--attr: units="[n_m]"
