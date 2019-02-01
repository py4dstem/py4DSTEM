import numpy as np, matplotlib.pyplot as plt,copy,sys,os
from scipy import optimize
from scipy.ndimage.measurements import label,center_of_mass

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


def get_dark_reference(datacube,ndrsamples=20,upper_limit=5,drwidth=100):
    '''For a 4D-STEM datacube subtract the dark reference'''
    #Get dimensions
    y,x,z,t = np.shape(datacube)

    #Sum a set of random frames from the data to create
    #the background reference
    rand = np.random.randint(0,min(z,t),(min(z*t,ndrsamples),2))
    refframe = np.zeros((y,x),dtype=np.int16)
    mean = 0
    var=0
    for i in range(ndrsamples):
        refframe+= datacube[:,:,rand[i,0],rand[i,1]]
        mean += np.mean(datacube[:,:,rand[i,0],rand[i,1]])
        var += np.var(datacube[:,:,rand[i,0],rand[i,1]])
    mean/=ndrsamples
    var/=ndrsamples
    #TODO: make GUI to choose region to take dark reference from
    #Create dark reference image by choosing a strip of the reference
    #image near the boundary, projecting this horizontally (axis 1)
    #and summing.
    #Normalisation by the width of the reference region and the
    #number of sample images necessary.
    dr = np.sum(refframe[:,:drwidth],axis=1)//drwidth//ndrsamples

    return dr,mean,np.sqrt(var)

def electron_list_to_array(list,array_size):
    '''For a list of electron positions as a fractional
       if the original pixel dimensions of the camera
       create a 2D image of the diffraction pattern for
       arbitrary array size array_size.'''
    nelectrons = np.shape(list)[0]
    Ny,Nx = [np.amax(list[:,2]),np.amax(list[:,3])]
    array_out = np.zeros(array_size+(Ny,Nx),dtype=np.int)
    sze = np.asarray(array_size)
    for i in range(nelectrons):
        y,x = [int(x) for x in np.floor(list[i,:2]*sze)]
        z,t = [int(x) for x in list[i,2:4]]
        array_out[y,x,z,t] += 1
    return array_out

def calculate_counting_threshhold(datacube,darkreference,sigmathresh=4,nsamples=20,
                                  upper_limit=10,plot_histogram=False):
    """For a datacube calculate the threshhold for an electron count to be
       registered."""
    y,x,z,t = np.shape(datacube)
    samples = np.random.randint(0,z*t,size=nsamples)
    #This flattens the array, it shouldn't make a copy of it
    sample = np.zeros((y,x,nsamples),dtype=np.int16)
    for i in range(nsamples):
        sample[:,:,i] = np.asarray(datacube[:,:,samples[i]//t,samples[i]%t],dtype=np.int16)
        sample[:,:,i] -= darkreference[:,np.newaxis]
    sample  = np.ravel(sample)
    #Remove X-rays
    mean = np.mean(sample)
    stddev = np.std(sample)
    #zero all X-ray counts
    sample[sample>(mean+upper_limit*stddev)]=mean
    #Since the data is 16 bit integer choosing bins is easy
    bins = np.arange(np.floor(np.amin(sample))
                     ,np.ceil(np.amax(sample)),
                        dtype=np.int)

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

def count_datacube(datacube,counted_shape,sigmathresh=4,nsamples=40,upper_limit=10,
                         drwidth=100,sub_pixel=True,
                         plot_histogram=False,plot_electrons=False):
    data = datacube.data4D

    #Get dimensions
    y,x,z,t = [datacube.Q_Ny, datacube.Q_Nx,datacube.R_Ny, datacube.R_Nx]

    print('Getting dark current reference')
    #Remove dark background
    dr,mean,stddev = get_dark_reference(data,ndrsamples=nsamples,upper_limit=upper_limit,
                                        drwidth=drwidth)

    print('Calculating threshhold')
    #Get threshhold
    thresh = calculate_counting_threshhold(data,dr,sigmathresh=sigmathresh
                                    ,nsamples=nsamples,upper_limit=upper_limit
                                    ,plot_histogram=plot_histogram)

    counted = np.zeros(counted_shape+(z,t),dtype=np.uint16)

    #S
    total_electrons =0
    e_list = None

    for tt in range(t):
        printProgressBar(tt, t-1, prefix = 'Counting:', suffix = 'Complete', length = 50)
        for zz in range(z):

            #Background subtract and remove X-rays\hot pixels
            workingarray = data[:,:,zz,tt]-dr[:,np.newaxis]
            workingarray[workingarray>mean+upper_limit*stddev]=0

            #Create a map of events (pixels with value greater than
            #the threshhold value)

            events = np.greater(workingarray,thresh)

            #Now find local maxima by circular shift and comparison to neighbours
            #if we want diagonal neighbours to be considered seperate electron
            #counts then we would add a second for loop to do these extra
            #comparisons
            for i in range(4):
                events = np.logical_and(np.greater(workingarray,
                                np.roll(workingarray,i%2*2-1,axis=i//2)),events)
            events[ 0, :]=False
            events[-1, :]=False
            events[ :, 0]=False
            events[ :,-1]=False

            electron_posn = np.asarray(np.argwhere(events),dtype=np.float)
            num_electrons = np.shape(electron_posn)[0]

            if(sub_pixel):
                #Now do center of mass in local region 3x3 region to refine position
                # estimate
                for i in range(num_electrons):
                    event = electron_posn[i,:]
                    electron_posn[i,:] += center_of_mass(workingarray[int(event[0]-1)
                                                    :int(event[0]+2),int(event[1]-1)
                                                    :int(event[1]+2)])
                electron_posn -= np.asarray([1,1])[np.newaxis,:]

            if(plot_electrons):
                if(not os.path.exists(count_plots)):os.mkdir('count_plots')
                figsize = max(np.shape(data[:,:,zz,tt])[:2])//200
                fig = plt.figure(figsize = (figsize,figsize))
                ax  = fig.add_subplot(111)
                ax.imshow(data[:,:,zz,tt],origin='lower',vmax=2*thresh)
                ax.plot(electron_posn[:,1],electron_posn[:,0],'rx')
                ax.set_title('Found {0} electrons'.format(num_electrons))
                plt.show()
                fig.savefig('count_plots/Countplot_{0}_{1}.pdf'.format(tt,zz))
            #Update total number of electrons
            total_electrons += num_electrons
            #Put the electron_posn in fractional coordinates
            #where the positions are fractions of the original array
            electron_posn /= np.asarray([max(y,x),max(y,x)])[np.newaxis,:]
            electron_posn = np.hstack((electron_posn
                           ,np.asarray([tt,zz],dtype=np.float32)[np.newaxis,:]))
            if(e_list is None): e_list = electron_posn
            else:e_list = np.vstack((e_list,electron_posn))
    return electron_posn

if __name__=="__main__": pass
