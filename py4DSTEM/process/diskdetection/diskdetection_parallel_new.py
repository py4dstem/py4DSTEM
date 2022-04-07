import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask
#import dask.bag as db
from py4DSTEM.io.datastructure import PointListArray, PointList
from .diskdetection import _find_Bragg_disks_single_DP_FK
from py4DSTEM.io import PointListArray, PointList, datastructure
import time
from dask.diagnostics import ProgressBar
import dill
#import distributed
import dask.config
from distributed.protocol.serialize import register_serialization_family
import distributed


#### SERIALISERS ####
# Define Serialiser 
# these are functions which allow the hdf5 objects to be passed. May not be required anymore 

def dill_dumps(x):
    header = {'serializer': 'dill'}
    frames = [dill.dumps(x)]
    return header, frames

def dill_loads(header, frames):
    if len(frames) > 1:
        frame = ''.join(frames)
    else:
        frame = frames[0]
        
    return dill.loads(frame)
# register the serialization method 
#register_serialization_family('dill', dill_dumps, dill_loads)

def register_dill_serializer():
    """
    This function registers the dill serializer allowing dask to work on h5py objects. 
    Not sure if this needs to be run and how often this need to be run. Keeping this in for now. 
    Args:
        None
    Returns:
        None
    """
    register_serialization_family('dill', dill_dumps, dill_loads)
    return None


#### END OF SERAILISERS ####


#### DASK WRAPPER FUNCTION #### 

# Each delayed objected is passed a 4D array, currently implementing only on 2D slices. 
# TODO add batching with fancy indexing  - needs to run a for loop over the batch of arrays
# TODO add cuda accelerated version 
# TODO add ML-AI version 
def _find_Bragg_disks_single_DP_FK_dask_wrapper(arr, *args,**kwargs):
    # THis is needed as _find_Bragg_disks_single_DP_FK takes 2D array these arrays have the wrong shape
    return _find_Bragg_disks_single_DP_FK(arr[0,0], *args, **kwargs)


#### END OF DASK WRAPPER FUNCTIONS ####



#### MAIN FUNCTION
# TODO add batching with fancy indexing - needs batch size, fancy indexing method 
# TODO add cuda accelerated function - needs dask GPU cluster. 

def beta_parallel_disk_detection(dataset, 
                            probe,
                            #rxmin=None, # these would allow selecting a sub section 
                            #rxmax=None,
                            #rymin=None,
                            #rymax=None,
                            #qxmin=None,
                            #qxmax=None,
                            #qymin=None,
                            #qymax=None,
                            probe_type="FT",
                            dask_client= None,
                            dask_client_params:dict=None,
                            restart_dask_client=True,
                            close_dask_client=False,
                            return_dask_client=True,
                            *args, **kwargs):
    """
    This is not fully validated currently so may not work, please report bugs on the py4DSTEM github page. 

    This parallellises the disk detetection for all probe posistions. This can operate on either in memory or out of memory datasets 
    
    There is an asumption that unless specifying otherwise you are parallelising on a single Local Machine. 
    If this is not the case its probably best to pass the dask_client into the function, although you can just pass the required arguments to dask_client_params.
    If no dask_client arguments are passed it will create a dask_client for a local machine 
    
    Note:
        Do not pass "peaks" argument as a kwarg, like you might in "_find_Bragg_disks_single_DP_FK", as the results will be unreliable and may cause the calculation to crash.
    Args:
        dataset (py4dSTEM datacube): 4DSTEM dataset
        probe (ndarray): can be regular probe kernel or fourier transormed
        probe_type (str): "FT" or None 
        dask_client (distributed.client.Client): dask client
        dask_client_params (dict): parameters to pass to dask client or dask cluster
        restart_dask_client (bool): if True, function will attempt to restart the dask_client.
        close_dask_client (bool): if True, function will attempt to close the dask_client.
        return_dask_client (bool): if True, function will return the dask_client.
        *args,kwargs will be passed to "_find_Bragg_disks_single_DP_FK" e.g. corrPower, sigma, edgeboundary...

    Returns:
        peaks (PointListArray): the Bragg peak positions and the correlenation intensities
        dask_client(optional) (distributed.client.Client): dask_client for use later.
    """
    #TODO add asserts abotu peaks not being passed
    # Dask Client stuff
    #TODO how to guess at default params for client, sqrt no.cores.  Something to do with the size of the diffraction patterm
    # write a function which can do this. 
    #TODO replace dask part with a with statement for easier clean up e.g.
    # with LocalCluser(params) as cluster, Client(cluster) as client: 
    #   ... dask stuff. 
    #TODO add assert statements and other checks. Think about reordering opperations
    
    if dask_client == None: 
        if dask_client_params !=None:

            dask.config.set({'distributed.worker.memory.spill': False,
                'distributed.worker.memory.target': False}) 
            cluster = LocalCluster(**dask_client_params)
            dask_client = Client(cluster, **dask_client_params)
        else:
            # AUTO MAGICALLY SET?
            # LET DASK SET?
            # HAVE A FUNCTION WHICH RUNS ON A SUBSET OF THE DATA TO PICK OPTIMIAL VALUE?
            # psutil could be used to count cores. 
            dask.config.set({'distributed.worker.memory.spill': False, # stops spilling to disk
                'distributed.worker.memory.target': False}) # stops spilling to disk and erroring out
            cluster = LocalCluster()
            dask_client = Client(cluster)

    else:
        assert type(dask_client) == distributed.client.Client
        if restart_dask_client:
            try:
                dask_client.restart()
            except Exception as e:
                print('Could not restart dask client. Try manually restarting outside or passing "restart_dask_client=False"') # WARNING STATEMENT
                return e 
        else:
            pass


    # Probe stuff
    assert (probe.shape == dataset.data.shape[2:]), "Probe and Diffraction Pattern Shapes are Mismatched"
    if probe_type != "FT":
    #TODO clean up and pull out redudant parts
    #if probe.dtype != (np.complex128 or np.complex64 or np.complex256):
        #DO FFT SHIFT THING
        probe_kernel_FT = np.conj(np.fft.fft2(probe))
        dask_probe_array = da.from_array(probe_kernel_FT, chunks=(dataset.Q_Nx, dataset.Q_Ny))
        dask_probe_delayed = dask_probe_array.to_delayed()
        # delayed_probe_kernel_FT = delayed(probe_kernel_FT)
    else:
        probe_kernel_FT = probe
        dask_probe_array = da.from_array(probe_kernel_FT, chunks=(dataset.Q_Nx, dataset.Q_Ny))
        dask_probe_delayed = dask_probe_array.to_delayed()

    # GET DATA 
    #TODO add another elif if it is a dask array then pass
    if type(dataset.data) == np.ndarray:
        dask_data = da.from_array(dataset.data, chunks=(1, 1,dataset.Q_Nx, dataset.Q_Ny))
    elif dataset.stack_pointer != None:
        dask_data = da.from_array(dataset.stack_pointer, chunks=(1, 1,dataset.Q_Nx, dataset.Q_Ny))
    else: 
        print("Couldn't access the data")
        return None

    # Convert the data to delayed 
    dataset_delayed = dask_data.to_delayed()
    # TODO Trim data e.g. rx,ry,qx,qy
    # I can pass the index values in here I should trim the probe and diffraction pattern first


    # Into the meat of the function 
    
    # create an empty list to which we will append the dealyed functions to. 
    res = []
    # loop over the dataset_delayed and create a delayed function of 
    for x in np.ndindex(dataset_delayed.shape):
        temp = delayed(_find_Bragg_disks_single_DP_FK_dask_wrapper)(dataset_delayed[x],
                                probe_kernel_FT=dask_probe_delayed[0,0],
                                #probe_kernel_FT=delayed_probe_kernel_FT,
                                *args, **kwargs) #passing through args from earlier or should I use 
                                #corrPower=corrPower,
                                #sigma=sigma_gaussianFilter,
                                #edgeBoundary=edgeBoundary,
                                #minRelativeIntensity=minRelativeIntensity,
                                #minPeakSpacing=minPeakSpacing,        
                                #maxNumPeaks=maxNumPeaks,
                                #subpixel='poly')
        res.append(temp)
    _temp_peaks = dask_client.compute(res, optimize_graph=True) # creates futures and starts computing 

    output = dask_client.gather(_temp_peaks) # gather the future objects 

    coords = [('qx',float),('qy',float),('intensity',float)]
    peaks = PointListArray(coordinates=coords, shape=dataset.data.shape[:-2])

    #temp_peaks[0][0]

    # operating over a list so we need the size (0->count) and re-create the probe positions (0->rx,0->ry),
    for (count,(rx, ry)) in zip([i for i in range(dataset.data[...,0,0].size)],np.ndindex(dataset.data.shape[:-2])):
        #peaks.get_pointlist(rx, ry).add_pointlist(temp_peaks[0][count])
        #peaks.get_pointlist(rx, ry).add_pointlist(output[count][0])
        peaks.get_pointlist(rx, ry).add_pointlist(output[count])

    # Clean up
    dask_client.cancel(_temp_peaks) # removes from the dask workers
    del _temp_peaks # deletes the object 
    if close_dask_client:
        dask_client.close()
        return peaks
    elif close_dask_client == False and return_dask_client == True:
        return peaks, dask_client
    elif close_dask_client and return_dask_client == False:
        return peaks
    else:
        print('Dask Client in unknown state, this may result in unpredicitable behaviour later')
        return peaks

