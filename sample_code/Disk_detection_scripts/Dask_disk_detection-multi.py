import py4DSTEM
import matplotlib.pyplot as plt 
import numpy as np
import h5py
import dask.array as da
from dask.distributed import Client
import dill
import dask
from distributed.protocol.serialize import register_serialization_family
import time
# configure dask to avoid spilling to disk
dask.config.set({'distributed.worker.memory.spill': False,
                'distributed.worker.memory.target': False})


# define dill based serialisers
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

def main():
    

    # register the serializer dill # not sure if this is nesercary everytime or just once
    register_serialization_family('dill', dill_dumps, dill_loads)
  
    print("loading data")

    # load the data
    
    file_path_input = 'Ge_SiGe_ML_ideal.h5'
    stack_pointer = '4DSTEM_simulation/data/datacubes/CBED_array_depth0000/data'

    file = h5py.File(file_path_input, 'r')

    dask_data = da.from_array(file[stack_pointer][...],
                              chunks=(1, 1, 420, 422))
    dataset = py4DSTEM.io.DataCube(data = dask_data)
    dataset.stack_pointer = file[stack_pointer]

    # Probe is a single 2D Image
    probe = py4DSTEM.io.read(file_path_input, data_id='probe');

    # Estimate the radius of the BF disk, and the center coordinates
    probe_semiangle, qx0, qy0 = py4DSTEM.process.calibration.get_probe_size(
        probe.data)
    # Generate the probe kernel
    probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(
        probe.data, 
        probe_semiangle * 0.0,                                                        
        probe_semiangle * 2.0,)
    # generate Fourier Transform of the probe 
    probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))
    
    print("starting client")
    
    # start dask client, leaving to dask automagic to configure the LocalCluster. 
    # we're looking into optimizing dask client parameters, e.g.
    # processes vs threads, 
    # nworkers:threads_per worker  
    client = Client()
    
    # set parameters
    disk_detect_params = {
        'minRelativeIntensity' : 0,
        'minAbsoluteIntensity' : 1e-6,
        'edgeBoundary' : 4,
        'minPeakSpacing' : 20,
        'subpixel' : 'multicorr',  # slower but more precise method
        'upsample_factor' : 32,
        'maxNumPeaks': 100,
    }
    
    
    print("starting disk detection")
    start = time.time()
    
    peaks = py4DSTEM.process.diskdetection.beta_parallel_disk_detection(
            dataset,
            probe_kernel_FT,
            dask_client = client, 
            return_dask_client=False,
            **disk_detect_params)
    
    end = time.time()

    run_time = end - start
    print(run_time) 
    
    file.close()
    return run_time, peaks

if __name__ == '__main__':
    main()
