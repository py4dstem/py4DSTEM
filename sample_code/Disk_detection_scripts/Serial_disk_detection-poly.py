import py4DSTEM
import numpy as np
import h5py
import time

def main():
    
    #set the file path
    file_path_input = 'Ge_SiGe_ML_ideal.h5'
    
    # load the dataset and the probe
    dataset = py4DSTEM.io.read(file_path_input, data_id=0)
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

    # set parameters
    disk_detect_params = {
        'minRelativeIntensity' : 0,
        'minAbsoluteIntensity' : 0.01,
        'edgeBoundary' : 4,
        'minPeakSpacing' : 0.45/0.0217, # 0.0217 is the pixelSizeInvAng
        'subpixel' : 'poly',   # quicker but less precise method
        'upsample_factor' : 32
    }
    
    start = time.time()
     
    serial_peaks = py4DSTEM.process.diskdetection.find_Bragg_disks(
                dataset,
                probe_kernel,
                **disk_detect_params
                )

    end = time.time()

    run_time = end - start
    print(run_time)
    return run_time, serial_peaks

if __name__ == '__main__':
    main()
