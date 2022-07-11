# Make a sample HDF5 file in the py4DSTEM format using v0.13

import py4DSTEM
import numpy as np

# Decide whether to visualize outputs
vis = True

# Set the filepath
fpath_datacube = "/home/ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.dm3"
fpath_h5 = "/home/ben/Work/Data/py4DSTEM_sampleData/smallDataSets/sample_h5_v13.h5"

# Load a datacube
datacube = py4DSTEM.io.read(fpath_datacube)
datacube.set_scan_shape(10,10)

# Get the max DP
datacube.get_max_dp()

# Select a few scan positions
positions = [
    [0,0],
    [3,3],
    [6,8]
]
N = len(positions)

# Get a few diffraction patterns
for rx,ry in positions:
    datacube.get_dp((rx,ry))

# Get a few virtual images
center = datacube.Q_Nx//2,datacube.Q_Ny//2
detector_circ = (center,20)
detector_ann = (center,(80,160))

datacube.capture_circular_detector(
    detector_circ,
    name = 'BF'
)
datacube.capture_annular_detector(
    detector_ann,
    name = 'ADF'
)

# Get a probe and correlation kernel
ROI = np.zeros((datacube.R_Nx,datacube.R_Ny),dtype=bool)
ROI[6:,6:] = True

datacube.get_probe_from_ROI(ROI)
datacube.get_probe_size()
datacube.get_probe_kernel(
    edge = 'sigmoid',
    ri = 0,
    ro = 4*datacube.calibrations.get_alpha())
)

# Find bragg scattering
datacube.find_Bragg_scattering()

# Save
py4DSTEM.io.save(
    fpath_h5,
    data=datacube,
    overwrite=True
)

# Visualize
if vis:

    # Max DP
    datacube.show('max_dp')

    # Bright field detector/image
    datacube.show(
        'BF',
        show_detector=True
    )

    # ADF detector/image
    datacube.show(
        'ADF',
        show_detector=True
    )

    # Selected scan positions
    datacube.show(
        'BF'
        points=np.array(positions)
    )

    # Selected diffraction images
    for rx,ry in positions:
        datacube.show(
            (rx,ry),
            show_scan_position=True
        )

    # Probe kernel
    datacube.show_probe_kernel(100,250,5)

    # Disk positions
    for rx,ry in positions:
        datacube.show(
            (rx,ry),
            show_disks = True
        )




