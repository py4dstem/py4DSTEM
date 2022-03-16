# Make a sample HDF5 file in the py4DSTEM format using v0.12

import py4DSTEM
import numpy as np

# Decide whether to visualize outputs
vis = True

# Set the filepath
fpath_datacube = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.dm3"
fpath_h5 = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/sample_h5_v12.h5"

# Load a datacube
datacube = py4DSTEM.io.read(fpath_datacube)
datacube.set_scan_shape(10,10)

# Get the max DP
max_dp = np.max(datacube.data,axis=(0,1))

# Select a few scan positions
positions = [
    [0,0],
    [3,3],
    [6,8]
]
N = len(positions)

# Get a few diffraction patterns
dps = np.zeros((datacube.Q_Nx,datacube.Q_Ny,N))
for i in range(N):
    rx,ry = positions[i]
    dps[:,:,i] = datacube.data[rx,ry,:,:]

# Get a few virtual images
center = datacube.Q_Nx//2,datacube.Q_Ny//2
detector_circ = (center,20)
detector_ann = (center,(80,160))

im1 = py4DSTEM.process.virtualimage.get_virtualimage(datacube,detector_circ)
im2 = py4DSTEM.process.virtualimage.get_virtualimage(datacube,detector_ann)

# Get a probe and correlation kernel
ROI = np.zeros((datacube.R_Nx,datacube.R_Ny),dtype=bool)
ROI[6:,6:] = True
probe = py4DSTEM.process.diskdetection.get_probe_from_4Dscan_ROI(datacube,ROI)
alpha,_,_ = py4DSTEM.process.calibration.get_probe_size(probe)
probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(probe,ri=0,ro=4*alpha)

# Find bragg scattering
braggpeaks = py4DSTEM.process.diskdetection.find_Bragg_disks(
    datacube,
    probe_kernel
)

# Create data objects for saving
maximum_diffraction_image = py4DSTEM.io.DiffractionSlice(
    max_dp,
    name='max_dp'
)
diffraction_images = py4DSTEM.io.DiffractionSlice(
    dps,
    slicelabels=['0','1','2'],
    name='diffraction_images'
)
probe = py4DSTEM.io.DiffractionSlice(
    np.dstack(
        [probe,
         probe_kernel]
    ),
    slicelabels=['probe','probe_kernel'],
    name='probe'
)
BF = py4DSTEM.io.RealSlice(
    im1,
    name='BF image'
)
ADF = py4DSTEM.io.RealSlice(
    im2,
    name='ADF image'
)
coords = [('rx',int),('ry',int)]
pos_data = np.zeros(N,dtype=coords)
pos_data['rx'] = np.array(positions)[:,0]
pos_data['ry'] = np.array(positions)[:,1]
pos = py4DSTEM.io.PointList(
    data=pos_data,
    coordinates=coords,
    name='positions'
)
braggpeaks.name = 'braggpeaks'
datacube.name = 'datacube'

# Save
py4DSTEM.io.save(
    fpath_h5,
    data=[
        datacube,
        braggpeaks,
        pos,
        ADF,
        BF,
        probe,
        diffraction_images,
        maximum_diffraction_image
    ],
    overwrite=True
)

# Visualize
if vis:

    import matplotlib.pyplot as plt
    from py4DSTEM.visualize import show,show_image_grid,show_kernel

    # Max DP
    show(maximum_diffraction_image.data,
         scaling='log')

    # Bright field detector/image
    fig,axs = plt.subplots(1,2,figsize=(8,4))
    show(
        maximum_diffraction_image.data,
        scaling='log',
        circle={
           'center':detector_circ[0],
           'R':detector_circ[1],
           'color':'r',
           'fill':True,
           'alpha':0.3
        },
        figax=(fig,axs[0])
    )
    show(BF.data,
         figax=(fig,axs[1]))
    plt.show()

    # ADF detector/image
    fig,axs = plt.subplots(1,2,figsize=(8,4))
    show(
        maximum_diffraction_image.data,
        scaling='log',
        annulus={
           'center':detector_ann[0],
           'Ri':detector_ann[1][0],
           'Ro':detector_ann[1][1],
           'color':'r',
           'fill':True,
           'alpha':0.3
        },
        figax=(fig,axs[0])
    )
    show(ADF.data,
         figax=(fig,axs[1]))
    plt.show()

    # Selected scan positions
    show(
        BF.data,
        points={
        'x':pos.data['rx'],
        'y':pos.data['ry']
        }
    )

    # Selected diffraction images
    show_image_grid(
        lambda i:diffraction_images.data[:,:,i],
        H=1,W=3,
        scaling='log'
    )

    # Probe kernel
    show_kernel(probe.slices['probe_kernel'],100,250,5)

    # Disk positions
    show_image_grid(
    lambda i:diffraction_images.data[:,:,i],
    H=1,W=3,
    scaling='log',
    get_x = lambda i:braggpeaks.get_pointlist(
        pos.data['rx'][i],pos.data['ry'][i]).data['qx'],
    get_y = lambda i:braggpeaks.get_pointlist(
        pos.data['rx'][i],pos.data['ry'][i]).data['qy'],
    open_circles=True,
    scale=1000
    )




