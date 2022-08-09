# Make a sample HDF5 file in the py4DSTEM format using v0.12

import py4DSTEM
import numpy as np

# Decide whether to visualize outputs
vis = True

# Set the filepath
fpath_h5 = "/home/ben/Work/Data/py4DSTEM_sampleData/smallDataSets/sample_h5_v12.h5"

# Read the data
maximum_diffraction_image = py4DSTEM.io.read(fpath_h5,data_id='max_dp')
diffraction_images = py4DSTEM.io.read(fpath_h5,data_id='diffraction_images')
probe = py4DSTEM.io.read(fpath_h5,data_id='probe')
BF = py4DSTEM.io.read(fpath_h5,data_id='BF image')
ADF = py4DSTEM.io.read(fpath_h5,data_id='ADF image')
pos = py4DSTEM.io.read(fpath_h5,data_id='positions')
braggpeaks = py4DSTEM.io.read(fpath_h5,data_id='braggpeaks')
datacube = py4DSTEM.io.read(fpath_h5,data_id='datacube')

# Define detectors
center = datacube.Q_Nx//2,datacube.Q_Ny//2
detector_circ = (center,20)
detector_ann = (center,(80,160))

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




