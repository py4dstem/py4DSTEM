"""
test_basic.py

######## DESCRIPTION ########
This script tests some of the basic functionality of py4DSTEM, including
visualization, disk detection, and read/write operations, by

- loading data from a dm3,
- computing some simple 2D data slices in real and diffraction space
- detecting Bragg disks

then if show_vis==True:
- displaying visualizations

and if save_vis==True:
- saving visualizations

and if test_io==True:
- generating each type of py4DSTEM DataObject,
- saving to an .h5,
- loading the objects from the .h5,
- and checking that the original and re-loaded objects are the same.


######## DATA ########
The 4DSTEM data was collected by Steven Zeltmann.
To download the data, please go to:

https://drive.google.com/file/d/1B-xX3F65JcWzAg0v7f1aVwnawPIfb5_o/view?usp=sharing

then update the filepaths at the beginning of the script accordingly.


######## VERSION INFO ########
Last updated on 2019-11-11 with py4DSTEM version 0.9.18.
"""

### PARAMETERS ###

# Filepaths
filepath_input = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.dm3"
filepath_output = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.h5"
dirpath_plots = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/test"

# BF detector
x0_BF,y0_BF = 242,272
R_BF = 50

# Sample DP
rx,ry = 2,5

# Grid of DPs
x0_grid,y0_grid = 3,1
xL_grid,yL_grid = 3,3

# Subregions to generate max DP
rx_min_1,rx_max_1 = 0,8
ry_min_1,ry_max_1 = 0,2
rx_min_2,rx_max_2 = 1,3
ry_min_2,ry_max_2 = 3,7
rx_min_3,rx_max_3 = 5,8
ry_min_3,ry_max_3 = 3,6
colors_maxdps = ['b','magenta','r']

# Sample DPs for disk detection
rxs = 3,3,3
rys = 0,4,7
colors_dps=['r','b','g']

# Disk detection params
corrPower=1
sigma=2
edgeBoundary=20
minRelativeIntensity=0.005
relativeToPeak=0
minPeakSpacing=60
maxNumPeaks=70
subpixel='multicorr'
upsample_factor=16

# Generate and show/save plots of analysis?
show_vis = False
save_vis = False
test_io = True

### END PARAMETERS ###



import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM

# Load the data
datacube = py4DSTEM.io.read(filepath_input)
datacube.set_scan_shape(10,10)
datacube.crop_data_real(2,10,2,10)

# Perform some analysis
max_dp = np.max(datacube.data, axis=(0,1))
BF_image = py4DSTEM.process.virtualimage.get_virtualimage_circ(
                                                 datacube,x0_BF,y0_BF,R_BF)
max_dp_1 = np.max(datacube.data[rx_min_1:rx_max_1,ry_min_1:ry_max_1,:,:],
                                                                axis=(0,1))
max_dp_2 = np.max(datacube.data[rx_min_2:rx_max_2,ry_min_2:ry_max_2,:,:],
                                                                axis=(0,1))
max_dp_3 = np.max(datacube.data[rx_min_3:rx_max_3,ry_min_3:ry_max_3,:,:],
                                                                axis=(0,1))
probe = py4DSTEM.process.diskdetection.get_probe_from_vacuum_4Dscan(datacube)
probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_subtrgaussian(probe,sigma_probe_scale=2)
dp1 = datacube.data[rxs[0],rys[0],:,:]
dp2 = datacube.data[rxs[1],rys[1],:,:]
dp3 = datacube.data[rxs[2],rys[2],:,:]
disks_selected = py4DSTEM.process.diskdetection.find_Bragg_disks_selected(
                        datacube,probe_kernel,rxs,rys,corrPower=corrPower,
                        sigma=sigma,edgeBoundary=edgeBoundary,
                        minRelativeIntensity=minRelativeIntensity,
                        relativeToPeak=relativeToPeak,
                        minPeakSpacing=minPeakSpacing,maxNumPeaks=maxNumPeaks,
                        subpixel=subpixel,upsample_factor=upsample_factor)
disks = py4DSTEM.process.diskdetection.find_Bragg_disks(datacube,probe_kernel,
                        corrPower=corrPower,sigma=sigma,edgeBoundary=edgeBoundary,
                        minRelativeIntensity=minRelativeIntensity,
                        relativeToPeak=relativeToPeak,
                        minPeakSpacing=minPeakSpacing,maxNumPeaks=maxNumPeaks,
                        subpixel=subpixel,upsample_factor=upsample_factor)
braggvectormap = py4DSTEM.process.diskdetection.get_bragg_vector_map(
                                        disks,datacube.Q_Nx,datacube.Q_Ny)


# Show plots
if show_vis:
    py4DSTEM.visualize.show(max_dp,0,2)
    py4DSTEM.visualize.show_circ(max_dp,0,2,center=(x0_BF,y0_BF),R=R_BF,alpha=0.25)
    py4DSTEM.visualize.show(BF_image,contrast='minmax')
    py4DSTEM.visualize.show_points(BF_image,rx,ry,contrast='minmax',figsize=(6,6))
    py4DSTEM.visualize.show(datacube.data[rx,ry,:,:],0,2,figsize=(6,6))
    py4DSTEM.visualize.show_grid_overlay(BF_image,x0_grid,y0_grid,xL_grid,yL_grid,
                             contrast='minmax',color='k',linewidth=5,figsize=(8,8))
    py4DSTEM.visualize.show_DP_grid(datacube,x0_grid,y0_grid,xL_grid,yL_grid,
                            min=0,max=2,bordercolor='k',borderwidth=5,axsize=(4,4))
    py4DSTEM.visualize.show_rect(BF_image,contrast='minmax',color=colors_maxdps,
          fill=False,linewidth=4,lims=[(rx_min_1,rx_max_1,ry_min_1,ry_max_1),
                                       (rx_min_2,rx_max_2,ry_min_2,ry_max_2),
                                       (rx_min_3,rx_max_3,ry_min_3,ry_max_3)])
    py4DSTEM.visualize.show_image_grid(lambda i: [max_dp_1,max_dp_2,max_dp_3][i],
                                       1,3,axsize=(4,4),cmap='gray',min=0.5,max=2,
                                       get_bc=lambda i: colors_maxdps[i])
    py4DSTEM.visualize.show(probe,0,10)
    py4DSTEM.visualize.show_kernel(probe_kernel,R=100,L=200,W=5)
    py4DSTEM.visualize.show_points(BF_image,contrast='minmax',x=rxs,y=rys,
                                   point_color=colors_dps)
    py4DSTEM.visualize.show_image_grid(lambda i:[dp1,dp2,dp3][i],1,3,min=0.5,max=2,
                                       axsize=(5,5),get_bc=lambda i:colors_dps[i])
    py4DSTEM.visualize.show_image_grid(lambda i:[dp1,dp2,dp3][i],1,3,min=0.5,max=2,
                               axsize=(5,5),
                               get_bc=lambda i:colors_dps[i],
                               get_x=lambda i:disks_selected[i].data['qx'],
                               get_y=lambda i:disks_selected[i].data['qy'],
                               #get_s=lambda i:disks_selected[i].data['intensity'],
                               get_pointcolors=lambda i:colors_dps[i])
    py4DSTEM.visualize.show(braggvectormap,0,2,cmap='viridis')


# Save .pdfs of plots
if save_vis:
    from pathlib import Path
    import matplotlib.pyplot as plt
    from os.path import exists
    from os import mkdir
    dpath = Path(dirpath_plots)
    if not exists(dpath):
        mkdir(dpath)

    f,a = py4DSTEM.visualize.show(max_dp,0,2,returnfig=True)
    plt.savefig(dpath/Path('max_DP.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_circ(max_dp,0,2,center=(x0_BF,y0_BF),R=R_BF,
                                                    alpha=0.25,returnfig=True)
    plt.savefig(dpath/Path('BF_detector.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show(BF_image,contrast='minmax',returnfig=True)
    plt.savefig(dpath/Path('BF_image.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_points(BF_image,rx,ry,contrast='minmax',
                                                figsize=(6,6),returnfig=True)
    plt.savefig(dpath/Path('sample_DP_scanposition.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show(datacube.data[rx,ry,:,:],0,2,figsize=(6,6),
                                                              returnfig=True)
    plt.savefig(dpath/Path('sample_DP.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_grid_overlay(BF_image,x0_grid,y0_grid,
                            xL_grid,yL_grid,contrast='minmax',color='k',
                            linewidth=5,figsize=(8,8),returnfig=True)
    plt.savefig(dpath/Path('DP_grid_imageoverlay.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_DP_grid(datacube,x0_grid,y0_grid,
                            xL_grid,yL_grid,min=0,max=2,bordercolor='k',
                            borderwidth=5,axsize=(4,4),returnfig=True)
    plt.savefig(dpath/Path('DP_grid.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_rect(BF_image,contrast='minmax',
                            color=colors_maxdps,fill=False,linewidth=4,
                            lims=[(rx_min_1,rx_max_1,ry_min_1,ry_max_1),
                                  (rx_min_2,rx_max_2,ry_min_2,ry_max_2),
                                  (rx_min_3,rx_max_3,ry_min_3,ry_max_3)],
                            returnfig=True)
    plt.savefig(dpath/Path('max_DP_subregions_imageoverlay.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_image_grid(
                                lambda i: [max_dp_1,max_dp_2,max_dp_3][i],
                                1,3,axsize=(4,4),cmap='gray',min=0.5,max=2,
                                get_bc=lambda i: colors_maxdps[i],
                                returnfig=True)
    plt.savefig(dpath/Path('max_DP_subregions.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show(probe,0,10,returnfig=True)
    plt.savefig(dpath/Path('probe.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_kernel(probe_kernel,R=100,L=200,W=5,
                                                        returnfig=True)
    plt.savefig(dpath/Path('probe_kernel.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_points(BF_image,contrast='minmax',x=rxs,y=rys,
                                   point_color=colors_dps,returnfig=True)
    plt.savefig(dpath/Path('selected_DPs_imageoverlay.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_image_grid(
                                lambda i:[dp1,dp2,dp3][i],1,3,min=0.5,max=2,
                                axsize=(5,5),get_bc=lambda i:colors_dps[i],
                                returnfig=True)
    plt.savefig(dpath/Path('selected_DPs.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show_image_grid(
                                lambda i:[dp1,dp2,dp3][i],1,3,min=0.5,max=2,
                                axsize=(5,5),
                                get_bc=lambda i:colors_dps[i],
                                get_x=lambda i:disks_selected[i].data['qx'],
                                get_y=lambda i:disks_selected[i].data['qy'],
                                #get_s=lambda i:disks_selected[i].data['intensity'],
                                get_pointcolors=lambda i:colors_dps[i],
                                returnfig=True)
    plt.savefig(dpath/Path('detected_Bragg_disks_selected_DPs.pdf'))
    plt.close()

    f,a = py4DSTEM.visualize.show(braggvectormap,0,2,cmap='viridis',
                                                      returnfig=True)
    plt.savefig(dpath/Path('Bragg_vector_map.pdf'))
    plt.close()


# i/o testing
if test_io:

    # Save an .h5 file
    max_dp_DiffSlice = py4DSTEM.datastructure.DiffractionSlice(
                                    data=max_dp,name='max_dp')
    BF_image_RealSlice = py4DSTEM.datastructure.RealSlice(
                                    data=BF_image,name='BF_image')
    three_dps = py4DSTEM.datastructure.DiffractionSlice(
                                    data=np.dstack([dp1,dp2,dp3]),
                                    slicelabels=['rx,ry={},{}'.format(rxs[i],rys[i]) for i in range(len(rxs))],
                                    name='three_dps')
    probe_DiffSlice = py4DSTEM.datastructure.DiffractionSlice(
                                    data=np.dstack([probe,probe_kernel]),
                                    slicelabels=['probe','probe_kernel'],
                                    name='probe')
    dp3_disks = disks_selected[2]
    dp3_disks.name = 'some_bragg_disks'
    disks.name = 'braggpeaks'
    datacube.name = '4ddatacube'

    data = [max_dp_DiffSlice,BF_image_RealSlice,three_dps,dp3_disks,
                                        probe_DiffSlice,disks,datacube]
    py4DSTEM.io.native.save(filepath_output,data,overwrite=True)
#    py4DSTEM.io.native.save(filepath_output,[],overwrite=True)
#    py4DSTEM.io.native.append(filepath_output,data,overwrite=True)

    # Load from the .h5
    max_dp_h5 = py4DSTEM.io.read(filepath_output,data_id='max_dp')
    max_dp_h5 = max_dp_h5.data

    BF_image_h5 = py4DSTEM.io.read(filepath_output,data_id='BF_image')
    BF_image_h5 = BF_image_h5.data

    three_dps_h5 = py4DSTEM.io.read(filepath_output,data_id="three_dps")
    three_dps_h5 = three_dps_h5.data
    dp1_h5,dp2_h5,dp3_h5 = [three_dps_h5[:,:,i] for i in range(3)]

    probe_DiffSlice_h5 = py4DSTEM.io.read(filepath_output,
                                                 data_id='probe')
    probe_h5 = probe_DiffSlice_h5.slices['probe']
    probe_kernel_h5 = probe_DiffSlice_h5.slices['probe_kernel']

    dp3_disks_h5 = py4DSTEM.io.read(filepath_output,
                                           data_id="some_bragg_disks")

    disk_h5 = py4DSTEM.io.read(filepath_output,data_id="braggpeaks")

    datacube_h5 = py4DSTEM.io.read(filepath_output,data_id='4ddatacube')

    # Confirm that saved and loaded data are identical
    print('Running io tests...')
    assert(np.sum(max_dp_h5-max_dp)==0)
    assert(np.sum(BF_image_h5-BF_image)==0)
    for dp_h5,dp in zip((dp1_h5,dp2_h5,dp3_h5),(dp1,dp2,dp3)):
        assert(np.sum(dp_h5-dp)==0)
    for dp_h5,dp in zip((probe_h5,probe_kernel_h5),(probe,probe_kernel)):
        assert(np.sum(dp_h5-dp)==0)
    for key in dp3_disks.data.dtype.names:
        assert(np.sum(dp3_disks_h5.data[key]-dp3_disks.data[key])==0)
    assert(np.sum(datacube_h5.data-datacube.data)==0)
    print('All tests passed!')




