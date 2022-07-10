# Calibration workflow

import py4DSTEM
from py4DSTEM.visualize import show
import numpy as np


# Set filepaths
#  - experimental aluminum dataset
#  - a same day/conditions vacuum scan
#  - a path to write to
filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
filepath_vacuum = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_141_vacuumScan/dataset_141.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"


# Load a datacube from a dm file
datacube = py4DSTEM.io.read(filepath_calibration_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])

print(f"Loaded a {datacube.data.shape} shaped datacube with tree:")
datacube.tree.print()




# Virtual diffraction

dp_max = datacube.get_dp_max()
dp_mean = datacube.get_dp_mean()
dp_median = datacube.get_dp_median()
#show(datacube.tree['dp_max'], scaling='log')
#show(datacube.tree['dp_mean'], scaling='log')
#show(datacube.tree['dp_median'], scaling='log')

print("After virtual diffraction")
datacube.tree.print()





# Virtual imaging

geometry_BF = (
    (432,432),
    30
)
geometry_ADF = (
    (432,432),
    (80,300)
)
im_BF = datacube.get_virtual_image(
    mode = 'circle',
    geometry = geometry_BF,
    name = 'vBF'
)
im_ADF = datacube.get_virtual_image(
    mode = 'annulus',
    geometry = geometry_ADF,
    name = 'vADF'
)
#show(datacube.tree['vBF'])
#show(datacube.tree['vADF'])

print("After virtual imaging")
datacube.tree.print()





# Probe

datacube_vacuum = py4DSTEM.io.read(
    filepath_vacuum,
    name = 'datacube_vacuum'
)
print('Loaded a vacuum datacube:')
print(datacube_vacuum)

probe = datacube_vacuum.get_vacuum_probe(
    ROI = (7,10,7,10)
)

#show(datacube_vacuum.tree['probe'].probe)
#show(datacube_vacuum.tree['probe'].kernel)

probe.get_kernel(
    mode = 'sigmoid',
    radii = (0,50)
)

datacube.add(probe)

print("After probe")
datacube.tree.print()







# disk detection
rxs = 0,3,5,10,5,8
rys = 8,5,13,12,14,3


# Tune disk detection parameters on selected DPs
detect_params = {
    'minAbsoluteIntensity':0.65,
    'minPeakSpacing':20,
    'maxNumPeaks':20,
    'subpixel':'poly',
    'sigma':2,
    'edgeBoundary':20,
    'corrPower':1,
}

selected_peaks = datacube.find_Bragg_disks(
    data = (rxs,rys),
    template = probe.kernel,
    **detect_params
)


# Get all disks
braggvectors = datacube.find_Bragg_disks(
    template=probe.kernel,
    **detect_params
)


# BVM

bvm = braggvectors.get_bvm(
    Qshape = datacube.Qshape,
    mode = 'raw'
)

# set viz params
bvm_vis_params = {
    'cmap':'inferno',
    'scaling':'power',
    'power':0.125,
    'clipvals':'manual',
    'min':0,
    'max':20
}
show(bvm,**bvm_vis_params,figsize=(10,10))





########## Calibration ###########


# Origin

# Because this data had a beamstop blocking the unscattered beam,
# we'll use conjugate bragg pairs to find the origin

# Specify an annular region of Q-space 
center_guess = 380,415
radii = 240,265

#show(
#    bvm,
#    points = {'x':center_guess[0],'y':center_guess[1]},
#    annulus = {'center':center_guess,'radii':radii,'fill':True,'alpha':0.5,'color':'g'},
#    **bvm_vis_params,
#    figsize=(10,10)
#)

# Compute the origin position pattern-by-pattern

origin_meas = braggvectors.measure_origin(
    mode = 5,
    center_guess = center_guess,
    radii = radii,
    Q_Nx = datacube.Q_Nx,
    Q_Ny = datacube.Q_Ny,
    max_dist = 16,
    max_iter = 2
)

# Show the measured origin shifts

qx0_meas,qy0_meas = origin_meas[0],origin_meas[1]
mask = ~qx0_meas.mask
#show(qx0_meas,cmap='RdBu',clipvals='centered',min=np.mean(qx0_meas),max=8)
#show(qy0_meas,cmap='RdBu',clipvals='centered',min=np.mean(qy0_meas),max=8)


# Some local variation in the position of the origin due to electron-sample interaction is
# expected, and constitutes meaningful signal that we would not want to subtract away.
# In fitting a plane or parabolic surface to the measured origin shifts, we aim to
# capture the systematic shift of the beam due to the changing scan coils,
# while removing as little physically meaningful signal we can.

x = py4DSTEM.process.calibration.fit_origin(
    (qx0_meas,qy0_meas),
    mask=~mask,
    fitfunction='plane')
qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = x
py4DSTEM.visualize.show_image_grid(
    lambda i:[qx0_meas,qx0_fit,qx0_residuals,
              qy0_meas,qy0_fit,qy0_residuals][i],
    H=2,W=3,cmap='RdBu',clipvals='centered')


# Calibrate the origin

datacube.calibration.set_origin((qx0_fit,qy0_fit))


# Center the disk positions about the origin

#braggvectors.calibrate()





# io

#py4DSTEM.io.save(
#    filepath_h5,
#    datacube,
#    tree = True,
#    mode = 'o'
#)
#py4DSTEM.io.print_h5_tree(filepath_h5)
#
#d = py4DSTEM.io.read(
#    filepath_h5,
#    root = '4DSTEM_experiment/datacube/probe',
#    tree = False
#)
#
#
#print(d)
#print(d.metadata)



