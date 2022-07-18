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
#show(dp_max, scaling='log')
#show(dp_mean, scaling='log')
#show(dp_median, scaling='log')






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




# Disk Detection

# Select scan positions
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

# Find disks in selected diffraction patterns
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


# Bragg vector map
bvm = braggvectors.get_bvm( mode = 'raw' )

# set viz params
bvm_vis_params = {
    'cmap':'inferno',
    'scaling':'power',
    'power':0.125,
    'clipvals':'manual',
    'min':0,
    'max':20
}
#show(bvm,**bvm_vis_params,figsize=(10,10))








########## Calibration ###########


# ORIGIN

# Because this data had a beamstop blocking the unscattered beam,
# we'll use conjugate bragg pairs to find the origin

# Specify an annular region of Q-space 
center_guess = 380,415
radii = 240,265
#show(
#    bvm,
#    **bvm_vis_params,
#    points = {'x':center_guess[0],'y':center_guess[1]},
#    annulus = {'center':center_guess,'radii':radii,'fill':True,'alpha':0.5,'color':'g'},
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
#py4DSTEM.visualize.show_image_grid(
#    lambda i:[qx0_meas,qx0_fit,qx0_residuals,
#              qy0_meas,qy0_fit,qy0_residuals][i],
#    H=2,W=3,cmap='RdBu',clipvals='centered')

# Calibrate the origin
datacube.calibration.set_origin( (qx0_fit,qy0_fit) )


# Center the disk positions about the origin
braggvectors.calibrate()


# Centered BVM
bvm_centered = braggvectors.get_bvm()
#show(
#    bvm_centered,
#    **bvm_vis_params
#)


# ELLIPSE

# Select a fitting region
qmin,qmax = 240,265
#show(
#    bvm_centered,
#    **bvm_vis_params,
#    annulus={
#        'center' : (datacube.Q_Nx/2.,datacube.Q_Ny/2.),
#        'radii' : (qmin,qmax),
#        'fill' : True,
#        'color' : 'y',
#        'alpha' : 0.2
#    }
#)

# Fit the elliptical distortions
p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
    bvm_centered,
    (datacube.Q_Nx/2.,datacube.Q_Ny/2.),
    (qmin,qmax)
)
#py4DSTEM.visualize.show_elliptical_fit(
#    bvm_centered,
#    (qmin,qmax),
#    p_ellipse,
#    **bvm_vis_params
#)

# Elliptical calibration
datacube.calibration.set_p_ellipse(p_ellipse)

# Elliptically correct bragg peak positions, stretching the elliptical
# semiminor axis to match the semimajor axis length
braggvectors.calibrate()

# Recompute the bvm
bvm_ellipsecorr = braggvectors.get_bvm()
#show(bvm_ellipsecorr,**bvm_vis_params)

# Confirm that elliptical distortions have been removed
# by measuring the ellipticity of the corrected data
p_ellipse_corr = py4DSTEM.process.calibration.fit_ellipse_1D(
    bvm_ellipsecorr,
    p_ellipse[:2],
    (qmin,qmax)
)
#py4DSTEM.visualize.show_elliptical_fit(
#    bvm_ellipsecorr,
#    (qmin,qmax),
#    p_ellipse_corr,
#    **bvm_vis_params
#)
# Print the ratio of the semi-axes before and after correction
#print("The ratio of the semimajor to semiminor axes was measured to be")
#print("")
#print("\t{:.2f}% in the original data and".format(100*p_ellipse[2]/p_ellipse[3]))
#print("\t{:.2f}% in the corrected data.".format(100*p_ellipse_corr[2]/p_ellipse_corr[3]))



# PIXEL SIZE

# Radial integration
ymax = 120
dq=0.25             # binsize for the x-axis
q,I_radial = py4DSTEM.process.utils.radial_integral(
    bvm_ellipsecorr,
    datacube.Q_Nx/2,
    datacube.Q_Ny/2,
    dr=dq
)
#py4DSTEM.visualize.show_qprofile(q=q,intensity=I_radial,ymax=ymax)

# Fit a gaussian to find a peak location
qmin,qmax = 240,270
A,mu,sigma = py4DSTEM.process.fit.fit_1D_gaussian(q,I_radial,qmin,qmax)
#fig,ax = py4DSTEM.visualize.show_qprofile(q=q,intensity=I_radial,ymax=ymax,
#                                          returnfig=True)
#ax.vlines((qmin,qmax),0,ax.get_ylim()[1],color='r')
#ax.vlines(mu,0,ax.get_ylim()[1],color='g')
#ax.plot(q,py4DSTEM.process.fit.gaussian(q,A,mu,sigma),color='r')
#plt.show()

# Get pixel calibration
# At time of writing, one peak with a known spacing
# must be manually identified and entered
d_spacing_A = 2.337                           # This is the Al 111 peak
#d_spacing_A = 2.024                         
inv_A_per_pixel = 1./(d_spacing_A * mu)
#py4DSTEM.visualize.show_qprofile(q=q*inv_A_per_pixel,intensity=I_radial,
#                                 ymax=ymax,xlabel='q (1/A)')

# Demonstrate consistency with known Al spacings
spacings_A = np.array([1.1685,1.22,1.431,2.024,2.337])   # 111, 200, 220, 113, 222
spacings_inv_A = 1./spacings_A
#fig,ax = py4DSTEM.visualize.show_qprofile(q=q*inv_A_per_pixel,intensity=I_radial,
#                                 ymax=ymax,xlabel='q (1/A)',returnfig=True)
#ax.vlines(spacings_inv_A,0,ax.get_ylim()[1],color='r')
#plt.show()

# Pixel calibration
datacube.calibration.set_Q_pixel_size(inv_A_per_pixel)
datacube.calibration.set_Q_pixel_units('A^-1')

# Calibrate bragg vector distances
braggvectors.calibrate()

# BVM
#bvm_cal = braggvectors.get_bvm()
#show(bvm_cal, **bvm_vis_params)








# io

py4DSTEM.io.save(
    filepath_h5,
    datacube,
    tree = 'noroot',
    mode = 'o'
)
py4DSTEM.io.print_h5_tree(filepath_h5)

d = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/datacube/',
    tree = True
)


#print(d)
#print(d.calibration)



#d.calibration = datacube.calibration
#print(d.calibration)

#d.tree['braggvectors'].calibrate()






