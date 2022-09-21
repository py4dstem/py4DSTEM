# Test the DataCube datastructure and its methods

import py4DSTEM
import numpy as np

run_virt_diff = True
run_virt_im = True
run_probe = False
run_disk_detection = False
run_calibration = False
show_plots = False
show_imgrids = True


# Load data
fp = "/Users/Ben/Work/Data/py4DSTEM_sampleData/calibration_simulatedAuNanoplatelet/calibrationData_simulatedAuNanoplatelet_binned.h5"
datacube = py4DSTEM.io.read(fp,data_id='polyAu_4DSTEM')
datacube.crop_data_real(30,50,30,50) # crop dataset for testing
probe_image = py4DSTEM.io.read(fp,data_id='probe_template').data
print(datacube.data.shape)
print(datacube.R_Nx,datacube.R_Ny,datacube.Q_Nx,datacube.Q_Ny)
datacube.set_bvm_vis_params(cmap='gray',scaling='power',power=0.5,
                            clipvals='manual',min=0,max=500)


if run_virt_diff:
    print('Running virtual diffraction methods.............')

    # single DP
    dp = datacube.get_dp((2,3))
    datacube.get_dp((2,3),name='dp 2,3')
    if show_plots:
        py4DSTEM.visualize.show(dp.data,scaling='log',title='dp 2,3')
        datacube.show('dp 2,3')
        datacube.show((2,3))
        pass

    # max, mean, median DP
    datacube.get_max_dp()
    datacube.get_mean_dp()
    datacube.get_median_dp()
    if show_plots:
        datacube.show() # default argument is 'max_dp'
        datacube.show('median_dp')
        datacube.show('mean_dp')

    if show_imgrids:
        # Show all three together

        if False:
            # first with the visualize.show function
            fig,axs = py4DSTEM.visualize.show(
                [datacube.diffractionslices['max_dp'].data,
                 datacube.diffractionslices['median_dp'].data,
                 datacube.diffractionslices['mean_dp'].data],
                scaling='log',returnfig=True
            )
            axs[0,0].set_title('max_dp')
            axs[0,1].set_title('median_dp')
            axs[0,2].set_title('mean_dp')
            import matplotlib.pyplot as plt
            plt.show()

        # then with the datacube.show function
        datacube.show(['max_dp','median_dp','mean_dp'])
        datacube.show([[(0,0),(5,5),(4,3),(7,6)],
                       [(2,2),(11,15),(6,1),(0,5)]])

    # max,mean,median - specify lims
    lims=(0,5,0,5)   #specify rectangular region
    if show_plots:
        datacube.position_realspace_lims(lims)
    datacube.get_max_dp(lims,name='max_dp_2')
    datacube.get_mean_dp(lims,name='mean_dp_2')
    datacube.get_median_dp(lims,name='median_dp_2')
    if show_plots:
        datacube.show('max_dp_2')
        datacube.show('mean_dp_2')
        datacube.show('median_dp_2')

    # max,mean,median - specify mask
    s = datacube.data.shape[:2]   #specify region w/boolean mask
    mask = np.arange(np.prod(s)).reshape(s)<np.prod(s)/8
    if show_plots:
        datacube.position_realspace_mask(mask)
    datacube.get_max_dp(mask,name='max_dp_3')
    datacube.get_mean_dp(mask,name='mean_dp_3')
    datacube.get_median_dp(mask,name='median_dp_3')
    if show_plots:
        datacube.show('max_dp_3')
        datacube.show('mean_dp_3')
        datacube.show('median_dp_3')

if run_virt_im:
    print('Running virtual image methods...................')

    geometry_point1 = (63,63)
    geometry_circ1 = ((63,63),12)
    geometry_rect1 = (53,73,53,73)
    geometry_ann1 = ((63,63),(12,24))
    geometry_point2 = (33,66)
    geometry_circ2 = ((45,80),12)
    geometry_rect2 = (57,87,68,92)
    geometry_ann2 = ((45,80),(8,16))
    if show_plots:
        datacube.position_point_detector(geometry_point1)
        datacube.position_point_detector(geometry_point2)
        datacube.position_circular_detector(geometry_circ1)
        datacube.position_circular_detector(geometry_circ2)
        datacube.position_rectangular_detector(geometry_rect1)
        datacube.position_rectangular_detector(geometry_rect2)
        datacube.position_annular_detector(geometry_ann1)
        datacube.position_annular_detector(geometry_ann2)
    datacube.capture_point_detector(geometry_point1,
                                    name='point detector 1')
    datacube.capture_circular_detector(geometry_circ1,
                                    name='circular detector 1')
    datacube.capture_rectangular_detector(geometry_rect1,
                                    name='rectangular detector 1')
    datacube.capture_annular_detector(geometry_ann1,
                                    name='annular detector 1')
    datacube.get_im(geometry_point2,
                    detector='point',
                    name='point detector 2')
    datacube.get_im(geometry_rect2,
                    detector='rect',
                    name='rectangular detector 2')
    datacube.get_im(geometry_circ2,
                    detector='circ',
                    name='circular detector 2')
    datacube.get_im(geometry_ann2,
                    detector='ann',
                    name='annular detector 2')
    datacube.get_sum_im()
    if show_plots:
        datacube.show_im('point detector 1')
        datacube.show_im('point detector 2')
        datacube.show_im('rectangular detector 1')
        datacube.show_im('rectangular detector 2')
        datacube.show_im('circular detector 1')
        datacube.show_im('circular detector 2')
        datacube.show_im('annular detector 1')
        datacube.show_im('annular detector 2')
        datacube.show_im('sum')

    if show_imgrids:
        datacube.show_im(['sum','rectangular detector 1',
            'circular detector 2','point detector 1'])

if run_probe:
    print('Running probe methods...........................')

    datacube.add_probe_image(probe_image)
    datacube.get_probe_size()
    datacube.get_probe_kernel(method='sigmoid',ri=0,
        ro=datacube.calibrations.get_convergence_semiangle_pixels()*4)
    if show_plots:
        datacube.show('probe_image')
        datacube.show_probe_size()
        datacube.show_probe_kernel()

if run_disk_detection:
    print('Running disk detection methods..................')

    positions = ((0,0),
                 (3,7),
                 (6,4))
    disk_detec_params = {
        'corrPower':1.0,
        'sigma':1,
        'edgeBoundary':4,
        'minAbsoluteIntensity':15,
        'minRelativeIntensity':0,
        'minPeakSpacing':8,
        'maxNumPeaks':40,
        'subpixel':'poly',
        'upsample_factor':16
    }
    datacube.find_some_bragg_disks(positions,**disk_detec_params)
    if show_plots:
        datacube.show_some_DPs(positions)
        datacube.show_some_bragg_disks()
    datacube.find_bragg_disks(**disk_detec_params)

if run_calibration:
    print('Running calibration methods.....................')

    # make calibration measurements

    ## origin
    datacube.measure_origin()
    datacube.fit_origin()
    if show_plots: datacube.show_origin_meas()
    if show_plots: datacube.show_origin_fit()

    ## ellipse
    fitradii = 40,47
    datacube.peaks.calibrate(which='origin')
    if show_plots:
        datacube.peaks.select_radii(fitradii)
        p_ellipse = datacube.peaks.fit_elliptical_distortions(fitradii)
        datacube.peaks.show_elliptical_fit(fitradii,p_ellipse)
    else:
        datacube.peaks.fit_elliptical_distortions(fitradii)

    ## pixel size
    qlims = 33,38
    datacube.peaks.calibrate(which='ellipse')
    datacube.peaks.get_radial_integral(which='ellipse')
    q_fit = datacube.peaks.fit_radial_peak(qlims,show=show_plots)
    if show_plots: datacube.peaks.show_radial_profile(which='ellipse')
    dq,dq_units = py4DSTEM.process.calibration.get_Q_pixel_size(
        q_fit,q_known=1.442,units='A')
    datacube.calibrations.set_Q_pixel_size(dq)
    datacube.calibrations.set_Q_pixel_units(dq_units)
    if show_plots:
        datacube.peaks.show_radial_profile()
        spacings_Ang = np.array([1.177,1.23,1.442,2.039,2.355])   # Au 222, 113, 022, 002, 111
        spacings_inv_Ang = 1./spacings_Ang
        datacube.peaks.show_radial_profile(q_ref=spacings_inv_Ang)


    ## rotation
    CoMx, CoMy = py4DSTEM.process.dpc.get_CoM_images(datacube, mask=1)
    if show_plots: py4DSTEM.visualize.show_image_grid(
        lambda i:[CoMx, CoMy][i],H=1,W=2,cmap='RdBu',title='CoM x,y')
    theta, flip =  py4DSTEM.process.dpc.get_rotation_and_flip_maxcontrast(
        CoMx, CoMy, 360)
    # Solve for minimum rotation from (-pi,pi) radian range
    theta = np.mod(theta + np.pi, 2*np.pi) - np.pi
    theta_deg = theta*180/np.pi
    print('Image flip detected =', flip);
    print('Best fit rotation = ', '%.4f' % theta_deg, 'degrees')
    datacube.calibrations.set_QR_rotation_degrees(theta_deg)
    datacube.calibrations.set_QR_flip(flip)

    # apply calibrations to the bragg positions
    datacube.peaks.calibrate()

    # get/show bvms before and after
    datacube.peaks.get_bvm(which='raw')
    datacube.peaks.get_bvm(which='all')
    if show_plots: datacube.peaks.show_bvm(which='raw')
    if show_plots: datacube.peaks.show_bvm(which='all')




