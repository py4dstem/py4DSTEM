import h5py
import py4DSTEM
from py4DSTEM.process import amorph
import matplotlib
from py4DSTEM.process.utils.elliptical_coords import *
from py4DSTEM.process.calibration import ellipse
from tqdm import tqdm
from scipy.signal import medfilt2d
from scipy.ndimage import binary_closing
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing
from scipy.signal import medfilt2d
from scipy.signal import ellip
from tqdm import tqdm

matplotlib.rcParams["figure.dpi"] = 100
plt.ion()

# which files to read
linux = False
mac = True
# flags to control which part of the script to run
run_test = False
run_test2 = False
make_data = False
load_data = False
run_data = False
analyze_data = True
# make ellipse
"""
The parameters in p are

    p[0] I0          the intensity of the first gaussian function
    p[1] I1          the intensity of the Janus gaussian
    p[2] sigma_ref      std of first gaussian
    p[3] sigma1      inner std of Janus gaussian
    p[4] sigma2      outer std of Janus gaussian
    p[5] c_bkgd      a constant offset
    p[6] R           center/radius of the Janus gaussian
    p[7:8] x0,y0       the origin
    p[9:11] A,B,C       Ax^2 + Bxy + Cy^2 = 1

"""


# def compare_double_sided_gaussian(data, p, power=1, mask=None, fig_num=12):
#     """
#     Plots a comparison between a diffraction pattern and a fit, given p.
#     """
#     if mask is None:
#         mask = np.ones_like(data)

#     yy, xx = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
#     data_fit = ellipse.double_sided_gaussian(p, xx, yy)

#     theta = np.arctan2(xx - p[7], yy - p[8])
#     theta_mask = np.cos(theta * 8) > 0
#     data_combined = (data * theta_mask + data_fit * (1 - theta_mask)) ** power
#     data_combined = mask * data_combined
#     plt.figure(fig_num, clear=True)
#     plt.imshow(data_combined)

#     return data_combined


if run_test:
    # this tests if double sided gaussian is functioning properly, and it's result can be properly fit by the fitting function
    yy, xx = np.meshgrid(np.arange(256), np.arange(256))
    coef = [250, 250, 5, 4, 4, 5, 127, 125, 2e-4, -2e-8, 2e-4]
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C = coef
    r2 = A * (xx - x0) ** 2 + B * (xx - x0) * (yy - y0) + C * (yy - y0) ** 2

    mask = np.logical_and(r2 ** 0.5 > .6, r2 ** 0.5 < 1.4)
    # mask = np.ones_like(mask, dtype=bool)

    ring = ellipse.double_sided_gaussian(coef, xx, yy) + np.random.rand(256, 256) * 100

    coef_fit = [250, 250, 5, 4, 4, 5, 128, 128, 2e-4, 0, 2e-4]
    fit = ellipse.fit_ellipse_amorphous_ring(
        ring, (128, 128), (10, 1000), p0=coef_fit, mask=mask
    )[1]

    plt.figure(1, clear=True)
    plt.imshow(ring * mask)

    plt.figure(2, clear=True)
    ring_fit = ellipse.double_sided_gaussian(fit, xx, yy)
    plt.imshow(ring_fit * mask)
    py4DSTEM.visualize.vis_special.show_amorphous_ring_fit(
        ring, fitradii=(40,None),p_dsg=fit, maskcenter=False
    )

    # try on affine transformed data
    ring2 = affine_transform(ring, [[1, 0.1], [0.1, 1]])
    fit2 = ellipse.fit_ellipse_amorphous_ring(ring2, (110, 110), (10, 200), p0=coef_fit)[1]
    py4DSTEM.visualize.vis_special.show_amorphous_ring_fit(
        ring2, p_dsg=fit2, fitradii=(10,None))  # TODO remove this

if run_test2:
    # this code tests to see from an arbitrary ellipse created via known strains, if the strains can be recovered. This is a more complex test than run_test, in that the double_sided_gaussian function is not used to create the data.
    # this test also can be used to check if the math behind using a reference ellipse works by changing ref_x and ref_y in the definitions below
    yy, xx = np.meshgrid(np.arange(256), np.arange(256))
    r = np.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
    r_ref = 65
    e11 = 10 / 100
    e22 = 2 / 100
    e12 = 5 / 100

    num_points = 360
    ref_x = 1.0
    ref_y = 1.2
    ref_rot = -np.pi / 4
    t = np.linspace(0, 2 * np.pi, num_points)
    x = ref_x * np.cos(t + ref_rot) * r_ref
    y = ref_y * np.sin(t + ref_rot) * r_ref

    m = np.array([[1 + e11, e12], [e12, 1 + e22]])
    m_inv = np.linalg.inv(m)

    xy_ref = np.stack((x, y))
    xy = np.matmul(m_inv, xy_ref)

    ring_ref = np.zeros((256, 256))
    ring_ref[
        np.round(xy_ref[0, :]).astype(int) + 128,
        np.round(xy_ref[1, :]).astype(int) + 128,
    ] = 1
    ring_ref = gaussian_filter(ring_ref, sigma=5)
    ring_ref *= 10000

    ring = np.zeros((256, 256))
    ring[np.round(xy[0, :]).astype(int) + 128, np.round(xy[1, :]).astype(int) + 128] = 1
    ring = gaussian_filter(ring, sigma=5)
    ring *= 10000

    plt.figure(1, clear=True)
    plt.plot(xy_ref[1, :], xy_ref[0, :])
    plt.plot(xy[1, :], xy[0, :])
    plt.plot(np.cos(t) * r_ref, np.sin(t) * r_ref, "--")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.legend(["reference", "strained", "circle"])

    plt.figure(2, clear=True)
    plt.imshow(ring, cmap="plasma")

    coef_fit = [250, 250, 5, 4, 4, 5, 128, 128, 2e-4, 0, 2e-4]
    fit = ellipse.fit_ellipse_amorphous_ring(ring, (128, 128), (10, 1000), p0=coef_fit)[1]
    fit_ref = ellipse.fit_ellipse_amorphous_ring(
        ring_ref, (128, 128), (10, 1000), p0=coef_fit
    )[1]
    py4DSTEM.visualize.vis_special.show_amorphous_ring_fit(
        ring,
        p_dsg=fit,
        fitradii=(10,None),
        scaling="none",
        fitbordercolor="pink",
        cmap=("gray", "plasma"),
    )
    # now we need to get appropriate A, B, C again compared to colin's code, so we can then extract e11, e22, and e12
    r_ratio = fit[6] / fit_ref[6]
    r_ratio=1
    A, B, C = fit[8], fit[9], fit[10]

    # print(f"r_ref = {r_ref}, fit_r = {fit[6]}")
    # print(f"A = {A:.2f}, A_new = {A / r_ratio ** 2:.4f}")
    # print(f"B = {B:.2f}, B_new = {B / r_ratio ** 2:.4f}")
    # print(f"C = {C:.2f}, C_new = {C / r_ratio ** 2:.4f}")

    r_ratio_ref = fit_ref[6] / fit_ref[6]
    r_ratio_ref=1
    A_ref, B_ref, C_ref = fit_ref[8], fit_ref[9], fit_ref[10]

    # print(f'r_ratio = {r_ratio}')

    # print(f"r_ref = {r_ref}, fit_r = {fit_ref[6]}")
    # print(f"A_ref = {A_ref:.2f}, A_ref_new = {A_ref / r_ratio_ref ** 2:.4f}")
    # print(f"B_ref = {B_ref:.2f}, B_ref_new = {B_ref / r_ratio_ref ** 2:.4f}")
    # print(f"C_ref = {C_ref:.2f}, C_ref_new = {C_ref / r_ratio_ref ** 2:.4f}")

    # now calculate strains
    A /= r_ratio ** 2
    B /= r_ratio ** 2
    C /= r_ratio ** 2

    m_ellipse = np.asarray([[A, B / 2], [B / 2, C]])
    e_vals, e_vecs = np.linalg.eig(m_ellipse)
    ang = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])

    rot_matrix = np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    transformation_matrix = np.diag(np.sqrt(e_vals))

    transformation_matrix = rot_matrix @ transformation_matrix @ rot_matrix.T

    # now fit reference ellipse
    m_ellipse_ref = np.asarray([[A_ref, B_ref / 2], [B_ref / 2, C_ref]])
    e_vals_ref, e_vecs_ref = np.linalg.eig(m_ellipse_ref)
    ang_ref = np.arctan2(e_vecs_ref[1, 0], e_vecs_ref[0, 0])

    rot_matrix_ref = np.asarray(
        [[np.cos(ang_ref), -np.sin(ang_ref)], [np.sin(ang_ref), np.cos(ang_ref)]]
    )

    transformation_matrix_ref = np.diag(np.sqrt(e_vals_ref))

    transformation_matrix_ref = (
        rot_matrix_ref @ transformation_matrix_ref @ rot_matrix_ref.T
    )

    # uncomment to confirm that we used to assume circle - the two printed results should match
    # transformation_matrix_ref = np.asarray([[1,0],[0,1]])

    # print(f"angle0 = {ang_ref:.4f}")
    print(
        f"This is the transformation from a circle to the reference ellipse\ntransformation_matrix_ref=\n{transformation_matrix_ref}\n"
    )
    # print(f"angle = {ang:.4f}")
    print(
        f"This is the transformation from a circle to the fitted (strained) ellipse\ntransformation_matrix=\n{transformation_matrix}\n"
    )

    # now compute strains
    e11_fit = transformation_matrix[0, 0] - 1
    e22_fit = transformation_matrix[1, 1] - 1
    e12_fit = 0.5 * (transformation_matrix[0, 1] + transformation_matrix[1, 0])

    # If you have a transformation matrix from A->B and from A->C, this is how
    # you get the transformation matrix from B->C. A->B represents circle to
    # reference ellipse, A->C is circle to strained ellipse, and so then B->C
    # is reference ellipse to strained ellipse, from which we can then extract
    # strains.
    transformation_matrix_from_ref = transformation_matrix @ np.linalg.inv(
        transformation_matrix_ref
    )
    e11_fit_from_ref = transformation_matrix_from_ref[0, 0] - 1
    e22_fit_from_ref = transformation_matrix_from_ref[1, 1] - 1
    e12_fit_from_ref = 0.5 * (
        transformation_matrix_from_ref[0, 1] + transformation_matrix_from_ref[1, 0]
    )

    print(
        "\nThe fit will not be exact as we are fitting the image, not points.\nStrain with respect to perfect circle:"
    )
    print(f"e11 = {e11:.3f}, e11_fit = {e11_fit:.3f}")
    print(f"e22 = {e22:.3f}, e22_fit = {e22_fit:.3f}")
    print(f"e12 = {e12:.3f}, e12_fit = {e12_fit:.3f}")
    print("Strain with respect to reference ellipse:")
    print(f"e11 = {e11:.3f}, e11_fit_from_ref = {e11_fit_from_ref:.3f}")
    print(f"e22 = {e22:.3f}, e22_fit_from_ref = {e22_fit_from_ref:.3f}")
    print(f"e12 = {e12:.3f}, e12_fit_from_ref = {e12_fit_from_ref:.3f}")
    print("These values should be the same as the input values. ")


if make_data:
    f = h5py.File(
        "/Volumes/tom_home/Documents/Research/Data/2020/amorphous_py4DSTEM_paper/Dataset38_bksbtr_20190918.h5",
        "r",
    )

    data = np.empty((162, 285, 112, 120))
    for i in tqdm(range(162)):
        for j in range(285):
            data[i, j, :, :] = bin2D(
                f["4DSTEM_experiment"]["data"]["datacubes"]["GTO_datacube_bksbtr"][
                    "data"
                ][i, j, :, :],
                4,
            )

# the data is binned by 4 and is now [162,285,112,120]

if load_data:
    print("loading data")
    if linux:
        data = py4DSTEM.file.io.read(
            "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/binned_data.h5"
        )
        helper_data_browser = py4DSTEM.file.io.FileBrowser(
            "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/Dataset38_20190918_processing.h5"
        )
        peaks = helper_data_browser.get_dataobject("braggpeaks_unshifted")
        mask_array = py4DSTEM.file.io.read(
            "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/data_mask_spots_and_radius.h5"
        )
    elif mac:
        data = py4DSTEM.io.read(
            "/Volumes/tom_home/Documents/Research/Data/2020/amorphous_py4DSTEM_paper/binned_data.h5",
            mem="MEMMAP",
            data_id=0,
        )

        peaks = py4DSTEM.io.read(
            "/Volumes/tom_home/Documents/Research/Data/2020/amorphous_py4DSTEM_paper/Dataset38_20190918_processing.h5",
            data_id="braggpeaks_unshifted",
        )

        mask_array = py4DSTEM.io.read(
            "/Volumes/tom_home/Documents/Research/Data/2020/amorphous_py4DSTEM_paper/data_mask_spots_and_radius.h5",
            data_id=0,
            mem="MEMMAP",
        )

if run_data:
    # make a mask of region to fit
    # mean_dp = np.mean(data.data, axis=(0, 1))
    # mean_im = np.mean(data.data, axis=(2, 3))
    yy, xx = np.meshgrid(np.arange(mean_dp.shape[1]), np.arange(mean_dp.shape[0]))
    center = [52, 59]
    r = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) ** 0.5
    mask_r = np.ones_like(r)
    ri, ro = 15, 40
    mask_r[r < ri] = 0
    mask_r[r > ro] = 0
    mask_r = mask_r.astype(bool)

    test_im = data.data[152, 101, :, :]
    # test_im = np.mean(data.data[-10:-1, -10:-1,:,:], axis=(0,1))
    plt.figure(10, clear=True)
    # test_im = np.log(test_im + 0.01)
    plt.imshow(test_im * mask_r)

    p_init = [2.95e5, 7e2, 1.5, 2.1, 2.2, 200, 51, 56, .001, 0, .001]
    # test_fit = ellipse.fit_ellipse_amorphous_ring(test_im, 50,60,10,40,p0=p_init,)[1]
    test_fit = ellipse.fit_ellipse_amorphous_ring(
        test_im, (50, 50), (10, 400), p0=p_init, mask=mask_r
    )[1]
    # compare_double_sided_gaussian(test_im, p_init, mask=mask_r)
    py4DSTEM.visualize.vis_special.show_amorphous_ring_fit(
        test_im,
        p_dsg=test_fit,
        fitradii=(ri,ro),
        scaling="none",
        fitbordercolor="pink",
        cmap=("gray", "gray"),
        maskcenter=True,
    )

    # mask_array = amorph.make_mask_array(peaks, data.data.shape, peak_radius=4.3, bin_factor=4, universal_mask=mask_r)
    # mask_array = py4DSTEM.file.datastructure.DataCube(mask_array)
    np.seterr(all="ignore")
    print("running whole fit")
    coef_array = amorph.strain.fit_stack(data, p_init, ri, ro, mask_array.data)

if analyze_data:
    # coef_array = py4DSTEM.file.io.read(
    #     "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/ellipse_coefs.h5"
    # )
    # coef_array = np.squeeze(coef_array.data)
    # coef_array[np.isnan(coef_array)] = 1
    if linux:
        coef_array = np.load(
            "/home/tom/Documents/Research/results/2020-1-10 Strain Mapping/coef_array_np.npy"
        )
    if mac:
        coef_array = np.load(
            "/Volumes/tom_home/Documents/Research/Data/2020/amorphous_py4DSTEM_paper/coef_array_np.npy"
        )
    # # remove zeros/nan values
    coef_array[np.where(np.isnan(coef_array))] = 10

    # used to set 0 strain value - remember that strain values are comparative! 29 seems good
    radius_ref = 29.25
    mean_ellipse_coeffs = np.mean(coef_array[150:, :, :], axis=(0, 1))
    radius_ref = (
        mean_ellipse_coeffs[6],
        mean_ellipse_coeffs[9],
        mean_ellipse_coeffs[10],
    )

    # mask of regions that are crystalline
    # strains = amorph.calculate_coef_strain(coef_array, r_ref=radius_ref, A_ref=np.median(coef_array[:,:,9]), B_ref=np.median(coef_array[:,:,10]), C_ref=np.median(coef_array[:,:,11]))
    strains = amorph.strain.calculate_coef_strain(coef_array, r_ref=radius_ref)

    print(f"reference radius = {radius_ref}")
    print(
        f"median strains [exx, eyy, exy] = {np.median(np.asarray(strains)[:,135:,:], axis=(1,2))}"
    )

    mask_strain = np.logical_or(np.abs(strains[0]) > 0.1, np.abs(strains[1]) > 0.1)
    mask_strain = binary_closing(mask_strain, iterations=3, border_value=1)

    normalized_strains = [
        medfilt2d(i) - np.median(medfilt2d(i)[135:, :]) for i in strains
    ]

    titles = [r"$\epsilon_{xx}$", r"$\epsilon_{yy}$", r"$\epsilon_{xy}$"]

    fig, axs = py4DSTEM.visualize.vis_grid.show_image_grid(
        lambda x: np.ma.array(normalized_strains[x], mask=mask_strain),
        1,
        3,
        (3, 3),
        title=titles,
        returnfig=True,
        clipvals="manual",
        min=-0.04,
        max=0.04,
        cmap="RdBu_r",
        num=11,
    )

    for i in axs:
        i.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.05])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu_r"), cax=cbar_ax, orientation="horizontal"
    )
    plt.tight_layout()
