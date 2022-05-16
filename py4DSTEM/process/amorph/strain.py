import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
from py4DSTEM.process.calibration import ellipse

# this fixes figure si
# bring strain mapping, fit stack code here, rely on visualization package for plotting. also bring in make_mask_array

# to fit an ellipse process.calibration.fit_ellipse_amorphous_ring
# Adjust strain code to handle (r, e, theta)


def make_mask_array(
    peak_pos_array, data_shape, peak_radius, bin_factor=1, universal_mask=None
):
    """
    Maybe make a mask module or leave here. Or make amorph module itself!

    This function needs a real home, I don't know where to put it yet. But this function will take in a peakListArray with all of the peaks in the diffraction pattern, and make a 4d array of masks such that they can be quickly applied to the diffraction patterns before fitting. 

    Accepts:
    peak_pos_array  - a peakListArray corresponding to all of the diffraction patterns in the dataset
    data_shape      - the 4-tuple shape of the data, essentially data.data.shape (qx, qy, x, y)
    peak_radius     - the peak radius
    bin_factor      - if the peak positions were measured at a different binning factor than the data, this will effectivly divide their location by bin_factor
    
    Returns:
    mask_array      - a 4D array of masks the same shape as the data
    """
    if universal_mask is None:
        universal_mask = np.ones(data_shape[2:])
    mask_array = np.ones(data_shape)
    yy, xx = np.meshgrid(np.arange(data_shape[3]), np.arange(data_shape[2]))

    for i in tqdm(np.arange(data_shape[0])):
        for j in np.arange(data_shape[1]):
            mask_array[i, j, :, :] = universal_mask
            for spot in peak_pos_array.get_pointlist(i, j).data:
                temp_inds = (
                    (xx - spot[0] / bin_factor) ** 2 + (yy - spot[1] / bin_factor) ** 2
                ) ** 0.5
                temp_inds = temp_inds < peak_radius
                mask_array[i, j, temp_inds] = 0
                # plt.figure(100,clear=True)
                # plt.imshow(temp_inds)
                # plt.pause(0.2)
    return mask_array.astype(bool)


def fit_stack(datacube, init_coefs, ri, ro, mask=None):
    """
    This will fit an ellipse using the polar elliptical transform code to all the diffraction patterns. It will take in a datacube and return a coefficient array which can then be used to map strain, fit the centers, etc.

    Accepts:
        datacute    - a datacube of diffraction data
        init_coefs  - an initial starting guess for the fit
        mask        - a mask, either 2D or 4D, for either one mask for the whole stack, or one per pattern. 
    Returns:
        coef_cube  - an array of coefficients of the fit
    """
    coefs_array = np.zeros([i for i in datacube.data.shape[0:2]] + [len(init_coefs)])
    for i in tqdm(range(datacube.R_Nx)):
        for j in tqdm(range(datacube.R_Ny)):
            if len(mask.shape) == 2:
                mask_current = mask
            elif len(mask.shape) == 4:
                mask_current = mask[i, j, :, :]

            coefs = ellipse.fit_ellipse_amorphous_ring(
                datacube.data[i, j, :, :],
                init_coefs[7],
                init_coefs[8],
                ri,
                ro,
                p0=init_coefs,
                mask=mask_current,
            )[1]
            coefs_array[i, j] = coefs

    return coefs_array


def calculate_coef_strain(coef_cube, r_ref):
    """
    This function will calculate the strains from a 3D matrix output by fit_stack

    Coefs order:
        I0          the intensity of the first gaussian function
        I1          the intensity of the Janus gaussian
        sigma0      std of first gaussian
        sigma1      inner std of Janus gaussian
        sigma2      outer std of Janus gaussian
        c_bkgd      a constant offset
        x0,y0       the origin
        A,B,C         x^2 + Bxy + Cy^2 = 1

    Accepts:
        coef_cube   - output from fit_stack
        r_ref       - a reference 0 strain radius - needed because we fit r as well as B and C
                      if r_ref is a number, it will assume that the reference ellipse is a perfect circle
                      r_ref can also be a 3-tuple, in which case it is (r_ref, B_ref, C_ref)
                      in this case, the reference 0 strain measurement is an ellipse defined by these numbers, which are defined as the outputs of fit_ellipse_amorphous_ring
    Returns:
        exx         - strain in the x axis direction in image coordinates
        eyy         - strain in the y axis direction in image coordinates
        exy         - shear

    """
    # parse r_ref input
    if isinstance(r_ref, (int, float)):
        # r_ref is integer, and we will measure strain with circle
        A_ref, B_ref, C_ref = 1, 0, 1
    elif isinstance(r_ref, (tuple, list)):
        if len(r_ref) != 3:
            raise AssertionError(
                "r_ref must be a 3 element tuple with elements(r_ref, B_ref, C_ref)."
            )
        # r_ref is a tuple with (r_ref, B_ref, C_ref)
        A_ref, B_ref, C_ref = r_ref[0], r_ref[1], r_ref[2]
    else:
        raise ValueError("r_ref must be a number, or 3 element tuple")

    # R = coef_cube[:, :, 6]
    # r_ratio = (
    #     R / r_ref
    # )  # this is a correction factor for what defines 0 strain, and must be applied to A, B and C. This has been found _experimentally_! TODO have someone else read this

    r_ratio=1
    A = coef_cube[:, :, 8] / r_ratio ** 2
    B = coef_cube[:, :, 9] / r_ratio ** 2
    C = coef_cube[:, :, 10] / r_ratio ** 2

    # make reference transformation matrix
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

    transformation_matrix_ref_inv = np.linalg.inv(transformation_matrix_ref)

    exx, eyy, exy = np.empty_like(A), np.empty_like(C), np.empty_like(B)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            m_ellipse = np.asarray([[A[i, j], B[i, j] / 2], [B[i, j] / 2, C[i, j]]])
            e_vals, e_vecs = np.linalg.eig(m_ellipse)
            ang = np.arctan2(e_vecs[1, 0], e_vecs[0, 0])
            rot_matrix = np.asarray(
                [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
            )
            transformation_matrix = np.diag(np.sqrt(e_vals))
            transformation_matrix = rot_matrix @ transformation_matrix @ rot_matrix.T

            transformation_matrix_from_ref = (
                transformation_matrix @ transformation_matrix_ref_inv
            )

            exx[i, j] = transformation_matrix_from_ref[0, 0] - 1
            eyy[i, j] = transformation_matrix_from_ref[1, 1] - 1
            exy[i, j] = 0.5 * (
                transformation_matrix_from_ref[0, 1]
                + transformation_matrix_from_ref[1, 0]
            )

    return exx, eyy, exy
