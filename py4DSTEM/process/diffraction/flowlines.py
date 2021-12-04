# Functions for creating flowline maps from diffraction spots




def make_orientation_histogram(
    braggpeaks,
    radial_ranges,
    theta_step_deg=1.0,
    ):
    """
    Create an 3D or 4D orientation histogram from a braggpeaks PointListArray,
    from user-specified radial ranges.
    
    Args:
        braggpeaks (PointListArray):    2D of pointlists containing centered peak locations.
        radial_ranges (np array):       [N x 2] array for N radial bins
        theta_step_deg (float):         

    Returns:
        orient_hist (array):          3D or 4D array containing Bragg peaks
    """
    
    print(braggpeaks.shape)

    orient_hist = 0

    return orient_hist

