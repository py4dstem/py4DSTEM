# Functions for differential phase contrast imaging

import numpy as np
from ...file.datastructure import DataCube

############################# DPC Functions ################################

def get_CoM_images(datacube, mask):
    """
    Calculates two images - center of mass x and y - from a 4D-STEM datacube.

    The centers of mass are calculated with respect to the uncalibrated coordinate system of the
    detector - i.e. pixel sizes are unity, and no rotation or shifting of the origin are performed.
    CoM images in different (rotated, shifted, rescaled) coordinate systems can be obtained by
    passing this functions output into change_CoM_coords().

    Accepts:
        datacube        (DataCube) the 4D-STEM data
    """
    assert isinstance(datacube, DataCube)

    qy,qx = np.meshgrid(np.arange(datacube.Q_Ny),np.arange(datacube.Q_Nx))
    mass = np.sum(datacube.data4D, axis=(2,3))
    CoMx = np.sum(dc.data4D*qx[np.newaxis,np.newaxis,:,:],axis=(2,3)) / mass
    CoMy = np.sum(dc.data4D*qy[np.newaxis,np.newaxis,:,:],axis=(2,3)) / mass

    return CoMx, CoMy

def change_CoM_coords(params):
    pass

def get_phase_from_CoM(CoMx, CoMy, padding=0):
    pass




#################### Functions for constructing the e-beam #################

def construct_illumation(shape, size, keV, aperture, ap_in_mrad=True,
                         df=0, cs=0, c5=0, return_qspace=False):
    """
    Makes a probe wave function, in the sample plane.

    The arguments shape and size together describe a rectangular region in which the
    illumination of the beam is calculated, with the probe placed at the center of this region.
    size gives the region size (xsize,ysize), in units of Angstroms.
    shape describes the sampling (Nx,Ny), i.e. the number of pixels spanning the grid, in the x
    and y directions.

    Accepts:
        shape           (2-tuple of ints) the number of pixels (Nx,Ny) making grid in which
                        the illumination is calculated.
        size            (2-tuple of floats) the size (xsize,ysize) of the grid, in real space.
        keV             (float) the energe of the probe electrons, in keV
        aperture        (float) the probe forming aperture size. Units are specified by ap_in_mrad.
        ap_in_mrad      (bool) if True, aperture describes the aperture size in mrads, i.e. it
                        specifies the convergence semi-angle.
                        If False, aperture describes the aperture size in inverse Angstroms
        df              (float) probe defocus, in Angstroms, with negative values corresponding to
                        overfocus.
        cs              (float) the 3rd order spherical aberration coefficient, in mm
        c5              (float) the 5th order spherical aberration coefficient, in mm
        return_qspace   (bool) if True, return the probe in the diffraction plane, rather than the
                        sample plane.
    """
    # Get shapes
    Nx,Ny = shape
    xsize,ysize = size

    # Get diffraction space coordinates
    qsize = (float(Nx)/xsize,float(Ny)/ysize)
    qx,qy = make_qspace_coords(shape, qsize)
    qr = np.sqrt(qx**2 + qy**2)

    # Get electron wavenumber and aperture size
    k = get_wavenumber(keV*1000)
    if ap_in_mrad is True:
        aperture = np.tan(aperture/1000)*k

    # Get the probe
    probe = np.asarray(qr<=aperture, dtype=complex)                          # Initialize probe
    probe *= np.exp(-1j*sph_aberration(qr, lam=1.0/k, df=df, cs=cs, c5=c5))  # Add aberrations
    if return_qspace is True:
        return probe
    probe = np.fft.ifft2(probe)                                         # Convert to real space
    probe /= np.sqrt(np.sum(np.square(np.abs(probe))))                  # Normalize
    return probe

def sph_aberration(qr, lam, df=0, cs=0, c5=0):
    """
    Calculates the aberration function chi as a function of diffraction space radial coordinates qr
    for an electron with wavelength lam.

    Note that this function only considers the rotationally symmetric terms of chi (i.e. spherical
    aberration) up to 5th order.  Non-rotationally symmetric terms (coma, stig, etc) and higher
    order terms (c7, etc) are not considered.

    Accepts:
        qr      (float or array) diffraction space radial coordinate(s), in inverse Angstroms
        lam     (float) wavelength of electron, in Angstroms
        df      (float) probe defocus, in Angstroms
        cs      (float) probe 3rd order spherical aberration coefficient, in mm
        c5      (float) probe 5th order spherical aberration coefficient, in mm

    Returns:
        chi     (float) the aberation function
    """
    p = lam*qr
    chi = df*np.square(p)/2.0 + cs*1e7*np.power(p,4)/4.0 + c5*1e7*np.power(p,6)/6.0
    chi = 2*np.pi*chi/lam
    return chi


##################### Electron physics functions ########################

def get_relativistic_mass_correction(E):
    """
    Calculates the relativistic mass correction (i.e. the Lorentz factor, gamma) for an electron
    with kinetic energy E, in eV.
    See, e.g., Kirkland, 'Advanced Computing in Electron Microscopy', Eq. 2.2.

    Accepts:
        E       (float) electron energy, in eV

    Returns:
        gamma   (float) relativistic mass correction factor
    """
    m0c2 = 5.109989461e5    # electron rest mass, in eV
    return (m0c2 + E)/m0c2

def get_wavenumber(E):
    """
    Calculates the relativistically corrected wavenumber k0 (reciprocal of wavelength) for an
    electron with kinetic energy E, in eV.
    See, e.g., Kirkland, 'Advanced Computing in Electron Microscopy', Eq. 2.5.

    Accepts:
        E       (float) electron energy, in eV

    Returns:
        k0      (float) relativistically corrected wavenumber
    """
    hc = 1.23984193e4       # Planck's constant times the speed of light in eV Angstroms
    m0c2 = 5.109989461e5    # electron rest mass, in eV
    return np.sqrt( E*(E + 2*m0c2) ) / hc

def get_interaction_constant(E):
    """
    Calculates the interaction constant, sigma, to convert electrostatic potential (in V Angstroms)
    to radians. Units of this constant are rad/(V Angstrom).
    See, e.g., Kirkland, 'Advanced Computing in Electron Microscopy', Eq. 2.5.

    Accepts:
        E       (float) electron energy, in eV

    Returns:
        m       (float) relativistically corrected electron mass
    """
    h = 6.62607004e-34      # Planck's constant in Js
    me = 9.10938356e-31     # Electron rest mass in kg
    qe = 1.60217662e-19     # Electron charge in C
    k0 = get_wavenumber(E)           # Electron wavenumber in inverse Angstroms
    gamma = get_relativistic_mass_correction(E)   # Relativistic mass correction
    return 2*np.pi*gamma*me*qe/(k0*1e-20*h**2)



####################### Utility functions ##########################

def make_qspace_coords(shape,qsize):
    """
    Creates a diffraction space coordinate grid.

    Number of pixels in the grid (sampling) is given by shape = (Nx,Ny).
    Extent of the grid is given by qsize = (xsize,ysize), where xsize,ysize are in inverse length
    units, and are the number of pixels divided by the real space size.

    Accepts:
        shape       (2-tuple of ints) grid shape
        qsize       (2-tuple of floats) grid size, in reciprocal length units

    Returns:
        qx          (2D ndarray) the x diffraction space coordinates
        qy          (2D ndarray) the y diffraction space coordinates
    """
    qx = np.fft.fftfreq(shape[0])*qsize[0]
    qy = np.fft.fftfreq(shape[1])*qsize[1]
    return qx,qy

def pad_shift(ar, shift_x, shift_y):
    """
    Similar to np.roll, but designed for special handling of zero padded matrices.

    In particular, for a zero-padded matrix ar and shift values (shift_x,shift_y) which are equal to
    or less than the pad width, pad_shift is identical to np.roll.
    For a zero-padded matrix ar and shift values (shift_x,shift_y) which are greater than the pad
    width, values of ar which np.roll would 'wrap around' are instead set to zero.

    For a 1D analog, np.roll and pad_shift are identical in the first case, but differ in the second:

    Case 1:
        np.roll(np.array([0,0,1,1,1,0,0],2) = array([0,0,0,0,1,1,1])
        pad_shift(np.array([0,0,1,1,1,0,0],2) = array([0,0,0,0,1,1,1])

    Case 2:
        np.roll(np.array([0,0,1,1,1,0,0],3) = array([1,0,0,0,0,1,1])
        pad_shift(np.array([0,0,1,1,1,0,0],3) = array([0,0,0,0,0,1,1])

    Accepts:
        ar          (ndarray) a 2D array
        shift_x     (int) the x shift
        shift_y     (int) the y shift

    Returns:
        shifted_ar  (ndarray) the shifted array
    """
    assert isinstance(shift_x,(int,np.integer))
    assert isinstance(shift_y,(int,np.integer))

    xend,yend = np.shape(ar)
    xend,yend = xend-x,yend-y

    return np.pad(ar, ((x*(x>=0),-x*(x<=0)),(y*(y>=0),-y*(y<=0))),
                  mode='constant')[-x*(x<=0):-x*(x>=0)+xend*(x<=0), \
                                   -y*(y<=0):-y*(y>=0)+yend*(y<=0)]

def rotate_point(origin, point, angle):
    """
    Rotates point counterclockwise by angle about origin.

    Accepts:
        origin          (2-tuple of floats) the (x,y) coords of the origin
        point           (2-tuple of floats) the (x,y) coords of the point
        angle           (float) the rotation angle, in radians

    Returns:
        rotated_point   (2-tuple of floats) the (x,y) coords of the rotated point
    """
    ox,oy = origin
    px,py = point

    qx = ox + np.cos(angle)*(px-ox) - np.sin(angle)*(py-oy)
    qy = oy + np.sin(angle)*(px-ox) + np.cos(angle)*(py-oy)

    return qx,qy



