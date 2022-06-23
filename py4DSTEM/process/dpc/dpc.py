# Functions for differential phase contrast imaging

import numpy as np
from ..utils import make_Fourier_coords2D, print_progress_bar
from ...io import DataCube

############################# DPC Functions ################################

def get_CoM_images(datacube, mask=None, normalize=True):
    """
    Calculates two images - center of mass x and y - from a 4D-STEM datacube.

    The centers of mass are returned in units of pixels and in the Qx/Qy detector
    coordinate system.

    Args:
        datacube (DataCube): the 4D-STEM data
        mask (2D array): optionally, calculate the CoM only in the areas where mask==True
        normalize (bool): if true, subtract off the mean of the CoM images

    Returns:
        (2-tuple of 2d arrays): the center of mass coordinates, (x,y)
    """
    assert isinstance(datacube, DataCube)
    assert isinstance(normalize, bool)

    # Coordinates
    qy,qx = np.meshgrid(np.arange(datacube.Q_Ny),np.arange(datacube.Q_Nx))
    if mask is not None:
        qx *= mask
        qy *= mask

    # Get CoM
    CoMx = np.zeros((datacube.R_Nx,datacube.R_Ny))
    CoMy = np.zeros((datacube.R_Nx,datacube.R_Ny))
    mass = np.zeros((datacube.R_Nx,datacube.R_Ny))
    for Rx in range(datacube.R_Nx):
        for Ry in range(datacube.R_Ny):
            DP = datacube.data[Rx,Ry,:,:]
            mass[Rx,Ry] = np.sum(DP*mask)
            CoMx[Rx,Ry] = np.sum(qx*DP) / mass[Rx,Ry]
            CoMy[Rx,Ry] = np.sum(qy*DP) / mass[Rx,Ry]

    if normalize:
        CoMx -= np.mean(CoMx)
        CoMy -= np.mean(CoMy)

    return CoMx, CoMy

def get_rotation_and_flip_zerocurl(CoMx, CoMy, Q_Nx, Q_Ny, n_iter=100, stepsize=1,
                                                           return_costs=False):
    """
    Find the rotation offset between real space and diffraction space, and whether there
    exists a relative axis flip their coordinate systems, starting from the premise that
    the CoM changes must have zero curl everywhere.

    The idea of the algorithm is to find the rotation which best preserves
    self-consistency in the observed CoM changes.  By 'self-consistency', we refer to the
    requirement that the CoM changes - because they correspond to a gradient - must be a
    conservative vector field (i.e. path independent).  This condition fails to be met
    when there exists some rotational offset between real and diffraction space. Thus
    this algorithm performs gradient descent to minimize the square of the sums of all
    the 4-pixel closed loop line integrals, while varying the rotation angle of the
    diffraction space (CoMx/CoMy) axes.

    Args:
        CoMx (2D array): the x coordinates of the diffraction space centers of mass
        CoMy (2D array): the y coordinates of the diffraction space centers of mass
        Q_Nx (int): the x shape of diffraction space
        Q_Ny (int): the y shape of diffraction space
        n_iter (int): the number of gradient descent iterations
        stepsize (float): the gradient descent step size (i.e. change to theta in a
            single step, relative to the gradient)
        return_costs (bool): if True, returns the theta values and costs, both with and
            without an axis flip, for all gradient descent steps, for diagnostic purposes

    Returns:
        (variable): If ``return_costs==False`` (default), returns a 2-tuple consisting of

            * **theta**: *(float)* the rotation angle between the real and diffraction
              space coordinate, in radians
            * **flip**: *(bool)* if True, the real and diffraction space coordinates are
              flipped relative to one another.  By convention, we take flip=True to
              correspond to the change CoMy --> -CoMy.

        If ``return_costs==True``, returns a 6-tuple consisting of the two values listed
        above, followed by:

            * **thetas**: *(float)* returned iff return_costs is True. The theta values
              at each gradient descent step for flip=False. In radians.
        * **costs**: *(float)* returned iff return_costs is True. The cost values at each
          gradient descent step for flip=False
        * **thetas_f**: *(float)* returned iff return_costs is True. The theta values for
          flip=True descent step for flip=False
        * **costs_f**: *(float)* returned iff return_costs is True. The cost values for
          flip=False
    """
    # Cost function coefficients, with / without flip
    term1 = np.roll(CoMx,(0,-1),axis=(0,1)) - np.roll(CoMx,( 0,+1),axis=(0,1)) + \
            np.roll(CoMy,(1, 0),axis=(0,1)) - np.roll(CoMy,(-1, 0),axis=(0,1))
    term2 = np.roll(CoMx,(1, 0),axis=(0,1)) - np.roll(CoMx,(-1, 0),axis=(0,1)) + \
            np.roll(CoMy,(0, 1),axis=(0,1)) - np.roll(CoMy,( 0,-1),axis=(0,1))

    term1_f = np.roll( CoMx,(0,-1),axis=(0,1)) - np.roll( CoMx,( 0,+1),axis=(0,1)) + \
              np.roll(-CoMy,(1, 0),axis=(0,1)) - np.roll(-CoMy,(-1, 0),axis=(0,1))
    term2_f = np.roll( CoMx,(1, 0),axis=(0,1)) - np.roll( CoMx,(-1, 0),axis=(0,1)) + \
              np.roll(-CoMy,(0, 1),axis=(0,1)) - np.roll(-CoMy,( 0,-1),axis=(0,1))

    # Gradient descent

    thetas = np.zeros(n_iter)
    costs = np.zeros(n_iter)
    theta = 0
    for i in range(n_iter):
        thetas[i] = theta
        gradAll = stepsize * ( term1*np.cos(theta) + term2*np.sin(theta)) * \
                             (-term1*np.sin(theta) + term2*np.cos(theta))
        grad = np.mean(gradAll)
        theta -= grad*stepsize
        costs[i] = np.mean((term1*np.cos(theta) + term2*np.sin(theta))**2)

    thetas_f = np.zeros(n_iter)
    costs_f = np.zeros(n_iter)
    theta = 0
    for i in range(n_iter):
        thetas_f[i] = theta
        gradAll = stepsize * ( term1_f*np.cos(theta) + term2_f*np.sin(theta)) * \
                             (-term1_f*np.sin(theta) + term2_f*np.cos(theta))
        grad = np.mean(gradAll)
        theta -= grad*stepsize
        costs_f[i] = np.mean((term1_f*np.cos(theta) + term2_f*np.sin(theta))**2)

    # Get rotation and flip
    if costs_f[-1] < costs[-1]:
        flip = True
        theta = thetas_f[-1]
    else:
        flip = False
        theta = thetas[-1]

    if return_costs:
        return theta, flip, thetas, costs, thetas_f, costs_f
    else:
        return theta, flip

def get_rotation_and_flip_maxcontrast(CoMx, CoMy, N_thetas, paddingfactor=2,
                                      regLowPass=0.5, regHighPass=100, stepsize=1,
                                      n_iter=1, return_stds=False, verbose=True):
    """
    Find the rotation offset between real space and diffraction space, and whether there
    exists a relative axis flip their coordinate systems, starting from the premise that
    the contrast of the phase reconstruction should be maximized when the RQ rotation is
    correctly set.

    The idea of the algorithm is to perform a phase reconstruction for various values of
    the RQ rotation, and with and without an RQ flip, and then calculate the standard
    deviation of the resulting images.  The rotation and flip which maximize the standard
    deviation are then returned. Note that answer should be correct up to a 180 degree
    rotation, corresponding to a complete contrast reversal.  From these two options, the
    correct rotation can then be selected manually by noting that for the correct
    rotation, atomic sites should be bright and the absence of atoms dark.  Physically,
    the presence of two degenerate solutions is related to the electron charge,
    with the incorrect, contrast reversed solution corresponding to electrons with a
    charge of +e.

    Args:
        CoMx (2D array): the x coordinates of the diffraction space centers of mass
        CoMy (2D array): the y coordinates of the diffraction space centers of mass
        N_thetas (int): the number of theta values to use
        regLowPass (float): passed to get_phase_from_CoM; low pass regularization term
            for the Fourier integration operators
        regHighPass (float): passed to get_phase_from_CoM; high pass regularization term
            for the Fourier integration operators
        paddingfactor (int): passed to get_phase_from_CoM; padding to add to the CoM
            arrays for boundry condition handling. 1 corresponds to no padding, 2 to
            doubling the array size, etc.
        stepsize (float): passed to get_phase_from_CoM; the stepsize in the iteration
            step which updates the phase
        n_iter (int): passed to get_phase_from_CoM; the number of iterations
        return_stds (bool): if True, returns the theta values and costs, both with and
            without an axis flip, for all gradient descent steps, for diagnostic purposes
        verbose (bool): if True, display progress bar during calculation

    Returns:
        (5-tuple) A 5-tuple containing:

            * **theta**: *(float)* the rotation angle between the real and diffraction
              space coordinates, in radians.
            * **flip**: *(bool)* if True, the real and diffraction space coordinates are
              flipped relative to one another.  By convention, we take flip=True to
              correspond to the change CoMy --> -CoMy.
            * **thetas**: *(float)* returned iff return_costs is True. The theta values.
              In radians.
            * **stds**: *(float)* returned iff return_costs is True. The cost values at
              each gradient descent step for flip=False
            * **stds_f**: *(float)* returned iff return_costs is True. The cost values
              for flip=False
    """
    thetas = np.linspace(0,2*np.pi,N_thetas)
    stds = np.zeros(N_thetas)
    stds_f = np.zeros(N_thetas)

    # Unflipped
    for i,theta in enumerate(thetas):
        phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=False,
                                          regLowPass=regLowPass, regHighPass=regHighPass,
                                          paddingfactor=paddingfactor, stepsize=stepsize,
                                          n_iter=n_iter)
        stds[i] = np.std(phase)
        if verbose:
            print_progress_bar(i+1, 2*N_thetas, prefix='Analyzing:', suffix='Complete.', length=50)

    # Flipped
    for i,theta in enumerate(thetas):
        phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=True,
                                          regLowPass=regLowPass, regHighPass=regHighPass,
                                          paddingfactor=paddingfactor, stepsize=stepsize,
                                          n_iter=n_iter)
        stds_f[i] = np.std(phase)
        if verbose:
            print_progress_bar(N_thetas+i+1, 2*N_thetas, prefix='Analyzing:', suffix='Complete.', length=50)

    flip = np.max(stds_f)>np.max(stds)
    if flip:
        theta = thetas[np.argmax(stds_f)]
    else:
        theta = thetas[np.argmax(stds)]

    if return_stds:
        return theta, flip, thetas, stds, stds_f
    else:
        return theta, flip

def get_phase_from_CoM(CoMx, CoMy, theta, flip, regLowPass=0.5, regHighPass=100,
                        paddingfactor=2, stepsize=1, n_iter=10, phase_init=None):
    """
    Calculate the phase of the sample transmittance from the diffraction centers of mass.
    A bare bones description of the approach taken here is below - for detailed
    discussion of the relevant theory, see, e.g.::

        Ishizuka et al, Microscopy (2017) 397-405
        Close et al, Ultramicroscopy 159 (2015) 124-137
        Wadell and Chapman, Optik 54 (1979) No. 2, 83-96

    The idea here is that the deflection of the center of mass of the electron beam in
    the diffraction plane scales linearly with the gradient of the phase of the sample
    transmittance. When this correspondence holds, it is therefore possible to invert the
    differential equation and extract the phase itself.* The primary assumption made is
    that the sample is well described as a pure phase object (i.e. the real part of the
    transmittance is 1). The inversion is performed in this algorithm in Fourier space,
    i.e. using the Fourier transform property that derivatives in real space are turned
    into multiplication in Fourier space.

    *Note: because in DPC a differential equation is being inverted - i.e. the
    fundamental theorem of calculus is invoked - one might be tempted to call this
    "integrated differential phase contrast".  Strictly speaking, this term is redundant
    - performing an integration is simply how DPC works.  Anyone who tells you otherwise
    is selling something.

    Args:
        CoMx (2D array): the diffraction space centers of mass x coordinates
        CoMy (2D array): the diffraction space centers of mass y coordinates
        theta (float): the rotational offset between real and diffraction space
            coordinates
        flip (bool): whether or not the real and diffraction space coords contain a
                        relative flip
        regLowPass (float): low pass regularization term for the Fourier integration
            operators
        regHighPass (float): high pass regularization term for the Fourier integration
            operators
        paddingfactor (int): padding to add to the CoM arrays for boundry condition
            handling. 1 corresponds to no padding, 2 to doubling the array size, etc.
        stepsize (float): the stepsize in the iteration step which updates the phase
        n_iter (int): the number of iterations
        phase_init (2D array): initial guess for the phase

    Returns:
        (2-tuple) A 2-tuple containing:

            * **phase**: *(2D array)* the phase of the sample transmittance, in radians
            * **error**: *(1D array)* the error - RMSD of the phase gradients compared
              to the CoM - at each iteration step
    """
    assert isinstance(flip,(bool,np.bool_))
    assert isinstance(paddingfactor,(int,np.integer))
    assert isinstance(n_iter,(int,np.integer))

    # Coordinates
    R_Nx,R_Ny = CoMx.shape
    R_Nx_padded,R_Ny_padded = R_Nx*paddingfactor,R_Ny*paddingfactor

    qx = np.fft.fftfreq(R_Nx_padded)
    qy = np.fft.rfftfreq(R_Ny_padded)
    qr2 = qx[:,None]**2 + qy[None,:]**2

    # Inverse operators
    denominator = qr2 + regHighPass + qr2**2*regLowPass
    _ = np.seterr(divide='ignore')
    denominator = 1./denominator
    denominator[0,0] = 0
    _ = np.seterr(divide='warn')
    f = 1j * -0.25*stepsize
    qxOperator = f*qx[:,None]*denominator
    qyOperator = f*qy[None,:]*denominator

    # Perform rotation and flipping
    if not flip:
        CoMx_rot = CoMx*np.cos(theta) - CoMy*np.sin(theta)
        CoMy_rot = CoMx*np.sin(theta) + CoMy*np.cos(theta)
    if flip:
        CoMx_rot = CoMx*np.cos(theta) + CoMy*np.sin(theta)
        CoMy_rot = CoMx*np.sin(theta) - CoMy*np.cos(theta)

    # Initializations
    phase = np.zeros((R_Nx_padded,R_Ny_padded))
    update = np.zeros((R_Nx_padded,R_Ny_padded))
    dx = np.zeros((R_Nx_padded,R_Ny_padded))
    dy = np.zeros((R_Nx_padded,R_Ny_padded))
    error = np.zeros(n_iter)
    mask = np.zeros((R_Nx_padded,R_Ny_padded),dtype=bool)
    mask[:R_Nx,:R_Ny] = True
    maskInv = mask==False
    if phase_init is not None:
        phase[:R_Nx,:R_Ny] = phase_init

    # Iterative reconstruction
    for i in range(n_iter):

        # Update gradient estimates using measured CoM values
        dx[mask] -= CoMx_rot.ravel()
        dy[mask] -= CoMy_rot.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0

        # Calculate reconstruction update
        update = np.fft.irfft2( np.fft.rfft2(dx)*qxOperator + np.fft.rfft2(dy)*qyOperator)

        # Apply update
        phase += stepsize*update

        # Measure current phase gradients
        dx = (np.roll(phase,(-1,0),axis=(0,1)) - np.roll(phase,(1,0),axis=(0,1))) / 2.
        dy = (np.roll(phase,(0,-1),axis=(0,1)) - np.roll(phase,(0,1),axis=(0,1))) / 2.

        # Estimate error from cost function, RMS deviation of gradients
        xDiff = dx[mask] - CoMx_rot.ravel()
        yDiff = dy[mask] - CoMy_rot.ravel()
        error[i] = np.sqrt(np.mean((xDiff-np.mean(xDiff))**2 + (yDiff-np.mean(yDiff))**2))

        # Halve step size if error is increasing
        if i>0:
            if error[i] > error[i-1]:
                stepsize /= 2

    phase = phase[:R_Nx,:R_Ny]

    return phase, error


#################### Functions for constructing the e-beam #################

def construct_illumation(shape, size, keV, aperture, ap_in_mrad=True,
                         df=0, cs=0, c5=0, return_qspace=False):
    """
    Makes a probe wave function, in the sample plane.

    The arguments shape and size together describe a rectangular region in which the
    illumination of the beam is calculated, with the probe placed at the center of this
    region. size gives the region size (xsize,ysize), in units of Angstroms.
    shape describes the sampling (Nx,Ny), i.e. the number of pixels spanning the grid,
    in the x and y directions.

    Args:
        shape (2-tuple of ints): the number of pixels (Nx,Ny) making grid in which
            the illumination is calculated.
        size (2-tuple of floats): the size (xsize,ysize) of the grid, in real space.
        keV (float): the energe of the probe electrons, in keV
        aperture (float): the probe forming aperture size. Units are specified by
            ap_in_mrad.
        ap_in_mrad (bool): if True, aperture describes the aperture size in mrads, i.e.
            it specifies the convergence semi-angle. If False, aperture describes the
            aperture size in inverse Angstroms
        df (float): probe defocus, in Angstroms, with negative values corresponding to
            overfocus.
        cs (float): the 3rd order spherical aberration coefficient, in mm
        c5 (float): the 5th order spherical aberration coefficient, in mm
        return_qspace (bool): if True, return the probe in the diffraction plane, rather
            than the sample plane.
    """
    # Get shapes
    Nx,Ny = shape
    xsize,ysize = size

    # Get diffraction space coordinates
    qsize = (float(Nx)/xsize,float(Ny)/ysize)
    qx, qy = make_Fourier_coords2D(qsize[0],qsize[1])
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
    Calculates the aberration function chi as a function of diffraction space radial
    coordinates qr for an electron with wavelength lam.

    Note that this function only considers the rotationally symmetric terms of chi (i.e.
    spherical aberration) up to 5th order.  Non-rotationally symmetric terms (coma, stig,
    etc) and higher order terms (c7, etc) are not considered.

    Args:
        qr (float or array): diffraction space radial coordinate(s), in inverse Angstroms
        lam (float): wavelength of electron, in Angstroms
        df (float): probe defocus, in Angstroms
        cs (float): probe 3rd order spherical aberration coefficient, in mm
        c5 (float): probe 5th order spherical aberration coefficient, in mm

    Returns:
        (float): the aberation function
    """
    p = lam*qr
    chi = df*np.square(p)/2.0 + cs*1e7*np.power(p,4)/4.0 + c5*1e7*np.power(p,6)/6.0
    chi = 2*np.pi*chi/lam
    return chi


##################### Electron physics functions ########################

def get_relativistic_mass_correction(E):
    """
    Calculates the relativistic mass correction (i.e. the Lorentz factor, gamma) for an
    electron with kinetic energy E, in eV. See, e.g., Kirkland, 'Advanced Computing in
    Electron Microscopy', Eq. 2.2.

    Args:
        E (float): electron energy, in eV

    Returns:
        (float): relativistic mass correction factor
    """
    m0c2 = 5.109989461e5    # electron rest mass, in eV
    return (m0c2 + E)/m0c2

def get_wavenumber(E):
    """
    Calculates the relativistically corrected wavenumber k0 (reciprocal of wavelength)
    for an electron with kinetic energy E, in eV. See, e.g., Kirkland, 'Advanced
    Computing in Electron Microscopy', Eq. 2.5.

    Args:
        E (float): electron energy, in eV

    Returns:
        (float): relativistically corrected wavenumber
    """
    hc = 1.23984193e4       # Planck's constant times the speed of light in eV Angstroms
    m0c2 = 5.109989461e5    # electron rest mass, in eV
    return np.sqrt( E*(E + 2*m0c2) ) / hc

def get_interaction_constant(E):
    """
    Calculates the interaction constant, sigma, to convert electrostatic potential (in
    V Angstroms) to radians. Units of this constant are rad/(V Angstrom).
    See, e.g., Kirkland, 'Advanced Computing in Electron Microscopy', Eq. 2.5.

    Args:
        E (float): electron energy, in eV

    Returns:
        (float): relativistically corrected electron mass
    """
    h = 6.62607004e-34      # Planck's constant in Js
    me = 9.10938356e-31     # Electron rest mass in kg
    qe = 1.60217662e-19     # Electron charge in C
    k0 = get_wavenumber(E)           # Electron wavenumber in inverse Angstroms
    gamma = get_relativistic_mass_correction(E)   # Relativistic mass correction
    return 2*np.pi*gamma*me*qe*1e-20/(k0*h**2)


####################### Utility functions ##########################


# def pad_shift(ar, x, y):
#     """
#     Similar to np.roll, but designed for special handling of zero padded matrices.
# 
#     In particular, for a zero-padded matrix ar and shift values (x,y) which are equal to
#     or less than the pad width, pad_shift is identical to np.roll.
#     For a zero-padded matrix ar and shift values (x,y) which are greater than the pad
#     width, values of ar which np.roll would 'wrap around' are instead set to zero.
# 
#     For a 1D analog, np.roll and pad_shift are identical in the first case, but differ in the second:
# 
#     Case 1:
#         np.roll(np.array([0,0,1,1,1,0,0],2) = array([0,0,0,0,1,1,1])
#         pad_shift(np.array([0,0,1,1,1,0,0],2) = array([0,0,0,0,1,1,1])
# 
#     Case 2:
#         np.roll(np.array([0,0,1,1,1,0,0],3) = array([1,0,0,0,0,1,1])
#         pad_shift(np.array([0,0,1,1,1,0,0],3) = array([0,0,0,0,0,1,1])
# 
#     Accepts:
#         ar          (ndarray) a 2D array
#         x           (int) the x shift
#         y           (int) the y shift
# 
#     Returns:
#         shifted_ar  (ndarray) the shifted array
#     """
#     assert isinstance(x,(int,np.integer))
#     assert isinstance(y,(int,np.integer))
# 
#     xend,yend = np.shape(ar)
#     xend,yend = xend-x,yend-y
# 
#     return np.pad(ar, ((x*(x>=0),-x*(x<=0)),(y*(y>=0),-y*(y<=0))),
#                   mode='constant')[-x*(x<=0):-x*(x>=0)+xend*(x<=0), \
#                                    -y*(y<=0):-y*(y>=0)+yend*(y<=0)]
# 
# def rotate_point(origin, point, angle):
#     """
#     Rotates point counterclockwise by angle about origin.
# 
#     Accepts:
#         origin          (2-tuple of floats) the (x,y) coords of the origin
#         point           (2-tuple of floats) the (x,y) coords of the point
#         angle           (float) the rotation angle, in radians
# 
#     Returns:
#         rotated_point   (2-tuple of floats) the (x,y) coords of the rotated point
#     """
#     ox,oy = origin
#     px,py = point
# 
#     qx = ox + np.cos(angle)*(px-ox) - np.sin(angle)*(py-oy)
#     qy = oy + np.sin(angle)*(px-ox) + np.cos(angle)*(py-oy)
# 
#     return qx,qy



