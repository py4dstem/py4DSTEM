# Defines utility functions used by other functions in the /process/ directory.

import numpy as np

def make_Fourier_coords1D(N, pixelSize=1):
    """
    Generates Fourier coordinates for a 1D array of length N.
	Specifying the pixelSize argument sets a unit size.
    """
    if N%2 == 0:
        q = np.roll( np.arange(-N/2,N/2)/(N*pixelSize), int(N/2))
    else:
        q = np.roll( np.arange((1-N)/2,(N+1)/2)/(N*pixelSize), int((1-N)/2))
    return q

def make_Fourier_coords2D(Nx, Ny, pixelSize=1):
    """
    Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
	Specifying the pixelSize argument sets a unit size.
	"""
    qx = make_Fourier_coords1D(Nx,pixelSize)
    qy = make_Fourier_coords1D(Ny,pixelSize)
    qy,qx = np.meshgrid(qy,qx)
    return qx,qy

def get_shift(ar1,ar2):
    """
	Determined the relative shift between a pair of identical arrays, or the shift giving
	best overlap.
	Determines best shift in the simplest way, using the brightest pixel in the cross
	correlation, and is limited to pixel resolution.

	Inputs:
		ar1,ar2	-	2D ndarrays
	Outputs:
		shiftx,shifty - relative image shift, in pixels
    """
    cc = np.fft.ifft2(np.fft.fft2(ar1)*np.conj(np.fft.fft2(ar2)))
    xshift,yshift = np.unravel_index(np.argmax(cc),ar1.shape)
    return xshift,yshift

def get_shifted_ar(ar,xshift,yshift):
    """
	Shifts array ar by the shift vector (xshift,yshift), using the Fourier shift theorem (i.e.
	with sinc interpolation).
    """
    nx,ny = np.shape(ar)
    qx,qy = make_Fourier_coords2D(nx,ny,1)
    nx,ny = float(nx),float(ny)

    w = np.exp(-(2j*np.pi)*( (yshift*qy) + (xshift*qx) ))
    shifted_ar = np.real(np.fft.ifft2((np.fft.fft2(ar))*w))
    return shifted_ar

def get_cross_correlation(ar, kernel, corrPower=1):
    """
    Calculates the cross correlation of ar with kernel.
    corrPower specifies the correlation type, where 1 is a cross correlation, 0 is a phase
    correlation, and values in between are hybrids.
    """
    m = np.fft.fft2(ar) * np.conj(np.fft.fft2(kernel))
    return np.real(np.fft.ifft2(np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))))

def get_cross_correlation_fk(ar, fourierkernel, corrPower=1):
    """
    Calculates the cross correlation of ar with fourierkernel.
    Here, fourierkernel = np.conj(np.fft.fft2(kernel)); speeds up computation when the same
    kernel is to be used for multiple cross correlations.
    corrPower specifies the correlation type, where 1 is a cross correlation, 0 is a phase
    correlation, and values in between are hybrids.
    """
    m = np.fft.fft2(ar) * fourierkernel
    return np.real(np.fft.ifft2(np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))))

def get_CoM(ar):
    """
    Finds and returns the center of mass of array ar.
    """
    nx,ny=np.shape(ar)
    ry,rx = np.meshgrid(np.arange(ny),np.arange(nx))
    tot_intens = np.sum(ar)
    xCoM = np.sum(rx*ar)/tot_intens
    yCoM = np.sum(ry*ar)/tot_intens
    return xCoM,yCoM

def get_maximal_points(ar):
    """
    Returns the points in an array with values larger than all 8 of their nearest neighbors.
    """
    return (ar>np.roll(ar,(-1,0),axis=(0,1))) & (ar>np.roll(ar,(1,0),axis=(0,1))) & \
           (ar>np.roll(ar,(0,-1),axis=(0,1))) & (ar>np.roll(ar,(0,1),axis=(0,1))) & \
           (ar>np.roll(ar,(-1,-1),axis=(0,1))) & (ar>np.roll(ar,(-1,1),axis=(0,1))) & \
           (ar>np.roll(ar,(1,-1),axis=(0,1))) & (ar>np.roll(ar,(1,1),axis=(0,1)))

def linear_interpolation_1D(ar,x):
    """
    Calculates the 1D linear interpolation of array ar at position x using the two nearest elements.
    """
    x0,x1 = int(np.floor(x)),int(np.ceil(x))
    if x0==x1:
        return ar[x0]
    else:
        dx = x-x0
        return (1-dx)*ar[x0] + dx*ar[x1]

def radial_integral(ar, x0, y0):
    """
    Computes the radial integral of array ar from center x0,y0.

    Based on efficient radial profile code found at www.astrobetter.com/wiki/python_radial_profiles,
    with credit to Jessica R. Lu, Adam Ginsburg, and Ian J. Crossfield.
    """
    y,x = np.meshgrid(np.arange(ar.shape[1]),np.arange(ar.shape[0]))
    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    # Get sorted radii and ar values
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    vals_sorted = ar.flat[ind]

    # Cast to int (i.e. set binsize = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels within each radial bin
    delta_r = r_int[1:] - r_int[:-1]
    rind = np.where(delta_r)[0]       # Gives nonzero elements of delta_r, i.e. where radius changes
    nr = rind[1:] - rind[:-1]         # Number of pixels represented in each bin

    # Cumulative sum in each radius bin
    cs_vals = np.cumsum(vals_sorted, dtype=float)
    bin_sum = cs_vals[rind[1:]] - cs_vals[rind[:-1]]

    return bin_sum / nr, bin_sum, nr, rind





