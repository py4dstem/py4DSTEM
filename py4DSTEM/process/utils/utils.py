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



