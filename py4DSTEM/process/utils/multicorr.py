'''
loosely based on multicorr.py found at:
https://github.com/ercius/openNCEM/blob/master/ncempy/algo/multicorr.py

modified by SEZ, May 2019 to integrate with py4DSTEM utility functions
 * rewrote upsampleFFT (previously did not work correctly)
 * modified upsampled_correlation to accept xyShift, the point around which to
 upsample the DFT
 * eliminated the factor-2 FFT upsample step in favor of using parabolic
 for first-pass subpixel (since parabolic is so fast)
 * rewrote the matrix multiply DFT to be more pythonic
'''

import numpy as np

def upsampled_correlation(imageCorr, upsampleFactor, xyShift):
    '''
    Refine the correlation peak of imageCorr around xyShift by DFT upsampling.

    There are two approaches to Fourier upsampling for subpixel refinement: (a) one
    can pad an (appropriately shifted) FFT with zeros and take the inverse transform,
    or (b) one can compute the DFT by matrix multiplication using modified
    transformation matrices. The former approach is straightforward but requires
    performing the FFT algorithm (which is fast) on very large data. The latter method
    trades one speedup for a slowdown elsewhere: the matrix multiply steps are expensive
    but we operate on smaller matrices. Since we are only interested in a very small
    region of the FT around a peak of interest, we use the latter method to get
    a substantial speedup and enormous decrease in memory requirement. This
    "DFT upsampling" approach computes the transformation matrices for the matrix-
    multiply DFT around a small 1.5px wide region in the original `imageCorr`.

    Following the matrix multiply DFT we use parabolic subpixel fitting to
    get even more precision! (below 1/upsampleFactor pixels)

    NOTE: previous versions of multiCorr operated in two steps: using the zero-
    padding upsample method for a first-pass factor-2 upsampling, followed by the
    DFT upsampling (at whatever user-specified factor). I have implemented it
    differently, to better support iterating over multiple peaks. **The DFT is always
    upsampled around xyShift, which MUST be specified to HALF-PIXEL precision
    (no more, no less) to replicate the behavior of the factor-2 step.**
    (It is possible to refactor this so that peak detection is done on a Fourier
    upsampled image rather than using the parabolic subpixel and rounding as now...
    I like keeping it this way because all of the parameters and logic will be identical
    to the other subpixel methods.)


    Accepts:
        imageCorr : ndarray complex
            Complex product of the FFTs of the two images to be registered
            i.e. m = np.fft.fft2(DP) * probe_kernel_FT;
            imageCorr = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
        upsampleFactor : int
            Upsampling factor. Must be greater than 2. (To do upsampling
            with factor 2, use upsampleFFT, which is faster.)
        xyShift
            Location in original image coordinates around which to upsample the
            FT. This should be given to exactly half-pixel precision to
            replicate the initial FFT step that this implementation skips

    Returns
    -------
        xyShift : 2-element np array
            Refined location of the peak in image coordinates.
    '''

    assert upsampleFactor > 2

    xyShift[0] = np.round(xyShift[0] * upsampleFactor) / upsampleFactor
    xyShift[1] = np.round(xyShift[1] * upsampleFactor) / upsampleFactor

    globalShift = np.fix(np.ceil(upsampleFactor * 1.5)/2)

    upsampleCenter = globalShift - upsampleFactor*xyShift

    imageCorrUpsample = np.conj(dftUpsample(np.conj(imageCorr), upsampleFactor, upsampleCenter ))

    xySubShift = np.unravel_index(imageCorrUpsample.argmax(), imageCorrUpsample.shape)

    # add a subpixel shift via parabolic fitting
    try:
        icc = np.real(imageCorrUpsample[xySubShift[0] - 1 : xySubShift[0] + 2, xySubShift[1] - 1 : xySubShift[1] + 2])
        dx = (icc[2,1] - icc[0,1]) / (4 * icc[1,1] - 2 * icc[2,1] - 2 * icc[0,1])
        dy = (icc[1,2] - icc[1,0]) / (4 * icc[1,1] - 2 * icc[1,2] - 2 * icc[1,0])
    except:
        dx, dy = 0, 0 # this is the case when the peak is near the edge and one of the above values does not exist

    xySubShift = xySubShift - globalShift

    xyShift = xyShift + (xySubShift + np.array([dx, dy])) / upsampleFactor

    return xyShift

def upsampleFFT(cc):
    '''
    Zero-padding FFT upsampling. Returns the real IFFT of the input with 2x
    upsampling. This may have an error for matrices with an odd size. Takes
    a complex np array as input.
    '''
    sz = cc.shape
    ups = np.zeros((sz[0]*2,sz[1]*2),dtype=complex)
    
    ups[:int(np.ceil(sz[0]/2)),:int(np.ceil(sz[1]/2))] = cc[:int(np.ceil(sz[0]/2)),:int(np.ceil(sz[1]/2))]
    ups[-int(np.ceil(sz[0]/2)):,:int(np.ceil(sz[1]/2))] = cc[-int(np.ceil(sz[0]/2)):,:int(np.ceil(sz[1]/2))]
    ups[:int(np.ceil(sz[0]/2)),-int(np.ceil(sz[1]/2)):] = cc[:int(np.ceil(sz[0]/2)),-int(np.ceil(sz[1]/2)):]
    ups[-int(np.ceil(sz[0]/2)):,-int(np.ceil(sz[1]/2)):] = cc[-int(np.ceil(sz[0]/2)):,-int(np.ceil(sz[1]/2)):]
    
    return np.real(np.fft.ifft2(ups))
  

def dftUpsample(imageCorr, upsampleFactor, xyShift):
    '''
    This performs a matrix multiply DFT around a small neighboring region of the inital correlation peak.
    By using the matrix multiply DFT to do the Fourier upsampling, the efficiency is greatly improved.
    This is adapted from the subfuction dftups found in the dftregistration function on the Matlab File Exchange.

    https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

    The matrix multiplication DFT is from 
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," 
    Opt. Lett. 33, 156-158 (2008). http://www.sciencedirect.com/science/article/pii/S0045790612000778

    Parameters
    ----------
        imageCorr : ndarray complex
            Correlation image between two images in Fourier space.
        upsampleFactor : int 
            Scalar integer of how much to upsample.
        xyShift : list of 2 floats 
            Coordinates in the UPSAMPLED GRID around which to upsample. 
            These must be single-pixel IN THE UPSAMPLED GRID
        
    Returns
    -------
        imageUpsample : ndarray
            Upsampled image from region around correlation peak.
        
    '''
    imageSize = imageCorr.shape
    pixelRadius = 1.5
    numRow = np.ceil(pixelRadius * upsampleFactor)
    numCol = numRow

    colKern = np.exp(
    (-1j * 2 * np.pi / (imageSize[1] * upsampleFactor))
    * np.outer( (np.fft.ifftshift( (np.arange(imageSize[1])) ) - np.floor(imageSize[1]/2)),  (np.arange(numCol) - xyShift[1]))
    )

    rowKern = np.exp(
    (-1j * 2 * np.pi / (imageSize[0] * upsampleFactor))
    * np.outer( (np.arange(numRow) - xyShift[0]), (np.fft.ifftshift(np.arange(imageSize[0])) - np.floor(imageSize[0]/2)))
    )

    imageUpsample = np.real(rowKern @ imageCorr @ colKern)
    return imageUpsample

