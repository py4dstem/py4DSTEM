'''
# multicorr.py copied from https://github.com/ercius/openNCEM/blob/master/ncempy/algo/multicorr.py
# modified by SEZ, May 2019 to integrate with py4DSTEM utility functions
'''

import numpy as np
#from pdb import set_trace

def upsampled_correlation(imageCorr, upsampleFactor, xyShift):
    #set_trace()
    '''Upsamples the correlation image by a set integer factor upsampleFactor.
    If upsampleFactor == 2, then it is naively Fourier upsampled.
    If the upsampleFactoris higher than 2, then it uses dftUpsample, which is
    a more efficient way to Fourier upsample the image.

    Parameters
    ----------
        imageCorr : ndarray complex 
            Fourier transformed correlation image returned by initial_correlation_image.
        upsampleFactor : int
            Upsampling factor.

    Returns
    -------
        xyShift : list
            Shift in x and y of G2 with respect to G1.
    '''

    imageCorrIFT = np.real(np.fft.ifft2(imageCorr))

    if upsampleFactor == 1:
        imageSize = imageCorrIFT.shape
        xyShift[0] = ((xyShift[0] + imageSize[0]/2) % imageSize[0]) - imageSize[0]/2
        xyShift[1] = ((xyShift[1] + imageSize[1]/2) % imageSize[1]) - imageSize[1]/2
        
    else:
        #imageCorrLarge = upsampleFFT(imageCorr, 2)
        #imageSizeLarge = imageCorrLarge.shape
        #xySubShift2 = list(np.unravel_index(imageCorrLarge.argmax(), imageSizeLarge, 'C'))
        #print('xySubShift2 = {}'.format(xySubShift2))
        #xySubShift2[0] = ((xySubShift2[0] + imageSizeLarge[0]/2) % imageSizeLarge[0]) - imageSizeLarge[0]/2
        #xySubShift2[1] = ((xySubShift2[1] + imageSizeLarge[1]/2) % imageSizeLarge[1]) - imageSizeLarge[1]/2
        #xyShift = [i/2 for i in xySubShift2] #signs have to flip, or mod wrong?
        #print('xyShiftln127 = {}.format(xyShift))

        if upsampleFactor > 2:
            # here is where we use DFT registration to make things much faster
            # we cut out and upsample a peak 1.5 by 1.5 px from our original correlation image.

            xyShift[0] = np.round(xyShift[0] * upsampleFactor) / upsampleFactor
            xyShift[1] = np.round(xyShift[1] * upsampleFactor) / upsampleFactor

            globalShift = np.fix(np.ceil(upsampleFactor * 1.5)/2)# this line might have an off by one error based. The associated matlab comment is "this will be used to center the output array at dftshift + 1"
            #print('globalShift', globalShift, 'upsampleFactor', upsampleFactor, 'xyShift', xyShift)

            imageCorrUpsample = np.conj(dftUpsample(np.conj(imageCorr), upsampleFactor, xyShift )) #/ (np.fix(imageSizeLarge[0]) * np.fix(imageSizeLarge[1]) * upsampleFactor ** 2)

            xySubShift = np.unravel_index(imageCorrUpsample.argmax(), imageCorrUpsample.shape, 'C')
            #print('xySubShift = {}'.format(xySubShift))

            # add a subpixel shift via parabolic fitting
            try:
                icc = np.real(imageCorrUpsample[xySubShift[0] - 1 : xySubShift[0] + 2, xySubShift[1] - 1 : xySubShift[1] + 2])
                dx = (icc[2,1] - icc[0,1]) / (4 * icc[1,1] - 2 * icc[2,1] - 2 * icc[0,1])
                dy = (icc[1,2] - icc[1,0]) / (4 * icc[1,1] - 2 * icc[1,2] - 2 * icc[1,0])
            except:
                dx, dy = 0, 0 # this is the case when the peak is near the edge and one of the above values does not exist
            #print('dxdy = {}, {}'.format(dx, dy))
            #print('xyShift = {}'.format(xyShift))
            xySubShift = xySubShift - globalShift
            #print('xysubShift2 = {}'.format(xySubShift))
            xyShift = xyShift + (xySubShift + np.array([dx, dy])) / upsampleFactor
            #print('xyShift2 = {}'.format(xyShift))

    return xyShift

def upsampleFFT(imageInit, upsampleFactor):
    '''This does a Fourier upsample of the imageInit. imageInit is the Fourier transform of the correlation image. 
    The function returns the real space correlation image that has been Fourier upsampled by the upsampleFactor.
    An upsample factor of 2 is generally sufficient.

    The way it works is that it embeds imageInit in a larger array of zeros, then does the inverse Fourier transform to return the Fourier upsampled image in real space.

    Parameters
    ----------
        imageInit : ndarray  complex
            The image to be Fourier upsampled. This should be in the Fourier domain.
        upsampleFactor : int
            THe upsample factor (usually 2).

    Returns
    -------
        imageUpsampleReal : ndarray complex 
            The inverse Fourier transform of imageInit upsampled by the upsampleFactor.
            
    TODO
    ----
        This can be fully replaced by np.fft.ffts(im,s=(1024,1024)) where im.shape = (512,512)
        Also change to use rfft2
    '''
    imageSize = imageInit.shape
    imageUpsample = np.zeros(tuple((i*upsampleFactor for i in imageSize))) + 0j
    imageUpsample[:imageSize[0], :imageSize[1]] = imageInit
    imageUpsample = np.roll(np.roll(imageUpsample, -int(imageSize[0]/2), 0), -int(imageSize[1]/2),1)
    imageUpsampleReal = np.real(np.fft.ifft2(imageUpsample))
    return imageUpsampleReal

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
        imageCorr : ndarray
            Correlation image between two images in Fourier space.
        upsampleFactor : int 
            Scalar integer of how much to upsample.
        xyShift : list of 2 floats 
            Single pixel shift between images previously computed. Used to center the matrix multiplication 
            on the correlation peak.
        
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
    * (np.fft.ifftshift( (np.arange(imageSize[1])) )
    - np.floor(imageSize[1]/2))
    * (np.arange(numCol) - xyShift[1])[:, np.newaxis]
    ) # I think this can be written differently without the need for np.newaxis. This might require np.outer to compute the matrix itself instead of just using np.dot.

    rowKern = np.exp(
    (-1j * 2 * np.pi / (imageSize[0] * upsampleFactor))
    * (np.arange(numRow) - xyShift[0])
    * (np.fft.ifftshift(np.arange(imageSize[0]))
    - np.floor(imageSize[0]/2))[:, np.newaxis]
    ) # Comment from above applies.

    imageUpsample = np.real(np.dot(np.dot(rowKern.transpose(), imageCorr), colKern.transpose()))

    return imageUpsample

