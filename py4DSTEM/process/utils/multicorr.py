# multicorr.py copied from https://github.com/ercius/openNCEM/blob/master/ncempy/algo/multicorr.py
# modified by SEZ, May 2019, to support specified corrPower instead of set options

'''
Module to correlate two images, functionally written.

TODO
----
    - Replace makeFourierCoords with np.fft.fftfreq
    - Add verbose output
    - Add testing for complex input for G1 and G2
    - upsampled_correlation has ```if upSample > 2``` but
      should it be if upSample >= 2?
    - Cant use rfft2 currently. This gives one shift as 1/2 the value. How
      can this be improved to improve speed?
    - Better to make this a class. Then you can set it
      up and change the inputs without having to run
      through all of the setup again
    - I think upSampleFFT can be replaced by np.fft.fft2
      with the s option.
    - imageShifter and multicorr output have opposite sign.

'''

import numpy as np

def multicorr(ar, fk, method = 'cross', upsampleFactor = 1):
    '''Align a template to an image by cross correlation. THe template
    and the image must have the same size.
    
    The function takes in FFTs so that any FFT algorithm can be used to 
    transform the image and template (fft2, mkl, scipack, etc.)
    
    Parameters   
    ----------
        G1 : complex ndarray
            Fourier transform of reference image.
        G2 : complex ndarray
            Fourier kernel = F(probe)*
        method : str, optional
            The correlation method to use. Must be 'phase' or 'cross' or 'hybrid' (default = 'cross')
        upsampleFactor : int
            Upsample factor for subpixel precision of cross correlation. (default = 1)

    Returns
    -------
        xyShift : list of floats
            The shift between G1 and G2 in pixels.
    
    Example
    -------
        Cross correlate two images already stored as ndarrays. You must input the FFT
        of the images.
        
        >>> import ncempy.multicorr as mc
        >>> import numpy as np
        >>> im0FFT = np.fft.fft2(im0)
        >>> im1FFT = np.fft.fft2(im1)
        >>> shifts = mc.multicorr(im0FFT,im1FFT)
    
    '''
    
    #method, upsampleFactor = parse_input(G1, G2, method, upsampleFactor)
    # Check to make sure both G1 and G2 are arrays
    if type(G1) is not np.ndarray:
        raise TypeError('G1 must be an ndarray')
    elif type(G2) is not np.ndarray:
        raise TypeError('G2 must be an ndarray')
    
    #Check that the inputs are complex FFTs (common error)
    if not np.iscomplexobj(G1) or not np.iscomplexobj(G2):
        raise TypeError('G1 and G2 must be complex FFTs.')
    
    # Check to make sure method and upsample factor are the correct values
    if method not in ['phase', 'cross', 'hybrid']:
        print('Unknown method used, setting to cross.')
        method = 'cross'

    if type(upsampleFactor) is not int and type(upsampleFactor) is not float:
        print('Upsample factor is not an integer or float, setting to 1')
        upsampleFactor = 1
    elif type(upsampleFactor) is not int:
        print('Upsample factor is not an integer, rounding down')
        upsampleFactor = int(upsampleFactor)
        if upsampleFactor < 1:
            print('Upsample factor is < 1, setting to 1')
            upsampleFactor = 1

    # Verify images are the same size.
    if G1.shape != G2.shape:
        raise TypeError('G1 and G2 are not the same size, G1 is {0} and G2 is {1}'.format(G1.shape, G2.shape))

    
    imageCorr = initial_correlation_image(G1, G2, corrPower)
    xyShift = upsampled_correlation(imageCorr, upsampleFactor)
    
    return xyShift

def initial_correlation_image(G1, G2, corrPower):
    '''Generate correlation image at initial resolution using the method specified.

    Parameters   
    ----------
        G1 : complex ndarray
            Fourier transform of reference image.
        G2 : complex ndarray
            Fourier kernel F(probe)*
        corrPower: Fourier correlation power

    Returns
    -------
        imageCorr : ndarray complex
            Correlation array which has not yet been inverse Fourier transformed.
    '''
    m = G1 * G2

    return np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))

def upsampled_correlation(imageCorr, upsampleFactor):
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
    xyShift = list(np.unravel_index(imageCorrIFT.argmax(), imageCorrIFT.shape, 'C'))

    if upsampleFactor == 1:
        imageSize = imageCorrIFT.shape
        xyShift[0] = ((xyShift[0] + imageSize[0]/2) % imageSize[0]) - imageSize[0]/2
        xyShift[1] = ((xyShift[1] + imageSize[1]/2) % imageSize[1]) - imageSize[1]/2
        
    else:
        imageCorrLarge = upsampleFFT(imageCorr, 2)
        imageSizeLarge = imageCorrLarge.shape
        xySubShift2 = list(np.unravel_index(imageCorrLarge.argmax(), imageSizeLarge, 'C'))
        #print('xySubShift2 = {}'.format(xySubShift2))
        xySubShift2[0] = ((xySubShift2[0] + imageSizeLarge[0]/2) % imageSizeLarge[0]) - imageSizeLarge[0]/2
        xySubShift2[1] = ((xySubShift2[1] + imageSizeLarge[1]/2) % imageSizeLarge[1]) - imageSizeLarge[1]/2
        xyShift = [i/2 for i in xySubShift2] #signs have to flip, or mod wrong?
        #print('xyShiftln127 = {}.format(xyShift))

        if upsampleFactor > 2:
            # here is where we use DFT registration to make things much faster
            # we cut out and upsample a peak 1.5 by 1.5 px from our original correlation image.

            xyShift[0] = np.round(xyShift[0] * upsampleFactor) / upsampleFactor
            xyShift[1] = np.round(xyShift[1] * upsampleFactor) / upsampleFactor

            globalShift = np.fix(np.ceil(upsampleFactor * 1.5)/2)# this line might have an off by one error based. The associated matlab comment is "this will be used to center the output array at dftshift + 1"
            #print('globalShift', globalShift, 'upsampleFactor', upsampleFactor, 'xyShift', xyShift)

            imageCorrUpsample = np.conj(dftUpsample(np.conj(imageCorr), upsampleFactor, globalShift - np.multiply(xyShift, upsampleFactor))) / (np.fix(imageSizeLarge[0]) * np.fix(imageSizeLarge[1]) * upsampleFactor ** 2)

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
            xySubShift = xySubShift - globalShift;
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

def imageShifter(G2, xyShift):
    '''
    This function multiplies G2 by a plane wave that has the real space effect of shifting ifft2(G2) by [x, y] pixels.

    Parameters
   -----------
        G2 : complex ndarray
            The Fourier transform of an image.
        xyShift : list
            A two element list of the shifts along each axis.

    Returns
    -------
        G2shift : complex ndarray
            Fourier shifted image FFT
            
    Example
    -------
    
        >>> shiftIm0 = np.real(np.fft.ifft2(multicorr.imageShifter(np.fft.fft2(im0),[11.1,22.2])))
        >>> plt.imshow(shiftIm0)
        
    '''
    imageSize = G2.shape
    qx = makeFourierCoords(imageSize[0], 1) # does this need to be a column vector
    if imageSize[1] == imageSize[0]:
        qy = qx
    else:
        qy = makeFourierCoords(imageSize[1], 1)

    G2shift = np.multiply(G2, np.outer( np.exp(-2j * np.pi * qx * xyShift[0]),  np.exp(-2j * np.pi * qy * xyShift[1])))

    return G2shift

def makeFourierCoords(N, pSize):
    '''
    This function creates Fourier coordinates such that (0,0) is in the center of the array.

    Parameters
    ----------
        N :int
            The maximum coordinate in the original frame.
        pSize : float
            The pixel size.

    Returns
    -------
        q : ndarray
            A single row array that has transformed 0:N to -N/2:N/2, such that the array sizes are the same.
    '''

    N = float(N)
    if N % 2 == 0:
        q = np.roll(np.arange(-N/2, N/2, dtype = 'float64') / (N * pSize), int(-N/2), axis=0)
    else:
        q = np.roll(np.arange((1-N)/2, (N+1)/2) / (N * pSize), int((1-N)/2), axis=0)
    return q
