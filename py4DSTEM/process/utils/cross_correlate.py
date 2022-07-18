# Cross correlation function

import numpy as np




def get_cross_correlation(
    ar,
    template,
    corrPower=1,
    _returnval='real'):
    """
    Get the cross/phase/hybrid correlation of `ar` with `template`, where
    the latter is in real space.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """
    assert( _returnval in ('real','fourier') )
    template_FT = np.conj(np.fft.fft2(template))
    return get_cross_correlation_FT(
        ar,
        template_FT,
        corrPower = corrPower,
        _returnval = _returnval)


def get_cross_correlation_FT(
    ar,
    template_FT,
    corrPower = 1,
    _returnval = 'real'
    ):
    """
    Get the cross/phase/hybrid correlation of `ar` with `template_FT`, where
    the latter is already in Fourier space (i.e. `template_FT` is
    `np.conj(np.fft.fft2(template))`.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """
    assert(_returnval in ('real','fourier'))
    m = np.fft.fft2(ar) * template_FT
    cc = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
    if _returnval == 'real':
        cc = np.maximum(np.real(np.fft.ifft2(cc)),0)
    return cc



def get_shift(
    ar1,
    ar2,
    corrPower=1
    ):
    """
	Determine the relative shift between a pair of arrays giving the best overlap.

	Shift determination uses the brightest pixel in the cross correlation, and is
    thus limited to pixel resolution. corrPower specifies the cross correlation
    power, with 1 corresponding to a cross correlation and 0 a phase correlation.

	Args:
		ar1,ar2 (2D ndarrays):
        corrPower (float between 0 and 1, inclusive): 1=cross correlation, 0=phase
            correlation

    Returns:
		(2-tuple): (shiftx,shifty) - the relative image shift, in pixels
    """
    cc = get_cross_correlation(ar1, ar2, corrPower)
    xshift, yshift = np.unravel_index(np.argmax(cc), ar1.shape)
    return xshift, yshift



