from PIL import Image
import numpy as np

def HSV_to_RGB(cin):
    """\
    HSV to RGB transformation.
    """

    # HSV channels
    h, s, v = cin

    i = (6. * h).astype(int)
    f = (6. * h) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    imout = np.zeros(h.shape + (3,), dtype=h.dtype)
    imout[:, :, 0] = 255 * (i0 * v + i1 * q + i2 * p + i3 * p + i4 * t + i5 * v)
    imout[:, :, 1] = 255 * (i0 * t + i1 * v + i2 * v + i3 * q + i4 * p + i5 * p)
    imout[:, :, 2] = 255 * (i0 * p + i1 * p + i2 * t + i3 * v + i4 * v + i5 * q)

    return imout

def P1A_to_HSV(cin, vmin=None, vmax=None):
    """\
    Transform a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.
    """
    # HSV channels
    h = .5 * np.angle(cin) / np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    assert vmin < vmax
    v = (v.clip(vmin, vmax) - vmin) / (vmax - vmin)

    return HSV_to_RGB((h, s, v))


def imsave(a, filename=None, vmin=None, vmax=None, cmap=None):
    """
    imsave(a) converts array a into, and returns a PIL image
    imsave(a, filename) returns the image and also saves it to filename
    imsave(a, ..., vmin=vmin, vmax=vmax) clips the array to values between vmin and vmax.
    imsave(a, ..., cmap=cmap) uses a matplotlib colormap.
    """

    if a.dtype.kind == 'c':
        # Image is complex
        if cmap is not None:
            print('imsave: Ignoring provided cmap - input array is complex')
        i = P1A_to_HSV(a, vmin, vmax)
        im = Image.fromarray(np.uint8(i), mode='RGB')

    else:
        if vmin is None:
            vmin = a.min()
        if vmax is None:
            vmax = a.max()
        im = Image.fromarray((255 * (a.clip(vmin, vmax) - vmin) / (vmax - vmin)).astype('uint8'))
        if cmap is not None:
            r = im.point(lambda x: cmap(x / 255.0)[0] * 255)
            g = im.point(lambda x: cmap(x / 255.0)[1] * 255)
            b = im.point(lambda x: cmap(x / 255.0)[2] * 255)
            im = Image.merge("RGB", (r, g, b))
            # b = (255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8')
            # im = Image.fromstring('L', a.shape[-1::-1], b.tostring())

    if filename is not None:
        im.save(filename)
    return im