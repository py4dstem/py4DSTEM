# Functions to become DataCube methods

import numpy as np




# Add to tree

from ..emd import Array
def add(
    self,
    data,
    name = ''
    ):
    """
    Adds a block of data to the DataCube's tree. If `data` is an instance of
    an EMD/py4DSTEM class, add it to the tree.  If it's a numpy array,
    turn it into an Array instance, then save to the tree.
    """
    if isinstance(data, np.ndarray):
        data = Array(
            data = data,
            name = name
        )
    self.tree[data.name] = data



# Diffraction imaging

from .diffractionimage import DiffractionImage
def get_diffraction_image(
    self,
    mode = 'max',
    geometry = None,
    shift_corr = False,
    name = 'diffraction_image',
    returncalc = True,
    ):
    """
    Get a diffraction image using `mode` and store it in the DataCube's
    tree with name `name`.

    Args:
        name (str): the name
        mode (str): must be in ('max','mean','median')
        geometry (variable): indicates the region the image will
            be computed over. Behavior depends on the argument type:
                - None: uses the whole image
                - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                - 2-tuple: (rx,ry), which are length L arrays.
                    Uses the specified scan positions.
                - `mask`: boolean 2D array
                - `mask_float`: floating point 2D array. Valid only for
                    `mean` mode
        shift_corr (bool): if True, correct for beam shift
        returncalc (bool): if True, returns the answer

    Returns:
        (DiffractionImage): the diffraction image
    """

    # perform computation
    from ....process.virtualdiffraction import get_diffraction_image
    dp = get_diffraction_image(
        self,
        mode = mode,
        geometry = geometry,
        shift_corr = shift_corr
    )

    # wrap with a py4dstem class
    dp = DiffractionImage(
        data = dp,
        name = name,
        mode = mode,
        geometry = geometry,
        shift_corr = shift_corr
    )

    # add to the tree
    self.tree[name] = dp

    # return
    if returncalc:
        return dp


def get_dp_max(
    self,
    geometry = None,
    shift_corr = False,
    name = 'dp_max',
    returncalc = True,
    ):
    """
    Get a maximal diffraction pattern and store it in the DataCube's
    tree with name `name`.

    Args:
        name (str): the name
        geometry (variable): indicates the region the image will
            be computed over. Behavior depends on the argument type:
                - None: uses the whole image
                - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                - 2-tuple: (rx,ry), which are length L arrays.
                    Uses the specified scan positions.
                - `mask`: boolean 2D array
        shift_corr (bool): if True, correct for beam shift
        returncalc (bool): if True, returns the answer

    Returns:
        (DiffractionImage): the diffraction image
    """
    dp = get_diffraction_image(
        self,
        name = name,
        mode = 'max',
        geometry = geometry,
        shift_corr = shift_corr,
        returncalc = True,
    )
    if returncalc:
        return dp


def get_dp_mean(
    self,
    geometry = None,
    shift_corr = False,
    name = 'dp_mean',
    returncalc = True,
    ):
    """
    Get a mean diffraction pattern and store it in the DataCube's
    tree with name `name`.

    Args:
        name (str): the name
        geometry (variable): indicates the region the image will
            be computed over. Behavior depends on the argument type:
                - None: uses the whole image
                - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                - 2-tuple: (rx,ry), which are length L arrays.
                    Uses the specified scan positions.
                - `mask`: boolean 2D array
                - `mask_float`: floating point 2D array.
        shift_corr (bool): if True, correct for beam shift
        returncalc (bool): if True, returns the answer

    Returns:
        (DiffractionImage): the diffraction image
    """
    dp = get_diffraction_image(
        self,
        name = name,
        mode = 'mean',
        geometry = geometry,
        shift_corr = shift_corr,
        returncalc = True,
    )
    if returncalc:
        return dp


def get_dp_median(
    self,
    geometry = None,
    shift_corr = False,
    name = 'dp_median',
    returncalc = True,
    ):
    """
    Get a median diffraction pattern and store it in the DataCube's
    tree with name `name`.

    Args:
        name (str): the name
        geometry (variable): indicates the region the image will
            be computed over. Behavior depends on the argument type:
                - None: uses the whole image
                - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                - 2-tuple: (rx,ry), which are length L arrays.
                    Uses the specified scan positions.
                - `mask`: boolean 2D array
        shift_corr (bool): if True, correct for beam shift
        returncalc (bool): if True, returns the answer

    Returns:
        (DiffractionImage): the diffraction image
    """
    dp = get_diffraction_image(
        self,
        name = name,
        mode = 'median',
        geometry = geometry,
        shift_corr = shift_corr,
        returncalc = True,
    )
    if returncalc:
        return dp





# Virtual imaging

from .virtualimage import VirtualImage
def get_virtual_image(
    self,
    mode,
    geometry,
    shift_corr = False,
    eager_compute = True,
    name = 'virtual_image',
    returncalc = True,
    ):
    """
    Get a virtual image and store it in `datacube`s tree under `name`.
    The kind of virtual image is specified by the `mode` argument.

    Args:
        mode (str): must be in
            ('point','circle','annulus','rectangle',
            'cpoint','ccircle','cannulus','csquare',
            'qpoint','qcircle','qannulus','qsquare',
            'mask').  The first four modes represent point, circular,
            annular, and rectangular detectors with geomtries specified
            in pixels, relative to the uncalibrated origin, i.e. the upper
            left corner of the diffraction plane. The next four modes
            represent point, circular, annular, and square detectors with
            geometries specified in pixels, relative to the calibrated origin,
            taken to be the mean posiion of the origin over all scans.
            'ccircle','cannulus', and 'csquare' are automatically centered
            about the origin. The next four modes are identical to these,
            except that the geometry is specified in q-space units, rather
            than pixels. In the last mode the geometry is specified with a
            user provided mask, which can be either boolean or floating point.
            Floating point masks are normalized by setting their maximum value
            to 1.
        geometry (variable): valid entries are determined by the `mode`
            argument, as follows:
                - 'point': 2-tuple, (qx,qy)
                - 'circle': nested 2-tuple, ((qx,qy),r)
                - 'annulus': nested 2-tuple, ((qx,qy),(ri,ro))
                - 'rectangle': 4-tuple, (xmin,xmax,ymin,ymax)
                - 'cpoint': 2-tuple, (qx,qy)
                - 'ccircle': number, r
                - 'cannulus': 2-tuple, (ri,ro)
                - 'csquare': number, s
                - 'qpoint': 2-tuple, (qx,qy)
                - 'qcircle': number, r
                - 'qannulus': 2-tuple, (ri,ro)
                - 'qsquare': number, s
                - `mask`: 2D array
        shift_corr (bool): if True, correct for beam shift. Works only with
            'c' and 'q' modes - uses the calibrated origin for each pixel,
            instead of the mean origin position.
        name (str): the output object's name
        returncalc (bool): if True, returns the output

    Returns:
        (Optional): if returncalc is True, returns the VirtualImage
    """

    # perform computation
    from ....process.virtualimage import get_virtual_image
    im = get_virtual_image(
        self,
        mode = mode,
        geometry = geometry,
        shift_corr = shift_corr,
        eager_compute = eager_compute
    )

    # wrap with a py4dstem class
    im = VirtualImage(
        data = im,
        name = name,
        mode = mode,
        geometry = geometry,
        shift_corr = shift_corr
    )

    # add to the tree
    self.tree[name] = im

    # return
    if returncalc:
        return im





# Probe

from .probe import Probe
def get_vacuum_probe(
    self,
    name = 'probe',
    returncalc = True,
    **kwargs
    ):
    """

    """

    # perform computation
    from ....process.probe import get_vacuum_probe
    x = get_vacuum_probe(
        self,
        **kwargs
    )

    # wrap with a py4dstem class
    x = Probe(
        data = x,
        **kwargs
    )

    # add to the tree
    self.tree[name] = x

    # return
    if returncalc:
        return x









