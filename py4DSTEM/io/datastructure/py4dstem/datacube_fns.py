# Functions to become DataCube methods

import numpy as np
from .diffractionimage import DiffractionImage







# Diffraction imaging


def get_diffraction_image(
    self,
    name = 'diffraction_image',
    mode = 'max',
    geometry = None,
    shift_corr = False,
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
    from ....process.virtualimage import get_diffraction_image
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
    name = 'dp_max',
    geometry = None,
    shift_corr = False,
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
    name = 'dp_mean',
    geometry = None,
    shift_corr = False,
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
    name = 'dp_median',
    geometry = None,
    shift_corr = False,
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




