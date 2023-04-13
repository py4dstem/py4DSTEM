"""
This sub-module contains functions for polar transform peak detection of amorphous / semicrystalline datasets.

"""

from py4DSTEM import tqdmnd
from itertools import product
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt



class PolarPeaks:
    """
    Primary class for polar transform peak detection.
    """


    def __init__(
        self,
        datacube,
        radial_min = 0.0,
        radial_max = None,
        radial_step = 1.0,
        num_annular_bins = 60,
        progress_bar = True,
        ):
        """
        Initialize class by performing an intensity-preserving polar transformation.

        Parameters
        --------
        datacube: py4DSTEM.io.DataCube
            4D-STEM dataset, requires origin calibration
        radial_min: float
            Minimum radius of polar transformation.
        radial_max: float
            Maximum radius of polar transformation.
        radial_step: float
            Width of radial bins of polar transformation.
        num_annular_bins: int
            Number of bins in annular direction. Note that we fold data over 180 degrees periodically, 
            so setting this value to 60 gives bin widths of 180/60 = 3.0 degrees.
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        

        """

        # radial bin coordinates
        if radial_max is None:
            radial_max = np.min(datacube.Qshape) / np.sqrt(2)
        self.radial_bins = np.arange(
            radial_min,
            radial_max,
            radial_step,
            )
        self.radial_step = np.array(radial_step)

        # annular bin coordinates
        self.annular_bins = np.linspace(
            0,
            np.pi,
            num_annular_bins,
            endpoint = False,
            )
        self.annular_step = self.annular_bins[1] - self.annular_bins[0]

        # init polar transformation array
        self.polar_shape = np.array((self.annular_bins.shape[0], self.radial_bins.shape[0]))
        self.polar_size = np.prod(self.polar_shape)
        self.data_polar = np.zeros((
            datacube.R_Nx,
            datacube.R_Ny,
            self.polar_shape[0],
            self.polar_shape[1],
            ))

        # init coordinates
        xa, ya = np.meshgrid(
            np.arange(datacube.Q_Nx),
            np.arange(datacube.Q_Ny),
            indexing = 'ij',
        )

        # polar transformation
        for rx, ry in tqdmnd(
            range(20,21),
            range(110,111),
            # range(datacube.R_Nx),
            # range(datacube.R_Ny),
            desc="polar transformation",
            unit=" images",
            disable=not progress_bar,
            ):

            # shifted coordinates
            x = xa - datacube.calibration.get_qx0(rx,ry)
            y = ya - datacube.calibration.get_qy0(rx,ry)

            # polar coordinate indices
            r_ind = (np.sqrt(x**2 + y**2) - self.radial_bins[0]) / self.radial_step
            t_ind = np.arctan2(y, x) / self.annular_step
            r_ind_floor = np.floor(r_ind).astype('int')
            t_ind_floor = np.floor(t_ind).astype('int')
            dr = r_ind - r_ind_floor
            dt = t_ind - t_ind_floor
            # t_ind_floor = np.mod(t_ind_floor, self.num_annular_bins)

            # polar transformation
            sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self.polar_shape[1])
            im = np.bincount(
                r_ind_floor[sub] + \
                np.mod(t_ind_floor[sub],self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (1 - dr[sub]) * (1 - dt[sub]),
                minlength = self.polar_size,
            )
            im += np.bincount(
                r_ind_floor[sub] + \
                np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (1 - dr[sub]) * (    dt[sub]),
                minlength = self.polar_size,
            )
            sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self.polar_shape[1]-1)
            im += np.bincount(
                r_ind_floor[sub] + 1 + \
                np.mod(t_ind_floor[sub],self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (    dr[sub]) * (1 - dt[sub]),
                minlength = self.polar_size,
            )
            im += np.bincount(
                r_ind_floor[sub] + 1 + \
                np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (    dr[sub]) * (    dt[sub]),
                minlength = self.polar_size,
            )

            # sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self.polar_shape[1])
            # im = accumarray(
            #     np.vstack((
            #         np.mod(t_ind_floor[sub],self.polar_shape[0]),
            #         r_ind_floor[sub],
            #     )).T, 
            #     datacube.data[rx,ry][sub] * (1 - dr[sub]) * (1 - dt[sub]),
            #     size=self.polar_shape,
            # )
            # im += accumarray(
            #     np.vstack((
            #         np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]),
            #         r_ind_floor[sub],
            #     )).T, 
            #     datacube.data[rx,ry][sub] * (1 - dr[sub]) * (    dt[sub]),
            #     size=self.polar_shape,
            # )
            # sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self.polar_shape[1]-1)
            # im += accumarray(
            #     np.vstack((
            #         np.mod(t_ind_floor[sub],self.polar_shape[0]),
            #         r_ind_floor[sub] + 1,
            #     )).T, 
            #     datacube.data[rx,ry][sub] * (    dr[sub]) * (1 - dt[sub]),
            #     size=self.polar_shape,
            # )            
            # im += accumarray(
            #     np.vstack((
            #         np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]),
            #         r_ind_floor[sub] + 1,
            #     )).T, 
            #     datacube.data[rx,ry][sub] * (    dr[sub]) * (    dt[sub]),
            #     size=self.polar_shape,
            # )

            # Output
            # self.data_polar[rx,ry] = im



            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(
                np.reshape(im, self.polar_shape),
                vmin = 0,
                vmax = 10,
                )





    # fig, ax = plt.subplots(figsize=(6,6))
    # ax.imshow(
    #     im,
    #     vmin = 0,
    #     vmax = 10,
    #     )



# def accumarray(subs, vals, size=None, fun=np.sum):

#     if len(subs.shape) == 1:
#         if size is None:
#             size = [subs.values.max() + 1, 0]

#         acc = val.groupby(subs).agg(fun)
#     else:
#         if size is None:
#             size = [subs.values.max()+1, subs.shape[1]]

#         # subs = subs.copy().reset_index()
#         subs = subs.copy()
#         by = subs.columns.tolist()[1:]
#         acc = subs.groupby(by=by)['index'].agg(list).apply(lambda x: val[x].agg(fun))
#         acc = acc.to_frame().reset_index().pivot_table(index=0, columns=1, aggfunc='first')
#         acc.columns = range(acc.shape[1])
#         acc = acc.reindex(range(size[1]), axis=1).fillna(0)

#     id_x = range(size[0])
#     acc = acc.reindex(id_x).fillna(0)

#     return acc


# def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
#     """
#     An accumulation function similar to Matlab's `accumarray` function.

#     Parameters
#     ----------
#     accmap : ndarray
#         This is the "accumulation map".  It maps input (i.e. indices into
#         `a`) to their destination in the output array.  The first `a.ndim`
#         dimensions of `accmap` must be the same as `a.shape`.  That is,
#         `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
#         has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
#         case `accmap[i,j]` gives the index into the output array where
#         element (i,j) of `a` is to be accumulated.  If the output is, say,
#         a 2D, then `accmap` must have shape (15,4,2).  The value in the
#         last dimension give indices into the output array. If the output is
#         1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
#     a : ndarray
#         The input data to be accumulated.
#     func : callable or None
#         The accumulation function.  The function will be passed a list
#         of values from `a` to be accumulated.
#         If None, numpy.sum is assumed.
#     size : ndarray or None
#         The size of the output array.  If None, the size will be determined
#         from `accmap`.
#     fill_value : scalar
#         The default value for elements of the output array. 
#     dtype : numpy data type, or None
#         The data type of the output array.  If None, the data type of
#         `a` is used.

#     Returns
#     -------
#     out : ndarray
#         The accumulated results.

#         The shape of `out` is `size` if `size` is given.  Otherwise the
#         shape is determined by the (lexicographically) largest indices of
#         the output found in `accmap`.


#     Examples
#     --------
#     >>> from numpy import array, prod
#     >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
#     >>> a
#     array([[ 1,  2,  3],
#            [ 4, -1,  6],
#            [-1,  8,  9]])
#     >>> # Sum the diagonals.
#     >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
#     >>> s = accum(accmap, a)
#     array([9, 7, 15])
#     >>> # A 2D output, from sub-arrays with shapes and positions like this:
#     >>> # [ (2,2) (2,1)]
#     >>> # [ (1,2) (1,1)]
#     >>> accmap = array([
#             [[0,0],[0,0],[0,1]],
#             [[0,0],[0,0],[0,1]],
#             [[1,0],[1,0],[1,1]],
#         ])
#     >>> # Accumulate using a product.
#     >>> accum(accmap, a, func=prod, dtype=float)
#     array([[ -8.,  18.],
#            [ -8.,   9.]])
#     >>> # Same accmap, but create an array of lists of values.
#     >>> accum(accmap, a, func=lambda x: x, dtype='O')
#     array([[[1, 2, 4, -1], [3, 6]],
#            [[-1, 8], [9]]], dtype=object)
#     """

#     # Check for bad arguments and handle the defaults.
#     if accmap.shape[:a.ndim] != a.shape:
#         raise ValueError("The initial dimensions of accmap must be the same as a.shape")
#     if func is None:
#         func = np.sum
#     if dtype is None:
#         dtype = a.dtype
#     if accmap.shape == a.shape:
#         accmap = np.expand_dims(accmap, -1)
#     adims = tuple(range(a.ndim))
#     if size is None:
#         size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
#     size = np.atleast_1d(size)

#     # Create an array of python lists of values.
#     vals = np.empty(size, dtype='O')
#     for s in product(*[range(k) for k in size]):
#         vals[s] = []
#     for s in product(*[range(k) for k in a.shape]):
#         indx = tuple(accmap[s])
#         val = a[s]
#         vals[indx].append(val)

#     # Create the output array.
#     out = np.empty(size, dtype=dtype)
#     for s in product(*[range(k) for k in size]):
#         if vals[s] == []:
#             out[s] = fill_value
#         else:
#             out[s] = func(vals[s])

#     return out
