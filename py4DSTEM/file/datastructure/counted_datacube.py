from collections.abc import Sequence

from ...process.utils import tqdmnd

import numpy as np
import dask as da
import numba as nb


class Sparse4D(Sequence):
	"""
	A wrapper for a PointListArray of electron events that returns
	a reconstructed diffraction pattern when sliced.
	NOTE: This class is meant to be constructed by the
	CountedDataCube object.
	"""
	def __init__(self,electrons,detector_shape,index_key='ind'):
		super().__init__()

		self.electrons = electrons
		self.detector_shape = detector_shape
		self.index_key = index_key

		# Needed for dask:
		self.shape = (electrons.shape[0], electrons.shape[1],
			detector_shape[0], detector_shape[1])
		self.dtype = np.uint8
		self.ndim = 4

	def __getitem__(self,i):
		pl = self.electrons.get_pointlist(i[0],i[1])

		return points_to_DP_numba_ravel(pl.data[self.index_key],
			int(self.detector_shape[0]),int(self.detector_shape[1]))

	def __len__(self):
		return np.prod(self.shape)


# Numba accelerated conversion of single-index 
# electron event list to full DP
@nb.njit
def points_to_DP_numba_ravel(pl,sz1,sz2):
    dp = np.zeros((sz1*sz2),dtype=np.uint8)
    for i in nb.prange(len(pl)):
        dp[pl[i]] += 1
    return dp.reshape((sz1,sz2))