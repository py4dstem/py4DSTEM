import pyqtgraph as pg

from PyQt5 import QtCore, QtGui

import numpy as np
import collections
import pyqtgraph.functions as fn
import pyqtgraph.debug as debug
from pyqtgraph import GraphicsObject
from pyqtgraph import Point
from pyqtgraph import getConfigOption

class ImageViewAlpha(pg.ImageItem):

	def __init__(self,image=None,alpha=None, **kargs):
		pg.ImageItem.__init__(self, image, **kargs)

		self.alpha = alpha

	def updateImage(self, *args, **kargs):
		## used for re-rendering qimage from self.image.
		
		## can we make any assumptions here that speed things up?
		## dtype, range, size are all the same?
		defaults = {
			'autoLevels': False,
		}
		defaults.update(kargs)
		return self.setImage(*args, **defaults)

	def render(self):
		# Convert data to QImage for display.
		
		profile = debug.Profiler()
		if self.image is None or self.image.size == 0:
			return
		
		# Request a lookup table if this image has only one channel
		if self.image.ndim == 2 or self.image.shape[2] == 1:
			if isinstance(self.lut, collections.Callable):
				lut = self.lut(self.image)
			else:
				lut = self.lut
		else:
			lut = None

		if self.autoDownsample:
			# reduce dimensions of image based on screen resolution
			o = self.mapToDevice(QtCore.QPointF(0,0))
			x = self.mapToDevice(QtCore.QPointF(1,0))
			y = self.mapToDevice(QtCore.QPointF(0,1))
			w = Point(x-o).length()
			h = Point(y-o).length()
			if w == 0 or h == 0:
				self.qimage = None
				return
			xds = max(1, int(1.0 / w))
			yds = max(1, int(1.0 / h))
			axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
			image = fn.downsample(self.image, xds, axis=axes[0])
			image = fn.downsample(image, yds, axis=axes[1])
			self._lastDownsample = (xds, yds)
		else:
			image = self.image

		# if the image data is a small int, then we can combine levels + lut
		# into a single lut for better performance
		levels = self.levels
		if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
			if self._effectiveLut is None:
				eflsize = 2**(image.itemsize*8)
				ind = np.arange(eflsize)
				minlev, maxlev = levels
				levdiff = maxlev - minlev
				levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
				if lut is None:
					efflut = fn.rescaleData(ind, scale=255./levdiff, 
											offset=minlev, dtype=np.ubyte)
				else:
					lutdtype = np.min_scalar_type(lut.shape[0]-1)
					efflut = fn.rescaleData(ind, scale=(lut.shape[0]-1)/levdiff,
											offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0]-1))
					efflut = lut[efflut]
				
				self._effectiveLut = efflut
			lut = self._effectiveLut
			levels = None
		
		# Convert single-channel image to 2D array
		if image.ndim == 3 and image.shape[-1] == 1:
			image = image[..., 0]
		
		# Assume images are in column-major order for backward compatibility
		# (most images are in row-major order)
		if self.axisOrder == 'col-major':
			image = image.transpose((1, 0, 2)[:image.ndim])
		
		argb, alpha = fn.makeARGB(image, lut=lut, levels=levels)

		if self.alpha is not None:
			argb[:,:,3] = self.alpha.T

		self.qimage = fn.makeQImage(argb, True, transpose=False)



