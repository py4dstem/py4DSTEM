import numpy as np
import mrcfile
from ..utils import get_shift_hybrid
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
from time import time
from ...file.datastructure import DataCube

def read_mrc_file(filename):
	# read in an *.mrc file as an MRC memmap
	return mrcfile.mmap(filename,mode='r')


def slice_subframes(frame, x_cent, y_cent, wx=500, wy=500):
	"""
	Slice a movie frame into subframes, given the center points x_cent and y_cent (which are numpy meshgrids)
	and the subframe width wx, wy
	Returns a 3D stack of diffraction patterns.
	"""
	nDP = xcents.shape[0] * xcents.shape[1]
	stack = np.zeros((wx,wy,nDP))

	dx = np.round(wx/2).astype(int)
	dy = np.round(wy/2).astype(int)

	for Rx in range(xcents.shape[0]):
		for Ry in range(xcents.shape[1]):
			DPind = np.ravel_multi_index((Rx,Ry),(xcents.shape[0],xcents.shape[1]))
			stack[:,:,DPind] = np.roll(DP,(-(xcents[Rx,Ry]-dx),-(ycents[Rx,Ry]-dy)),axis=(0,1))[:wx,:wy]

	return stack


def subframeAlign(testDP, xcent, ycent, wx=500, wy=500, niter=10, maxshift=80, anchorDP=0, corrPower=0.5, damping=0.5 ):
	"""
	Detrmine the alignment of the subframes in a Relativity frame by gradient descent. Arguments are:

	testDP			(numpy array) a single frame from a Relativity movie, ideally with all subframes similarly illuminated
	xcent, ycent 	(numpy meshgrids) initial guesses of the centers
	wx,wy 			(ints) frame x- and y-sizes
	niter			(int) number of iterations of gradient descent to optimize shifts
	maxshift		(int) maximum allowable shift. This only triggers an error if the shifts exceed it, it doesn't constrain fitting
	anchorDP		(int) index of the DP that should be kept static. Other positions are fitted to align subframes with this one.
	corrPower		(float) Hybrid correlation power. Phase-y correlation works well when you have a beamstop
	damping			(float) damps the update steps. Set to 1 for no damping (you can probably get away with no damping)
	"""

	err = np.zeros(niter)

	nudgex = np.zeros_like(xcent)
	nudgey = np.zeros_like(ycent)

	anchorx,anchory = np.unravel_index(anchorDP,xcent.shape)

	for iter in range(niter):

		#get the stack using the current centers
		stack = slice_subframes(testDP,xcent+nudgex,ycent+nudgey,wx,wy)

		#compute the error metric
		shifterr = np.zeros(stack.shape[2])
		for ind in range(stack.shape[2]):
			shiftx, shifty = get_shift_hybrid(stack[:,:,anchorDP],stack[:,:,ind],corrPower)

			#correct for the wraparound of the shifts from get_shift
			if shiftx > int(wx/2): shiftx = shiftx - wx
			if shifty > int(wy/2): shifty = shifty - wy

			DPx,DPy = np.unravel_index(ind,xcent.shape)

			nudgex[DPx,DPy] -= np.rint(shiftx*damping)
			nudgey[DPx,DPy] -= np.rint(shifty*damping)

			shifterr[ind] = shiftx**2 + shifty**2

		newerr = np.sqrt(np.mean(shifterr.astype(float)))
		err[iter] = newerr

		assert np.max(np.abs((nudgex,nudgey))) < maxshift, "Maximum shift exceeded."

	bestx = xcent + nudgex
	besty = ycent + nudgey

	stack0 = slice_subframes(testDP,xcent,ycent,wx,wy)

	fig, ax = subplots(xcent.shape[0],xcent.shape[1],figsize=(10,10)):

	for Rx in np.arange(xcent.shape[0]):
		for Ry in np.arange(xcent.shape[1]):
			ind = np.ravel_multi_index((Rx,Ry),(xcent.shape[0],xcent.shape[1]))
			ax[Rx,Ry].imshow(stack[:,:,ind]-stack0[:,:,ind], cmap='RdBu')
			ax[Rx,Ry].axis('off')

	plt.show()

	fig2, ax2 = subplots(xcent.shape[0],xcent.shape[1],figsize=(10,10)):

	for Rx in np.arange(xcent.shape[0]):
		for Ry in np.arange(xcent.shape[1]):
			ind = np.ravel_multi_index((Rx,Ry),(xcent.shape[0],xcent.shape[1]))
			ax2[Rx,Ry].imshow(stack[:,:,ind])
			ax2[Rx,Ry].axis('off')

	plt.show()

	return bestx, besty, err


def slice_mrc_stack(mrc, scratch, scanshape, optx, opty, wx=500, wy=500):
	"""
	Slice the *.mrc movie into all of its subframes

	mrc 		(MrcMemmap) memory map into the mrc file (opened by py4DSTEM.file.readwrite.read(...,load='relativity'))
	scratch 	(str) path to a scratch file where a numpy memmap containing the re-sliced stack will be buffered
				      NOTE! this will overwrite whatever file is at this path! be careful!
  	scanshape	(numpy array) 2-element array containing the scan shape (Rx, Ry)
	optx, opty 	(numpy meshgrids) the optimized centers of the subframes from subframeAlign(...)
	wx,wy 		(ints) subframe sizes x and y

	Returns:
	dc 			(DataCube) a py4DSTEM DataCube containing the sliced up stack, in the correct order
	"""

	nframe = scansize.prod()//optx.size

	dshape = (int(nframe),int(optx.size),wx,wy)

	vstack = np.memmap(scratch, mode='w+', dtype='<i2', shape=dshape)

	f = FloatProgress(min=0,max=nframe-1)
	display(f)

	t0 = time()

	for i in np.arange(startframe,startframe+nframe):
		f.value = i
		frame = mrc.data[int(i),:,:]
		stack = slice_subframes(frame,optx,opty,wx,wy)
		vstack[int(i-startframe),:,:,:] = np.transpose(stack,(2,0,1))

	t = time() - t0

	print("Sliced {} diffraction patterns in {}h {}m {}s".format(scansize.prod(), int(t/3600),int(t/60), int(t%60)))
	mrc.close()

	dc = DataCube(vstack)
	dc.set_scan_shape(scanshape[0],scanshape[1])


