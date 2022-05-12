# -*- coding: utf-8 -*-

# Reads an IDES Relativity file, determines the locations of the subframes, and splits them into a
# stack.
#
# The IDES Relativity splits the camera area into a matrix of subframes, which gives a higher
# framerate than the native capability of the camera (1280 fps in 4x4 mode on the NCEM ThemIS).
# The frames are slightly overlapping and highly distorted. This file provides functions for reading
# the *.mrc movies containing the frames, locating the subframes and cropping them, measuring the
# distortion in each subframe, and finally extracting a corrected DataCube from the movie.
# 
# Created on 7 May 2019
# @author: sezeltmann

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
from time import time
from ..utils import get_shift
from ...io import DataCube

def slice_subframes(frame, x_cent, y_cent, wx=500, wy=500):
	"""
	Slice a movie frame into subframes.

    x_cent and y_cent are shape (N,M) arrays, where the camera frames have been divided
    into an NxM grid of subframes.

    Args:
        frame (array): the original camera frame containing the tiled subframes
        x_cent (array): x coords of the centers of the frames
        y_cent (array): y coords of the centers of the frames
        wx (int): the x width of the subframes
        wy (int): the y width of the subframes

	Returns:
        (ndarray) a 3D stack of diffraction patterns
	"""
	nDP = x_cent.shape[0] * x_cent.shape[1]
	stack = np.zeros((wx,wy,nDP))

	dx = np.round(wx/2).astype(int)
	dy = np.round(wy/2).astype(int)

	for Rx in range(x_cent.shape[0]):
		for Ry in range(x_cent.shape[1]):
			DPind = np.ravel_multi_index((Rx,Ry),(x_cent.shape[0],x_cent.shape[1]))
			stack[:,:,DPind] = np.roll(frame,(-(x_cent[Rx,Ry]-dx),-(y_cent[Rx,Ry]-dy)),axis=(0,1))[:wx,:wy]

	return stack


def subframe_align(testDP, xcent, ycent, wx=500, wy=500, niter=10, maxshift=80, anchorDP=0, corrPower=0.5, damping=1 ):
	"""
	Determine the alignment of the subframes in a Relativity frame by gradient descent.

    Args:
        testDP (array): a single frame from a Relativity movie, ideally with all
            subframes similarly illuminated
        xcent (array): initial guess at the x position of the frame of the centers
        ycent (array): initial guess at the y position of the frame of the centers
        wx,wy (int): subframe x- and y-sizes
        niter (int): number of iterations of gradient descent to optimize shifts
        maxshift (int): maximum allowable shift. This only triggers an error if the
            shifts exceed it; it doesn't constrain fitting
        anchorDP (int): index of the DP that should be kept static. Other positions are
            fitted to align subframes with this one.
        corrPower (float): hybrid correlation power. Phase-y correlation works well when
            you have a beamstop
        damping	(float): damps the update steps. Set to 1 for no damping (you can
            probably get away with no damping)

    Returns
        (3-tuple) A 3-tuple containing:

            * **bestx**: *(2D array)* optimized subframe centers
            * **besty**: *(2D array)* optimized subframe centers
            * **err**: *(1D array)* fit error for each gradient descent step
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
			shiftx, shifty = get_shift(stack[:,:,anchorDP],stack[:,:,ind],corrPower)

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

	fig, ax = plt.subplots(xcent.shape[0],xcent.shape[1],figsize=(10,10))

	for Rx in np.arange(xcent.shape[0]):
		for Ry in np.arange(xcent.shape[1]):
			ind = np.ravel_multi_index((Rx,Ry),(xcent.shape[0],xcent.shape[1]))
			ax[Rx,Ry].imshow(stack[:,:,ind]-stack0[:,:,ind], cmap='RdBu')
			ax[Rx,Ry].axis('off')

	plt.show()

	fig2, ax2 = plt.subplots(xcent.shape[0],xcent.shape[1],figsize=(10,10))

	for Rx in np.arange(xcent.shape[0]):
		for Ry in np.arange(xcent.shape[1]):
			ind = np.ravel_multi_index((Rx,Ry),(xcent.shape[0],xcent.shape[1]))
			ax2[Rx,Ry].imshow(stack[:,:,ind])
			ax2[Rx,Ry].axis('off')

	plt.show()

	return bestx, besty, err


def slice_mrc_stack(mrc, scratch, scanshape, optx, opty, startframe=0, wx=500, wy=500):
	"""
	Slice the *.mrc movie into all of its subframes

    Args:
        mrc (MrcMemmap): memory map into the mrc file (such as opened by
            py4DSTEM.file.io.read(...,load='relativity'))
        scratch (str): path to a scratch file where a numpy memmap containing the
            re-sliced stack will be buffered.
            NOTE! this will overwrite whatever file is at this path! be careful!
            ALSO NOTE! this file is where the data in the DataCube will actually live!
            Either save the DataCube as a py4DSTEM *.h5 or use separate scratches for
            different data!
        scanshape (numpy array): 2-element array containing the scan shape (Rx, Ry)
        optx, opty (numpy meshgrids): the optimized centers of the subframes from
            subframeAlign(...)
        wx,wy (ints): subframe sizes x and y

	Returns:
        (DataCube): a py4DSTEM DataCube containing the sliced up stack, in the correct
        order
	"""

	nframe = scanshape.prod()//optx.size

	dshape = (int(nframe),int(optx.size),wx,wy)

	vstack = np.memmap(scratch, mode='w+', dtype='<i2', shape=dshape)

	t0 = time()

	for i in np.arange(startframe,startframe+nframe):
		frame = mrc.data[int(i),:,:]
		stack = slice_subframes(frame,optx,opty,wx,wy)
		vstack[int(i-startframe),:,:,:] = np.transpose(stack,(2,0,1))

	t = time() - t0

	print("Sliced {} diffraction patterns in {}h {}m {}s".format(scanshape.prod(), int(t/3600),int(t/60), int(t%60)))
	mrc.close()

	dc = DataCube(vstack)
	dc.set_scan_shape(scanshape[0],scanshape[1])

	return dc


