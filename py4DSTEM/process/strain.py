# Defines the Strain class

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from py4DSTEM.data import RealSlice, Data
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.preprocess.utils import get_maxima_2D
from py4DSTEM.visualize import show,add_pointlabels,add_vector, add_bragg_index_labels
from py4DSTEM import PointList

class StrainMap(RealSlice,Data):
    """
    Stores strain map.

    TODO add docs

    """

    def __init__(
        self,
        braggvectors: BraggVectors,
        name: Optional[str] = 'strainmap'
        ):
        """
        TODO
        """
        assert(isinstance(braggvectors,BraggVectors)), f"braggvectors myst be BraggVectors, not type {type(braggvectors)}"

        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            name = name,
            data = np.empty((
                6,
                braggvectors.Rshape[0],
                braggvectors.Rshape[1],
            )),
            slicelabels = [
                'exx',
                'eyy',
                'exy',
                'theta',
                'mask',
                'error'
            ]
        )

        # set up braggvectors
        # this assigns the bvs, ensures the origin is calibrated,
        # and adds the strainmap to the bvs' tree
        self.braggvectors = braggvectors

        # initialize as Data
        Data.__init__(self)

        # set calstate
        # this property is used only to check to make sure that
        # the braggvectors being used throughout a workflow are
        # the same. The state of calibration of the vectors is noted
        # here, and then checked each time the vectors are used -
        # if they differ, an error message and instructions for
        # re-calibration are issued
        self.calstate = self.braggvectors.calstate
        assert self.calstate["center"], "braggvectors must be centered"
        # get the BVM
        # a new BVM using the current calstate is computed
        self.bvm = self.braggvectors.histogram()



    # braggvector properties

    @property
    def braggvectors(self):
        return self._braggvectors
    @braggvectors.setter
    def braggvectors(self,x):
        assert(isinstance(x,BraggVectors)), f".braggvectors must be BraggVectors, not type {type(x)}"
        assert(x.calibration.origin is not None), f"braggvectors must have a calibrated origin"
        self._braggvectors = x
        self._braggvectors.tree(self,force=True)


    def reset_calstate(self):
        """
        Resets the calibration state. This recomputes the BVM, and removes any computations
        this StrainMap instance has stored, which will need to be recomputed.
        """
        for attr in (
            'g0',
            'g1',
            'g2',
        ):
            if hasattr(self,attr):
                delattr(self,attr)
        self.calstate = self.braggvectors.calstate
        pass



    # Class methods

    def choose_lattice_vectors(
        self,
        index_g0,
        index_g1,
        index_g2,
        subpixel = 'multicorr',
        upsample_factor = 16,
        sigma=0,
        minAbsoluteIntensity=0,
        minRelativeIntensity=0,
        relativeToPeak=0,
        minSpacing=0,
        edgeBoundary=1,
        maxNumPeaks=10,
        figsize=(12,6),
        c_indices='lightblue',
        c0='g',
        c1='r',
        c2='r',
        c_vectors='r',
        c_vectorlabels='w',
        size_indices=20,
        width_vectors=1,
        size_vectorlabels=20,
        vis_params = {},
        returncalc = False,
        returnfig=False,
        ):
        """
        Choose which lattice vectors to use for strain mapping.

        Overlays the bvm with the points detected via local 2D
        maxima detection, plus an index for each point. User selects
        3 points using the overlaid indices, which are identified as
        the origin and the termini of the lattice vectors g1 and g2.

        Parameters
        ----------
        index_g0 : int
            selected index for the origin
        index_g1 : int
            selected index for g1
        index_g2 :int
            selected index for g2
        subpixel : str in ('pixel','poly','multicorr')
            See the docstring for py4DSTEM.preprocess.get_maxima_2D
        upsample_factor : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        sigma : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minAbsoluteIntensity : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minRelativeIntensity : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        relativeToPeak : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        minSpacing : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        edgeBoundary : number
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        maxNumPeaks : int
            See the py4DSTEM.preprocess.get_maxima_2D docstring
        figsize : 2-tuple
            the size of the figure
        c_indices : color
            color of the maxima
        c0 : color
            color of the origin
        c1 : color
            color of g1 point
        c2 : color
            color of g2 point
        c_vectors : color
            color of the g1/g2 vectors
        c_vectorlabels : color
            color of the vector labels
        size_indices : number
            size of the indices
        width_vectors : number
            width of the vectors
        size_vectorlabels : number
            size of the vector labels
        vis_params : dict
            additional visualization parameters passed to `show`
        returncalc : bool
            toggles returning the answer
        returnfig : bool
            toggles returning the figure

        Returns
        -------
        (optional) : None or (g0,g1,g2) or (fig,(ax1,ax2)) or both of the latter
        """
        # validate inputs
        for i in (index_g0,index_g1,index_g2):
            assert isinstance(i,(int,np.integer)), "indices must be integers!"
        # check the calstate
        assert(self.calstate == self.braggvectors.calstate), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        # find the maxima
        g = get_maxima_2D(
            self.bvm.data,
            subpixel = subpixel,
            upsample_factor = upsample_factor,
            sigma = sigma,
            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minSpacing = minSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,
        )

        # get the lattice vectors
        gx,gy = g['x'],g['y']
        g0 = gx[index_g0],gy[index_g0]
        g1x = gx[index_g1] - g0[0]
        g1y = gy[index_g1] - g0[1]
        g2x = gx[index_g2] - g0[0]
        g2y = gy[index_g2] - g0[1]
        g1,g2 = (g1x,g1y),(g2x,g2y)

        # make the figure
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
        show(self.bvm.data,figax=(fig,ax1),**vis_params)
        show(self.bvm.data,figax=(fig,ax2),**vis_params)

        # Add indices to left panel
        d = {'x':gx,'y':gy,'size':size_indices,'color':c_indices}
        d0 = {'x':gx[index_g0],'y':gy[index_g0],'size':size_indices,
              'color':c0,'fontweight':'bold','labels':[str(index_g0)]}
        d1 = {'x':gx[index_g1],'y':gy[index_g1],'size':size_indices,
              'color':c1,'fontweight':'bold','labels':[str(index_g1)]}
        d2 = {'x':gx[index_g2],'y':gy[index_g2],'size':size_indices,
              'color':c2,'fontweight':'bold','labels':[str(index_g2)]}
        add_pointlabels(ax1,d)
        add_pointlabels(ax1,d0)
        add_pointlabels(ax1,d1)
        add_pointlabels(ax1,d2)

        # Add vectors to right panel
        dg1 = {'x0':gx[index_g0],'y0':gy[index_g0],'vx':g1[0],'vy':g1[1],'width':width_vectors,
               'color':c_vectors,'label':r'$g_1$','labelsize':size_vectorlabels,'labelcolor':c_vectorlabels}
        dg2 = {'x0':gx[index_g0],'y0':gy[index_g0],'vx':g2[0],'vy':g2[1],'width':width_vectors,
               'color':c_vectors,'label':r'$g_2$','labelsize':size_vectorlabels,'labelcolor':c_vectorlabels}
        add_vector(ax2,dg1)
        add_vector(ax2,dg2)

        # store vectors
        self.g = g
        self.g0 = g0
        self.g1 = g1
        self.g2 = g2

        # return
        if returncalc and returnfig:
            return (g0,g1,g2),(fig,(ax1,ax2))
        elif returncalc:
            return (g0,g1,g2)
        elif returnfig:
            return (fig,(ax1,ax2))
        else:
            return
    
    def fit_lattice_vectors(
        self,
        x0 = None,
        y0 = None,
        max_peak_spacing = 2,
        mask = None,
        plot = True,
        vis_params = {},
        returncalc = False,
        ):
        """
        From an origin (x0,y0), a set of reciprocal lattice vectors gx,gy, and an pair of
        lattice vectors g1=(g1x,g1y), g2=(g2x,g2y), find the indices (h,k) of all the
        reciprocal lattice directions.

        Args:
            x0 : floagt
                x-coord of origin
            y0 : float
                y-coord of origin
            max_peak_spacing: float
                Maximum distance from the ideal lattice points
                to include a peak for indexing
            mask: bool
                Boolean mask, same shape as the pointlistarray, indicating which
                locations should be indexed. This can be used to index different regions of
                the scan with different lattices
            plot:bool
                plot results if tru
            vis_params : dict
                additional visualization parameters passed to `show`
            returncalc : bool
                if True, returns braggdirections, bragg_vectors_indexed, g1g2_map
        """
        # check the calstate
        assert(self.calstate == self.braggvectors.calstate), "The calibration state has changed! To resync the calibration state, use `.reset_calstate`."

        if x0 is None:
            x0 = self.braggvectors.Qshape[0]/2
        if y0 is None:
            y0 = self.braggvectors.Qshape[0]/2

        #index braggvectors
        from py4DSTEM.process.latticevectors import index_bragg_directions
        _, _, braggdirections = index_bragg_directions(
            x0,
            y0,
            self.g['x'],
            self.g['y'],
            self.g1,
            self.g2
        )

        self.braggdirections = braggdirections

        if plot:
            self.show_bragg_indexing(
                self.bvm,
                braggdirections = braggdirections,
                points = True,
                **vis_params,
            )

        #add indicies to braggvectors
        from py4DSTEM.process.latticevectors import add_indices_to_braggvectors

        bragg_vectors_indexed = add_indices_to_braggvectors(
            self.braggvectors,
            self.braggdirections,
            maxPeakSpacing = max_peak_spacing,
            qx_shift = self.braggvectors.Qshape[0]/2,
            qy_shift = self.braggvectors.Qshape[1]/2,
            mask = mask
        )

        self.bragg_vectors_indexed = bragg_vectors_indexed

        #fit bragg vectors
        from py4DSTEM.process.latticevectors import fit_lattice_vectors_all_DPs
        g1g2_map = fit_lattice_vectors_all_DPs(
            self.bragg_vectors_indexed
        )
        self.g1g2_map = g1g2_map

        if returncalc:
            braggdirections, bragg_vectors_indexed, g1g2_map
    
    def show_lattice_vectors(ar,x0,y0,g1,g2,color='r',width=1,labelsize=20,labelcolor='w',returnfig=False,**kwargs):
        """ Adds the vectors g1,g2 to an image, with tail positions at (x0,y0).  g1 and g2 are 2-tuples (gx,gy).
        """
        fig,ax = show(ar,returnfig=True,**kwargs)

        # Add vectors
        dg1 = {'x0':x0,'y0':y0,'vx':g1[0],'vy':g1[1],'width':width,
               'color':color,'label':r'$g_1$','labelsize':labelsize,'labelcolor':labelcolor}
        dg2 = {'x0':x0,'y0':y0,'vx':g2[0],'vy':g2[1],'width':width,
               'color':color,'label':r'$g_2$','labelsize':labelsize,'labelcolor':labelcolor}
        add_vector(ax,dg1)
        add_vector(ax,dg2)

        if returnfig:
            return fig,ax
        else:
            plt.show()
            return



    

    def show_bragg_indexing(
        self, 
        ar,
        braggdirections,
        voffset=5,
        hoffset=0,
        color='w',
        size=20,
        points=True,
        pointcolor='r',
        pointsize=50,
        returnfig=False,
        **kwargs
    ):
        """
        Shows an array with an overlay describing the Bragg directions

        Accepts:
            ar                  (arrray) the image
            bragg_directions    (PointList) the bragg scattering directions; must have coordinates
                                'qx','qy','h', and 'k'. Optionally may also have 'l'.
        """
        assert isinstance(braggdirections,PointList)
        for k in ('qx','qy','h','k'):
            assert k in braggdirections.data.dtype.fields

        fig,ax = show(ar,returnfig=True,**kwargs)
        d = {'braggdirections':braggdirections,'voffset':voffset,'hoffset':hoffset,'color':color,
             'size':size,'points':points,'pointsize':pointsize,'pointcolor':pointcolor}
        add_bragg_index_labels(ax,d)

        if returnfig:
            return fig,ax
        else:
            plt.show()
            return











# def index_bragg_directions(x0, y0, gx, gy, g1, g2):
#     """
#     From an origin (x0,y0), a set of reciprocal lattice vectors gx,gy, and an pair of
#     lattice vectors g1=(g1x,g1y), g2=(g2x,g2y), find the indices (h,k) of all the
#     reciprocal lattice directions.

#     The approach is to solve the matrix equation
#             ``alpha = beta * M``
#     where alpha is the 2xN array of the (x,y) coordinates of N measured bragg directions,
#     beta is the 2x2 array of the two lattice vectors u,v, and M is the 2xN array of the
#     h,k indices.

#     Args:
#         x0 (float): x-coord of origin
#         y0 (float): y-coord of origin
#         gx (1d array): x-coord of the reciprocal lattice vectors
#         gy (1d array): y-coord of the reciprocal lattice vectors
#         g1 (2-tuple of floats): g1x,g1y
#         g2 (2-tuple of floats): g2x,g2y

#     Returns:
#         (3-tuple) A 3-tuple containing:

#             * **h**: *(ndarray of ints)* first index of the bragg directions
#             * **k**: *(ndarray of ints)* second index of the bragg directions
#             * **bragg_directions**: *(PointList)* a 4-coordinate PointList with the
#               indexed bragg directions; coords 'qx' and 'qy' contain bragg_x and bragg_y
#               coords 'h' and 'k' contain h and k.
#     """
#     # Get beta, the matrix of lattice vectors
#     beta = np.array([[g1[0],g2[0]],[g1[1],g2[1]]])

#     # Get alpha, the matrix of measured bragg angles
#     alpha = np.vstack([gx-x0,gy-y0])

#     # Calculate M, the matrix of peak positions
#     M = lstsq(beta, alpha, rcond=None)[0].T
#     M = np.round(M).astype(int)

#     # Get h,k
#     h = M[:,0]
#     k = M[:,1]

#     # Store in a PointList
#     coords = [('qx',float),('qy',float),('h',int),('k',int)]
#     temp_array = np.zeros([], dtype = coords)
#     bragg_directions = PointList(data = temp_array)
#     bragg_directions.add_data_by_field((gx,gy,h,k))

#     return h,k, bragg_directions



def add_indices_to_braggvectors(braggpeaks, lattice, maxPeakSpacing, qx_shift=0,
                              qy_shift=0, mask=None):
    """
    Using the peak positions (qx,qy) and indices (h,k) in the PointList lattice,
    identify the indices for each peak in the PointListArray braggpeaks.
    Return a new braggpeaks_indexed PointListArray, containing a copy of braggpeaks plus
    three additional data columns -- 'h','k', and 'index_mask' -- specifying the peak
    indices with the ints (h,k) and indicating whether the peak was successfully indexed
    or not with the bool index_mask. If `mask` is specified, only the locations where
    mask is True are indexed.

    Args:
        braggpeaks (PointListArray): the braggpeaks to index. Must contain
            the coordinates 'qx', 'qy', and 'intensity'
        lattice (PointList): the positions (qx,qy) of the (h,k) lattice points.
            Must contain the coordinates 'qx', 'qy', 'h', and 'k'
        maxPeakSpacing (float): Maximum distance from the ideal lattice points
            to include a peak for indexing
        qx_shift,qy_shift (number): the shift of the origin in the `lattice` PointList
            relative to the `braggpeaks` PointListArray
        mask (bool): Boolean mask, same shape as the pointlistarray, indicating which
            locations should be indexed. This can be used to index different regions of
            the scan with different lattices

    Returns:
        (PointListArray): The original braggpeaks pointlistarray, with new coordinates
        'h', 'k', containing the indices of each indexable peak.
    """

    assert isinstance(braggpeaks,PointListArray)
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy','intensity')])
    assert isinstance(lattice, PointList)
    assert np.all([name in lattice.dtype.names for name in ('qx','qy','h','k')])

    if mask is None:
        mask = np.ones(braggpeaks.shape,dtype=bool)

    assert mask.shape == braggpeaks.shape, 'mask must have same shape as pointlistarray'
    assert mask.dtype == bool, 'mask must be boolean'

    indexed_braggpeaks = braggpeaks.copy()

    # add the coordinates if they don't exist
    if not ('h' in braggpeaks.dtype.names):
        indexed_braggpeaks = indexed_braggpeaks.add_fields([('h',int)])
    if not ('k' in braggpeaks.dtype.names):
        indexed_braggpeaks = indexed_braggpeaks.add_fields([('k',int)])

    # loop over all the scan positions
    for Rx, Ry in tqdmnd(mask.shape[0],mask.shape[1]):
        if mask[Rx,Ry]:
            pl = indexed_braggpeaks.get_pointlist(Rx,Ry)
            rm_peak_mask = np.zeros(pl.length,dtype=bool)

            for i in range(pl.length):
                r2 = (pl.data['qx'][i]-lattice.data['qx'] + qx_shift)**2 + \
                     (pl.data['qy'][i]-lattice.data['qy'] + qy_shift)**2
                ind = np.argmin(r2)
                if r2[ind] <= maxPeakSpacing**2:
                    pl.data['h'][i] = lattice.data['h'][ind]
                    pl.data['k'][i] = lattice.data['k'][ind]
                else:
                    rm_peak_mask[i] = True
            pl.remove(rm_peak_mask)

    indexed_braggpeaks.name = braggpeaks.name + "_indexed"
    return indexed_braggpeaks









    









    # IO methods

    # TODO - copy method

    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = RealSlice._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
        }
        return args



