# Functions to become DataCube methods

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes, gaussian_filter1d
from scipy.interpolate import interp1d

from emdfile import Array, Metadata, Node



class DataCubeMethods:
    """
    A container for DataCube object instance methods.
    """

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
        self.tree( data )

    def set_scan_shape(
        self,
        Rshape
        ):
        """
        Reshape the data given the real space scan shape.

        Accepts:
            Rshape (2-tuple)
        """
        from py4DSTEM.preprocess import set_scan_shape
        assert len(Rshape)==2, "Rshape must have a length of 2"
        d = set_scan_shape(self,Rshape[0],Rshape[1])
        return d


    def swap_RQ(
        self
        ):
        """
        Swaps the first and last two dimensions of the 4D datacube.
        """
        from py4DSTEM.preprocess import swap_RQ
        d = swap_RQ(self)
        return d

    def swap_Rxy(
        self
        ):
        """
        Swaps the real space x and y coordinates.
        """
        from py4DSTEM.preprocess import swap_Rxy
        d = swap_Rxy(self)
        return d

    def swap_Qxy(
        self
        ):
        """
        Swaps the diffraction space x and y coordinates.
        """
        from py4DSTEM.preprocess import swap_Qxy
        d = swap_Qxy(self)
        return d

    def crop_Q(
        self,
        ROI
        ):
        """
        Crops the data in diffraction space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Qx_min,Qx_max,Qy_min,Qy_max)
        """
        from py4DSTEM.preprocess import crop_data_diffraction
        assert len(ROI)==4, "Crop region `ROI` must have length 4"
        d = crop_data_diffraction(self,ROI[0],ROI[1],ROI[2],ROI[3])
        return d

    def crop_R(
        self,
        ROI
        ):
        """
        Crops the data in real space about the region specified by ROI.

        Accepts:
            ROI (4-tuple): Specifies (Rx_min,Rx_max,Ry_min,Ry_max)
        """
        from py4DSTEM.preprocess import crop_data_real
        assert len(ROI)==4, "Crop region `ROI` must have length 4"
        d = crop_data_real(self,ROI[0],ROI[1],ROI[2],ROI[3])
        return d

    def bin_Q(
        self,
        N,
        dtype = None
        ):
        """
        Bins the data in diffraction space by bin factor N

        Parameters
        ----------
        N : int
            The binning factor
        dtype : a datatype (optional)
            Specify the datatype for the output. If not passed, the datatype
            is left unchanged

        Returns
        ------
        datacube : DataCube
        """
        from py4DSTEM.preprocess import bin_data_diffraction
        d = bin_data_diffraction(self,N,dtype)
        return d

    def pad_Q(
        self,
        N = None,
        output_size = None
        ):
        """
        Pads the data in diffraction space by pad factor N, or to match output_size.

        Accepts:
            N (float, or Sequence[float]): the padding factor
            output_size ((int,int)): the padded output size
        """
        from py4DSTEM.preprocess import pad_data_diffraction
        d = pad_data_diffraction(self,pad_factor=N,output_size=output_size)
        return d

    def resample_Q(
        self,
        N = None,
        output_size = None,
        method='bilinear'
        ):
        """
        Resamples the data in diffraction space by resampling factor N, or to match output_size,
        using either 'fourier' or 'bilinear' interpolation.

        Accepts:
            N (float, or Sequence[float]): the resampling factor
            output_size ((int,int)): the resampled output size
            method (str): 'fourier' or 'bilinear' (default)
        """
        from py4DSTEM.preprocess import resample_data_diffraction
        d = resample_data_diffraction(self,resampling_factor=N,output_size=output_size,method=method)
        return d

    def bin_Q_mmap(
        self,
        N,
        dtype=np.float32
        ):
        """
        Bins the data in diffraction space by bin factor N for memory mapped data

        Accepts:
            N (int): the binning factor
            dtype: the data type
        """
        from py4DSTEM.preprocess import bin_data_mmap
        d = bin_data_mmap(self,N)
        return d

    def bin_R(
        self,
        N
        ):
        """
        Bins the data in real space by bin factor N

        Accepts:
            N (int): the binning factor
        """
        from py4DSTEM.preprocess import bin_data_real
        d = bin_data_real(self,N)
        return d

    def thin_R(
        self,
        N
        ):
        """
        Reduces the data in real space by skipping every N patterns in the x and y directions.

        Accepts:
            N (int): the thinning factor
        """
        from py4DSTEM.preprocess import thin_data_real
        d = thin_data_real(self,N)
        return d

    def filter_hot_pixels(
        self,
        thresh,
        ind_compare=1,
        return_mask=False
        ):
        """
        This function performs pixel filtering to remove hot / bright pixels. We first compute a moving local ordering filter,
        applied to the mean diffraction image. This ordering filter will return a single value from the local sorted intensity
        values, given by ind_compare. ind_compare=0 would be the highest intensity, =1 would be the second hightest, etc.
        Next, a mask is generated for all pixels which are least a value thresh higher than the local ordering filter output.
        Finally, we loop through all diffraction images, and any pixels defined by mask are replaced by their 3x3 local median.

        Args:
            datacube (DataCube):
            thresh (float): threshold for replacing hot pixels, if pixel value minus local ordering filter exceeds it.
            ind_compare (int): which median filter value to compare against. 0 = brightest pixel, 1 = next brightest, etc.
            return_mask (bool): if True, returns the filter mask

        Returns:
            datacube (DataCube)
            mask (optional, boolean Array) the bad pixel mask
        """
        from py4DSTEM.preprocess import filter_hot_pixels
        d = filter_hot_pixels(
            self,
            thresh,
            ind_compare,
            return_mask,
        )
        return d






    # Diffraction imaging

    def get_virtual_diffraction(
        self,
        method = 'max',
        mode = None,
        geometry = None,
        calibrated = False,
        shift_center = False,
        verbose = True,
        name = 'virtual_diffracton',
        returncalc = True,
        ):
        """
        Function to calculate virtual diffraction patterns

        Args:
            datacube (Datacube) : datacube class object which stores 4D-dataset
                needed for calculation
            method (str) : defines method used for diffraction pattern, options
                are 'mean', 'median', and 'max'
            mode (str) : defines mode for selecting area in real space to use
                for virtual diffraction. The default is None, which means no
                geometry will be applied and the whole datacube will be used
                for the calculation. Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright
                      field
                    - 'annular' or 'annulus' uses annular detector, like dark
                      field
                    - 'rectangle', 'square', 'rectangular', uses rectangular
                      detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`,
                values in pixels argument, as follows. The default is None, which
                means no geometry will be applied and the whole datacube will be
                used for the calculation. If mode is None the geometry will not
                be applied.
                    - 'point': 2-tuple, (rx,ry), ints
                    - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                    - 'annular' or 'annulus': nested 2-tuple,
                      ((rx,ry),(radius_i,radius_o))
                    - 'rectangle', 'square', 'rectangular': 4-tuple,
                      (rxmin,rxmax,rymin,rymax)
                    - `mask`: flexible detector, any boolean or floating point 2D
                      array with the same shape as datacube.Rshape
            calibrated (bool): if True, geometry is specified in units of 'A'
                instead of pixels. The datacube's calibrations must have its
                `"R_pixel_units"` parameter set to "A". If mode is None the
                geometry and calibration will not be applied.
            shift_center (bool): if True, the difraction patterns are shifted to
                account for beam shift or the changing of the origin through the
                scan. The datacube's calibration['origin'] parameter must be set.
                Only 'max' and 'mean' supported for this option.
            verbose (bool): if True, show progress bar

        Returns:
            (VirtualDiffraction): the diffraction image
        """

        # perform computation
        from py4DSTEM.classes.virtualdiffraction import VirtualDiffraction
        from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
        dp = get_virtual_diffraction(
            self,
            method = method,
            mode = mode,
            geometry = geometry,
            shift_center = shift_center,
            calibrated = calibrated,
            verbose = verbose,
        )

        # wrap with a py4dstem class
        dp = VirtualDiffraction(
            data = dp,
            name = name
        )

        # add the args used to gen this dp as metadata
        dp.metadata = Metadata(
            name='gen_params',
            data = {
                #'gen_func' : 
                'method' : method,
                'mode' : mode,
                'geometry' : geometry,
                'shift_center' : shift_center,
                'calibrated' : calibrated,
            }
        )

        # add to the tree
        self.tree( dp )

        # return
        if returncalc:
            return dp


    def get_dp_max(
        self,
        method = 'max',
        mode = None,
        geometry = None,
        calibrated = False,
        shift_center = False,
        verbose = True,
        name = 'dp_max',
        returncalc = True,
        ):
        """
        Function to calculate maximum virtual diffraction. Default captures
        pattern across entire 4D-dataset.

        Args:
            datacube (Datacube) : datacube class object which stores 4D-dataset
                needed for calculation
            mode (str) : defines mode for selecting area in real space to use for
                virtual diffraction. The default is None, which means no
                geometry will be applied and the whole datacube will be used
                for the calculation. Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright
                      field
                    - 'annular' or 'annulus' uses annular detector, like dark
                      field
                    - 'rectangle', 'square', 'rectangular', uses rectangular
                      detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`,
                values in pixels argument, as follows. The default is None, which
                means no geometry will be applied and the whole datacube will be
                used for the calculation. If mode is None the geometry will not
                be applied.
                    - 'point': 2-tuple, (rx,ry),
                       rx and ry are each single float or int to define center
                    - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                    - 'annular' or 'annulus': nested 2-tuple,
                      ((rx,ry),(radius_i,radius_o)),
                    - 'rectangle', 'square', 'rectangular': 4-tuple,
                      (xmin,xmax,ymin,ymax)
                    - `mask`: flexible detector, any boolean or floating point 2D
                      array with the same shape as datacube.Rshape
            calibrated (bool): if True, geometry is specified in units of 'A'
                instead of pixels. The datacube's calibrations must have its
                `"R_pixel_units"` parameter set to "A". If mode is None the
                geometry and calibration will not be applied.
            shift_center (bool) : if True, the difraction patterns are shifted to
                account for beam shift or the changing of the origin through the
                scan. The datacube's calibration['origin'] parameter must be set.
                Only 'max' and 'mean' supported for this option.
            verbose (bool): if True, show progress bar

        Returns:
            (VirtualDiffraction): the diffraction image
        """

        # perform computation
        from py4DSTEM.classes.virtualdiffraction import VirtualDiffraction
        from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
        dp = get_virtual_diffraction(
            self,
            method = method,
            mode = mode,
            geometry = geometry,
            shift_center = shift_center,
            calibrated = calibrated,
            verbose = verbose,
        )

        # wrap with a py4dstem class
        dp = VirtualDiffraction(
            data = dp,
            name = name
        )

        # add the args used to gen this dp as metadata
        dp.metadata = Metadata(
            name='gen_params',
            data = {
                #'gen_func' : 
                'method' : method,
                'mode' : mode,
                'geometry' : geometry,
                'shift_center' : shift_center,
                'calibrated' : calibrated,
            }
        )

        # add to the tree
        self.tree( dp )

        # return
        if returncalc:
            return dp


    def get_dp_mean(
        self,
        method = 'mean',
        mode = None,
        geometry = None,
        calibrated = False,
        shift_center = False,
        verbose = True,
        name = 'dp_mean',
        returncalc = True,
        ):
        """
        Function to calculate mean virtual diffraction. Default captures pattern
        across entire 4D-dataset.

        Args:
            datacube (Datacube) : datacube class object which stores 4D-dataset
                needed for calculation
            mode (str) : defines mode for selecting area in real space to use for
                virtual diffraction. The default is None, which means no
                geometry will be applied and the whole datacube will be used
                for the calculation. Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright
                      field
                    - 'annular' or 'annulus' uses annular detector, like dark
                      field
                    - 'rectangle', 'square', 'rectangular', uses rectangular
                      detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`,
                values in pixels argument, as follows. The default is None,
                which means no geometry will be applied and the whole datacube
                will be used for the calculation. If mode is None the geometry
                will not be applied.
                    - 'point': 2-tuple, (rx,ry),
                       qx and qy are each single float or int to define center
                    - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                       qx, qy and radius, are each single float or int
                    - 'annular' or 'annulus': nested 2-tuple,
                      ((rx,ry),(radius_i,radius_o)),
                    - 'rectangle', 'square', 'rectangular': 4-tuple,
                      (xmin,xmax,ymin,ymax)
                    - `mask`: flexible detector, any boolean or floating point 2D
                      array with the same shape as datacube.Rshape
            calibrated (bool): if True, geometry is specified in units of 'A'
                instead of pixels. The datacube's calibrations must have its
                `"R_pixel_units"` parameter set to "A". If mode is None the
                geometry and calibration will not be applied.
            shift_center (bool): if True, the diffraction patterns are shifted to
                account for beam shift or the changing of the origin through the
                scan. The datacube's calibration['origin'] parameter must be set.
                Only 'max' and 'mean' supported for this option.
            verbose (bool) : if True, show progress bar

        Returns:
            (VirtualDiffraction): the diffraction image
        """

        # perform computation
        from py4DSTEM.classes.virtualdiffraction import VirtualDiffraction
        from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
        dp = get_virtual_diffraction(
            self,
            method = method,
            mode = mode,
            geometry = geometry,
            shift_center = shift_center,
            calibrated = calibrated,
            verbose = verbose,
        )

        # wrap with a py4dstem class
        dp = VirtualDiffraction(
            data = dp,
            name = name,
        )

        # add the args used to gen this dp as metadata
        dp.metadata = Metadata(
            name='gen_params',
            data = {
                #'gen_func' : 
                'method' : method,
                'mode' : mode,
                'geometry' : geometry,
                'shift_center' : shift_center,
                'calibrated' : calibrated,
            }
        )

        # add to the tree
        self.tree( dp )

        # return
        if returncalc:
            return dp

    def get_dp_median(
        self,
        method = 'median',
        mode = None,
        geometry = None,
        calibrated = False,
        shift_center = False,
        verbose = True,
        name = 'dp_median',
        returncalc = True,
        ):
        """
        Function to calculate median virtual diffraction. Default captures
        pattern across entire 4D-dataset.

        Args:
            datacube (Datacube) : datacube class object which stores 4D-dataset
                needed for calculation
            mode (str) : defines mode for selecting area in real space to use for
                virtual diffraction. The default is None, which means no
                geometry will be applied and the whole datacube will be used
                for the calculation. Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright
                      field
                    - 'annular' or 'annulus' uses annular detector, like dark
                      field
                    - 'rectangle', 'square', 'rectangular', uses rectangular
                      detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`,
                values in pixels argument, as follows. The default is None,
                which means no geometry will be applied and the whole datacube
                will be used for the calculation. If mode is None the geometry
                will not be applied.
                    - 'point': 2-tuple, (rx,ry),
                    - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                    - 'annular' or 'annulus': nested 2-tuple,
                      ((rx,ry),(radius_i,radius_o)),
                    - 'rectangle', 'square', 'rectangular': 4-tuple,
                      (xmin,xmax,ymin,ymax)
                    - `mask`: flexible detector, any boolean or floating point 2D
                      array with the same shape as datacube.Rshape
            calibrated (bool): if True, geometry is specified in units of 'A'
                instead of pixels. The datacube's calibrations must have its
                `"R_pixel_units"` parameter set to "A". If mode is None the
                geometry and calibration will not be applied.
            shift_center (bool) : if True, the diffraction patterns are shifted to
                account for beam shift or the changing of the origin through the
                scan. The datacube's calibration['origin'] parameter must be set.
                Only 'max' and 'mean' supported for this option.
            verbose (bool): if True, show progress bar

        Returns:
            (VirtualDiffraction): the diffraction image
        """

        # perform computation
        from py4DSTEM.classes.virtualdiffraction import VirtualDiffraction
        from py4DSTEM.process.virtualdiffraction import get_virtual_diffraction
        dp = get_virtual_diffraction(
            self,
            method = method,
            mode = mode,
            geometry = geometry,
            shift_center = shift_center,
            calibrated = calibrated,
            verbose = verbose,
        )

        # wrap with a py4dstem class
        dp = VirtualDiffraction(
            data = dp,
            name = name,
        )

        # add the args used to gen this dp as metadata
        dp.metadata = Metadata(
            name='gen_params',
            data = {
                #'gen_func' : 
                'method' : method,
                'mode' : mode,
                'geometry' : geometry,
                'shift_center' : shift_center,
                'calibrated' : calibrated,
            }
        )

        # add to the tree
        self.tree( dp )

        # return
        if returncalc:
            return dp



    # Virtual imaging

    def get_virtual_image(
        self,
        mode,
        geometry,
        centered = False,
        calibrated = False,
        shift_center = False,
        verbose = True,
        dask = False,
        return_mask = False,
        name = 'virtual_image',
        returncalc = True,
        test_config = False
        ):
        """
        Get a virtual image and store it in `datacube`s tree under `name`.
        The kind of virtual image is specified by the `mode` argument.

        Args:
            mode (str): defines geometry mode for calculating virtual image
                options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright
                      field
                    - 'annular' or 'annulus' uses annular detector, like dark
                      field
                    - 'rectangle', 'square', 'rectangular', uses rectangular
                      detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`,
                values in
            pixels argument, as follows:
                - 'point': 2-tuple, (qx,qy), ints
                - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius),
                - 'annular' or 'annulus': nested 2-tuple,
                  ((qx,qy),(radius_i,radius_o)),
                - 'rectangle', 'square', 'rectangular': 4-tuple,
                  (xmin,xmax,ymin,ymax)
                - `mask`: any boolean or floating point 2D array with the same
                  size as datacube.Qshape
            centered (bool): if False, the origin is in the upper left corner.
                 If True, the origin is set to the mean origin in the datacube
                 calibrations, so that a bright-field image could be specified
                 with, e.g., geometry = ((0,0),R).  The origin can set with
                 datacube.calibration.set_origin().  For `mode="mask"`,
                 has no effect. Default is False.
            calibrated (bool): if True, geometry is specified in units of 'A^-1'
                instead of pixels. The datacube's calibrations must have its
                `"Q_pixel_units"` parameter set to "A^-1". For `mode="mask"`, has
                no effect. Default is False.
            shift_center (bool): if True, the mask is shifted at each real space
                position to account for any shifting of the origin of the
                diffraction images. The datacube's calibration['origin']
                parameter must be set. The shift applied to each pattern is the
                difference between the local origin position and the mean origin
                position over all patterns, rounded to the nearest integer for
                speed. Default is False.
            verbose (bool): if True, show progress bar
            dask (bool): if True, use dask arrays
            return_mask (bool): if False (default) returns a virtual image as
                usual. If True, does *not* generate or return a virtual image,
                instead returning the mask that would be used in virtual image
                computation for any call to this function where
                `shift_center = False`. Otherwise, must be a 2-tuple of integers
                corresponding to a scan position (rx,ry); in this case, returns
                the mask that would be used for virtual image computation at this
                scan position with `shift_center` set to `True`. Setting
                return_mask to True does not add anything to the datacube's tree
            name (str): the output object's name
            returncalc (bool): if True, returns the output
        Returns:
            (Optional): if returncalc is True, returns the VirtualImage
        """
        # perform computation
        from py4DSTEM.classes.virtualimage import VirtualImage
        from py4DSTEM.process.virtualimage import get_virtual_image
        im = get_virtual_image(
            self,
            mode = mode,
            geometry = geometry,
            centered = centered,
            calibrated = calibrated,
            shift_center = shift_center,
            verbose = verbose,
            dask = dask,
            return_mask = return_mask,
            test_config = test_config
        )

        # if a mask is requested, skip the remaining i/o functionality
        if return_mask is not False:
            return im

        # wrap with a py4dstem class
        im = VirtualImage(
            data = im,
            name = name,
        )

        # add generating params as metadata
        im.metadata = Metadata(
            name = 'gen_params',
            data = {
                'mode' : mode,
                'geometry' : geometry,
                'shift_center' : shift_center,
                'centered' : centered,
                'calibrated' : calibrated,
                'verbose' : verbose,
                'dask' : dask,
                'return_mask' : return_mask,
                'test_config' : test_config
            }
        )

        # add to the tree
        self.tree( im )

        # return
        if returncalc:
            return im


    # Position detector

    def position_detector(
        self,
        mode,
        geometry,
        scan_position = None,
        centered = None,
        calibrated = None,
        shift_center = None,
        invert = False,
        color = 'r',
        alpha = 0.7,
    ):
        """
        Display a diffraction space image with an overlaid mask representing
        a virtual detector.

        Args:
            mode: see py4DSTEM.process.get_virtual_image
            geometry: see py4DSTEM.process.get_virtual_image
            scan_position: if None, positions the unshifted detector over the
                mean or max diffraction pattern. Otherwise, must be a tuple
                (rx,ry) of ints, and a detector is positioned over the
                diffraction pattern at this position, including shifts if they
                would be applied for this dataset (i.e. if it contains the
                appropriate calibrations)
            centered (bool): if False, the origin is in the upper left corner.
                 If True, the origin is set to the mean origin in the datacube
                 calibrations, so that a bright-field image could be specified
                 with, e.g., geometry = ((0,0),R). The origin can set with
                 datacube.calibration.set_origin().  For `mode="mask"`,
                 has no effect. Default is False.
            calibrated (bool): if True, geometry is specified in units of 'A^-1'
                instead of pixels. The datacube's calibrations must have its
                `"Q_pixel_units"` parameter set to "A^-1". For `mode="mask"`, has
                no effect.
            shift_center (bool): if True, the mask is shifted at each real space
                position to account for any shifting of the origin of the
                diffraction images. The datacube's calibration['origin']
                parameter must be set. The shift applied to each pattern is the
                difference between the local origin position and the mean origin
                position over all patterns, rounded to the nearest integer for
                speed.
            invert (bool): if True, invert the display mask
        """

        # parse inputs
        if scan_position is None:
            data = self
        else:
            data = (self,scan_position[0],scan_position[1])

        # make and show visualization
        from py4DSTEM.visualize import position_detector
        position_detector(
            data,
            mode,
            geometry,
            centered,
            calibrated,
            shift_center,
            invert = invert,
            color = color,
            alpha = alpha,
        )




    # Probe

    def get_vacuum_probe(
        self,
        ROI = None,
        name = 'probe',
        returncalc = True,
        ):
        """
        Computes a vacuum probe from the DataCube by aligning and averaging
        either all or some subset of the diffraction patterns.

        Args:
            ROI (None or boolean array or tuple): if None, uses the whole
                datacube. Otherwise, uses a subset of diffraction patterns.
                If `ROI` is a boolean array, it should be Rspace shaped, and
                diffraction patterns where True are used. Else should be
                a 4-tuple representing (Rxmin,Rxmax,Rymin,Rymax) of a
                rectangular region to use.

        Returns:
            (Probe) a Probe instance

        """

        # perform computation
        from py4DSTEM.classes.probe import Probe
        from py4DSTEM.process.probe import get_vacuum_probe
        if ROI is None:
            x = get_vacuum_probe(
                self
            )
        else:
            x = get_vacuum_probe(
                self,
                ROI = ROI
            )

        # wrap with a py4dstem class
        x = Probe(
            data = x
        )

        # add to the tree
        self.tree( x )

        # return
        if returncalc:
            return x



    def get_probe_size(
        self,
        dp = None,
        thresh_lower=0.01,
        thresh_upper=0.99,
        N=100,
        plot = True,
        returncal = True,
        write_to_cal = True,
        **kwargs,
        ):
        """
        Gets the center and radius of the probe in the diffraction plane.

        The algorithm is as follows:
        First, create a series of N binary masks, by thresholding the diffraction
        pattern DP with a linspace of N thresholds from thresh_lower to
        thresh_upper, measured relative to the maximum intensity in DP.
        Using the area of each binary mask, calculate the radius r of a circular
        probe. Because the central disk is typically very intense relative to
        the rest of the DP, r should change very little over a wide range of
        intermediate values of the threshold. The range in which r is trustworthy
        is found by taking the derivative of r(thresh) and finding identifying
        where it is small.  The radius is taken to be the mean of these r values.
        Using the threshold corresponding to this r, a mask is created and the
        CoM of the DP times this mask it taken.  This is taken to be the origin
        x0,y0.

        Args:
            dp (str or array): specifies the diffraction pattern in which to
                find the central disk. A position averaged, or shift-corrected
                and averaged, DP works best. If mode is None, the diffraction
                pattern stored in the tree from 'get_dp_mean' is used. If mode
                is a string it specifies the name of another virtual diffraction
                pattern in the tree. If mode is an array, the array is used to
                calculate probe size.
            thresh_lower (float, 0 to 1): the lower limit of threshold values
            thresh_upper (float, 0 to 1): the upper limit of threshold values
            N (int): the number of thresholds / masks to use
            plot (bool): if True plots results
            plot_params(dict): dictionary to modify defaults in plot
            return_calc (bool): if True returns 3-tuple described below
            write_to_cal (bool): if True, looks for a Calibration instance
                and writes the measured probe radius there

        Returns:
            (3-tuple): A 3-tuple containing:

                * **r**: *(float)* the central disk radius, in pixels
                * **x0**: *(float)* the x position of the central disk center
                * **y0**: *(float)* the y position of the central disk center
        """
        #perform computation        
        from py4DSTEM.process.calibration import get_probe_size

        if dp is None:
            assert 'dp_mean' in self._branch.keys(), "calculate .get_dp_mean()"
            DP = self.tree( 'dp_mean' ).data
        elif type(dp) == str:
            assert dp in self._branch.keys(), "mode not found"
            DP = self.tree( dp )
        elif type(dp) == np.ndarray:
            assert len(dp.shape) == 2, "must be a 2D array"
            DP = dp

        x = get_probe_size(
            DP,
            thresh_lower = thresh_lower,
            thresh_upper = thresh_upper,
            N = N,
        )

        # try to add to calibration
        if write_to_cal:
            try:
                self.calibration.set_probe_param(x)
            except AttributeError:
                # should we raise an error here?
                pass

        #plot results 
        if plot:
            from py4DSTEM.visualize import show_circles
            show_circles(
                DP,
                (x[1], x[2]),
                x[0],
                vmin = 0,
                vmax = 1,
                **kwargs
            )

        # return
        if returncal:
            return x


    # Bragg disks

    def find_Bragg_disks(
        self,

        template,
        data = None,

        radial_bksb = False,
        filter_function = None,

        corrPower = 1,
        sigma = None,
        sigma_dp = 0,
        sigma_cc = 2,
        subpixel = 'multicorr',
        upsample_factor = 16,

        minAbsoluteIntensity = 0,
        minRelativeIntensity = 0.005,
        relativeToPeak = 0,
        minPeakSpacing = 60,
        edgeBoundary = 20,
        maxNumPeaks = 70,

        CUDA = False,
        CUDA_batched = True,
        distributed = None,

        ML = False,
        ml_model_path = None,
        ml_num_attempts = 1,
        ml_batch_size = 8,

        name = 'braggvectors',
        returncalc = True,
        ):
        """
        Finds the Bragg disks in the diffraction patterns represented by `data` by
        cross/phase correlatin with `template`.

        Behavior depends on `data`. If it is None (default), runs on the whole DataCube,
        and stores the output in its tree. Otherwise, nothing is stored in tree,
        but some value is returned. Valid entries are:

            - a 2-tuple of numbers (rx,ry): run on this diffraction image,
                and return a QPoints instance
            - a 2-tuple of arrays (rx,ry): run on these diffraction images,
                and return a list of QPoints instances
            - an Rspace shapped 2D boolean array: run on the diffraction images
                specified by the True counts and return a list of QPoints
                instances

        For disk detection on a full DataCube, the calculation can be performed
        on the CPU, GPU or a cluster. By default the CPU is used.  If `CUDA` is set
        to True, tries to use the GPU.  If `CUDA_batched` is also set to True,
        batches the FFT/IFFT computations on the GPU. For distribution to a cluster,
        distributed must be set to a dictionary, with contents describing how
        distributed processing should be performed - see below for details.


        For each diffraction pattern, the algorithm works in 4 steps:

        (1) any pre-processing is performed to the diffraction image. This is
            accomplished by passing a callable function to the argument
            `filter_function`, a bool to the argument `radial_bksb`, or a value >0
            to `sigma_dp`. If none of these are passed, this step is skipped.
        (2) the diffraction image is cross correlated with the template.
            Phase/hybrid correlations can be used instead by setting the
            `corrPower` argument. Cross correlation can be skipped entirely,
            and the subsequent steps performed directly on the diffraction
            image instead of the cross correlation, by passing None to
            `template`.
        (3) the maxima of the cross correlation are located and their
            positions and intensities stored. The cross correlation may be
            passed through a gaussian filter first by passing the `sigma_cc`
            argument. The method for maximum detection can be set with
            the `subpixel` parameter. Options, from something like fastest/least
            precise to slowest/most precise are 'pixel', 'poly', and 'multicorr'.
        (4) filtering is applied to remove untrusted or undesired positive counts,
            based on their intensity (`minRelativeIntensity`,`relativeToPeak`,
            `minAbsoluteIntensity`) their proximity to one another or the
            image edge (`minPeakSpacing`, `edgeBoundary`), and the total
            number of peaks per pattern (`maxNumPeaks`).


        Parameters
        ----------
        template : 2D array
            the vacuum probe template, in real space. For Probe instances,
            this is `probe.kernel`.  If None, does not perform a cross
            correlation.
        data : variable
            see above
        radial_bksb : bool
            if True, computes a radial background given by the median of the
            (circular) polar transform of each each diffraction pattern, and
            subtracts this background from the pattern before applying any
            filter function and computing the cross correlation. The origin
            position must be set in the datacube's calibrations. Currently
            only supported for full datacubes on the CPU.
        filter_function : callable
            filtering function to apply to each diffraction pattern before
            peak finding. Must be a function of only one argument (the
            diffraction pattern) and return the filtered diffraction pattern.
            The shape of the returned DP must match the shape of the probe
            kernel (but does not need to match the shape of the input
            diffraction pattern, e.g. the filter can be used to bin the
            diffraction pattern). If using distributed disk detection, the
            function must be able to be pickled with by dill.
        corrPower : float between 0 and 1, inclusive
            the cross correlation power. A value of 1 corresponds to a cross
            correlation, 0 corresponds to a phase correlation, and intermediate
            values correspond to hybrid correlations.
        sigma : float
            alias for `sigma_cc`
        sigma_dp : float
            if >0, a gaussian smoothing filter with this standard deviation
            is applied to the diffraction pattern before maxima are detected
        sigma_cc : float
            if >0, a gaussian smoothing filter with this standard deviation
            is applied to the cross correlation before maxima are detected
        subpixel : str
            Whether to use subpixel fitting, and which algorithm to use.
            Must be in ('none','poly','multicorr').
                * 'none': performs no subpixel fitting
                * 'poly': polynomial interpolation of correlogram peaks (default)
                * 'multicorr': uses the multicorr algorithm with DFT upsampling
        upsample_factor : int
            upsampling factor for subpixel fitting (only used when
            subpixel='multicorr')
        minAbsoluteIntensity : float
            the minimum acceptable correlation peak intensity, on an absolute scale
        minRelativeIntensity : float
            the minimum acceptable correlation peak intensity, relative to the
            intensity of the brightest peak
        relativeToPeak : int
            specifies the peak against which the minimum relative intensity is
            measured -- 0=brightest maximum. 1=next brightest, etc.
        minPeakSpacing : float
            the minimum acceptable spacing between detected peaks
        edgeBoundary (int): minimum acceptable distance for detected peaks from
            the diffraction image edge, in pixels.
        maxNumPeaks : int
            the maximum number of peaks to return
        CUDA : bool
            If True, import cupy and use an NVIDIA GPU to perform disk detection
        CUDA_batched : bool
            If True, and CUDA is selected, the FFT and IFFT steps of disk detection
            are performed in batches to better utilize GPU resources.
        distributed : dict
            contains information for parallel processing using an IPyParallel or
            Dask distributed cluster.  Valid keys are:
                * ipyparallel (dict):
                * client_file (str): path to client json for connecting to your
                    existing IPyParallel cluster
                * dask (dict): client (object): a dask client that connects to
                    your existing Dask cluster
                * data_file (str): the absolute path to your original data
                    file containing the datacube
                * cluster_path (str): defaults to the working directory during
                    processing
            if distributed is None, which is the default, processing will be in
            serial
        name : str
            name for the output BraggVectors
        returncalc : bool
            if True, returns the answer

        Returns
        -------
        variable
            See above.
        """
        from py4DSTEM.process.diskdetection import find_Bragg_disks

        sigma_cc = sigma if sigma is not None else sigma_cc

        # parse args
        if data is None:
            x = self
        elif isinstance(data, tuple):
            x = self, data[0], data[1]
        elif isinstance(data, np.ndarray):
            assert data.dtype == bool, 'array must be boolean'
            assert data.shape == self.Rshape, 'array must be Rspace shaped'
            x = self.data[data,:,:]
        else:
            raise Exception(f'unexpected type for `data` {type(data)}')


        # compute
        peaks = find_Bragg_disks(
            data = x,
            template = template,

            radial_bksb = radial_bksb,
            filter_function = filter_function,

            corrPower = corrPower,
            sigma_dp = sigma_dp,
            sigma_cc = sigma_cc,
            subpixel = subpixel,
            upsample_factor = upsample_factor,

            minAbsoluteIntensity = minAbsoluteIntensity,
            minRelativeIntensity = minRelativeIntensity,
            relativeToPeak = relativeToPeak,
            minPeakSpacing = minPeakSpacing,
            edgeBoundary = edgeBoundary,
            maxNumPeaks = maxNumPeaks,

            CUDA = CUDA,
            CUDA_batched = CUDA_batched,
            distributed = distributed,
            ML = ML,
            ml_model_path = ml_model_path,
            ml_num_attempts = ml_num_attempts,
            ml_batch_size = ml_batch_size,
        )

        if isinstance(peaks,Node):

            # add metadata
            peaks.name = name
            peaks.metadata = Metadata(
                name = 'gen_params',
                data = {
                    #'gen_func' : 
                    'template' : template,
                    'filter_function' : filter_function,
                    'corrPower' : corrPower,
                    'sigma_dp' : sigma_dp,
                    'sigma_cc' : sigma_cc,
                    'subpixel' : subpixel,
                    'upsample_factor' : upsample_factor,
                    'minAbsoluteIntensity' : minAbsoluteIntensity,
                    'minRelativeIntensity' : minRelativeIntensity,
                    'relativeToPeak' : relativeToPeak,
                    'minPeakSpacing' : minPeakSpacing,
                    'edgeBoundary' : edgeBoundary,
                    'maxNumPeaks' : maxNumPeaks,
                    'CUDA' : CUDA,
                    'CUDA_batched' : CUDA_batched,
                    'distributed' : distributed,
                    'ML' : ML,
                    'ml_model_path' : ml_model_path,
                    'ml_num_attempts' : ml_num_attempts,
                    'ml_batch_size' : ml_batch_size,

                }
            )

            # add to tree
            if data is None:
                self.tree( peaks )

        # return
        if returncalc:
            return peaks





    def get_beamstop_mask(
        self,
        threshold = 0.25,
        distance_edge = 4.0,
        include_edges = True,
        name = "mask_beamstop",
        returncalc = True,
        ):
        """
        This function uses the mean diffraction pattern plus a threshold to
        create a beamstop mask.

        Args:
            threshold (float):  Value from 0 to 1 defining initial threshold for
                beamstop mask, taken from the sorted intensity values - 0 is the
                dimmest pixel, while 1 uses the brighted pixels.
            distance_edge (float): How many pixels to expand the mask.
            include_edges (bool): If set to True, edge pixels will be included
                in the mask.
            name (string): Name of the output array.
            returncalc (bool): Set to true to return the result.

        Returns:
            (Optional): if returncalc is True, returns the beamstop mask

        """

        # Calculate dp_mean if needed
        if not "dp_mean" in self._branch.keys():
            self.get_dp_mean();

        # normalized dp_mean
        int_sort = np.sort(self.tree("dp_mean").data.ravel())
        ind = np.round(np.clip(
                int_sort.shape[0]*threshold,
                0,int_sort.shape[0])).astype('int')
        intensity_threshold = int_sort[ind]

        # Use threshold to calculate initial mask
        mask_beamstop = self.tree("dp_mean").data >= intensity_threshold

        # clean up mask
        mask_beamstop = np.logical_not(binary_fill_holes(np.logical_not(mask_beamstop)))
        mask_beamstop = binary_fill_holes(mask_beamstop)

        # Edges
        if include_edges:
            mask_beamstop[0,:] = False
            mask_beamstop[:,0] = False
            mask_beamstop[-1,:] = False
            mask_beamstop[:,-1] = False


        # Expand mask
        mask_beamstop = distance_transform_edt(mask_beamstop) < distance_edge

        # Wrap beamstop mask in a class
        x = Array(
            data = mask_beamstop,
            name = name
        )

        # Add metadata
        x.metadata = Metadata(
            name = 'gen_params',
            data = {
                #'gen_func' : 
                'threshold' : threshold,
                'distance_edge' : distance_edge,
                'include_edges' : include_edges,
                'name' : "mask_beamstop",
                'returncalc' : returncalc,
            }
        )

        # Add to tree
        self.tree(x)

        # return
        if returncalc:
            return mask_beamstop



    def get_radial_bkgrnd(
        self,
        rx,
        ry,
        sigma = 2
    ):
        """
        Computes and returns a background image for the diffraction
        pattern at (rx,ry), populated by radial rings of constant intensity
        about the origin, with the value of each ring given by the median
        value of the diffraction pattern at that radial distance.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        sigma : number
            If >0, applying a gaussian smoothing in the radial direction
            before returning

        Returns
        -------
        background : ndarray
            The radial background
        """
        # ensure a polar cube and origin exist
        assert(self.polar is not None), "No polar datacube found!"
        assert(self.calibration.get_origin() is not None), "No origin found!"

        # get the 1D median background
        bkgrd_ma_1d = np.ma.median( self.polar.data[rx,ry], axis=0 )
        bkgrd_1d = bkgrd_ma_1d.data
        bkgrd_1d[bkgrd_ma_1d.mask] = 0

        # smooth
        if sigma>0:
            bkgrd_1d = gaussian_filter1d(bkgrd_1d, sigma)

        # define the 2D cartesian coordinate system
        origin = self.calibration.get_origin()
        origin = origin[0][rx,ry],origin[1][rx,ry]
        qxx,qyy = self.qxx-origin[0], self.qyy-origin[1]

        # get distance qr in polar-elliptical coords
        ellipse = self.calibration.get_ellipse()
        ellipse = (1,1,0) if ellipse is None else ellipse
        a,b,theta = ellipse

        qrr = np.sqrt(
            ( (qxx*np.cos(theta)) + (qyy*np.sin(theta)) )**2 +
            ( (qxx*np.sin(theta)) - (qyy*np.cos(theta)) )**2 / (b/a)**2
        )

        # make an interpolation function and get the 2D background
        f = interp1d(
            self.polar.radial_bins,
            bkgrd_1d,
            fill_value = 'extrapolate'
        )
        background = f(qrr)

        # return
        return background



    def get_radial_bksb_dp(
        self,
        rx,
        ry,
        sigma = 2
    ):
        """
        Computes and returns the diffraction pattern at beam position (rx,ry)
        with a radial background subtracted.  See the docstring for
        datacube.get_radial_background for more info.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        sigma : number
            If >0, applying a gaussian smoothing in the radial direction
            before returning

        Returns
        -------
        data : ndarray
            The radial background subtracted diffraction image
        """
        # get 2D background
        background = self.get_radial_bkgrnd( rx,ry,sigma )

        # subtract, zero negative values, return
        ans = self.data[rx,ry] - background
        ans[ans<0] = 0
        return ans



    def get_local_ave_dp(
        self,
        rx,
        ry,
        radial_bksb = False,
        sigma = 2,
        braggmask = False,
        braggvectors = None,
        braggmask_radius = None
    ):
        """
        Computes and returns the diffraction pattern at beam position (rx,ry)
        after weighted local averaging with its nearest-neighbor patterns,
        using a 3x3 gaussian kernel for the weightings.

        Parameters
        ----------
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        radial_bksb : bool
            It True, apply a radial background subtraction to each pattern
            before averaging
        sigma : number
            If radial_bksb is True, use this sigma for radial smoothing of
            the background
        braggmask : bool
            If True, masks bragg scattering at each scan position before
            averaging. `braggvectors` and `braggmask_radius` must be
            specified.
        braggvectors : BraggVectors
            The Bragg vectors to use for masking
        braggmask_radius : number
            The radius about each Bragg point to mask

        Returns
        -------
        data : ndarray
            The radial background subtracted diffraction image
        """
        # define the kernel
        kernel = np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]])/16.

        # get shape and check for valid inputs
        nx,ny = self.data.shape[:2]
        assert(rx>=0 and rx<nx), "rx outside of scan range"
        assert(ry>=0 and ry<ny), "ry outside of scan range"

        # get the subcube, checking for edge patterns
        # and modifying the kernel as needed
        if rx!=0 and rx!=(nx-1) and ry!=0 and ry!=(ny-1):
            subcube = self.data[rx-1:rx+2,ry-1:ry+2,:,:]
        elif rx==0 and ry==0:
            subcube = self.data[:2,:2,:,:]
            kernel = kernel[1:,1:]
        elif rx==0 and ry==(ny-1):
            subcube = self.data[:2,-2:,:,:]
            kernel = kernel[1:,:-1]
        elif rx==(nx-1) and ry==0:
            subcube = self.data[-2:,:2,:,:]
            kernel = kernel[:-1,1:]
        elif rx==(nx-1) and ry==(ny-1):
            subcube = self.data[-2:,-2:,:,:]
            kernel = kernel[:-1,:-1]
        elif rx==0:
            subcube = self.data[:2,ry-1:ry+2,:,:]
            kernel = kernel[1:,:]
        elif rx==(nx-1):
            subcube = self.data[-2:,ry-1:ry+2,:,:]
            kernel = kernel[:-1,:]
        elif ry==0:
            subcube = self.data[rx-1:rx+2,:2,:,:]
            kernel = kernel[:,1:]
        elif ry==(ny-1):
            subcube = self.data[rx-1:rx+2,-2:,:,:]
            kernel = kernel[:,:-1]
        else:
            raise Exception(f'Invalid (rx,ry) = ({rx},{ry})...')

        # normalize the kernel
        kernel /= np.sum(kernel)


        # compute...

        # ...in the simple case
        if not(radial_bksb) and not(braggmask):
            ans = np.tensordot(subcube,kernel,axes=((0,1),(0,1)))

        # ...with radial background subtration
        elif radial_bksb and not(braggmask):
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx!=0 else 0
            _ys = 1 if ry!=0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            for (i,j),w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                ans += self.get_radial_bksb_dp(x,y,sigma) * w

        # ...with bragg masking
        elif not(radial_bksb) and braggmask:
            assert(braggvectors is not None), "`braggvectors` must be specified or `braggmask` must be turned off!"
            assert(braggmask_radius is not None), "`braggmask_radius` must be specified or `braggmask` must be turned off!"
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx!=0 else 0
            _ys = 1 if ry!=0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            weights = np.zeros(self.Qshape)
            for (i,j),w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                mask = self.get_braggmask(
                    braggvectors,
                    x,
                    y,
                    braggmask_radius
                )
                weights_curr = mask * w
                ans += self.data[x,y] * weights_curr
                weights += weights_curr
            # normalize
            out = np.full_like(ans, np.nan)
            ans_mask = weights>0
            ans = np.divide(
                ans,
                weights,
                out = out,
                where = ans_mask
            )
            # make masked array
            ans = np.ma.array(
                data = ans,
                mask = np.logical_not(ans_mask)
            )
            pass

        # ...with both radial background subtraction and bragg masking
        else:
            assert(braggvectors is not None), "`braggvectors` must be specified or `braggmask` must be turned off!"
            assert(braggmask_radius is not None), "`braggmask_radius` must be specified or `braggmask` must be turned off!"
            # get position of (rx,ry) relative to kernel
            _xs = 1 if rx!=0 else 0
            _ys = 1 if ry!=0 else 0
            x0 = rx - _xs
            y0 = ry - _ys
            # compute
            ans = np.zeros(self.Qshape)
            weights = np.zeros(self.Qshape)
            for (i,j),w in np.ndenumerate(kernel):
                x = x0 + i
                y = y0 + j
                mask = self.get_braggmask(
                    braggvectors,
                    x,
                    y,
                    braggmask_radius
                )
                weights_curr = mask * w
                ans += self.get_radial_bksb_dp(x,y,sigma) * weights_curr
                weights += weights_curr
            # normalize
            out = np.full_like(ans, np.nan)
            ans_mask = weights>0
            ans = np.divide(
                ans,
                weights,
                out = out,
                where = ans_mask
            )
            # make masked array
            ans = np.ma.array(
                data = ans,
                mask = np.logical_not(ans_mask)
            )
            pass

        # return
        return ans




    def get_braggmask(
        self,
        braggvectors,
        rx,
        ry,
        radius
    ):
        """
        Returns a boolean mask which is False in a radius of `radius` around
        each bragg scattering vector at scan position (rx,ry).

        Parameters
        ----------
        braggvectors : BraggVectors
            The bragg vectors
        rx : int
            The x-coord of the beam position
        ry : int
            The y-coord of the beam position
        radius : number
            mask pixels about each bragg vector to this radial distance

        Returns
        -------
        mask : boolean ndarray
        """
        # allocate space
        mask = np.ones( self.Qshape, dtype=bool )
        # get the vectors
        vects = braggvectors.raw[rx,ry]
        # loop
        for idx in range(len(vects.data)):
            qr = np.hypot(self.qxx-vects.qx[idx], self.qyy-vects.qy[idx])
            mask = np.logical_and(mask, qr>radius)
        return mask





