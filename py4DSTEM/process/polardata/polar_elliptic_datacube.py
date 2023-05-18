import numpy as np
from py4DSTEM.classes import DataCube
from py4DSTEM.process.polardata.polar_datacube import PolarDatacube,PolarDataGetter



class PolarEllipticDatacube(PolarDatacube):

    """
    An interface to a 4D-STEM datacube under polar-elliptical transformation.
    """

    def __init__(
        self,
        datacube,
        qmin = 0.0,
        qmax = None,
        qstep = 1.0,
        n_annular = 180,
        qscale = None
        ):
        """
        Parameters
        ----------
        datacube : DataCube
            The datacube in cartesian coordinates
        qmin : number
            Minumum radius of the polar transformation
        qmax : number or None
            Maximum radius of the polar transformation
        qstep : number
            Width of radial bins
        n_annular = 180
            Number of bins in the annular direction. Bins will each
            have a width of 360/num_annular_bins, in degrees
        """

        # initialize as a PolarDatacube
        PolarDatacube.__init__(
            self,
            datacube = datacube,
            qmin = qmin,
            qmax = qmax,
            qstep = qstep,
            n_annular = n_annular,
            qscale = qscale
        )

        # overwrite PolarDatacube's getter
        self._set_polar_elliptic_data_getter()

    def _set_polar_elliptic_data_getter(self):
        self._polar_data_getter = PolarEllipticDataGetter(
            polarcube = self
        )



class PolarEllipticDataGetter(PolarDataGetter):

    def __init__(
        self,
        polarcube,
    ):
        self._polarcube = polarcube

    def __getitem__(self,pos):
        # unpack scan position
        x,y = pos
        # get the data
        cartesian_data = self._polarcube._datacube[x,y]
        # find the origin
        origin = self._get_origin(
            (x,y),
            self._polarcube.calibration
        )
        # get the ellipticity
        ellipse = self._get_ellipticity(
            (x,y),
            self._polarcube.calibration
        )
        # apply the transform
        ans = self._transform(
            cartesian_data = cartesian_data,
            origin = origin,
            ellipse = ellipse
        )
        # get a normalization array
        norm = self._transform(
            np.ones_like(cartesian_data),
            origin = origin,
            ellipse = ellipse
        )
        # normalize
        ans = np.divide(ans, norm, out=np.zeros_like(ans), where=norm!=0)
        # scaling
        if self._polarcube.qscale is not None:
            ans *= self._polarcube._qscale_ar[np.newaxis,:]
        # return
        return ans


    def _get_ellipticity(
        self,
        pos,
        cal
        ):
        x,y = pos
        assert(cal.get_ellipse(x,y) is not None), "No elliptical calibrations found!"
        ellipse = cal.get_ellipse(x,y)
        return ellipse



    def _transform(
        self,
        cartesian_data,
        origin = None,
        ellipse = None
        ):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.
        """
        # get calibrations
        if origin is None:
            origin = self._polarcube.calibration.get_origin_mean()
        if ellipse is None:
            ellipse = self._polarcube.calibration.get_ellipse()

        # unpack ellipse
        a,b,theta = ellipse

        # shifted coordinates
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]


        # transformation matrix (elliptic cartesian -> circular cartesian)
        A = (a/b)*np.cos(theta)
        B = - np.sin(theta)
        C = (a/b)*np.sin(theta)
        D = np.cos(theta)
        det = 1 / (A*D - B*C)

        # circular cartesian coords
        xc =  x*D - y*B
        yc = -x*C + y*A

        # polar coords
        rr = det * np.hypot(
            xc,
            yc
        )
        tt = np.arctan2(yc,xc) - np.pi/2

        # get bins and interpolation vals
        r_ind = (rr - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = tt / self._polarcube.annular_step
        r_ind_floor = np.floor(r_ind).astype('int')
        t_ind_floor = np.floor(t_ind).astype('int')
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor

        # get polar resampled data
        sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self._polarcube.polar_shape[1])
        im = np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (1 - dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (1 - dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self._polarcube.polar_shape[1]-1)
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub],self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (    dr[sub]) * (1 - dt[sub]),
            minlength = self._polarcube.polar_size,
        )
        im += np.bincount(
            r_ind_floor[sub] + 1 + \
            np.mod(t_ind_floor[sub] + 1,self._polarcube.polar_shape[0]) * self._polarcube.polar_shape[1],
            weights = cartesian_data[sub] * (    dr[sub]) * (    dt[sub]),
            minlength = self._polarcube.polar_size,
        )

        # output
        ans = np.reshape(im, self._polarcube.polar_shape)
        return ans




