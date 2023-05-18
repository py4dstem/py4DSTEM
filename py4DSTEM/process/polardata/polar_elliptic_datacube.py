import numpy as np
from py4DSTEM.classes import DataCube
from py4DSTEM.process.polardata.polar_datacube import PolarDatacube



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
        )

        # overwrite PolarDatacube's getter
        self._set_polar_elliptic_data_getter()

    def _set_polar_elliptic_data_getter(self):
        self._polar_data_getter = PolarEllipticDataGetter(
            polarcube = self
        )



class PolarEllipticDataGetter:

    def __init__(
        self,
        polarcube,
    ):
        self._polarcube = polarcube

    def __getitem__(self,pos):
        x,y = pos
        ans = self._polarcube._datacube[x,y]
        ans = self._transform(
            cartesian_data = ans,
            cal = self._polarcube.calibration,
            scanxy = (x,y),
        )
        norm = self._transform(
            np.ones_like(self._polarcube._datacube[x,y]),
            cal = self._polarcube.calibration,
            scanxy = (x,y)
        )
        ans = np.divide(ans, norm, out=np.zeros_like(ans), where=norm!=0)
        return ans

    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += "Retrieves the diffraction pattern at scan position (x,y) in polar elliptic coordinates when sliced with [x,y]."
        return string

    def _transform(
        self,
        cartesian_data,
        cal,
        scanxy,
        ):
        """
        Return a transformed copy of the diffraction pattern `cartesian_data`.
        """
        # scan position
        x,y = scanxy

        # get calibrations
        assert(cal.get_origin(x,y) is not None), "No center found! Try setting the origin."
        assert(cal.get_ellipse(x,y) is not None), "No elliptical calibrations found!"
        origin = cal.get_origin(x,y)
        ellipse = cal.get_ellipse(x,y)
        a,b,theta = ellipse

        # shifted coordinates
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]


        # polar-elliptic coordinates
        tt = np.arctan2(y,x) + np.pi/2 - theta
        rr = np.hypot(x,y) * (1 - (1-(b/a))*np.square(np.sin(tt))) # is this right?????
        r_ind = (rr - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = tt / self._polarcube.annular_step
        r_ind_floor = np.floor(r_ind).astype('int')
        t_ind_floor = np.floor(t_ind).astype('int')
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor


        # polar transformation
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




