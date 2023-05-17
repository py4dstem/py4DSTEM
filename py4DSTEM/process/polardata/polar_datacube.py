import numpy as np
from py4DSTEM.classes import DataCube




class PolarDatacube:

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
        assert(isinstance(datacube,DataCube))
        self._datacube = datacube

        # check for calibrations
        assert(hasattr(self._datacube,'calibration')), "No .calibration found"
        self.calibration = self._datacube.calibration

        # setup data getter
        self._set_polar_data_getter()


        # setup sampling

        # polar
        if qmax is None:
            qmax = np.min(self._datacube.Qshape) / np.sqrt(2)
        self.set_radial_bins(qmin,qmax,qstep)
        self.set_annular_bins(n_annular)

        # cartesian
        self._xa,self._ya = np.meshgrid(
            np.arange(self._datacube.Q_Nx),
            np.arange(self._datacube.Q_Ny),
            indexing = 'ij'
        )

        pass



    # sampling methods + properties

    def set_radial_bins(
        self,
        qmin,
        qmax,
        qstep,
    ):
        self._qmin = qmin
        self._qmax = qmax
        self._qstep = qstep

        self.radial_bins = np.arange(
            self._qmin,
            self._qmax,
            self._qstep
        )
        self.set_polar_shape()

    @property
    def qmin(self):
        return self._qmin
    @qmin.setter
    def qmin(self, x):
        self.set_radial_bins(
            x,
            self._qmax,
            self._qstep
        )

    @property
    def qmax(self):
        return self._qmax
    @qmin.setter
    def qmax(self, x):
        self.set_radial_bins(
            self._qmin,
            x,
            self._qstep
        )

    @property
    def qstep(self):
        return self._qstep
    @qstep.setter
    def qstep(self, x):
        self.set_radial_bins(
            self._qmin,
            self._qmax,
            x
        )

    def set_annular_bins(
        self,
        n_annular
    ):
        self._n_annular = n_annular
        self._annular_bins = np.linspace(
            0,
            2*np.pi,
            self._n_annular,
            endpoint = False
        )
        self.set_polar_shape()
        self._annular_step = self.annular_bins[1] - self.annular_bins[0]

    @property
    def annular_bins(self):
        return self._annular_bins
    @property
    def annular_step(self):
        return self._annular_step

    @property
    def n_annular(self):
        return self._n_annular
    @n_annular.setter
    def n_annular(self, x):
        self.set_annular_bins(x)

    def set_polar_shape(self):
        if hasattr(self,'radial_bins') and hasattr(self,'annular_bins'):
            self.polar_shape = np.array((
                self.annular_bins.shape[0],
                self.radial_bins.shape[0]
            ))
            self.polar_size = np.prod(self.polar_shape)


    # expose raw data
    @property
    def data_raw(self):
        return self._datacube


    # expose transformed data
    @property
    def data(self):
        return self._polar_data_getter

    def _set_polar_data_getter(self):
        self._polar_data_getter = PolarDataGetter(
            polarcube = self
        )




class PolarDataGetter:

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
        return ans

    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += "Retrieves the diffraction pattern at scan position (x,y) in polar coordinates when sliced with [x,y]."
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

        # origin
        assert(cal.get_origin(x,y) is not None), "No center found! Try setting the origin."
        origin = cal.get_origin(x,y)

        # shifted coordinates
        x = self._polarcube._xa - origin[0]
        y = self._polarcube._ya - origin[1]

        # polar coordinate indices
        r_ind = (np.sqrt(x**2 + y**2) - self._polarcube.radial_bins[0]) / self._polarcube.qstep
        t_ind = np.arctan2(y, x) / self._polarcube.annular_step
        r_ind_floor = np.floor(r_ind).astype('int')
        t_ind_floor = np.floor(t_ind).astype('int')
        dr = r_ind - r_ind_floor
        dt = t_ind - t_ind_floor
        # t_ind_floor = np.mod(t_ind_floor, self.num_annular_bins)

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


