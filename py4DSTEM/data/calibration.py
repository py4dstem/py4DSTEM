# Defines the Calibration class, which stores calibration metadata

import numpy as np
from numbers import Number
from typing import Optional
from warnings import warn

from emdfile import Metadata, Root
from py4DSTEM.data.propagating_calibration import call_calibrate


class Calibration(Metadata):
    """
    Stores calibration measurements.

    Usage
    -----
    For some calibration instance `c`

        >>> c['x'] = y

    will set the value of some calibration item called 'x' to y, and

        >>> _y = c['x']

    will return the value currently stored as 'x' and assign it to _y.
    Additionally, for calibration items in the list `l` given below,
    the syntax

        >>> c.set_p(p)
        >>> p = c.get_p()

    is equivalent to

        >>> c.p = p
        >>> p = c.p

    is equivalent to

        >>> c['p'] = p
        >>> p = c['p']

    where in the first line of each couplet the parameter `p` is set and in
    the second it's retrieved, for parameters p in the list

                                                  calibrate
                                                  ---------
    l = [
        Q_pixel_size,                                 *
        R_pixel_size,                                 *
        Q_pixel_units,                                *
        R_pixel_units,                                *
        qx0,
        qy0,
        qx0_mean,
        qy0_mean,
        qx0shift,
        qy0shift,
        origin,                                       *
        origin_meas,
        origin_meas_mask,
        origin_shift,
        a,                                            *
        b,                                            *
        theta,                                        *
        p_ellipse,                                    *
        ellipse,                                      *
        QR_rotation_degrees,                          *
        QR_flip,                                      *
        QR_rotflip,                                   *
        probe_semiangle,
        probe_param,
        probe_center,
        probe_convergence_semiangle_pixels,
        probe_convergence_semiangle_mrad,
    ]

    There are two advantages to using the getter/setter syntax for parameters
    in `l` (e.g. either c.set_p or c.p) instead of the normal dictionary-like
    getter/setter syntax (i.e. c['p']).  These are (1) enabling retrieving
    parameters by beam scan position, and (2) enabling propagation of any
    calibration changes to downstream data objects which are affected by the
    altered calibrations.  See below.

    Get a parameter by beam scan position
    -------------------------------------
    Some parameters support retrieval by beam scan position.  In these cases,
    calling

        >>> c.get_p(rx,ry)

    will return the value of parameter p at beam position (rx,ry). This works
    only for the above syntax. Using either of

        >>> c.p
        >>> c['p']

    will return an R-space shaped array.

    Trigger downstream calibrations
    -------------------------------
    Some objects store their own internal calibration state, which depends on
    the calibrations stored here.  For example, a DataCube stores dimension
    vectors which calibrate its 4 dimensions, and which depend on the pixel
    sizes and the origin position.

    Modifying certain parameters therefore can trigger other objects which
    depend on these parameters to re-calibrate themselves by calling their
    .calibrate() method, if the object has one. Methods marked with a * in the
    list `l` above have this property. Only objects registered with the
    Calibration instance will have their .calibrate method triggered by changing
    these parameters.  An object `data` can be registered by calling

        >>> c.register_target( data )

    and deregistered with

        >>> c.deregister_target( data )

    If an object without a .calibrate method is registerd when a * method is
    called, nothing happens.

    The .calibrate methods are triggered by setting some parameter `p` using
    either

        >>> c.set_p( val )

    or

        >>> c.p = val

    syntax.  Setting the parameter with

        >>> c['p'] = val

    will not trigger re-calibrations.

    Calibration + Data
    ------------------
    Data in py4DSTEM is stored in filetree like representations, and
    Calibration instances are the top-level objects in these trees,
    in that they live here:

    Root
      |--metadata
      |     |-- *****---> calibration <---*****
      |
      |--some_object(e.g.datacube)
      |     |--another_object(e.g.max_dp)
      |             |--etc.
      |--etc.
      :

    Every py4DSTEM Data object has a tree with a calibration, and calling

        >>> data.calibration

    will return the that Calibration instance. See also the docstring
    for the `Data` class.

    Attaching an object to a different Calibration
    ----------------------------------------------
    To modify the calibration associated with some object `data`, use

        >>> c.attach( data )

    where `c` is the new calibration instance. This (1) moves `data` into the
    top level of `c`'s data tree, which means the new calibration will now be
    accessible normally at

        >>> data.calibration

    and (2) if and only if `data` was registered with its old calibration,
    de-registers it there and registers it with the new calibration. If
    `data` was not registered with the old calibration and it should be
    registered with the new one, `c.register_target( data )` should be
    called.

    To attach `data` to a different location in the calibration instance's
    tree, use `node.attach( data )`. See the Data.attach docstring.
    """

    def __init__(
        self,
        name: Optional[str] = "calibration",
        root: Optional[Root] = None,
    ):
        """
        Args:
            name (optional, str):
        """
        Metadata.__init__(self, name=name)

        # Set the root
        if root is None:
            root = Root(name="py4DSTEM_root")
        self.set_root(root)

        # List to hold objects that will re-`calibrate` when
        # certain properties are changed
        self._targets = []

        # set initial pixel values
        self["Q_pixel_size"] = 1
        self["R_pixel_size"] = 1
        self["Q_pixel_units"] = "pixels"
        self["R_pixel_units"] = "pixels"
        self["QR_flip"] = False

    # EMD root property
    @property
    def root(self):
        return self._root

    @root.setter
    def root(self):
        raise Exception(
            "Calibration.root does not support assignment; to change the root, use self.set_root"
        )

    def set_root(self, root):
        assert isinstance(root, Root), f"root must be a Root, not type {type(root)}"
        self._root = root

    # Attach data to the calibration instance
    def attach(self, data):
        """
        Attach `data` to this calibration instance, placing it in the top
        level of the Calibration instance's tree. If `data` was in a
        different data tree, remove it. If `data` was registered with
        a different calibration instance, de-register it there and
        register it here. If `data` was not previously registerd and it
        should be, after attaching it run `self.register_target(data)`.
        """
        from py4DSTEM.data import Data

        assert isinstance(data, Data), "data must be a Data instance"
        self.root.attach(data)

    # Register for auto-calibration
    def register_target(self, new_target):
        """
        Register an object to recieve calls to it `calibrate`
        method when certain calibrations get updated
        """
        if new_target not in self._targets:
            self._targets.append(new_target)

    def unregister_target(self, target):
        """
        Unlink an object from recieving calls to `calibrate` when
        certain calibration values are changed
        """
        if target in self._targets:
            self._targets.remove(target)

    @property
    def targets(self):
        return tuple(self._targets)

    ######### Begin Calibration Metadata Params #########

    # pixel size/units

    @call_calibrate
    def set_Q_pixel_size(self, x):
        self._params["Q_pixel_size"] = x

    def get_Q_pixel_size(self):
        return self._get_value("Q_pixel_size")

    # aliases
    @property
    def Q_pixel_size(self):
        return self.get_Q_pixel_size()

    @Q_pixel_size.setter
    def Q_pixel_size(self, x):
        self.set_Q_pixel_size(x)

    @property
    def qpixsize(self):
        return self.get_Q_pixel_size()

    @qpixsize.setter
    def qpixsize(self, x):
        self.set_Q_pixel_size(x)

    @call_calibrate
    def set_R_pixel_size(self, x):
        self._params["R_pixel_size"] = x

    def get_R_pixel_size(self):
        return self._get_value("R_pixel_size")

    # aliases
    @property
    def R_pixel_size(self):
        return self.get_R_pixel_size()

    @R_pixel_size.setter
    def R_pixel_size(self, x):
        self.set_R_pixel_size(x)

    @property
    def qpixsize(self):
        return self.get_R_pixel_size()

    @qpixsize.setter
    def qpixsize(self, x):
        self.set_R_pixel_size(x)

    @call_calibrate
    def set_Q_pixel_units(self, x):
        assert x in (
            "pixels",
            "A^-1",
            "mrad",
        ), "Q pixel units must be 'A^-1', 'mrad' or 'pixels'."
        self._params["Q_pixel_units"] = x

    def get_Q_pixel_units(self):
        return self._get_value("Q_pixel_units")

    # aliases
    @property
    def Q_pixel_units(self):
        return self.get_Q_pixel_units()

    @Q_pixel_units.setter
    def Q_pixel_units(self, x):
        self.set_Q_pixel_units(x)

    @property
    def qpixunits(self):
        return self.get_Q_pixel_units()

    @qpixunits.setter
    def qpixunits(self, x):
        self.set_Q_pixel_units(x)

    @call_calibrate
    def set_R_pixel_units(self, x):
        self._params["R_pixel_units"] = x

    def get_R_pixel_units(self):
        return self._get_value("R_pixel_units")

    # aliases
    @property
    def R_pixel_units(self):
        return self.get_R_pixel_units()

    @R_pixel_units.setter
    def R_pixel_units(self, x):
        self.set_R_pixel_units(x)

    @property
    def rpixunits(self):
        return self.get_R_pixel_units()

    @rpixunits.setter
    def rpixunits(self, x):
        self.set_R_pixel_units(x)

    # origin

    # qx0,qy0
    def set_qx0(self, x):
        self._params["qx0"] = x
        x = np.asarray(x)
        qx0_mean = np.mean(x)
        qx0_shift = x - qx0_mean
        self._params["qx0_mean"] = qx0_mean
        self._params["qx0_shift"] = qx0_shift

    def set_qx0_mean(self, x):
        self._params["qx0_mean"] = x

    def get_qx0(self, rx=None, ry=None):
        return self._get_value("qx0", rx, ry)

    def get_qx0_mean(self):
        return self._get_value("qx0_mean")

    def get_qx0shift(self, rx=None, ry=None):
        return self._get_value("qx0_shift", rx, ry)

    def set_qy0(self, x):
        self._params["qy0"] = x
        x = np.asarray(x)
        qy0_mean = np.mean(x)
        qy0_shift = x - qy0_mean
        self._params["qy0_mean"] = qy0_mean
        self._params["qy0_shift"] = qy0_shift

    def set_qy0_mean(self, x):
        self._params["qy0_mean"] = x

    def get_qy0(self, rx=None, ry=None):
        return self._get_value("qy0", rx, ry)

    def get_qy0_mean(self):
        return self._get_value("qy0_mean")

    def get_qy0shift(self, rx=None, ry=None):
        return self._get_value("qy0_shift", rx, ry)

    def set_qx0_meas(self, x):
        self._params["qx0_meas"] = x

    def get_qx0_meas(self, rx=None, ry=None):
        return self._get_value("qx0_meas", rx, ry)

    def set_qy0_meas(self, x):
        self._params["qy0_meas"] = x

    def get_qy0_meas(self, rx=None, ry=None):
        return self._get_value("qy0_meas", rx, ry)

    def set_origin_meas_mask(self, x):
        self._params["origin_meas_mask"] = x

    def get_origin_meas_mask(self, rx=None, ry=None):
        return self._get_value("origin_meas_mask", rx, ry)

    # aliases
    @property
    def qx0(self):
        return self.get_qx0()

    @qx0.setter
    def qx0(self, x):
        self.set_qx0(x)

    @property
    def qx0_mean(self):
        return self.get_qx0_mean()

    @qx0_mean.setter
    def qx0_mean(self, x):
        self.set_qx0_mean(x)

    @property
    def qx0shift(self):
        return self.get_qx0shift()

    @property
    def qy0(self):
        return self.get_qy0()

    @qy0.setter
    def qy0(self, x):
        self.set_qy0(x)

    @property
    def qy0_mean(self):
        return self.get_qy0_mean()

    @qy0_mean.setter
    def qy0_mean(self, x):
        self.set_qy0_mean(x)

    @property
    def qy0_shift(self):
        return self.get_qy0_shift()

    @property
    def qx0_meas(self):
        return self.get_qx0_meas()

    @qx0_meas.setter
    def qx0_meas(self, x):
        self.set_qx0_meas(x)

    @property
    def qy0_meas(self):
        return self.get_qy0_meas()

    @qy0_meas.setter
    def qy0_meas(self, x):
        self.set_qy0_meas(x)

    @property
    def origin_meas_mask(self):
        return self.get_origin_meas_mask()

    @origin_meas_mask.setter
    def origin_meas_mask(self, x):
        self.set_origin_meas_mask(x)

    # origin = (qx0,qy0)

    @call_calibrate
    def set_origin(self, x):
        """
        Args:
            x (2-tuple of numbers or of 2D, R-shaped arrays): the origin
        """
        qx0, qy0 = x
        self.set_qx0(qx0)
        self.set_qy0(qy0)

    def get_origin(self, rx=None, ry=None):
        qx0 = self._get_value("qx0", rx, ry)
        qy0 = self._get_value("qy0", rx, ry)
        ans = (qx0, qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans

    def get_origin_mean(self):
        qx0 = self._get_value("qx0_mean")
        qy0 = self._get_value("qy0_mean")
        return qx0, qy0

    def get_origin_shift(self, rx=None, ry=None):
        qx0 = self._get_value("qx0_shift", rx, ry)
        qy0 = self._get_value("qy0_shift", rx, ry)
        ans = (qx0, qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans

    def set_origin_meas(self, x):
        """
        Args:
            x (2-tuple or 3 uple of 2D R-shaped arrays): qx0,qy0,[mask]
        """
        qx0, qy0 = x[0], x[1]
        self.set_qx0_meas(qx0)
        self.set_qy0_meas(qy0)
        try:
            m = x[2]
            self.set_origin_meas_mask(m)
        except IndexError:
            pass

    def get_origin_meas(self, rx=None, ry=None):
        qx0 = self._get_value("qx0_meas", rx, ry)
        qy0 = self._get_value("qy0_meas", rx, ry)
        ans = (qx0, qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans

    # aliases
    @property
    def origin(self):
        return self.get_origin()

    @origin.setter
    def origin(self, x):
        self.set_origin(x)

    @property
    def origin_meas(self):
        return self.get_origin_meas()

    @origin_meas.setter
    def origin_meas(self, x):
        self.set_origin_meas(x)

    @property
    def origin_shift(self):
        return self.get_origin_shift()

    # ellipse

    @call_calibrate
    def set_a(self, x):
        self._params["a"] = x

    def get_a(self, rx=None, ry=None):
        return self._get_value("a", rx, ry)

    @call_calibrate
    def set_b(self, x):
        self._params["b"] = x

    def get_b(self, rx=None, ry=None):
        return self._get_value("b", rx, ry)

    @call_calibrate
    def set_theta(self, x):
        self._params["theta"] = x

    def get_theta(self, rx=None, ry=None):
        return self._get_value("theta", rx, ry)

    @call_calibrate
    def set_ellipse(self, x):
        """
        Args:
            x (3-tuple): (a,b,theta)
        """
        a, b, theta = x
        self._params["a"] = a
        self._params["b"] = b
        self._params["theta"] = theta

    @call_calibrate
    def set_p_ellipse(self, x):
        """
        Args:
            x (5-tuple): (qx0,qy0,a,b,theta) NOTE: does *not* change qx0,qy0!
        """
        _, _, a, b, theta = x
        self._params["a"] = a
        self._params["b"] = b
        self._params["theta"] = theta

    def get_ellipse(self, rx=None, ry=None):
        a = self.get_a(rx, ry)
        b = self.get_b(rx, ry)
        theta = self.get_theta(rx, ry)
        ans = (a, b, theta)
        if any([x is None for x in ans]):
            ans = None
        return ans

    def get_p_ellipse(self, rx=None, ry=None):
        qx0, qy0 = self.get_origin(rx, ry)
        a, b, theta = self.get_ellipse(rx, ry)
        return (qx0, qy0, a, b, theta)

    # aliases
    @property
    def a(self):
        return self.get_a()

    @a.setter
    def a(self, x):
        self.set_a(x)

    @property
    def b(self):
        return self.get_b()

    @b.setter
    def b(self, x):
        self.set_b(x)

    @property
    def theta(self):
        return self.get_theta()

    @theta.setter
    def theta(self, x):
        self.set_theta(x)

    @property
    def p_ellipse(self):
        return self.get_p_ellipse()

    @p_ellipse.setter
    def p_ellipse(self, x):
        self.set_p_ellipse(x)

    @property
    def ellipse(self):
        return self.get_ellipse()

    @ellipse.setter
    def ellipse(self, x):
        self.set_ellipse(x)

    # Q/R-space rotation and flip

    @call_calibrate
    def set_QR_rotation(self, x):
        self._params["QR_rotation"] = x
        self._params["QR_rotation_degrees"] = np.degrees(x)

    def get_QR_rotation(self):
        return self._get_value("QR_rotation")

    @call_calibrate
    def set_QR_rotation_degrees(self, x):
        self._params["QR_rotation"] = np.radians(x)
        self._params["QR_rotation_degrees"] = x

    def get_QR_rotation_degrees(self):
        return self._get_value("QR_rotation_degrees")

    @call_calibrate
    def set_QR_flip(self, x):
        self._params["QR_flip"] = x

    def get_QR_flip(self):
        return self._get_value("QR_flip")

    @call_calibrate
    def set_QR_rotflip(self, rot_flip):
        """
        Args:
            rot_flip (tuple), (rot, flip) where:
                rot (number): rotation in degrees
                flip (bool): True indicates a Q/R axes flip
        """
        rot, flip = rot_flip
        self._params["QR_rotation"] = rot
        self._params["QR_rotation_degrees"] = np.degrees(rot)
        self._params["QR_flip"] = flip

    @call_calibrate
    def set_QR_rotflip_degrees(self, rot_flip):
        """
        Args:
            rot_flip (tuple), (rot, flip) where:
                rot (number): rotation in degrees
                flip (bool): True indicates a Q/R axes flip
        """
        rot, flip = rot_flip
        self._params["QR_rotation"] = np.radians(rot)
        self._params["QR_rotation_degrees"] = rot
        self._params["QR_flip"] = flip

    def get_QR_rotflip(self):
        rot = self.get_QR_rotation()
        flip = self.get_QR_flip()
        if rot is None or flip is None:
            return None
        return (rot, flip)

    def get_QR_rotflip_degrees(self):
        rot = self.get_QR_rotation_degrees()
        flip = self.get_QR_flip()
        if rot is None or flip is None:
            return None
        return (rot, flip)

    # aliases
    @property
    def QR_rotation_degrees(self):
        return self.get_QR_rotation_degrees()

    @QR_rotation_degrees.setter
    def QR_rotation_degrees(self, x):
        self.set_QR_rotation_degrees(x)

    @property
    def QR_flip(self):
        return self.get_QR_flip()

    @QR_flip.setter
    def QR_flip(self, x):
        self.set_QR_flip(x)

    @property
    def QR_rotflip(self):
        return self.get_QR_rotflip()

    @QR_rotflip.setter
    def QR_rotflip(self, x):
        self.set_QR_rotflip(x)

    # probe

    def set_probe_semiangle(self, x):
        self._params["probe_semiangle"] = x

    def get_probe_semiangle(self):
        return self._get_value("probe_semiangle")

    def set_probe_param(self, x):
        """
        Args:
            x (3-tuple): (probe size, x0, y0)
        """
        probe_semiangle, qx0, qy0 = x
        self.set_probe_semiangle(probe_semiangle)
        self.set_qx0_mean(qx0)
        self.set_qy0_mean(qy0)

    def get_probe_param(self):
        probe_semiangle = self._get_value("probe_semiangle")
        qx0 = self._get_value("qx0")
        qy0 = self._get_value("qy0")
        ans = (probe_semiangle, qx0, qy0)
        if any([x is None for x in ans]):
            ans = None
        return ans

    def set_convergence_semiangle_pixels(self, x):
        self._params["convergence_semiangle_pixels"] = x

    def get_convergence_semiangle_pixels(self):
        return self._get_value("convergence_semiangle_pixels")

    def set_convergence_semiangle_mrad(self, x):
        self._params["convergence_semiangle_mrad"] = x

    def get_convergence_semiangle_mrad(self):
        return self._get_value("convergence_semiangle_mrad")

    def set_probe_center(self, x):
        self._params["probe_center"] = x

    def get_probe_center(self):
        return self._get_value("probe_center")

    # aliases
    @property
    def probe_semiangle(self):
        return self.get_probe_semiangle()

    @probe_semiangle.setter
    def probe_semiangle(self, x):
        self.set_probe_semiangle(x)

    @property
    def probe_param(self):
        return self.get_probe_param()

    @probe_param.setter
    def probe_param(self, x):
        self.set_probe_param(x)

    @property
    def probe_center(self):
        return self.get_probe_center()

    @probe_center.setter
    def probe_center(self, x):
        self.set_probe_center(x)

    @property
    def probe_convergence_semiangle_pixels(self):
        return self.get_probe_convergence_semiangle_pixels()

    @probe_convergence_semiangle_pixels.setter
    def probe_convergence_semiangle_pixels(self, x):
        self.set_probe_convergence_semiangle_pixels(x)

    @property
    def probe_convergence_semiangle_mrad(self):
        return self.get_probe_convergence_semiangle_mrad()

    @probe_convergence_semiangle_mrad.setter
    def probe_convergence_semiangle_mrad(self, x):
        self.set_probe_convergence_semiangle_mrad(x)

    ######## End Calibration Metadata Params ########

    # calibrate targets
    @call_calibrate
    def calibrate(self):
        pass

    # For parameters which can have 2D or (2+n)D array values,
    # this function enables returning the value(s) at a 2D position,
    # rather than the whole array
    def _get_value(self, p, rx=None, ry=None):
        """Enables returning the value of a pixel (rx,ry),
        if these are passed and `p` is an appropriate array
        """
        v = self._params.get(p)

        if v is None:
            return v

        if (rx is None) or (ry is None) or (not isinstance(v, np.ndarray)):
            return v

        else:
            er = f"`rx` and `ry` must be ints; got values {rx} and {ry}"
            assert np.all([isinstance(i, (int, np.integer)) for i in (rx, ry)]), er
            return v[rx, ry]

    def copy(self, name=None):
        """ """
        if name is None:
            name = self.name + "_copy"
        cal = Calibration(name=name)
        cal._params.update(self._params)
        return cal

    # HDF5 i/o

    # write is inherited from Metadata
    def to_h5(self, group):
        """
        Saves the metadata dictionary _params to group, then adds the
        calibration's target's list
        """
        # Add targets list to metadata
        targets = [x._treepath for x in self.targets]
        self["_target_paths"] = targets
        # Save the metadata
        Metadata.to_h5(self, group)
        del self._params["_target_paths"]

    # read
    @classmethod
    def from_h5(cls, group):
        """
        Takes a valid group for an HDF5 file object which is open in
        read mode. Determines if it's a valid Metadata representation, and
        if so loads and returns it as a Calibration instance. Otherwise,
        raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A Calibration instance
        """
        # load the group as a Metadata instance
        metadata = Metadata.from_h5(group)

        # convert it to a Calibration instance
        cal = Calibration(name=metadata.name)
        cal._params.update(metadata._params)

        # return
        return cal


########## End of class ##########
