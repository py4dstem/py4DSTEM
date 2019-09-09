# Rotational calibrations

import numpy as np

class RQ_axes(object):
    """
    An object which specifies the relative orientation of real and reciprocal space, keeps track
    of internal coordinate systems used for output (e.g. cartesian axes along which to define strain,
    which may be different from the aces of the detector frame), and facilitates mapping directions
    and coordinate axes between R and Q space.

    Notation:
    Two unit vectors, u and v, are used.  These are definted in both real and diffraction space,
    enabling mapping between the two spaces, as
        Qu = (self.Qux, self.Quy)
        Qv = (self.Qvx, self.Qvy)
        Ru = (self.Qux, self.Ruy)
        Rv = (self.Rvx, self.Rvy)
    The Q and R unit vectors are rotated relative to one another by self.RQ_offset, as described
    in the method docstrings.  u and v define a right-handed coordinate system, such that
        np.cross(Qu,Qv) == np.corss(Ru,Rv) == +1
    """
    def __init__(self, RQ_offset, flip=False):
        """
        Initialize the RQ axes object.

        Accepts:
            RQ_offset   (float) the counterclockwise rotation of real space with respect to
                        diffraction space, i.e. the two spaces will be rotationally aligned if
                        diffraction space is rotated by this amount in the counterclockwise
                        direction.  In degrees.
            flip        (bool) indicates if there is an axes flip between real and diffraction space.
        """
        self.RQ_offset = RQ_offset
        self.flip = flip

    def set_RQ_offset(RQ_offset):
        """
        Change the RQ_offset value.

        Does not change the Qu, Qv, Ru, or Rv vectors; to do so, run set_Q_axes() or set_R_axes().

        Accepts:
            RQ_offset   (float) the counterclockwise rotation of real space with respect to
                        diffraction space, in degrees.
        """
        self.RQ_offset = RQ_offset

    def set_Q_axes(self, Qux, Quy):
        """
        Sets the orientation of the u,v internal coordinate system, which can be used for outputs
        such as strain mapping, by specifying the direction of u in Q.

        This function sets both the Q coordinate system Qu,Qv and also the R coordinate system
        Ru,Rv, using the known RQ_offset.  Note that only the direction of the (Qux,Quy) inputs is
        important; the final vectors Qu, Qv, Ru, Rv will be scaled and stored as unit vectors.

        Accepts:
            Qux         (float) the x coodinate of u in diffraction space
            Quy         (float) the y coodinate of u in diffraction space
        """
        length = np.hypot(Qux,Quy)
        self.Qux = Qux/length
        self.Quy = Quy/length
        self.Qvx = -self.Quy
        self.Qvy = self.Qux

        theta = np.radians(self.RQ_offset)
        T = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        Qvectors = np.array([[self.Qux, self.Qvx],[self.Quy, self.Qvy]])
        Rvectors = np.matmul(T,Qvectors)
        self.Rux = Rvectors[0,0]
        self.Ruy = Rvectors[1,0]
        self.Rvx = Rvectors[0,1]
        self.Rvy = Rvectors[1,1]

        if self.flip:
            self.Ruy = -self.Ruy
            self.Rvy = -self.Rvy

    def set_R_axes(self, Rux, Ruy):
        """
        Sets the orientation of the u,v internal coordinate system, which can be used for outputs
        such as strain mapping, by specifying the direction of u in R.

        This function sets both the R coordinate system Ru,Rv and also the Q coordinate system
        Qu,Qv, using the known RQ_offset.  Note that only the direction of the (Rux,Ruy) inputs is
        important; the final vectors Qu, Qv, Ru, Rv will be scaled and stored as unit vectors.

        Accepts:
            Rux         (float) the x coodinate of u in real space
            Ruy         (float) the y coodinate of u in real space
        """
        length = np.hypot(Rux,Ruy)
        self.Rux = Rux/length
        self.Ruy = Ruy/length
        self.Rvx = -self.Ruy
        self.Rvy = self.Rux

        theta = np.radians(self.RQ_offset)
        T = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        Rvectors = np.array([[self.Rux, self.Rvx],[self.Ruy, self.Rvy]])
        Qvectors = np.matmul(T,Rvectors)
        self.Qux = Qvectors[0,0]
        self.Quy = Qvectors[1,0]
        self.Qvx = Qvectors[0,1]
        self.Qvy = Qvectors[1,1]

        if self.flip:
            self.Quy = -self.Quy
            self.Qvy = -self.Qvy



