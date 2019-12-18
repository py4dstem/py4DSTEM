# This file defines the coordinate systems used thoughout py4DSTEM, and routines for converting
# between various coordinate systems.  
#
# This includes both
#   - transformation of the coordinates, e.g. getting the location of some point (r,theta) in
#     polar coordinates from some point (x,y) in cartesian coordinates, and
#   - transformation of data arrays, e.g. getting some (Nr,Nt)-shaped array in the (r,theta) polar
#     coordinate system from some (Nx,Ny)-shaped array in the (x,y) cartesian coordinate system
#
# In general, the coordinate systems described here should be thought of as paramatrizations of
# diffraction space, as it is the diffraction patterns which most often require this level of
# description.  In some cases, it will be valuable to define coordinate systems with distinct values
# of the parameters at each scan position - e.g. moving the origin to account for diffraction shifts.
# In these cases the parameters of the coordinate system will be (R_Nx,R_Ny)-shaped arrays.
#
# All angular quantities are in radians.
#
#
#
# ### The coordinate systems ###
#
# --Cartesian--
# All the other coordinate systems will ultimately refer back to this one in connecting coordinates
# to values in an array of data.  We'll therefore be somewhat explicit here about how our cartesian
# coordinates are defined.
# Here we mean cartesian coordinates with the origin and axes defined relative to some
# (Q_Nx,Q_Ny)-shaped data array ar as follows:
#   - the origin at ar[0,0]
#   - the x-axis along the first axis, such that ar[:,i] is a slice at y=i
#   - the y-axis along the second axis, such that ar[i,:] is a slice at x=i
# Advancing along the x-axis is thus advancing down the rows, while advancing along the y-axis is
# advancing across the columns.
# In other words, relative to array ar, the origin is placed in the upper left corner, the x-axis
# points down, and the y-axis point to the right.
#
#   Coordinates: x,y
#
#
# --Shifted Cartesian--
# This refers to a cartesian coordinate system with a shifted origin.
#
#   Coordinates: x',y'
#   Parameters: x0,y0
#   Def:
#       x' = x - x0
#       y' = y - y0
#
#
# --Polar-Elliptical--
# We'll use two different polar-elliptical coordinate parametrizations.
#
#   --Parametrization 1--
#   Coordinates: r,theta
#   Parameters: x0,y0,A,B,phi
#   Def:
#       x = x0 + A*r*cos(theta)*cos(phi) + B*r*sin(theta)*sin(phi)
#       y = y0 + B*r*sin(theta)*sin(phi) - A*r*cos(theta)*cos(phi)
#
#
#
#
#
#
#
#







