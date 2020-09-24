# `io.native`: native file format reading and writing

This module includes functions to perform the following operations on the native
fileformat:
- read
- write
- append data
- remove data
- copy
- repack (remove unlinked data)



# The py4DSTEM file format

py4DSTEM files are HDF5 files with a specified group/attribute/data structure.

Last updated for py4DSTEM v0.10.1

############################## File structure ###############################

/
|--grp: 4DSTEM_experiment (*)
        |--attr: emd_group_type=2
        |--attr: version_major=0
        |--attr: version_minor=10
        |--attr: version_release=1
        |
        |--grp: data
        |         |--grp: datacubes (**)
        |         |   |--grp: datacube_1
        |         |   |--grp: datacube_2
        |         |   :    :
        |         |
        |         |--grp: counted_datacubes
        |         |   |--grp: counteddatacube_1
        |         |   :    :
        |         |   
        |         |--grp: diffractionslices
        |         |   |--grp: diffractionslice_1
        |         |   :    :
        |         |
        |         |--grp: realslices
        |         |   |--grp: realslice_1
        |         |   :    :
        |         |
        |         |--grp: pointlist
        |         |   |--grp: pointlist_1
        |         |   :    :
        |         |
        |         |--grp: pointlistarrays
        |             |--grp: pointlistarray_1
        |             :    :
        |
        |--grp: metadata (***)
                  |--grp: microscope
                  |   |--# Acquisition parameters
                  |   |--# Accelerating voltage, camera length, convergence angle, 
                  |   |--# C2 aperture, spot size, exposure time, scan rotation angle,
                  |   |--# scan shape, probe FWHM
                  |
                  |--grp: sample
                  |   |--# Material, preparation
                  |
                  |--grp: user
                  |   |--# Name, instituion, dept, contact email
                  |
                  |--grp: calibration
                  |   |--# R pixel size, Q pixel size, R/Q rotation offset
                  |   |--# In case of duplicates here and in grp: microscope (e.g. pixel
                  |   |--# sizes), quantities here are calculated from data rather than
                  |   |--# being read from the instrument
                  |
                  |--grp: comments

#############################################################################

(*)  The specification here is for a certain heirarchy nested under an HDF5 group.
     In principle that group might live anywhere in some .h5 file; in practice,
     the program's read/write functions assume this is a group sitting right under
     the file root.  The default name for this toplevel group is '4DSTEM_experiment',
     but any name will do (as long as it has the right tags!). This enables the
     possibility of saving arbitrarily many py4DSTEM 'files' inside a single .h5.
(**) The data blocks stored in the datacubes, diffractionslices, and realslices
     are all EMD type 1 files. Below is the EMD type 1 specification, followed
     by the specifications for the data blocks in each of the py4DSTEM data
     subgroups.

### EMD type 1 ###

/
|--grp: some_emd1_data_group
        |--attr: emd_group_type=1
        |--dataset: data
        |--dataset: dim1
        |         |--attr: name
        |         |--attr: units
        |
        |--dataset: dim2
        |         |--attr: name
        |         |--attr: units
        :         : 

For an N dimensional dataset, there are N dim datasets.  The data associated with
each dim represents the calibration for a corresponding dimension of the primary
dataset. The dim datasets are 1D, and can have a length equal to the corresponding
data axis, or may have a length of 2, from which the remaining values can be
extrapolated linearly.

### datacubes ###

TKTKTKTK


        |         |   |--grp: datacube_1 (**)
        |         |   |   |--attr: emd_group_type=1
        |         |   |   |--data: data
        |         |   |   |--data: dim1 (***)
        |         |   |   |    |--attr: name="R_x"
        |         |   |   |    |--attr: units="[n_m]"
        |         |   |   |--data: dim2
        |         |   |   |    |--attr: name="R_y"
        |         |   |   |    |--attr: units="[n_m]"
        |         |   |   |--data: dim3
        |         |   |   |    |--attr: name="Q_x"
        |         |   |   |    |--attr: units="[n_m^-1]"
        |         |   |   |--data: dim4
        |         |   |        |--attr: name="Q_y"
        |         |   |        |--attr: units="[n_m^-1]"



        |         |   |-- grp: counted_data_1 (**)
        |         |   |   |--attr: emd_group_type=1
        |         |   |   |--data: data (stored as v0.7 PointListArray)
        |         |   |   |--data: dim1 (***)
        |         |   |   |    |--attr: name="R_x"
        |         |   |   |    |--attr: units="[n_m]"
        |         |   |   |--data: dim2
        |         |   |   |    |--attr: name="R_y"
        |         |   |   |    |--attr: units="[n_m]"
        |         |   |   |--data: dim3
        |         |   |   |    |--attr: name="Q_x"
        |         |   |   |    |--attr: units="[n_m^-1]"
        |         |   |   |--data: dim4
        |         |   |        |--attr: name="Q_y"
        |         |   |        |--attr: units="[n_m^-1]"
        |         |   |   |--data: index_coords (string array)
        |         |   |             (used only if data is numpy structured |         |   |              array, to indicate fields for data)



        |         |   |--grp: diffractionslice_1 (**)
        |         |   |    |--attr: emd_group_type=1
        |         |   |    |--data: data
        |         |   |    |--data: dim1,dim2 (***)
        |         |   |    |--data: dim3 (****)
        |         |   |


        |         |   |--grp: realslice_1
        |         |   |    |--attr: emd_group_type=1
        |         |   |    |--data: data
        |         |   |    |--data: dim1,dim2 (***)
        |         |   |    |--data: dim3 (****)
        |         |   |


        |         |   |--grp: pointlist_1 (**)
        |         |   |    |--attr: coordinates='coord_1, coord_2, ...'
        |         |   |    |--attr: dimensions=val
        |         |   |    |--attr: length=val
        |         |   |    |--grp: coord_1
        |         |   |    |    |--attr: dtype
        |         |   |    |    |--data
        |         |   |    |--grp: coord_2
        |         |   |    :    :
        |         |   |

        |             |--grp: pointlistarray_1
        |             |    |--attr: coordinates='Qx, Qy, Rx, Ry, Int, ...'
        |             |    |--attr: dimensions=val
        |             |    |--data (array of numpy arrays)
        |             |    :
        |             |




################################### Notes ###################################

(*) the top level h5 group name defaults to '4DSTEM_experiment' for files written from py4DSTEM,
    and to '4DSTEM_simulation' for files written from Prismatic simulations.  However, any top level
    group name is acceptable -- the files are recognized as py4DSTEM structured by the three
    attributes in this toplevel group version_major, version_minor, and emd_group_type=2.

(**) Each data group -- datacubes, diffractionslices, realslices, pointlists, pointlistarrays --
    contains any number of subgroups, each corresponding to a single dataobject of that type. These
    can have any name (and it is *strongly* encouraged to name objects carefully and clearly), but
    if left unnamed at write time, they default to datacube_0, diffractionslice_5, etc.

(***) The indivudual dataobjects corresponding to datacube, realslice, and diffractionslice objects
    are emd_group_type=1 objects, and conform to the standards for such objects, described in detail
    at https://emdatasets.com/format/.  In particular, these h5 groups contain an h5 dataset called
    data with the data itself, and N h5 datasets called dim1, dim2 ... dimN for N dimensional data,
    which specify the meaning of the N axes of the data.  Each dim has the attributes 'name' and
    'units' which name each axis and specify its units.

(****) The diffractionslice and realslice groups can be 2 or 3 dimensional.  When 3 dimensional, the
    third axis may correspond to a continuous variable (e.g. a series of images from a depth
    sectioning experiment, a series of diffraction patterns acquired in time as a sample is heated,
    etc.) or it may correspond to a set of discrete but thematically related slices with named
    labels (e.g. a strain map consisting of 4 'image' shaped objects giving the in-plane strain and
    rotation components 'e_xx', 'e_yy', 'e_xy', 'theta').  These two cases are distinguished only
    by the typing of dim3: in the former case, dim3 is populated by numbers, and in the latter, by
    strings (i.e. the slice names).

(*****) Pointlists contain N points (attr: length), with M values for each point (attr: dimensions).
    The M values each correspond to some labelled coordinate (attr: coordinates).  For instance,
    if 10 Bragg peaks are identified in some diffraction pattern, they might be stored in a
    pointlist with the three coordinates 'qx', 'qy', and 'intensity' corresponding to the (x,y)
    position and intensity of each Bragg peak. The data itself is then stored in M h5 groups, with
    names cooresponding to the coordinates ('qx',...), and each containing an h5 dataset of length N
    and an attribute speficifying the data type.

(******) Pointlistarrays are 2D arrays of structured data arrays with
    H5 special variable length (vlen) datatype. All PointLists in the
    array MUST have the same coordinates. For counted datacubes there
    is an additional dataset that stores a list of strings corresponding
    to the structured array fields that store the electron counts (typically
    "ind" for 1D and "qx"/"qy" for 2D indexing). If the data type of the array is a base dtype (i.e. not a "structured" array with named fields) then this is ignored and the reader assumes the indexing is 1D (this is the Kitware format–numpy structured arrays seem to be specific to h5py). 



