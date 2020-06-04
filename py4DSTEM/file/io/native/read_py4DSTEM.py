# Reads a py4DSTEM formatted (EMD type 2) 4D-STEM dataset

def read_py4DSTEM(fp, mem="RAM", binfactor=1, **kwargs):
    """
    Read a py4DSTEM formatted (EMD type 2) 4D-STEM file.

    Accepts:
        fp          str or Path Path to the file
        mem         str         (opt) Specifies how the data should be stored; must be "RAM" or "MEMMAP". See
                                docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor   int         (opt) Bin the data, in diffraction space, as it's loaded. See docstring for
                                py4DSTEM.file.io.read.  Default is 1.
        **kwargs                (opt) When reading the native h5 file format, additional keyword arguments are
                                used to indicate loading behavior in the case where the source file contains
                                multiple data objects.

                                Recognized keywords are:

                                    TKTKkwarg1       int         descrption TKTKTK

    Returns:
        data        DataCube    The 4D-STEM data.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"

    # TK TODO load the data

    # TK TODO load the metadata

    return data, md


