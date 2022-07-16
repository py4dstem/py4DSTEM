from os.path import splitext

def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    _, fext = splitext(fp)
    fext = fext.lower()
    if fext in [
        ".h5",
        ".H5",
        ".hdf5",
        ".HDF5",
        ".py4dstem",
        ".py4DSTEM",
        ".PY4DSTEM",
        ".emd",
        ".EMD",
    ]:
        return "py4DSTEM"
    elif fext in [
        ".dm",
        ".dm3",
        ".dm4",
        ".DM",
        ".DM3",
        ".DM4"
    ]:
        return "dm"
    elif fext in [".raw"]:
       return "empad"
    elif fext in [".mrc"]:
       return "mrc_relativity"
    elif fext in [".gtg", ".bin"]:
       return "gatan_K2_bin"
    elif fext in [".kitware_counted"]:
       return "kitware_counted"
    else:
        raise Exception(f"Unrecognized file extension {fext}.")
