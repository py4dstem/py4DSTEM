# File parser utility

from os.path import splitext
import py4DSTEM.io.legacy as legacy
import emdfile as emd
import h5py

import emdfile as emd
import h5py
import py4DSTEM.io.legacy as legacy


def _parse_filetype(fp):
    """
    Accepts a path to a data file, and returns the file type as a string.
    """
    _, fext = splitext(fp)
    fext = fext.lower()
    if fext in [
        ".h5",
        ".hdf5",
        ".py4dstem",
        ".emd",
    ]:
        if emd._is_EMD_file(fp):
            return "emd"

        elif legacy.is_py4DSTEM_file(fp):
            return "legacy"

        elif _is_arina(fp):
            return "arina"

        elif _is_abTEM(fp):
            return "abTEM"
        else:
            raise Exception("not supported `h5` data type")

    elif fext in [
        ".dm",
        ".dm3",
        ".dm4",
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
    elif fext in [".mib", ".MIB"]:
        return "mib"
    else:
        raise Exception(f"Unrecognized file extension {fext}.")


def _is_arina(filepath):
    """
    Check if an h5 file is an Arina file.
    """
    with h5py.File(filepath, "r") as f:
        try:
            assert "entry" in f.keys()
        except AssertionError:
            return False
        try:
            assert "NX_class" in f["entry"].attrs.keys()
        except AssertionError:
            return False
    return True


def _is_abTEM(filepath):
    """
    Check if an h5 file is an abTEM file.
    """
    with h5py.File(filepath, "r") as f:
        try:
            assert "array" in f.keys()
        except AssertionError:
            return False
    return True


def _is_arina(filepath):
    """
    Check if an h5 file is an Arina file.
    """
    with h5py.File(filepath, "r") as f:
        try:
            assert "entry" in f.keys()
        except AssertionError:
            return False
        try:
            assert "NX_class" in f["entry"].attrs.keys()
        except AssertionError:
            return False
    return True


def _is_abTEM(filepath):
    """
    Check if an h5 file is an abTEM file.
    """
    with h5py.File(filepath, "r") as f:
        try:
            assert "array" in f.keys()
        except AssertionError:
            return False
    return True
