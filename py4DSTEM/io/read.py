# Reader for native files

import warnings
from os.path import exists
from pathlib import Path
from typing import Optional, Union

import emdfile as emd
import py4DSTEM.io.legacy as legacy
from py4DSTEM.data import Data
from py4DSTEM.io.parsefiletype import _parse_filetype


def read(
    filepath: Union[str, Path],
    datapath: Optional[str] = None,
    tree: Optional[Union[bool, str]] = True,
    verbose: Optional[bool] = False,
    **kwargs,
):
    """
    A file reader for native py4DSTEM / EMD files.  To read non-native
    formats, use `py4DSTEM.import_file`.

    For files written by py4DSTEM version 0.14+, the function arguments
    are those listed here - filepath, datapath, and tree.  See below for
    descriptions.

    Files written by py4DSTEM v0.14+ are EMD 1.0 files, an HDF5 based
    format.  For a description and complete file specification, see
    https://emdatasets.com/format/. For the Python implementation of
    EMD 1.0 read-write routines which py4DSTEM is build on top of, see
    https://github.com/py4dstem/emdfile.

    To read file written by older verions of py4DSTEM, different keyword
    arguments should be passed. See the docstring for
    `py4DSTEM.io.native.legacy.read_py4DSTEM_legacy` for a complete list.
    For example, `data_id` may need to be specified to select dataset.

    Args:
        filepath (str or Path): the file path
        datapath (str or None): the path within the H5 file to the data
            group to read from. If there is a single EMD data tree in the
            file, `datapath` may be left as None, and the path will
            be set to the root node of that tree.  If `datapath` is None
            and there are multiple EMD trees, this function will issue a
            warning a return a list of paths to the root nodes of all
            EMD trees it finds. Otherwise, should be a '/' delimited path
            to the data node of interest, for example passing
            'rootnode/somedata/someotherdata' will set the node called
            'someotherdata' as the point to read from. To print the tree
            of data nodes present in a file to the screen, use
            `py4DSTEM.print_h5_tree(filepath)`.
        tree (True or False or 'noroot'): indicates what data should be loaded,
            relative to the target data group specified with `datapath`.
            Enables reading the target data node only if `tree` is False,
            reading the target node as well as recursively reading the tree
            of data underneath it if `tree` is True, or recursively reading
            the tree of data underneath the target node but excluding the
            target node itself if `tree` is to 'noroot'.
    Returns:
        (the data)
    """

    # parse filetype
    er1 = f"filepath must be a string or Path, not {type(filepath)}"
    er2 = f"specified filepath '{filepath}' does not exist"
    assert isinstance(filepath, (str, Path)), er1
    assert exists(filepath), er2

    filetype = _parse_filetype(filepath)
    assert filetype in (
        "emd",
        "legacy",
    ), f"`py4DSTEM.read` loads native HDF5 formatted files, but a file of type {filetype} was detected.  Try loading it using py4DSTEM.import_file"

    # support older `root` input
    if datapath is None:
        if "root" in kwargs:
            datapath = kwargs["root"]

    # EMD 1.0 formatted files (py4DSTEM v0.14+)
    if filetype == "emd":
        # check version
        version = emd._get_EMD_version(filepath)
        if verbose:
            print(
                f"EMD version {version[0]}.{version[1]}.{version[2]} detected. Reading..."
            )
        assert emd._version_is_geq(
            version, (1, 0, 0)
        ), f"EMD version {version} detected. Expected version >= 1.0.0"

        # read
        data = emd.read(filepath, emdpath=datapath, tree=tree)
        if verbose:
            print("Data was read from file. Adding calibration links...")

        # add calibration links
        if isinstance(data, Data):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal = data.calibration
        elif isinstance(data, emd.Root):
            try:
                cal = data.metadata["calibration"]
            except KeyError:
                cal = None
        else:
            cal = None
        if cal is not None:
            try:
                root_treepath = cal["_root_treepath"]
                target_paths = cal["_target_paths"]
                del cal._params["_target_paths"]
                for p in target_paths:
                    try:
                        p = p.replace(root_treepath, "")
                        d = data.root.tree(p)
                        cal.register_target(d)
                        if hasattr(d, "setcal"):
                            d.setcal()
                    except AssertionError:
                        pass
            except KeyError:
                pass
            cal.calibrate()

        # return
        if verbose:
            print("Done.")
        return data

    # legacy py4DSTEM files (v <= 0.13)
    else:
        assert (
            filetype == "legacy"
        ), "path points to an H5 file which is neither an EMD 1.0+ file, nor a recognized legacy py4DSTEM file."

        # read v13
        if legacy.is_py4DSTEM_version13(filepath):
            # load the data
            if verbose:
                print("Legacy py4DSTEM version 13 file detected. Reading...")
            kwargs["root"] = datapath
            kwargs["tree"] = tree
            data = legacy.read_legacy13(
                filepath=filepath,
                **kwargs,
            )
            if verbose:
                print("Done.")
            return data

        # read <= v12
        else:
            # parse the root/data_id from the datapath arg
            if datapath is not None:
                datapath = datapath.split("/")
                try:
                    datapath.remove("")
                except ValueError:
                    pass
                rootgroup = datapath[0]
                if len(datapath) > 1:
                    datapath = "/".join(rootgroup[1:])
                else:
                    datapath = None
            else:
                rootgroups = legacy.get_py4DSTEM_topgroups(filepath)
                if len(rootgroups) > 1:
                    print(
                        "multiple root groups in a legacy file found - returning list of root names; please pass one as `datapath`"
                    )
                    return rootgroups
                elif len(rootgroups) == 0:
                    raise Exception("No rootgroups found")
                else:
                    rootgroup = rootgroups[0]
                    datapath = None

            # load the data
            if verbose:
                print("Legacy py4DSTEM version <= 12 file detected. Reading...")
            kwargs["topgroup"] = rootgroup
            if datapath is not None:
                kwargs["data_id"] = datapath
            data = legacy.read_legacy12(
                filepath=filepath,
                **kwargs,
            )
            if verbose:
                print("Done.")
            return data
