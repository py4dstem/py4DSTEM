from emdfile import save as _save
import warnings


def save(filepath, data, mode="w", emdpath=None, tree=True):
    """
    Saves data to an EMD 1.0 formatted HDF5 file at filepath.

    For the full docstring, see py4DSTEM.emdfile.save.
    """
    # This function wraps emdfile's save and adds a small piece
    # of metadata to the calibration to allow linking to calibrated
    # data items on read

    cal = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(data, "calibration") and data.calibration is not None:
            cal = data.calibration
            rp = "/".join(data._treepath.split("/")[:-1])
            cal["_root_treepath"] = rp

    _save(filepath, data=data, mode=mode, emdpath=emdpath, tree=tree)

    if cal is not None:
        del cal._params["_root_treepath"]
