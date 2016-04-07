import h5py
import time

"""
recommended HDF5 file format for ScopeFoundry
* = group
- = attr
D = data_set

* /
    - scope_foundry_version = 100
    - emd_version = 102
    * gui
        - log_quant_1
        - log_quant_1_unit
        - ...
    * hardware
        * hardware_component_1
            - ScopeFoundry_Type = Hardware
            - name = hardware_component_1
            - log_quant_1
            - log_quant_1_unit
            - ...
            * units
                - log_quant_1 = '[n_m]'
        * ...
    * measurement_1
        - ScopeFoundry_Type = Measurement
        - name = measurement_1
        - log_quant_1
        - ...
        * units
            - log_quant_1 = '[n_m]'
        * image_like_data_set_1
            - emd_group_type = 1
            D data
            D dim0
                - name = 'x'
                - unit = '[n_m]'
            D ...
            D dimN
        D simple_data_set_2
        D ...

other thoughts:
    store git revision of code
    store git revision of ScopeFoundry

"""

def h5_base_file(gui, fname):
    h5_file = h5py.File(fname)
    root = h5_file['/']
    root.attrs["ScopeFoundry_version"] = 100
    t0 = time.time()
    root.attrs['time_id'] = t0

    h5_save_gui_lq(gui, root)
    h5_save_hardware_lq(gui, root)
    return h5_file

def h5_save_gui_lq(gui, h5group):
    h5_gui_group = h5group.create_group('gui/')
    h5_gui_group.attrs['ScopeFoundry_type'] = "Gui"
    settings_group = h5_gui_group.create_group('settings')
    h5_save_lqs_to_attrs(gui.logged_quantities, settings_group)


def h5_save_hardware_lq(gui, h5group):
    h5_hardware_group = h5group.create_group('hardware/')
    h5_hardware_group.attrs['ScopeFoundry_type'] = "HardwareList"
    for hc_name, hc in gui.hardware_components.items():
        h5_hc_group = h5_hardware_group.create_group(hc_name)
        h5_hc_group.attrs['name'] = hc.name
        h5_hc_group.attrs['ScopeFoundry_type'] = "Hardware"
        h5_hc_settings_group = h5_hc_group.create_group("settings")
        h5_save_lqs_to_attrs(hc.logged_quantities, h5_hc_settings_group)
    return h5_hardware_group

def h5_save_lqs_to_attrs(logged_quantities, h5group):
    """
    Take a dictionary of logged_quantities
    and create attributes inside h5group

    :param logged_quantities:
    :param h5group:
    :return: None
    """
    unit_group = h5group.create_group('units')
    # TODO decide if we should specify h5 attr data type based on LQ dtype
    for lqname, lq in logged_quantities.items():
        h5group.attrs[lqname] = lq.val
        if lq.unit:
            unit_group.attrs[lqname] = lq.unit


def h5_create_measurement_group(measurement, h5group):
    h5_meas_group = h5group.create_group('measurement/' + measurement.name)
    h5_save_measurement_settings(measurement, h5_meas_group)
    return h5_meas_group

def h5_save_measurement_settings(measurement, h5_meas_group):
    h5_meas_group.attrs['name'] = measurement.name
    h5_meas_group.attrs['ScopeFoundry_type'] = "Measurement"
    settings_group = h5_meas_group.create_group("settings")
    h5_save_lqs_to_attrs(measurement.logged_quantities, settings_group)
    
    
def h5_create_emd_dataset(name, h5parent, shape=None, data = None, maxshape = None, 
                          dim_arrays = None, dim_names= None, dim_units = None,  **kwargs):
    """
    create an EMD dataset v0.2 inside h5parent
    returns an h5 group emd_grp
    
    to access N-dim dataset:    emd_grp['data']
    to access a specific dimension array: emd_grp['dim1']

    HDF5 Hierarchy:
    ---------------
    * h5parent
        * name [emd_grp] (<--returned)
            - emd_group_type = 1
            D data [shape = shape] 
            D dim1 [shape = shape[0]]
                - name
                - units
            ...
            D dimN [shape = shape[-1]]      

    Parameters
    ----------
    
    h5parent : parent HDF5 group 
    
    shape : Dataset shape of N dimensions.  Required if "data" isn't provided.

    data : Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.
    
    Keyword Args:
    
    dtype : Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
            
    dim_arrays : optional, a list of N dimension arrays
    
    dim_names : optional, a list of N strings naming the dataset dimensions 
    
    dim_units : optional, a list of N strings specifying units of dataset dimensions
    
    Other keyword arguments follow from h5py.File.create_dataset
    
    Returns
    -------
    emd_grp : h5 group containing dataset and dimension arrays, see hierarchy below
    
    """
    #set the emd version tag at root of h5 file
    h5parent.file['/'].attrs['version_major'] = 0
    h5parent.file['/'].attrs['version_minor'] = 2
    
    from matplotlib import pyplot
    pyplot.acorr
    
    # create the EMD data group
    emd_grp = h5parent.create_group(name)
    emd_grp.attrs['emd_group_type'] = 1
    
    if data is not None:
        shape = data.shape
    
    # data set where the N-dim data is stored
    data_dset = emd_grp.create_dataset("data", shape=shape, maxshape=maxshape, data=data, **kwargs)
    
    if dim_arrays is not None: assert len(dim_arrays) == len(shape)
    if dim_names  is not None: assert len(dim_names)  == len(shape)
    if dim_units  is not None: assert len(dim_units)  == len(shape)
    if maxshape   is not None: assert len(maxshape)   == len(shape)
    
    # Create the dimension array datasets
    for ii in range(len(shape)):
        if dim_arrays is not None:
            dim_array = dim_arrays[ii]
            dim_dtype =  dim_array.dtype            
        else:
            dim_array = None
            dim_dtype = float
        if dim_names is not None:
            dim_name = dim_names[ii]
        else:
            dim_name = "dim" + str(ii+1)
        if dim_units is not None:
            dim_unit = dim_units[ii]
        else:
            dim_unit = None
        if maxshape is not None:
            dim_maxshape = (maxshape[ii],)
        else:
            dim_maxshape = None
        
        # create dimension array dataset
        dim_dset = emd_grp.create_dataset("dim" + str(ii+1), shape=(shape[ii],), 
                                           dtype=dim_dtype, data=dim_array, 
                                           maxshape=dim_maxshape)
        dim_dset.attrs['name'] = dim_name
        if dim_unit is not None:
            dim_dset.attrs['unit'] = dim_unit
            
    return emd_grp
    
