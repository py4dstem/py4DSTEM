#### this file contains a function/s that will check if various 
# libaries/compute options are available
import importlib
from operator import mod

modules = ["cupy",
           "numpy",
           "jax",
           "tensorflow",
           "dask",
           "pymatgen"
          ]



#### Class and Functions to Create Coloured Strings ####



class colours:
    CEND = '\x1b[0m'
    WARNING = '\x1b[7;93m'
    SUCCESS = '\x1b[7;92m'
    FAIL = '\x1b[7;91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_warning(s:str):
    s = colours.WARNING + s + colours.CEND
    return s
def create_success(s:str):
    s = colours.SUCCESS + s + colours.CEND
    return s
def create_failure(s:str):
    s = colours.FAIL + s + colours.CEND
    return s
def create_bold(s:str):
    s = colours.BOLD + s + colours.CEND
    return s
def create_underline(s:str):
    s = colours.UNDERLINE + s + colours.CEND
    return s



def import_tester(m:str):
    
    state = True
    try:
        importlib.import_module(m) 
    except:
        state = False
    
    
    if state:
        s = f" Module {m.capitalize()} Imported Successfully "
        s = create_success(s)
        s = f"{s: <80}"
        print(s)
    
    else: 
        s = f" Module {m.capitalize()} Import Failed "
        s = create_failure(s)
        s = f"{s: <80}"
        print(s)

    return state

def check_cupy_gpu(
    verbose:bool = False,
    gratuitously_verbose:bool = False,
    **kwargs
    ):
    
    from pprint import pprint
    import cupy as cp
    
    cuda_availability = cp.cuda.is_available()
    if cuda_availability:
        s = f" CUDA is Available "
        s = create_success(s)
        s = f"{s: <80}"
        print(s)
    else: 
        s = f" CUDA is Unavailable "
        s = create_failure(s)
        s = f"{s: <80}"
        print(s)
        
    for i in range(24):
        try:
            d = cp.cuda.Device(i)
            hasattr(d, 'attributes')
        except:
            num_gpus_detected = i
            break
    if num_gpus_detected == 0:
        s = " Detected no GPUs "
        s = create_failure(s)
        s = f"{s: <80}"
        print(s)
    elif num_gpus_detected >= 24:
        s = " Detected at least 24 GPUs, could be more " 
        s = create_warning(s)
        s = f"{s: <80}"
        print(s)
    else:
        s = f" Detected {num_gpus_detected} GPUs "
        s = create_success(s)
        s = f"{s: <80}"
        print(s)
    if verbose:
        cuda_path = cp.cuda.get_cuda_path()
        print(f"Detected CUDA Path:\t{cuda_path}")
        cupy_version = cp.__version__
        print(f"Cupy Version:\t\t{cupy_version}")
    if gratuitously_verbose:
        for i in range(num_gpus_detected):
            d = cp.cuda.Device(i)
            s = f"GPU: {i}" 
            s = create_warning(s)
            print(f" {s} ")
            pprint(d.attributes)

def print_no_extra_checks(m:str):
    s = f" There are no Extra Checks for {m} "
    s = create_warning(s)
    s = f" {s} "
    print(s)
    
    return None

def check_config(
    modules:list = modules,
    verbose:bool = False,
    gratuitously_verbose:bool = False
    ):
    
    
    # create an empty dict to check the import states
    states_dict = {}
    
    # dict of extra check functions
    funcs_dict = {
        "cupy"  : check_cupy_gpu
    }
    
    
    # Print that Import Checks are happening
    imports_check_message = "Running Import Checks"
    imports_check_message = create_bold(imports_check_message)
    print(f"{imports_check_message}")
    
    # check if the modules import 
    # and update the states dict to reflect this 
    for m in modules:
        x = import_tester(m)
        states_dict[m] = x
    
    extra_checks_message = "Running Extra Checks"
    extra_checks_message = create_bold(extra_checks_message)
    print(f"{extra_checks_message}")
    # For modules that import run any extra checks
    for key, val in states_dict.items():
        if val:
            s = create_underline(key.capitalize())
            print(s)
            func = funcs_dict.get(key)
            if func is not None:
                func(verbose, gratuitously_verbose)
            else:
                print_no_extra_checks(key)
    return None