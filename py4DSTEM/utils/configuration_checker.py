#### this file contains a function/s that will check if various 
# libaries/compute options are available
import importlib
from operator import mod

# list of modules we expect/may expect to be installed
#  as part of a standard py4DSTEM installation 
# this needs to be the import name e.g. import mp_api not mp-api
modules = [
        'PyQt5',
        'crystal4D',
        'cupy',
        'dask',
        'dill',
        'distributed',
        'gdown',
        'h5py',
        'ipyparallel',
        'ipywidgets',
        'jax',
        'matplotlib',
        'mp_api',
        'ncempy',
        'numba',
        'numpy',
        'orix',
        'pymatgen',
        'pyqtgraph',
        'qtconsole',
        'skimage',
        'sklearn',
        'scipy',
        'tensorflow',
        'tensorflow-addons',
        'tqdm'
]   

# currently this was copy and pasted from setup.py, 
# hopefully there's a programatic way to do this. 
module_depenencies = {
    'base' : [
        'numpy',
        'scipy',
        'h5py',
        'ncempy',
        'matplotlib',
        'skimage',
        'sklearn',
        'PyQt5',
        'pyqtgraph',
        'qtconsole',
        'ipywidgets',
        'tqdm',
        'dill',
        'gdown',
        'dask',
        'distributed'
        ],
    'ipyparallel': ['ipyparallel', 'dill'],
    'cuda': ['cupy'],
    'acom': ['pymatgen', 'mp_api', 'orix'],
    'aiml': ['tensorflow','tensorflow-addons','crystal4D'],
    'aiml-cuda': ['tensorflow','tensorflow-addons','crystal4D','cupy'],
    'numba': ['numba']
    }


#### Class and Functions to Create Coloured Strings ####
class colours:
    CEND = '\x1b[0m'
    WARNING = '\x1b[7;93m'
    SUCCESS = '\x1b[7;92m'
    FAIL = '\x1b[7;91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_warning(s:str)->str:
    """
    Creates a yellow shaded with white font version of string s 

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.WARNING + s + colours.CEND
    return s
def create_success(s:str)-> str:
    """
    Creates a yellow shaded with white font version of string s 

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.SUCCESS + s + colours.CEND
    return s
def create_failure(s:str)->str:
    """
    Creates a yellow shaded with white font version of string s 

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.FAIL + s + colours.CEND
    return s
def create_bold(s:str)->str:
    """
    Creates a yellow shaded with white font version of string s 

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.BOLD + s + colours.CEND
    return s
def create_underline(s:str)->str:
    """
    Creates an underlined version of string s 

    Args:
        s (str): string to be turned into an underlined style string

    Returns:
        str: stylized version of string s
    """
    s = colours.UNDERLINE + s + colours.CEND
    return s



def import_tester(m:str)->bool:
    """
    This function will try and import the module, m,
    it returns the success as boolean and prints a message. 
    Args:
        m (str): string name of a module

    Returns:
        bool: boolean if the module was able to be imported
    """
    # set a boolean switch
    state = True

    # try and import the module
    try:
        importlib.import_module(m) 
    except:
        state = False
    # if able to import
    if state:
        s = f" Module {m.capitalize()} Imported Successfully "
        s = create_success(s)
        s = f"{s: <80}"
        print(s)
    #if unable to import 
    else: 
        s = f" Module {m.capitalize()} Import Failed "
        s = create_failure(s)
        s = f"{s: <80}"
        print(s)
    # return whether it was able to be imported
    return state

#### ADDTIONAL CHECKS ####

def check_cupy_gpu(
    verbose:bool = False,
    gratuitously_verbose:bool = False,
    **kwargs
    ):
    """
    This function performs some additional tests which may be useful in 
    diagnosing Cupy GPU performance

    Args:
        verbose (bool, optional): Will print additional information e.g. CUDA path, Cupy version. Defaults to False
        gratuitously_verbose (bool, optional): Will print out atributes of all  Defaults to False.
    """
    # import some libaries
    from pprint import pprint
    import cupy as cp
    
    # check that CUDA is detected correctly
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
    
    # Count how many GPUs Cupy can detect
    # probably should change this to a while loop ... 
    for i in range(24):
        try:
            d = cp.cuda.Device(i)
            hasattr(d, 'attributes')
        except:
            num_gpus_detected = i
            break

    # print how many GPUs were detected, filter for a couple of special conditons
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

    # if verbose print extra information 
    if verbose:
        cuda_path = cp.cuda.get_cuda_path()
        print(f"Detected CUDA Path:\t{cuda_path}")
        cupy_version = cp.__version__
        print(f"Cupy Version:\t\t{cupy_version}")
    # print the atributes of the GPUs detected. 
    if gratuitously_verbose:
        for i in range(num_gpus_detected):
            d = cp.cuda.Device(i)
            s = f"GPU: {i}" 
            s = create_warning(s)
            print(f" {s} ")
            pprint(d.attributes)
    return None


def print_no_extra_checks(m:str):
    """
    This function prints a warning style message that the module m
    currently has no extra checks. 

    Args:
        m (str): This is the name of the module

    Returns:
        None
    """
    s = f" There are no Extra Checks for {m} "
    s = create_warning(s)
    s = f"{s}"
    print(s)
    
    return None


def check_module_functionality(state_dict:dict)->None:
    """


    """


    # create an empty dict to put module states into:
    module_states = {}

    # key is the name of the module e.g. ACOM
    # val is a list of its dependencies 
    for key, val in module_depenencies.items():
        
        # create a list to store the status of the depencies
        temp_lst = []

        # loop over all the dependencies required for the module to work
        # append the bool if they could be imported
        for depend in val:
            temp_lst.append(state_dict[depend])
        
        # check that all the depencies could be imported i.e. state == True
        # and set the state of the module to that 
        module_states[key] = all(temp_lst) == True

    # Print out the state of all the modules in colour code
    for key, val in module_states.items():
        # if the state is True
        if val:
            s = f" All Dependencies for {key.capitalize()} are Installed "
            s = create_success(s)
            print(s)
        # if something is missing
        else: 
            s = f" Not All Dependencies for {key.capitalize()} are Installed"
            s = create_failure(s)
            print(s)

    return None # module_states

    


#### main function used to check the configuration of the installation
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

    modules_checks_message = "Checking Module Dependencies"
    modules_checks_message = create_bold(modules_checks_message)
    print(modules_checks_message)

    check_module_functionality(state_dict=states_dict)
    return None