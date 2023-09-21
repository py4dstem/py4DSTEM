#### this file contains a function/s that will check if various
# libaries/compute options are available
import importlib
from operator import mod

# list of modules we expect/may expect to be installed
#  as part of a standard py4DSTEM installation
# this needs to be the import name e.g. import mp_api not mp-api
modules = [
    "crystal4D",
    "cupy",
    "dask",
    "dill",
    "distributed",
    "gdown",
    "h5py",
    "ipyparallel",
    "jax",
    "matplotlib",
    "mp_api",
    "ncempy",
    "numba",
    "numpy",
    "pymatgen",
    "skimage",
    "sklearn",
    "scipy",
    "tensorflow",
    "tensorflow-addons",
    "tqdm",
]

# currently this was copy and pasted from setup.py,
# hopefully there's a programatic way to do this.
module_depenencies = {
    "base": [
        "numpy",
        "scipy",
        "h5py",
        "ncempy",
        "matplotlib",
        "skimage",
        "sklearn",
        "tqdm",
        "dill",
        "gdown",
        "dask",
        "distributed",
    ],
    "ipyparallel": ["ipyparallel", "dill"],
    "cuda": ["cupy"],
    "acom": ["pymatgen", "mp_api"],
    "aiml": ["tensorflow", "tensorflow-addons", "crystal4D"],
    "aiml-cuda": ["tensorflow", "tensorflow-addons", "crystal4D", "cupy"],
    "numba": ["numba"],
}


#### Class and Functions to Create Coloured Strings ####
class colours:
    CEND = "\x1b[0m"
    WARNING = "\x1b[7;93m"
    SUCCESS = "\x1b[7;92m"
    FAIL = "\x1b[7;91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def create_warning(s: str) -> str:
    """
    Creates a yellow shaded with white font version of string s

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.WARNING + s + colours.CEND
    return s


def create_success(s: str) -> str:
    """
    Creates a yellow shaded with white font version of string s

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.SUCCESS + s + colours.CEND
    return s


def create_failure(s: str) -> str:
    """
    Creates a yellow shaded with white font version of string s

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.FAIL + s + colours.CEND
    return s


def create_bold(s: str) -> str:
    """
    Creates a yellow shaded with white font version of string s

    Args:
        s (str): string to be turned into a warning style string

    Returns:
        str: stylized version of string s
    """
    s = colours.BOLD + s + colours.CEND
    return s


def create_underline(s: str) -> str:
    """
    Creates an underlined version of string s

    Args:
        s (str): string to be turned into an underlined style string

    Returns:
        str: stylized version of string s
    """
    s = colours.UNDERLINE + s + colours.CEND
    return s


#### Functions to check imports etc.
### here I use the term state to define a boolean condition as to whether a libary/module was sucessfully imported/can be used


def get_import_states(modules: list = modules) -> dict:
    """
    Check the ability to import modules and store the results as a boolean value. Returns as a dict.
    e.g. import_states_dict['numpy'] == True/False

    Args:
        modules (list, optional): List of modules to check the ability to import. Defaults to modules.

    Returns:
        dict: dictionary where key is the name of the module, and val is the boolean state if it could be imported
    """
    # Create the import states dict
    import_states_dict = {}

    # check if the modules import
    # and update the states dict to reflect this
    for m in modules:
        state = import_tester(m)
        import_states_dict[m] = state

    return import_states_dict


def get_module_states(state_dict: dict) -> dict:
    """_summary_

    Args:
        state_dict (dict): _description_

    Returns:
        dict: _description_
    """

    # create an empty dict to put module states into:
    module_states = {}

    # key is the name of the module e.g. ACOM
    # val is a list of its dependencies
    # module_dependencies comes from the namespace
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

    return module_states


def print_import_states(import_states: dict) -> None:
    """_summary_

    Args:
        import_states (dict): _description_

    Returns:
        _type_: _description_
    """
    # m is the name of the import module
    # state is whether it was importable
    for m, state in import_states.items():
        # if it was importable i.e. state == True
        if state:
            s = f" Module {m.capitalize()} Imported Successfully "
            s = create_success(s)
            s = f"{s: <80}"
            print(s)
        # if unable to import i.e. state == False
        else:
            s = f" Module {m.capitalize()} Import Failed "
            s = create_failure(s)
            s = f"{s: <80}"
            print(s)
    return None


def print_module_states(module_states: dict) -> None:
    """_summary_

    Args:
        module_states (dict): _description_

    Returns:
        _type_: _description_
    """
    # Print out the state of all the modules in colour code
    # key is the name of a py4DSTEM Module
    # Val is the state i.e. True/False
    for key, val in module_states.items():
        # if the state is True i.e. all dependencies are installed
        if val:
            s = f" All Dependencies for {key.capitalize()} are Installed "
            s = create_success(s)
            print(s)
        # if something is missing
        else:
            s = f" Not All Dependencies for {key.capitalize()} are Installed"
            s = create_failure(s)
            print(s)
    return None


def perfrom_extra_checks(
    import_states: dict, verbose: bool, gratuitously_verbose: bool, **kwargs
) -> None:
    """_summary_

    Args:
        import_states (dict): _description_
        verbose (bool): _description_
        gratuitously_verbose (bool): _description_

    Returns:
        _type_: _description_
    """

    # print a output module
    extra_checks_message = "Running Extra Checks"
    extra_checks_message = create_bold(extra_checks_message)
    print(f"{extra_checks_message}")
    # For modules that import run any extra checks
    for key, val in import_states.items():
        if val:
            # s = create_underline(key.capitalize())
            # print(s)
            func = funcs_dict.get(key)
            if func is not None:
                s = create_underline(key.capitalize())
                print(s)
                func(verbose=verbose, gratuitously_verbose=gratuitously_verbose)
            else:
                # if gratuitously_verbose print out all modules without checks
                if gratuitously_verbose:
                    s = create_underline(key.capitalize())
                    print(s)
                    print_no_extra_checks(key)
                else:
                    pass

    return None


def import_tester(m: str) -> bool:
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

    return state


def check_module_functionality(state_dict: dict) -> None:
    """
    This function checks all the py4DSTEM modules, e.g. acom, ml-ai, and whether all the required dependencies are importable

    Args:
        state_dict (dict): dictionary of the state, i.e. boolean, of all the modules and the ability to import.
        It will then print in a 'Success' or 'Failure' message. All dependencies must be available to succeed.

    Returns:
        None: Prints the state of a py4DSTEM module's libaray depencencies
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

    return None  # module_states


#### ADDTIONAL CHECKS ####


def check_cupy_gpu(gratuitously_verbose: bool, **kwargs):
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
            hasattr(d, "attributes")
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

    cuda_path = cp.cuda.get_cuda_path()
    print(f"Detected CUDA Path:\t{cuda_path}")
    cupy_version = cp.__version__
    print(f"Cupy Version:\t\t{cupy_version}")

    # if verbose print extra information
    if gratuitously_verbose:
        for i in range(num_gpus_detected):
            d = cp.cuda.Device(i)
            s = f"GPU: {i}"
            s = create_warning(s)
            print(f" {s} ")
            pprint(d.attributes)
    return None


def print_no_extra_checks(m: str):
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


# dict of extra check functions
funcs_dict = {"cupy": check_cupy_gpu}


#### main function used to check the configuration of the installation
def check_config(
    # modules:list = modules, # removed to not be user editable as this will often break. Could make it append to modules... but for now just removing
    verbose: bool = False,
    gratuitously_verbose: bool = False,
    # egregiously_verbose:bool = False
) -> None:
    """
    This function checks the state of required imports to run py4DSTEM.

    Default behaviour will provide a summary of install dependencies for each module e.g. Base, ACOM etc.

    Args:
        verbose (bool, optional): Will provide the status of all possible requriements for py4DSTEM, and perform any additonal checks. Defaults to False.
        gratuitously_verbose (bool, optional): Provides more indepth analysis. Defaults to False.

    Returns:
        None
    """

    # get the states of all imports
    states_dict = get_import_states(modules)

    # get the states of all modules dependencies
    modules_dict = get_module_states(states_dict)

    # print the modules compatiabiltiy
    # prepare a message
    modules_checks_message = "Checking Module Dependencies"
    modules_checks_message = create_bold(modules_checks_message)
    print(modules_checks_message)
    # print the status
    print_module_states(modules_dict)

    if verbose:
        # Print that Import Checks are happening
        imports_check_message = "Running Import Checks"
        imports_check_message = create_bold(imports_check_message)
        print(f"{imports_check_message}")

        print_import_states(states_dict)

        perfrom_extra_checks(
            import_states=states_dict,
            verbose=verbose,
            gratuitously_verbose=gratuitously_verbose,
        )

    return None
