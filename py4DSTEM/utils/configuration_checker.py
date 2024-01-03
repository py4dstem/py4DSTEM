#### this file contains a function/s that will check if various
# libaries/compute options are available
import importlib
from importlib.metadata import requires
import re
from importlib.util import find_spec

# need a mapping of pypi/conda names to import names
import_mapping_dict = {
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "scikit-optimize": "skopt",
    "mp-api": "mp_api",
}


# programatically get all possible requirements in the import name style
def get_modules_list():
    # Get the dependencies from the installed distribution
    dependencies = requires("py4DSTEM")

    # Define a regular expression pattern for splitting on '>', '>=', '='
    delimiter_pattern = re.compile(r">=|>|==|<|<=")

    # Extract only the module names without versions
    module_names = [
        delimiter_pattern.split(dependency.split(";")[0], 1)[0].strip()
        for dependency in dependencies
    ]

    # translate pypi names to import names e.g. scikit-image->skimage, mp-api->mp_api
    for index, module in enumerate(module_names):
        if module in import_mapping_dict.keys():
            module_names[index] = import_mapping_dict[module]

    return module_names


# programatically get all possible requirements in the import name style,
# split into a dict where optional import names are keys
def get_modules_dict():
    package_name = "py4DSTEM"
    # Get the dependencies from the installed distribution
    dependencies = requires(package_name)

    # set the dictionary for modules and packages to go into
    # optional dependencies will be added as they are discovered
    modules_dict = {
        "base": [],
    }
    # loop over the dependencies
    for depend in dependencies:
        # all the optional have extra in the name
        # if its not there append it to base
        if "extra" not in depend:
            # String looks like: 'numpy>=1.19'
            modules_dict["base"].append(depend)

        # if it has extra in the string
        else:
            # get the name of the optional name
            # depend looks like this 'numba>=0.49.1; extra == "numba"'
            # grab whatever is in the double quotes i.e. numba
            optional_name = re.search(r'"(.*?)"', depend).group(1)
            # if the optional name is not in the dict as a key i.e. first requirement of hte optional dependency
            if optional_name not in modules_dict:
                modules_dict[optional_name] = [depend]
            # if the optional_name is already in the dict then just append it to the list
            else:
                modules_dict[optional_name].append(depend)
    # STRIP all the versioning and semi-colons
    # Define a regular expression pattern for splitting on '>', '>=', '='
    delimiter_pattern = re.compile(r">=|>|==|<|<=")
    for key, val in modules_dict.items():
        # modules_dict[key] = [dependency.split(';')[0].split(' ')[0] for dependency in val]
        modules_dict[key] = [
            delimiter_pattern.split(dependency.split(";")[0], 1)[0].strip()
            for dependency in val
        ]

    # translate pypi names to import names e.g. scikit-image->skimage, mp-api->mp_api
    for key, val in modules_dict.items():
        for index, module in enumerate(val):
            if module in import_mapping_dict.keys():
                val[index] = import_mapping_dict[module]

    return modules_dict


# module_depenencies = get_modules_dict()
modules = get_modules_list()


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


# get the state of each modules as a dict key-val e.g. "numpy" : True
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


# Check
def get_module_states(state_dict: dict) -> dict:
    """
    given a state dict for all modules e.g. "numpy" : True,
    this parses through and checks if all modules required for a state are true

    returns dict "base": True, "ai-ml": False etc.
    """

    # get the modules_dict
    module_depenencies = get_modules_dict()
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
        module_states[key] = all(temp_lst) is True

    return module_states


def print_import_states(import_states: dict) -> None:
    """
    print with colours if the library could be imported or not
    takes dict
        "numpy" : True ->  prints success
        "pymatgen" : False -> prints failure

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
    """
    print with colours if all the imports required for module could be imported or not
    takes dict
        "base" : True ->  prints success
        "ai-ml" : Fasle -> prints failure
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


def perform_extra_checks(
    import_states: dict, verbose: bool, gratuitously_verbose: bool, **kwargs
) -> None:
    """_summary_

    Args:
        import_states (dict): dict of modules and if they could be imported or not
        verbose (bool): will show module states and all import states
        gratuitously_verbose (bool): will run extra checks - Currently only for cupy

    Returns:
        _type_: _description_
    """
    if gratuitously_verbose:
        # print a output module
        extra_checks_message = "Running Extra Checks"
        extra_checks_message = create_bold(extra_checks_message)
        print(f"{extra_checks_message}")
        # For modules that import run any extra checks
    # get all the dependencies
    dependencies = requires("py4DSTEM")
    # Extract only the module names with versions
    depends_with_requirements = [
        dependency.split(";")[0] for dependency in dependencies
    ]
    # print(depends_with_requirements)
    # need to go from
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
                    # check
                    generic_versions(
                        key, depends_with_requires=depends_with_requirements
                    )
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
    except Exception:
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
    module_depenencies = get_modules_dict()

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
        module_states[key] = all(temp_lst) is True

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


def generic_versions(module: str, depends_with_requires: list[str]) -> None:
    # module will be like numpy, skimage
    # depends_with_requires look like: numpy >= 19.0, scikit-image
    # get module_translated_name
    # mapping scikit-image : skimage
    for key, value in import_mapping_dict.items():
        # if skimage == skimage get scikit-image
        # print(f"{key = } - {value = } - {module = }")
        if module in value:
            module_depend_name = key
            break
        else:
            # if cant find mapping set the search name to the same
            module_depend_name = module
    # print(f"{module_depend_name = }")
    # find the requirement
    for depend in depends_with_requires:
        if module_depend_name in depend:
            spec_required = depend
    # print(f"{spec_required = }")
    # get the version installed
    spec_installed = find_spec(module)
    if spec_installed is None:
        s = f"{module} unable to import - {spec_required} required"
        s = create_failure(s)
        s = f"{s: <80}"
        print(s)

    else:
        try:
            version = importlib.metadata.version(module_depend_name)
        except Exception:
            version = "Couldn't test version"
        s = f"{module} imported: {version = } - {spec_required} required"
        s = create_warning(s)
        s = f"{s: <80}"
        print(s)


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
    num_gpus_detected = cp.cuda.runtime.getDeviceCount()

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
funcs_dict = {
    "cupy": check_cupy_gpu,
}


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

        perform_extra_checks(
            import_states=states_dict,
            verbose=verbose,
            gratuitously_verbose=gratuitously_verbose,
        )

    return None
