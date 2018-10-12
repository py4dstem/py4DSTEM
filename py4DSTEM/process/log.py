# Creates a function decorator which causes a function to be logged every time it's used

from collections import OrderedDict
from inspect import signature

def log(function):

    # Get the parameters and default arguments
    sig = inspect.signature(function)
    params = OrderedDict()
    for key,value in sig.parameters.items():
        if value.default is inspect._empty:
            params[key] = None
        else:
            params[key] = value.default

    # Define the new function
    def logged_function(*args,**kwargs):
        for i in range(len(args)):
            key = list(params.items())[i][0]
            params[key] = args[i]
        for key,value in kwargs.items():
            params[key] = value

        # Perform logging
        print(params)

        return function(*args,**kwargs)

    return logged_function

