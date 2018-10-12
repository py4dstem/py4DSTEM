# Creates a function decorator which causes a function to be logged every time it's used

from collections import OrderedDict
import inspect

def log(function):

    # Get the parameters and default arguments
    signature = inspect.signature(function)
    params = OrderedDict()
    for key,value in signature.parameters.items():
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
        print("Log entry:")
        print("Function called: "+function.__name__)
        print("Parameters and values:")
        for key, value in params.items():
            print("{} : {}".format(key,value))

        return function(*args,**kwargs)

    return logged_function

