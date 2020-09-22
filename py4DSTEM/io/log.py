# Creates a log singleton class, Logger, which will contain all program logs.
#
# All instances of the Logger class point to a single instance, which keeps a running inventory
# of function calls which are flagged for logging. 
# Each such function call is stored in a single LogItem object, which is stored at a particular
# index in the Logger.
# A function decorator, @log, is defined which causes decorated functions to get logged whenever
# called.

from collections import OrderedDict
from time import localtime
from functools import wraps
import inspect

# Get the current version in __version__
from ..version import __version__

class Logger(object):
    """
    The Logger class is a singleton, ensuring that only one logger exists.
    """
    instance = None

    class __Logger(object):
        def __init__(self):
            self.log_index = 0
            self.logged_items = OrderedDict()

        def add_item(self, function, inputs, version, datetime):
            log_item = LogItem(function=function,
                               inputs=inputs,
                               version=version,
                               datetime=datetime)
            self.logged_items[self.log_index] = log_item
            self.log_index += 1

        def show_item(self, index):
            log_item = self.logged_items[index]
            print("*** Log index {}, at time {}{}{}_{}:{}:{} ***".format(index,
                                                                  log_item.datetime.tm_year,
                                                                  log_item.datetime.tm_mon,
                                                                  log_item.datetime.tm_mday,
                                                                  log_item.datetime.tm_hour,
                                                                  log_item.datetime.tm_min,
                                                                  log_item.datetime.tm_sec))
            print("Function: \t{}".format(log_item.function))
            print("Inputs:")
            for key, value in log_item.inputs.items():
                if type(value).__name__=='DataCube':
                    print("\t\t{}\t{}".format(key,"DataCube_id"+str(id(value))))
                else:
                    print("\t\t{}\t{}".format(key,value))
            print("Version: \t{}\n".format(log_item.version))

        def show_log(self):
            for i in range(self.log_index):
                self.show_item(i)

    def __new__(cls):
        if not Logger.instance:
            Logger.instance = Logger.__Logger()
        return Logger.instance

    def __getattr__(self,name):
        return getattr(self.instance,name)

    def __setattr__(self,name):
        return setattr(self.instance,name)

class LogItem(object):

    def __init__(self, function, inputs, version, datetime):
        self.function = function
        self.inputs = inputs
        self.version = version
        self.datetime = datetime

logger = Logger()

def log(function):

    global logger

    # Get the parameters and default arguments
    signature = inspect.signature(function)
    inputs = OrderedDict()
    for key,value in signature.parameters.items():
        if value.default is inspect._empty:
            inputs[key] = None
        else:
            inputs[key] = value.default

    # Define the new function
    @wraps(function)
    def logged_function(*args,**kwargs):
        for i in range(len(args)):
            key = list(inputs.items())[i][0]
            inputs[key] = args[i]
        for key,value in kwargs.items():
            inputs[key] = value

        # Perform logging
        logger.add_item(function = function.__name__,
                                 inputs = inputs,
                                 version = __version__,
                                 datetime = localtime())

        return function(*args,**kwargs)

    return logged_function


