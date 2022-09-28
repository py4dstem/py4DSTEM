import warnings

class propagating_calibration(object):
    """
    A decorator which, when attached to a method of Calibration,
    causes `calibrate` to be called on any objects in the 
    Calibration object's `_targets` list, following execution of
    the decorated function.
    This allows objects associated with the Calibration to 
    automatically respond to changes in the calibration state.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        """
        Update the parameters the caller wanted by calling the wrapped 
        method, then loop through the list of targetsand call their 
        `calibrate` methods.
        """
        self.func(*args,**kwargs)

        calibration = args[0]
        try:
            for target in calibration._targets:
                if hasattr(target,'calibrate') and callable(target.calibrate):
                    target.calibrate()
                else:
                    warnings.warn(f"{target} is registered as a target for calibration propagation but does not appear to have a calibrate() method")
        except:
            pass

    def __get__(self, instance, owner):
        """
        This is some magic to make sure that the Calibration instance
        on which the decorator was called gets passed through and
        everything dispatches correctly (by making sure `instance`, 
        the Calibration instance to which the call was directed, gets
        placed in the `self` slot of the wrapped method (which is *not*
        actually bound to the instance due to this decoration.) using
        partial application of the method.)
        """
        from functools import partial
        return partial(self.__call__, instance)

