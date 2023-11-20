# Define decorators call_* which, when used to decorate class methods,
# calls all objects in a list _targets? to call some method *.

import warnings


# This is the abstract pattern:


class call_method(object):
    """
    A decorator which, when attached to a method of SomeClass,
    causes `method` to be called on any objects in the
    instance's `_targets` list, following execution of
    the decorated function.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        """
        Update the parameters the caller wanted by calling the wrapped
        method, then loop through the list of targets and call their
        `calibrate` methods.
        """
        self.func(*args, **kwargs)
        some_object = args[0]
        assert hasattr(
            some_object, "_targets"
        ), "SomeObject object appears to be in an invalid state. _targets attribute is missing."
        for target in some_object._targets:
            if hasattr(target, "method") and callable(target.method):
                try:
                    target.method()
                except Exception as err:
                    print(
                        f"Attempted to call .method(), but this raised an error: {err}"
                    )
            else:
                # warn or pass or error out here, as needs be
                # pass
                warnings.warn(
                    f"{target} is registered as a target but does not appear to have a .method() callable"
                )

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


# This is a functional decorator, @call_calibrate:

# calls: calibrate()
# targets: _targets


class call_calibrate(object):
    """
    Decorated methods cause all targets in _targets to call .calibrate().
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        """ """
        self.func(*args, **kwargs)
        calibration = args[0]
        assert hasattr(
            calibration, "_targets"
        ), "Calibration object appears to be in an invalid state. _targets attribute is missing."
        for target in calibration._targets:
            if hasattr(target, "calibrate") and callable(target.calibrate):
                try:
                    target.calibrate()
                except Exception as err:
                    print(
                        f"Attempted to calibrate object {target} but this raised an error: {err}"
                    )
            else:
                pass

    def __get__(self, instance, owner):
        """ """
        from functools import partial

        return partial(self.__call__, instance)
