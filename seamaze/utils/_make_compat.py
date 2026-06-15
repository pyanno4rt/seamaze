"""Function signature compatibilizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from functools import wraps
from inspect import signature, Parameter

# %% Function definition


def make_compat(func):
    """
    Ensure signature compatibility of a callable by filtering excess arguments.

    Parameters
    ----------
    func : Callable or None
        The function to be adapted for signature compatibility.

    Returns
    -------
    Callable or None
        The wrapped robust function, or None if the input was None.
    """

    # Check if None has been passed
    if func is None:

        return None

    # Get the signature of the function
    sig = signature(func)
    parameters = list(sig.parameters.values())

    # Check for variable (key-worded) arguments (*args and **kwargs)
    has_args = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters)
    has_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters)

    # Determine the allowed number of additional positional arguments
    num_pos = sum(
        1 for parameter in parameters if parameter.kind in (
            Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        )
    max_extra_args = max(0, num_pos - 1)

    @wraps(func)
    def safe_call(x, *args, **kwargs):
        """Call the wrapped function with compliant filtered arguments."""

        # Check if the function does not support *args
        if not has_args:

            # Slice the arguments
            args = args[:max_extra_args]

        # Check if the function does not support *kwargs
        if not has_kwargs:

            # Filter unknown keyword arguments
            kwargs = {
                key: value for key, value in kwargs.items()
                if key in sig.parameters}

        return func(x, *args, **kwargs)

    return safe_call
