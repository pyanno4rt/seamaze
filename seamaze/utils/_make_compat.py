"""Function compatibilizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from functools import wraps
from inspect import signature, Parameter

# %% Compatibilizer function


def make_compat(func):
    """
    Ensure compatibility of an objective function with specific signature.

    Parameters
    ----------
    func : Callable
        The objective function to be wrapped.

    Returns
    -------
    Callable, optional
        A wrapped version of the function that only gets supported arguments.
    """

    # Check if no function has been provided
    if func is None:

        return None

    # Get the function signature and parameter list
    sig = signature(func)
    parameters = list(sig.parameters.values())

    # Pre-calculate if the function accepts variable arguments
    has_args = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters)
    has_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters)

    @wraps(func)
    def safe_call(x, *args, **kwargs):
        """Call the wrapped function with compatible arguments."""

        # Check if no variable arguments are included
        if not has_args:

            # Get the number of variable arguments
            arg_count = sum(
                1 for parameter in parameters
                if parameter.kind in (
                        Parameter.POSITIONAL_ONLY,
                        Parameter.POSITIONAL_OR_KEYWORD)
                )

            # Get the variable arguments
            args = args[:arg_count]

        # Check if no variable keyworded arguments are included
        if not has_kwargs:

            # Get the variable keyworded arguments
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Return the function with filtered arguments
        return func(x, *args, **kwargs)

    return safe_call
