"""Function compatibilizer."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from functools import wraps
from inspect import signature

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

    # Get the function signature
    target = func if not hasattr(func, '__call__') else func.__call__
    sig = signature(target)

    # Pre-calculate if the function accepts variable keyword arguments
    has_kwargs = any(
        p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
        )

    @wraps(func)
    def safe_call(x, *args, **kwargs):
        """Call the wrapped function with compatible arguments."""

        # Check if the function accepts any parameters
        if has_kwargs:

            # Return the function as-is
            return func(x, *args, **kwargs)

        # Filter out the keyworded arguments from the signature
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in sig.parameters
            }

        # Return the function with filtered keyworded arguments
        return func(x, *args, **filtered_kwargs)

    return safe_call
