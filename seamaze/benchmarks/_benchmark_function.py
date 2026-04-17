"""Benchmark function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% Class definition


class BenchmarkFunction:
    """
    Base class for benchmark functions.

    Parameters
    ----------
    name : str
        Name of the function.

    bounds : tuple
        Lower and upper variable bounds.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    bounds : tuple
        See 'Parameters'.
    """

    def __init__(
            self,
            name,
            bounds):

        # Get the arguments
        self.name = name
        self.bounds = bounds

    def __call__(
            self,
            x):
        """
        Compute the function value.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        float
            Function value.
        """

        raise NotImplementedError

    def gradient(
            self,
            x):
        """
        Compute the gradient.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        ndarray
            Gradient value.
        """

        raise NotImplementedError
