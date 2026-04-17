"""Styblinski-Tang function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class StyblinskiTang(BenchmarkFunction):
    """
    Styblinski-Tang function class.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    ndim : int
        Number of dimensions.

    Notes
    -----
    Global optimum: x=(-2.903534, -2.903534, ..., -2.903534), \
    f(x)=-39.166165*n.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Styblinski-Tang',
            bounds=(full(ndim, -5.0), full(ndim, 5.0)))

        # Get the argument
        self.ndim = ndim

    def __call__(
            self,
            x):
        """
        Compute the Styblinski-Tang function value.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        float
            Function value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return nsum(x**4 - 16*x**2 + 5*x) / 2

    def gradient(
            self,
            x):
        """
        Compute the Styblinski-Tang gradient.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        ndarray
            Gradient value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return 2.0 * x**3 - 16 * x + 2.5
