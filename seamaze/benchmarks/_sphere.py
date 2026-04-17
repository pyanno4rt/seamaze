"""Sphere function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Sphere(BenchmarkFunction):
    """
    Sphere function class.

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
    Global optimum: x=(0, 0, ..., 0), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Sphere',
            bounds=(full(ndim, -5), full(ndim, 5)))

        # Get the argument
        self.ndim = ndim

    def __call__(
            self,
            x):
        """
        Compute the sphere function value.

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

        return nsum(x**2)

    def gradient(
            self,
            x):
        """
        Compute the sphere gradient.

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

        return 2.0 * x
