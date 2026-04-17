"""Rastrigin function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from math import pi
from numpy import asarray, cos, full, sin
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function class.

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
            name='Rastrigin',
            bounds=(full(ndim, -5.12), full(ndim, 5.12)))

        # Get the argument
        self.ndim = ndim

    def __call__(
            self,
            x):
        """
        Compute the Rastrigin function value.

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

        return 10 * self.ndim + nsum(x**2 - 10 * cos(2*pi*x))

    def gradient(
            self,
            x):
        """
        Compute the Rastrigin gradient.

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

        return 2 * x + 20 * pi * sin(2*pi*x)
