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

    A highly multimodal, separable landscape defined by a frequent cosine
    modulation superposed on a parabolic baseline.

    Parameters
    ----------
    ndim : int, default=2
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
            ndim=ndim,
            bounds=(full(ndim, -5.12), full(ndim, 5.12))
            )

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

        # Ensure that the input is an array
        x = asarray(x)

        return 10.0 * self.ndim + nsum(x**2 - 10.0 * cos(2.0 * pi * x))

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

        # Ensure that the input is an array
        x = asarray(x)

        return 2.0 * x + 20.0 * pi * sin(2.0*pi*x)
