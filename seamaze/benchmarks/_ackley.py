"""Ackley function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, cos, exp, full, pi, sin, sqrt, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Ackley(BenchmarkFunction):
    """
    Ackley function class.

    A highly multimodal function with a nearly flat outer region full of local
    minima and a steep, narrow central valley.

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
            name='Ackley',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the Ackley function value.

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

        # Check if the number of dimensions is zero
        if self.ndim == 0:

            # Return zero
            return 0.0

        return (
            -20.0 * exp(-0.2 * sqrt(nsum(x**2) / self.ndim))
            - exp(nsum(cos(2.0 * pi * x)) / self.ndim)
            + 20.0 + exp(1.0)
            )

    def gradient(
            self,
            x):
        """
        Compute the Ackley gradient.

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

        # Check if the input is zero-dimensional or equal to the origin
        if self.ndim == 0 or nsum(x**2) == 0.0:

            # Return zeros
            return zeros(self.ndim)

        return (
            (4.0 * exp(-0.2 * sqrt(nsum(x**2) / self.ndim))
             / (self.ndim * sqrt(nsum(x**2) / self.ndim))) * x
            +
            (2.0 * pi * exp(nsum(cos(2.0 * pi * x)) / self.ndim)
             / self.ndim) * sin(2.0 * pi * x)
            )
