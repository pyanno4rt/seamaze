"""Discus function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Discus(BenchmarkFunction):
    """
    Discus function class.

    A poorly conditioned, unimodal function characterized by a flat, disc-like
    shape. The first dimension is heavily penalized by 1e6, while all other
    dimensions are smoothly scaled.

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
            name='Discus',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the discus function value.

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

        # Check if the number of dimensions is smaller or equal to one
        if self.ndim <= 1:

            # Return the scaled sphere function value
            return 1e6 * nsum(x**2)

        return 1e6 * (x[0]**2) + nsum(x[1:]**2)

    def gradient(
            self,
            x):
        """
        Compute the discus gradient.

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

        # Check if the number of dimensions is zero
        if self.ndim == 0:

            # Return zeros
            return zeros(0)

        # Check if the number of dimensions is one
        if self.ndim == 1:

            # Return the scaled sphere gradient
            return 2e6 * x

        # Compute the uniform gradient
        grad = 2.0 * x

        # Adapt the first element
        grad[0] = 2e6 * x[0]

        return grad
