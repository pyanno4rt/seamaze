"""Ellipsoid function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import arange, asarray, full
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Ellipsoid(BenchmarkFunction):
    """
    Ellipsoid function class.

    A strictly convex and unimodal quadratic hyper-surface characterized by
    severe ill-conditioning, where the scaling coefficients increase
    exponentially across the parameter dimensions.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    coefficients : ndarray
        Exponentially growing scaling factors for each dimension.

    Notes
    -----
    Global optimum: x=(0, 0, ..., 0), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Ellipsoid',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

        # Check if the number of dimensions is smaller or equal to one
        if self.ndim <= 1:

            # Set the coefficient to 1.0
            self.coefficients = full(self.ndim, 1.0)

        else:

            # Precompute the exponential scaling coefficients
            self.coefficients = (
                10.0**(6.0 * arange(self.ndim) / (self.ndim - 1.0))
                )

    def __call__(
            self,
            x):
        """
        Compute the ellipsoid function value.

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

        return nsum(self.coefficients * (x**2))

    def gradient(
            self,
            x):
        """
        Compute the ellipsoid gradient.

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

        return 2.0 * self.coefficients * x
