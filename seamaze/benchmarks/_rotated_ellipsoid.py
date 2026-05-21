"""Rotated ellipsoid function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, cumsum, full
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class RotatedEllipsoid(BenchmarkFunction):
    """
    Rotated ellipsoid function class.

    An ill-conditioned function where a coordinate rotation matrix misaligns
    the exponentially scaled parabolic axes, preventing coordinate-wise
    optimization.

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
            name='Rotated Ellipsoid',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the rotated ellipsoid function value.

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

        return nsum(cumsum(x)**2)

    def gradient(
            self,
            x):
        """
        Compute the rotated ellipsoid gradient.

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

        # Compute the cumulative sum
        cum_sums = cumsum(x)

        return 2.0 * cumsum(cum_sums[::-1])[::-1]
