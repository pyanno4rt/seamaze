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

    A smooth, non-separable landscape featuring an asymmetric multi-well
    structure where the global minimum is surrounded by steep walls and
    competing local basins.

    Parameters
    ----------
    ndim : int, default=2
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
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

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

        # Ensure that the input is an array
        x = asarray(x)

        return nsum(x**4 - 16.0 * (x**2) + 5.0 * x) / 2

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

        # Ensure that the input is an array
        x = asarray(x)

        return 2.0 * (x**3) - 16.0 * x + 2.5
