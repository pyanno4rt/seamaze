"""Sum-of-different-powers function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import arange, asarray, full, sign
from numpy import abs as nabs
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class SumOfDiffPowers(BenchmarkFunction):
    """
    Sum-of-different-powers function class.

    A continuous, unimodal hyper-surface characterized by variable scaling and
    structural asymmetry, where polynomial exponents increase progressively
    across the parameter dimensions.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    exponents : ndarray
        Linearly increasing exponents (from 2.0 to 6.0).

    Notes
    -----
    Global optimum: x=(0, 0, ..., 0), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Sum of Different Powers',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

        # Check if the number of dimension is smaller or equal to one
        if self.ndim <= 1:

            # Set the exponent to 2.0
            self.exponents = full(self.ndim, 2.0)

        else:

            # Precompute the exponent series
            self.exponents = 2.0 + 4.0 * arange(self.ndim) / (self.ndim - 1.0)

    def __call__(
            self,
            x):
        """
        Compute the sum-of-different-powers function value.

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

        return nsum(nabs(x) ** self.exponents)

    def gradient(
            self,
            x):
        """
        Compute the sum-of-different-powers gradient.

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

        return self.exponents * sign(x) * (nabs(x) ** (self.exponents - 1.0))
