"""Schwefel function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, cos, full, sign, sin, sqrt, where, zeros
from numpy import abs as nabs
from numpy import any as nany
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Schwefel(BenchmarkFunction):
    """
    Schwefel (2.26) function class.

    A deceptive, multimodal landscape where the global optimum is far from
    local traps, forcing algorithms to cross massive barriers to avoid
    premature convergence.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Notes
    -----
    Global optimum: x=(420.9687, 420.9687, ..., 420.9687), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Schwefel',
            ndim=ndim,
            bounds=(full(ndim, -500.0), full(ndim, 500.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the Schwefel function value.

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

        # Check if the evaluation point lies outside
        if nany(x < -500.0) or nany(x > 500.0):

            # Return infinity
            return 1e6

        return 418.9829 * self.ndim - nsum(x * sin(sqrt(nabs(x))))

    def gradient(
            self,
            x):
        """
        Compute the Schwefel gradient.

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

        # Check if the number of dimensions is zero or the point lies outside
        if self.ndim == 0 or (nany(x < -500.0) or nany(x > 500.0)):

            # Return zeros
            return zeros(self.ndim)

        # Precompute the square root of the absolute values
        sqrt_abs_x = sqrt(nabs(x))

        # Calculate the gradient
        grad = -sin(sqrt_abs_x) - 0.5 * sign(x) * sqrt_abs_x * cos(sqrt_abs_x)

        return where(x == 0.0, 0.0, grad)
