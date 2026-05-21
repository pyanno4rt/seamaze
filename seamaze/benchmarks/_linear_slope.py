"""Linear slope function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class LinearSlope(BenchmarkFunction):
    """
    Linear slope function class.

    A strictly linear, non-separable function featuring a constant
    directional gradient that forces the optimizer to slide down a smooth,
    endless slope toward the boundaries of the search space.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    weights : ndarray
        Linear weights (slopes) for each dimension.

    Notes
    -----
    Global optimum: x=(5, 5, ..., 5), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Linear Slope',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

        # Set the weights
        self.weights = full(self.ndim, -1.0)

    def __call__(
            self,
            x):
        """
        Compute the linear slope function value.

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

        return nsum(self.weights * (x - 5.0))

    def gradient(
            self,
            x):
        """
        Compute the linear slope gradient.

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
            return zeros(self.ndim)

        return self.weights
