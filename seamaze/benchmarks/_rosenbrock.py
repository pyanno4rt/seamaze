"""Rosenbrock function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function class.

    A non-linear, non-convex landscape characterized by a narrow, parabolic
    valley where the gradient vectors point almost orthogonal to the ridge
    line.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Notes
    -----
    Global optimum: x=(1, 1, ..., 1), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Rosenbrock',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the Rosenbrock function value.

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

            # Return zero
            return 0.0

        return nsum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    def gradient(
            self,
            x):
        """
        Compute the Rosenbrock gradient.

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

        # Initialize the gradient
        grad = zeros(self.ndim)

        # Compute the derivative from x_0 to x_{n-2}
        grad[:-1] = 400.0 * x[:-1] * (x[:-1]**2 - x[1:]) + 2.0 * (x[:-1] - 1.0)

        # Add the derivative from x_1 to x_{n-1}
        grad[1:] += 200.0 * (x[1:] - x[:-1]**2)

        return grad
