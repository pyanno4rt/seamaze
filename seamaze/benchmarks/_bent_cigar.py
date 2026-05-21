"""Bent-Cigar function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import asarray, full, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class BentCigar(BenchmarkFunction):
    """
    Bent-Cigar function class.

    A poorly conditioned, unimodal benchmark function characterized by an
    extremely narrow, elongated cigar-like shape. Only the first dimension is
    smoothly scaled, while all other dimensions are heavily penalized by 1e6.

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
            name='Bent-Cigar',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

    def __call__(
            self,
            x):
        """
        Compute the Bent-Cigar function value.

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

            # Return the sphere function value
            return nsum(x**2)

        return x[0]**2 + 1e6 * nsum(x[1:]**2)

    def gradient(
            self,
            x):
        """
        Compute the Bent-Cigar gradient.

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

            # Return the sphere gradient
            return 2.0 * x

        # Compute the uniform gradient
        grad = 2e6 * x

        # Adapt the first element
        grad[0] = 2.0 * x[0]

        return grad
