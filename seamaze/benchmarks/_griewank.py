"""Griewank function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import arange, asarray, cos, full, prod, sin, sqrt, zeros
from numpy import sum as nsum

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class Griewank(BenchmarkFunction):
    """
    Griewank function class.

    A non-linear, multimodal landscape defined by a global parabolic profile
    multi-fractionally perturbed by a high-frequency cosine product term.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    indices_sqrt : ndarray
        Precomputed square root of index positions for the cosine term.

    Notes
    -----
    Global optimum: x=(0, 0, ..., 0), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Griewank',
            ndim=ndim,
            bounds=(full(ndim, -5.0), full(ndim, 5.0))
            )

        # Precompute the square root of index positions
        self.indices_sqrt = sqrt(arange(1, self.ndim + 1))

    def __call__(
            self,
            x):
        """
        Compute the Griewank function value.

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

        return nsum(x**2) / 4000.0 - prod(cos(x/self.indices_sqrt)) + 1.0

    def gradient(
            self,
            x):
        """
        Compute the Griewank gradient.

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

        # Precompute individual cosine and sine terms
        cos_terms = cos(x/self.indices_sqrt)
        sin_terms = sin(x/self.indices_sqrt)

        # Check if any cosine term is close to zero
        if (abs(cos_terms) < 1e-15).any():

            # Initialize the gradient product
            grad_prod = zeros(self.ndim)

            # Loop over the dimensions
            for j in range(self.ndim):

                # Calculate the gradient product without the j-th term
                grad_prod[j] = (
                    prod(cos_terms[arange(self.ndim) != j]) * sin_terms[j]
                    )

        else:

            # Calculate the gradient product directly
            grad_prod = (prod(cos_terms) / cos_terms) * sin_terms

        return (x / 2000.0) + (grad_prod / self.indices_sqrt)
