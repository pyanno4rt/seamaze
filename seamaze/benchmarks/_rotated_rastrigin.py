"""Rotated Rastrigin function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from math import pi
from numpy import array, asarray, cos, eye, full, sin, zeros
from numpy import sum as nsum
from scipy.linalg import block_diag

# %% Internal package import

from seamaze.benchmarks import BenchmarkFunction

# %% Class definition


class RotatedRastrigin(BenchmarkFunction):
    """
    Rotated Rastrigin function class.

    A highly multimodal function where an orthogonal rotation completely skews
    the local cosine ripples, eliminating all coordinate-wise separability.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.

    Attributes
    ----------
    rotation_matrix : ndarray
        Orthogonal matrix used to rotate the coordinate system.

    Notes
    -----
    Global optimum: x=(0, 0, ..., 0), f(x)=0.
    """

    def __init__(
            self,
            ndim=2):

        # Initialize the superclass
        super().__init__(
            name='Rotated Rastrigin',
            ndim=ndim,
            bounds=(full(ndim, -5.12), full(ndim, 5.12))
            )

        # Check if the number of dimensions is smaller or equal to one
        if self.ndim <= 1:

            # Set the rotation matrix to the unit matrix
            self.rotation_matrix = eye(self.ndim)

            return

        # Set the angle
        angle = pi / 6.0
        cosine, sine = cos(angle), sin(angle)

        # Create the default rotation block
        g_block = array([[cosine, -sine], [sine, cosine]])

        # Create a list of blocks
        blocks = [g_block] * (self.ndim // 2)

        # Check if the number of dimensions is odd
        if self.ndim % 2 != 0:

            # Append a neutral element
            blocks.append(array([[1.0]]))

        # Create the full orthogonal rotation matrix
        self.rotation_matrix = block_diag(*blocks)

    def __call__(
            self,
            x):
        """
        Compute the rotated Rastrigin function value.

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

        # Rotate the evaluation point
        x = x @ self.rotation_matrix.T

        return 10.0 * self.ndim + nsum(x**2 - 10.0 * cos(2.0*pi*x))

    def gradient(
            self,
            x):
        """
        Compute the rotated Rastrigin gradient.

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

        # Rotate the coordinates
        x = x @ self.rotation_matrix.T

        # Get the rotated gradient
        grad = 2.0 * x + 20.0 * pi * sin(2.0*pi*x)

        return grad @ self.rotation_matrix
