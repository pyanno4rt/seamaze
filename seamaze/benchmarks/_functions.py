"""Benchmark functions."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from math import inf, pi
from numpy import asarray, cos, full, sin
from numpy import sum as nsum

# %% Benchmark functions


class BenchmarkFunction:
    """
    Base class for benchmark functions.

    Parameters
    ----------
    name : str
        Name of the function.

    bounds : tuple
        Lower and upper variable bounds.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    bounds : tuple
        See 'Parameters'.
    """

    def __init__(
            self,
            name,
            bounds):

        # Get the arguments
        self.name = name
        self.bounds = bounds

    def __call__(
            self,
            x):
        """
        Compute the function value.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        float
            Function value.
        """

        raise NotImplementedError

    def gradient(
            self,
            x):
        """
        Compute the gradient.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        ndarray
            Gradient value.
        """

        raise NotImplementedError


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function.

    Parameters
    ----------
    n_dimensions : int, default=2
        Number of dimensions.

    Attributes
    ----------
    _n : int
        Number of dimensions.
    """

    def __init__(
            self,
            n_dimensions=2):

        # Initialize the superclass
        super().__init__(
            name='Rastrigin',
            bounds=(full(n_dimensions, -5.12), full(n_dimensions, 5.12)))

        # Get the argument
        self._n = n_dimensions

    def __call__(
            self,
            x):
        """
        Compute the Rastrigin function value.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        float
            Function value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return 10 * self._n + nsum(x**2 - 10 * cos(2*pi*x))

    def gradient(
            self,
            x):
        """
        Compute the Rastrigin gradient.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        ndarray
            Gradient value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return 2 * x + 20 * pi * sin(2*pi*x)


class Sphere(BenchmarkFunction):
    """
    Sphere function.

    Parameters
    ----------
    n_dimensions : int, default=2
        Number of dimensions.

    Attributes
    ----------
    _n : int
        Number of dimensions.
    """

    def __init__(
            self,
            n_dimensions=2):

        # Initialize the superclass
        super().__init__(
            name='Sphere',
            bounds=(full(n_dimensions, -inf), full(n_dimensions, inf)))

        # Get the argument
        self._n = n_dimensions


    def __call__(
            self,
            x):
        """
        Compute the sphere function value.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        float
            Function value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return nsum(x**2)

    def gradient(
            self,
            x):
        """
        Compute the sphere gradient.

        Parameters
        ----------
        x : ndarray
            Evaluation point.

        Returns
        -------
        ndarray
            Gradient value.
        """

        # Ensure that x is an array
        x = asarray(x)

        return 2.0 * x


class StyblinskiTang(BenchmarkFunction):
    """
    Styblinski-Tang function.

    Parameters
    ----------
    n_dimensions : int, default=2
        Number of dimensions.

    Attributes
    ----------
    _n : int
        Number of dimensions.
    """

    def __init__(
            self,
            n_dimensions=2):

        # Initialize the superclass
        super().__init__(
            name='Styblinski-Tang',
            bounds=(full(n_dimensions, -5.0), full(n_dimensions, 5.0)))

        # Get the argument
        self._n = n_dimensions


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

        # Ensure that x is an array
        x = asarray(x)

        return nsum(x**4 - 16*x**2 + 5*x) / 2

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

        # Ensure that x is an array
        x = asarray(x)

        return 2.0 * x**3 - 16 * x + 2.5
