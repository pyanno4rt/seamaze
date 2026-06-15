"""
Benchmarks module.

==================================================================

This module aims to provide methods and classes for benchmarking the
implemented algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.benchmarks._benchmark_function import BenchmarkFunction
from seamaze.benchmarks._ackley import Ackley
from seamaze.benchmarks._bent_cigar import BentCigar
from seamaze.benchmarks._discus import Discus
from seamaze.benchmarks._ellipsoid import Ellipsoid
from seamaze.benchmarks._griewank import Griewank
from seamaze.benchmarks._linear_slope import LinearSlope
from seamaze.benchmarks._rastrigin import Rastrigin
from seamaze.benchmarks._rosenbrock import Rosenbrock
from seamaze.benchmarks._rotated_ellipsoid import RotatedEllipsoid
from seamaze.benchmarks._rotated_rastrigin import RotatedRastrigin
from seamaze.benchmarks._schwefel import Schwefel
from seamaze.benchmarks._sphere import Sphere
from seamaze.benchmarks._styblinski_tang import StyblinskiTang
from seamaze.benchmarks._sum_of_diff_powers import SumOfDiffPowers

__all__ = [
    'BenchmarkFunction',
    'Ackley',
    'BentCigar',
    'Discus',
    'Ellipsoid',
    'Griewank',
    'LinearSlope',
    'Rastrigin',
    'Rosenbrock',
    'RotatedEllipsoid',
    'RotatedRastrigin',
    'Schwefel',
    'Sphere',
    'StyblinskiTang',
    'SumOfDiffPowers']
