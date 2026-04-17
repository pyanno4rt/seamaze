"""
Benchmarks module.

==================================================================

This module aims to provide methods and classes for benchmarking the \
implemented algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.benchmarks._benchmark_function import BenchmarkFunction
from seamaze.benchmarks._rastrigin import Rastrigin
from seamaze.benchmarks._sphere import Sphere
from seamaze.benchmarks._stybtang import StyblinskiTang

__all__ = [
    'BenchmarkFunction',
    'Rastrigin',
    'Sphere',
    'StyblinskiTang']
