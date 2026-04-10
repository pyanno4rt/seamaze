"""
Optimizers module.

==================================================================

This module aims to provide methods and classes for (low-rank) covariance \
matrix adaptation evolutionary algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers import evolutionary
from seamaze.optimizers import low_rank

__all__ = [
    'evolutionary',
    'low_rank']
