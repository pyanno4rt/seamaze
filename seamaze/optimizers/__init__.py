"""
Optimizers module.

==================================================================

This module aims to provide methods and classes for (low-rank) covariance \
matrix adaptation evolutionary algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers.evolutionary._cmaes import CMAES
from seamaze.optimizers.low_rank._dlrcmaes import DLRCMAES

__all__ = [
    'CMAES',
    'DLRCMAES']
