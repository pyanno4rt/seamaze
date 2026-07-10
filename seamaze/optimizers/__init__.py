"""
Optimizers module.

==================================================================

This module aims to provide methods and classes for (low-rank) covariance \
matrix adaptation evolutionary strategies.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers._cmaes import CMAES
from seamaze.optimizers._dlrcmaes import DLRCMAES
from seamaze.optimizers._lmmaes import LMMAES

__all__ = [
    'CMAES',
    'DLRCMAES',
    'LMMAES']
