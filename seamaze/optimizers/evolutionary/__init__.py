"""
Evolutionary algorithms module.

==================================================================

This module aims to provide methods and classes for covariance matrix \
adaptation evolutionary algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers.evolutionary._cmaes import CMAES
from seamaze.optimizers.evolutionary._lmcmaes import LMCMAES

__all__ = [
    'CMAES',
    'LMCMAES']
