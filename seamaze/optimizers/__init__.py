"""
Optimizers module.

==================================================================

This module aims to provide methods and classes for (low-rank) covariance \
matrix adaptation evolutionary strategies.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers._low_rank_integrator import LowRankIntegrator

from seamaze.optimizers._cmaes import CMAES
from seamaze.optimizers._dlrcmaes import DLRCMAES
from seamaze.optimizers._lmcmaes import LMCMAES

__all__ = [
    'LowRankIntegrator',
    'CMAES',
    'DLRCMAES',
    'LMCMAES']
