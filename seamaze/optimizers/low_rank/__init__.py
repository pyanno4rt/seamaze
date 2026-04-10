"""
Low-rank evolutionary algorithms module.

==================================================================

This module aims to provide methods and classes for low-rank covariance \
matrix adaptation evolutionary algorithms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.optimizers.low_rank._low_rank_integrator import LowRankIntegrator
from seamaze.optimizers.low_rank._dlrcmaes import DLRCMAES

__all__ = [
    'LowRankIntegrator',
    'DLRCMAES']
