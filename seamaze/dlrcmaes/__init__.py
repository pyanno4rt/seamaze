"""
Dynamical low-rank covariance matrix adaptation evolution strategy (DLRA-CMA-ES) module.

==================================================================

This module aims to provide methods and classes for dynamical low-rank CMA-ES solvers.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze.dlrcmaes._low_rank_integrator import LowRankIntegrator
from seamaze.dlrcmaes._dlrcmaes import DLRCMAES

__all__ = [
    'LowRankIntegrator',
    'DLRCMAES']
