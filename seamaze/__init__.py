"""
seamaze: a python library for classical and dynamical low-rank CMA-ES variants.

==================================================================

seamaze is a Python library for classical and Dynamical Low-Rank (DLR) CMA-ES variants. It is designed to navigate complex, high-dimensional fitness landscapes by iteratively adapting a multivariate Gaussian search space to the objective's local topography. By leveraging DLR approximations, seamaze remains computationally efficient even on ill-conditioned or rugged black-box problems. This implementation further extends to the integration of first-order information, constraints, and robust restart mechanisms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze import cmaes, dlrcmaes, visualization

__all__ = [
    'cmaes',
    'dlrcmaes',
    'visualization']
