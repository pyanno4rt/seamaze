"""
seamaze: a python library for classical, limited-memory, and dynamical
low-rank CMA-ES variants.

==================================================================

seamaze is a Python library for classical, limited-memory and Dynamical
Low-Rank (DLR) variants of the Covariance Matrix Adaptation Evolution Strategy
(CMA-ES). It provides state-of-the-art, derivative-free algorithms designed
for continuous, non-linear, and non-convex real-parameter optimization,
excelling in ill-conditioned, non-separable, or rugged fitness landscapes.
By leveraging limited-memory and DLR approximations, seamaze maintains
computational efficiency even on high-dimensional black-box problems. This
implementation further incorporates first-order information, constraint
handling, and multi-stage restart mechanisms.
"""

# Author: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from seamaze import (
    benchmarks, diagnostics, logging, optimizers, plotting, utils)

__all__ = [
    'benchmarks',
    'diagnostics',
    'logging',
    'optimizers',
    'plotting',
    'utils']
