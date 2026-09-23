# Unit tests for benchmark optimization functions
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import pytest
from numpy import array, full

from seamaze.benchmarks import (
    Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)
from seamaze.optimizers import CMAES

# Global optimum per benchmark: (coordinate of x* in every dimension,
# f(x*) per dimension). Most benchmarks have x*=zeros(dim) and f(x*)=0,
# the exceptions are listed explicitly.
OPTIMA = {
    Ackley: (0.0, 0.0),
    BentCigar: (0.0, 0.0),
    Discus: (0.0, 0.0),
    Ellipsoid: (0.0, 0.0),
    Griewank: (0.0, 0.0),
    LinearSlope: (5.0, 0.0),
    Rastrigin: (0.0, 0.0),
    Rosenbrock: (1.0, 0.0),
    RotatedEllipsoid: (0.0, 0.0),
    RotatedRastrigin: (0.0, 0.0),
    Schwefel: (420.9687462275036, 0.0),
    Sphere: (0.0, 0.0),
    StyblinskiTang: (-2.903534027771178, -39.16616570377142), #opt. value in multi-D is -39.166...*nDim
    SumOfDiffPowers: (0.0, 0.0),
    }

DIMENSIONS = [1, 2, 5, 10]


# Check obj. function value at optimum:
# f(x*) should equal the known optimal value for several dimensions
@pytest.mark.parametrize('ndim', DIMENSIONS)
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_objectiveAtOpt(benchmark, ndim):
    x_opt, f_opt = OPTIMA[benchmark]
    problem = benchmark(ndim)

    # Schwefel's constant 418.9829 is rounded, so f(x*) is only ~1e-5 per dim
    assert problem(full(ndim, x_opt)) == pytest.approx(
        f_opt * ndim, abs=1e-4 * ndim)
