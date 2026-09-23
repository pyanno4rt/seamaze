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

# Check optimum is found in 1D
# -> CMA-ES started inside the global basin should converge to x*
# (Rosenbrock is excluded since it is constant in 1d)
@pytest.mark.parametrize(
    'benchmark', [b for b in OPTIMA if b is not Rosenbrock],
    ids=lambda b: b.__name__)
def test_findOpt(benchmark):
    x_opt, f_opt = OPTIMA[benchmark]
    problem = benchmark(1)
    lower, upper = problem.bounds

    # Start slightly off the optimum (inside the bounds for Linear Slope,
    # whose optimum lies on the upper bound)
    start = x_opt - 0.25 if x_opt == upper[0] else x_opt + 0.25

    solver = CMAES(
        number_of_variables=1,
        objective=problem.__call__,
        lower_variable_bounds=lower,
        upper_variable_bounds=upper,
        initial_sigma=0.1,
        min_log_level='critical',
        random_state=42)
    result = solver.optimize(array([start]))

    assert result['optimal_point'][0] == pytest.approx(x_opt, abs=1e-6)
    assert result['optimal_value'] == pytest.approx(f_opt, abs=1e-4)
