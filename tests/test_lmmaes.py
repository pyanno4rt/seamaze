# Unit tests for benchmark optimization functions
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import pytest
from numpy import full

from seamaze.benchmarks import (
    Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)
from seamaze.optimizers import LMMAES

# Global optimum per benchmark: (coordinate of x* in every dimension,
# f(x*) per dimension). 
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

# Problem dimension 
NDIM = 5

# Tolerances for the optimal point and value (softer for limited memory
# opt., which stalls around f~1e-3 on the ill-conditioned Bent-Cigar/Discus)
X_TOL = 1e-1
F_TOL = 1e-2

# Sum of Different Powers is very flat around x*, so the optimal point can only be checked loosely
X_TOL_FLAT = 1e-1


# Check optimum is found in NDIM dimensions
# -> LM-MA-ES started inside the global basin should converge to x*
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_findOpt(benchmark):
    x_opt, f_opt = OPTIMA[benchmark]
    problem = benchmark(NDIM)
    lower, upper = problem.bounds

    # Start slightly off the optimum (inside the bounds for Linear Slope,
    # whose optimum lies on the upper bound)
    start = x_opt - 0.05 if x_opt == upper[0] else x_opt + 0.05

    solver = LMMAES(
        number_of_variables=NDIM,
        objective=problem.__call__,
        lower_variable_bounds=lower,
        upper_variable_bounds=upper,
        initial_sigma=0.1,
        min_log_level='critical',
        random_state=42)
    result = solver.optimize(full(NDIM, start))

    x_tol = X_TOL_FLAT if benchmark is SumOfDiffPowers else X_TOL
    assert result['optimal_point'] == pytest.approx(
        full(NDIM, x_opt), abs=x_tol)
    assert result['optimal_value'] == pytest.approx(f_opt * NDIM, abs=F_TOL)
