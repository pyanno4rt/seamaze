# Unit tests for benchmark optimization functions
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import pytest
from numpy import array, eye, full, zeros
from numpy.random import default_rng

from seamaze.benchmarks import (
    BenchmarkFunction, Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)

# Check base class 
# -> only defines the interface, so calling it must raise NotImplementedError
# until a subclass overrides the method
@pytest.mark.parametrize('method', ['__call__', 'gradient'])
def test_baseClassNotImplemented(method):
    base = BenchmarkFunction(
        name='Base', ndim=2, bounds=(full(2, -1.0), full(2, 1.0)))

    with pytest.raises(NotImplementedError):
        getattr(base, method)(zeros(2))

# Check actual benchmark functions
# Global optimum per benchmark: (coordinate of x* in every dimension,
# f(x*) per dimension). Most benchmarks have x*=zeros(dim) and f(x*)=0,
# the exceptions are listed explicitly
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

# Check zero dimensional problem returns zero
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_zeroDimReturnsZero(benchmark):
    problem = benchmark(0)

    assert problem(full(0, zeros(0))) == 0.0
    assert problem.gradient(zeros(0)).shape == (0,)

# Check obj. function value at optimum:
# f(x*) should equal the known optimal value for several dimensions
@pytest.mark.parametrize('ndim', DIMENSIONS)
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_objectiveAtOpt(benchmark, ndim):
    x_opt, f_opt = OPTIMA[benchmark]
    problem = benchmark(ndim)

    assert problem(full(ndim, x_opt)) == pytest.approx(
        f_opt * ndim, abs=1e-4 * ndim)


# Check gradient at optimum:
# grad f(x*) should be zero up to numerical inaccuracies, except for Linear Slope, whose optimum lies on
# the upper bound, so the gradient there is the constant slope -1
@pytest.mark.parametrize('ndim', DIMENSIONS)
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_gradientAtOpt(benchmark, ndim):
    x_opt, _ = OPTIMA[benchmark]
    problem = benchmark(ndim)
    grad_opt = -1.0 if benchmark is LinearSlope else 0.0

    assert problem.gradient(full(ndim, x_opt)) == pytest.approx(
        full(ndim, grad_opt), abs=1e-6)


# Check gradient against central finite differences:
# evaluated at a few fixed random points 
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_gradientFiniteDiff(benchmark):
    ndim, num_points, step = 5, 3, 1e-4
    problem = benchmark(ndim)
    lower, upper = problem.bounds
    rng = default_rng(0)

    for x in rng.uniform(0.9 * lower, 0.9 * upper, size=(num_points, ndim)):
        finite_diff = array([
            (problem(x + step * e) - problem(x - step * e)) / (2.0 * step)
            for e in eye(ndim)])
        assert problem.gradient(x) == pytest.approx(
            finite_diff, rel=1e-4, abs=1e-4)
