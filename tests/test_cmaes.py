# Unit tests for benchmark optimization functions
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

from signal import SIGINT, getsignal
from time import time

import pytest
from numpy import (
    arange, argsort, array, ceil, diag, diff, exp, eye, float64, full, inf,
    int64, linspace, log, maximum, ones, outer, sqrt, where, zeros)
from numpy import mean as nmean
from numpy import sum as nsum
from numpy.linalg import eigh, eigvalsh, norm, qr
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

from seamaze.benchmarks import (
    Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)
from seamaze.optimizers import CMAES
from seamaze.optimizers._cmaes import _tell, _update_covariance

# Global optimum per benchmark: (coordinate of x* in every dimension,
# f(x*) per dimension)
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

# Tolerances for the optimal point and value
X_TOL = 1e-4
F_TOL = 1e-4

# Sum of Different Powers is very flat around x*, so the optimal point can only be checked loosely
X_TOL_FLAT = 1e-1


# Check optimum is found
@pytest.mark.parametrize('benchmark', OPTIMA, ids=lambda b: b.__name__)
def test_findOpt(benchmark):
    x_opt, f_opt = OPTIMA[benchmark]
    problem = benchmark(NDIM)
    lower, upper = problem.bounds

    # Start slightly off the optimum
    start = x_opt - 0.1 if x_opt == upper[0] else x_opt + 0.1

    solver = CMAES(
        number_of_variables=NDIM,
        objective=problem.__call__,
        lower_variable_bounds=lower,
        upper_variable_bounds=upper,
        initial_sigma=0.1,
        min_log_level='critical',
        random_state=42,
        tolerance=1e-8,
        sigma_threshold=1e-8)
    result = solver.optimize(full(NDIM, start))

    x_tol = X_TOL_FLAT if benchmark is SumOfDiffPowers else X_TOL
    assert result['optimal_point'] == pytest.approx(
        full(NDIM, x_opt), abs=x_tol)
    assert result['optimal_value'] == pytest.approx(f_opt * NDIM, abs=F_TOL)

#separate tests for each function:


def make_solver(**kwargs):
    """Create a CMA-ES on the NDIM-dimensional sphere (quiet, fixed seed)."""
    return CMAES(NDIM, Sphere(NDIM), **{
        'min_log_level': 'critical', 'random_state': 42, **kwargs})


def bounds(lower, upper):
    """Build the bound keyword arguments for a box [lower, upper]^NDIM."""
    return {'lower_variable_bounds': full(NDIM, lower),
            'upper_variable_bounds': full(NDIM, upper)}


#test __init__
#expectation: when called with certain inputs those are class variables afterwards, when called with no/minimal input default values are filled in

def test_init_storesArguments():
    lower, upper = full(NDIM, -2.0), full(NDIM, 3.0)

    def callback(solver):
        pass

    solver = CMAES(
        number_of_variables=NDIM, objective=Sphere(NDIM),
        lower_variable_bounds=lower, upper_variable_bounds=upper,
        number_of_individuals=12, initial_sigma=0.3, maximum_iterations=77,
        maximum_wall_time=5, fitness_threshold=1e-3, fitness_window_size=20,
        tolerance=1e-9, sigma_threshold=1e-10, update_interval=3,
        min_log_level='critical', callback=callback, random_state=7)

    assert solver._number_of_variables == NDIM
    assert solver.objective(full(NDIM, 1.0)) == NDIM
    assert solver.lower_variable_bounds is lower
    assert solver.upper_variable_bounds is upper
    assert solver._is_bound
    assert solver._pop_size == 12
    assert solver._sigma == 0.3
    assert solver.maximum_iterations == 77
    assert solver.maximum_wall_time == 5
    assert solver.fitness_threshold == 1e-3
    assert solver._fitness_history.maxlen == 20
    assert solver.tolerance == 1e-9
    assert solver.sigma_threshold == 1e-10
    assert solver._update_interval == 3
    assert solver._callback is callback


def test_init_defaults():
    solver = CMAES(NDIM, Sphere(NDIM), min_log_level='critical')

    # Unbounded problem with default population size and step size
    assert_array_equal(solver.lower_variable_bounds, full(NDIM, -inf))
    assert_array_equal(solver.upper_variable_bounds, full(NDIM, inf))
    assert not solver._is_bound
    assert solver.gradient is None
    assert solver._pop_size == 4 + int(3 * log(NDIM))
    assert solver._sigma == 1.0

    # Default stopping criteria
    assert solver.maximum_iterations == 100000
    assert solver.maximum_wall_time == 43200
    assert solver.fitness_threshold == -inf
    assert solver._fitness_history.maxlen == 50
    assert solver.tolerance == 1e-6
    assert solver.sigma_threshold == 1e-8
    assert solver._update_interval == ceil(sqrt(1 / solver._lr_cov))
    assert solver._callback is None

    # Standard normal search distribution around the origin, empty result
    assert solver._opt_iter == 0
    assert_array_equal(solver._mean, zeros(NDIM))
    assert_array_equal(solver._path_sigma, zeros(NDIM))
    assert_array_equal(solver._path_cov, zeros(NDIM))
    for matrix in (solver._cov, solver._root_cov, solver._basis):
        assert_array_equal(matrix, eye(NDIM))
    assert_array_equal(solver._core, ones(NDIM))
    assert solver._gamma is None
    assert solver._result == {
        'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
        'wall_time': None, 'iterations': None}


# A gradient adds two individuals for the mirrored gradient steps
def test_init_gradientIncreasesPopulation():
    solver = make_solver(gradient=Sphere(NDIM).gradient)

    assert solver._pop_size == 4 + int(3 * log(NDIM)) + 2


# A disabled step-size threshold (None) is replaced by 0.0
def test_init_sigmaThresholdNone():
    assert make_solver(sigma_threshold=None).sigma_threshold == 0.0


def test_init_weightsAndLearningRates():
    solver = make_solver()
    weights = solver._weights.ravel()
    positive = weights[:solver._elite_size]

    # The better half of the population gets positive weights summing to 1,
    # the weights decrease with the rank and the rest are negative
    assert solver._weights.shape == (solver._pop_size, 1)
    assert solver._elite_size == solver._pop_size // 2
    assert (positive > 0.0).all()
    assert (weights[solver._elite_size:] < 0.0).all()
    assert nsum(positive) == pytest.approx(1.0)
    assert (diff(weights) < 0.0).all()
    assert solver._mu_eff == pytest.approx(1.0 / nsum(positive**2))

    # Learning rates are in (0, 1] and the covariance update keeps a
    # non-negative share of the old covariance matrix
    for rate in (solver._lr_sigma, solver._lr_cov, solver._lr_rank_one,
                 solver._lr_rank_mu, solver._lr_mean):
        assert 0.0 < rate <= 1.0
    assert solver._lr_rank_one + solver._lr_rank_mu <= 1.0
    assert solver._damp_sigma >= 1.0

    # Expected length of a standard normal vector (approx. sqrt(NDIM))
    assert solver._expected_path_length == pytest.approx(sqrt(NDIM), rel=0.1)


#test ask
#expectation: Generates a new population with the correct size and fulfilling given properties (e.g. staying within given bounds)

def test_ask_populationAroundMean():
    solver = make_solver(initial_sigma=0.5)
    solver._mean[:] = arange(NDIM)
    solver.ask()

    assert solver._population.shape == (solver._pop_size, NDIM)
    assert_allclose(solver._population, solver._mean + 0.5 * solver._steps)
    assert solver._mean_squared_bound_errors is None


# The steps are samples from N(0, C) with C = root_cov @ root_cov.T
def test_ask_samplesFromCovariance():
    solver = make_solver(number_of_individuals=20000)
    rotation, _ = qr(default_rng(0).standard_normal((NDIM, NDIM)))
    solver._root_cov[:] = rotation * arange(1.0, NDIM + 1)
    cov = solver._root_cov @ solver._root_cov.T
    solver.ask()

    assert_allclose(nmean(solver._steps, axis=0), zeros(NDIM), atol=0.1)
    assert_allclose(solver._steps.T @ solver._steps / 20000, cov,
                    atol=0.05 * cov.max())


def test_ask_reproducibleWithSeed():
    solvers = [make_solver(random_state=seed) for seed in (1, 1, 2)]
    for solver in solvers:
        solver.ask()

    assert_array_equal(solvers[0]._population, solvers[1]._population)
    assert (solvers[0]._population != solvers[2]._population).any()


# Violating individuals are mirrored back into the box, the others are kept,
# and the mean squared bound error of the unrepaired samples is recorded
def test_ask_repairsIntoBounds():
    solver = make_solver(initial_sigma=2.0, **bounds(-1.0, 1.0))
    solver.ask()
    raw = solver._mean + solver._sigma * solver._steps
    eps_lower, eps_upper = maximum(0.0, -1.0 - raw), maximum(0.0, raw - 1.0)
    inside = (eps_lower == 0.0) & (eps_upper == 0.0)
    sigma_rel = 2.0 / 3.0
    mirrored = where(raw < -1.0, -1.0 + sigma_rel * eps_lower, raw)
    mirrored = where(raw > 1.0, 1.0 - sigma_rel * eps_upper, mirrored)

    assert inside.any() and not inside.all()
    assert ((solver._population >= -1.0) & (solver._population <= 1.0)).all()
    assert_array_equal(solver._population[inside], raw[inside])
    assert_allclose(solver._population, mirrored.clip(-1.0, 1.0))
    assert_allclose(solver._mean_squared_bound_errors,
                    nmean((eps_lower + eps_upper)**2, axis=1))


# Without violations, an initialized penalty factor decays by 0.999 but is
# never clipped below 1e-5
@pytest.mark.parametrize('gamma, expected', [(2.0, 2.0 * 0.999), (1e-5, 1e-5)])
def test_ask_decaysPenaltyFactor(gamma, expected):
    solver = make_solver(initial_sigma=1e-3, **bounds(-1.0, 1.0))
    solver._gamma = gamma
    solver.ask()

    assert solver._gamma == pytest.approx(expected)


# With a gradient, the last two steps are the mirrored natural gradient
# direction, normalized in the metric of C (here C = I)
def test_ask_gradientSteps():
    problem = Sphere(NDIM)
    solver = make_solver(gradient=problem.gradient)
    solver._mean[:] = arange(1.0, NDIM + 1)
    solver.ask()
    direction = problem.gradient(solver._mean)
    direction /= norm(direction)

    assert_allclose(solver._steps[-1], direction)
    assert_allclose(solver._steps[-2], -direction)


#test evaluate
# expectation: evaluates fitness and checks several conditions to see whether pure fitness value needs to be used or penalized/selected differently
#(possible checks, e.g. selection_fitness = true_fitness depending on whether condition is fulfilled), also checks whether optimal value needs to be updates (-> check whether this is done/not done correctly)

# Population whose i-th individual is SCALES[i] * (1, ..., 1), so that its
# sphere fitness is NDIM * SCALES[i]**2 (best: index 1)
SCALES = array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0])


def test_evaluate_unboundedUsesTrueFitness():
    solver = make_solver()
    solver._population[:] = SCALES[:, None]
    true_fitness, selection_fitness = solver.evaluate()

    assert_allclose(true_fitness, NDIM * SCALES**2)
    assert_array_equal(selection_fitness, true_fitness)
    assert solver._gamma is None


# On first use the penalty factor is scaled to the fitness range, and the
# selection fitness adds the weighted bound errors
def test_evaluate_boundedPenalizesViolations():
    solver = make_solver(**bounds(-10.0, 10.0))
    solver._population[:] = SCALES[:, None]
    errors = linspace(0.0, 1.0, SCALES.size)
    solver._mean_squared_bound_errors = errors
    true_fitness, selection_fitness = solver.evaluate()
    gamma = NDIM * (SCALES.max()**2 - SCALES.min()**2)

    assert solver._gamma == pytest.approx(gamma)
    assert_allclose(selection_fitness, true_fitness + gamma * errors)

    # An initialized penalty factor is kept
    solver._gamma = 3.0
    _, selection_fitness = solver.evaluate()

    assert solver._gamma == 3.0
    assert_allclose(selection_fitness, true_fitness + 3.0 * errors)


# A flat fitness range falls back to a penalty factor of 1.0
def test_evaluate_flatFitnessPenaltyFactor():
    solver = make_solver(**bounds(-10.0, 10.0))
    solver._population[:] = 1.0
    solver._mean_squared_bound_errors = zeros(SCALES.size)
    solver.evaluate()

    assert solver._gamma == 1.0


def test_evaluate_tracksOptimum():
    solver = make_solver()
    best_point, best_value = full(NDIM, 1.0), NDIM * 1.0

    # First evaluation: the best individual becomes the optimum (as a copy)
    solver._population[:] = SCALES[:, None]
    solver.evaluate()
    solver._population[:] = 10.0

    assert solver._result['optimal_value'] == best_value
    assert_array_equal(solver._result['optimal_point'], best_point)

    # A worse population leaves the optimum unchanged
    solver.evaluate()

    assert solver._result['optimal_value'] == best_value
    assert_array_equal(solver._result['optimal_point'], best_point)

    # A better population replaces it
    solver._population[:] = 0.1 * SCALES[:, None]
    solver.evaluate()

    assert solver._result['optimal_value'] == pytest.approx(NDIM * 0.01)
    assert_allclose(solver._result['optimal_point'], full(NDIM, 0.1))

    # The best fitness of every generation is stored in the history
    assert list(solver._fitness_history) == pytest.approx(
        [best_value, NDIM * 100.0, NDIM * 0.01])


# The optimum is tracked with the true (unpenalized) fitness
def test_evaluate_optimumIgnoresPenalty():
    solver = make_solver(**bounds(-10.0, 10.0))
    solver._population[:] = SCALES[:, None]
    solver._mean_squared_bound_errors = where(SCALES == 1.0, 100.0, 0.0)
    _, selection_fitness = solver.evaluate()

    assert argsort(selection_fitness)[0] != 1
    assert solver._result['optimal_value'] == NDIM * 1.0


#test tell
#expectation: calls state variable update and correctly uodates variables, checks whether covariance update is necessary and calls update if necessary

def test_tell_movesMeanTowardsBestSteps():
    solver = make_solver(initial_sigma=0.5)
    solver._mean[:] = 1.0
    solver._opt_iter = 1
    solver.ask()
    fitness = default_rng(0).random(solver._pop_size)
    elite = argsort(fitness)[:solver._elite_size]
    old_mean = solver._mean.copy()
    solver.tell(fitness)

    # New mean: old mean plus the weighted steps of the best individuals
    elite_step = nsum(
        solver._steps[elite] * solver._weights[:solver._elite_size], axis=0)
    assert_allclose(solver._mean, old_mean + 0.5 * elite_step)


# tell stores exactly what _tell computes from the current state
def test_tell_storesStateUpdate():
    solver = make_solver()
    solver._opt_iter = 1
    solver.ask()
    fitness = default_rng(0).random(solver._pop_size)
    path_sigma, mean, sigma, path_cov, _ = _tell(
        solver._steps.copy(), argsort(fitness), solver._weights,
        solver._basis.copy(), solver._core.copy(), solver._path_sigma.copy(),
        solver._path_cov.copy(), solver._mean.copy(), solver._sigma,
        solver._lr_sigma, solver._lr_cov, solver._lr_mean, solver._mu_eff,
        solver._damp_sigma, solver._expected_path_length, solver._opt_iter,
        solver._elite_size)
    solver.tell(fitness)

    assert_allclose(solver._path_sigma, path_sigma)
    assert_allclose(solver._mean, mean)
    assert solver._sigma == pytest.approx(sigma)
    assert_allclose(solver._path_cov, path_cov)


# The covariance factors are only updated every update_interval iterations,
# and then consistently (root_cov @ root_cov.T == cov)
@pytest.mark.parametrize('opt_iter, updated', [
    (1, False), (2, False), (3, True), (4, False), (6, True)])
def test_tell_covarianceUpdateInterval(opt_iter, updated):
    solver = make_solver(update_interval=3)
    solver._opt_iter = opt_iter
    solver.ask()
    cov_before = solver._cov.copy()
    solver.tell(default_rng(0).random(solver._pop_size))

    assert (solver._cov != cov_before).any() == updated
    assert_allclose(solver._root_cov @ solver._root_cov.T, solver._cov,
                    atol=1e-12)


#test optimize
#expectation: wrappper function that calls components of optimization and then correctly reads out results in the end and gived them as output

def test_optimize_returnsResult():
    solver = make_solver(maximum_iterations=5)
    result = solver.optimize(full(NDIM, 2.0))

    assert result is solver._result
    assert set(result) == {'optimal_point', 'optimal_value', 'solver_info',
                           'wall_time', 'iterations'}
    assert result['iterations'] == 5
    assert result['solver_info'] == 'MAX_ITER_REACHED'
    assert result['wall_time'] >= 0.0
    assert result['optimal_value'] == pytest.approx(
        Sphere(NDIM)(result['optimal_point']))
    assert result['optimal_value'] == min(solver._fitness_history)


@pytest.mark.parametrize('initial_mean, expected', [
    (None, zeros(NDIM)), (arange(NDIM), arange(float(NDIM)))],
    ids=['default_origin', 'integer_array'])
def test_optimize_initialMean(initial_mean, expected):
    solver = make_solver(maximum_iterations=0)
    solver.optimize(initial_mean)

    assert solver._mean.dtype == float64
    assert_array_equal(solver._mean, expected)


# Every iteration runs ask, evaluate, tell and the callback in this order
def test_optimize_callsStepsInOrder(monkeypatch):
    calls = []
    solver = make_solver(
        maximum_iterations=3,
        callback=lambda s: calls.append(('callback', s._opt_iter)))
    for name in ('ask', 'evaluate', 'tell'):
        method = getattr(solver, name)

        def recorded(*args, _name=name, _method=method):
            calls.append((_name, solver._opt_iter))
            return _method(*args)

        monkeypatch.setattr(solver, name, recorded)
    solver.optimize(full(NDIM, 2.0))

    assert calls == [(step, iteration) for iteration in (1, 2, 3)
                     for step in ('ask', 'evaluate', 'tell', 'callback')]


# Ctrl+C (SIGINT) stops the run gracefully after the current iteration, and
# the previous signal handler is restored afterwards
def test_optimize_stopsOnInterrupt():
    original_handler = getsignal(SIGINT)

    def interrupt(solver):
        if solver._opt_iter == 2:
            getsignal(SIGINT)(SIGINT, None)

    result = make_solver(maximum_iterations=100, callback=interrupt).optimize(
        full(NDIM, 2.0))

    assert result['solver_info'] == 'STOPPED_BY_USER'
    assert result['iterations'] == 2
    assert getsignal(SIGINT) is original_handler


# The signal handler is also restored if an exception interrupts the run
def test_optimize_restoresHandlerOnError():
    original_handler = getsignal(SIGINT)

    def fail(solver):
        raise RuntimeError('callback failed')

    with pytest.raises(RuntimeError, match='callback failed'):
        make_solver(callback=fail).optimize(full(NDIM, 2.0))

    assert getsignal(SIGINT) is original_handler


#test check_termination
#expectation: if I give certain termination criteria it returns True otherwise False

def reach_max_iterations(solver):
    solver._opt_iter = solver.maximum_iterations


def exceed_wall_time(solver):
    solver.maximum_wall_time = 5
    solver._wall_start = time() - 10


def shrink_sigma(solver):
    solver._sigma = 1e-9


def zero_eigenvalue(solver):
    solver._core[0] = 1e-15


def huge_condition_number(solver):
    solver._core[0] = 1e15


def reach_fitness_threshold(solver):
    solver.fitness_threshold = 1.0
    solver._result['optimal_value'] = 0.5


def absolute_plateau(solver):
    solver._fitness_history.extend([1.0] * 50)


# Range 0.5 is above the tolerance, but tiny relative to the values (1e6)
def relative_plateau(solver):
    solver._fitness_history.extend(1e6 + linspace(0.0, 0.5, 50))


# Adding 0.1 * sigma * axis to a huge mean has no effect in floating point
def no_effect_axis(solver):
    solver._mean[:] = 1e20


def max_iterations_and_small_sigma(solver):
    reach_max_iterations(solver)
    shrink_sigma(solver)


@pytest.mark.parametrize('setup, solver_info', [
    (reach_max_iterations, 'MAX_ITER_REACHED'),
    (exceed_wall_time, 'MAX_WALL_TIME_REACHED'),
    (shrink_sigma, 'SIGMA_BELOW_THRESH'),
    (zero_eigenvalue, 'MAX_COND_NUM_EXCEEDED'),
    (huge_condition_number, 'MAX_COND_NUM_EXCEEDED'),
    (reach_fitness_threshold, 'FITNESS_BELOW_THRESH'),
    (absolute_plateau, 'ABSOLUTE_FITNESS_PLATEAU_REACHED'),
    (relative_plateau, 'RELATIVE_FITNESS_PLATEAU_REACHED'),
    (no_effect_axis, 'NO_EFFECT_AXIS'),
    # Criteria are checked in order, the first fulfilled one is reported
    (max_iterations_and_small_sigma, 'MAX_ITER_REACHED')],
    ids=lambda value: getattr(value, '__name__', value))
def test_checkTermination_stops(setup, solver_info):
    solver = make_solver()
    setup(solver)

    assert solver.check_termination() is True
    assert solver._result['solver_info'] == solver_info


def fresh_solver(solver):
    pass


def below_max_iterations(solver):
    solver._opt_iter = solver.maximum_iterations - 1


def wall_clock_not_started(solver):
    solver.maximum_wall_time = 0


def history_not_full(solver):
    solver._fitness_history.extend([1.0] * 49)


def history_not_finite(solver):
    solver._fitness_history.extend([inf] * 50)


def history_still_improving(solver):
    solver._fitness_history.extend(linspace(10.0, 1.0, 50))


def fitness_threshold_disabled(solver):
    solver.fitness_threshold = None
    solver._result['optimal_value'] = -1e300


@pytest.mark.parametrize('setup', [
    fresh_solver, below_max_iterations, wall_clock_not_started,
    history_not_full, history_not_finite, history_still_improving,
    fitness_threshold_disabled],
    ids=lambda setup: setup.__name__)
def test_checkTermination_continues(setup):
    solver = make_solver()
    setup(solver)

    assert solver.check_termination() is False
    assert solver._result['solver_info'] is None


#test _tell
#expectation: state variables are updated and maybe also find some sanity checks (update switch is 0.0 or 1.0, ..?)

def call_tell(solver, steps, sorting, core=None, path_sigma=None,
              path_cov=None, sigma=1.0, opt_iter=1):
    """Call _tell with the solver's parameters, identity basis, zero mean."""
    return _tell(
        steps, sorting, solver._weights, eye(NDIM),
        ones(NDIM) if core is None else core,
        zeros(NDIM) if path_sigma is None else path_sigma,
        zeros(NDIM) if path_cov is None else path_cov, zeros(NDIM), sigma,
        solver._lr_sigma, solver._lr_cov, solver._lr_mean, solver._mu_eff,
        solver._damp_sigma, solver._expected_path_length, opt_iter,
        solver._elite_size)


def random_steps(solver, scale=1.0):
    """Draw random steps and a random fitness ranking."""
    rng = default_rng(0)
    steps = scale * rng.standard_normal((solver._pop_size, NDIM))
    return steps, argsort(rng.random(solver._pop_size)).astype(int64)


def test_tell_updateFormulas():
    solver = make_solver()
    steps, sorting = random_steps(solver)
    path_sigma, mean, sigma, path_cov, switch = call_tell(
        solver, steps, sorting, sigma=0.5)
    elite_step = nsum(steps[sorting[:solver._elite_size]]
                      * solver._weights[:solver._elite_size], axis=0)
    lr_sigma, lr_cov, mu_eff = solver._lr_sigma, solver._lr_cov, solver._mu_eff
    expected_path_sigma = sqrt(lr_sigma * (2 - lr_sigma) * mu_eff) * elite_step
    ps_norm = norm(expected_path_sigma)

    assert_allclose(mean, 0.5 * elite_step)
    assert_allclose(path_sigma, expected_path_sigma)
    assert sigma == pytest.approx(0.5 * exp(
        lr_sigma / solver._damp_sigma
        * (ps_norm / solver._expected_path_length - 1.0)))
    assert switch == 1.0
    assert_allclose(path_cov, sqrt(lr_cov * (2 - lr_cov) * mu_eff) * elite_step)


# The step-size path is whitened with C^(-1/2): for C = 4 I it is halved,
# while the mean update is not affected
def test_tell_whitensStepSizePath():
    solver = make_solver()
    steps, sorting = random_steps(solver)
    path_identity, mean_identity, *_ = call_tell(solver, steps, sorting)
    path_scaled, mean_scaled, *_ = call_tell(
        solver, steps, sorting, core=full(NDIM, 4.0))

    assert_allclose(path_scaled, 0.5 * path_identity)
    assert_allclose(mean_scaled, mean_identity)


# The step size grows for long and shrinks for short step-size paths, and
# is clipped from below at 1e-15
@pytest.mark.parametrize('path_length, sigma_in, relation', [
    (3.0, 1.0, 'larger'), (0.0, 1.0, 'smaller'), (0.0, 1e-20, 'clipped')])
def test_tell_stepSizeAdaptation(path_length, sigma_in, relation):
    solver = make_solver()
    steps = zeros((solver._pop_size, NDIM))
    path_sigma = full(NDIM, path_length * solver._expected_path_length
                      / sqrt(NDIM) / (1.0 - solver._lr_sigma))
    sigma = call_tell(solver, steps, arange(solver._pop_size, dtype=int64),
                      path_sigma=path_sigma, sigma=sigma_in)[2]

    if relation == 'larger':
        assert sigma > sigma_in
    elif relation == 'smaller':
        assert sigma < sigma_in
    else:
        assert sigma == 1e-15


# A too long step-size path switches off the covariance path update (switch
# is exactly 0.0 or 1.0): the path then only decays
@pytest.mark.parametrize('scale, switch', [(1.0, 1.0), (1e3, 0.0)])
def test_tell_updateSwitch(scale, switch):
    solver = make_solver()
    steps, sorting = random_steps(solver, scale)
    path_cov_in = ones(NDIM)
    *_, path_cov, update_switch = call_tell(
        solver, steps, sorting, path_cov=path_cov_in.copy())

    assert update_switch == switch
    if switch == 0.0:
        assert_allclose(path_cov, (1.0 - solver._lr_cov) * path_cov_in)


#test _update_covariance
#expectation: output variables are updated and fullfil some properties: e.g. root_cov²=cov, etc.

def call_update(solver, basis, core, steps, sorting, weights, path_cov,
                switch=1.0):
    """Call _update_covariance with the solver's learning rates."""
    return _update_covariance(
        basis, core, steps, sorting, weights, path_cov, solver._lr_cov,
        solver._lr_rank_one, solver._lr_rank_mu, switch)


# Starting from a random covariance matrix, the outputs are consistent:
# symmetric positive definite cov = basis diag(core) basis^T = root root^T,
# orthonormal basis and descending positive eigenvalues
def test_updateCovariance_consistentFactors():
    solver = make_solver()
    rng = default_rng(0)
    factor = rng.standard_normal((NDIM, NDIM))
    core, basis = eigh(factor @ factor.T + eye(NDIM))
    steps, sorting = random_steps(solver)
    cov, root_cov, basis_new, core_new = call_update(
        solver, basis, core, steps, sorting, solver._weights,
        rng.standard_normal(NDIM))
    atol = 1e-10 * norm(cov)

    assert_allclose(cov, cov.T, atol=atol)
    assert eigvalsh(cov).min() > 0.0
    assert (core_new > 0.0).all() and (diff(core_new) <= 0.0).all()
    assert_allclose(basis_new.T @ basis_new, eye(NDIM), atol=1e-10)
    assert_allclose((basis_new * core_new) @ basis_new.T, cov, atol=atol)
    assert_allclose(root_cov @ root_cov.T, cov, atol=atol)


# Exact update for C = I and positive weights:
#   C_new = (1 - c1 - cmu + (1 - switch) c1 cc (2 - cc)) I
#           + c1 p p^T + cmu sum_i w_i y_(i) y_(i)^T
@pytest.mark.parametrize('switch', [1.0, 0.0])
def test_updateCovariance_formula(switch):
    solver = make_solver()
    rng = default_rng(0)
    steps = rng.standard_normal((3, NDIM))
    sorting = array([2, 0, 1], dtype=int64)
    weights = array([[0.5], [0.3], [0.2]])
    path_cov = rng.standard_normal(NDIM)
    cov = call_update(solver, eye(NDIM), ones(NDIM), steps, sorting, weights,
                      path_cov, switch)[0]
    lr_one, lr_mu, lr_cov = (solver._lr_rank_one, solver._lr_rank_mu,
                             solver._lr_cov)
    rank_mu = sum(w * outer(steps[i], steps[i])
                  for w, i in zip(weights.ravel(), sorting))
    keep = 1.0 - lr_one - lr_mu + (1.0 - switch) * lr_one * lr_cov * (2 - lr_cov)

    assert_allclose(cov, keep * eye(NDIM) + lr_one * outer(path_cov, path_cov)
                    + lr_mu * rank_mu, atol=1e-12)


# Negative weights of long steps are rescaled by NDIM / |z|^2, so that a
# bad step cannot make the covariance matrix indefinite; the input weights
# are not modified
def test_updateCovariance_rescalesNegativeWeights():
    solver = make_solver()
    steps = zeros((2, NDIM))
    steps[0, 0], steps[1, 1] = 1.0, 100.0
    weights = array([[1.0], [-0.5]])
    cov = call_update(solver, eye(NDIM), ones(NDIM), steps,
                      array([0, 1], dtype=int64), weights, zeros(NDIM))[0]
    effective = -0.5 * NDIM / 100.0**2
    lr_one, lr_mu = solver._lr_rank_one, solver._lr_rank_mu
    expected = (1.0 - lr_one - lr_mu) * eye(NDIM) + lr_mu * (
        outer(steps[0], steps[0]) + effective * outer(steps[1], steps[1]))

    assert_allclose(cov, expected, atol=1e-12)
    assert eigvalsh(cov).min() > 0.0
    assert_array_equal(weights, [[1.0], [-0.5]])
