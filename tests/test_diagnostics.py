# Unit (and integration) tests for the monitoring classes
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import sys
from types import SimpleNamespace

import pytest
from numpy import array, float64, full, inf, ones, zeros
from numpy.linalg import norm

from seamaze.benchmarks import Sphere
from seamaze.diagnostics import MonitorCMAES, MonitorDLRCMAES, MonitorLMMAES
from seamaze.optimizers import CMAES, DLRCMAES, LMMAES

# Keys recorded by all monitors in every recorded iteration
SERIES_KEYS = {
    'iteration', 'optimal_value', 'best_fitness', 'mean_fitness',
    'worst_fitness', 'max_bound_viol', 'mean_bound_viol', 'gamma',
    'path_sigma_norm', 'sigma', 'cov_cn', 'cov_spectr_norm'}

# Additional keys recorded by the 'full' callback
FULL_KEYS = {'mean', 'mean_change_norm', 'cov_svs'}

# Solver-monitor pairs
PAIRS = [(CMAES, MonitorCMAES), (LMMAES, MonitorLMMAES),
         (DLRCMAES, MonitorDLRCMAES)]
PAIR_IDS = ['CMAES', 'LMMAES', 'DLRCMAES']

# Problem dimension and number of iterations for the real solver runs
NDIM = 5
NUM_ITER = 10


# Tests with a small fake solver (unit tests)
def make_fake_solver(fitness=(3.0, 1.0, 2.0), errors=None, core=(4.0, 1.0)):
    """Build a fake CMA-ES exposing all attributes read by MonitorCMAES."""
    ndim = len(core)
    return SimpleNamespace(
        # Static parameters
        lower_variable_bounds=full(ndim, -inf),
        upper_variable_bounds=full(ndim, inf),
        _damp_sigma=float64(1.0),
        _elite_size=2,
        _expected_path_length=float64(1.0),
        fitness_threshold=-inf,
        _lr_cov=0.1,
        _lr_mean=1.0,
        _lr_rank_one=float64(0.1),
        _lr_rank_mu=float64(0.1),
        _lr_sigma=float64(0.1),
        maximum_iterations=100,
        maximum_wall_time=10,
        _mu_eff=float64(1.5),
        _number_of_variables=ndim,
        _pop_size=len(fitness),
        sigma_threshold=1e-8,
        tolerance=1e-6,
        _update_interval=1,
        _weights=array([0.7, 0.3]),
        # Iteration state
        _opt_iter=1,
        _mean=zeros(ndim),
        _population=zeros((len(fitness), ndim)),
        _cov=None,
        _core=array(core),
        _sigma=0.5,
        _gamma=None,
        _fitness=array(fitness),
        _mean_squared_bound_errors=errors,
        _result={'optimal_value': float64(min(fitness))},
        _path_sigma=ones(ndim),
        _path_cov=zeros(ndim),
        )


def test_fake_fitnessStatistics():
    monitor = MonitorCMAES()
    monitor.base(make_fake_solver(fitness=(3.0, 1.0, 2.0)))

    assert monitor.data['best_fitness'] == [1.0]
    assert monitor.data['mean_fitness'] == [2.0]
    assert monitor.data['worst_fitness'] == [3.0]


# Bound violations are 0.0 while the solver has not computed any errors
# (unbounded problem), otherwise their maximum and mean are recorded
@pytest.mark.parametrize('errors, max_viol, mean_viol', [
    (None, 0.0, 0.0),
    (array([0.0, 0.5, 0.25]), 0.5, 0.25)])
def test_fake_boundViolations(errors, max_viol, mean_viol):
    monitor = MonitorCMAES()
    monitor.base(make_fake_solver(errors=errors))

    assert monitor.data['max_bound_viol'] == [max_viol]
    assert monitor.data['mean_bound_viol'] == [pytest.approx(mean_viol)]


# A singular covariance matrix must not produce inf/nan in the condition
# number (guarded by the +1e-12 in the denominator)
def test_fake_conditionNumberSingular():
    monitor = MonitorCMAES()
    monitor.base(make_fake_solver(core=(2.0, 0.0)))

    assert monitor.data['cov_cn'] == [pytest.approx(2.0 / 1e-12)]
    assert monitor.data['cov_spectr_norm'] == [2.0]


# Data is only recorded every 'interval' calls, static parameters only once
def test_fake_interval():
    monitor = MonitorCMAES(interval=3)
    solver = make_fake_solver()
    for iteration in range(1, 8):
        solver._opt_iter = iteration
        monitor.base(solver)

    assert monitor.data['iteration'] == [3, 6]
    assert monitor.data['pop_size'] == [3]


# Arrays are updated in place by the solvers, so the monitor must store
# copies instead of references
def test_fake_recordsCopies():
    monitor = MonitorCMAES()
    solver = make_fake_solver()
    monitor.full(solver)
    solver._mean[:] = 1.0
    solver._core[:] = 0.0

    assert (monitor.data['mean'][0] == 0.0).all()
    assert (monitor.data['cov_svs'][0] == array([4.0, 1.0])).all()


# Tests with the real solver-monitor pairs

def run_monitored(solver_class, monitor_class, callback='full', interval=1,
                  bounded=False):
    """Run a short optimization on Sphere with a monitor callback."""
    problem = Sphere(NDIM)
    bounds = (
        {'lower_variable_bounds': problem.bounds[0],
         'upper_variable_bounds': problem.bounds[1]}
        if bounded else {})
    with monitor_class(interval=interval) as monitor:
        solver = solver_class(
            number_of_variables=NDIM,
            objective=problem.__call__,
            maximum_iterations=NUM_ITER,
            min_log_level='critical',
            callback=getattr(monitor, callback),
            random_state=42,
            **bounds)
        result = solver.optimize(full(NDIM, 2.0))
    return solver, monitor, result


@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_recordedKeys(solver_class, monitor_class):
    _, base_monitor, _ = run_monitored(
        solver_class, monitor_class, callback='base')
    _, full_monitor, _ = run_monitored(
        solver_class, monitor_class, callback='full')

    assert SERIES_KEYS <= set(base_monitor.data)
    assert not FULL_KEYS & set(base_monitor.data)
    assert SERIES_KEYS | FULL_KEYS <= set(full_monitor.data)


@pytest.mark.parametrize('interval', [1, 3])
@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_seriesLengths(solver_class, monitor_class, interval):
    solver, monitor, _ = run_monitored(
        solver_class, monitor_class, interval=interval)
    data = monitor.data

    # One entry per recorded iteration
    iterations = list(range(interval, NUM_ITER + 1, interval))
    assert data['iteration'] == iterations
    for key in SERIES_KEYS | FULL_KEYS:
        assert len(data[key]) == len(iterations), key

    # Static parameters are recorded once and match the solver
    assert data['pop_size'] == [solver._pop_size]
    assert data['number_of_variables'] == [NDIM]


@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_valueSanity(solver_class, monitor_class):
    _, monitor, result = run_monitored(solver_class, monitor_class)
    data = monitor.data

    for best, avg, worst in zip(
            data['best_fitness'], data['mean_fitness'], data['worst_fitness']):
        assert best <= avg <= worst

    # The best-so-far value never increases and ends at the final result
    values = data['optimal_value']
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] == result['optimal_value']

    assert all(sigma > 0.0 for sigma in data['sigma'])

    # Condition numbers are >= 1, up to tolerance
    assert all(cn >= 1.0 - 1e-9 for cn in data['cov_cn'])


@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_boundHandling(solver_class, monitor_class):
    _, unbounded, _ = run_monitored(solver_class, monitor_class)
    _, bounded, _ = run_monitored(solver_class, monitor_class, bounded=True)

    # Without bounds, no penalty factor and no violations are recorded
    assert all(gamma is None for gamma in unbounded.data['gamma'])
    assert set(unbounded.data['max_bound_viol']) == {0.0}

    # With bounds, the penalty factor is initialized after the first iteration
    assert all(gamma is not None for gamma in bounded.data['gamma'][1:])


@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_meanHistory(solver_class, monitor_class):
    solver, monitor, _ = run_monitored(solver_class, monitor_class)
    means = monitor.data['mean']
    changes = monitor.data['mean_change_norm']

    # Each entry is an independent copy of the solver's mean
    solver._mean[:] = 123.0
    assert not (means[-1] == 123.0).any()
    assert (means[0] != means[-1]).any()

    # The change norms match the recorded means
    for i in range(1, len(means)):
        assert changes[i] == pytest.approx(norm(means[i] - means[i-1]))


# Known fail: the monitor only sees the mean after the first iteration, so
# the first change norm is always 0 instead of the step from the start point
@pytest.mark.xfail(strict=True, reason='initial mean is not recorded')
def test_firstMeanChange():
    _, monitor, _ = run_monitored(CMAES, MonitorCMAES)

    assert monitor.data['mean_change_norm'][0] > 0.0


# Tests of the context manager and the interactive mode

class FakeVisualizer:
    """Replace the matplotlib visualizer, recording the update calls."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.update_calls = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)


@pytest.fixture
def show_calls(monkeypatch):
    """Replace plt.show so no test opens a window, recording the calls."""
    calls = []
    monkeypatch.setattr(
        'matplotlib.pyplot.show', lambda **kwargs: calls.append(kwargs))
    return calls


@pytest.mark.parametrize('monitor_class', [p[1] for p in PAIRS], ids=PAIR_IDS)
def test_contextManager(monitor_class, show_calls):
    with monitor_class() as monitor:
        assert isinstance(monitor, monitor_class)

    # Exceptions inside the context are not swallowed
    with pytest.raises(ValueError):
        with monitor_class():
            raise ValueError

    # Silent mode never creates a visualizer or shows a plot
    assert monitor.visualizer is None
    assert show_calls == []


@pytest.mark.parametrize('solver_class, monitor_class', PAIRS, ids=PAIR_IDS)
def test_interactiveMode(solver_class, monitor_class, show_calls,
                         monkeypatch):
    monkeypatch.setattr(
        sys.modules[monitor_class.__module__], 'Visualizer', FakeVisualizer)

    problem = Sphere(NDIM)
    with monitor_class(interval=2, mode='interactive',
                       plot_bounds=((-5, -5), (5, 5))) as monitor:
        solver = solver_class(
            number_of_variables=NDIM,
            objective=problem.__call__,
            maximum_iterations=NUM_ITER,
            min_log_level='critical',
            callback=monitor.base,
            random_state=42)
        solver.optimize(full(NDIM, 2.0))

        # The visualizer is created once and updated per recorded iteration
        assert isinstance(monitor.visualizer, FakeVisualizer)
        assert monitor.visualizer.init_kwargs['dimensions'] == NDIM
        assert [call['iteration'] for call in monitor.visualizer.update_calls
                ] == list(range(2, NUM_ITER + 1, 2))
        assert show_calls == []

    # The plot is held open once when leaving the context
    assert show_calls == [{'block': True}]
