# Unit tests for the numerical safeguards of the optimizers
# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

import pytest
from numpy import diag, eye, full, inf, isfinite, isnan, maximum, nan
from numpy.linalg import eigvalsh, norm

from seamaze.benchmarks import (
    BentCigar, Discus, Ellipsoid, RotatedEllipsoid, Sphere)
from seamaze.optimizers import CMAES, DLRCMAES, LMMAES

OPTIMIZERS = [CMAES, LMMAES, DLRCMAES]
OPTIMIZER_IDS = ['CMAES', 'LMMAES', 'DLRCMAES']

# Problem dimension and number of iterations per run
NDIM = 10
NUM_ITER = 300


class InvariantChecker:
    """
    Callback checking the safeguarded solver quantities after every
    iteration, recording the first violation of each invariant.
    """

    def __init__(self):
        self.violations = {}

    def _check(self, name, condition, value):
        if not condition and name not in self.violations:
            self.violations[name] = f'iteration {self.iteration}: {value}'

    def __call__(self, solver):
        self.iteration = solver._opt_iter

        # Step size is finite and not below its lower clip value
        self._check('sigma', isfinite(solver._sigma)
                    and solver._sigma >= 1e-15, solver._sigma)
        self._check('mean finite', isfinite(solver._mean).all(), solver._mean)
        self._check('population finite', isfinite(solver._population).all(),
                    'non-finite individual')

        # Penalty factor stays within its clip range
        if solver._gamma is not None:
            self._check('gamma', 1e-5 <= solver._gamma <= 1e10, solver._gamma)

        # Population is repaired into the bounds
        if solver._is_bound:
            self._check(
                'population in bounds',
                (solver._population >= solver.lower_variable_bounds).all()
                and (solver._population <= solver.upper_variable_bounds).all(),
                'individual outside the bounds')

        if isinstance(solver, CMAES):
            self._check_cmaes(solver)
        elif isinstance(solver, LMMAES):
            self._check('memory finite', isfinite(solver._memory).all(),
                        'non-finite memory entry')
        elif isinstance(solver, DLRCMAES):
            self._check_dlrcmaes(solver)

    def _check_cmaes(self, solver):
        cov = solver._cov

        # Eigenvalues are clipped from below, the covariance is symmetric
        # positive definite and the eigenbasis orthonormal
        self._check('core', solver._core.min() >= 1e-15, solver._core.min())
        self._check('cov symmetric', norm(cov - cov.T) <= 1e-12 * norm(cov),
                    norm(cov - cov.T))
        self._check('cov positive definite', eigvalsh(cov).min() > 0.0,
                    eigvalsh(cov).min())
        basis = solver._basis
        self._check('basis orthonormal',
                    norm(basis.T @ basis - eye(basis.shape[1])) <= 1e-8,
                    norm(basis.T @ basis - eye(basis.shape[1])))

    def _check_dlrcmaes(self, solver):
        basis = solver._basis

        # Rank stays within its limits, the residual variances are
        # non-negative and the low-rank basis orthonormal
        self._check('rank',
                    1 <= solver.rank <= solver._low_rank_max_dimension,
                    solver.rank)
        self._check('psi', solver._psi.min() >= 0.0, solver._psi.min())
        self._check('basis orthonormal',
                    norm(basis.T @ basis - eye(basis.shape[1])) <= 1e-8,
                    norm(basis.T @ basis - eye(basis.shape[1])))

        # The covariance actually used for sampling and step-size adaptation
        # (with clipped core values) is positive semidefinite
        cov = (basis * maximum(solver._core, 0.0)) @ basis.T + diag(solver._psi)
        eigenvalues = eigvalsh(cov)
        self._check('effective cov positive semidefinite',
                    eigenvalues.min() >= -1e-12 * eigenvalues.max(),
                    eigenvalues.min())


def run_checked(optimizer, objective, start, **kwargs):
    """Run a seeded optimization with the invariant checker as callback."""
    checker = InvariantChecker()
    solver = optimizer(
        number_of_variables=NDIM,
        objective=objective,
        maximum_iterations=NUM_ITER,
        min_log_level='critical',
        callback=checker,
        random_state=42,
        **kwargs)
    result = solver.optimize(full(NDIM, start))
    return result, checker


def bounds(lower, upper):
    """Build the bound keyword arguments for a box [lower, upper]^NDIM."""
    return {'lower_variable_bounds': full(NDIM, lower),
            'upper_variable_bounds': full(NDIM, upper)}


# "Bad" but valid inputs: (objective, start value, solver keyword arguments)
SCENARIOS = {
    # Severely ill-conditioned or non-separable landscapes (cond. 1e6)
    'ill_conditioned_ellipsoid': (Ellipsoid(NDIM), 3.0, {}),
    'ill_conditioned_bent_cigar': (BentCigar(NDIM), 3.0, {}),
    'ill_conditioned_discus': (Discus(NDIM), 3.0, {}),
    'rotated_ellipsoid': (RotatedEllipsoid(NDIM), 3.0, {}),
    # Extreme step sizes
    'tiny_initial_sigma': (Sphere(NDIM), 3.0, {'initial_sigma': 1e-6}),
    'huge_initial_sigma_bounded': (
        Sphere(NDIM), 3.0, {'initial_sigma': 1e3, **bounds(-5.0, 5.0)}),
    # Difficult bound situations
    'start_far_outside_bounds': (Sphere(NDIM), 100.0, bounds(-5.0, 5.0)),
    'optimum_on_narrow_box_bound': (Sphere(NDIM), 1.0, bounds(1.0, 2.0)),
    # Degenerate objectives
    'flat_objective': (lambda x: 1.0, 3.0, {}),
    'huge_objective_scale': (lambda x: 1e150 * float(x @ x), 3.0, {}),
    # Smallest sensible population
    'minimal_population': (Sphere(NDIM), 3.0, {'number_of_individuals': 3}),
    }


# Check that all safeguarded quantities stay valid in every iteration and
# that no numerical warnings (overflow, invalid values) occur
@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('optimizer', OPTIMIZERS, ids=OPTIMIZER_IDS)
def test_invariantsUnderBadInput(optimizer, scenario):
    objective, start, kwargs = SCENARIOS[scenario]

    _, checker = run_checked(optimizer, objective, start, **kwargs)

    assert checker.violations == {}


# Known issue: the raw DLR-CMA-ES core values can become negative, so the covariance update itself
# does not preserve positive semidefiniteness
@pytest.mark.xfail(strict=False, reason='core update can produce negatives')
def test_dlrcmaesRawCoreNonNegative():
    min_core = []
    solver = DLRCMAES(
        number_of_variables=NDIM,
        objective=BentCigar(NDIM),
        maximum_iterations=NUM_ITER,
        min_log_level='critical',
        callback=lambda s: min_core.append(s._core.min()),
        random_state=42)
    solver.optimize(full(NDIM, 3.0))

    assert min(min_core) >= 0.0


# Check that stopping before the first iteration (no optimal point recorded
# yet) returns a result instead of crashing
@pytest.mark.parametrize('kwargs, solver_info', [
    ({'initial_sigma': 1e-12}, 'SIGMA_BELOW_THRESH'),
    ({'maximum_iterations': 0}, 'MAX_ITER_REACHED')],
    ids=['sigma_below_threshold', 'zero_iterations'])
@pytest.mark.parametrize('optimizer', OPTIMIZERS, ids=OPTIMIZER_IDS)
def test_stopBeforeFirstIteration(optimizer, kwargs, solver_info):
    solver = optimizer(
        number_of_variables=NDIM,
        objective=Sphere(NDIM),
        min_log_level='critical',
        **kwargs)
    result = solver.optimize(full(NDIM, 3.0))

    assert result['iterations'] == 0
    assert result['solver_info'] == solver_info


# Check that objectives returning inf or NaN (e.g. for infeasible points) are
# handled: NaN never becomes the optimum, and no invariant or warning is hit
@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('objective, kwargs', [
    (lambda x: float(x @ x) if x @ x < 1.0 else inf, {}),
    (lambda x: float(x @ x) if x @ x < 1.0 else inf, bounds(-5.0, 5.0)),
    (lambda x: float(x @ x) if x[0] >= 0.0 else nan, {})],
    ids=['inf_outside_unit_ball', 'inf_outside_unit_ball_bounded',
         'nan_for_negative_x0'])
@pytest.mark.parametrize('optimizer', OPTIMIZERS, ids=OPTIMIZER_IDS)
def test_nonFiniteObjective(optimizer, objective, kwargs):
    result, checker = run_checked(optimizer, objective, 0.5, **kwargs)

    # The recorded optimum is never NaN, and a finite optimum has a point
    assert not isnan(result['optimal_value'])
    if isfinite(result['optimal_value']):
        assert result['optimal_point'] is not None
    assert checker.violations == {}
