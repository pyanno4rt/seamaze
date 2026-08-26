"""
Dynamical low-rank covariance matrix adaptation evolution strategy
(DLR-CMA-ES).
"""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from signal import getsignal, SIGINT, signal
from time import time
import warnings

from collections import deque
from math import inf
from numba import njit, types
from numba.core.errors import NumbaPerformanceWarning
from numpy import (
    add, arange, argmin, argsort, array, ascontiguousarray, asfortranarray,
    ceil, clip, diag, exp, eye, finfo, float64, full, isinf, log, maximum,
    minimum, ptp, ones, outer, sort, sqrt, where, zeros)
from numpy import abs as nabs
from numpy import max as nmax
from numpy import mean as nmean
from numpy import min as nmin
from numpy import sum as nsum
from numpy.linalg import eigh, norm, qr, solve
from numpy.random import default_rng, randn

# %% Internal package import

from seamaze.logging import Logging
from seamaze.utils import make_compat

# %% Disable numba warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# %% Class definition


class DLRCMAES:
    """
    Dynamical low-rank covariance matrix adaptation evolution strategy \
    (DLR-CMA-ES) class.

    This class implements the novel DLR-CMA-ES algorithm, introduced in our
    upcoming paper:

        Ortkamp, T., Patwardhan, C. and Stammer, P. (2026). A dynamical
        low-rank covariance matrix adaptation hybrid evolution strategy for
        computationally efficient large-scale constrained optimization.

    Here, the covariance matrix is represented as a sum of a low-rank component
    and a residual-variance component:

        C = USU^T + diag(psi),

    where USU^T captures the learned dependency structure, while psi accounts
    for residual marginal variances.

    Parameters
    ----------
    number_of_variables : int
        Dimension of the search space (number of decision variables).

    objective : Callable[[ndarray], float]
        The objective function to be minimized. Must accept a 1D ``ndarray``
        and return a scalar ``float``.

    gradient : Callable[[ndarray], ndarray], optional
        Optional gradient function used for hybrid evolution steps.

    lower_variable_bounds : ndarray, default=None
        Lower bounds on the decision variables. Must be a 1D array of length
        `number_of_variables`. Defaults to -inf for all variables.

    upper_variable_bounds : ndarray, default=None
        Upper bounds on the decision variables. Must be a 1D array of length
        `number_of_variables`. Defaults to +inf for all variables.

    number_of_individuals : int, default=None
        Population size. Defaults to 4 + int(3*log(`number_of_variables`)).

    initial_sigma : float, default=1.0
        Initial step size.

    low_rank_init_dimension : int, default=None
        Initial rank of the approximation. Defaults to 1.

    low_rank_max_dimension : int, default=None
        Maximum rank of the approximation. Defaults to `number_of_variables`.

    low_rank_is_adaptive : bool, default=True
        Indicator for adaptive low-rank selection.

    low_rank_energy_tolerance : float, default=1e-2
        Maximum fraction of discarded low-rank energy for rank truncation.

    maximum_iterations : int, default=100000
        Maximum number of generations (iterations) to run before stopping.

    maximum_wall_time : int or float, default=43200
        Maximum allowed wall-clock time in seconds.

    fitness_threshold : int or float, default=-inf
        Target fitness value. If the objective value reaches this threshold,
        optimization stops (success criterion).

    fitness_window_size : int, default=50
        Number of past iterations to consider for the fitness range
        stagnation check.

    tolerance : float, default=1e-6
        Absolute and relative termination tolerance: stops if the change in
        fitness range over `fitness_window_size` is below this value.

    sigma_threshold : float, default=1e-8
        Minimum allowed step size. If the step size falls below this limit, \
        optimization stops (convergence/collapse criterion).

    update_interval : int, default=None
        Frequency of the covariance update (in generations). Larger values
        (e.g. 10) can significantly speed up the algorithm for
        high-dimensional problems. Defaults to

            ceil(sqrt(rank/(n*lr_cov))),

        where `rank` is the current low-rank dimension, `n` is the problem
        dimension, and `lr_cov` is the covariance learning rate.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}, \
        default='debug'
        Minimum logging level for passing messages to the console.

    callback : Callable[[DLRCMAES], None], default=None
        Optional function called at the end of each iteration. Must accept
        the solver instance.

    random_state : int, default=42
        Control seed for the internal random number generator.
    """

    def __init__(
            self,
            number_of_variables,
            objective,
            gradient=None,
            lower_variable_bounds=None,
            upper_variable_bounds=None,
            number_of_individuals=None,
            initial_sigma=1.0,
            low_rank_init_dimension=None,
            low_rank_max_dimension=None,
            low_rank_is_adaptive=True,
            low_rank_energy_tolerance=1e-2,
            maximum_iterations=100000,
            maximum_wall_time=43200,
            fitness_threshold=-inf,
            fitness_window_size=50,
            tolerance=1e-6,
            sigma_threshold=1e-8,
            update_interval=None,
            min_log_level='debug',
            callback=None,
            random_state=42):

        # Initialize the logger
        self.logger = Logging('DLR-CMA-ES', min_log_level)

        # Log a message about the initialization
        self.logger.info('Initializing DLR-CMA-ES ...')

        # Check if no random state has been passed
        if random_state is None:

            # Initialize the default RNG
            self._rng = default_rng()

        else:

            # Initialize the RNG with control seed
            self._rng = default_rng(int(random_state))

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = make_compat(objective)
        self.gradient = gradient
        self.lower_variable_bounds = (
            full(self._number_of_variables, -inf)
            if lower_variable_bounds is None
            else array(lower_variable_bounds, dtype=float64)
            )
        self.upper_variable_bounds = (
            full(self._number_of_variables, inf)
            if upper_variable_bounds is None
            else array(upper_variable_bounds, dtype=float64)
            )

        # Initialize the boundedness indicator
        self._is_bound = (
            ~isinf(self.lower_variable_bounds).all() or
            ~isinf(self.upper_variable_bounds).all()
            )

        # Initialize the population size
        self._pop_size = (
            4 + int(3 * log(self._number_of_variables))
            if number_of_individuals is None else number_of_individuals
            )
        self._pop_size += (2 if self.gradient is not None else 0)

        # Initialize the base weights
        base_weights = (
            log((self._pop_size + 1) / 2) - log(arange(1, self._pop_size + 1))
            ).reshape(-1, 1)

        # Initialize the elite size
        self._elite_size = int(nsum(base_weights > 0))

        # Get the low-rank adaptivity parameters
        default_rank = 1

        self._low_rank_max_dimension = (
            self._number_of_variables
            if low_rank_max_dimension is None
            else min(low_rank_max_dimension, self._number_of_variables)
            )

        self._low_rank_init_dimension = (
            default_rank
            if low_rank_init_dimension is None
            else low_rank_init_dimension
            )
        self._low_rank_init_dimension = min(
            max(1, self._low_rank_init_dimension),
            self._number_of_variables,
            self._low_rank_max_dimension
            )

        self._low_rank_is_adaptive = low_rank_is_adaptive
        self._low_rank_energy_tolerance = low_rank_energy_tolerance

        # Determine the integrator rank
        self.rank = self._low_rank_init_dimension

        # Determine the sums of positive and negative base weights
        bw_pos_sum = nsum(base_weights[:self._elite_size])
        bw_neg_sum = nsum(base_weights[self._elite_size:])

        # Initialize the variance effective selection mass
        self._mu_eff = bw_pos_sum**2 / nsum(base_weights[:self._elite_size]**2)
        mu_eff_neg = bw_neg_sum**2 / nsum(base_weights[self._elite_size:]**2)

        # Initialize the learning rates
        self._lr_sigma = (
            (self._mu_eff + 2) / (self._number_of_variables + self._mu_eff + 5)
            )
        self._lr_cov = (
            (4 + self._mu_eff / self._number_of_variables) /
            (self._number_of_variables + 4
             + 2 * self._mu_eff/self._number_of_variables)
            )
        self._lr_rank_one_base = (
            2 / ((self._number_of_variables + 1.3)**2 + self._mu_eff)
            )
        self._lr_rank_mu_base = min(
            1.0 - self._lr_rank_one_base,
            2.0 * (
                (0.25 + self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                ((self._number_of_variables + 2.0)**2
                 + 2.0 * self._mu_eff / 2.0)
                )
            )
        self._lr_rank_one, self._lr_rank_mu = self._get_learning_rates()
        self._lr_mean = 1.0

        # Determine the alpha values
        alpha_mu_neg = 1.0 + self._lr_rank_one / (self._lr_rank_mu + 1e-12)
        alpha_mu_eff_neg = 1.0 + (2.0 * mu_eff_neg) / (self._mu_eff + 2.0)
        alpha_posdef_neg = (
            (1.0 - self._lr_rank_one - self._lr_rank_mu)
            / (self._number_of_variables * self._lr_rank_mu + 1e-12)
            )
        alpha_min = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_posdef_neg)

        # Set the weights
        self._weights = where(
            base_weights > 0,
            (1.0 / bw_pos_sum) * base_weights,
            (alpha_min / (abs(bw_neg_sum) + 1e-12)) * base_weights
            ).reshape(-1, 1)

        # Initialize the damping coefficient
        self._damp_sigma = (
            1.0 + 2.0 * max(
                0.0,
                sqrt((self._mu_eff - 1.0)/(self._number_of_variables + 1.0))
                - 1.0
                )
            + self._lr_sigma
            )

        # Initialize the expected path length
        self._expected_path_length = (
            sqrt(self._number_of_variables) * (
                1.0
                - 1.0 / (4.0 * self._number_of_variables)
                + 1.0 / (21.0 * self._number_of_variables**2))
            )

        # Initialize the bound/constraint handling parameters
        self._mean_squared_bound_errors = None
        self._gamma = None

        # Initialize the adaptive state variables, arrays, and matrices
        self._wall_start = None
        self._opt_iter = 0
        self._sigma = initial_sigma

        self._steps = zeros(
            (self._pop_size, self._number_of_variables), dtype=float64
            )
        self._population = zeros(
            (self._pop_size, self._number_of_variables), dtype=float64
            )

        self._path_sigma = zeros(self._number_of_variables, dtype=float64)
        self._path_cov = zeros(self._number_of_variables, dtype=float64)
        self._mean = zeros(self._number_of_variables, dtype=float64)

        self._basis = eye(
            self._number_of_variables, self.rank, dtype=float64
            )
        self._core = ones(self.rank, dtype=float64)

        self._psi = ones(self._number_of_variables, dtype=float64)
        self._psi[:self.rank] *= 0.0

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness = None
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._update_interval_user = update_interval
        self._update_interval = (
            self._get_update_interval() if update_interval is None
            else update_interval
            )
        self._current_expansion_reasons = []
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None, 'iterations': None
            }

        # Initialize the stop flag
        self._stop_requested = False

    def _get_learning_rates(self):
        """
        Get the learning rates depending on the rank.

        Returns
        -------
        float
            Rank-1 learning rate.

        float
            Rank-mu learning rate.
        """

        # Determine the scaling factor
        beta = 0.0
        factor = (self._number_of_variables / self.rank) ** beta

        return factor * self._lr_rank_one_base, factor * self._lr_rank_mu_base

    def _get_update_interval(self):
        """
        Get the update interval depending on the rank.

        Returns
        -------
        int
            Update interval.
        """

        return ceil(sqrt(
            self.rank / (self._number_of_variables * self._lr_cov)
            ))

    def ask(self):
        """Generate a new population."""

        # Sample from the standard multivariate Gaussian
        num_random = (
            self._pop_size - 2 if self.gradient is not None
            else self._pop_size
            )
        z_low_rank = self._rng.standard_normal((num_random, self.rank))
        z_noise = self._rng.standard_normal(
            (num_random, self._number_of_variables)
            )

        # Calculate the root of the covariance matrix
        sqrt_core = sqrt(maximum(self._core, 0.0))

        # Transform the samples into low-rank and noisy components
        structured_part = (z_low_rank * sqrt_core) @ self._basis.T
        noise_part = z_noise * sqrt(self._psi)

        # Get the steps from the split-Gaussian sampling
        self._steps[:num_random] = structured_part + noise_part

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            low_rank_gradient = (
                self._basis @ (self._core * (self._basis.T @ gradient))
                )
            natural_gradient = low_rank_gradient + self._psi * gradient

            # Compute the rescaling factor
            rescale = 1.0 / (sqrt(gradient @ natural_gradient) + 1e-15)

            # Compute the natural gradient step
            gradient_step = natural_gradient * rescale

            # Add the mirrored gradient steps
            self._steps[-2] = -gradient_step
            self._steps[-1] = gradient_step

        # Sample the new population
        add(self._mean, self._sigma * self._steps, out=self._population)

        # Check if the decision variables are bounded
        if self._is_bound:

            # Compute the distance to the lower and upper bounds
            eps_lower = maximum(
                0.0, self.lower_variable_bounds - self._population
                )
            eps_upper = maximum(
                0.0, self._population - self.upper_variable_bounds
                )

            # Compute the squared total bound errors
            bound_errors_squared = (eps_lower + eps_upper) ** 2

            # Average the squared errors for each individual
            self._mean_squared_bound_errors = nmean(
                bound_errors_squared, axis=1)

            # Compute the relative severity of bound violations
            violation_severity = nmean(
                nsum(bound_errors_squared, axis=1) /
                (nsum(bound_errors_squared, axis=1) +
                 nsum((self._sigma * self._steps) ** 2, axis=1) +
                 1e-15
                 )
                ) ** 2

            # Check if the penalty factor has been initialized
            if self._gamma is not None:

                # Compute the gamma factor
                gamma_factor = (
                    1.001 ** violation_severity *
                    0.999 ** (1.0 - violation_severity)
                    )

                # Adapt the penalty factor
                self._gamma = clip(self._gamma * gamma_factor, 1e-5, 1e10)

            # Get the relative step size
            sigma_rel = self._sigma / (1 + self._sigma)

            # Mirror the violating individuals back into the feasible region
            self._population[:] = where(
                self._population < self.lower_variable_bounds,
                self.lower_variable_bounds + sigma_rel * eps_lower,
                self._population
                )
            self._population[:] = where(
                self._population > self.upper_variable_bounds,
                self.upper_variable_bounds - sigma_rel * eps_upper,
                self._population
                )

            # Enforce hard constraints to avoid numerical round-off errors
            clip(self._population, a_min=self.lower_variable_bounds,
                 a_max=self.upper_variable_bounds, out=self._population)

    def evaluate(self):
        """
        Evaluate the fitness of the population and track the global optimum.

        Returns
        -------
        ndarray
            Unpenalized fitness values for the population.

        ndarray
            Penalized fitness values for the population.
        """

        # Compute the unpenalized fitness values
        true_fitness = array([
            self.objective(individual, track=False)
            for individual in self._population
            ])

        # Check if the decision variables are bounded
        if self._is_bound:

            # Check if the penalty factor has not been initialized
            if self._gamma is None:

                # Compute the unpenalized fitness range
                fitness_range = nmax(true_fitness) - nmin(true_fitness)

                # Scale the penalty factor to the fitness range
                self._gamma = (
                    fitness_range if fitness_range > 1e-8 else 1.0
                    )

            # Compute the penalized fitness values
            selection_fitness = (
                true_fitness + self._gamma * self._mean_squared_bound_errors
                )

        else:

            # Apply unpenalized fitness values for selection
            selection_fitness = true_fitness

        # Get the best unpenalized fitness
        best_index = argmin(true_fitness)
        true_best_fitness = true_fitness[best_index]

        # Append the best (unpenalized) fitness to the history
        self._fitness_history.append(true_best_fitness)

        # Check if an improved solution has been found
        if true_best_fitness < self._result['optimal_value']:

            # Update the optimal value and point
            self._result['optimal_value'] = true_best_fitness
            self._result['optimal_point'] = self._population[best_index].copy()

        # Re-evaluate the current best individual for tracking
        self.objective(self._result['optimal_point'])

        return true_fitness, selection_fitness

    def tell(
            self,
            fitness):
        """
        Update the state variables and perform an adaptive BUG step.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Update the state variables
        (path_sigma_new, mean_new, sigma_new, path_cov_new, update_switch
         ) = _tell(
            fitness,
            self._steps,
            self._weights,
            self._basis,
            self._core,
            self._psi,
            self._path_sigma,
            self._path_cov,
            self._mean,
            self._sigma,
            self._lr_sigma,
            self._lr_cov,
            self._lr_mean,
            self._mu_eff,
            self._damp_sigma,
            self._expected_path_length,
            self._opt_iter,
            self._elite_size
            )

        # Save the state variables
        self._path_sigma[:] = path_sigma_new
        self._mean[:] = mean_new
        self._sigma = sigma_new
        self._path_cov[:] = path_cov_new

        # Check if the covariance factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Get the current rank
            rank_current = self.rank

            # Get the steps sorted by fitness
            steps_sorted = self._steps[argsort(fitness)]

            # Update the covariance factors
            basis_new, core_new, psi_new, rank_new = _adaptive_bug_step(
                self._basis,
                self._core,
                self._psi,
                steps_sorted,
                self._weights,
                self._path_cov,
                self._lr_cov,
                self._lr_rank_one,
                self._lr_rank_mu,
                self._low_rank_max_dimension,
                self._low_rank_is_adaptive,
                self._low_rank_energy_tolerance,
                update_switch=update_switch,
                force_expansion=self.check_rank_expansion(
                    steps_sorted[:self._elite_size, :]
                    )
                )

            # Save the covariance state variables
            self._basis = basis_new
            self._core = core_new
            self._psi[:] = psi_new
            self.rank = rank_new

            # Check if the rank has changed
            if rank_new != rank_current:

                # Refresh the learning rates
                (self._lr_rank_one,
                 self._lr_rank_mu) = self._get_learning_rates()

                # Check if no update interval has been passed
                if self._update_interval_user is None:

                    # Refresh the update interval
                    self._update_interval = self._get_update_interval()

    def optimize(
            self,
            initial_mean=None):
        """
        Run the optimization algorithm.

        Parameters
        ----------
        initial_mean : ndarray, default=None
            Initial mean vector. Default corresponds to the zero vector.

        Returns
        -------
        dict
            Dictionary with the optimization results.
        """

        # Start the runtime recordings
        self._wall_start = time()

        # Check if an initial mean has been provided
        if initial_mean is not None:

            # Set the initial mean
            self._mean = initial_mean.astype(float)

        # Get the previous signal handler
        old_handler = getsignal(SIGINT)

        # Define a custom signal handler
        def _sigint_handler(_, __):
            self._stop_requested = True

        # Register the new handler
        signal(SIGINT, _sigint_handler)

        try:

            # Continue until termination criteria are fulfilled
            while not self.check_termination():

                # Check if a program stop has been requested
                if self._stop_requested:

                    # Log a message about the user stopping
                    self.logger.warning("Optimization interrupted by user ...")
                    self._result['solver_info'] = 'STOPPED_BY_USER'

                    break

                # Increment the iteration counter
                self._opt_iter += 1

                # "Ask" for a new population
                self.ask()

                # Evaluate the population's fitness
                self._fitness, selection_fitness = self.evaluate()

                # "Tell" the algorithm to update its parameters
                self.tell(selection_fitness)

                # Check if a callback has been provided
                if self._callback is not None:

                    # Pass the current results to the callback
                    self._callback(self)

                # Log a message about the current result
                self.logger.info(
                    f'Iteration {self._opt_iter}: '
                    f'f={round(self._result["optimal_value"], 6)} '
                    f'(r={self.rank})'
                    )

        finally:

            # Restore the original signal handler
            signal(SIGINT, old_handler)

        # Store the runtime and iterations
        self._result['wall_time'] = time() - self._wall_start
        self._result['iterations'] = self._opt_iter

        # Get the optimal point
        opt_point = self._result['optimal_point']

        # Check if the length is greater than 5
        if len(opt_point) > 5:

            # Format the first elements and append suspension points
            short_sol = (
                "[" + ", ".join(f"{x:.4f}" for x in opt_point[:5]) + " ...]"
                )

        else:

            # Format the full solution string
            short_sol = "[" + ", ".join(f"{x:.4f}" for x in opt_point) + "]"

        # Log a message about the optimization results
        self.logger.info(
            'Optimization finished | '
            f'Best value: {round(self._result["optimal_value"], 6)} | '
            f'Best solution: {short_sol} | '
            f'Iterations: {self._opt_iter} | '
            f'Wall-clock: {self._result["wall_time"]} seconds | '
            f'Solver info: "{self._result["solver_info"]}" ...'
            )

        return self._result

    def check_rank_expansion(
            self,
            elite_steps):
        """
        Check the evolutionary-state rank expansion criteria.

        Parameters
        ----------
        elite_steps : ndarray
            Elite mutation steps.

        Returns
        -------
        bool
            Indicator for rank expansion.
        """

        # Check if the current rank equals the number of variables
        if self.rank == self._number_of_variables:

            return False

        # Initialize the expansion pressure flags
        path_pressure = False
        steps_pressure = False
        fitness_pressure = False

        # Initialize the expansion reasons
        reasons = []

        # Get the expected path energy fraction from a random subspace
        expected_explained_ratio = max(
            self.rank / self._number_of_variables, 1e-6
            )

        # Project the evolution path into the current low-rank subspace
        path_cov_coords = self._basis.T @ self._path_cov

        # Calculate the explained and total path energy
        path_explained_energy = norm(path_cov_coords)**2
        path_total_energy = norm(self._path_cov)**2

        # Calculate the explained path energy fraction
        path_explained_ratio = (
            path_explained_energy / (path_total_energy + 1e-15)
            )

        # Get the ratio of explained versus expected energy
        path_ratio = path_explained_ratio / (expected_explained_ratio + 1e-15)

        # Check if the ratio is smaller than 80%
        if path_ratio < 0.8:

            # Set the path pressure flag
            path_pressure = True

            # Add the expansion reason
            reasons.append('path_excess_residual')

        # Project the elite steps into the current low-rank subspace
        elite_coords = elite_steps @ self._basis

        # Calculate the explained and total weighted elite step energies
        elite_explained_energy = nsum(
            self._weights[:self._elite_size]
            * nsum(elite_coords * elite_coords, axis=1)
            )
        elite_total_energy = nsum(
            self._weights[:self._elite_size]
            * nsum(elite_steps * elite_steps, axis=1)
            )

        # Calculate the explained elite step energy fraction
        elite_explained_ratio = (
            elite_explained_energy / (elite_total_energy + 1e-15)
            )

        # Get the ratio of explained versus expected energy
        elite_ratio = (
            elite_explained_ratio / (expected_explained_ratio + 1e-15)
            )

        # Check if the ratio is smaller than 80%
        if elite_ratio < 0.8:

            # Set the steps pressure flag
            steps_pressure = True

            # Add the expansion reason
            reasons.append('steps_excess_residual')

        # Check if the fitness history has been completely filled
        if len(self._fitness_history) == self._fitness_history.maxlen:

            # Get the fitness history
            history = array(self._fitness_history)

            # Split the history
            half = history.size // 2

            # Compute the average fitness
            fit_old = nmean(history[:half])
            fit_new = nmean(history[half:])

            # Get the fitness range
            fit_range = ptp(history)

            # Check if the fitness improvement is reasonably small
            if (fit_old - fit_new) / (fit_range + 1e-15) < 1e-2:

                # Set the fitness pressure flag
                fitness_pressure = True

                # Add the expansion reason
                reasons.append('fitness_stagnation')

        # Update the current expansion reasons
        self._current_expansion_reasons = reasons

        return (path_pressure + steps_pressure + fitness_pressure) >= 2

    def check_termination(self):
        """
        Check the termination criteria.

        Returns
        -------
        bool
            Indicator for termination.
        """

        # Check if the maximum number of iterations has been reached
        if self._opt_iter >= self.maximum_iterations:

            # Add the solver info
            self._result['solver_info'] = 'MAX_ITER_REACHED'

            return True

        # Check if the wall clock timer has been started
        if self._wall_start is not None:

            # Check if the maximum runtime has been reached
            if time()-self._wall_start >= self.maximum_wall_time:

                # Add the solver info
                self._result['solver_info'] = 'MAX_WALL_TIME_REACHED'

                return True

        # Check if the step size is below the threshold
        if self._sigma <= self.sigma_threshold:

            # Add the solver info
            self._result['solver_info'] = 'SIGMA_BELOW_THRESH'

            return True

        # Check if the condition number criterion should be re-evaluated
        if self._opt_iter % self._update_interval == 0:

            # Approximate the maximum and minimum eigenvalue
            max_eval, min_eval = _lanczos_spectrum_extremes(
                self._basis,
                self._core,
                self._psi
                )

            # Reference the floor and the condition-number bound off the
            # current largest eigenvalue and double-precision machine
            # epsilon, rather than fixed absolute constants -- a
            # legitimately well-adapted, highly anisotropic shape (small
            # psi relative to a large max_eval) should not be confused
            # with numerical breakdown of the shape matrix itself
            machine_eps = finfo(float64).eps
            eigenvalue_floor = max_eval * machine_eps

            # Check if any eigenvalue is (relatively) zero or the
            # condition number has reached the precision limit of float64
            if (max_eval <= 0.0
                    or min_eval < eigenvalue_floor
                    or max_eval / (min_eval + eigenvalue_floor)
                    >= 1.0 / machine_eps):

                # Add the solver info
                self._result['solver_info'] = 'MAX_COND_NUM_EXCEEDED'

                return True

        # Check if the optimal value is below a threshold
        if (self.fitness_threshold is not None
                and self._result['optimal_value'] < self.fitness_threshold):

            # Add the solver info
            self._result['solver_info'] = 'FITNESS_BELOW_THRESH'

            return True

        # Check if the history is completely filled
        if len(self._fitness_history) == self._fitness_history.maxlen:

            # Convert the history to a list
            history = array(self._fitness_history)

            # Get the fitness range
            fit_range = ptp(history)

            # Check if the absolute fitness range is below tolerance
            if fit_range < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'ABSOLUTE_FITNESS_PLATEAU_REACHED')

                return True

            # Check if the relative fitness range is below tolerance
            if fit_range / max(nmax(nabs(history)), 1.0) < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'RELATIVE_FITNESS_PLATEAU_REACHED')

                return True

        return False


# Set the variable types
f8_2d = types.float64[:, :]
f8_1d = types.float64[:]
i8_1d = types.int64[:]
f8 = types.float64
i8 = types.int64
bo = types.bool_

@njit(
    types.Tuple((f8, f8))(
        # Return: Tuple(max_eval, min_eval)
        f8_2d,          # basis
        f8_1d,          # core
        f8_1d,          # psi
        ),
    fastmath=True
    )
def _lanczos_spectrum_extremes(basis, core, psi):
    """Approximate the smallest and largest eigenvalues of C."""

    # Get the problem dimensionality
    dim = basis.shape[0]

    # Determine the Krylov subspace dimension
    krylov_dim = min(dim, 20)

    # Initialize the Lanczos coefficients
    alpha = zeros(krylov_dim, dtype=float64)
    beta = zeros(krylov_dim, dtype=float64)

    # Initialize the Krylov basis
    krylov_basis = zeros((krylov_dim, dim)).T

    # Start from a random normalized direction
    vector = randn(dim)
    vector_norm = norm(vector)

    # Check if the norm is sufficiently small
    if vector_norm < 1e-15:

        # Fall back to the diagonal component
        return nmax(psi), nmin(psi)

    # Normalize the initial Krylov vector
    krylov_basis[:, 0] = vector / vector_norm

    # Track the current Krylov subspace dimensionality
    current_dim = krylov_dim

    # Loop over the dimensions of the Krylov subspace
    for index in range(krylov_dim):

        # Get the current Krylov vector
        current_basis = krylov_basis[:, index]

        # Apply the covariance matrix to the current direction
        update_vec = (
            basis @ (core * (basis.T @ current_basis)) + psi * current_basis
            )

        # Perfom full reorthogonalization twice
        for _ in range(2):

            # Loop until the current dimension of the Krylov subspace
            for sub in range(index + 1):

                # Get the i-th Krylov vector
                sub_basis = krylov_basis[:, sub]

                # Project the next vector onto the Krylov vector
                projection = sub_basis.T @ update_vec

                # Remove the projected component
                update_vec -= projection * sub_basis

                # Check if the penultimate dimension has been reached
                if _ == 0 and sub == index:

                    # Store the current diagonal coefficient
                    alpha[index] = projection

        # Check if the final Krylov vector has been reached
        if index < krylov_dim - 1:

            # Compute the norm of the residual vector
            beta_next = norm(update_vec)

            # Store the off-diagonal Lanczos coefficient
            beta[index + 1] = beta_next

            # Check for "happy breakdown" (exact solution found)
            if beta_next < 1e-12:

                # Set the current dimensionality to the final index
                current_dim = index + 1

                break

            # Normalize and store the next Krylov vector
            krylov_basis[:, index + 1] = update_vec / beta_next

    # Construct the symmetric tridiagonal Lanczos matrix
    tridiagonal = diag(alpha[:current_dim])

    # Check if the current Krylov dimension is larger than one
    if current_dim > 1:

        # Fill the first upper and lower subdiagonals
        tridiagonal += (
            diag(beta[1:current_dim], k=1) + diag(beta[1:current_dim], k=-1)
            )

    # Perform an eigendecomposition of the projected matrix
    tridiagonal_evals, _ = eigh(tridiagonal)

    # Get the smallest and largest Ritz eigenvalues
    min_eval = tridiagonal_evals[0]
    max_eval = tridiagonal_evals[-1]

    # Prevent numerical errors
    min_eval = max(min_eval, 1e-15)

    # Keep the estimated spectrum consistent
    max_eval = max(max_eval, min_eval)

    return max_eval, min_eval


@njit(
    f8_1d(
        # Return: approximation
        f8_1d,          # elite_mean_step
        f8_2d,          # basis
        f8_1d,          # core
        f8_1d,          # psi
        ),
    fastmath=True
    )
def _lanczos_inverse_sqrt_product(elite_mean_step, basis, core, psi):
    """Approximate the inverse matrix square root-vector product C^(-1/2)*x."""

    # Get the norm of the elite mean step
    norm_step = norm(elite_mean_step)

    # Check if the norm is sufficiently small
    if norm_step < 1e-15:

        # Return the zero vector
        return zeros(elite_mean_step.shape[0], dtype=float64)

    # Get the problem dimensionality
    dim = basis.shape[0]

    # Determine the Krylov subspace dimension
    krylov_dim = min(dim, 20)

    # Initialize the Lanczos coefficients
    alpha = zeros(krylov_dim, dtype=float64)
    beta = zeros(krylov_dim, dtype=float64)

    # Initialize the Krylov basis
    krylov_basis = zeros((krylov_dim, dim)).T

    # Normalize the initial Krylov vector
    krylov_basis[:, 0] = elite_mean_step / norm_step

    # Track the current Krylov subspace dimensionality
    current_dim = krylov_dim

    # Loop over the dimensions of the Krylov subspace
    for index in range(krylov_dim):

        # Get the current Krylov vector
        current_basis = krylov_basis[:, index]

        # Generate the next Krylov vector
        update_vec = (
            basis @ (core * (basis.T @ current_basis)) + psi * current_basis
            )

        # Perfom full reorthogonalization twice
        for _ in range(2):

            # Loop until the current dimension of the Krylov subspace
            for sub in range(index + 1):

                # Get the i-th Krylov vector
                sub_basis = krylov_basis[:, sub]

                # Project the next vector onto the Krylov vector
                projection = sub_basis.T @ update_vec

                # Remove the projected component
                update_vec -= projection * sub_basis

                # Check if the penultimate dimension has been reached
                if _ == 0 and sub == index:

                    # Store the current diagonal coefficient
                    alpha[index] = projection

        # Check if the final Krylov vector has been reached
        if index < krylov_dim - 1:

            # Compute the norm of the residual vector
            beta_next = norm(update_vec)

            # Store the off-diagonal Lanczos coefficient
            beta[index + 1] = beta_next

            # Check for "happy breakdown" (exact solution found)
            if beta_next < 1e-12:

                # Set the current dimensionality to the final index
                current_dim = index + 1

                break

            # Normalize and store the next Krylov vector
            krylov_basis[:, index + 1] = update_vec / beta_next

    # Construct the symmetric tridiagonal Lanczos matrix
    tridiagonal = diag(alpha[:current_dim])

    # Check if the current Krylov dimension is larger than one
    if current_dim > 1:

        # Fill the first upper and lower subdiagonals
        tridiagonal += (
            diag(beta[1:current_dim], k=1) + diag(beta[1:current_dim], k=-1)
            )

    # Perform an eigendecomposition on the projection matrix
    tridiagonal_evals, tridiagonal_evecs = eigh(tridiagonal)

    # Get the largest Ritz eigenvalue
    lambda_max = tridiagonal_evals[-1]

    # Apply a relative eigenvalue floor for numerical stability
    threshold = max(1e-15, 1e-12 * lambda_max)

    # Safeguard the projected spectrum against non-positive Ritz values
    tridiagonal_evals = maximum(tridiagonal_evals, threshold)

    # Calculate the inner vector (T^(-1/2) * e_1||x||)
    inv_sqrt_evals = 1.0 / sqrt(tridiagonal_evals)
    inner_vector = (
        tridiagonal_evecs @ (
            inv_sqrt_evals * (tridiagonal_evecs[0, :] * norm_step))
        )

    # Restrict the Krylov basis to the generated subspace
    krylov_slice = asfortranarray(krylov_basis[:, :current_dim])

    # Map the projected result back to the original space
    approximation = krylov_slice @ inner_vector

    return approximation


@njit(
    types.Tuple((f8_1d, f8_1d, f8, f8_1d, f8))(
        # Return: Tuple(path_sigma, mean, sigma, path_cov, update_switch)
        f8_1d,          # fitness
        f8_2d,          # steps
        f8_2d,          # weights
        f8_2d,          # basis
        f8_1d,          # core
        f8_1d,          # psi
        f8_1d,          # path_sigma
        f8_1d,          # path_cov
        f8_1d,          # mean
        f8,             # sigma
        f8,             # lr_sigma
        f8,             # lr_cov
        f8,             # lr_mean
        f8,             # mu_eff
        f8,             # damp_sigma
        f8,             # expected_path_length
        i8,             # opt_iter
        i8              # elite_size
        ),
    fastmath=True
    )
def _tell(
    fitness, steps, weights, basis, core, psi, path_sigma, path_cov, mean,
    sigma, lr_sigma, lr_cov, lr_mean, mu_eff, damp_sigma, expected_path_length,
    opt_iter, elite_size):
    """Update the state variables."""

    # Get the elite indices
    sorted_indices = argsort(fitness)
    elite_indices = sorted_indices[:elite_size]

    # Initialize the elite mean step
    elite_weights = weights[:elite_size]
    elite_mean_step = nsum(
        steps[elite_indices] * elite_weights, axis=0
        )

    # Update the step size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff)
        * _lanczos_inverse_sqrt_product(elite_mean_step, basis, core, psi)
        )

    # Get the norm of the step size evolution path
    ps_norm = norm(path_sigma)

    # Update the mean
    mean += (lr_mean * sigma) * elite_mean_step

    # Update the step size
    sigma = sigma * exp(
        (lr_sigma / damp_sigma) * (ps_norm / expected_path_length - 1.0)
        )

    # Check if the updated sigma is sufficiently small
    if sigma < 1e-15:

        # Clip to 1e-15
        sigma = 1e-15

    # Compute the update switch for the covariance evolution path
    update_switch = (
        1.0
        if ps_norm / sqrt(1.0 - (1.0 - lr_sigma)**(2.0 * opt_iter))
        < (1.4 + 2.0 / (mean.size + 1.0)) * expected_path_length
        else 0.0
        )

    # Compute the 'keep' term of the covariance evolution path
    path_cov *= (1.0 - lr_cov)

    # Precompute the coefficient
    coeff = update_switch * sqrt(lr_cov * (2.0 - lr_cov) * mu_eff)

    # Update the covariance evolution path with the elite mean step
    path_cov += coeff * elite_mean_step

    return path_sigma, mean, sigma, path_cov, update_switch


@njit(
    i8(
        # Return: rank
        f8_1d,          # eigenvalues
        f8,             # energy_fraction
        i8,             # min_rank
        ),
    fastmath=True
    )
def _energy_rank_selection(eigenvalues, energy_fraction, min_rank):
    """Determine the optimal rank from the total energy criterium."""

    # Initialize the total energy
    total_energy = 0.0

    # Loop over the eigenvalues
    for index in range(eigenvalues.size):

        # Cumulate the total energy
        total_energy += eigenvalues[index]**2

    # Check if the total energy is sufficiently small
    if total_energy <= 1e-15:

        # Return the minimum rank
        return min_rank

    # Initialize the cumulative energy
    cumulative = 0.0

    # Loop over the eigenvalues
    for index in range(eigenvalues.size):

        # Cumulate the energy
        cumulative += eigenvalues[index]**2

        # Check if the cumulated energy exceeds the requested fraction
        if cumulative >= energy_fraction * total_energy:

            # Return the current rank
            return max(min_rank, index + 1)

    return max(min_rank, eigenvalues.size)


@njit(
    types.Tuple((f8_2d, f8_1d, f8_1d, i8))(
        # Return: Tuple(basis_new, core_new, psi_new, rank_new)
        f8_2d,          # basis
        f8_1d,          # core
        f8_1d,          # psi
        f8_2d,          # steps_sorted
        f8_2d,          # weights
        f8_1d,          # path_cov
        f8,             # lr_cov
        f8,             # lr_rank_one
        f8,             # lr_rank_mu
        i8,             # low_rank_max_dimension
        bo,             # low_rank_is_adaptive
        f8,             # low_rank_energy_tolerance
        f8,             # update_switch
        bo              # force_expansion
        ),
    fastmath=True
    )
def _adaptive_bug_step(
    basis, core, psi, steps_sorted, weights, path_cov, lr_cov, lr_rank_one,
    lr_rank_mu, low_rank_max_dimension, low_rank_is_adaptive,
    low_rank_energy_tolerance, update_switch, force_expansion):
    """Perform an update step of the adaptive BUG integrator."""

    # Ensure array layouts
    basis = asfortranarray(basis)
    steps_sorted = ascontiguousarray(steps_sorted)

    # Get the problem dimensionality and current rank
    dim, rank = basis.shape

    # Determine the maximum rank and the augmentation size
    max_rank = min(2 * rank, dim)
    aug_size = max_rank - rank

    # Copy and flatten the weights
    weights_sorted = weights.ravel().copy()

    # Get the indices for the negative weights
    neg_indices = where(weights_sorted < 0.0)[0]

    # Check if any negative weights are present
    if neg_indices.size > 0:

        # Extract negative steps into a contiguous block
        steps_neg = steps_sorted[neg_indices]

        # Compute the inverse diagonal component
        inv_psi = 1.0 / maximum(psi, 1e-15)

        # Project the inverse onto the basis
        inv_psi_basis = inv_psi[:, None] * basis

        # Build the small Woodbury matrix
        woodbury = diag(1.0 / maximum(core, 1e-15)) + basis.T @ inv_psi_basis

        # Apply the inverse scaling to the negative steps
        inv_psi_steps = inv_psi[:, None] * steps_neg.T

        # Project the scaled steps onto the basis
        rhs = basis.T @ inv_psi_steps

        # Solve the small Woodbury system
        correction = solve(woodbury, rhs)

        # Measure the step size in the inverse covariance metric
        quadratic_forms = (
            nsum(steps_neg.T * inv_psi_steps, axis=0)
            - nsum(rhs * correction, axis=0)
            )

        # Ensure non-negative length
        quadratic_forms = maximum(quadratic_forms, 0.0)

        # Convert the step size into a scaling factor
        factors = dim / (quadratic_forms + 1e-15)

        # Prevent the weights from increasing
        factors = minimum(1.0, factors)

        # Apply the weight rescaling using the factors
        weights_sorted[neg_indices] *= factors

    # Lift the weights
    weights_sorted_2d = weights_sorted.reshape((-1, 1))

    # Get the adjusted rank-1 learning rate
    lr_rank_one_adj = (
        (1.0 - update_switch) * lr_rank_one * lr_cov * (2.0 - lr_cov)
        )

    # Get the decay rate
    lr_decay = lr_rank_one + lr_rank_mu - lr_rank_one_adj

    # Compute the rank-one update
    rank_one_term_u = path_cov[:, None] * (path_cov @ basis)

    # Compute the rank-mu update
    steps_basis = steps_sorted @ basis
    rank_mu_term_u = steps_sorted.T @ (weights_sorted_2d * steps_basis)

    # Get the diagonal of the covariance update
    lr_diag_upd = (
        lr_rank_one * path_cov**2
        + lr_rank_mu * nsum(weights_sorted_2d * steps_sorted**2, axis=0)
        + (1.0 - lr_decay) * (nsum(basis**2 * core, axis=1) + psi)
        )

    # Apply the update to the current low-rank covariance factor
    k_slice = (
        lr_rank_one * rank_one_term_u
        + lr_rank_mu * rank_mu_term_u
        + (1.0 - lr_decay) * (basis * core)
        )

    # Initialize the augmented low-rank covariance factor
    k_aug = zeros((dim, max_rank))
    k_aug[:, :rank] = k_slice

    # Compute the orthogonal basis of k_slice
    q_slice, _ = qr(k_slice)

    # Initialize the augmentation counter
    n_added = 0

    # Loop over the number of required augmentation vectors
    for index in range(aug_size):

        # Check if the index is zero
        if index == 0:

            # Start with the covariance evolution path
            vector = path_cov.copy()

        # Check if an intermediate index is used
        elif index - 1 < steps_sorted.shape[0]:

            # Try a successful mutation step
            vector = steps_sorted[index - 1].copy()

        else:

            # Fall back to a random direction
            vector = randn(dim)

        # Perform full reorthogonalization twice
        for _ in range(2):

            # Project onto the orthogonal complement of the current basis
            vector -= q_slice @ (q_slice.T @ vector)

            # Check if the basis has already been augmented
            if n_added > 0:

                # Get the the previous augmentation vectors
                previous = k_aug[:, rank:rank + n_added]

                # Orthogonalize the current vector against the previous
                vector -= previous @ (previous.T @ vector)

        # Get the norm
        vector_norm = norm(vector)

        # Check if the norm is sufficiently small
        if vector_norm > 1e-12:

            # Add the normalized vector
            k_aug[:, rank + n_added] = vector / vector_norm

            # Increase the counter
            n_added += 1

    # Compute an orthonormal basis of the augmented low-rank directions
    uhat_aug, _ = qr(k_aug)
    uhat_aug_tr = uhat_aug.T

    # Compute the rank-one update in the augmented space
    path_aug = uhat_aug.T @ path_cov
    rank_one_term_s = outer(path_aug, path_aug)

    # Compute the rank-mu update in the augmented space
    steps_aug = steps_sorted @ uhat_aug
    rank_mu_term_s = steps_aug.T @ (weights_sorted_2d * steps_aug)

    # Project the old covariance structure onto the augmented space
    proj = uhat_aug_tr @ basis
    ext_s = (proj * core) @ proj.T

    # Apply the update to the augmented core matrix
    shat = (
        lr_rank_one * rank_one_term_s
        + lr_rank_mu * rank_mu_term_s
        + (1.0 - lr_decay) * ext_s
        )

    # Enforce the symmetry
    shat = 0.5 * (shat + shat.T)

    # Decompose the updated core matrix
    sigma, basis_sigma = eigh(shat)

    # Sort the eigenvalues and -vectors in descending order
    idx = argsort(sigma)[::-1]
    sigma = sigma[idx]
    basis_sigma = basis_sigma[:, idx]

    # Check if the rank should be adapted
    if low_rank_is_adaptive:

        # Initialize the rank delta
        rank_delta = 0

        # Check if the rank should be expanded due to evolutionary signals
        if force_expansion:

            # Set the rank increment to one
            rank_delta = 1

        else:

            # Get the off-diagonal core matrix
            shat_off = shat - diag(diag(shat))

            # Calculate and sort the eigenvalues
            sigma_off, _ = eigh(shat_off)
            sigma_off = sort(nabs(sigma_off))[::-1]

            # Determine the rank from the spectral energy
            rank_energy = _energy_rank_selection(
                sigma_off, energy_fraction=1-low_rank_energy_tolerance,
                min_rank=1
                )

            # Check if the proposed rank is smaller
            if rank_energy < rank:

                # Set the rank increment to minus one
                rank_delta = -1

            # Check if the proposed rank is larger
            elif rank_energy > rank:

                # Set the rank increment to one
                rank_delta = 1

        # Change the rank within the allowed range
        rank_new = rank + rank_delta
        rank_new = max(1, min(rank_new, low_rank_max_dimension))

    else:

        # Keep the current rank fixed
        rank_new = min(rank, low_rank_max_dimension)

    # Truncate the low-rank factors to the new rank
    basis_new = uhat_aug @ basis_sigma[:, :rank_new]
    core_new = sigma[:rank_new]

    # Get the diagonal of the updated low-rank component
    lr_diag_new = nsum(basis_new**2 * core_new, axis=1)

    # Update psi by the residual diagonal
    psi_new = lr_diag_upd - lr_diag_new

    # Keep the residual diagonal strictly positive
    psi_new = maximum(psi_new, 1e-15)

    return basis_new, core_new, psi_new, rank_new
