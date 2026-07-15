"""
Dynamical low-rank covariance matrix adaptation evolution strategy
(DLR-CMA-ES).
"""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from signal import getsignal, SIGINT, signal
from time import time

from collections import deque
from math import inf
from numba import njit, types
from numpy import (
    add, arange, argmin, argsort, array, asfortranarray, clip, diag, exp, eye,
    float64, full, isinf, log, maximum, minimum, ptp, ones, outer, sqrt, where,
    zeros)
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from numpy import mean as nmean
from numpy import min as nmin
from numpy import sum as nsum
from numpy.linalg import eigh, norm, qr
from numpy.random import default_rng, randn

# %% Internal package import

from seamaze.logging import Logging
from seamaze.utils import make_compat

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
        (unpublished).

    Here, the covariance matrix is represented as a sum of a diagonal variance
    component and a diagonally-free low-rank correlation component:

        C = diag(psi) + USU^T,

    where psi captures marginal variances and USU^T models off-diagonal
    dependencies only, satisfying diag(USU^T) = 0.

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

    initial_sigma : float, default=0.3
        Initial step size.

    low_rank_init_dimension : int, default=None
        Initial rank of the approximation. Defaults to
        4 + int(3*log(`number_of_variables`)).

    low_rank_max_dimension : int, default=None
        Maximum rank of the approximation. Defaults to `number_of_variables`.

    low_rank_is_adaptive : bool, default=True
        Indicator for adaptive low-rank selection.

    low_rank_energy_tolerance : float, default=1e-3
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
        high-dimensional problems. Defaults to max(1, 0.1/β), where β is the
        sum of the rank-1 and rank-mu learning rates.

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
            initial_sigma=0.3,
            low_rank_init_dimension=None,
            low_rank_max_dimension=None,
            low_rank_is_adaptive=True,
            low_rank_energy_tolerance=1e-3,
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

        # Set the random seed
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
        self._lr_rank_one = (
            2 / ((self._number_of_variables + 1.3)**2 + self._mu_eff)
            )
        self._lr_rank_mu = min(
            1.0 - self._lr_rank_one,
            2.0 * (
                (0.25 + self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                ((self._number_of_variables + 2.0)**2
                 + 2.0 * self._mu_eff / 2.0)
                )
            )
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

        # Get the low-rank adaptivity parameters
        default_rank = 4 + int(3 * log(self._number_of_variables))

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
            self._low_rank_max_dimension,
            )

        self._low_rank_is_adaptive = low_rank_is_adaptive
        self._low_rank_energy_tolerance = low_rank_energy_tolerance

        # Determine the integrator rank
        self.rank = self._low_rank_init_dimension

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
        self._squared_bound_errors = None
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

        init_var = 1.0
        alpha = min(0.5, self.rank / self._number_of_variables)
        self._basis = eye(
            self._number_of_variables, self.rank, order='F', dtype=float64
            )
        self._core = (alpha * init_var) * ones(self.rank, dtype=float64)

        self._psi = init_var * ones(self._number_of_variables, dtype=float64)
        self._psi[:self.rank] *= (1.0 - alpha)

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness = None
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._update_interval = (
            max(1, int(0.1/(self._lr_rank_one + self._lr_rank_mu)))
            if update_interval is None else update_interval
            )
        self._current_expansion_reasons = []
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None, 'iterations': None
            }

        # Initialize the stop flag
        self._stop_requested = False

    def ask(self):
        """
        Generate a new population.

        Returns
        -------
        ndarray
            Sample population (bound to the feasible region).

        ndarray
            Sample steps.
        """

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
        sqrt_core = sqrt(maximum(self._core, 1e-15))

        # Transform the samples into low-rank and noisy components
        structured_part = (z_low_rank * sqrt_core) @ self._basis.T
        noise_part = z_noise * sqrt(self._psi)

        # Get the steps by adding the components
        self._steps[:num_random] = structured_part + noise_part

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            low_rank_gradient = (
                self._basis
                @ (self._core * (self._basis.T @ gradient))
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

            # Sum the squared errors for each individual
            self._squared_bound_errors = nsum(
                (eps_lower + eps_upper) ** 2, axis=1
                )

            # Mirror the violating individuals back into the feasible region
            self._population[:] = where(
                self._population < self.lower_variable_bounds,
                self.lower_variable_bounds + eps_lower,
                self._population
                )
            self._population[:] = where(
                self._population > self.upper_variable_bounds,
                self.upper_variable_bounds - eps_upper,
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

            # Check if the penalty factor has been initialized
            if self._gamma is None:

                # Compute the unpenalized fitness range
                fitness_range = nmax(true_fitness) - nmin(true_fitness)

                # Scale the penalty factor to the fitness range
                self._gamma = (
                    (fitness_range if fitness_range > 1e-8 else 10.0) /
                    (self._number_of_variables + 1e-15)
                    )

            # Check if any bound violations occurred
            if nany(self._squared_bound_errors > 0):

                # Set the gamma factor > 1
                gamma_factor = 1.1

            else:

                # Set the gamma factor < 1
                gamma_factor = 0.99

            # Adapt the penalty factor
            self._gamma = clip(self._gamma * gamma_factor, 1e-5, 1e10)

            # Compute the penalized fitness values
            selection_fitness = (
                true_fitness + self._gamma * self._squared_bound_errors
                )

        else:

            # Apply unpenalized fitness values for selection
            selection_fitness = true_fitness

        # Get the best unpenalized fitness
        best_index = argmin(selection_fitness)
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

    def tell(self, fitness):
        """
        Update the state variables and perform an adaptive BUG step.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Update the state variables
        (path_sigma_new, sigma_new, mean_new, path_cov_new, update_switch
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
        self._sigma = sigma_new
        self._mean[:] = mean_new
        self._path_cov[:] = path_cov_new

        # Check if the low-rank factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Update the low-rank and noise factors
            basis_new, core_new, psi_new, rank_new = _adaptive_bug_step(
                self._basis,
                self._core,
                self._psi,
                self._steps[argsort(fitness)],
                self._weights,
                self._path_cov,
                self._lr_cov,
                self._lr_rank_one,
                self._lr_rank_mu,
                self._low_rank_max_dimension,
                self._low_rank_is_adaptive,
                self._low_rank_energy_tolerance,
                update_switch=update_switch,
                force_expansion=self.check_rank_expansion()
                )

            # Save the factor states
            self._basis = basis_new
            self._core = core_new
            self._psi[:] = psi_new
            self.rank = rank_new

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

    def check_rank_expansion(self):
        """
        Check the evolutionary-state rank expansion criteria.

        Returns
        -------
        bool
            Indicator for rank expansion.
        """

        # Check if the current rank equals the dimensionality
        if self.rank == self._number_of_variables:

            return False

        # Initialize the expansion pressure flags
        path_pressure = False
        fitness_pressure = False

        # Initialize the expansion reasons
        reasons = []

        # Get the expected path energy fraction from a random subspace
        expected_explained_ratio = self.rank / self._number_of_variables

        # Calculate the explained path energy fraction
        path_norm_sq = norm(self._path_cov)**2 + 1e-15
        path_cov_proj = self._basis.T @ self._path_cov
        path_explained_ratio = norm(path_cov_proj)**2 / path_norm_sq

        # Calculate the path underrepresentation
        path_underrepresentation = (
            (expected_explained_ratio - path_explained_ratio)
            / (expected_explained_ratio + 1e-15)
            )

        # Check if the underrepresentation is larger than 50%
        if path_underrepresentation > 0.5:

            # Set the path pressure flag
            path_pressure = True

            # Add the expansion reason
            reasons.append('path_excess_residual')

        # Check if the fitness history has been completely filled
        if len(self._fitness_history) == self._fitness_history.maxlen:

            # Get the fitness history
            history = array(self._fitness_history)

            # Split the history
            half = history.size // 2

            # Compute the average fitness
            fit_old = nmean(history[:half])
            fit_new = nmean(history[half:])

            # Check if the relative improvement is very small
            if abs(fit_old - fit_new) / max(abs(fit_old), 1.0) < 1e-3:

                # Set the fitness pressure flag
                fitness_pressure = True

            # Get the mean fitness
            fit_mean = nmean(history)

            # Get the fitness range
            fit_range = ptp(history)

            # Check if the fitness variation is very small
            if fit_range / (abs(fit_mean) + 1e-15) < 1e-3:

                # Set the fitness pressure flag
                fitness_pressure = True

            # Check if fitness pressure is applied
            if fitness_pressure:

                # Add the expansion reason
                reasons.append('fitness_stagnation')

        # Update the current expansion reasons
        self._current_expansion_reasons = reasons

        return path_pressure and fitness_pressure

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

        # Get the maximum and minimum eigenvalue
        max_eval, min_eval = _approx_spectrum_extremes(
            self._basis,
            self._core,
            self._psi
            )

        # Check if any eigenvalue is zero or the condition number explodes
        if min_eval < 1e-14 or (max_eval / (min_eval + 1e-15)) >= 1e14:

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

            # Get the mean fitness
            fit_mean = nmean(history)

            # Get the fitness range
            fit_range = ptp(history)

            # Check if the absolute fitness range is below tolerance
            if fit_range < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'ABSOLUTE_FITNESS_PLATEAU_REACHED')

                return True

            # Check if the relative median difference is below tolerance
            if fit_range / (nabs(fit_mean) + 1e-15) < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'RELATIVE_FITNESS_PLATEAU_REACHED')

                return True

        return False


# Set the variable types
f8_2d = types.float64[:, :]
f8_2d_f = types.float64[::1, :]
f8_2d_c = types.float64[:, ::1]
f8_1d = types.float64[:]
i8_1d = types.int64[:]
f8 = types.float64
i8 = types.int64
bo = types.bool_

@njit(
    types.Tuple((f8, f8))(
        # Return: Tuple(max_eval, min_eval)
        f8_2d_f,        # basis
        f8_1d,          # core
        f8_1d,          # psi
        ),
    fastmath=True
    )
def _approx_spectrum_extremes(basis, core, psi):
    """Approximate the covariance spectrum extremes."""

    # Get the dimensionality
    dim = basis.shape[0]

    # Estimate the minimum eigenvalue from psi (Weyl's inequality)
    min_eval = nmin(psi)

    # Check if the value is very small
    if min_eval < 1e-12:

        # Set the value to 1e-12 to prevent numerical issues
        min_eval = 1e-12

    # Draw a random vector from the standard normal distribution
    sample = randn(dim)

    # Get the sample norm
    sample_norm = norm(sample)

    # Check if the norm is positive
    if sample_norm > 0:

        # Normalize the vector
        sample = sample / sample_norm

    # Loop for a fixed number of power iterations
    for _ in range(8):

        # Compute the product C*x efficiently
        update_vec = basis @ (core * (basis.T @ sample)) + psi * sample

        # Get the norm of the update vector
        vec_norm = norm(update_vec)

        # Check if the norm is very small
        if vec_norm < 1e-12:

            break

        # Normalize the update vector
        sample = update_vec / vec_norm

    # Compute the Rayleigh quotient to get the maximum eigenvalue
    max_eval = sample @ update_vec

    return max_eval, min_eval


@njit(
    f8_1d(
        # Return: approximation
        f8_1d,          # elite_mean_step
        f8_2d_f,        # basis
        f8_1d,          # core
        f8_1d,          # psi
        ),
    fastmath=True
    )
def _lanczos_matrix_inverse_sqrt_product(elite_mean_step, basis, core, psi):
    """Compute the inverse matrix square root-vector product C^(-1/2)*x."""

    # Get the norm of the elite mean step
    norm_step = norm(elite_mean_step)

    # Check if the mean step is very small
    if norm_step < 1e-15:

        # Return zeros
        return zeros(elite_mean_step.shape[0], dtype=float64)

    # Get the shape of the basis matrix
    dim, rank = basis.shape

    # Determine the Krylov dimension
    krylov_dim = min(dim, max(30, 3 * rank), 100)

    # Initialize the coefficients and the Krylov subspace basis
    alpha = zeros(krylov_dim, dtype=float64)
    beta = zeros(krylov_dim, dtype=float64)
    krylov_basis = zeros((krylov_dim, dim)).T

    # Insert the first basis vector
    krylov_basis[:, 0] = elite_mean_step / norm_step

    # Track the current Krylov subspace dimensionality
    current_dim = krylov_dim

    # Loop over the dimensions of the Krylov subspace
    for index in range(krylov_dim):

        # Get the current basis vector
        current_basis = krylov_basis[:, index]

        # Generate the next Krylov vector
        update_vec = (
            basis @ (core * (basis.T @ current_basis)) + psi * current_basis
            )

        # Loop until the current dimension of the Krylov subspace
        for sub in range(index + 1):

            # Get the i-th Krylov basis vector
            sub_basis = krylov_basis[:, sub]

            # Project the next vector onto the i-th Krylov basis
            projection = sub_basis.T @ update_vec

            # Check if the penultimate dimension has been reached
            if sub == index:

                # Store the alpha coefficient
                alpha[index] = projection

            # "Clean" the update vector by the projection
            update_vec = update_vec - projection * sub_basis

        # Check if the final dimension has not been reached
        if index < krylov_dim - 1:

            # Get the beta value by the update vector norm
            beta_next = norm(update_vec)

            # Insert the value at the next index of the beta vector
            beta[index + 1] = beta_next

            # Check for "happy breakdown" (exact solution found)
            if beta_next < 1e-12:

                # Set the current dimensionality to the final index
                current_dim = index + 1

                break

            # Store the new Krylov basis vector
            krylov_basis[:, index + 1] = update_vec / beta_next

    # Initialize the tridiagonal projection matrix by the alpha values
    tridiagonal = diag(alpha[:current_dim])

    # Check if the current Krylov dimension is larger than one
    if current_dim > 1:

        # Fill the subdiagonals of the projection matrix
        tridiagonal += (
            diag(beta[1:current_dim], k=1) + diag(beta[1:current_dim], k=-1)
            )

    # Perform an eigendecomposition on the projection matrix
    tridiagonal_evals, tridiagonal_evecs = eigh(tridiagonal)

    # Safeguard against small eigenvalues
    threshold = max(1e-15, tridiagonal_evals[-1] * 1e-12)
    tridiagonal_evals = maximum(tridiagonal_evals, threshold)

    # Calculate the inner vector (T^(-1/2) * e_1||x||)
    inv_sqrt_evals = 1.0 / sqrt(tridiagonal_evals)
    inner_vector = (
        tridiagonal_evecs @ (
            inv_sqrt_evals * (tridiagonal_evecs[0, :] * norm_step))
        )

    # Get the approximated inverse matrix square root-vector product
    krylov_slice = asfortranarray(krylov_basis[:, :current_dim])
    approximation = krylov_slice @ inner_vector

    return approximation


@njit(
    types.Tuple((f8_1d, f8, f8_1d, f8_1d, f8))(
        # Return: Tuple(path_sigma, sigma, mean, path_cov, update_switch)
        f8_1d,          # fitness
        f8_2d,          # steps
        f8_2d,          # weights
        f8_2d_f,        # basis
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
        * _lanczos_matrix_inverse_sqrt_product(
            elite_mean_step, basis, core, psi
            )
        )

    # Get the norm of the step size evolution path
    ps_norm = norm(path_sigma)

    # Update the mean
    mean += (lr_mean * sigma) * elite_mean_step

    # Update the step size
    sigma = sigma * exp(
        (lr_sigma / damp_sigma) * (ps_norm / expected_path_length - 1.0)
        )

    # Check if the updated sigma is lower than 1e-15
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

    # Pre-compute the coefficient
    coeff = update_switch * sqrt(lr_cov * (2.0 - lr_cov) * mu_eff)

    # Update the covariance evolution path with the elite mean step
    path_cov += coeff * elite_mean_step

    return path_sigma, sigma, mean, path_cov, update_switch


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

        # Check if the eigenvalue is positive
        if eigenvalues[index] > 0.0:

            # Cumulate the total energy
            total_energy += eigenvalues[index]

    # Check if the total energy is very small
    if total_energy <= 1e-15:

        # Return the minimum rank
        return min_rank

    # Initialize the cumulative energy
    cumulative = 0.0

    # Loop over the eigenvalues
    for index in range(eigenvalues.size):

        # Check if the eigenvalue is positive
        if eigenvalues[index] > 0.0:

            # Cumulate the energy
            cumulative += eigenvalues[index]

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
        f8_2d,          # elite_steps
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

    # Get the dimension and rank
    dim, rank = basis.shape

    # Get the transposed basis and the basis-core product
    basis_tr = basis.T
    basis_core = basis * core

    # Determine the maximum rank and the augmentation size
    max_rank = min(2 * rank, dim)
    aug_size = max_rank - rank

    # Copy and flatten the weights
    weights_sorted = weights.ravel().copy()
    weights_sorted_2d = weights_sorted.reshape((-1, 1))

    # Transform the steps for negative-weight stabilization
    steps_sorted_tr = steps_sorted @ basis

    # Get the indices for the negative weights
    neg_indices = where(weights_sorted < 0.0)[0]

    # Check if any negative weights are present
    if neg_indices.size > 0:

        # Get the inverse square root of the core matrix
        safe_core = maximum(1e-14, core)
        inv_sqrt_core = 1.0 / sqrt(safe_core)

        # Perform an isotropic transformation
        steps_neg_iso = steps_sorted_tr[neg_indices] * inv_sqrt_core

        # Get the squared norms of the isotropic vectors
        squared_z_norms = nsum(steps_neg_iso**2, axis=1)

        # Get the scaling factors
        factors = dim / (squared_z_norms + 1e-15)

        # Rescale the weights to guarantee positive definiteness
        weights_sorted[neg_indices] *= minimum(1.0, factors)

    # Get the adjusted rank-1 learning rate
    lr_rank_one_adj = (
        (1.0 - update_switch) * lr_rank_one * lr_cov * (2.0 - lr_cov)
        )

    # Get the decay rate
    lr_decay = lr_rank_one + lr_rank_mu - lr_rank_one_adj

    # Compute the diagonal-free rank-one update
    rank_one_diag = path_cov ** 2
    rank_one_term_u = (
        path_cov[:, None] * (path_cov @ basis)
        - rank_one_diag[:, None] * basis
        )

    # Compute the diagonal-free rank-mu update
    rank_mu_diag = nsum(weights_sorted_2d * steps_sorted**2, axis=0)
    steps_basis = steps_sorted @ basis
    rank_mu_term_u = (
        steps_sorted.T @ (weights_sorted_2d * steps_basis)
        - rank_mu_diag[:, None] * basis
        )

    # Determine the variance (diagonal) of the growth field
    growth_var = clip(
        lr_rank_one * rank_one_diag + lr_rank_mu * rank_mu_diag, -0.1 * psi,
        None
        )

    # Update psi with the growth field variance
    psi_new = maximum(1e-12, (1.0 - lr_decay) * psi + growth_var)

    # Set the initial state for the K-slice
    k_slice_init = basis_core

    # Integrate the low-rank velocity field
    k_slice = (
        + lr_rank_one * rank_one_term_u
        + lr_rank_mu * rank_mu_term_u
        + (1.0 - lr_decay) * k_slice_init
        )

    # Augment the K-slice with random directions to allow rank adaptation
    k_aug = zeros((dim, max_rank))
    k_aug[:, :rank] = k_slice
    if aug_size > 0:
        random_noise = randn(dim, aug_size)
        orthogonal_noise = random_noise - basis @ (basis_tr @ random_noise)
        q_orth, _ = qr(orthogonal_noise)
        k_aug[:, rank:max_rank] = q_orth

    # Compute an orthonormal basis of the augmented low-rank subspace
    uhat_aug, _ = qr(k_aug)
    uhat_aug_tr = uhat_aug.T

    # Project the existing low-rank covariance into the augmented subspace
    proj = uhat_aug_tr @ basis
    ext_s = (proj * core) @ proj.T
    ext_s -= diag(diag(ext_s))

    # Compute the diagonal-free rank-one update in the augmented space
    path_cov_aug = uhat_aug_tr @ path_cov
    rank_one_term_s = outer(path_cov_aug, path_cov_aug)
    rank_one_term_s -= diag(diag(rank_one_term_s))

    # Compute the diagonal-free rank-mu update in the augmented space
    steps_aug_tr = steps_sorted @ uhat_aug
    rank_mu_term_s = steps_aug_tr.T @ (weights_sorted_2d * steps_aug_tr)
    rank_mu_term_s -= diag(diag(rank_mu_term_s))

    # Assemble the low-rank correlation velocity field
    f_core = (
        + lr_rank_one * rank_one_term_s
        + lr_rank_mu * rank_mu_term_s
        - lr_decay * ext_s
        )

    # Integrate the low-rank matrix flow
    shat = ext_s + f_core

    # Enforce symmetry
    shat = 0.5 * (shat + shat.T)

    # Extract the updated low-rank eigensystem
    sigma, basis_sigma = eigh(shat)

    # Sort the eigenvalues and -vectors
    idx = argsort(sigma)[::-1]
    sigma = sigma[idx]
    basis_sigma = basis_sigma[:, idx]

    # Apply lower threshold to the eigenvalues
    threshold = max(1e-12, abs(sigma[0]) / 1e12)
    sigma_safe = maximum(sigma, threshold)

    # Check if the rank should be adapted
    if low_rank_is_adaptive:

        # Initialize the rank delta
        rank_delta = 0

        # Check if the rank should be expanded due to the evolutionary state
        if force_expansion:

            # Increase the rank
            rank_delta += 1

        else:

            # Determine rank from retained covariance energy
            rank_energy = _energy_rank_selection(
                sigma_safe, energy_fraction=1-low_rank_energy_tolerance,
                min_rank=2
                )

            # Check if the proposed rank is smaller
            if rank_energy < rank:

                # Reduce the rank delta
                rank_delta -= 1

            # Check if the proposed rank is larger
            elif rank_energy > rank:

                # Increase the rank delta
                rank_delta += 1

        # Clip the rank to the minimum/maximum allowed rank
        rank_new = rank + rank_delta
        rank_new = max(2, min(rank_new, low_rank_max_dimension))

    else:

        # Fix the rank
        rank_new = min(rank, low_rank_max_dimension)

    # Truncate to the new rank
    basis_new = uhat_aug @ basis_sigma[:, :rank_new]
    core_new = sigma_safe[:rank_new]

    # Ensure that the basis remains a Fortran array
    basis_new = asfortranarray(basis_new)

    return basis_new, core_new, psi_new, rank_new
