"""Covariance matrix adaptation evolution strategy (CMA-ES)."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

from collections import deque
from math import inf
from numba import njit
from numpy import (
    add, allclose, arange, argmax, argmin, argsort, array, asfortranarray,
    clip, exp, eye, float64, full, log, maximum, minimum, ones, outer, sqrt,
    where, zeros)
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from numpy import min as nmin
from numpy import sum as nsum
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.linalg import eigh

# %% Internal package import

from seamaze.logging import Logging
from seamaze.utils import make_compat

# %% Class definition


class CMAES:
    """
    Covariance matrix adaptation evolution strategy (CMA-ES) class.

    This class implements the CMA-ES algorithm according to the paper:

        Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. ArXiv,
        abs/1604.00772.

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
        Minimum allowed step size. If the step size falls below this limit,
        optimization stops (convergence/collapse criterion).

    update_interval : int, default=1
        Frequency of the covariance update (in generations). Larger values
        (e.g. 10) can significantly speed up the algorithm for
        high-dimensional problems.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}, \
        default='debug'
        Minimum logging level for passing messages to the console.

    callback : Callable[[dict], None], default=None
        Optional function called at the end of each iteration. Must accept
        the solver instance.
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
            maximum_iterations=100000,
            maximum_wall_time=43200,
            fitness_threshold=-inf,
            fitness_window_size=50,
            tolerance=1e-6,
            sigma_threshold=1e-8,
            update_interval=1,
            min_log_level='debug',
            callback=None):

        # Initialize the logger
        self.logger = Logging('CMA-ES', min_log_level)

        # Log a message about the initialization
        self.logger.info('Initializing CMA-ES ...')

        # Set the random seed
        self._rng = default_rng(42)

        # Initialize the optimization problem variables
        self._number_of_variables = number_of_variables
        self.objective = make_compat(objective)
        self.gradient = gradient
        self.lower_variable_bounds = (
            full(self._number_of_variables, -inf)
            if lower_variable_bounds is None else lower_variable_bounds
            )
        self.upper_variable_bounds = (
            full(self._number_of_variables, inf)
            if upper_variable_bounds is None else upper_variable_bounds
            )

        # Initialize the population size
        self._pop_size = (
            4 + int(3 * log(self._number_of_variables))
            if number_of_individuals is None else number_of_individuals
            )
        self._pop_size += (2 if gradient is not None else 0)

        # Initialize the base weights
        base_weights = (
            log((self._pop_size + 1) / 2) - log(arange(1, self._pop_size + 1))
            )

        # Initialize the elite size
        self._elite_size = int(nsum(base_weights > 0))

        # Determine the sums of positive and negative base weights
        bw_pos_sum = nsum(base_weights[:self._elite_size])
        bw_neg_sum = nsum(base_weights[self._elite_size:])

        # Initialize the variance effective selection mass
        self._mu_eff = (
            bw_pos_sum**2 / nsum(base_weights[:self._elite_size]**2)
            )
        mu_eff_neg = (
            bw_neg_sum**2 / nsum(base_weights[self._elite_size:]**2)
            )

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

        # Adjust the weights to ensure positive definiteness
        alpha_mu_neg = 1.0 + self._lr_rank_one / (self._lr_rank_mu + 1e-12)
        alpha_mu_eff_neg = 1.0 + (2.0 * mu_eff_neg) / (self._mu_eff + 2.0)
        alpha_posdef_neg = (
            (1.0 - self._lr_rank_one - self._lr_rank_mu)
            / (self._number_of_variables * self._lr_rank_mu + 1e-12)
            )
        alpha_min = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_posdef_neg)

        self._weights = where(
            base_weights > 0,
            (1.0 / bw_pos_sum) * base_weights,
            (alpha_min / (abs(bw_neg_sum) + 1e-12)) * base_weights
            )

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
            (self._pop_size, self._number_of_variables),
            order='F', dtype=float64)
        self._population = zeros(
            (self._pop_size, self._number_of_variables),
            order='F', dtype=float64)

        self._path_sigma = zeros(self._number_of_variables, dtype=float64)
        self._path_cov = zeros(self._number_of_variables, dtype=float64)
        self._mean = zeros(self._number_of_variables, dtype=float64)

        self._cov = eye(self._number_of_variables, dtype=float64)
        self._root_cov = eye(
            self._number_of_variables, order='F', dtype=float64)
        self._left_basis = eye(
            self._number_of_variables, order='F', dtype=float64)
        self._core_vector = ones(self._number_of_variables, dtype=float64)

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness = None
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._update_interval = update_interval
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None
            }

    def ask(self):
        """Generate a new population."""

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Partially sample from a standard multivariate Gaussian
            num_random = self._pop_size - 2
            zsamples = self._rng.standard_normal(
                (num_random, self._number_of_variables)
                )

            # Sample the stochastic steps from the multivariate Gaussian
            self._steps[:num_random] = zsamples @ self._root_cov.T

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            natural_gradient = self._cov @ gradient

            # Compute the rescaling factor
            rescale = 1 / (sqrt(gradient @ natural_gradient) + 1e-15)

            # Compute the natural gradient step
            gradient_step = natural_gradient * rescale

            # Add the mirrored gradient steps
            self._steps[-2] = -gradient_step
            self._steps[-1] = gradient_step

        else:

            # Fully sample from the standard multivariate Gaussian
            zsamples = self._rng.standard_normal(
                (self._pop_size, self._number_of_variables)
                )

            # Sample all steps from the multivariate Gaussian
            self._steps[:] = zsamples @ self._root_cov.T

        # Sample the new population
        add(self._mean, self._sigma * self._steps, out=self._population)

        # Compute the distance to the lower and upper bounds
        eps_lower = maximum(0.0, self.lower_variable_bounds - self._population)
        eps_upper = maximum(0.0, self._population - self.upper_variable_bounds)

        # Sum the squared errors for each individual
        self._squared_bound_errors = nsum((eps_lower + eps_upper) ** 2, axis=1)

        # Mirror the violating individuals back into the feasible region
        self._population = where(
            self._population < self.lower_variable_bounds,
            self.lower_variable_bounds + eps_lower,
            self._population
            )
        self._population = where(
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

        # Check if the penalty factor has been initialized
        if self._gamma is None:

            # Compute the unpenalized fitness range
            fitness_range = nmax(true_fitness) - nmin(true_fitness)

            # Scale the penalty factor to the fitness range
            self._gamma = (
                (fitness_range if fitness_range > 1e-8 else 10.0) /
                (self._number_of_variables + 1e-15)
                )

        # Compute the penalized fitness values
        selection_fitness = (
            true_fitness + self._gamma * self._squared_bound_errors
            )

        # Check if any bound violations occurred
        if nany(self._squared_bound_errors > 0):

            # Set the gamma factor > 1
            gamma_factor = 1.1

        else:

            # Set the gamma factor < 1
            gamma_factor = 0.95

        # Adapt the penalty factor
        self._gamma = clip(self._gamma * gamma_factor, 1e-5, 1e10)

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

    def tell(
            self,
            fitness):
        """
        Update the adaptive variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Update the basis variables
        self._sigma = _tell(
            fitness,
            self._steps,
            self._weights,
            self._left_basis,
            self._core_vector,
            self._path_sigma,
            self._path_cov,
            self._mean,
            self._sigma,
            self._cov,
            self._lr_sigma,
            self._lr_cov,
            self._lr_mean,
            self._lr_rank_one,
            self._lr_rank_mu,
            self._mu_eff,
            self._damp_sigma,
            self._expected_path_length,
            self._opt_iter,
            self._elite_size
            )

        # Check if the low-rank factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Update the low-rank factors via eigendecomposition
            self._core_vector, self._left_basis = eigh(
                self._cov, overwrite_a=False, check_finite=False
                )

            # Sort the singular values and vectors in descending order
            self._core_vector = self._core_vector[::-1]
            self._left_basis = asfortranarray(self._left_basis[:, ::-1])

            # Clip the singular values
            maximum(self._core_vector, 1e-12, out=self._core_vector)

            # Update the sampling matrix
            sqrt_core = sqrt(self._core_vector)
            self._root_cov = asfortranarray(self._left_basis * sqrt_core)

            # Reconstruct the covariance matrix
            self._cov = (
                (self._left_basis * self._core_vector) @ self._left_basis.T
                )

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

        try:

            # Continue until termination criteria are fulfilled
            while self.check_termination() is False:

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
                    f'f={round(self._result["optimal_value"], 6)}'
                    )

        except KeyboardInterrupt:

            # Log a message about the user stopping
            self.logger.warning("Optimization interrupted by user ...")
            self._result['solver_info'] = 'STOPPED_BY_USER'

        # Store the runtimes
        self._result['wall_time'] = time() - self._wall_start

        # Get the optimal point
        opt_point = self._result['optimal_point']

        # Check if the length is greater than 5
        if len(opt_point) > 5:

            # Truncate the solution string
            short_sol = (
                str(opt_point[:5]).replace("\n", "")[:-1] + " ...]"
                )

        else:

            # Return the full solution string
            short_sol = str(opt_point).replace("\n", "")

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
        max_ev = nmax(self._core_vector)
        min_ev = nmin(self._core_vector)

        # Check if any eigenvalue is zero or the condition number explodes
        if min_ev <= 0 or (max_ev / min_ev + 1e-15) >= 1e14:

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
            history = list(self._fitness_history)

            # Get the best and worst fitness within the window
            min_fit = nmin(history)
            max_fit = nmax(history)
            abs_range = max_fit - min_fit

            # Check if the absolute median difference is below tolerance
            if abs_range < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'ABSOLUTE_FITNESS_PLATEAU_REACHED')

                return True

            # Check if the relative median difference is below tolerance
            if abs_range / (nabs(min_fit) + 1e-15) < self.tolerance:

                # Add the solver info
                self._result['solver_info'] = (
                    'RELATIVE_FITNESS_PLATEAU_REACHED')

                return True

        # Get the longest axis
        longest_axis = self._root_cov[:, argmax(norm(self._root_cov, axis=0))]

        # Check if the mean is not shifted along the longest axis
        if allclose(
                self._mean, self._mean + 0.1 * self._sigma * longest_axis,
                rtol=0.0, atol=1e-15):

            # Add the solver info
            self._result['solver_info'] = 'NO_EFFECT_AXIS'

            return True

        return False


@njit(fastmath=True)
def _tell(
    fitness, steps, weights, left_basis, core_vector, path_sigma, path_cov,
    mean, sigma, cov, lr_sigma, lr_cov, lr_mean, lr_rank_one, lr_rank_mu,
    mu_eff, damp_sigma, expected_path_length, opt_iter, elite_size):
    """Update the basic CMAES variables."""

    # Get the search space dimension
    dim = mean.shape[0]

    # Sort the population by fitness
    sorted_indices = argsort(fitness)
    elite_indices = sorted_indices[:elite_size]

    # Initialize the elite mean step
    elite_weights = weights[:elite_size].reshape(-1, 1)
    elite_mean_step = nsum(
        steps[elite_indices] * elite_weights, axis=0
        )

    # Calculate the inverse rooted eigenvalues
    inv_root_core_vec = 1.0 / (sqrt(core_vector) + 1e-15)

    # Transform the elite mean step
    latent_step = left_basis.T @ elite_mean_step
    elite_mean_step_tr = left_basis @ (inv_root_core_vec * latent_step)

    # Update the step-size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff) * elite_mean_step_tr
        )

    # Get the norm of the step-size evolution path
    ps_norm = norm(path_sigma)

    # Update the step size
    sigma_update = sigma * exp(
        (lr_sigma / damp_sigma) * (ps_norm / expected_path_length - 1.0)
        )

    # Check if the updated sigma is lower than 1e-15
    if sigma_update < 1e-15:

        # Clip to 1e-15
        sigma_update = 1e-15

    # Update the mean
    mean += (lr_mean * sigma) * elite_mean_step

    # Compute the update switch for the covariance evolution path
    update_switch = (
        1.0
        if ps_norm / sqrt(1.0 - (1.0 - lr_sigma)**(2.0 * opt_iter))
        < (1.4 + 2.0/(mean.shape[0] + 1.0)) * expected_path_length
        else 0.0
        )

    # Compute the 'keep' term of the covariance evolution path
    path_cov *= (1.0 - lr_cov)

    # Precompute the coefficient
    coeff = update_switch * sqrt(lr_cov * (2.0 - lr_cov) * mu_eff)

    # Update the covariance evolution path with the elite mean step
    path_cov += coeff * elite_mean_step

    # Get the adjusted rank-1 learning rate
    lr_rank_one_adj = (1.0-update_switch) * lr_rank_one * lr_cov * (2.0-lr_cov)

    # Synchronize steps and weights with sorted fitness order
    steps_sorted = steps[sorted_indices]
    weights_sorted = weights.copy()

    # Get the indices for the negative weights
    neg_indices = where(weights_sorted < 0.0)[0]

    # Check if any negative weights are present
    if len(neg_indices) > 0:

        # Extract negative steps into a contiguous block
        steps_neg = steps_sorted[neg_indices]

        # Perform an isotropic transformation
        steps_neg_tr = (left_basis.T @ steps_neg.T).T * inv_root_core_vec

        # Get the squared norms of the isotropic vectors
        squared_z_norms = nsum(steps_neg_tr**2, axis=1)

        # Get the scaling factors
        factors = dim / (squared_z_norms + 1e-15)

        # Rescale the weights to guarantee positive definiteness
        weights_sorted[neg_indices] *= minimum(1.0, factors)

        # Enumerate the indices of the negative weights
        for i, index in enumerate(neg_indices):

            # Adjust the sorted weights
            weights_sorted[index] *= min(1.0, factors[i])

    # Compute the rank-mu term for the covariance
    rank_mu_term = (
        (steps_sorted.T * weights_sorted.reshape(1, -1)) @ steps_sorted
        )

    # Update the covariance matrix
    cov *= (1.0 - lr_rank_one - lr_rank_mu + lr_rank_one_adj)
    cov += lr_rank_one * outer(path_cov, path_cov)
    cov += lr_rank_mu * rank_mu_term

    return sigma_update
