"""
Dynamical low-rank covariance matrix adaptation evolution strategy
(DLR-CMA-ES).
"""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

from collections import deque
from math import inf
from numba import njit, types
from numpy import (
    add, arange, argmax, argmin, argsort, array, ascontiguousarray, clip, diag,
    exp, eye, float64, full, isinf, log, maximum, minimum, ptp, ones, outer,
    sqrt, where, zeros)
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from numpy import mean as nmean
from numpy import min as nmin
from numpy import sum as nsum
from numpy.linalg import eigh, norm, qr
from numpy.random import default_rng

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

    low_rank_dimension : int, default=None
        Initial rank of the approximation. Defaults to `number_of_variables`.

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
            low_rank_dimension=None,
            maximum_iterations=100000,
            maximum_wall_time=43200,
            fitness_threshold=-inf,
            fitness_window_size=50,
            tolerance=1e-6,
            sigma_threshold=1e-8,
            min_log_level='debug',
            callback=None,
            random_state=42):

        # Initialize the logger
        self.logger = Logging('DLR-CMA-ES', min_log_level)

        # Log a message about the initialization
        self.logger.info('Initializing DLR-CMA-ES with SPD-BUG integrator ...')

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

        # Determine the integrator rank
        self.rank = min(
            max(1, low_rank_dimension or self._number_of_variables),
            self._number_of_variables
            )

        # Initialize the base weights
        self._base_weights = (
            log((self._pop_size + 1) / 2) - log(arange(1, self._pop_size + 1))
            ).reshape(-1, 1)

        # Initialize the elite size
        self._elite_size = int(nsum(self._base_weights > 0))

        # Determine the sums of positive and negative base weights
        self._bw_pos_sum = nsum(self._base_weights[:self._elite_size])
        self._bw_neg_sum = nsum(self._base_weights[self._elite_size:])

        # Initialize the variance effective selection mass
        self._mu_eff = (
            self._bw_pos_sum**2 /
            nsum(self._base_weights[:self._elite_size]**2)
            )
        self._mu_eff_neg = (
            self._bw_neg_sum**2 /
            nsum(self._base_weights[self._elite_size:]**2)
            )

        # Initialize the dynamic variables
        self._lr_sigma = 0.0
        self._lr_cov = 0.0
        self._lr_rank_one = 0.0
        self._lr_rank_mu = 0.0
        self._lr_mean = 1.0
        self._weights = zeros((self._pop_size, 1), dtype=float64)
        self._damp_sigma = 0.0
        self._expected_path_length = 0.0

        # Update the dynamic variables
        self._update_dynamics()

        # Initialize the bound/constraint handling parameters
        self._squared_bound_errors = None
        self._gamma = None

        # Initialize the adaptive state variables, arrays, and matrices
        self._wall_start = None
        self._opt_iter = 0
        self._sigma = initial_sigma

        self._steps = zeros(
            (self._pop_size, self._number_of_variables), order='F',
            dtype=float64
            )
        self._population = zeros(
            (self._pop_size, self._number_of_variables), order='F',
            dtype=float64
            )

        self._path_sigma = zeros(self._number_of_variables, dtype=float64)
        self._path_cov = zeros(self._number_of_variables, dtype=float64)
        self._mean = zeros(self._number_of_variables, dtype=float64)

        self._basis = eye(
            self._number_of_variables, order='F', dtype=float64
            )
        self._core_vector = ones(self.rank, dtype=float64)

        self._psi = full(
            self._number_of_variables, 1.0, dtype=float64
            )

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness = None
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None, 'iterations': None
            }

    def _update_dynamics(self):
        """Update the dynamic variables."""

        # Get the current rank and dimensionality of the problem
        rank = self.rank
        dim = self._number_of_variables

        # Initialize the step size learning rate (depending on rank)
        self._lr_sigma = (self._mu_eff + 2.0) / (rank + self._mu_eff + 5.0)

        # Initialize the covariance learning rates (depending on full space)
        self._lr_cov = (
            (4.0 + self._mu_eff / dim) / (dim + 4.0 + 2.0 * self._mu_eff / dim)
            )
        self._lr_rank_one = (
            2.0 / ((dim + 1.3)**2 + self._mu_eff)
            )
        self._lr_rank_mu = min(
            1.0 - self._lr_rank_one,
            2.0 * ((0.25 + self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                   ((dim + 2.0)**2 + 2.0 * self._mu_eff / 2.0))
            )

        # Initialize the default mean learning rate
        self._lr_mean = 1.0

        # Determine the alpha values
        alpha_mu_neg = 1.0 + self._lr_rank_one / (self._lr_rank_mu + 1e-12)
        alpha_mu_eff_neg = (
            1.0 + (2.0 * self._mu_eff_neg) / (self._mu_eff + 2.0)
            )
        alpha_posdef_neg = (
            (1.0 - self._lr_rank_one - self._lr_rank_mu)
            / (dim * self._lr_rank_mu + 1e-12)
            )
        alpha_min = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_posdef_neg)

        # Generate a mask for positive base weights
        pos_mask = self._base_weights > 0

        # Set the weights
        self._weights[pos_mask] = (
            (1.0 / self._bw_pos_sum) * self._base_weights[pos_mask]
            )
        self._weights[~pos_mask] = (
            (alpha_min / (nabs(self._bw_neg_sum) + 1e-12))
            * self._base_weights[~pos_mask]
            )

        # Initialize the damping coefficient (depending on rank)
        self._damp_sigma = (
            1.0 + 2.0 * max(
                0.0, sqrt((self._mu_eff - 1.0) / (rank + 1.0)) - 1.0)
            + self._lr_sigma
            )

        # Initialize the expected path length (depending on rank)
        self._expected_path_length = (
            sqrt(rank) * (1.0 - 1.0 / (4.0 * rank) + 1.0 / (21.0 * rank**2))
            )

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

        # Get the current rank
        rank = self.rank

        # Sample from the standard multivariate Gaussian
        num_random = (
            self._pop_size - 2 if self.gradient is not None
            else self._pop_size
            )
        z_low_rank = self._rng.standard_normal((num_random, rank))
        z_noise = self._rng.standard_normal(
            (num_random, self._number_of_variables)
            )

        # Calculate the root of the covariance matrix
        root_cov = self._basis[:, :rank] * sqrt(self._core_vector)

        # Transform the samples into low-rank and noisy components
        structured_part = z_low_rank @ root_cov.T
        noise_part = z_noise * sqrt(self._psi)

        # Get the steps by adding the components
        self._steps[:num_random] = structured_part + noise_part

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            basis_grad = root_cov.T @ gradient
            natural_gradient = root_cov @ basis_grad + self._psi * gradient

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
        Update the state variables and perform an SPD-BUG step.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Get the current rank
        rank = self.rank

        # Update the state variables
        (elite_indices, path_sigma_new, sigma_new, mean_new, path_cov_new,
         update_switch)  = _tell(
            fitness,
            self._steps,
            self._weights,
            self._basis[:, :rank],
            self._core_vector,
            self._path_sigma[:rank],
            self._path_cov[:rank],
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
        self._path_sigma[:rank] = path_sigma_new
        self._sigma = sigma_new
        self._mean = mean_new
        self._path_cov[:rank] = path_cov_new

        # Store the old basis before updating
        basis_old = self._basis[:, :rank].copy()

        # Update the low-rank and noise factors
        basis_new, core_vector_new, psi_new, rank_new = _spd_bug_step(
            self._basis[:, :rank],
            self._core_vector,
            self._psi,
            self._steps[elite_indices],
            self._weights,
            self._path_cov[:rank],
            self._lr_cov,
            self._lr_rank_one,
            self._lr_rank_mu,
            update_switch=update_switch,
            force_expansion=self.check_rank_expansion()
            )

        # Compute the rotation matrix (from basis_old to basis_new)
        projection_matrix = basis_new.T @ basis_old

        # Rotate the step size evolution path
        rotated_path_sigma = projection_matrix @ self._path_sigma[:rank]

        # Rotate the covariance evolution path
        rotated_path_cov = projection_matrix @ self._path_cov[:rank]

        # Check if the subspace shrinks
        if rank_new < rank:

            # Get the retention ratio
            ratio = rank_new / rank

            # Damp the evolution paths
            rotated_path_cov *= ratio
            rotated_path_sigma *= ratio

        # Save the factor states
        self._basis[:, :rank_new] = basis_new
        self._core_vector = core_vector_new
        self._psi[:] = psi_new
        self.rank = rank_new

        # Assign the rotated sigma and covariance evolution path
        self._path_sigma[:rank_new] = rotated_path_sigma
        self._path_cov[:rank_new] = rotated_path_cov

        # Update the dynamic variables
        self._update_dynamics()

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
            while not self.check_termination():

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
            Indicator for expanding the rank.
        """

        # Get the current rank
        rank = self.rank

        # Initialize the number of positive and required votes
        votes = 0
        required = 2

        # Compute the eigenvalues of the covariance matrix
        cov_evals = self._core_vector + self._psi[:rank]

        # Get the maximum eigenvalue
        max_eval = nmax(cov_evals)

        # Check if the current rank is smaller than the problem dimension
        if rank < self._number_of_variables:

            # Get the minimum eigenvalue by considering the full noise term
            min_eval = minimum(nmin(cov_evals), nmin(self._psi[rank:]))

        else:

            # Get the minimum eigenvalue
            min_eval = nmin(cov_evals)

        # Check if the condition number is greater than 1e6
        if max_eval / (min_eval + 1e-15) > 1e6:

            # Increment the votes
            votes += 1

        # Check if the fitness history has been completely filled
        if len(self._fitness_history) == self._fitness_history.maxlen:

            # Get the fitness history
            history = array(self._fitness_history)

            # Get the mean fitness
            fit_mean = nmean(history)

            # Get the fitness range
            fit_range = ptp(history)

            # Check if the fitness stagnates prior to convergence
            if (fit_range > self.tolerance and
                    (fit_range / (nabs(fit_mean) + 1e-15) < 1e-3)):

                # Increment the votes
                votes += 1

            # Get the centered time steps
            indices = arange(len(history)) - (len(history) - 1) / 2.0

            # Compute the slope over the time steps
            slope = nsum(indices * (history - fit_mean)) / nsum(indices**2)

            # Check if the linear trend is too small
            if abs(slope) / (nabs(fit_mean) + 1e-15) < 1e-4:

                # Increment the votes
                votes += 1

        return votes >= required

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

        # Get the current rank
        rank = self.rank

        # Compute the eigenvalues of the covariance matrix
        cov_evals = self._core_vector + self._psi[:rank]

        # Get the maximum eigenvalue
        max_eval = nmax(cov_evals)

        # Check if the current rank is smaller than the problem dimension
        if rank < self._number_of_variables:

            # Get the minimum eigenvalue by considering the full noise term
            min_eval = minimum(nmin(cov_evals), nmin(self._psi[rank:]))

        else:

            # Get the minimum eigenvalue from the spectrum
            min_eval = nmin(cov_evals)

        # Check if any eigenvalue is zero or the condition number explodes
        if min_eval <= 0 or (max_eval / (min_eval + 1e-15)) >= 1e14:

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

        # Get the longest axis
        root_cov = self._basis[:, :self.rank] * sqrt(self._core_vector)
        longest_axis = root_cov[:, argmax(norm(root_cov, axis=0))]

        # Check if the mean is not shifted along the longest axis
        if all(self._mean == self._mean + 0.1 * self._sigma * longest_axis):

            # Add the solver info
            self._result['solver_info'] = 'NO_EFFECT_AXIS'

            return True

        return False


# Set the variable types
f8_1d = types.float64[:]
f8_2d = types.float64[:, :]
f8 = types.float64
i8_1d = types.int64[:]
i8 = types.int64
bo = types.bool_

@njit(
    types.Tuple((i8_1d, f8_1d, f8, f8_1d, f8_1d, f8))(
        # Return: Tuple(elite_indices, path_sigma, sigma, mean, path_cov)
        f8_1d,          # fitness
        f8_2d,          # steps
        f8_2d,          # weights
        f8_2d,          # basis
        f8_1d,          # evals
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
    fitness, steps, weights, basis, core_vector, path_sigma, path_cov, mean,
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

    # Calculate the inverse rooted eigenvalues
    inv_root_core_vec = 1.0 / (sqrt(core_vector) + 1e-15)

    # Transform the elite mean step
    latent_step = basis.T @ elite_mean_step
    elite_mean_step_tr = inv_root_core_vec * latent_step

    # Update the step size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff) * elite_mean_step_tr
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
    path_cov += coeff * latent_step

    return elite_indices, path_sigma, sigma, mean, path_cov, update_switch


# @njit(
#     types.Tuple((f8_2d, f8_1d, f8_1d, i8))(
#         # Return: Tuple(basis_new, ev_new, psi_new, rank_new)
#         f8_2d,          # basis
#         f8_1d,          # core_vector
#         f8_1d,          # psi
#         f8_2d,          # elite_steps
#         f8_2d,          # weights
#         f8_1d,          # path_cov
#         f8,             # lr_cov
#         f8,             # lr_rank_one
#         f8,             # lr_rank_mu
#         f8,             # update_switch
#         bo              # force_expansion
#         ),
#     fastmath=True
#     )
def _spd_bug_step(
        basis, core_vector, psi, elite_steps, weights, path_cov, lr_cov,
        lr_rank_one, lr_rank_mu, update_switch, force_expansion):
    """Perform the SPD-BUG step with adaptive rank truncation."""

    from numpy import hstack, isfinite, newaxis, nan_to_num, sign
    from numpy.linalg import LinAlgError

    dim, rank = basis.shape
    max_rank = min(2 * rank, dim)

    # 0. Berechne den angepassten Term genau wie im klassischen CMA-ES
    lr_rank_one_adj = (1.0 - update_switch) * lr_rank_one * lr_cov * (2.0 - lr_cov)

    # Fortlaufenden Speicher garantieren
    B = ascontiguousarray(basis)

    # 1. Prüfen, ob die Richtung 'psi' den Unterraum erweitern kann
    projection_weights = B.T @ psi
    psi_orthogonal = psi - B @ projection_weights
    psi_norm = norm(psi_orthogonal)

    # CMA-ES Dämpfung auf das alte Kernsystem anwenden
    c_decay = 1.0 - lr_rank_one - lr_rank_mu + lr_rank_one_adj

    if psi_norm > 1e-10 and rank < max_rank:
        # Basis temporär um die neue orthogonale Richtung erweitern
        psi_new_dir = psi_orthogonal / psi_norm
        B_augmented = hstack([B, psi_new_dir[:, newaxis]])

        # Altes Kernsystem gedämpft einbetten (Größe: rank + 1)
        K_expanded = zeros((rank + 1, rank + 1))
        K_expanded[:rank, :rank] = diag(core_vector * c_decay)

        # WEGWEISENDE KORREKTUR: path_cov lebt im Unterraum (Größe rank).
        # Da wir die Basis um 1 erweitert haben, hängen wir eine 0 an den Pfad an.
        y_path = zeros(rank + 1)
        y_path[:rank] = path_cov

    else:
        # Keine Erweiterung, Update im bestehenden Unterraum
        B_augmented = B
        K_expanded = diag(core_vector * c_decay)
        # Keine Erweiterung der Basis -> y_path entspricht exakt dem aktuellen path_cov
        y_path = path_cov

    # 2. Projektion der hochdimensionalen CMA-ES Update-Terme in den BUG-Unterraum
    # Rank-One Term: Evolutionspfad projizieren
    K_rank_one = lr_rank_one * outer(y_path, y_path)

    # Rank-Mu Term: Projiziere alle Schritte (Form: rank_augmented x pop_size, hier: 1000 x 12)
    Y_steps = B_augmented.T @ elite_steps.T

    # Wie viele Individuen (Spalten) haben wir in dieser Iteration tatsächlich?
    pop_size_current = Y_steps.shape[1]

    # Wir schneiden die Gewichte exakt auf die Anzahl der aktuellen Individuen zu
    # und richten sie als Zeilenvektor (1, pop_size) aus
    w_sliced = weights

    # Hocheffizientes spaltenweises Skalieren via Broadcasting:
    # Jedes Individuum (Spalte) bekommt sein passendes CMA-ES Gewicht.
    K_rank_mu = lr_rank_mu * ((Y_steps @ w_sliced) @ Y_steps.T)

    # 3. Das kombinierte BUG-Kernsystem aufstellen
    K_total = K_expanded + K_rank_one + K_rank_mu

    # Numerischer Schutz gegen NaN/Inf (falls sigma kollabiert)
    if not isfinite(K_total).all():
        K_total = nan_to_num(K_total, nan=0.0, posinf=0.0, neginf=0.0)

    # Perfekte Symmetrie erzwingen
    K_total = 0.5 * (K_total + K_total.T)

    # 4. Eigenwertzerlegung des kleinen Kernsystems (viel stabiler als SVD bei CMA-ES)
    try:
        S_new, U_core = eigh(K_total)

        # Absteigend nach der Größe der Eigenwerte sortieren
        idx = argsort(S_new)[::-1]
        S_new = S_new[idx]
        U_core = U_core[:, idx]

        # Negative Eigenwerte durch numerisches Rauschen abschneiden
        S_new = clip(S_new, a_min=0.0, a_max=None)

    except LinAlgError:
        # Fallback bei numerischem Totalausfall
        dim_k = K_total.shape[0]
        S_new = ones(dim_k) * 1e-15
        S_new[:len(core_vector)] = core_vector
        U_core = eye(dim_k)

    # 5. Adaptives Abschneiden (Truncation)
    # =========================================================================
    if force_expansion:
        target_rank = min(max_rank, len(S_new))
        truncated_energy = 0.0
    else:
        energy_threshold = 1e-7 * S_new[0] if len(S_new) > 0 else 1e-7
        keep_mask = S_new > energy_threshold
        target_rank = nsum(keep_mask)

        # WICHTIG: Berechne die Energie der weggeschnittenen Richtungen!
        # Diese Varianzen dürfen nicht verpuffen, sondern müssen zurück zu Psi fließen.
        truncated_energy = nsum(S_new[~keep_mask]) if nsum(~keep_mask) > 0 else 0.0

        # Auf der Sphere darf der Rang ruhig klein werden (z. B. mindestens 2 oder 3)
        target_rank = max(2, min(target_rank, max_rank))

    # =========================================================================
    # 6. Rückprojektion & Relativer Floor
    # =========================================================================
    new_basis = B_augmented @ U_core[:, :target_rank]

    max_ev = S_new[0]
    relative_floor = max_ev * 1e-7
    absolute_floor = 1e-15
    floor_value = max(relative_floor, absolute_floor)

    new_core = clip(S_new[:target_rank], a_min=floor_value, a_max=None)
    new_rank = len(new_core)

    # =========================================================================
    # 7. Numerische Nachreinigung (QR)
    # =========================================================================
    new_basis, R = qr(new_basis, mode='reduced')
    new_basis = new_basis * sign(diag(R))

    # =========================================================================
    # 8. Dynamisches Psi-Update mit Energie-Rückfluss
    # =========================================================================
    # Das alte Psi dämpfen (CMA-ES Dämpfung), genau wie das Kernsystem!
    psi_decayed = psi * c_decay

    # Der orthogonale Anteil des aktuellen Schritts (Residuum)
    new_psi_projection = new_basis @ (new_basis.T @ psi_decayed)
    residual_psi = psi_decayed - new_psi_projection

    # ENERGIE-ERHALTUNG: Wir fügen die weggeschnittene Varianz (verteilt auf die
    # verbleibenden unstrukturierten Dimensionen) als isotropen Anteil hinzu.
    unstructured_dims = dim - target_rank
    if unstructured_dims > 0:
        isotropic_injection = truncated_energy / unstructured_dims
        new_psi = residual_psi + isotropic_injection
    else:
        new_psi = residual_psi

    # Relativer Bodenschutz auch für Psi, damit es proportional zu S_new mitschrumpft,
    # anstatt auf einem harten 1e-3 einzufrieren!
    psi_floor = max_ev * 1e-7
    new_psi = clip(new_psi, a_min=max(psi_floor, 1e-15), a_max=None)

    return new_basis, new_core, new_psi, new_rank
