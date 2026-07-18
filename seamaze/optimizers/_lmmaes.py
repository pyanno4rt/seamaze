"""Limited-memory matrix adaptation evolution strategy (LM-MA-ES)."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from signal import getsignal, SIGINT, signal
from time import time

from collections import deque
from math import inf
from numba import njit, types
from numpy import (
    add, arange, argmin, argsort, array, clip, exp, float64, full, isinf,
    isnan, log, maximum, ptp, sqrt, where, zeros)
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from numpy import mean as nmean
from numpy import min as nmin
from numpy import sum as nsum
from numpy.linalg import norm
from numpy.random import default_rng

# %% Internal package import

from seamaze.logging import Logging
from seamaze.utils import make_compat

# %% Class definition


class LMMAES:
    """
    Limited-memory matrix adaptation evolution strategy (LM-MA-ES) class.

    This class implements the LM-MA-ES algorithm according to the paper:

        Loshchilov, I., Glasmachers, T. and Beyer, H.-G. (2017). Limited-Memory
        Matrix Adaptation for Large Scale Black-box Optimization. ArXiv,
        abs/1705.06693.

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

    memory_size : int, default=None
        The limited memory capacity parameter 'm'. Determines the maximum
        number of directional history vectors retained in memory. Defaults to
        4 + int(3*log(`number_of_variables`)).

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

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}, \
        default='debug'
        Minimum logging level for passing messages to the console.

    callback : Callable[[LMMAES], None], default=None
        Optional function called at the end of each iteration. Must accept
        the solver instance.

    random_state : int or None, default=42
        Control seed for the random number generator.
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
            memory_size=None,
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
        self.logger = Logging('LM-MA-ES', min_log_level)

        # Log a message about the initialization
        self.logger.info('Initializing LM-MA-ES ...')

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
            if lower_variable_bounds is None else lower_variable_bounds
            )
        self.upper_variable_bounds = (
            full(self._number_of_variables, inf)
            if upper_variable_bounds is None else upper_variable_bounds
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
        self._pop_size += (2 if gradient is not None else 0)

        # Initialize the memory size
        self._memory_size = (
            4 + int(3 * log(self._number_of_variables))
            if memory_size is None else memory_size
            )
        self._memory_size += (2 if gradient is not None else 0)

        # Initialize the base weights
        base_weights = (
            log((self._pop_size + 1) / 2) - log(arange(1, self._pop_size + 1))
            )

        # Initialize the elite size
        self._elite_size = int(nsum(base_weights > 0))

        # Determine the sums of positive and negative base weights
        bw_pos_sum = nsum(base_weights[:self._elite_size])

        # Set the weights
        self._weights = (
            base_weights[:self._elite_size] / bw_pos_sum
            ).reshape(-1, 1)

        # Initialize the variance effective selection mass
        self._mu_eff = nsum(self._weights)**2 / nsum(self._weights**2)

        # Initialize the learning rates
        epsilon = 1e-12
        self._lr_sigma = clip(
            (2 * self._pop_size) / self._number_of_variables, epsilon, 0.99
            )
        exponents = arange(self._memory_size)
        self._lr_mem = clip(
            1.0 / clip(1.5**exponents * self._number_of_variables, 1e-15, inf),
            epsilon, 0.99
            )
        self._lr_cov = clip(
            self._pop_size / clip(
                4.0**exponents * self._number_of_variables, 1e-15, inf),
            epsilon, 0.99
            )
        self._lr_mean = 1.0

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

        self._zsamples = zeros(
            (self._pop_size, self._number_of_variables), dtype=float64
            )
        self._steps = zeros(
            (self._pop_size, self._number_of_variables), dtype=float64
            )
        self._population = zeros(
            (self._pop_size, self._number_of_variables), dtype=float64
            )

        self._path_sigma = zeros(self._number_of_variables, dtype=float64)
        self._mean = zeros(self._number_of_variables, dtype=float64)

        self._memory = zeros(
            (self._memory_size, self._number_of_variables), dtype=float64
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

        # Initialize the stop flag
        self._stop_requested = False

    def ask(self):
        """Generate a new population."""

        # Partially sample from a standard multivariate Gaussian
        num_random = (
            self._pop_size - 2 if self.gradient is not None
            else self._pop_size
            )
        self._zsamples[:num_random] = self._rng.standard_normal(
            (num_random, self._number_of_variables)
            )

        # Copy the samples to the steps
        self._steps[:] = self._zsamples

        # Get the current iteration of memory filling
        num_iter = min(self._opt_iter, self._memory_size)

        # Transform the steps
        self._steps[:num_random] = _transform_steps(
            self._steps[:num_random], self._memory, self._lr_mem, num_iter
            )

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Create a temporary copy
            natural_gradient = gradient.copy()

            # Transform the natural gradient
            natural_gradient = _transform_gradient(
                natural_gradient, self._memory, self._lr_cov, num_iter
                )

            # Compute the rescaling factor
            rescale = 1.0 / sqrt(gradient @ natural_gradient + 1e-15)

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

    def tell(
            self,
            fitness):
        """
        Update the state variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Update the state variables
        path_sigma_new, mean_new, sigma_new, memory_new = _tell(
            fitness,
            self._zsamples,
            self._steps,
            self._weights,
            self._path_sigma,
            self._mean,
            self._sigma,
            self._memory,
            self._lr_sigma,
            self._lr_cov,
            self._lr_mean,
            self._mu_eff,
            self._damp_sigma,
            self._expected_path_length,
            self._elite_size,
            self._memory_size
            )

        # Save the state variables
        self._path_sigma[:] = path_sigma_new
        self._mean[:] = mean_new
        self._sigma = sigma_new
        self._memory[:] = memory_new

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
            while self.check_termination() is False:

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
                    f'f={round(self._result["optimal_value"], 6)}'
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

        # Check if the first iteration has passed
        if self._opt_iter > 0:

            # Get the maximum absolute value from the memory matrix
            m_max = nmax(nabs(self._memory))

            # Check if the memory matrix contains infinity or NaN
            if isinf(m_max) or nany(isnan(self._memory)):

                # Add the solver info
                self._result['solver_info'] = 'MEMORY_MATRIX_INSTABLE'

                return True

            # Check if the longest axis is already "dead"
            if nmax(nabs(self._memory[0, :])) < 1e-15:

                # Add the solver info
                self._result['solver_info'] = 'MEMORY_MATRIX_COLLAPSED'

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

        # Check if the first memory vector has been saved
        if min(self._opt_iter, self._memory_size) > 0:

            # Get the longest axis by the first memory vector
            longest_axis = self._memory[0, :]

            # Check if a marginal shift along the axis has no effect
            if all(self._mean == self._mean + 0.1 * self._sigma * longest_axis):

                # Add the solver info
                self._result['solver_info'] = 'NO_EFFECT_AXIS'

                return True

        return False


# Set the variable types
f8_2d = types.float64[:, :]
f8_2d_c = types.float64[:, ::1]
f8_1d = types.float64[:]
f8_1d_c = types.float64[::1]
f8 = types.float64
i8 = types.int64

@njit(
      f8_1d(            # Return: gradient
        f8_1d_c,        # gradient
        f8_2d_c,        # memory
        f8_1d_c,        # lr_cov
        i8,             # num_iter
        ),
    fastmath=True
    )
def _transform_gradient(gradient, memory, lr_cov, num_iter):
    """Transform the gradient with the memory matrix."""

    # Loop backward over the memory index
    for j in range(num_iter - 1, -1, -1):

        # Project the gradient onto the j-th memory vector
        dot_product = gradient @ memory[j, :]

        # Transform the gradient
        gradient *= (1.0 - lr_cov[j])
        gradient += (lr_cov[j] * dot_product) * memory[j, :]

    # Loop foward over the memory index
    for j in range(num_iter):

        # Project the gradient onto the j-th memory vector
        dot_product = gradient @ memory[j, :]

        # Transform the gradient
        gradient *= (1.0 - lr_cov[j])
        gradient += (lr_cov[j] * dot_product) * memory[j, :]

    return gradient

@njit(
      f8_2d(            # Return: steps
        f8_2d_c,        # steps
        f8_2d_c,        # memory
        f8_1d_c,        # lr_mem
        i8,             # num_iter
        ),
    fastmath=True
    )
def _transform_steps(steps, memory, lr_mem, num_iter):
    """Transform the steps with the memory matrix."""

    # Loop over the current memory
    for j in range(num_iter):

        # Project the steps onto the j-th memory vector
        dot_products = steps @ memory[j, :]

        # Loop over the step vectors
        for idx in range(steps.shape[0]):

            # Transform the step vector
            steps[idx, :] = (
                (1.0 - lr_mem[j]) * steps[idx, :]
                + lr_mem[j] * dot_products[idx] * memory[j, :]
                )

    return steps

@njit(
      types.Tuple((f8_1d, f8_1d, f8, f8_2d))(
        # Return: path_sigma, mean, sigma, memory
        f8_1d,          # fitness
        f8_2d,          # zsamples
        f8_2d,          # steps
        f8_2d,          # weights
        f8_1d,          # path_sigma
        f8_1d,          # mean
        f8,             # sigma
        f8_2d,          # memory
        f8,             # lr_sigma
        f8_1d,          # lr_cov
        f8,             # lr_mean
        f8,             # mu_eff
        f8,             # damp_sigma
        f8,             # expected_path_length
        i8,             # elite_size
        i8,             # num_iter
        ),
    fastmath=True
    )
def _tell(
    fitness, zsamples, steps, elite_weights, path_sigma, mean, sigma, memory,
    lr_sigma, lr_cov, lr_mean, mu_eff, damp_sigma, expected_path_length,
    elite_size, memory_size):
    """Update the state variables."""

    # Get the elite indices
    sorted_indices = argsort(fitness)
    elite_indices = sorted_indices[:elite_size]

    # Initialize the elite mean step
    elite_mean_step = nsum(
        steps[elite_indices] * elite_weights, axis=0
        )

    # Calculate the latent isotropic step
    latent_step = nsum(
        zsamples[elite_indices] * elite_weights, axis=0
        )

    # Update the step size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff) * latent_step
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

    # Loop over the rows of the memory matrix
    for i in range(memory_size):

        # Update the row
        memory[i, :] *= (1.0 - lr_cov[i])
        memory[i, :] += (
            sqrt(mu_eff * lr_cov[i] * (2.0 - lr_cov[i])) * latent_step
            )

    return path_sigma, mean, sigma, memory
