"""
Dynamical low-rank covariance matrix adaptation evolution strategy
(DLR-CMA-ES).
"""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from time import time

from collections import deque
from math import inf
from numba import njit
from numpy import (
    add, allclose, arange, argmax, argmin, argsort, array, clip, copyto, diag,
    empty, exp, eye, fill_diagonal, float64, full, isinf, log, matmul, maximum,
    minimum, multiply, ones, outer, sqrt, take, where, zeros)
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
from seamaze.optimizers import LowRankIntegrator
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

    low_rank_integrator : \
        {'fixedBUG', 'fixedsymmetricBUG', 'fixedaugBUG', 'fixedSPDBUG', \
         'isofixedSPDBUG', 'augBUG', 'symmetricaugBUG', 'SPDaugBUG', \
         'isoSPDaugBUG'}, default='fixedsymmetricBUG'
        Name of the low-rank integrator.

    low_rank_dimension : int, default=None
        Initial rank of the approximation. Defaults to
        `number_of_variables` // 10.

    low_rank_tolerance_rel : float, default=1e-2
        Relative tolerance of the rank truncation.

    low_rank_tolerance_abs : float, default=1e-8
        Absolute tolerance of the rank truncation.

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
        Frequency of the low-rank factorization update (in generations).
        Larger values (e.g. 10) can significantly speed up the algorithm for
        high-dimensional problems. Defaults to \
        max(1, 0.1/(`number_of_variables`*β)), where β is the sum of the
        rank-1 and rank-mu learning rates.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}, \
        default='debug'
        Minimum logging level for passing messages to the console.

    callback : Callable[[DLRCMAES], None], default=None
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
            low_rank_integrator='SPDaugBUG',
            low_rank_dimension=None,
            low_rank_tolerance_rel=1e-2,
            low_rank_tolerance_abs=1e-8,
            maximum_iterations=100000,
            maximum_wall_time=43200,
            fitness_threshold=-inf,
            fitness_window_size=50,
            tolerance=1e-6,
            sigma_threshold=1e-8,
            update_interval=None,
            min_log_level='debug',
            callback=None):

        # Initialize the logger
        self.logger = Logging('DLR-CMA-ES', min_log_level)

        # Log a message about the initialization
        self.logger.info(
            f'Initializing DLR-CMA-ES with "{low_rank_integrator}"...')

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
        rank = min(
            max(1, low_rank_dimension or self._number_of_variables),
            self._number_of_variables
            )

        # Initialize the dynamical low-rank integrator
        self.integrator = LowRankIntegrator(
            name=low_rank_integrator,
            rank=rank,
            truncation_tolerance_rel=low_rank_tolerance_rel,
            truncation_tolerance_abs=low_rank_tolerance_abs,
            N_conserved_basis=0,
            K_step=self.K_step,
            L_step=self.L_step,
            S_step=self.S_step
            )

        # Initialize the integrator buffers
        self.integrator.set_buffers(self._number_of_variables)

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

        # Initialize the learning rates, damping, and expected path length
        self._update_dynamics()

        # Adjust the weights
        _alpha_mu_neg = 1.0 + self._lr_rank_one / self._lr_rank_mu
        _alpha_mu_eff_neg = 1.0 + (2.0 * mu_eff_neg) / (self._mu_eff + 2.0)
        _alpha_posdef_neg = (
            (1.0 - self._lr_rank_one - self._lr_rank_mu)
            / (rank * self._lr_rank_mu)
            )
        _alpha_min = min(_alpha_mu_neg, _alpha_mu_eff_neg, _alpha_posdef_neg)

        self._weights = array([
            (1.0 / bw_pos_sum) * wi if wi > 0.0 else
            (_alpha_min / abs(bw_neg_sum)) * wi
            for wi in base_weights]
            )
        self._weights_2d = self._weights[:, None]

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
        self._elite_steps = zeros(
            (self._elite_size, self._number_of_variables), order='F',
            dtype=float64
            )
        self._population = zeros(
            (self._pop_size, self._number_of_variables), order='F',
            dtype=float64
            )

        self._path_sigma = zeros(rank, dtype=float64)
        self._path_cov = zeros((rank, 1), order='F', dtype=float64)
        self._mean = zeros(self._number_of_variables, dtype=float64)

        self._left_basis = zeros(
            (self._number_of_variables, self._number_of_variables), order='F',
            dtype=float64
            )
        self._left_basis[:rank, :rank] = eye(rank)
        self._core_matrix = eye(rank, dtype=float64)

        self._core_eig_cache = (ones(rank), eye(rank))
        self._root_cov = zeros(
            (self._number_of_variables, self._number_of_variables), order='F',
            dtype=float64
            )
        self._root_cov[:rank, :rank] = eye(rank)

        self._left_basis_old = empty(
            (self._number_of_variables, self._number_of_variables), order='F',
            dtype=float64
            )
        self._path_cov_padded = zeros(
            (self._number_of_variables, 1), order='F', dtype=float64
            )
        self._elite_projection = zeros(
            (self._elite_size, self._number_of_variables), order='F',
            dtype=float64
            )
        self._path_full = empty(
            (self._number_of_variables, 1), order='F', dtype=float64)
        self._weighted_steps = empty(
            (self._elite_size, self._number_of_variables), order='F',
            dtype=float64
            )
        self._mu_projection = empty(
            (self._elite_size, self._number_of_variables), order='F',
            dtype=float64
            )

        # Initialize the stopping criteria and tracking variables
        self.maximum_iterations = maximum_iterations
        self.maximum_wall_time = maximum_wall_time
        self.fitness_threshold = fitness_threshold
        self.sigma_threshold = sigma_threshold or 0.0
        self.tolerance = tolerance
        self._fitness = None
        self._fitness_history = deque(maxlen=fitness_window_size)
        self._update_interval = (
            max(1, int(0.1/(self._number_of_variables*(
                self._lr_rank_one + self._lr_rank_mu))))
            if update_interval is None else update_interval
            )
        self._callback = callback
        self._result = {
            'optimal_point': None, 'optimal_value': inf, 'solver_info': None,
            'wall_time': None, 'iterations': None
            }

    def K_step(
            self,
            left_basis_product,
            right_basis,
            _time_step):
        """
        Perform the K-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis_product :
            ...

        right_basis :
            ...

        _time_step :
            ...

        Returns
        -------
        ...
        """

        return self._update_rank_terms(left_basis_product, right_basis)

    def L_step(
            self,
            left_basis,
            right_basis_product,
            _time_step):
        """
        Perform the L-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis :
            ...

        right_basis_product :
            ...

        _time_step :
            ...

        Returns
        -------
        ...
        """

        return self._update_rank_terms(right_basis_product, left_basis)

    def S_step(
            self,
            left_basis,
            core_matrix,
            *args):
        """
        Perform the S-step of the dynamical low-rank integrator.

        Parameters
        ----------
        left_basis :
            ...

        core_matrix :
            ...

        *args : tuple
            ...

        Returns
        -------
        ...
        """

        # Get the ranks of the covariance evolution path and the core matrix
        path_rank = self._path_cov.shape[0]
        core_rank = core_matrix.shape[0]

        # Check if the ranks are different
        if path_rank != core_rank:

            # Add a padding to the covariance evolution path
            path_cov_padded = self._path_cov_padded[:core_rank, :]
            path_cov_padded.fill(0.0)
            path_cov_padded[:path_rank, :] = self._path_cov

        else:

            # Use the covariance evolution path directly
            path_cov_padded = self._path_cov

        # Get the projections of the covariance evolution path and elite steps
        matmul(
            self._elite_steps, left_basis,
            out=self._elite_projection[:, :core_rank])
        elite_projection = self._elite_projection[:, :core_rank]

        ps_norm = norm(self._path_sigma)
        update_switch = (
            1.0
            if ps_norm / sqrt(
                    1.0 - (1.0 - self._lr_sigma)**(2.0 * self._opt_iter)
                    )
            < (1.4 + 2.0/(self._mean.shape[0] + 1.0))
            * self._expected_path_length
            else 0.0
            )

        core_matrix *= (
            1.0
            - self._lr_rank_one
            - self._lr_rank_mu
            + (1.0 - update_switch)
            * self._lr_rank_one * self._lr_cov * (2.0 - self._lr_cov)
            )

        # Add the rank-1 update
        core_matrix += self._lr_rank_one * outer(
            path_cov_padded, path_cov_padded)

        # Add the rank-mu update
        core_matrix += self._lr_rank_mu * (
            elite_projection.T @ (
                self._weights_2d[:self._elite_size] * elite_projection))

        return core_matrix

    def _update_rank_terms(
            self,
            target_matrix,
            projection_matrix):
        """
        Compute the rank-1 and rank-mu update terms.

        Parameters
        ----------
        target_matrix :
            ...

        projection_matrix :
            ...

        Returns
        -------
        ...
        """

        # Get the current rank
        rank = target_matrix.shape[1]

        # Compute the full-rank covariance evolution path
        rank_cma = self._path_cov.shape[0]
        matmul(
            self._left_basis[:, :rank_cma], self._path_cov, out=self._path_full
            )

        ps_norm = norm(self._path_sigma)
        update_switch = (
            1.0
            if ps_norm / sqrt(
                    1.0 - (1.0 - self._lr_sigma)**(2.0 * self._opt_iter)
                    )
            < (1.4 + 2.0/(self._mean.shape[0] + 1.0))
            * self._expected_path_length
            else 0.0
            )

        #
        target_matrix *= (
            1.0
            - self._lr_rank_one
            - self._lr_rank_mu
            + (1.0 - update_switch)
            * self._lr_rank_one * self._lr_cov * (2.0 - self._lr_cov)
            )

        # Add the rank-1 update
        target_matrix += self._lr_rank_one * (self._path_full @ (
            self._path_full.T @ projection_matrix))

        # Compute the weighted elite steps
        multiply(
            self._weights_2d[:self._elite_size], self._elite_steps,
            out=self._weighted_steps)

        # Project the weighted steps
        matmul(
            self._weighted_steps, projection_matrix,
            out=self._mu_projection[:, :rank])

        # Add the rank-mu update
        target_matrix += self._lr_rank_mu * (
            self._elite_steps.T @ self._mu_projection[:, :rank])

        return target_matrix

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
        rank = self.integrator.rank

        # Sample from the standard multivariate Gaussian
        num_random = (
            self._pop_size - 2 if self.gradient is not None
            else self._pop_size
            )
        z_low_rank = self._rng.standard_normal((num_random, rank))
        z_noise = self._rng.standard_normal(
            (num_random, self._number_of_variables)
            )

        # Sample steps from the multivariate Gaussian
        structured_part = z_low_rank @ self._root_cov[:, :rank].T
        noise_part = z_noise @ sqrt(self.integrator._psi).T
        self._steps[:num_random] = structured_part + noise_part

        # Check if a gradient has been provided
        if self.gradient is not None:

            # Compute the gradient
            gradient = self.gradient(self._mean)

            # Compute the unscaled natural gradient
            natural_gradient = self._root_cov[:, :rank] @ (
                self._root_cov[:, :rank].T @ gradient)

            # Compute the rescaling factor
            rescale = 1 / (sqrt(gradient @ natural_gradient) + 1e-15)

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
                gamma_factor = 0.95

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
        Update the adaptive variables.

        Parameters
        ----------
        fitness : ndarray
            Fitness values of the new population.
        """

        # Get the current rank
        rank_old = self.integrator.rank

        # Get the eigenvalues and eigenvectors of the singular value matrix
        ev, eq = self._core_eig_cache

        # Update the basic variables
        self._sigma, elite_indices = _tell(
            fitness,
            self._steps,
            self._weights,
            self._left_basis[:, :rank_old],
            self._path_sigma,
            self._path_cov,
            self._mean,
            self._sigma,
            ev[:rank_old],
            eq[:rank_old, :rank_old],
            self._lr_sigma,
            self._lr_cov,
            self._lr_mean,
            self._mu_eff,
            self._damp_sigma,
            self._expected_path_length,
            self._opt_iter,
            self._elite_size)

        # Get the elite steps
        take(self._steps, elite_indices, axis=0, out=self._elite_steps)

        # Check if the dynamical low-rank factors should be updated
        if self._opt_iter % self._update_interval == 0:

            # Copy the current basis to the buffer
            copyto(
                self._left_basis_old[:, :rank_old],
                self._left_basis[:, :rank_old])

            # Update the dynamical low-rank factors
            U_new, S_new, _ = self.integrator.update(
                self._left_basis[:, :rank_old], self._core_matrix,
                self._left_basis[:, :rank_old], self._lr_cov)

            #
            rank_new = U_new.shape[1]

            #
            copyto(self._left_basis[:, :rank_new], U_new)
            self._core_matrix = S_new

            # Rotate the evolution paths
            projection = (
                self._left_basis[:, :rank_new].T
                @ self._left_basis_old[:, :rank_old])
            self._path_sigma = (projection @ self._path_sigma).flatten()
            self._path_cov = (projection @ self._path_cov).reshape(-1, 1)

            # Check if the rank has changed
            if rank_new != rank_old:

                # Update the dynamic parameters
                self._update_dynamics()

            # Update the eigenvalues and -vectors of the singular value matrix
            ev, eq = eigh(
                self._core_matrix, overwrite_a=True, check_finite=False)

            # Sort in descending order
            ev, eq = ev[::-1], eq[:, ::-1]

            # Clip the eigenvalues
            maximum(ev, 1e-15, out=ev)

            # Update the eigendecomposition cache
            self._core_eig_cache = (ev, eq)

            # Update the sampling matrix
            matmul(
                self._left_basis[:, :rank_new], eq * sqrt(ev),
                out=self._root_cov[:, :rank_new])

            #
            diagonal_block = self._root_cov[:, rank_new:]
            diagonal_block.fill(0.0)
            fill_diagonal(diagonal_block, sqrt(self.integrator._psi))

    def _update_dynamics(self):
        """Update the dynamic parameters for the current rank."""

        # Get the current rank
        rank = self.integrator.rank

        # Update the learning rates
        self._lr_sigma = (self._mu_eff + 2) / (rank + self._mu_eff + 5)
        self._lr_cov = (
            (4 + self._mu_eff / rank) / (rank + 4 + 2*self._mu_eff / rank)
            )
        self._lr_rank_one = 2 / ((rank + 1.3)**2 + self._mu_eff)
        self._lr_rank_mu = min(
            1.0 - self._lr_rank_one,
            2.0 * (
                (0.25 + self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                ((rank + 2.0)**2 + 2.0 * self._mu_eff / 2.0)
                )
            )
        self._lr_mean = 1.0

        # Update the damping coefficient
        self._damp_sigma = (
            1.0
            + 2.0 * max(0.0, sqrt((self._mu_eff - 1.0) / (rank + 1.0)) - 1.0)
            + self._lr_sigma)

        # Initialize the expected path length
        self._expected_path_length = (
            sqrt(rank) * (1.0 - 1.0 / (4.0 * rank) + 1.0 / (21.0 * rank**2)))

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

        # Get the maximum and minimum eigenvalue
        max_ev = nmax(diag(self._core_matrix))
        min_ev = nmin(diag(self._core_matrix))

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


@njit("Tuple((float64, int64[:]))(float64[:], float64[:,:], float64[::1], "
      "float64[::1,:], float64[:], float64[:,:], float64[:], float64, "
      "float64[:], float64[:,:], float64, float64, float64, float64, float64, "
      "float64, int64, int64)", fastmath=True)
def _tell(
    fitness, steps, weights, left_basis, path_sigma, path_cov, mean, sigma, ev,
    eq, lr_sigma, lr_cov, lr_mean, mu_eff, damp_sigma, expected_path_length,
    opt_iter, elite_size):
    """Update the basic DLR-CMAES variables."""

    # Get the indices of the elite fitness values
    elite_indices = argsort(fitness)[:elite_size]

    # Initialize the elite mean step
    elite_weights = weights[:elite_size].reshape(-1, 1)
    elite_mean_step = nsum(
        steps[elite_indices] * elite_weights, axis=0)

    # Calculate the inverse rooted eigenvalues
    inv_root_ev = 1.0 / (sqrt(ev) + 1e-15)

    # Transform the elite mean step
    latent_mean_step = left_basis.T @ elite_mean_step
    eq_step = eq.T @ latent_mean_step
    latent_step_whitened = eq @ (inv_root_ev * eq_step)

    # Update the step-size evolution path
    path_sigma *= (1.0 - lr_sigma)
    path_sigma += (
        sqrt(lr_sigma * (2.0 - lr_sigma) * mu_eff) * latent_step_whitened
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

    return sigma_update, elite_indices
