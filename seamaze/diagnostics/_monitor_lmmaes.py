"""LM-MA-ES monitor."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import eye, float64, mean, ndarray, sort
from numpy.linalg import eigvalsh, norm

# %% Internal package import

from seamaze.plotting import Visualizer

# %% Class definition


class MonitorLMMAES:
    """
    LM-MA-ES monitor class.

    This class implements a monitoring system for tracking optimization \
    parameters of LM-MA-ES.

    Parameters
    ----------
    interval : int, default=1
        Frequency of data collection. Defaults to 1.

    mode : {'interactive', 'silent'}, default='silent'
        The monitoring mode. If 'interactive', an interactive 2D \
        visualization of the optimization progress is displayed.

    plot_bounds : tuple, list or ndarray, default=None
        Visualization bounds as coordinate pairs ((x1, y1), (x2, y2)). Only \
        used (and mandatory) if mode is set to 'interactive'.

    delay : int or float, default=0.001
        The time of delay for interactive plotting (in seconds). Only used if \
        mode is set to 'interactive'.
    """

    def __init__(
            self,
            interval=1,
            mode='silent',
            plot_bounds=None,
            delay=0.001):

        # Get the arguments
        self.interval = interval
        self.mode = mode
        self.plot_bounds = plot_bounds
        self.delay = delay

        # Initialize the visualizer
        self.visualizer = None

        # Initialize the data dictionary
        self._data = {}

        # Initialize the counter
        self._counter = 0

        # Initialize the last mean vector
        self._last_mean = None

        # Initialize the covariance matrix and singular values
        self._cov = None
        self._svs = None

    @property
    def data(self):
        """Return the data dictionary."""

        return self._data

    def _record(
            self,
            key,
            value):
        """
        Update the tracked parameters.

        Parameters
        ----------
        key : str
            Name of the parameter to be recorded.

        value : any
            Value to be stored.
        """

        # Copy the value if it is an array
        value = (
            value.copy()
            if hasattr(value, 'copy') or isinstance(value, ndarray)
            else value
            )

        # Check if the key is not available yet
        if key not in self._data:

            # Add the key-value pair
            self._data[key] = [value]

        else:

            # Append the value
            self._data[key].append(value)

    def base(
            self,
            solver):
        """
        Record the basic parameter set.

        Parameters
        ----------
        solver : object of class \
            :class:`~seamaze.optimizers._lmmaes.LMMAES`
            The object used to represent the LM-MA-ES algorithm.
        """

        # Increment the counter
        self._counter += 1

        # Check if the first iteration is reached
        if self._counter == 1:

            # Check if an interactive plot should be created
            if self.mode == 'interactive':

                # Initialize the visualizer
                self.visualizer = Visualizer(
                    bounds=self.plot_bounds,
                    dimensions=solver._number_of_variables,
                    pop_size=solver._pop_size)

            # Get the static parameters
            parameters = {
                'bounds': (
                    solver.lower_variable_bounds, solver.upper_variable_bounds
                    ),
                'damp_sigma': solver._damp_sigma,
                'elite_size': solver._elite_size,
                'expected_path_length': solver._expected_path_length,
                'fitness_threshold': solver.fitness_threshold,
                'lr_cov': solver._lr_cov,
                'lr_mean': solver._lr_mean,
                'lr_mem': solver._lr_mem,
                'lr_sigma': solver._lr_sigma,
                'max_iter': solver.maximum_iterations,
                'max_time': solver.maximum_wall_time,
                'memory_size': solver._memory_size,
                'mu_eff': solver._mu_eff.item(),
                'number_of_variables': solver._number_of_variables,
                'pop_size': solver._pop_size,
                'sigma_threshold': solver.sigma_threshold,
                'tolerance': solver.tolerance,
                'weights': solver._weights,
                }

            # Loop over the parameter dictionary
            for key, value in parameters.items():

                # Record the key/value pair
                self._record(key, value)

            # Copy the initial mean
            self._last_mean = solver._mean.copy()

        # Check if the data should be updated
        if self._counter % self.interval == 0:

            # Pre-store solver variables
            errors = solver._squared_bound_errors
            fitness = solver._fitness
            optimal_value = solver._result['optimal_value'].item()

            # Reproduce the covariance matrix and singular values
            self._cov = (
                eye(solver._number_of_variables, order='F', dtype=float64)
                + solver._memory.T @ solver._memory
                )
            self._svs = sort(
                1.0 + eigvalsh(solver._memory @ solver._memory.T)
                )[::-1]

            # Check if the interactive plot should be updated
            if self.visualizer:

                # Update the interactive plot
                self.visualizer.update(
                    iteration=solver._opt_iter,
                    population=solver._population,
                    mean=solver._mean,
                    cov=self._cov,
                    svs=self._svs,
                    sigma=solver._sigma,
                    fitness=fitness,
                    squared_bound_errors=errors,
                    optimal_value=optimal_value,
                    delay=self.delay)

            # Record the iteration and the current best objective value
            self._record('iteration', solver._opt_iter)
            self._record('optimal_value', optimal_value)

            # Record the best, mean and worst objective value
            self._record('best_fitness', min(fitness))
            self._record('mean_fitness', mean(fitness))
            self._record('worst_fitness', max(fitness))

            # Record the bound violation
            self._record(
                'max_bound_viol', 0.0 if errors is None else max(errors))
            self._record(
                'mean_bound_viol', 0.0 if errors is None else mean(errors))
            self._record('gamma', solver._gamma)

    def full(
            self,
            solver):
        """
        Record the full parameter set, including vectors and norms.

        Parameters
        ----------
        solver : object of class \
            :class:`~seamaze.optimizers._lmmaes.LMMAES`
            The object used to represent the LM-MA-ES algorithm.
        """

        # Record the basic parameters
        self.base(solver)

        # Check if the data should be updated
        if self._counter % self.interval == 0:

            # Record the sigma evolution path norm
            self._record('path_sigma_norm', norm(solver._path_sigma).item())

            # Record the step size
            self._record('sigma', solver._sigma)

            # Record the mean vector
            mean_vec = solver._mean
            self._record('mean', mean_vec)
            self._record(
                'mean_change_norm', norm(mean_vec-self._last_mean).item())
            self._last_mean = mean_vec.copy()

            # Record the singular values
            svs = self._svs
            self._record('cov_svs', svs)

            # Record the covariance metrics
            max_sv, min_sv = max(svs), min(svs)
            self._record('cov_norm', norm(self._cov, ord=2).item())
            self._record('cov_cn', (max_sv / (min_sv + 1e-12)).item())
            self._record('cov_spectr_norm', max_sv.item())

    def __enter__(self):
        """Enter the runtime context and return the monitor object."""

        return self

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb):
        """
        Exit the runtime context and hold the interactive plot.

        Parameters
        ----------
        exc_type : type or None
            The type of the exception that caused the context to be exited.

        exc_val : Exception or None
            The exception instance that caused the context to be exited.

        exc_tb : traceback or None
            The traceback object associated with the exception.

        Returns
        -------
        bool
            False, allowing any exceptions to propagate.
        """

        # Check if the interactive visualizer exists
        if self.visualizer is not None:

            # Import matplotlib locally
            import matplotlib.pyplot as plt

            # Transfer control to GUI
            plt.show(block=True)

        return False
