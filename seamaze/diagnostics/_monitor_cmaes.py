"""CMA-ES monitor."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import mean
from numpy.linalg import norm

# %% Internal package import

from seamaze.plotting import VisualizerCMAES

# %% Class definition


class MonitorCMAES:
    """
    CMA-ES monitor class.

    This class implements a monitoring system for tracking optimization \
    parameters of CMA-ES.

    Parameters
    ----------
    interval : int, default=1
        Frequency of data collection. Defaults to 1.

    mode : {'interactive', 'silent'}, default='silent'
        The monitoring mode. If 'interactive', an interactive 2D \
        visualization of the optimization progress is displayed.

    delay : int or float, default=0.001
        The time of delay for interactive plotting (in seconds). Only used if \
        mode is set to 'interactive'.

    Attributes
    ----------
    interval : int
        See 'Parameters'.

    mode : {'interactive', 'silent'}
        See 'Parameters'.

    delay : int or float
        See 'Parameters'.

    _visualizer : object of class \
        :class:`~seamaze.plotting._visualizer_cmaes.VisualizerCMAES`
        The object used to interactively plot the tracked results.

    _data : dict
        Dictionary with the tracked parameters.

    _counter : int
        Internal counter.

    _last_mean : ndarray
        Cache for the mean vector.
    """

    def __init__(
            self,
            interval=1,
            mode=False,
            delay=0.001):

        # Initialize the interval
        self.interval = interval

        # Initialize the visualization attributes
        self.mode = mode
        self._visualizer = None

        # Initialize the delay
        self.delay = delay

        # Initialize the data dictionary
        self._data = {}

        # Initialize the counter
        self._counter = 0

        # Initialize the last mean vector
        self._last_mean = 0

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
        value = value.copy() if hasattr(value, 'copy') else value

        # Check if the key is not available yet
        if key not in self._data:

            # Add the key/value pair
            self._data[key] = value

        else:

            # Check if the value is not a list
            if not isinstance(self._data[key], list):

                # Expand the value into a list
                self._data[key] = [self._data[key], value]

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
            :class:`~seamaze.optimizers.evolutionary._cmaes.CMAES`
            The object used to represent the CMA-ES algorithm.
        """

        # Increment the counter
        self._counter += 1

        # Check if the first iteration is reached
        if self._counter == 1:

            # Check if an interactive plot should be created
            if self.mode == 'interactive':

                # Get the variable bounds
                bounds = (
                    solver.lower_variable_bounds, solver.upper_variable_bounds)

                # Initialize the visualizer
                self._visualizer = VisualizerCMAES(
                    objective=solver.objective,
                    bounds=bounds,
                    dimensions=solver._number_of_variables)

            # Get the static parameters
            parameters = {
                'number_of_variables': solver._number_of_variables,
                'bounds': (
                    solver.lower_variable_bounds,
                    solver.upper_variable_bounds
                    ),
                'max_iter': solver.maximum_iterations,
                'max_time': solver.maximum_wall_time,
                'fitness_threshold': solver.fitness_threshold,
                'sigma_threshold': solver.sigma_threshold,
                'tolerance': solver.tolerance,
                'update_interval': solver._update_interval,
                'pop_size': solver._pop_size,
                'elite_size': solver._elite_size,
                'weights': solver._weights,
                'mu_eff': solver._mu_eff.item(),
                'lr_sigma': solver._lr_sigma.item(),
                'lr_cov': solver._lr_cov,
                'lr_rank_one': solver._lr_rank_one.item(),
                'lr_rank_mu': solver._lr_rank_mu.item(),
                'lr_mean': solver._lr_mean,
                'damp_sigma': solver._damp_sigma.item(),
                'expected_path_length': solver._expected_path_length.item()
                }

            # Loop over the parameter dictionary
            for key, value in parameters.items():

                # Record the key/value pair
                self._record(key, value)

        # Check if the data should be updated
        if self._counter % self.interval == 0:

            # Check if the interactive plot should be updated
            if self._visualizer:

                # Update the interactive plot
                self._visualizer.update(
                    iteration=solver._opt_iter,
                    mean=solver._mean,
                    cov=solver._cov,
                    sigma=solver._sigma,
                    fitness=solver._result['optimal_value'].item(),
                    delay=self.delay)

            # Record the iteration and the current best objective value
            self._record('iteration', solver._opt_iter)
            self._record(
                'optimal_value', solver._result['optimal_value'].item())

            # Record the best, mean and worst objective value
            self._record('best_fitness', min(solver._fitness))
            self._record('mean_fitness', mean(solver._fitness))
            self._record('worst_fitness', max(solver._fitness))

    def full(
            self,
            solver):
        """
        Record the full parameter set, including vectors and norms.

        Parameters
        ----------
        solver : object of class \
            :class:`~seamaze.optimizers.evolutionary._cmaes.CMAES`
            The object used to represent the CMA-ES algorithm.
        """

        # Record the basic parameters
        self.base(solver)

        # Check if the data should be updated
        if self._counter % self.interval == 0:

            # Record the step size
            self._record('sigma', solver._sigma)

            # Record the mean vector
            self._record('mean', solver._mean)
            self._record(
                'mean_change_norm', norm(solver._mean-self._last_mean).item())
            self._last_mean = solver._mean.copy()

            # Record the singular values
            svs = solver._core_vector
            self._record('cov_svs', svs)

            # Record the covariance metrics
            max_sv, min_sv = max(svs), min(svs)
            self._record('cov_norm', norm(solver._cov, ord=2).item())
            self._record('cov_cn', (max_sv / (min_sv + 1e-12)).item())
            self._record('cov_spectr_norm', max_sv.item())

            # Record the evolution path norms
            self._record('path_sigma_norm', norm(solver._path_sigma).item())
            self._record('path_cov_norm', norm(solver._path_cov).item())
