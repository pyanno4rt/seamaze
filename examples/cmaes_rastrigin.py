"""CMA-ES benchmarking: Rastrigin function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Rastrigin
from seamaze.diagnostics import MonitorCMAES
from seamaze.optimizers import CMAES
from seamaze.plotting import plot_results

# %% Rastrigin


# Initialize the problem
rastrigin = Rastrigin(100)

# Initialize the monitor
monitor = MonitorCMAES(interval=1, mode='interactive', delay=0.001)

# Initialize the CMA-ES solver
solver = CMAES(
    number_of_variables=rastrigin.ndim,
    objective=rastrigin.__call__,
    gradient=rastrigin.gradient,
    lower_variable_bounds=array(rastrigin.bounds[0]),
    upper_variable_bounds=array(rastrigin.bounds[1]),
    number_of_individuals=1000,
    initial_sigma=2.0,  # Set to ~20% of the range
    maximum_iterations=1000,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-6,
    sigma_threshold=1e-3,
    update_interval=1,  # Update every iteration
    callback=monitor.full  # Enable full monitoring
    )

# Optimize the decision variables
result = solver.optimize(array([5]*rastrigin.ndim))

# Plot the results
plot_results(
    data=monitor.data,
    label='Rastrigin',
    show_objective=True,
    show_fitness=True,
    show_step_size=True,
    show_mean_change_norm=True,
    show_sigma_path_norm=True,
    show_cov_path_norm=True,
    show_cov_svs=True,
    show_cov_norm=True,
    show_cov_cn=True,
    show_cov_spectr_norm=True,
    save_folder=None
    )
