"""CMA-ES benchmarking: Sphere function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Sphere
from seamaze.diagnostics import MonitorCMAES
from seamaze.optimizers import CMAES
from seamaze.plotting import plot_results

# %% Sphere

# Initialize the problem
sphere = Sphere(2)

# Initialize the monitor
monitor = MonitorCMAES(interval=1, mode='interactive', delay=3)

# Initialize the CMA-ES solver
solver_sphere = CMAES(
    number_of_variables=sphere.ndim,
    objective=sphere.__call__,
    gradient=sphere.gradient,
    lower_variable_bounds=array(sphere.bounds[0]),
    upper_variable_bounds=array(sphere.bounds[1]),
    number_of_individuals=2000,
    initial_sigma=2.0,
    maximum_iterations=2000,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-6,
    sigma_threshold=1e-3,
    update_interval=1,  # Update every iteration
    callback=monitor.full)  # Enable full monitoring

# Optimize the decision variables
result_sphere = solver_sphere.optimize(array([5]*sphere.ndim))

# Plot the results
plot_results(
    data=monitor.data,
    label='Sphere',
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
    save_path=None)
