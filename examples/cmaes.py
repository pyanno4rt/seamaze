"""CMA-ES benchmarking."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import (
    Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)
from seamaze.diagnostics import MonitorCMAES
from seamaze.optimizers import CMAES
from seamaze.plotting import plot_results

# %% Workflow

"""
This script provides a workflow to run benchmarking tests for CMA-ES.

Available benchmark functions are:

    - 'Ackley'
    - 'Bent-Cigar'
    - 'Discus'
    - 'Ellipsoid'
    - 'Griewank'
    - 'Linear Slope'
    - 'Rastrigin'
    - 'Rosenbrock'
    - 'Rotated Ellipsoid'
    - 'Rotated Rastrigin'
    - 'Schwefel'
    - 'Sphere'
    - 'Styblinski-Tang'
    - 'Sum of Different Powers'
"""

# Enter the function name
name = 'Griewank'

# Enter the problem dimensionality
ndim = 2

# Get the benchmark function class
problems = {
    'Ackley': Ackley,
    'Bent-Cigar': BentCigar,
    'Discus': Discus,
    'Ellipsoid': Ellipsoid,
    'Griewank': Griewank,
    'Linear Slope': LinearSlope,
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'Rotated Ellipsoid': RotatedEllipsoid,
    'Rotated Rastrigin': RotatedRastrigin,
    'Schwefel': Schwefel,
    'Sphere': Sphere,
    'Styblinski-Tang': StyblinskiTang,
    'Sum of Different Powers': SumOfDiffPowers
    }

# Initialize the problem
problem = problems[name](ndim)

# Initialize the monitor (optional)
monitor = MonitorCMAES(
    interval=1, mode='interactive', plot_bounds=((-5, -5), (5, 5)),
    delay=0.001)

# Initialize the CMA-ES solver
solver = CMAES(
    number_of_variables=problem.ndim,
    objective=problem.__call__,
    # gradient=problem.gradient,
    # lower_variable_bounds=array(problem.bounds[0]),
    # upper_variable_bounds=array(problem.bounds[1]),
    number_of_individuals=1000,
    initial_sigma=2.0,
    maximum_iterations=10000,
    maximum_wall_time=43200,
    fitness_threshold=None,
    fitness_window_size=50,
    tolerance=1e-6,
    sigma_threshold=1e-8,
    update_interval=1,  # Update every iteration
    callback=monitor.full  # Enable full monitoring
    )

# Optimize the decision variables
result = solver.optimize(array([5.0]*problem.ndim))

# Plot the results
plot_results(
    data=monitor.data,
    label=problem.name,
    show_objective=True,
    show_fitness=True,
    show_bound_viol=True,
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
