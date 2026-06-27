"""DLR-CMA-ES benchmarking."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import (
    Ackley, BentCigar, Discus, Ellipsoid, Griewank, LinearSlope, Rastrigin,
    Rosenbrock, RotatedEllipsoid, RotatedRastrigin, Schwefel, Sphere,
    StyblinskiTang, SumOfDiffPowers)
from seamaze.diagnostics import MonitorDLRCMAES
from seamaze.optimizers import DLRCMAES
from seamaze.plotting import ResultPlotter

# %% Workflow

"""
This script provides a workflow to run benchmarking tests for DLR-CMA-ES.

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
name = 'Sphere'

# Enter the problem dimensionality
ndim = 10

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

# Initialize the monitor as a context manager
with MonitorDLRCMAES(
    interval=1, mode='silent', plot_bounds=((-5, -5), (5, 5)),
    delay=0.001) as monitor:

    # Initialize the DLR-CMA-ES solver
    solver = DLRCMAES(
        number_of_variables=problem.ndim,
        objective=problem.__call__,
        # gradient=problem.gradient,
        # lower_variable_bounds=array(problem.bounds[0]),
        # upper_variable_bounds=array(problem.bounds[1]),
        number_of_individuals=None,
        initial_sigma=3.0,  # ~20-30 % of the search range
        low_rank_dimension=3,
        maximum_iterations=100000,
        maximum_wall_time=43200,
        fitness_threshold=None,
        fitness_window_size=50,
        tolerance=1e-6,
        sigma_threshold=1e-8,
        min_log_level='debug',
        callback=monitor.full,  # Enable full monitoring
        random_state=42
        )

    # Optimize the decision variables
    result = solver.optimize(array([3.0]*problem.ndim))

    # Initialize the result plotter (optional)
    plotter = ResultPlotter(
        data=monitor.data, label=problem.name, save_folder=None)

    # Select the plots
    plotter.show_objective = True
    plotter.show_fitness = True
    plotter.show_bound_viol = False
    plotter.show_step_size = True
    plotter.show_mean_change_norm = True
    plotter.show_sigma_path_norm = True
    plotter.show_cov_path_norm = True
    plotter.show_cov_svs = True
    plotter.show_cov_norm = True
    plotter.show_cov_cn = True
    plotter.show_cov_spectr_norm = True
    plotter.show_integrator_rank = True

    # Plot all selected results
    # plotter.plot_all()
