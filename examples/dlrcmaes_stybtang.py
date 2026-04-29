"""DLR-CMA-ES benchmarking: Styblinski-Tang function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import StyblinskiTang
from seamaze.diagnostics import MonitorDLRCMAES
from seamaze.optimizers import DLRCMAES
from seamaze.plotting import plot_results

# %% Styblinski-Tang


# Initialize the problem
stybtang = StyblinskiTang(100)

# Initialize the monitor
monitor = MonitorDLRCMAES(interval=1, mode='interactive', delay=0.001)

# Initialize the DLR-CMA-ES solver
solver = DLRCMAES(
    number_of_variables=stybtang.ndim,
    objective=stybtang.__call__,
    gradient=stybtang.gradient,
    lower_variable_bounds=array(stybtang.bounds[0]),
    upper_variable_bounds=array(stybtang.bounds[1]),
    number_of_individuals=1000,
    initial_sigma=2.0,  # Set to ~20% of the range
    low_rank_integrator='fixedSPDBUG',
    low_rank_dimension=None,
    low_rank_tolerance_rel=1e-2,
    low_rank_tolerance_abs=1e-8,
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
result = solver.optimize(array([5]*stybtang.ndim))

# Plot the results
plot_results(
    data=monitor.data,
    label='Styblinski-Tang',
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
    show_integrator_rank=True,
    save_folder=None
    )
