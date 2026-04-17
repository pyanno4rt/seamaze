"""DLR-CMA-ES benchmarking: Styblinski-Tang function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import StyblinskiTang
from seamaze.optimizers.low_rank import DLRCMAES

# %% Styblinski-Tang


# Initialize the problem
stybtang = StyblinskiTang(100)

# Initialize the DLR-CMA-ES solver
solver_stybtang = DLRCMAES(
    number_of_variables=stybtang.ndim,
    objective=stybtang.__call__,
    gradient=stybtang.gradient,
    lower_variable_bounds=array(stybtang.bounds[0]),
    upper_variable_bounds=array(stybtang.bounds[1]),
    number_of_individuals=2000,
    initial_sigma=3.0,
    low_rank_integrator='fixedSPDBUG',
    low_rank_dimension=None,
    low_rank_tolerance_rel=1e-2,
    low_rank_tolerance_abs=1e-8,
    maximum_iterations=2000,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-6,
    sigma_threshold=1e-3,
    update_interval=1,
    callback=None)

# Optimize the decision variables
result_stybtang = solver_stybtang.optimize(array([5]*stybtang.ndim))
