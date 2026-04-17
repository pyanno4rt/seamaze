"""DLR-CMA-ES benchmarking: Rastrigin function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Rastrigin
from seamaze.optimizers.low_rank import DLRCMAES

# %% Rastrigin


# Initialize the problem
rastrigin = Rastrigin(100)

# Initialize the DLR-CMA-ES solver
solver_rastrigin = DLRCMAES(
    number_of_variables=rastrigin.ndim,
    objective=rastrigin.__call__,
    gradient=rastrigin.gradient,
    lower_variable_bounds=array(rastrigin.bounds[0]),
    upper_variable_bounds=array(rastrigin.bounds[1]),
    number_of_individuals=2000,
    initial_sigma=2.0,
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
result_rastrigin = solver_rastrigin.optimize(array([5]*rastrigin.ndim))
