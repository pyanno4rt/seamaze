"""DLR-CMA-ES benchmarking: Sphere function."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Sphere
from seamaze.optimizers.low_rank import DLRCMAES

# %% Sphere


# Initialize the problem
sphere = Sphere(100)

# Initialize the DLR-CMA-ES solver
solver_sphere = DLRCMAES(
    number_of_variables=sphere.ndim,
    objective=sphere.__call__,
    gradient=sphere.gradient,
    lower_variable_bounds=array(sphere.bounds[0]),
    upper_variable_bounds=array(sphere.bounds[1]),
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
result_sphere = solver_sphere.optimize(array([5]*sphere.ndim))
