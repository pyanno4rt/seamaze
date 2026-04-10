"""DLR-CMA-ES benchmarking."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Rastrigin, Sphere, StyblinskiTang
from seamaze.optimizers.low_rank import DLRCMAES

# %% Benchmark simulations


# Initialize the problems
rastrigin = Rastrigin(5)
sphere = Sphere(5)
stybtang = StyblinskiTang(5)

# Initialize the CMA-ES solvers
solver_rastrigin = DLRCMAES(
    number_of_variables=rastrigin._n,
    objective=rastrigin.__call__,
    gradient=rastrigin.gradient,
    lower_variable_bounds=array(rastrigin.bounds[0]),
    upper_variable_bounds=array(rastrigin.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    low_rank_integrator='fixedsymmetricBUG',
    low_rank_dimension=None,
    low_rank_tolerance_rel=1e-2,
    low_rank_tolerance_abs=1e-8,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    update_interval=1,
    callback=None)

solver_sphere = DLRCMAES(
    number_of_variables=sphere._n,
    objective=sphere.__call__,
    gradient=sphere.gradient,
    lower_variable_bounds=array(sphere.bounds[0]),
    upper_variable_bounds=array(sphere.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    low_rank_integrator='fixedsymmetricBUG',
    low_rank_dimension=None,
    low_rank_tolerance_rel=1e-2,
    low_rank_tolerance_abs=1e-8,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    update_interval=1,
    callback=None)

solver_stybtang = DLRCMAES(
    number_of_variables=stybtang._n,
    objective=stybtang.__call__,
    gradient=stybtang.gradient,
    lower_variable_bounds=array(stybtang.bounds[0]),
    upper_variable_bounds=array(stybtang.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    low_rank_integrator='fixedsymmetricBUG',
    low_rank_dimension=None,
    low_rank_tolerance_rel=1e-2,
    low_rank_tolerance_abs=1e-8,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    update_interval=1,
    callback=None)

# Optimize the decision variables
result_rastrigin = solver_rastrigin.optimize(array([5]*rastrigin._n))
result_sphere = solver_sphere.optimize(array([5]*sphere._n))
result_stybtang = solver_stybtang.optimize(array([5]*stybtang._n))

# Plot ...
