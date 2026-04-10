"""CMAES benchmarking."""

# Authors: Tim Ortkamp, Chinmay Patwardhan, Pia Stammer

# %% External package import

from numpy import array

# %% Internal package import

from seamaze.benchmarks import Rastrigin, Sphere, StyblinskiTang
from seamaze.optimizers.evolutionary import CMAES

# %% Benchmark simulations


# Initialize the problems
rastrigin = Rastrigin(5)
sphere = Sphere(5)
stybtang = StyblinskiTang(5)

# Initialize the CMA-ES solvers
solver_rastrigin = CMAES(
    number_of_variables=rastrigin._n,
    objective=rastrigin.__call__,
    gradient=rastrigin.gradient,
    lower_variable_bounds=array(rastrigin.bounds[0]),
    upper_variable_bounds=array(rastrigin.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    store_singular_values=False,
    update_interval=1,
    rank=None,
    callback=None)

solver_sphere = CMAES(
    number_of_variables=sphere._n,
    objective=sphere.__call__,
    gradient=sphere.gradient,
    lower_variable_bounds=array(sphere.bounds[0]),
    upper_variable_bounds=array(sphere.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    store_singular_values=False,
    update_interval=1,
    rank=None,
    callback=None)

solver_stybtang = CMAES(
    number_of_variables=stybtang._n,
    objective=stybtang.__call__,
    gradient=stybtang.gradient,
    lower_variable_bounds=array(stybtang.bounds[0]),
    upper_variable_bounds=array(stybtang.bounds[1]),
    number_of_individuals=100,
    initial_sigma=2.0,
    maximum_iterations=200,
    maximum_wall_time=7200,
    fitness_threshold=None,
    fitness_window_size=30,
    tolerance=1e-3,
    sigma_threshold=1e-3,
    store_singular_values=False,
    update_interval=1,
    rank=None,
    callback=None)

# Optimize the decision variables
result_rastrigin = solver_rastrigin.optimize(array([5]*rastrigin._n))
result_sphere = solver_sphere.optimize(array([5]*sphere._n))
result_stybtang = solver_stybtang.optimize(array([5]*stybtang._n))

# Get the iteration-wise singular values
sv_rastrigin = solver_rastrigin._singular_values
sv_sphere = solver_sphere._singular_values
sv_stybtang = solver_stybtang._singular_values

# Plot ...
