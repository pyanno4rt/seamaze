"""Settings of the rank-rule study. Edit here; run_study.py reads this file.

Problems, dimensions, start points and solver settings follow the seamaze
cross-benchmarking script (DLR-CMA-ES part).
"""

# Benchmark problems, by their seamaze names (problem.name); any seamaze
# benchmark can be added by its name
PROBLEMS = ['Discus', 'Bent-Cigar', 'Griewank', 'Rosenbrock', 'Rotated Ellipsoid', 'Sphere']

# Problem dimensions. For each n the start points are read from
# starting_points_dim{n}.npz (key 'starting_points') in this folder. If that
# file does not exist, N_GENERATED_STARTS points are drawn once uniformly from
# the benchmark domain [-5, 5]^n (seed START_SEED + n) and saved as
# starting_points_dim{n}_generated.npz; a provided file always takes priority.
DIMS = [10, 20, 50, 100, 200, 500]
N_GENERATED_STARTS = 5
START_SEED = 1000

# Parameter combinations (benchmark script: [None, 100] and [False, True])
POP_SIZES = [None]                # None: default population size
USE_GRADIENT = [False]

# Results folder (inside this folder). results_benchmark: runs with the
# covariance decay 1 - c1 - c_mu of the earlier optimizer version;
# results_decay_fixed: decay 1 - c1 - c_mu * sum(w) (Hansen 2016, eq. 47)
RESULTS_DIR = 'results_decay_fixed'

# Solver settings of the benchmark script
SOLVER = dict(
    initial_sigma=3.0,
    low_rank_init_dimension=None,
    low_rank_max_dimension=None,
    low_rank_is_adaptive=True,
    low_rank_energy_tolerance=1e-2,
    maximum_iterations=int(1e7),
    maximum_wall_time=43200,
    fitness_threshold=None,
    fitness_window_size=50,
    tolerance=1e-12,    # benchmark script: 1e-6, which stops slowly converging high-n runs early
    sigma_threshold=1e-8,
    update_interval=1,
    random_state=43)

# Seed of the global numpy generator (initial basis, augmentation vectors).
# The benchmark script leaves it unseeded; seeding makes runs reproducible
# and the rules comparable. None: leave unseeded.
GLOBAL_SEED = 43

# A run counts as solved in the plots and tables if its final best f is below
SUCCESS_F = 1e-8

# Phases run one after another; plots are written after each phase and
# regularly during it. Rules: 'matrix_bernstein', 'current' (0.02 soft
# filter, unchanged optimizer), 'soft_eps' (soft filter with the noise level),
# 'bernstein_age' (Bernstein bound with age-dependent noise level). Plots of a
# phase compare its rules with those of all earlier phases. A single phase can
# be run on its own with --phase.
PHASES = [('1_bernstein', ['matrix_bernstein']),
          ('2_comparison', ['current', 'soft_eps']),
          ('3_age_noise', ['bernstein_age'])]

# Minutes between intermediate plot updates
PLOT_EVERY_MIN = 20
