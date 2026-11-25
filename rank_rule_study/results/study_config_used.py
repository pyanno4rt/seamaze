"""Settings of the rank-rule study. Edit here; run_study.py reads this file."""

# Benchmark problems (see runner.make_problem)
PROBLEMS = ['sphere', 'ellipsoid', 'discus', 'schwefel12', 'rosenbrock',
            'discus_rot', 'ellipsoid_rot']

# Problem dimensions
DIMS = [10, 20, 50, 100, 200, 500]

# Starting points: key -> label (see runner.make_x0)
STARTS = {'ones10': 'x0 = 10·1',             # standard start
          'u50': 'x0 ~ U[-50,50]^n'}         # far, random start (harder)

SEEDS = [1, 2, 3]

# Phases run one after another; plots are written after each phase and
# regularly during it. Rules: 'matrix_bernstein', 'current' (0.02 soft
# filter, unchanged optimizer), 'soft_eps' (soft filter with the noise level),
# 'bernstein_age' (Bernstein bound with age-dependent noise level). Plots of a
# phase compare its rules with those of all earlier phases. A single phase can
# be run on its own with --phase (e.g. alongside a study that is running).
PHASES = [('1_bernstein', ['matrix_bernstein']),
          ('2_comparison', ['current', 'soft_eps']),
          ('3_age_noise', ['bernstein_age'])]

# Budget in generations: per dimension (rotated problems need more), but at
# least MIN_BUDGET
BUDGET_PER_N = 600
BUDGET_PER_N_ROTATED = 3000
MIN_BUDGET = 1_000_000

# Wall-time limit per run in hours (the run stops and is marked TIME_LIMIT)
TIME_LIMIT_H = 12.0

# Solver settings
SIGMA0 = 3.0
FITNESS_TARGET = 1e-8
TOLERANCE = 1e-12

# Minutes between intermediate plot updates
PLOT_EVERY_MIN = 20
