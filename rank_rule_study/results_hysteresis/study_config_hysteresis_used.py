"""Settings for the hysteresis comparison: every rule with and without the
hysteresis of the rank steps of the branch version (5 consecutive proposals
up, 2 down). The rules of study_config.py get a '_hyst' copy (variants.py);
the branch rule (Marchenko-Pastur filter + hysteresis, optimizer_branch/) is
also run without hysteresis ('mp_no_hysteresis'). Default λ, no gradient,
n = 10 ... 500; all other settings as in study_config.py.
Selected with STUDY_CONFIG=study_config_hysteresis (start_hysteresis.sh).
"""
from study_config import *    # noqa: F401,F403

PROBLEMS = ['Sphere', 'Discus', 'Rotated Ellipsoid', 'Rosenbrock']
DIMS = [10, 20, 50, 100, 200, 500]
POP_SIZES = [None]
USE_GRADIENT = [False]

EXTRA_VARIANTS = {'mp_hysteresis': 'optimizer_branch/dlrcmaes_mp_hysteresis.py'}

# Phase 1 without, phase 2 with hysteresis; the plots of phase 2 show all rules
PHASES = [('1_no_hysteresis', ['current', 'soft_eps', 'matrix_bernstein', 'bernstein_age',
                               'mp_no_hysteresis']),
          ('2_hysteresis', ['current_hyst', 'soft_eps_hyst', 'matrix_bernstein_hyst',
                            'bernstein_age_hyst', 'mp_hysteresis'])]

RESULTS_DIR = 'results_hysteresis'
