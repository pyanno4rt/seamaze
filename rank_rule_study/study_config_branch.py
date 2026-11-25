"""Settings for the rank rule of the branch version of _dlrcmaes.py
(Marchenko-Pastur noise filter + hysteresis of the rank steps), stored as a
fixed copy in optimizer_branch/, so the optimizer in seamaze/ is not needed.
Problems, dimensions, start points and solver settings as in study_config.py;
default λ and λ = 100, without gradient. The runs are written to the folder of
the main study, so the plots of phase 4 compare with the other rules.
Selected with STUDY_CONFIG=study_config_branch (start_branch.sh).
"""
from study_config import *    # noqa: F401,F403
from study_config import PHASES

POP_SIZES = [None, 100]
USE_GRADIENT = [False]

EXTRA_VARIANTS = {'mp_hysteresis': 'optimizer_branch/dlrcmaes_mp_hysteresis.py'}
PHASES = PHASES + [('4_mp_hysteresis', ['mp_hysteresis'])]

RESULTS_DIR = 'results_decay_fixed'
