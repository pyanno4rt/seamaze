"""Settings for the rank rule of the develop version of _dlrcmaes.py
(spectral noise threshold from a Rademacher bootstrap of the weights,
hysteresis of the rank steps), stored as a fixed copy in optimizer_branch/,
so the optimizer in seamaze/ is not needed. Same settings as
study_config_branch.py (default λ and λ = 100, without gradient), run as
phase 5 in the folder of the main study, so the plots compare with all other
rules. Selected with STUDY_CONFIG=study_config_bootstrap (start_bootstrap.sh).
"""
from study_config_branch import *    # noqa: F401,F403
from study_config_branch import EXTRA_VARIANTS, PHASES

EXTRA_VARIANTS = dict(EXTRA_VARIANTS, bootstrap='optimizer_branch/dlrcmaes_bootstrap.py')
PHASES = PHASES + [('5_bootstrap', ['bootstrap'])]
