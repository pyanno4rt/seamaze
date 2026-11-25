"""Settings for the bootstrap rank rule of the develop branch without the
hysteresis of the rank steps (optimizer_branch/dlrcmaes_bootstrap_nohyst.py:
copy of dlrcmaes_bootstrap.py with both persistence counts set to 1, so each
rank step follows the rank pressure of its generation). Same settings as
phase 5, run as phase 6 in the folder of the main study.
Selected with STUDY_CONFIG=study_config_bootstrap_nohyst (start_bootstrap_nohyst.sh).
"""
from study_config_bootstrap import *    # noqa: F401,F403
from study_config_bootstrap import EXTRA_VARIANTS, PHASES

EXTRA_VARIANTS = dict(EXTRA_VARIANTS, bootstrap_nohyst='optimizer_branch/dlrcmaes_bootstrap_nohyst.py')
PHASES = PHASES + [('6_bootstrap_nohyst', ['bootstrap_nohyst'])]
