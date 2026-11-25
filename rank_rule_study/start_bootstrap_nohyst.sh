#!/usr/bin/env bash
# Start the runs of the bootstrap rule without hysteresis (study_config_bootstrap_nohyst.py,
# phase 6_bootstrap_nohyst) as several processes in the background;
# results in rank_rule_study/results_decay_fixed/.
# Usage: bash rank_rule_study/start_bootstrap_nohyst.sh [N_PARTS] [WORKERS_PER_PART]
#        e.g. bash rank_rule_study/start_bootstrap_nohyst.sh 4 8
STUDY_CONFIG=study_config_bootstrap_nohyst exec bash "$(dirname "$0")/start_all.sh" "${1:-4}" "${2:-8}" --phase 6_bootstrap_nohyst
