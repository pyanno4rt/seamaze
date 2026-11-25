#!/usr/bin/env bash
# Start the runs of the branch rule (study_config_branch.py, phase 4_mp_hysteresis)
# as several processes in the background; results in rank_rule_study/results_decay_fixed/.
# Usage: bash rank_rule_study/start_branch.sh [N_PARTS] [WORKERS_PER_PART]
#        e.g. bash rank_rule_study/start_branch.sh 4 8
STUDY_CONFIG=study_config_branch exec bash "$(dirname "$0")/start_all.sh" "${1:-4}" "${2:-8}" --phase 4_mp_hysteresis
