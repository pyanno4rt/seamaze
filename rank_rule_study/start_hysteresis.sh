#!/usr/bin/env bash
# Start the hysteresis comparison (study_config_hysteresis.py: all rules with and
# without hysteresis; Sphere, Discus, Rotated Ellipsoid, Rosenbrock; n = 10 ... 500;
# default λ) as several processes in the background; results in
# rank_rule_study/results_hysteresis/.
# Usage: bash rank_rule_study/start_hysteresis.sh [N_PARTS] [WORKERS_PER_PART] [further options]
#        e.g. bash rank_rule_study/start_hysteresis.sh 4 8
#             bash rank_rule_study/start_hysteresis.sh 4 8 --phase 2_hysteresis
# Progress: tail -f rank_rule_study/results_hysteresis/log.txt   Stop: bash rank_rule_study/stop.sh
STUDY_CONFIG=study_config_hysteresis exec bash "$(dirname "$0")/start_all.sh" "$@"
