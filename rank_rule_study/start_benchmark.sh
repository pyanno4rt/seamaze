#!/usr/bin/env bash
# Start the study with the settings of the benchmark script (study_config_benchmark.py)
# as several processes in the background; results in rank_rule_study/results_benchmark_settings/.
# Usage: bash rank_rule_study/start_benchmark.sh [N_PARTS] [WORKERS_PER_PART] [further options]
#        e.g. bash rank_rule_study/start_benchmark.sh 4 8
STUDY_CONFIG=study_config_benchmark exec bash "$(dirname "$0")/start_all.sh" "$@"
