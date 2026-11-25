#!/usr/bin/env bash
# Start the study in the background (keeps running after logout).
# Usage: bash rank_rule_study/start.sh [--workers N] [--quick] [--phase NAME]
# Progress: tail -f rank_rule_study/<RESULTS_DIR>/log.txt  (RESULTS_DIR in study_config.py)
cd "$(dirname "$0")/.." || exit 1
RD="$(grep "^RESULTS_DIR" "rank_rule_study/${STUDY_CONFIG:-study_config}.py" | cut -d"'" -f2)"    # results folder
PY="${PYTHON:-python3}"
OUT="rank_rule_study/nohup_$(date +%Y%m%d_%H%M%S).out"    # one file per start
nohup nice -n 10 "$PY" rank_rule_study/run_study.py "$@" > "$OUT" 2>&1 &
echo "started (PID $!); log: rank_rule_study/$RD/log.txt (${RD}_quick/ with --quick); output: $OUT"
