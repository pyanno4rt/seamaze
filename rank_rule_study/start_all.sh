#!/usr/bin/env bash
# Start the study as several independent processes in the background (each
# runs every N-th run, so the parts do not overlap; keeps running after logout).
# Usage: bash rank_rule_study/start_all.sh [N_PARTS] [WORKERS_PER_PART] [further options]
#        e.g. bash rank_rule_study/start_all.sh 4 8
#             bash rank_rule_study/start_all.sh 4 8 --phase 3_age_noise
# Progress: tail -f rank_rule_study/<RESULTS_DIR>/log.txt  (RESULTS_DIR in study_config.py)   Stop: bash rank_rule_study/stop.sh
cd "$(dirname "$0")/.." || exit 1
RD="$(grep "^RESULTS_DIR" "rank_rule_study/${STUDY_CONFIG:-study_config}.py" | cut -d"'" -f2)"    # results folder
PY="${PYTHON:-python3}"
NPARTS="${1:-4}"
WORKERS="${2:-8}"
shift $(( $# < 2 ? $# : 2 ))
STAMP="$(date +%Y%m%d_%H%M%S)"
for K in $(seq 1 "$NPARTS"); do
    OUT="rank_rule_study/nohup_${STAMP}_part${K}.out"
    nohup nice -n 10 "$PY" rank_rule_study/run_study.py --part "$K/$NPARTS" --workers "$WORKERS" "$@" > "$OUT" 2>&1 &
    echo "part $K/$NPARTS started (PID $!, $WORKERS workers); output: $OUT"
    sleep 2
done
echo "log: rank_rule_study/$RD/log.txt"
