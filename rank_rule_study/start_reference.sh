#!/usr/bin/env bash
# Start the reference runs (CMA-ES, LM-MA-ES on Rosenbrock and Griewank) as several
# processes in the background; results in rank_rule_study/results_reference/.
# Usage: bash rank_rule_study/start_reference.sh [N_PARTS] [WORKERS_PER_PART]
# Progress: tail -f rank_rule_study/results_reference/log.txt
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
NPARTS="${1:-1}"
WORKERS="${2:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
for K in $(seq 1 "$NPARTS"); do
    OUT="rank_rule_study/nohup_reference_${STAMP}_part${K}.out"
    nohup nice -n 10 "$PY" rank_rule_study/run_reference.py --part "$K/$NPARTS" --workers "$WORKERS" > "$OUT" 2>&1 &
    echo "part $K/$NPARTS started (PID $!, $WORKERS workers); output: $OUT"
    sleep 2
done
