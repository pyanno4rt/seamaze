#!/usr/bin/env bash
# Stop a running study (main process and its worker processes).
# Finished runs are kept; runs in progress are lost and redone on restart.
found=0
for p in $(pgrep -f "rank_rule_study/run_(study|reference).py"); do
    found=1
    kids=$(pgrep -P "$p")              # workers, collected before the main process stops
    kill -9 "$p"                       # main first, so no new workers are started
    [ -n "$kids" ] && kill $kids 2>/dev/null
    echo "stopped $p (workers: ${kids//$'\n'/ })"
done
[ "$found" = 0 ] && echo "no running study found"
sleep 2
pgrep -fa "multiprocessing" | grep -v grep && echo "(processes above may be leftover workers; kill them with: kill <PID>)"
exit 0
