#!/bin/bash
# Align a group of route-1' runs to the same number of iterations (plan §3: the runs of one comparison use the latest
# run's stopping iteration). Waits until every listed run has stopped (state.json 'stopped_at'), then relaunches
# each run that stopped earlier with --target <max> and its original flags.
# Usage: bash experiments/20261128_dsbm_rbsmall/align.sh <name1> <name2> ...
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP_NAME="$(basename "$EXP_DIR")"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
R="$REPO/runs/experiments/$EXP_NAME"
st() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('stopped_at',''))" "$R/${EXP_NAME}_$1/state.json" 2>/dev/null; }
while :; do
    all=1; for n in "$@"; do [ -n "$(st $n)" ] || all=0; done
    [ $all = 1 ] && break; sleep 300
done
MAX=0; for n in "$@"; do s=$(st $n); [ "$s" -gt "$MAX" ] && MAX=$s; done
echo "$(date '+%F %T') all stopped; aligning to $MAX iterations"
for n in "$@"; do
    s=$(st $n)
    if [ "$s" -lt "$MAX" ]; then
        echo "$(date '+%F %T') $n: stopped at $s, extending to $MAX"
        python3 - "$R/${EXP_NAME}_$n/state.json" "$EXP_DIR/launch.sh" "$n" "$MAX" <<'PY'
import json, subprocess, sys
state, launch, name, target = sys.argv[1:5]
a = json.load(open(state))['args']
argv = ['bash', launch, name]
for k, v in a.items():
    if k in ('out', 'collect_eval', 'dry', 'target') or v is None or v is False:
        continue
    argv += ['--' + k] if v is True else ['--' + k, str(v)]
subprocess.run(argv + ['--target', target], check=True)
PY
    fi
done
