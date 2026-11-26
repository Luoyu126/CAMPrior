#!/bin/bash
# Stage E summary (plan_endpoint.md §4): validation anchor, best of 20, evaluation seed 0. For every run with results:
# the MIS size at renoise 50 / 10 / 5 and posterior 50 steps; paired differences (paired.py, graph bootstrap 95% CI)
# against the 1C reference (SB omega 0.35, original multi-solutions, seeds 0-1 pooled; the pre-registered "works" test)
# and, for SB runs, against the omega = 0 control on the same pi1.
# Usage: bash experiments/20261126_sb_endpoint/summary.sh [tag, default anchor_renoise_t50_seed0]
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source cam/scripts/env.sh
R=runs/experiments/20261126_sb_endpoint; P=experiments/20261125_sbdifusco_rbsmall/paired.py
REF="runs/experiments/20261125_sbdifusco_rbsmall/20261125_sbdifusco_rbsmall_sb_w0.35_soft_s[01]"
TAG=${1:-anchor_renoise_t50_seed0}
v() { grep -h "Avg. size" "$1/$2_result.txt" 2>/dev/null | tail -1 | awk '{printf "%.3f", $3}'; }
pd() { "$CAM_PYTHON" $P "$1" "$2" $TAG --json 2>/dev/null | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(f\"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]\")" 2>/dev/null; }
echo "tag $TAG; reference 1C SB omega 0.35 (2 seeds): $(for d in $REF; do v $d $TAG; echo -n ' '; done)"
echo "| run | r50 | r10 | r5 | p50 | − 1C SB ($TAG) | SB − own control ($TAG) |"
echo "|---|---|---|---|---|---|---|"
for d in $R/20261126_sb_endpoint_*; do
    [ -f "$d/${TAG}_result.txt" ] || continue
    n=$(basename $d); n=${n#20261126_sb_endpoint_}
    c=""; if [[ $n =~ _w0\.[0-9]+ ]]; then cd_=$(echo "$d" | sed -E 's/_w0\.[0-9]+/_w0/'); [ -f "$cd_/${TAG}_result.txt" ] && c=$(pd "$cd_" "$d"); fi
    echo "| $n | $(v $d anchor_renoise_t50_seed0) | $(v $d anchor_renoise_t10_seed0) | $(v $d anchor_renoise_t5_seed0) | $(v $d anchor_posterior_t50_seed0) | $(pd "$REF" "$d") | $c |"
done
