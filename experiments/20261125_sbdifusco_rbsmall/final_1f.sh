#!/bin/bash
# Stage 1F final table (plan.md 1F): per arm the mean over training seeds and the seed standard deviation of the
# best-of-20 mean MIS size, on the validation anchor and on the test set (1000 graphs, looked at only after 1F
# training), and the paired per-graph difference SB - baseline (seeds pooled per arm, bootstrap 95% CI, paired.py).
# Arms: SB omega = 0.35 soft target, seeds 0-2, 800 epochs; matched independent coupling (omega = 0 on the SB code path,
# single GPU), seeds 0-1, 800 epochs.
# Usage: bash experiments/20261125_sbdifusco_rbsmall/final_1f.sh > OUT.md
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source cam/scripts/env.sh
R=runs/experiments/20261125_sbdifusco_rbsmall; N=20261125_sbdifusco_rbsmall
SB="${SB:-$R/${N}_sb_w0.35_soft_s[012]_e800}"; BL="${BL:-$R/${N}_sb_w0_soft_s[01]_e800}"   # overridable for a dry run
stat() {   # <glob> <tag>: mean and sample std over seeds of "Avg. size"
    for f in $1/$2_result.txt; do grep -h "Avg. size" "$f" 2>/dev/null | tail -1 | awk '{print $3}'; done |
        awk '{s+=$1; q+=$1*$1; n++} END{if(n){m=s/n; sd=(n>1)?sqrt((q-n*m*m)/(n-1)):0; printf "%.3f ± %.3f (n=%d)", m, sd, n} else printf "missing"}'
}
echo "| set | sampler | steps | SB ω = 0.35 (3 seeds) | baseline ω = 0 (2 seeds) | SB − baseline, paired [95% CI] |"
echo "|---|---|---|---|---|---|"
for SET in anchor test; do
  for S in renoise posterior; do
    if [ $SET = anchor ]; then STEPS="50 20 10 5"; else STEPS="50 10"; fi
    for T in $STEPS; do
      TAG=${SET}_${S}_t${T}_seed0
      P=$("$CAM_PYTHON" experiments/20261125_sbdifusco_rbsmall/paired.py "$BL" "$SB" $TAG --json 2>/dev/null |
          python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(f\"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['verdict']}\")" 2>/dev/null)
      echo "| $SET | $S | $T | $(stat "$SB" $TAG) | $(stat "$BL" $TAG) | ${P:-missing} |"
    done
  done
done
echo
echo "References (test set, 50 steps x 20 samples, from earlier runs): DiffUCO 19.880, CAM 19.907, DiffUCO multi-solution soft target 800 epochs 19.890."
