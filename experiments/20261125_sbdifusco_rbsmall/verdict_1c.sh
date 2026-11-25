#!/bin/bash
# Stage 1C verdict (plan.md 1C) and 1B criterion 4, from the anchor evaluations (evaluation seed 0, best of 20).
# Paired per-graph differences (paired.py), both training seeds pooled per arm:
#   main:      SB omega - omega = 0 on the SB code path (single GPU, 40 graphs per step, the same training setup)
#   secondary: SB omega - original independent-coupling baselines (soft targets; 4-GPU seed 0, 2-GPU seed 1)
#   1B crit 4: single-GPU original code path (indep_soft_s0) - single-GPU omega = 0 (sb_w0_soft_s0), within +-0.03
# Usage: bash experiments/20261125_sbdifusco_rbsmall/verdict_1c.sh > OUT.txt
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
R=runs/experiments/20261125_sbdifusco_rbsmall; N=20261125_sbdifusco_rbsmall
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source cam/scripts/env.sh; PY=$CAM_PYTHON
W0="$R/${N}_sb_w0_soft_s[01]"
ORIG="$R/baseline_indep_soft_s0,$R/${N}_indep_soft_s1"
for S in renoise posterior; do
  for T in 50 20 10 5; do
    TAG=anchor_${S}_t${T}_seed0
    for W in 0.35 0.5 1.0; do
      echo "main omega=$W $($PY experiments/20261125_sbdifusco_rbsmall/paired.py "$W0" "$R/${N}_sb_w${W}_soft_s[01]" $TAG)"
      echo "orig omega=$W $($PY experiments/20261125_sbdifusco_rbsmall/paired.py "$ORIG" "$R/${N}_sb_w${W}_soft_s[01]" $TAG)"
    done
    echo "orig omega=0 $($PY experiments/20261125_sbdifusco_rbsmall/paired.py "$ORIG" "$W0" $TAG)"
    echo "1B-crit4 $($PY experiments/20261125_sbdifusco_rbsmall/paired.py "$R/${N}_sb_w0_soft_s0" "$R/${N}_indep_soft_s0" $TAG 2>&1 | tail -1)"
  done
done
