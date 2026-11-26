#!/bin/bash
# Stage 1E (plan.md): paired per-graph differences on the validation anchor (evaluation seed 0, best of 20; one training
# seed per arm) answering the three questions of the acceptance:
#   SB gain per label form: SB (omega = 0.35) - its independent-coupling control (omega = 0), same labels
#   perturbation:           multi + perturbations - multi only (1C runs, seed 0), for both couplings
#   weighting:              T = 1 / T = 0.5 - uniform, per label family and coupling
# Usage: bash experiments/20261125_sbdifusco_rbsmall_labelvar/verdict_1e.sh > OUT.txt
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source cam/scripts/env.sh
L=runs/experiments/20261125_sbdifusco_rbsmall_labelvar/20261125_sbdifusco_rbsmall_labelvar
M=runs/experiments/20261125_sbdifusco_rbsmall/20261125_sbdifusco_rbsmall
P() { echo "$1 $("$CAM_PYTHON" experiments/20261125_sbdifusco_rbsmall/paired.py "$2" "$3" "$4" 2>&1 | tail -1)"; }
for S in renoise posterior; do
  for T in 50 10 5; do
    TAG=anchor_${S}_t${T}_seed0
    for v in single single_T1 single_T0.5 mkamis mkamis_T1 mkamis_T0.5 k128; do
      fam=${v%%_*}; w=${v#$fam}
      P "sb-gain $v" "${L}_${fam}_w0${w}_s0" "${L}_${fam}_w0.35${w}_s0" $TAG
    done
    P "perturb indep" "${M}_sb_w0_soft_s0" "${L}_mkamis_w0_s0" $TAG
    P "perturb sb" "${M}_sb_w0.35_soft_s0" "${L}_mkamis_w0.35_s0" $TAG
    P "k128 vs multi indep" "${M}_sb_w0_soft_s0" "${L}_k128_w0_s0" $TAG
    P "k128 vs multi sb" "${M}_sb_w0.35_soft_s0" "${L}_k128_w0.35_s0" $TAG
    for fam in single mkamis; do
      for c in 0 0.35; do
        for t in T1 T0.5; do P "weight $fam w$c $t" "${L}_${fam}_w${c}_s0" "${L}_${fam}_w${c}_${t}_s0" $TAG; done
      done
    done
  done
done
