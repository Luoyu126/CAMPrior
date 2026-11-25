#!/bin/bash
# Stage 1A of docs/20261125/plan.md: static SB coupling on 200 RB-small training graphs (no training).
# Usage: bash experiments/20261125_sbdifusco_rbsmall_coupling/launch.sh <main|omega1> <gpu0> [<gpu1> ...]
#   main   : all six omegas, K = 2^16, 2^18, 2^20, measured on 2^22 fresh starts (output coupling_stats_<shard>.jsonl)
#   omega1 : omega = 1 only, K = 2^21, 2^22, measured on 2^24 fresh starts (coupling_stats_w1_<shard>.jsonl)
set -e
cd "$(dirname "$0")/../../cam"
PY=/usr0/home/shengyuf/yunyi/envs/sb/bin/python
OUT=../runs/experiments/20261125_sbdifusco_rbsmall_coupling
mkdir -p $OUT
MODE=$1; shift
GPUS=("$@"); N=${#GPUS[@]}
if [ "$MODE" = main ]; then
  ARGS="--omegas 0.05 0.1 0.2 0.35 0.5 1.0 --K 65536 262144 1048576 --K_test 4194304 --check64 3"
else
  ARGS="--omegas 1.0 --K 2097152 4194304 --K_test 16777216 --check64 0 --tag _w1"
fi
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} nohup $PY scripts/sb/coupling_analysis.py --out $OUT --graphs 200 --seed 0 \
    $ARGS --shard $i --nshards $N > $OUT/shard_${MODE}_$i.log 2>&1 &
  echo "$MODE shard $i gpu ${GPUS[$i]} pid $!"
done
