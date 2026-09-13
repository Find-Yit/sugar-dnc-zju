#!/bin/bash
# S2.2 — coarse SuGaR baseline (density 正则, 7k→15k)，GPU0
source /scratch/e1351071/zju_test/env.sh
cd $PROJ_ROOT/repo/SuGaR
echo "=== S2.2 START $(date) ==="
python train_coarse_density.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -c $PROJ_ROOT/outputs/baseline/gs_truck/ \
  -i 7000 \
  -o $PROJ_ROOT/outputs/runs/coarse_baseline \
  --eval True --gpu 0
echo "=== S2.2 END $(date) exit=$? ==="
