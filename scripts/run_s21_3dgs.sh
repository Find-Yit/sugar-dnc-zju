#!/bin/bash
# S2.1 — 3DGS 7000 iter on Truck（GPU0）
source /scratch/e1351071/zju_test/env.sh
export CUDA_VISIBLE_DEVICES=0
cd $PROJ_ROOT/repo/SuGaR
echo "=== S2.1 START $(date) ==="
python gaussian_splatting/train.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -m $PROJ_ROOT/outputs/baseline/gs_truck \
  --iterations 7000 --save_iterations 7000 --test_iterations 7000 --eval
echo "=== S2.1 END $(date) exit=$? ==="
