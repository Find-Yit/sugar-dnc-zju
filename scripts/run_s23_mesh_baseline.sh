#!/bin/bash
# S2.3 — 从 coarse SuGaR baseline 提取 Poisson mesh，GPU0
# 用法: run_s23_mesh_baseline.sh <coarse_model .pt 绝对路径>
source /scratch/e1351071/zju_test/env.sh
COARSE_PT="$1"
if [ -z "$COARSE_PT" ]; then echo "需要传入 coarse .pt 路径"; exit 1; fi
cd $PROJ_ROOT/repo/SuGaR
echo "=== S2.3 START $(date) ==="
echo "coarse_model_path = $COARSE_PT"
python extract_mesh.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -c $PROJ_ROOT/outputs/baseline/gs_truck/ \
  -i 7000 \
  -m "$COARSE_PT" \
  -l 0.3 -d 200000 \
  --eval True --gpu 0 \
  -o $PROJ_ROOT/outputs/runs/mesh_baseline
echo "=== S2.3 END $(date) exit=$? ==="
