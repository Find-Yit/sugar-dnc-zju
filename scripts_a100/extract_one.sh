#!/usr/bin/env bash
# M — 单组 mesh 提取（NSCC 4xA100 版，脱胎于 repo/scripts/run_s3f_extract_one.sh）
# 用法:
#   TAG=base_d10_q01 GPU=0 bash scripts/m/extract_one.sh
#   TAG=q0 GPU=1 EXTRA="--vertices_density_quantile 0" bash scripts/m/extract_one.sh
#   TAG=pdauto GPU=2 EXTRA="--poisson_depth_bg auto --extract_seed 0" bash scripts/m/extract_one.sh
#   TAG=pruned GPU=3 COARSE_PT=/path/to/pruned.pt bash scripts/m/extract_one.sh
# EXTRA 原样拼到 extract_mesh.py 命令末尾（可覆盖前面的同名参数）。
set -uo pipefail
source /scratch/users/nus/e1351071/test_zju/env.sh
REPO=${REPO:-$PROJ_ROOT/repo/sugar-dnc-zju}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck/}
[[ "$GS_CKPT" == */ ]] || GS_CKPT="$GS_CKPT/"
SRC_RUN=${SRC_RUN:-coarse_base_seed0}
COARSE_PT=${COARSE_PT:-$PROJ_ROOT/outputs/runs/$SRC_RUN/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt}
TAG=${TAG:?TAG must be set}
PD=${PD:-10}; VDQ=${VDQ:-0.1}; RATIO=${RATIO:-100}; GPU=${GPU:-0}
EXTRA=${EXTRA:-}
OUT=$PROJ_ROOT/outputs/m/mesh_${TAG}
LOG=$PROJ_ROOT/logs/m_mesh_${TAG}.log
export OMP_NUM_THREADS=${M_OMP:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS; export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$OUT" "$PROJ_ROOT/logs"
CMD="python extract_mesh.py -s $SCENE -c $GS_CKPT -i 7000 -m $COARSE_PT -l 0.3 -d 200000 --eval True --gpu $GPU -o $OUT --poisson_depth $PD --vertices_density_quantile $VDQ --cell_size_nn_distance_ratio $RATIO $EXTRA"
{
  echo "=== M MESH START $(date) tag=$TAG ==="
  echo "repo              : $REPO"
  echo "coarse .pt        : $COARSE_PT"
  echo "mesh out dir      : $OUT"
  echo "EXTRA             : $EXTRA"
  echo "OMP_NUM_THREADS   : $OMP_NUM_THREADS   GPU: $GPU"
  echo "command           : $CMD"
  echo "========================================="
} > "$LOG"
cd "$REPO" || exit 2
T0=$(date +%s)
$CMD >> "$LOG" 2>&1
RC=$?
if [ "$RC" -eq 139 ]; then
  echo "=== M MESH SIGSEGV(139), retry once $(date) ===" >> "$LOG"
  sleep 5
  $CMD >> "$LOG" 2>&1
  RC=$?
fi
T1=$(date +%s)
echo "=== M MESH END $(date) tag=$TAG exit=$RC wallclock=$((T1-T0))s ===" >> "$LOG"
echo "$RC" > "$OUT/.extract_rc"
echo "$((T1-T0))" > "$OUT/.extract_wallclock_s"
exit $RC
