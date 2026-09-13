#!/usr/bin/env bash
# S3.f — one mesh-extraction-side ablation run on the SHARED coarse model
# (coarse_base_seed0). Only the new Poisson flags change; -l 0.3 -d 200000 --eval True
# are identical to stage 2/3.4, so every group is comparable to mesh_base_seed0.
#
#   TAG=pdauto PD=auto            bash scripts/run_s3f_extract_one.sh
#   TAG=q0     PD=10  VDQ=0       bash scripts/run_s3f_extract_one.sh
#   TAG=pd8    PD=8              bash scripts/run_s3f_extract_one.sh
set -uo pipefail
PROJ_ROOT=${PROJ_ROOT:-/scratch/e1351071/zju_test}
REPO=${REPO:-$PROJ_ROOT/repo/SuGaR_dev}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck/}
[[ "$GS_CKPT" == */ ]] || GS_CKPT="$GS_CKPT/"
SRC_RUN=${SRC_RUN:-coarse_base_seed0}
COARSE_PT=${COARSE_PT:-$PROJ_ROOT/outputs/runs/$SRC_RUN/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt}
TAG=${TAG:?TAG must be set}
PD=${PD:-10}
VDQ=${VDQ:-0.1}
RATIO=${RATIO:-100}
GPU=${GPU:-0}
OUT=$PROJ_ROOT/outputs/runs/mesh_base_seed0_${TAG}
LOG=$PROJ_ROOT/logs/s3f_mesh_${TAG}.log
source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false; export CUDA_HOME=/usr/local/cuda
mkdir -p "$OUT"
CMD="python extract_mesh.py -s $SCENE -c $GS_CKPT -i 7000 -m $COARSE_PT -l 0.3 -d 200000 --eval True --gpu $GPU -o $OUT --poisson_depth $PD --vertices_density_quantile $VDQ --cell_size_nn_distance_ratio $RATIO"
{
  echo "=== S3.f MESH START $(date) tag=$TAG ==="
  echo "source coarse run : $SRC_RUN"
  echo "coarse .pt        : $COARSE_PT"
  echo "poisson_depth     : $PD"
  echo "vertices_density_quantile : $VDQ"
  echo "cell_size_nn_distance_ratio : $RATIO"
  echo "mesh out dir      : $OUT"
  echo "command           : $CMD"
  echo "OMP_NUM_THREADS   : $OMP_NUM_THREADS"
  echo "========================================="
} > "$LOG"
T0=$(date +%s)
cd "$REPO"
$CMD >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
echo "=== S3.f MESH END $(date) tag=$TAG exit=$RC wallclock=$((T1-T0))s ===" >> "$LOG"
exit $RC
