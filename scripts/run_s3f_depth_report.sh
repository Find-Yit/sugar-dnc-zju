#!/usr/bin/env bash
# S3.f — only compute + print the Frosting automatic Poisson depth for a coarse SuGaR model.
# Does NOT extract any mesh (--only_report_depth True).
set -uo pipefail
PROJ_ROOT=${PROJ_ROOT:-/scratch/e1351071/zju_test}
REPO=${REPO:-$PROJ_ROOT/repo/SuGaR_dev}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck/}
COARSE_PT=${COARSE_PT:-$PROJ_ROOT/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt}
OUT=${OUT:-$PROJ_ROOT/outputs/runs/_depth_report_base_seed0}
RATIO=${RATIO:-100}
GPU=${GPU:-0}
LOG=${LOG:-$PROJ_ROOT/logs/s3f_depth_report.log}
source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false; export CUDA_HOME=/usr/local/cuda
mkdir -p "$OUT"
{
  echo "=== S3.f DEPTH REPORT START $(date) ==="
  echo "coarse .pt : $COARSE_PT"
  echo "ratio      : $RATIO"
  echo "command    : python extract_mesh.py -s $SCENE -c $GS_CKPT -i 7000 -m $COARSE_PT -l 0.3 -d 200000 --eval True --gpu $GPU -o $OUT --poisson_depth auto --cell_size_nn_distance_ratio $RATIO --only_report_depth True"
  echo "======================================="
} > "$LOG"
T0=$(date +%s)
cd "$REPO"
python extract_mesh.py -s "$SCENE" -c "$GS_CKPT" -i 7000 -m "$COARSE_PT" \
  -l 0.3 -d 200000 --eval True --gpu "$GPU" -o "$OUT" \
  --poisson_depth auto --cell_size_nn_distance_ratio "$RATIO" --only_report_depth True >> "$LOG" 2>&1
RC=$?
echo "=== S3.f DEPTH REPORT END $(date) exit=$RC wallclock=$(( $(date +%s)-T0 ))s ===" >> "$LOG"
exit $RC
