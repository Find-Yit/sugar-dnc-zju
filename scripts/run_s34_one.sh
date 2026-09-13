#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# S3.4 — one of the three official coarse-SuGaR controlled runs, followed
# immediately by its Poisson mesh extraction (identical parameters for all
# three groups: -l 0.3 -d 200000 --eval True).
#
# Usage:
#   RUN=coarse_base_seed0 DNC_FACTOR=0   GPU=0 bash scripts/run_s34_one.sh
#   RUN=coarse_dnc005     DNC_FACTOR=0.05 GPU=1 bash scripts/run_s34_one.sh
#   RUN=coarse_dnc02      DNC_FACTOR=0.2  GPU=1 bash scripts/run_s34_one.sh
#   RUN=coarse_dnc02_detach DNC_FACTOR=0.2 GPU=0 \
#       EXTRA_ARGS="--dnc_detach_depth True" bash scripts/run_s34_one.sh
#
# EXTRA_ARGS (optional, default empty) is appended verbatim to the training
# command.  Empty by default, so the three original runs are reproduced by the
# exact same commands as before.
#
# Mesh output dir follows the stage-2 convention (coarse_baseline -> mesh_baseline):
#   coarse_base_seed0 -> mesh_base_seed0 ; coarse_dnc005 -> mesh_dnc005 ; ...
# ---------------------------------------------------------------------------
set -uo pipefail

PROJ_ROOT=${PROJ_ROOT:-/scratch/e1351071/zju_test}
REPO=${REPO:-$PROJ_ROOT/repo/SuGaR_dev}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck/}   # trailing slash REQUIRED
[[ "$GS_CKPT" == */ ]] || GS_CKPT="$GS_CKPT/"

RUN=${RUN:?RUN must be set}
DNC_FACTOR=${DNC_FACTOR:?DNC_FACTOR must be set}
GPU=${GPU:?GPU must be set}
DNC_START=${DNC_START:-9000}
SEED=${SEED:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra flags appended to the training command (default: none)

OUT=$PROJ_ROOT/outputs/runs/$RUN
MESH_OUT=$PROJ_ROOT/outputs/runs/mesh_${RUN#coarse_}
TRAIN_LOG=$PROJ_ROOT/logs/s34_${RUN}.log
MESH_LOG=$PROJ_ROOT/logs/s34_mesh_${RUN#coarse_}.log

source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda
# 3 concurrent torch processes on 24 cores -> cap threads to avoid oversubscription.
# Applied identically to all three groups so the comparison stays internally fair.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$OUT" "$MESH_OUT" "$PROJ_ROOT/logs"

# ============================ TRAIN ============================
{
  echo "=== S3.4 TRAIN START $(date) ==="
  echo "run          : $RUN"
  echo "gpu          : $GPU"
  echo "repo         : $REPO"
  echo "git commit   : $(git -C "$REPO" rev-parse HEAD)"
  echo "git dirty    : $(git -C "$REPO" status --short | tr '\n' ' ')"
  echo "scene        : $SCENE"
  echo "3dgs ckpt    : $GS_CKPT"
  echo "output dir   : $OUT"
  echo "seed         : $SEED"
  echo "dnc_factor   : $DNC_FACTOR"
  echo "dnc_start    : $DNC_START"
  echo "extra args   : ${EXTRA_ARGS:-<none>}"
  echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
  echo "command      : python train_coarse_density.py -s $SCENE -c $GS_CKPT -i 7000 -o $OUT --eval True --gpu $GPU --seed $SEED --dnc_factor $DNC_FACTOR --dnc_start $DNC_START ${EXTRA_ARGS}"
  echo "==============================================="
} > "$TRAIN_LOG"

TRAIN_T0=$(date +%s)
cd "$REPO"
python train_coarse_density.py \
    -s "$SCENE" \
    -c "$GS_CKPT" \
    -i 7000 \
    -o "$OUT" \
    --eval True \
    --gpu "$GPU" \
    --seed "$SEED" \
    --dnc_factor "$DNC_FACTOR" \
    --dnc_start "$DNC_START" \
    ${EXTRA_ARGS} \
    >> "$TRAIN_LOG" 2>&1
TRAIN_RC=$?
TRAIN_T1=$(date +%s)
echo "=== S3.4 TRAIN END $(date) exit=$TRAIN_RC wallclock=$((TRAIN_T1-TRAIN_T0))s ===" >> "$TRAIN_LOG"

if [[ $TRAIN_RC -ne 0 ]]; then
    echo "TRAIN FAILED (exit $TRAIN_RC) -- skipping mesh extraction" >> "$TRAIN_LOG"
    exit $TRAIN_RC
fi

# ============================ MESH ============================
COARSE_PT=$(ls -1 "$OUT"/sugarcoarse_*/15000.pt 2>/dev/null | head -1)
{
  echo "=== S3.4 MESH START $(date) ==="
  echo "run          : $RUN"
  echo "gpu          : $GPU"
  echo "coarse .pt   : $COARSE_PT"
  echo "mesh out dir : $MESH_OUT"
  echo "command      : python extract_mesh.py -s $SCENE -c $GS_CKPT -i 7000 -m $COARSE_PT -l 0.3 -d 200000 --eval True --gpu $GPU -o $MESH_OUT"
  echo "==============================================="
} > "$MESH_LOG"

if [[ -z "$COARSE_PT" ]]; then
    echo "MESH FAILED: no 15000.pt found under $OUT" >> "$MESH_LOG"
    exit 2
fi

MESH_T0=$(date +%s)
python extract_mesh.py \
    -s "$SCENE" \
    -c "$GS_CKPT" \
    -i 7000 \
    -m "$COARSE_PT" \
    -l 0.3 \
    -d 200000 \
    --eval True \
    --gpu "$GPU" \
    -o "$MESH_OUT" \
    >> "$MESH_LOG" 2>&1
MESH_RC=$?
MESH_T1=$(date +%s)
echo "=== S3.4 MESH END $(date) exit=$MESH_RC wallclock=$((MESH_T1-MESH_T0))s ===" >> "$MESH_LOG"

# also record the mesh outcome at the tail of the train log, so one file has both
echo "=== S3.4 MESH ($RUN) finished $(date) exit=$MESH_RC wallclock=$((MESH_T1-MESH_T0))s log=$MESH_LOG ===" >> "$TRAIN_LOG"
exit $MESH_RC
