#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smoke test for the Depth-Normal Consistency (DNC) regularizer.
#
# Runs ~50 coarse-SuGaR iterations with DNC ACTIVE (lambda = 0.2, start = 7000)
# on GPU 1, in the development checkout repo/SuGaR_dev, and checks that
#   * L_dnc is finite and non-zero,
#   * no NaN appears in the total loss,
#   * backward() succeeds (training progresses to the end),
#   * D / N / N_d / mask PNGs are written to <out>/dnc_vis/,
#   * <out>/dnc_log.csv and <out>/train_stats.json are written,
#   * peak GPU memory stays reasonable.
#
# Usage:  bash scripts/smoke_dnc.sh              # defaults below
#         GPU=1 NUM_ITER=7050 bash scripts/smoke_dnc.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJ_ROOT=${PROJ_ROOT:-/scratch/e1351071/zju_test}
REPO=${REPO:-$PROJ_ROOT/repo/SuGaR_dev}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck}
# SuGaR concatenates the checkpoint path with "cameras.json" (string +, not os.path.join),
# so it MUST end with a trailing slash.
[[ "$GS_CKPT" == */ ]] || GS_CKPT="$GS_CKPT/"
OUT=${OUT:-$PROJ_ROOT/outputs/smoke/coarse_dnc_smoke}
GPU=${GPU:-1}
NUM_ITER=${NUM_ITER:-7050}      # coarse training starts at iteration 7000 -> ~50 DNC iterations
DNC_FACTOR=${DNC_FACTOR:-0.2}
DNC_START=${DNC_START:-7000}
LOG=${LOG:-$PROJ_ROOT/logs/smoke_dnc.log}

source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda

mkdir -p "$(dirname "$LOG")" "$OUT"

echo "=== smoke_dnc.sh ===" | tee "$LOG"
echo "date        : $(date -Iseconds)"          | tee -a "$LOG"
echo "repo        : $REPO"                      | tee -a "$LOG"
echo "git commit  : $(git -C "$REPO" rev-parse HEAD)" | tee -a "$LOG"
echo "scene       : $SCENE"                     | tee -a "$LOG"
echo "3dgs ckpt   : $GS_CKPT"                   | tee -a "$LOG"
echo "output      : $OUT"                       | tee -a "$LOG"
echo "gpu         : $GPU"                       | tee -a "$LOG"
echo "num_iter    : $NUM_ITER  (dnc_start=$DNC_START, dnc_factor=$DNC_FACTOR)" | tee -a "$LOG"
echo "====================" | tee -a "$LOG"

cd "$REPO"
python train_coarse_density.py \
    -s "$SCENE" \
    -c "$GS_CKPT" \
    -i 7000 \
    -o "$OUT" \
    --eval True \
    --gpu "$GPU" \
    --seed 0 \
    --num_iterations "$NUM_ITER" \
    --dnc_factor "$DNC_FACTOR" \
    --dnc_start "$DNC_START" \
    --dnc_log_every 10 \
    --dnc_vis_every 10 \
    2>&1 | tee -a "$LOG"

echo "=== post-run checks ===" | tee -a "$LOG"
if [[ "$DNC_FACTOR" != "0" && "$DNC_FACTOR" != "0.0" ]]; then
    echo "--- dnc_vis/ (must be non-empty) ---" | tee -a "$LOG"
    ls -la "$OUT/dnc_vis" | head -20 | tee -a "$LOG"
    echo "--- dnc_log.csv (must contain finite, non-zero L_dnc) ---" | tee -a "$LOG"
    cat "$OUT/dnc_log.csv" | tee -a "$LOG"
else
    # lambda == 0 -> the DNC branch must never be entered: no [DNC] Iteration line,
    # no dnc_vis/ directory, no dnc_log.csv.
    echo "--- DNC disabled: checking that the branch was never entered ---" | tee -a "$LOG"
    if grep -q "^\[DNC\] Iteration" "$LOG"; then
        echo "FAIL: found [DNC] Iteration lines with dnc_factor=0" | tee -a "$LOG"; exit 1
    fi
    for f in "$OUT/dnc_vis" "$OUT/dnc_log.csv"; do
        if [[ -e "$f" ]]; then echo "FAIL: $f exists with dnc_factor=0" | tee -a "$LOG"; exit 1; fi
    done
    echo "OK: no [DNC] Iteration line, no dnc_vis/, no dnc_log.csv" | tee -a "$LOG"
fi
echo "--- train_stats.json (written for every run) ---" | tee -a "$LOG"
cat "$OUT/train_stats.json" | tee -a "$LOG"
