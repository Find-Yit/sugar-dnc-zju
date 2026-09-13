#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# S3.e — control group `coarse_official_dnc`:
#   the OFFICIAL SuGaR depth-normal-consistency coarse trainer
#   (sugar_trainers/coarse_density_and_dn_consistency.py, factor 0.05, start 9000)
# run under exactly the same protocol as the other S3.4 runs, followed by
# Poisson mesh extraction (-l 0.3 -d 200000 --eval True) and the S4 evaluation.
#
# GPU policy: there is only ONE H200 in this PBS job, so the script FIRST WAITS
# for executor 7's `coarse_dnc02_detach` chain to announce
#     "MESH (coarse_dnc02_detach) finished"
# in logs/s34_coarse_dnc02_detach.log (or logs/s34_mesh_dnc02_detach.log).
# If the signal has not appeared by $DEADLINE (default 0935) the script exits
# WITHOUT launching anything, so two jobs can never share the GPU.
# ---------------------------------------------------------------------------
set -uo pipefail

PROJ_ROOT=${PROJ_ROOT:-/scratch/e1351071/zju_test}
REPO=${REPO:-$PROJ_ROOT/repo/SuGaR_dev}
SCENE=${SCENE:-$PROJ_ROOT/data/tandt/truck}
GS_CKPT=${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck/}   # trailing slash REQUIRED
[[ "$GS_CKPT" == */ ]] || GS_CKPT="$GS_CKPT/"

RUN=coarse_official_dnc
GPU=${GPU:-0}
SEED=${SEED:-0}
DEADLINE=${DEADLINE:-0935}      # HHMM local time; no launch after this
POLL=${POLL:-20}                # seconds between polls
SKIP_WAIT=${SKIP_WAIT:-0}

OUT=$PROJ_ROOT/outputs/runs/$RUN
MESH_OUT=$PROJ_ROOT/outputs/runs/mesh_official_dnc
TRAIN_LOG=$PROJ_ROOT/logs/s34_${RUN}.log
MESH_LOG=$PROJ_ROOT/logs/s34_mesh_official_dnc.log
WAIT_LOG=$PROJ_ROOT/logs/s3e_official_dnc_waiter.log

SIGNAL='MESH (coarse_dnc02_detach) finished'
SIG_FILES=("$PROJ_ROOT/logs/s34_coarse_dnc02_detach.log" "$PROJ_ROOT/logs/s34_mesh_dnc02_detach.log")

# ============================ WAIT ============================
{
echo "=== S3.e WAITER START $(date) deadline=$DEADLINE poll=${POLL}s ==="
if [[ "$SKIP_WAIT" == "1" ]]; then
    echo "SKIP_WAIT=1 -> not waiting for the signal."
else
    while true; do
        for f in "${SIG_FILES[@]}"; do
            if grep -qF "$SIGNAL" "$f" 2>/dev/null; then
                echo "=== signal found in $f at $(date) ==="
                FOUND=1
            fi
        done
        [[ "${FOUND:-0}" == "1" ]] && break
        NOW=$(date +%H%M)
        if [[ 10#$NOW -ge 10#$DEADLINE ]]; then
            echo "=== DEADLINE $DEADLINE reached at $(date); signal never appeared. NOT launching. ==="
            exit 3
        fi
        sleep "$POLL"
    done
fi

# Belt and braces: make sure no SuGaR job of ours is still holding the GPU.
for i in $(seq 1 60); do
    BUSY=$(pgrep -u "$(id -u)" -f 'train_coarse_density.py|extract_mesh.py|train_coarse_official_dnc.py' | grep -v "^$$\$" | wc -l)
    [[ "$BUSY" == "0" ]] && break
    echo "waiting for $BUSY of our GPU processes to exit ($(date))"
    sleep 10
done
echo "=== S3.e WAITER END $(date), launching $RUN ==="
} 2>&1 | tee -a "$WAIT_LOG"
WAIT_RC=${PIPESTATUS[0]}
[[ $WAIT_RC -ne 0 ]] && exit $WAIT_RC

source /scratch/e1351071/zju_test/env.sh
# Same thread cap as the three reported S3.4 runs (run_s34_one.sh default).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$OUT" "$MESH_OUT" "$PROJ_ROOT/logs"

# ============================ TRAIN ============================
{
  echo "=== S3.e TRAIN START $(date) ==="
  echo "run          : $RUN"
  echo "gpu          : $GPU"
  echo "repo         : $REPO"
  echo "git commit   : $(git -C "$REPO" rev-parse HEAD)"
  echo "git dirty    : $(git -C "$REPO" status --short | tr '\n' ' ')"
  echo "scene        : $SCENE"
  echo "3dgs ckpt    : $GS_CKPT"
  echo "output dir   : $OUT"
  echo "seed         : $SEED"
  echo "regularizer  : OFFICIAL dn_consistency (factor 0.05, start 9000, hard-coded upstream)"
  echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
  echo "command      : python train_coarse_official_dnc.py -s $SCENE -c $GS_CKPT -i 7000 -o $OUT --eval True --gpu $GPU --seed $SEED"
  echo "==============================================="
} > "$TRAIN_LOG"

TRAIN_T0=$(date +%s)
cd "$REPO" || exit 9
python train_coarse_official_dnc.py \
    -s "$SCENE" \
    -c "$GS_CKPT" \
    -i 7000 \
    -o "$OUT" \
    --eval True \
    --gpu "$GPU" \
    --seed "$SEED" \
    >> "$TRAIN_LOG" 2>&1
TRAIN_RC=$?
TRAIN_T1=$(date +%s)
echo "=== S3.e TRAIN END $(date) exit=$TRAIN_RC wallclock=$((TRAIN_T1-TRAIN_T0))s ===" >> "$TRAIN_LOG"

if [[ $TRAIN_RC -ne 0 ]]; then
    echo "TRAIN FAILED (exit $TRAIN_RC) -- skipping mesh extraction" >> "$TRAIN_LOG"
    exit $TRAIN_RC
fi

# ============================ MESH ============================
COARSE_PT=$(ls -1 "$OUT"/sugarcoarse_*/15000.pt 2>/dev/null | head -1)
{
  echo "=== S3.e MESH START $(date) ==="
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
echo "=== S3.e MESH END $(date) exit=$MESH_RC wallclock=$((MESH_T1-MESH_T0))s ===" >> "$MESH_LOG"
echo "=== S3.e MESH ($RUN) finished $(date) exit=$MESH_RC wallclock=$((MESH_T1-MESH_T0))s log=$MESH_LOG ===" >> "$TRAIN_LOG"
[[ $MESH_RC -ne 0 ]] && exit $MESH_RC

# ============================ EVAL ============================
MESH_PLY=$(ls -1 "$MESH_OUT"/sugarmesh_*level03_decim200000.ply 2>/dev/null | head -1)
echo "=== S3.e EVAL START $(date) mesh_ply=$MESH_PLY ===" >> "$TRAIN_LOG"
if [[ -z "$MESH_PLY" ]]; then
    echo "EVAL FAILED: no mesh .ply found under $MESH_OUT" >> "$TRAIN_LOG"
    exit 4
fi
cd "$PROJ_ROOT" || exit 9
bash scripts/eval/run_eval_for_run.sh "$RUN" "$COARSE_PT" "$MESH_PLY" "$GPU"
EVAL_RC=$?
python scripts/eval/summarize.py >> "$TRAIN_LOG" 2>&1
echo "=== S3.e EVAL END $(date) exit=$EVAL_RC ===" >> "$TRAIN_LOG"
echo "=== S3.e ALL DONE ($RUN) $(date) ===" >> "$TRAIN_LOG"
exit $EVAL_RC
