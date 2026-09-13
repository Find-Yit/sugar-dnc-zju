#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# S3.4b — wait for the sequential 1-GPU chain (run_s34_seq_1gpu.sh) to print
# "SEQ END", then immediately launch the 4th pre-registered run of plan §3d:
#   coarse_dnc02_detach  =  lambda 0.2  +  --dnc_detach_depth True
# Everything else (scene, 3DGS ckpt, seed, dnc_start, mesh params) is identical
# to the three reported runs, so the only difference is the detach switch.
#
# Gives up without launching if SEQ END has not appeared by $DEADLINE
# (plan §3d: "if not finished by 09:30, abandon and report the diagnosis only").
# ---------------------------------------------------------------------------
set -uo pipefail
PROJ_ROOT=/scratch/e1351071/zju_test
SEQ_LOG=$PROJ_ROOT/logs/s34_seq_1gpu.log
DEADLINE=${DEADLINE:-0905}          # HHMM, local time; no launch after this
POLL=${POLL:-20}                    # seconds between polls

echo "=== WAITER START $(date) deadline=$DEADLINE ==="
while true; do
    if grep -q "SEQ END" "$SEQ_LOG" 2>/dev/null; then
        echo "=== SEQ END detected at $(date) -> launching coarse_dnc02_detach ==="
        break
    fi
    NOW=$(date +%H%M)
    if [[ 10#$NOW -ge 10#$DEADLINE ]]; then
        echo "=== DEADLINE $DEADLINE reached at $(date); SEQ END never appeared. NOT launching. ==="
        exit 3
    fi
    sleep "$POLL"
done

export OMP_NUM_THREADS=12           # same as coarse_dnc02 / coarse_dnc005
cd "$PROJ_ROOT"
RUN=coarse_dnc02_detach DNC_FACTOR=0.2 GPU=0 \
    EXTRA_ARGS="--dnc_detach_depth True" \
    bash scripts/run_s34_one.sh
RC=$?
echo "=== coarse_dnc02_detach chain exit=$RC $(date) ==="
echo "=== WAITER END $(date) ==="
exit $RC
