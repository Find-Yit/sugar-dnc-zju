#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# S3.4b (retry) — wait until executor 6's mesh_dnc005 extraction has finished,
# then launch the 4th pre-registered run of plan §3d:
#   coarse_dnc02_detach = lambda 0.2 + --dnc_detach_depth True
#
# Why a retry: the first launch (08:33:17) fired on "SEQ END" and overlapped
# executor 6's mesh_dnc005 on GPU0, which would contaminate both wall-clock
# measurements.  That attempt was stopped at 08:35:22 (it had reached ~iter
# 9001, no checkpoint written) and is relaunched here with an exclusive GPU.
#
# NOTE: this is a NEW file on purpose.  Never edit a shell script while an
# instance of it is running -- on /hpfs-main even an atomic rename breaks the
# running bash ("error reading input file"), which is exactly what truncated
# the dnc005 chain at 08:33:01.
# ---------------------------------------------------------------------------
set -uo pipefail
PROJ_ROOT=/scratch/e1351071/zju_test
MESH_LOG=$PROJ_ROOT/logs/s34_mesh_dnc005.log
MESH_DIR=$PROJ_ROOT/outputs/runs/mesh_dnc005
DEADLINE=${DEADLINE:-0850}          # HHMM; launch anyway after this, and report the overlap
POLL=${POLL:-10}

echo "=== WAITER2 START $(date) deadline=$DEADLINE ==="
OVERLAP=no
while true; do
    if grep -q "Mesh saved" "$MESH_LOG" 2>/dev/null && compgen -G "$MESH_DIR/*.ply" > /dev/null; then
        echo "=== mesh_dnc005 done at $(date) ($(ls $MESH_DIR/*.ply)) -> launching coarse_dnc02_detach ==="
        break
    fi
    NOW=$(date +%H%M)
    if [[ 10#$NOW -ge 10#$DEADLINE ]]; then
        echo "=== DEADLINE $DEADLINE reached at $(date); mesh_dnc005 not finished. Launching ANYWAY (overlap!) ==="
        OVERLAP=yes
        break
    fi
    sleep "$POLL"
done
echo "=== overlap_with_mesh_dnc005=$OVERLAP ==="

export OMP_NUM_THREADS=12           # same as coarse_dnc02 / coarse_dnc005
cd "$PROJ_ROOT"
RUN=coarse_dnc02_detach DNC_FACTOR=0.2 GPU=0 \
    EXTRA_ARGS="--dnc_detach_depth True" \
    bash scripts/run_s34_one.sh
RC=$?
echo "=== coarse_dnc02_detach chain exit=$RC $(date) ==="
echo "=== WAITER2 END $(date) ==="
exit $RC
