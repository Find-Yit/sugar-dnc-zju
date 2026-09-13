#!/usr/bin/env bash
# Wait for the coarse_dnc02_detach mesh to be written, then run the standard
# evaluation (S4.1 render / S4.2 geometry / S4.3 mesh views) and re-summarize.
# New file on purpose -- never edit a script that may be running.
set -uo pipefail
PROJ_ROOT=/scratch/e1351071/zju_test
MESH_LOG=$PROJ_ROOT/logs/s34_mesh_dnc02_detach.log
MESH_DIR=$PROJ_ROOT/outputs/runs/mesh_dnc02_detach
PT=$PROJ_ROOT/outputs/runs/coarse_dnc02_detach/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt
DEADLINE=${DEADLINE:-0920}
POLL=${POLL:-10}

echo "=== EVALWAIT START $(date) deadline=$DEADLINE ==="
while true; do
    if grep -q "Mesh saved" "$MESH_LOG" 2>/dev/null && compgen -G "$MESH_DIR/*.ply" > /dev/null; then
        echo "=== mesh ready at $(date) ==="
        break
    fi
    if grep -q "MESH FAILED\|TRAIN FAILED" "$PROJ_ROOT/logs/s34_coarse_dnc02_detach.log" 2>/dev/null; then
        echo "=== chain reported FAILED; aborting eval ==="; exit 4
    fi
    NOW=$(date +%H%M)
    if [[ 10#$NOW -ge 10#$DEADLINE ]]; then
        echo "=== DEADLINE $DEADLINE reached, mesh never appeared. Aborting eval. ==="; exit 3
    fi
    sleep "$POLL"
done

PLY=$(ls -1 "$MESH_DIR"/*.ply | head -1)
echo "pt  = $PT"
echo "ply = $PLY"
cd "$PROJ_ROOT"
source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
bash scripts/eval/run_eval_for_run.sh coarse_dnc02_detach "$PT" "$PLY" 0
RC=$?
echo "=== eval exit=$RC $(date) ==="
python scripts/eval/summarize.py
echo "=== summarize exit=$? $(date) ==="
echo "=== EVALWAIT END $(date) ==="
exit $RC
