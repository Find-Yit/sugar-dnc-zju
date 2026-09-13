#!/usr/bin/env bash
# S3.f — 等第一波（pdauto/q0，已在跑）结束后，依次跑剩余的提取扫参 + 每组的几何评测。
# 波次：wave1 = pdauto,q0（外部已启动） -> wave2 = pdauto_q0,pd8 -> wave3 = pd9,q005
set -uo pipefail
PROJ_ROOT=/scratch/e1351071/zju_test
cd "$PROJ_ROOT"

wait_tag () {  # $1=tag ; 等其 MESH END 行出现
  local tag=$1
  for _ in $(seq 1 240); do
    if grep -q "MESH END" "logs/s3f_mesh_${tag}.log" 2>/dev/null; then return 0; fi
    sleep 10
  done
  echo "[TIMEOUT] $tag"; return 1
}

eval_tag () { TAG=$1 bash scripts/eval/run_eval_extract_only.sh >/dev/null 2>&1; echo "eval $1 rc=$?"; }

run_wave () {  # $1,$2 = "TAG:PD:VDQ"
  for spec in "$@"; do
    IFS=: read -r tag pd vdq <<< "$spec"
    nohup env TAG=$tag PD=$pd VDQ=$vdq bash scripts/run_s3f_extract_one.sh \
      > "logs/s3f_wrap_${tag}.log" 2>&1 &
    sleep 2
  done
  for spec in "$@"; do IFS=: read -r tag _ _ <<< "$spec"; wait_tag "$tag"; done
  for spec in "$@"; do IFS=: read -r tag _ _ <<< "$spec"; eval_tag "$tag"; done
}

echo "=== CHAIN START $(date) ==="
wait_tag pdauto; wait_tag q0
echo "--- wave1 meshes done $(date) ---"
eval_tag pdauto; eval_tag q0
python scripts/eval/summarize.py > logs/s3f_summarize_wave1.log 2>&1; echo "summarize wave1 rc=$?"
echo "--- wave2 start $(date) ---"
run_wave "pdauto_q0:auto:0" "pd8:8:0.1"
python scripts/eval/summarize.py > logs/s3f_summarize_wave2.log 2>&1; echo "summarize wave2 rc=$?"
echo "--- wave3 start $(date) ---"
run_wave "pd9:9:0.1" "q005:10:0.05"
python scripts/eval/summarize.py > logs/s3f_summarize_wave3.log 2>&1; echo "summarize wave3 rc=$?"
echo "=== CHAIN END $(date) ==="
