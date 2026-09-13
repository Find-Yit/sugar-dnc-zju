#!/usr/bin/env bash
# S3.f 顺序执行器（每次只跑一组提取，避免 3 个 Poisson 并发导致的段错误）。
# 队列：pdauto(重跑) -> pdauto_q0 -> pd8 -> pd9 -> q005；每组结束立刻评测。
set -uo pipefail
PROJ_ROOT=/scratch/e1351071/zju_test
cd "$PROJ_ROOT"

done_tag () { grep -q "MESH END" "logs/s3f_mesh_$1.log" 2>/dev/null; }
ok_tag ()   { grep -q "MESH END.*exit=0" "logs/s3f_mesh_$1.log" 2>/dev/null; }

wait_tag () { for _ in $(seq 1 300); do done_tag "$1" && return 0; sleep 10; done; echo "[TIMEOUT] $1"; return 1; }

eval_tag () {
  if ! ls "outputs/runs/mesh_base_seed0_$1"/*.ply >/dev/null 2>&1; then
    echo "[SKIP EVAL] $1 : no .ply"; return 1; fi
  TAG=$1 bash scripts/eval/run_eval_extract_only.sh >/dev/null 2>&1
  echo "eval $1 rc=$?"
}

run_tag () {   # tag pd vdq  —— 前台串行，最多重试 1 次
  local tag=$1 pd=$2 vdq=$3
  for attempt in 1 2; do
    echo "--- run $tag attempt $attempt $(date) ---"
    TAG=$tag PD=$pd VDQ=$vdq bash scripts/run_s3f_extract_one.sh
    local rc=$?
    echo "--- $tag attempt $attempt exit=$rc $(date) ---"
    [ $rc -eq 0 ] && break
    mv "logs/s3f_mesh_${tag}.log" "logs/s3f_mesh_${tag}_attempt${attempt}_exit${rc}.log" 2>/dev/null
  done
  eval_tag "$tag"
  python scripts/eval/summarize.py > "logs/s3f_summarize_${tag}.log" 2>&1
  echo "summarize after $tag rc=$?"
}

echo "=== SEQ RUNNER START $(date) ==="
wait_tag q0 && eval_tag q0
python scripts/eval/summarize.py > logs/s3f_summarize_q0.log 2>&1; echo "summarize after q0 rc=$?"
run_tag pdauto    auto 0.1
run_tag pdauto_q0 auto 0
run_tag pd8       8    0.1
run_tag pd9       9    0.1
run_tag q005      10   0.05
echo "=== SEQ RUNNER END $(date) ==="
