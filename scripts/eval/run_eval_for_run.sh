#!/usr/bin/env bash
# 对某一组 run 依次跑 S4.1 渲染指标 / S4.2 几何指标 / S4.3 mesh 可视化。
# 用法：
#   scripts/eval/run_eval_for_run.sh <run_name> <coarse_pt 或 ""> <mesh_ply> [gpu] [depth_range_from_json]
# 例：
#   scripts/eval/run_eval_for_run.sh baseline /path/15000.pt /path/mesh.ply 1
set -euo pipefail

RUN_NAME="${1:?需要 run_name}"
COARSE_PT="${2:-}"
MESH_PLY="${3:?需要 mesh .ply}"
GPU="${4:-1}"
DEPTH_FROM="${5:-}"

PROJ_ROOT="${PROJ_ROOT:-/scratch/e1351071/zju_test}"
SCENE="$PROJ_ROOT/data/tandt/truck"
GS_CKPT="$PROJ_ROOT/outputs/baseline/gs_truck"
EVAL_DIR="$PROJ_ROOT/scripts/eval"
LOG="$PROJ_ROOT/logs/eval_${RUN_NAME}_$(date +%H%M%S).log"

{
  echo "=== EVAL START $(date) run=$RUN_NAME gpu=$GPU ==="
  echo "coarse_pt=$COARSE_PT"
  echo "mesh_ply=$MESH_PLY"

  echo "--- S4.1 eval_render ---"
  if [ -n "$COARSE_PT" ]; then
    python "$EVAL_DIR/eval_render.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
      --iteration 7000 --coarse_pt "$COARSE_PT" --run_name "$RUN_NAME" --gpu "$GPU"
  else
    python "$EVAL_DIR/eval_render.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
      --iteration 7000 --run_name "$RUN_NAME" --gpu "$GPU"
  fi

  echo "--- S4.2 eval_geometry ---"
  python "$EVAL_DIR/eval_geometry.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
    --mesh_path "$MESH_PLY" --run_name "$RUN_NAME"

  echo "--- S4.3 render_mesh_views ---"
  if [ -n "$DEPTH_FROM" ]; then
    python "$EVAL_DIR/render_mesh_views.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
      --mesh_path "$MESH_PLY" --run_name "$RUN_NAME" --gpu "$GPU" --depth_range_from "$DEPTH_FROM"
  else
    python "$EVAL_DIR/render_mesh_views.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
      --mesh_path "$MESH_PLY" --run_name "$RUN_NAME" --gpu "$GPU"
  fi

  echo "--- S4.4 summarize ---"
  python "$EVAL_DIR/summarize.py"
  echo "=== EVAL END $(date) exit=0 ==="
} 2>&1 | tee "$LOG"
echo "日志 -> $LOG"
