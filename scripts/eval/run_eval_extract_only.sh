#!/usr/bin/env bash
# S3.f — 对「只做 mesh 提取端消融」的 run 跑几何指标 + mesh 可视化。
# 不重跑渲染指标：这些 run 用的是同一个 coarse 模型（高斯完全相同），
# 渲染指标与 source_coarse 那一组逐字相同，由 summarize.py 通过 provenance_<run>.json 继承。
#   TAG=pdauto bash scripts/eval/run_eval_extract_only.sh
set -uo pipefail
PROJ_ROOT="${PROJ_ROOT:-/scratch/e1351071/zju_test}"
TAG="${TAG:?TAG must be set}"
SRC_COARSE="${SRC_COARSE:-coarse_base_seed0}"
RUN_NAME="${RUN_NAME:-base_seed0_${TAG}}"
MESH_DIR="$PROJ_ROOT/outputs/runs/mesh_base_seed0_${TAG}"
MESH_PLY=$(ls -1 "$MESH_DIR"/*.ply 2>/dev/null | head -1)
SCENE="$PROJ_ROOT/data/tandt/truck"
GS_CKPT="$PROJ_ROOT/outputs/baseline/gs_truck"
EVAL_DIR="$PROJ_ROOT/scripts/eval"
GPU="${GPU:-0}"
LOG="$PROJ_ROOT/logs/s3f_eval_${TAG}.log"
source /scratch/e1351071/virtualenvs/zju_test/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
{
  echo "=== S3.f EVAL START $(date) tag=$TAG run_name=$RUN_NAME ==="
  echo "mesh_ply=$MESH_PLY"
  if [ -z "$MESH_PLY" ]; then echo "[FAIL] no .ply under $MESH_DIR"; exit 3; fi

  # provenance：标注该 run 只改了提取参数，coarse 模型来自哪一组
  python - "$RUN_NAME" "$SRC_COARSE" "$MESH_DIR" <<'PY'
import json, os, sys
run, src, mesh_dir = sys.argv[1], sys.argv[2], sys.argv[3]
proj = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")
stats_path = os.path.join(mesh_dir, "extract_stats.json")
extra = {}
if os.path.exists(stats_path):
    st = json.load(open(stats_path))
    extra = {k: st.get(k) for k in ("poisson_depth_arg", "poisson_depth_used",
                                    "vertices_density_quantile", "cell_size_nn_distance_ratio",
                                    "extraction_wall_clock_s")}
out = os.path.join(proj, "outputs", "metrics", f"provenance_{run}.json")
json.dump({"source_coarse": src,
           "note": "仅 mesh 提取端消融：coarse 模型与 " + src + " 完全相同，只改 Poisson 参数；"
                   "渲染指标 / 训练代价由 summarize.py 从该 run 继承。",
           "extract_stats": extra}, open(out, "w"), ensure_ascii=False, indent=2)
print("provenance ->", out)
PY

  echo "--- S4.2 eval_geometry ---"
  python "$EVAL_DIR/eval_geometry.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
    --mesh_path "$MESH_PLY" --run_name "$RUN_NAME"

  echo "--- S4.3 render_mesh_views ---"
  python "$EVAL_DIR/render_mesh_views.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
    --mesh_path "$MESH_PLY" --run_name "$RUN_NAME" --gpu "$GPU"

  echo "=== S3.f EVAL END $(date) ==="
} 2>&1 | tee "$LOG"
