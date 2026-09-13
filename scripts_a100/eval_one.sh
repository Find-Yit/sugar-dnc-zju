#!/usr/bin/env bash
# M — 单组几何评测（只跑 eval_geometry.py，默认跳过 render_mesh_views 省时间）
#   TAG=base_d10_q01 bash scripts/m/eval_one.sh
#   SKIP_VIS=0 TAG=xx bash scripts/m/eval_one.sh   # 也跑 mesh 可视化
set -uo pipefail
source /scratch/users/nus/e1351071/test_zju/env.sh
TAG="${TAG:?TAG must be set}"
SRC_COARSE="${SRC_COARSE:-coarse_base_seed0}"
RUN_NAME="${RUN_NAME:-m_${TAG}}"
MESH_DIR="${MESH_DIR:-$PROJ_ROOT/outputs/m/mesh_${TAG}}"
MESH_PLY=$(ls -1S "$MESH_DIR"/*.ply 2>/dev/null | head -1)
SCENE="${SCENE:-$PROJ_ROOT/data/tandt/truck}"
GS_CKPT="${GS_CKPT:-$PROJ_ROOT/outputs/baseline/gs_truck}"
EVAL_DIR="${EVAL_DIR:-$PROJ_ROOT/repo/sugar-dnc-zju/scripts/eval}"
GPU="${GPU:-0}"
SKIP_VIS="${SKIP_VIS:-1}"
LOG="$PROJ_ROOT/logs/m_eval_${TAG}.log"
export OMP_NUM_THREADS=${M_OMP:-12}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$PROJ_ROOT/outputs/metrics" "$PROJ_ROOT/logs"
{
  echo "=== M EVAL START $(date) tag=$TAG run_name=$RUN_NAME ==="
  echo "mesh_ply=$MESH_PLY"
  if [ -z "$MESH_PLY" ]; then echo "[FAIL] no .ply under $MESH_DIR"; exit 3; fi

  # provenance：合并 extract_stats.json / prune_stats.json（如有）
  TAG="$TAG" RUN_NAME="$RUN_NAME" SRC_COARSE="$SRC_COARSE" MESH_DIR="$MESH_DIR" python - <<'PY'
import json, os
run = os.environ["RUN_NAME"]; tag = os.environ["TAG"]
src = os.environ["SRC_COARSE"]; mesh_dir = os.environ["MESH_DIR"]
proj = os.environ["PROJ_ROOT"]
def load(name):
    p = os.path.join(mesh_dir, name)
    if os.path.exists(p):
        try: return json.load(open(p))
        except Exception as e: return {"_parse_error": str(e)}
    return None
def load_beside_pt():
    """剪枝统计常与剪枝后的 .pt 放在一起（<stem>_prune_stats.json / prune_stats.json）。"""
    es = load("extract_stats.json") or {}
    pt = es.get("coarse_model_path")
    if not pt:
        return None
    d, stem = os.path.dirname(pt), os.path.splitext(os.path.basename(pt))[0]
    for cand in (os.path.join(d, stem + "_prune_stats.json"),
                 os.path.join(d, "prune_stats.json")):
        if os.path.exists(cand):
            try: return json.load(open(cand))
            except Exception as e: return {"_parse_error": str(e), "_path": cand}
    return None
def readf(name):
    p = os.path.join(mesh_dir, name)
    return open(p).read().strip() if os.path.exists(p) else None
wc = readf(".extract_wallclock_s")
out = {
    "tag": tag, "run_name": run, "source_coarse": src, "mesh_dir": mesh_dir,
    "note": "仅 mesh 提取端消融：coarse 高斯与 %s 相同（除非 COARSE_PT 被覆盖）；渲染指标不重跑。" % src,
    "extract_stats": load("extract_stats.json"),
    "prune_stats": load("prune_stats.json") or load_beside_pt(),
    "extract_rc": readf(".extract_rc"),
    "extract_wallclock_s_shell": int(wc) if wc and wc.isdigit() else None,
}
os.makedirs(os.path.join(proj, "outputs", "metrics"), exist_ok=True)
dst = os.path.join(proj, "outputs", "metrics", "provenance_%s.json" % run)
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=2)
print("provenance ->", dst)
PY

  echo "--- eval_geometry ---"
  python "$EVAL_DIR/eval_geometry.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
    --mesh_path "$MESH_PLY" --run_name "$RUN_NAME" --out_dir "$PROJ_ROOT/outputs/metrics"
  RC=$?
  if [ "${SKIP_VIS}" != "1" ]; then
    echo "--- render_mesh_views ---"
    python "$EVAL_DIR/render_mesh_views.py" --scene_path "$SCENE" --gs_checkpoint "$GS_CKPT" \
      --mesh_path "$MESH_PLY" --run_name "$RUN_NAME" --gpu "$GPU"
  fi
  echo "=== M EVAL END $(date) tag=$TAG exit=$RC ==="
} 2>&1 | tee "$LOG"
