#!/usr/bin/env bash
# =============================================================================
# 一键复现：3DGS 7k -> coarse SuGaR（λ=0 / 0.05 / 0.2）-> 泊松网格 -> 评测 -> 汇总
#
# 用法：
#   export PROJ_ROOT=/path/to/workdir          # 产物根目录（数据也放这里）
#   export SUGAR_DIR=/path/to/this/repo        # 本仓库（含 train_coarse_density.py）
#   bash scripts/reproduce_all.sh              # 单卡顺序执行
#
#   GPUS=0,1 bash scripts/reproduce_all.sh     # 两卡：λ=0 在 GPU0，两个 λ>0 组并行在 GPU1
#   STAGES=eval bash scripts/reproduce_all.sh  # 只重跑某几个阶段（gs,coarse,mesh,eval,report）
#   DRY_RUN=1 bash scripts/reproduce_all.sh    # 只打印命令，不执行
#
# 所有路径都走变量，没有写死的绝对路径。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- 配置（可用环境变量覆盖）
PROJ_ROOT="${PROJ_ROOT:?请先 export PROJ_ROOT=产物根目录}"
SUGAR_DIR="${SUGAR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCENE="${SCENE:-$PROJ_ROOT/data/tandt/truck}"

GS_ITER="${GS_ITER:-7000}"                 # 3DGS 训练迭代数（coarse 从这里续训）
COARSE_ITER="${COARSE_ITER:-15000}"        # coarse SuGaR 结束迭代（SuGaR 默认值）
SURFACE_LEVEL="${SURFACE_LEVEL:-0.3}"      # 泊松等值面
DECIMATION="${DECIMATION:-200000}"         # 前景/背景各自的目标三角形数
SEED="${SEED:-0}"
DNC_START="${DNC_START:-9000}"             # DNC 生效起点（与 SuGaR 的 SDF 正则同期）
LAMBDAS="${LAMBDAS:-0 0.05 0.2}"           # 三组对照的 λ
GPUS="${GPUS:-0}"                          # 逗号分隔；只给一个就是单卡顺序执行
STAGES="${STAGES:-gs,coarse,mesh,eval,report}"
DRY_RUN="${DRY_RUN:-0}"

GS_OUT="$PROJ_ROOT/outputs/baseline/gs_truck"
GS_CKPT="$GS_OUT/"                         # ⚠️ 结尾斜杠必需：SuGaR 用字符串拼接 cameras.json
RUNS_DIR="$PROJ_ROOT/outputs/runs"
LOG_DIR="$PROJ_ROOT/logs"
COARSE_SUBDIR="sugarcoarse_3Dgs${GS_ITER}_densityestim02_sdfnorm02"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPU="${#GPU_ARR[@]}"

mkdir -p "$RUNS_DIR" "$LOG_DIR" "$PROJ_ROOT/outputs/metrics"

# λ -> run 名（去掉小数点即可：0.05 -> dnc005，0.2 -> dnc02；λ=0 单独命名）
run_name_for() {
  case "$1" in
    0|0.0|0.00) echo "coarse_base_seed0" ;;
    *)          echo "coarse_dnc$(printf '%s' "$1" | tr -d '.')" ;;
  esac
}

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

run() {
  echo "+ $*"
  [[ "$DRY_RUN" == "1" ]] && return 0
  "$@"
}

banner() { echo; echo "=============== $* ($(date '+%F %T')) ==============="; }

# ---------------------------------------------------------------- 第一步：3DGS 7k
if has_stage gs; then
  banner "STEP 1  3DGS ${GS_ITER} iterations"
  if [[ -f "$GS_OUT/point_cloud/iteration_${GS_ITER}/point_cloud.ply" ]]; then
    echo "已存在，跳过：$GS_OUT/point_cloud/iteration_${GS_ITER}/point_cloud.ply"
  else
    run env CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python "$SUGAR_DIR/gaussian_splatting/train.py" \
      -s "$SCENE" -m "$GS_OUT" \
      --iterations "$GS_ITER" --save_iterations "$GS_ITER" --test_iterations "$GS_ITER" --eval
  fi
fi

# ---------------------------------------------------------------- 第二步：三组 coarse 训练
train_one() {
  local lam="$1" gpu="$2" run_name out log
  run_name="$(run_name_for "$lam")"
  out="$RUNS_DIR/$run_name"
  log="$LOG_DIR/train_${run_name}.log"
  banner "STEP 2  coarse SuGaR  λ=$lam  ->  $run_name  (GPU $gpu)"
  if [[ -f "$out/$COARSE_SUBDIR/${COARSE_ITER}.pt" ]]; then
    echo "已存在，跳过：$out/$COARSE_SUBDIR/${COARSE_ITER}.pt"
    return 0
  fi
  run python "$SUGAR_DIR/train_coarse_density.py" \
    -s "$SCENE" -c "$GS_CKPT" -i "$GS_ITER" -o "$out" \
    --eval True --gpu "$gpu" --seed "$SEED" \
    --dnc_factor "$lam" --dnc_start "$DNC_START" 2>&1 | tee "$log"
}

if has_stage coarse; then
  i=0
  pids=()
  for lam in $LAMBDAS; do
    gpu="${GPU_ARR[$(( i % N_GPU ))]}"
    if (( N_GPU > 1 )); then
      train_one "$lam" "$gpu" &          # 多卡：后台并行
      pids+=($!)
    else
      train_one "$lam" "$gpu"            # 单卡：顺序执行（显存够也不建议并行，会互相拖慢）
    fi
    i=$(( i + 1 ))
  done
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && wait "$pid"; done
fi

# ---------------------------------------------------------------- 第三步：泊松网格提取
if has_stage mesh; then
  for lam in $LAMBDAS; do
    run_name="$(run_name_for "$lam")"
    mesh_out="$RUNS_DIR/${run_name/coarse_/mesh_}"
    banner "STEP 3  extract_mesh  $run_name"
    if compgen -G "$mesh_out/sugarmesh_*.ply" > /dev/null; then
      echo "已存在，跳过：$mesh_out/sugarmesh_*.ply"
      continue
    fi
    run python "$SUGAR_DIR/extract_mesh.py" \
      -s "$SCENE" -c "$GS_CKPT" -i "$GS_ITER" \
      -m "$RUNS_DIR/$run_name/$COARSE_SUBDIR/${COARSE_ITER}.pt" \
      -l "$SURFACE_LEVEL" -d "$DECIMATION" \
      --eval True --gpu "${GPU_ARR[0]}" -o "$mesh_out" 2>&1 \
      | tee "$LOG_DIR/mesh_${run_name}.log"
  done
fi

# ---------------------------------------------------------------- 第四步：评测 + 汇总
if has_stage eval; then
  for lam in $LAMBDAS; do
    run_name="$(run_name_for "$lam")"
    mesh_ply="$(ls -1 "$RUNS_DIR/${run_name/coarse_/mesh_}"/sugarmesh_*.ply 2>/dev/null | head -1 || true)"
    if [[ -z "$mesh_ply" ]]; then
      echo "[WARN] 找不到 $run_name 的网格，跳过评测"
      continue
    fi
    banner "STEP 4  eval  $run_name"
    run bash "$SUGAR_DIR/scripts/eval/run_eval_for_run.sh" "$run_name" \
      "$RUNS_DIR/$run_name/$COARSE_SUBDIR/${COARSE_ITER}.pt" "$mesh_ply" "${GPU_ARR[0]}"
  done
  banner "STEP 4  summarize"
  run python "$SUGAR_DIR/scripts/eval/summarize.py"
fi

# ---------------------------------------------------------------- 第五步：图表 + 报告
if has_stage report; then
  banner "STEP 5  figures & report"
  DOC_PY="${DOC_PY:-$PROJ_ROOT/tools/doc_env/bin/python}"
  if [[ -x "$DOC_PY" ]]; then
    run "$DOC_PY" "$SUGAR_DIR/scripts/deliverables/make_figures.py"
    run "$DOC_PY" "$SUGAR_DIR/scripts/deliverables/make_report.py"
  else
    echo "[WARN] 找不到文档 venv（$DOC_PY），跳过图表与报告生成。"
    echo "       图表/报告脚本只依赖 matplotlib + python-docx + Pillow，可用任意 venv 运行。"
  fi
fi

banner "ALL DONE"
echo "指标汇总：$PROJ_ROOT/outputs/metrics/summary.csv"
echo "图表：    $PROJ_ROOT/outputs/figures/"
echo "报告：    $PROJ_ROOT/outputs/reports/report.docx"
