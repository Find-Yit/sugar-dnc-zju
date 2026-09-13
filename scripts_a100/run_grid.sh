#!/usr/bin/env bash
# M — 网格调度：同 GPU 串行、不同 GPU 并行，每组提取完立刻评测。
# 配置文件每行： TAG|GPU|COARSE_PT|EXTRA      （# 开头或空行忽略；COARSE_PT 可留空用默认）
#   bash scripts/m/run_grid.sh scripts/m/grid_example.txt
set -uo pipefail
source /scratch/users/nus/e1351071/test_zju/env.sh
CFG=${1:?usage: run_grid.sh <config file>}
MDIR=$PROJ_ROOT/scripts/m
GLOG=$PROJ_ROOT/logs/m_grid.log
STATE=$PROJ_ROOT/outputs/m/.grid_state
mkdir -p "$STATE" "$PROJ_ROOT/logs"
echo "=== M GRID START $(date) cfg=$CFG ===" >> "$GLOG"

GPUS=$(grep -v '^\s*#' "$CFG" | grep -v '^\s*$' | awk -F'|' '{gsub(/ /,"",$2); print $2}' | sort -u)
for G in $GPUS; do
  CHAIN=$(mktemp "$PROJ_ROOT/outputs/m/.chain_gpu${G}_XXXX.sh")
  {
    echo "#!/usr/bin/env bash"
    echo "source $PROJ_ROOT/env.sh"
    grep -v '^\s*#' "$CFG" | grep -v '^\s*$' | while IFS='|' read -r TAG GPU CPT EXTRA; do
      TAG=$(echo "$TAG" | xargs); GPU=$(echo "$GPU" | xargs); CPT=$(echo "$CPT" | xargs)
      [ "$GPU" = "$G" ] || continue
      echo "echo \"[\$(date)] GPU $G TAG $TAG extract start\" >> $GLOG"
      echo "echo running > $STATE/$TAG.extract"
      printf 'env TAG=%q GPU=%q %s EXTRA=%q bash %s/extract_one.sh; RC=$?\n' \
        "$TAG" "$GPU" "$([ -n "$CPT" ] && printf 'COARSE_PT=%q' "$CPT")" "${EXTRA:-}" "$MDIR"
      echo "echo \$RC > $STATE/$TAG.extract"
      echo "echo \"[\$(date)] GPU $G TAG $TAG extract exit=\$RC\" >> $GLOG"
      echo "if [ \$RC -eq 0 ]; then"
      echo "  echo running > $STATE/$TAG.eval"
      printf '  env TAG=%q GPU=%q bash %s/eval_one.sh >/dev/null 2>&1; ERC=$?\n' "$TAG" "$GPU" "$MDIR"
      echo "  echo \$ERC > $STATE/$TAG.eval"
      echo "  echo \"[\$(date)] GPU $G TAG $TAG eval exit=\$ERC\" >> $GLOG"
      echo "fi"
    done
    echo "echo \"[\$(date)] GPU $G chain done\" >> $GLOG"
  } > "$CHAIN"
  chmod +x "$CHAIN"
  nohup bash "$CHAIN" >> "$GLOG" 2>&1 &
  echo "[launch] GPU $G chain pid=$! script=$CHAIN" | tee -a "$GLOG"
done
echo "总日志: $GLOG ; 状态: bash $MDIR/grid_status.sh $CFG"
