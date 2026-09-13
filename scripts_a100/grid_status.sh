#!/usr/bin/env bash
# M — 网格状态：排队 / 提取中 / 评测中 / 完成 / 失败(exit code)
#   bash scripts/m/grid_status.sh [config file]
set -uo pipefail
source /scratch/users/nus/e1351071/test_zju/env.sh
CFG=${1:-}
STATE=$PROJ_ROOT/outputs/m/.grid_state
printf "%-24s %-4s %-12s %s\n" TAG GPU STATUS DETAIL
if [ -n "$CFG" ]; then LINES=$(grep -v '^\s*#' "$CFG" | grep -v '^\s*$')
else LINES=$(ls "$STATE" 2>/dev/null | sed 's/\.\(extract\|eval\)$//' | sort -u | sed 's/$/|?||/'); fi
echo "$LINES" | while IFS='|' read -r TAG GPU _ _; do
  TAG=$(echo "$TAG" | xargs); GPU=$(echo "$GPU" | xargs)
  E=$(cat "$STATE/$TAG.extract" 2>/dev/null); V=$(cat "$STATE/$TAG.eval" 2>/dev/null)
  M=$PROJ_ROOT/outputs/metrics/geometry_m_${TAG}.json
  if [ -z "$E" ]; then S=排队; D=""
  elif [ "$E" = running ]; then S=提取中; D=$(tail -1 "$PROJ_ROOT/logs/m_mesh_${TAG}.log" 2>/dev/null | cut -c1-70)
  elif [ "$E" != 0 ]; then S=失败; D="extract exit=$E (log: logs/m_mesh_${TAG}.log)"
  elif [ "$V" = running ]; then S=评测中; D=""
  elif [ -n "$V" ] && [ "$V" != 0 ]; then S=失败; D="eval exit=$V"
  elif [ -f "$M" ]; then S=完成; D="$M"
  else S=评测中; D=""; fi
  printf "%-24s %-4s %-12s %s\n" "$TAG" "$GPU" "$S" "$D"
done
