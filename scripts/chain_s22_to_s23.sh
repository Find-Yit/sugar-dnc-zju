#!/bin/bash
# 等 S2.2 结束 → 找到 coarse .pt → 自动跑 S2.3 mesh 提取
source /scratch/e1351071/zju_test/env.sh
LOG=$PROJ_ROOT/logs/s22_coarse_baseline.log
i=0
while ! grep -q "S2.2 END" $LOG && [ $i -lt 2400 ]; do sleep 5; i=$((i+1)); done
echo "S2.2 END detected at $(date) (waited $((i*5))s)"
PT=$(ls -t $PROJ_ROOT/outputs/runs/coarse_baseline/*/15000.pt 2>/dev/null | head -1)
if [ -z "$PT" ]; then
  echo "!!! 未找到 15000.pt，列出目录："
  find $PROJ_ROOT/outputs/runs/coarse_baseline -maxdepth 3 -type f | head -20
  exit 1
fi
echo "找到 coarse checkpoint: $PT"
bash $PROJ_ROOT/scripts/run_s23_mesh_baseline.sh "$PT"
