#!/usr/bin/env bash
# 07:59 新作业(1×H200, hopper-13)后由 Fable 启动：顺序重跑两组 λ，各自独占 GPU0
cd /scratch/e1351071/zju_test
echo "=== SEQ START $(date) host=$(hostname) job=$PBS_JOBID ==="
RUN=coarse_dnc02  DNC_FACTOR=0.2  GPU=0 bash scripts/run_s34_one.sh
echo "=== dnc02 chain exit=$? $(date) ==="
RUN=coarse_dnc005 DNC_FACTOR=0.05 GPU=0 bash scripts/run_s34_one.sh
echo "=== dnc005 chain exit=$? $(date) ==="
echo "=== SEQ END $(date) ==="
