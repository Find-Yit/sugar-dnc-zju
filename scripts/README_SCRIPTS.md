# 运行脚本索引（按考核流程顺序）

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 环境 | `scripts/setup_env_step2.sh` | 装依赖（torch 2.4.1+cu121 → pytorch3d wheel → 其余包 → 编译两个 CUDA 扩展），配合 `requirements.txt` |
| 一键复现 | `scripts/reproduce_all.sh` | 3DGS 7k → coarse SuGaR（λ=0 / 0.05 / 0.2 / detach / 官方 dn_consistency）→ mesh → 评测 → 汇总 |
| 3DGS | `scripts/run_s21_3dgs.sh` | 训练 vanilla 3DGS 7000 iter（--eval） |
| coarse 基线 | `scripts/run_s22_coarse_baseline.sh`、`scripts/run_s23_mesh_baseline.sh`、`scripts/chain_s22_to_s23.sh` | 原版 coarse 训练 + Poisson 提取 |
| DNC 对照 | `scripts/run_s34_one.sh`（RUN / DNC_FACTOR / GPU / EXTRA_ARGS 环境变量）、`scripts/run_s34_seq_1gpu.sh` | 单组或单卡顺序跑 λ 组（训练 + 提取） |
| detach 诊断 | `scripts/run_s34_detach_after_seq.sh`、`scripts/run_s34_detach_after_mesh005.sh`、`scripts/run_eval_detach_after_mesh.sh` | λ=0.2 + --dnc_detach_depth |
| 官方 dn_consistency | `scripts/run_s3e_official_dnc.sh` | 用 `train_coarse_official_dnc.py` 跑官方正则对照 |
| 提取端扫参（hopper） | `scripts/run_s3f_depth_report.sh`、`scripts/run_s3f_extract_one.sh`、`scripts/run_s3f_sweep_chain.sh`、`scripts/s3f_seq_runner.sh` | Frosting 自动 Poisson 深度 / D / quantile 扫描 |
| 评测 | `scripts/eval/run_eval_for_run.sh <run> <pt> <ply> <gpu>`、`scripts/eval/run_eval_extract_only.sh`、`scripts/eval/summarize.py` | 渲染 + 几何指标，汇总 summary.csv |
| 单元测试 | `scripts/test_dnc_normal.py`、`scripts/test_dnc_detach.py`、`scripts/smoke_dnc.sh` | DNC 法向/损失/梯度检查与冒烟 |
| 提取端扩展（A100） | `scripts_a100/README.md`、`scripts_a100/extract_one.sh`、`scripts_a100/eval_one.sh`、`scripts_a100/grid_*.txt` | M1+a / M2-B 扫参（见 A100_DELIVERY.md） |
| 报告 | `scripts/deliverables/make_figures.py`、`make_report.py` | 图表与 DOCX 报告（需独立文档 venv） |

所有脚本默认路径基于 `PROJ_ROOT=/scratch/e1351071/zju_test`，换机器请改脚本顶部的变量或 `env.sh`。
