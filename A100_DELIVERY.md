# A100 侧交付说明（分支 `a100`，对应 `plans/A100侧交付清单_供合并报告.md`）

**结果一句话**：在同一个 coarse 模型（hopper 训练的 `coarse_base_seed0`，本机复用、未重训）上只改提取端：q=0 清洗 + M2-B DBSCAN 剪枝（p95，剪 14.2%）+ M1+a 背景 Poisson 深度 9，碎片分量 2094→1194（本机基线，**−43%**；对 hopper 基线 2137 为 −44%），边界边 38450→11225（−71%，其中一部分是 Poisson 在天空区域封口的大面片，见负结果），点到面距离 G1 +0.5%（噪声内），4 视角可见表面覆盖率 92.0%→99.98%，提取耗时 340 s→240 s；PSNR/SSIM 严格不变（高斯未动）。Frosting 原样自动深度、M1+b 表面点估密度为零收益；2D-SuGaR 原样"只留最大簇"是失败案例（背景删 80.7%，覆盖率抓不到，法向图可见）。

硬件：NSCC 节点 4×A100-SXM4-40GB（Singularity `pytorch_23.05_py3.sif`），torch 2.4.1+cu121，pytorch3d 0.7.8，open3d 0.19.0；代码基于本仓库 `dnc` 分支 commit 39c0f56（官方 SuGaR 7c10c4a）。

| 清单项 | 路径 |
|---|---|
| A1 指标汇总 | `results_a100/metrics/summary.csv`（列说明 `COLUMNS.md`）、`geometry_m_<TAG>.json`、`provenance_m_<TAG>.json`、`meshvis_m_<TAG>.json`、`meshvis_summary_m.csv`、`extract_stats_<TAG>.json`、`prune_stats_<variant>.json`、`verify_independent.json`（独立复算） |
| A2 来源与可比性 | `results_a100/notes/PROVENANCE.md`（coarse 模型复用 hopper 的 .pt；本机自己的原版提取基线是 2094，2137 是 hopper 数字） |
| A3 代码改动 | `patches_a100/m1a_q0_extract.patch`（M1/M1+a/M1+b/q 参数化/seed）、`patches_a100/m2b_cluster_prune.patch`（M2-B，规则="保留所有 size≥frac×组内点数的簇"+ 前/背景分区 eps，**不是** M2+a 贡献度评分）、`patches_a100/PARAMS.md`（参数表 + 默认=原版证据）；源码本身也在本分支（`extract_mesh.py`、`sugar_extractors/coarse_mesh.py`、`sugar_utils/cluster_prune.py`、`prune_coarse_model.py`） |
| A4 确切命令 | `results_a100/notes/COMMANDS.md` |
| A5 覆盖率指标定义 | `scripts_a100/COVERAGE_METRIC.md`（脚本 `scripts_a100/eval/render_mesh_views.py`，pytorch3d 光栅化 zbuf 命中像素 / 979×546，视角 000001/000065/000129/000193；原版 92% 是孔洞非碎片缺失；局限：奖励 Poisson 封口、抓不到背景缺失、非原版组已饱和） |
| A6 M2-B 剪枝统计 | `results_a100/notes/M2B_PRUNE_STATS.md` + `results_a100/metrics/prune_stats_*.json` |
| A7 M1+a 深度取值 | `results_a100/notes/M1A_DEPTH_STATS.md` + `results_a100/metrics/depth_report_{centers,surface}.json` |
| A8 可视化 PNG | `results_a100/figures/`：`fig_fragments_compare_base_d10_q01_vs_m2p95_bg9_q0.png`（碎片着色 2×4）、`fig_sky_closure_evidence.png`（天空封口证据）、`fig_mesh_normal_compare.png` / `fig_mesh_depth_compare.png`（6 配置×4 视角）、`fig_failure_background.png`、`fig_topology_bars.png`、单视角原图 `fragments_<TAG>_view*.png`、`<run>_mesh_normal_view*.png` |
| A9 负结果与 DEVIATION | `results_a100/notes/NEGATIVE_RESULTS.md`；完整结果记录 `results_a100/notes/RESULTS_m_nscc_2026-09-13.md`（§4 结论、§9 与 hopper 训练端统一对照、§10 封口面片限制） |
| A10 SELF_CHECK / RUNLOG | `results_a100/notes/SELF_CHECK.md`、`ENV.md`、`pip_freeze_nscc.txt` |
| B11 泛化场景 | 未提供 |
| B12 M2-A 训练中剪枝 | 未提供（1 小时时限内未做） |
| B13 网格渲染 PSNR | 未提供 |
| B14 扫参图 | `results_a100/figures/fig_sweep_fragments_G1_largest.png` |
| 复现脚本 | `scripts_a100/README.md`（剪枝 → 网格 → 评测 → 汇总 → 复算 → 可视化） |

来源披露：M1 逐字移植自 Anttwo/Frosting `frosting_extractors/coarse_shell.py` L17-49；M2 思路来自 prajwalcr/2d-sugar `gaussian_model.py:429`；M1+a/M1+b/M2-B 保留大簇规则与分区 eps 为本次自有改动。
