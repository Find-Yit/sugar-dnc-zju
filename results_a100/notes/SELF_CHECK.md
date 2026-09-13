# SELF_CHECK.md — A100 侧自检（清单第 10 项）

> 逐条对照 `plans/新方案_A100四卡_SuGaR改进.md` §5「验收标准」与
> `plans/A100侧交付清单_供合并报告.md` A 组 1–10 项。
> 判定口径：**PASS** = 有可追溯的文件/数字支撑；**FAIL** = 做了但未达标；**未提供** = 本次未做。
> 路径相对项目根 `/scratch/users/nus/e1351071/test_zju`；`R = repo/sugar-dnc-zju/results_a100`。

## 一、计划 §5 验收标准

| # | 条目 | 实际值 / 路径 | 判定 |
|---|---|---|---|
| 5.1 | 默认参数下的提取结果与原版**逐位**一致 | 提取含随机采样且**原版无种子**，逐位不可能。实测：同一 .pt、同一参数重复两次（`base_d10_q0` vs `frosting_q0_s0`）连通分量 1651 vs 1645（**0.4%**），G1/G2/G5 差 <0.2%。默认参数组 `base_d10_q01`（D=10/10, q=0.1）碎片 2094 vs hopper 原版 2137（**2%**，跨机）。→ **噪声内一致，非逐位** | **PASS（口径放宽）**，逐位一致一项 **FAIL**（不可达，已在 `PROVENANCE.md` §2 披露） |
| 5.2 | 每组 run 有 `extract_stats.json` / 日志 / 命令；数字只来自 `summary.csv` | 9 组齐全：`R/metrics/extract_stats_<TAG>.json`（9 个）、`logs/m_mesh_<TAG>.log`（9 个）、命令见 `R/notes/COMMANDS.md`；所有表格数字均出自 `R/metrics/summary.csv` | **PASS** |
| 5.2b | `train_stats.json` | **未提供**：本机不重训，coarse 由 hopper 训练（hopper 侧持有 `outputs/runs/coarse_base_seed0/train_stats.json`；其关键数字已摘入 `PROVENANCE.md` §1.2：8001 iter / 12.15 min / 91.1 ms per iter / 7593 MiB / 429423 高斯） | **未提供**（不适用） |
| 5.3 | 独立复算：抽 3 组 mesh 重算碎片数与 G1 中位 | `R/metrics/verify_independent.json`，`scripts/m/verify_independent.py`（自写 union-find / 边表 / 精确点面距离，不走 `eval_geometry.py` 代码路径）。3 组 `base_d10_q01` / `m1a_q0` / `m2p95_q0`：顶点数、面数、连通分量、碎片数、边界边 **五项逐位一致（rel = 0.0）**；G1 精确口径相对差 **0.39% / 1.07% / 1.68%**（3000 点子样本抽样噪声内）；D_fg/D_bg 与剪枝比核对通过；`cameras_extent` 复算 5.847915 一致 | **PASS** |
| 5.3b | 抽 3 个视角从 PNG 重算 PSNR | **未提供**：提取端不改高斯，渲染 PSNR 严格不变（恒等于 hopper baseline 24.6530）。替代做法：新增**网格像素覆盖率**在同样 4 个固定视角上光栅化复核（`R/metrics/meshvis_summary_m.csv`） | **未提供**（有替代证据） |
| 5.4 | 披露 M1/M2 的来源仓库与行号；自有改动单独成节；负结果如实写 | 来源见 `R/notes/RESULTS_m_nscc_2026-09-13.md` §6：M1 = Anttwo/Frosting `frosting_extractors/coarse_shell.py` **L17-49**（`compute_optimal_poisson_depth`，ECCV 2024）；M2 = prajwalcr/2d-sugar `gaussian_model.py:429` + `train.py:140`。自有改动 M1+a/M1+b（`sugar_extractors/coarse_mesh.py` 的 `compute_poisson_depth_from_points` / `compute_optimal_poisson_depth_bg`）、M2-B 保留大簇规则 + M2+b 分区 eps（`sugar_utils/cluster_prune.py`、`prune_coarse_model.py`）。负结果见 `R/notes/NEGATIVE_RESULTS.md` | **PASS** |
| 5.5 | 报告结构：问题 → 借鉴（注明来源）→ 我们的改动 → 三列对照证据 → 结论与代价 | 三列对照齐备（原版 `base_d10_q01` / 移植原样 `frosting_q0_s0`、`m2largest_q0` / 改进版 `m1a_q0`、`m2p95_q0`、`m2p95_bg9_q0`），见 `RESULTS_m_nscc_2026-09-13.md` §2 与 §9；代价见其 §9 读法 4 | **PASS**（报告正文由 hopper 侧合并） |

## 二、交付清单 A 组 1–10 项

| # | 条目 | 实际值 / 路径 | 判定 |
|---|---|---|---|
| **1** | 指标汇总 `summary.csv` + 各 run JSON | `R/metrics/summary.csv`（= `summary_m.csv`，9 行）+ `R/metrics/COLUMNS.md`（列说明及与 hopper 列名的一一对应）；`geometry_m_*.json` ×9、`provenance_m_*.json` ×9、`meshvis_m_*.json` ×6、`meshvis_summary_m.csv`、`extract_stats_*.json` ×9、`prune_stats_*.json` ×4、`depth_report_{centers,surface}.json`、`verify_independent.json`。覆盖：原版提取 `base_d10_q01`、q0 `base_d10_q0`、M1+a `m1a_q0`/`m1a_q01`、M2-B `m2p95_q0`、最优组合 `m2p95_bg9_q0`，外加 `frosting_q0_s0`、`m2p95_m1a_q0`、`m2largest_q0` | **PASS** |
| 1b | `render_<run>.json` / `train_stats.json` | **未提供**：提取端不改高斯，渲染指标恒定（hopper baseline 值）；不重训故无 train_stats | **未提供**（不适用） |
| **2** | run 来源与可比性说明 | `R/notes/PROVENANCE.md`（coarse 非本机重训、3DGS 7k 与 coarse 15k 的命令/seed 0/墙钟 2m13s 与 779 s、**2137 是 hopper 数字而本机原版是 2094**、硬件 4×A100-40GB、torch 2.4.1+cu121、pytorch3d 0.7.8、commit 39c0f56 / 7c10c4a） | **PASS** |
| **3** | 代码改动 diff / .patch + 新增参数默认值 | **本 Agent 未负责**（由另一路交付 `patches_a100/`）。相关新增参数：`--poisson_depth_bg`、`--depth_estimate_source`、`--extract_seed`、`--knn_percentile`、`--keep_cluster_min_frac`、`--prune_rule`、`--no_separate_fg_bg`；默认路径不变的证据见本表 5.1 | 不在本 Agent 范围 |
| **4** | 确切命令 / 日志 / 起止 / 耗时 | `R/notes/COMMANDS.md`：9 组 `command :` 行原样、START/END 时间、日志 wallclock 与 `.extract_wallclock_s` 双列，含 `m2p95_bg9_q0`（10:00:16→10:04:22，**240.16 s**）；「原版 352 s vs 最优 240 s」出处已标注（352 = `base_d10_q0` 351.66 s）；4 条剪枝命令见 §2 | **PASS**（剪枝原始日志 **未提供**，参数由 JSON 反推） |
| **5** | 覆盖率指标的定义与脚本 | **本 Agent 未负责**。数值已在 `R/metrics/meshvis_summary_m.csv` 与 `R/notes/RESULTS_m_nscc_2026-09-13.md` §8.1 给出（定义：pytorch3d `MeshRasterizer` 在 test 序号 0/8/16/24 上光栅化，判定 `pix_to_face >= 0` 的像素比例，分母 = 979×546 全幅；脚本 `scripts/eval/render_mesh_views.py`）。92.0% 低的原因：**q=0.1 的顶点密度分位数裁剪造成真孔洞**（非碎片缺失），见 `NEGATIVE_RESULTS.md` N6 | 数值 **PASS**，脚本文件本身不在本 Agent 范围 |
| **6** | M2-B 剪枝统计 | `R/notes/M2B_PRUNE_STATS.md`：4 变体的 `min_samples=6`、`knn_percentile` 90/95、fg/bg 实际 eps、簇数/保留簇数/剪掉数与比例、`cluster_sizes_top10`、**`largest_global` 背景被整片剪掉 80.7%**、明确写明用的是「保留 size ≥ frac×组内点数」规则而非 M2+a 贡献度评分；含 p90/p95 × 0.005/0.002 扫参整表（剪枝比 14.21 / 19.86 / 24.80 / 59.72%） | **PASS** |
| **7** | M1+a 的 D_fg/D_bg 与中间量 + M1+b 零收益依据 | `R/notes/M1A_DEPTH_STATS.md`：D_fg=10 / **D_bg=9**；`raw_depth_before_floor` fg 10.8732 / bg **9.9907**、surface fg 14.9248 / bg 13.1672；`quantile_dist_normalized_SQUARED`、`bbox_size` 12.8548 / 51.4612、点数（fg 77554、bg 136798、surface 7.45M / 1.93M 抽样至 500k）；centers 与 surface 两种来源；剪枝后 bg raw **9.99→10.08** 回落现象及其解法（强制 `--poisson_depth_bg 9`） | **PASS** |
| **8** | 可视化 PNG（4 固定视角） | 图片由并行另一路交付到 `R/figures/`（本 Agent 未负责生成/复制）：`fig_mesh_normal_compare.png`（法向，6 方法 × 4 视角）、`fig_mesh_depth_compare.png`（深度）、`fig_failure_background.png` 与 `fig_sky_closure_evidence.png`（**天空封口 / 背景缺失证据图**）、`fig_fragments_compare_base_d10_q01_vs_m2p95_bg9_q0.png` 与 `fragments_<TAG>_view{00,08,16,24}.png`（**碎片分量着色图**，附 `fragstats_<TAG>.json`）、`fig_topology_bars.png`、`fig_sweep_fragments_G1_largest.png`（扫参曲线）、`<TAG>_mesh_normal_view*.png`（原版 vs 最优组合单视角原图）。本机原始位置 `outputs/figures/m_nscc/`、`outputs/vis/m_<TAG>/` | **PASS**（不在本 Agent 范围，已核实文件存在） |
| **9** | 负结果与 DEVIATION | `R/notes/NEGATIVE_RESULTS.md`：N1 Frosting 零收益（raw 10.873 截断，0.4% 噪声）、N2 M1+b 零收益（raw 14.92/13.17 双截断）、N3 前景肉眼无差别（G1 跨组差 ≤2.6%）、N4 天空封口面片（边界边 −70.8% 需改口径）、N5 覆盖率抓不到背景缺失（剪 59.7% 仍 99.96%）、N6 覆盖率方向说明；DEVIATION D1–D9 含未做 M2-A / M2+a / 泛化 / 网格渲染 PSNR、提取重启一次（64 线程超订，`logs/*_aborted.log`） | **PASS** |
| **10** | SELF_CHECK / RUNLOG 索引 | 本文件 + 下方 §三 RUNLOG 索引；另附 `R/notes/RESULTS_m_nscc_2026-09-13.md`（完整结果记录）、`R/notes/ENV.md`、`R/notes/pip_freeze_nscc.txt` | **PASS** |

## 三、RUNLOG 索引（`logs/m_*.log`，共 26 个）

### 3.1 提取（`m_mesh_<TAG>.log`，9 个有效 + 4 个中止留档）
| 日志 | TAG | 状态 |
|---|---|---|
| `logs/m_mesh_base_d10_q01.log` | base_d10_q01 | exit=0，347 s |
| `logs/m_mesh_base_d10_q0.log` | base_d10_q0 | exit=0，359 s |
| `logs/m_mesh_frosting_q0_s0.log` | frosting_q0_s0 | exit=0，356 s |
| `logs/m_mesh_m1a_q0.log` | m1a_q0 | exit=0，309 s |
| `logs/m_mesh_m1a_q01.log` | m1a_q01 | exit=0，305 s |
| `logs/m_mesh_m2p95_q0.log` | m2p95_q0 | exit=0，328 s |
| `logs/m_mesh_m2p95_m1a_q0.log` | m2p95_m1a_q0 | exit=0，327 s |
| `logs/m_mesh_m2p95_bg9_q0.log` | **m2p95_bg9_q0** | exit=0，**246 s**（最优） |
| `logs/m_mesh_m2largest_q0.log` | m2largest_q0 | exit=0，309 s |
| `logs/m_mesh_base_d10_q0_aborted.log` | base_d10_q0 | **中止**（64 线程超订） |
| `logs/m_mesh_base_d10_q01_aborted.log` | base_d10_q01 | **中止** |
| `logs/m_mesh_m1a_q0_aborted.log` | m1a_q0 | **中止** |
| `logs/m_mesh_m1a_q01_aborted.log` | m1a_q01 | **中止** |

### 3.2 批量驱动
`logs/m_grid.log`（重启后有效批次）、`logs/m_grid_aborted.log`（首轮中止）

### 3.3 几何评测（`m_eval_<TAG>.log`，9 个）
`m_eval_base_d10_q0.log`、`m_eval_base_d10_q01.log`、`m_eval_frosting_q0_s0.log`、
`m_eval_m1a_q0.log`、`m_eval_m1a_q01.log`、`m_eval_m2p95_q0.log`、
`m_eval_m2p95_m1a_q0.log`、`m_eval_m2p95_bg9_q0.log`、`m_eval_m2largest_q0.log`
（另有 `m_autoeval_base_d10_q0.log`、`m_autoeval_base_d10_q01.log` 两个自动串接评测日志）

### 3.4 网格可视化 / 覆盖率（`meshvis_m_<TAG>.log`，5 个）
`meshvis_m_base_d10_q0.log`、`meshvis_m_base_d10_q01.log`、`meshvis_m_m1a_q0.log`、
`meshvis_m_m2p95_q0.log`、`meshvis_m_m2largest_q0.log`
（`m2p95_bg9_q0` 的渲染在 10:06 追加，7.9 s，GPU 0；其 JSON `meshvis_m_m2p95_bg9_q0.json` 已交付，
独立日志 **未提供**）

### 3.5 hopper 侧训练日志（供追溯，本机留存副本）
`logs/s21_3dgs.log`（3DGS 7k，05:48:58→05:51:11）、`logs/s34_coarse_base_seed0.log`（coarse 15k，06:15:43→06:28:42）、
RUNLOG：`notes/RUNLOG.md`、`notes/RUNLOG_s34.md`、`notes/RUNLOG_eval.md`

## 四、"未提供"汇总
1. `render_<run>.json` / `train_stats.json`（提取端不改高斯 + 不重训，不适用）
2. 逐位一致性证明（原版提取无种子，改以 0.4% 噪声底论证）
3. 从 PNG 重算 PSNR（同 1，改以覆盖率光栅化复核）
4. 剪枝脚本的原始运行日志（参数由 `prune_stats_*.json` 反推）
5. `default` / `frac002` 两个剪枝变体的下游网格指标（只剪未提取）
6. `m2p95_bg9_q0` 的 meshvis 独立日志（JSON 已有）
7. M2-A、M2+a、泛化场景（train/playroom）、网格渲染 PSNR（清单 B-11/12/13）
