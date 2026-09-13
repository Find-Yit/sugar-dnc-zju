# COLUMNS.md — `results_a100/metrics/summary.csv` 列说明与 hopper 侧 `summary.csv` 的对应

`summary.csv` 与 `summary_m.csv` 内容完全相同（同一文件两个名字）：源文件是本机
`outputs/metrics/summary_m.csv`，由 `scripts/m/summarize.py` 从每组 run 的
`outputs/m/mesh_<TAG>/extract_stats.json` + `outputs/metrics/geometry_m_<TAG>.json` 汇总生成。

> **重要**：本表与 hopper 侧 `results/metrics/summary.csv` **不是同列格式**。
> hopper 的 summary.csv 一行 = 一个「训练 run」（含训练与渲染指标）；
> 本表一行 = 一次「网格提取 run」（同一个 coarse 模型的不同提取参数），
> 因此**没有** PSNR/SSIM/LPIPS/训练列 —— 提取端改动不修改高斯，渲染指标严格不变，
> 恒等于 hopper `baseline` 行的 PSNR 24.6530 / SSIM 0.85644 / LPIPS 0.20280。

## 列对应表

| 本表列 | 含义 | hopper `summary.csv` 对应列 |
|---|---|---|
| `TAG` | 提取 run 名（= `outputs/m/mesh_<TAG>`、`logs/m_mesh_<TAG>.log`） | `run`（语义不同：hopper 是训练 run） |
| `D_fg` | 前景 Poisson 重建深度（实际生效值，来自 `extract_stats.json:poisson_depth_fg_used`） | 无（hopper 无该参数） |
| `D_bg` | 背景 Poisson 深度（`poisson_depth_bg_used`）；= D_fg 即原版行为 | 无 |
| `quantile` | `--vertices_density_quantile`，原版默认 0.1 | 无 |
| `seed` | `--extract_seed`（本次新增参数；原版提取无种子、不可复现） | 无 |
| `prune_ratio` | M2-B DBSCAN 剪掉的高斯比例（空 = 未剪枝），来自 `prune_stats_*.json:pruned_ratio` | 无 |
| `n_components` | 网格连通分量数（越小越好） | `G4_n_components_down` |
| `n_fragments_lt100` | 面数 <100 的碎片分量数（核心指标） | `G4_n_fragments_lt100faces_down` |
| `largest_component_ratio` | 最大连通分量的面数占比，本表为 **0–1 小数** | `G4_largest_comp_face_ratio_pct_up`（hopper 为**百分数**，数值 ×100） |
| `n_boundary_edges` | 边界边数（孔洞代理量） | `G4_n_boundary_edges_down` |
| `G1_median_abs` | COLMAP 稀疏点 → mesh 的点到面距离中位数（绝对值，场景单位） | `G1_median_abs_down` |
| `G2_ratio_below_1pct` | G1 距离 < 场景尺度 1% 的点比例 | `G2_ratio_lt_1pct_up` |
| `G5_dihedral_abs_mean` | 二面角绝对值均值（度） | `G5_dihedral_abs_deg_mean_down` |
| `n_faces` | 网格面数 | `G4_n_faces` |
| `extract_wallclock_s` | 提取墙钟秒（`outputs/m/mesh_<TAG>/.extract_wallclock_s`） | 无（hopper 无该列） |

## hopper 有、本表没有的列
`n_gaussians`、`PSNR_dB_up`、`PSNR_dB_std`、`SSIM_up`、`LPIPS_VGG_down`、`n_test_views`、
`G1_mean_abs_down`、`G1_p90_abs_down`、`G1_*_rel_pct_down`、`G1_n_reference_points`、
`G2_ratio_lt_0p5pct_up`、`G3_*`、`G4_n_vertices`、`G4_fragment_faces_total_down`、
`G4_n_non_manifold_edges_down`、`G5_dihedral_deg_mean_down`、`G5_*_p90_down`、
`G5_frac_raw_gt_90deg_pct`、`train_*`、`dnc_factor`、`cameras_extent`、`*_json`。
其中 G1/G2/G3/G4/G5 的**完整字段**在本目录的 `geometry_m_<TAG>.json` 中一一俱全，
与 hopper 的 `geometry_*.json` 同口径同脚本（`scripts/eval/eval_geometry.py`），可直接按 hopper 列名重排。
`cameras_extent` = 5.847915（本机复算见 `verify_independent.json`，与 hopper 一致）。

## 附加汇总表
`meshvis_summary_m.csv`：新增的「网格像素覆盖率」指标（hopper 侧没有）。
列：`TAG, label, cov_view00_000001, cov_view08_000065, cov_view16_000129, cov_view24_000193, cov_mean, n_faces`，
值为 0–1 小数，4 个视角与 hopper `eval_render.py` 的固定测试视角完全一致（test 序号 0/8/16/24）。
