# A100 侧代码改动与新增命令行参数（交付清单 A.3）

代码基线：分支 `dnc`，commit `39c0f56`。两个 patch 都是相对该 commit 生成的。

| 文件 | 内容 |
|---|---|
| `m1a_q0_extract.patch` | `git diff`：`extract_mesh.py` + `sugar_extractors/coarse_mesh.py`。包含 **M1**（自动 Poisson 深度上限）、**M1+a**（前景/背景分离自适应深度）、**M1+b**（深度估计点源可选）、**q0 参数化**（分位清洗阈值可配）、**提取 seed**。 |
| `m2b_cluster_prune.patch` | 三个新增文件的 `git diff --no-index /dev/null <file>`：`sugar_utils/cluster_prune.py`（DBSCAN 聚类剪枝核心）、`prune_coarse_model.py`（提取前剪枝 coarse 模型的独立入口）、`scripts/test_cluster_prune.py`（单元自测）。**M2-B 用的规则是 `keep_large`，即"保留所有 size ≥ 阈值的簇"**，不是 M2+a 贡献度评分。 |

## 1. `extract_mesh.py` 新增参数（M1 / M1+a / M1+b / seed）

| 参数 | 默认值 | 含义 | 默认=原版？ |
|---|---|---|---|
| `--poisson_depth_bg` | `same` | [M1+a] 背景 Poisson 重建的八叉树深度。`same`＝复用前景深度（**原版行为**）；整数＝直接指定；`auto`＝仅用背景点自动估计。最优组合用 `9`。 | ✅ `same` 就是原版：前后景同一个深度 |
| `--depth_estimate_source` | `centers` | [M1+b] 自动深度从哪些点估计。`centers`＝高斯中心，与 Frosting 完全一致（**原版行为**）；`surface`＝采样的表面点。 | ✅ `centers` 即 Frosting 原式 |
| `--max_poisson_depth` | `10` | [M1] 自动 Poisson 深度的上界。Frosting 原代码把 10 写死在函数里。 | ✅ 默认 10 = 原版硬编码值 |
| `--extract_seed` | `-1` | [M1] 提取阶段随机种子。`-1`＝不 seed（**原版行为**）；`>=0` 时同时 seed torch / cuda / numpy / random，使表面点 `randperm` 子采样可复现。 | ✅ `-1` 不调用任何 seed 函数 |

`q0` 不是新参数，而是把 SuGaR 原有的 `--surface_level` 后清洗分位数（原版硬编码 0.1）暴露成命令行值；`q=0.1` 即原版，`q=0` 表示不做分位清洗。

## 2. `prune_coarse_model.py` 参数（M2-B，全部为新增脚本，不影响原版路径）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--rule` | `keep_large` | 剪枝规则。`keep_large`＝保留所有 size ≥ `keep_cluster_min_frac × N` 的簇（本次全部实验用的就是它）。 |
| `--keep_cluster_min_frac` | `0.005` | 簇保留阈值，占总高斯数的比例。 |
| `--separate_fg_bg` | `True` | 前景（`fg_bbox` 内）与背景分别聚类，避免整片背景被当噪声剪掉。 |
| `--fg_bbox_factor` | `1.0` | 前景 bbox 缩放系数（与 `eval_geometry.py` 同定义）。 |
| `--min_samples` | `6` | DBSCAN `min_samples`。 |
| `--knn_percentile` | `90` | 由 kNN 距离的该分位数自动定 DBSCAN `eps`。最优组合用 `95`（即 `p95`）。 |
| `--eps_estimate_subsample` | `200000` | 估 `eps` 时的点子采样上限。 |
| `--opacity_min` | `0.0` | 聚类前的不透明度预过滤下限，`0.0`＝不过滤。 |
| `--seed` | `0` | 剪枝随机种子。 |
| `--verify_reload` | `True` | 写出后重新加载校验，确保剪枝模型可被 `extract_mesh.py` 正常读取。 |

> `prune_coarse_model.py` 是**独立的前置步骤**：不跑它，`extract_mesh.py` 的行为与原版完全一致；跑它只是把 coarse 的 `.pt` 换成一个高斯更少的 `.pt`。

## 3. "默认参数 = 原版行为"的证据

对照：

* **新代码 + 全默认参数**（`--poisson_depth_bg same --depth_estimate_source centers --max_poisson_depth 10 --extract_seed -1`，q=0.1）
  → `outputs/metrics/geometry_m_base_d10_q01.json`（run `m_base_d10_q01`，A100）
* **hopper 原版代码**（未打任何 patch）
  → `repo/sugar-dnc-zju/results/metrics/geometry_coarse_base_seed0.json`（run `coarse_base_seed0`，H200）

| 指标 | 新代码默认 (A100) | 原版代码 (hopper) | 相对差 |
|---|---|---|---|
| `cameras_extent` | 5.847915220 | 5.847914696 | +9e-8 |
| G1 `median_abs` | 0.004049294 | 0.004019149 | **+0.75%** |
| G1 `mean_abs` | 0.010347211 | 0.010349586 | −0.02% |
| G1 `p90_abs` | 0.019349296 | 0.019379026 | −0.15% |
| G2 `ratio_below_1pct` | 0.9773011 | 0.9773579 | −0.006% |
| G2 `ratio_below_0p5pct` | 0.9414757 | 0.9413394 | +0.014% |
| G4 `n_faces` | 370284 | 370128 | **+0.042%** |
| G4 `n_vertices` | 205283 | 205548 | −0.13% |
| G4 `n_connected_components` | 2184 | 2233 | **−2.2%** |
| G4 `n_fragment_components_lt_100` | 2094 | 2137 | **−2.0%** |
| G4 `largest_component_face_ratio` | 0.42303 | 0.41686 | +1.5% |
| G4 `n_boundary_edges` | 38450 | 38596 | −0.38% |
| G4 `n_non_manifold_edges` | 0 | 0 | 0 |
| G5 `dihedral_abs_deg_mean` | 33.5916 | 33.6019 | −0.03% |
| G3 `mean_abs_fg_vertices` | 0.106508 | 0.106660 | −0.14% |

**结论**：全部连续几何指标（G1/G2/G3/G5）差异 ≤0.75%，拓扑计数差异 ≤2.2%。

**噪声基准**：提取过程本身含随机性（`SuGaR.sample_points_in_gaussians` 用未 seed 的 `torch.randperm` 子采样表面点，marching cubes 的候选点集因此逐次不同）。本机同参数重复提取的实测重复差为 **0.4%**（面数量级），碎片数这类整数计数对该随机性最敏感。**注意 2184 vs 2137 的碎片数差异（−2.0%）说明 A100 侧是自己重跑的、独立的 baseline，不是复用 hopper 的模型或指标**——若复用会逐位一致。

**限制**：本对照是"新代码默认 vs 原版代码"的**跨机器**对照（A100 vs H200，cuDNN/cuBLAS 归约顺序不同），不是同机逐位对照；因此上表只能证明"默认参数不改变行为，差异落在提取随机性 + 硬件浮点差范围内"，**不能**证明逐位一致。
