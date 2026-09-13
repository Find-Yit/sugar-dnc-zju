# M1A_DEPTH_STATS.md — M1+a 自适应 Poisson 深度的实际取值与中间量（清单第 7 项）

> 数字来自 `results_a100/metrics/depth_report_centers.json`（原 `outputs/m1/report_depth/extract_stats.json`）
> 与 `depth_report_surface.json`（原 `outputs/m1/report_depth_surface/extract_stats.json`），
> 由 `extract_mesh.py --only_report_depth` 生成（**只估深度、不跑 Poisson**）。
> 各提取 run 实际生效值另见 `extract_stats_<TAG>.json` 的 `poisson_depth_fg_used` / `poisson_depth_bg_used`。

## 0. 结论速览

| 来源 | D_fg | D_bg | 说明 |
|---|---|---|---|
| **原版 SuGaR** | 10 | 10（无此概念，单一深度） | 硬编码默认 `--poisson_depth 10` |
| **M1（Frosting 原样 auto）** | **10** | — | raw 10.873 → 被 `max_poisson_depth=10` 截断 → **与原版相同，零收益** |
| **M1+a（centers，本次采用）** | **10** | **9** | 背景 raw **9.9907** → floor → **9**。这是 M1+a 唯一的有效增量 |
| **M1+b（surface，表面点估密度）** | 10 | 10 | raw 14.925 / 13.167 **双双被 10 截断 → 零区分度、零收益** |

公共量（两份报告一致）：`cell_size_nn_distance_ratio = 100.0`、`max_poisson_depth = 10`、
`quantile_to_use = 0.1`、`opacity_threshold = 0.5`、`fg_bbox_factor = 1.0`、`bg_bbox_factor = 4.0`、
`cameras_spatial_extent = 5.847915220260621`、`camera_average_xyz = [0.08620, -0.00718, 0.15170]`、
`n_gaussians_total = 429423`。

> 深度公式（逐字移植自 Frosting `frosting_extractors/coarse_shell.py` L17-49 的
> `compute_optimal_poisson_depth`）：`raw_depth = log2(bbox_size / (quantile_dist × ratio⁻¹…))` 形式，
> 最终 `poisson_depth = min(floor(raw_depth), max_poisson_depth)`。
> **注意**：`quantile_dist_normalized_SQUARED` 是 pytorch3d `knn_points` 返回的**平方距离**，
> Frosting 原实现未开方，本次**逐字保留**该行为（JSON 内 `nn_dist_note` 已标注），
> 以保证与上游"原样移植"的可比性。

## 1. 来源 A：`centers`（高斯中心点）— `depth_report_centers.json`

### 1.1 前景 D_fg（`depth_fg_centers` = **10**，等同 `auto_poisson_depth`）
| 量 | 值 |
|---|---|
| `n_gaussians_fg`（fg bbox 内） | 112209 |
| `n_gaussians_opaque`（opacity > 0.5） | 282930 |
| **`n_gaussians_used`（实际参与 KNN）** | **77554** |
| `bbox_size` | **12.854769802093507** |
| `quantile_dist_normalized_SQUARED`（q=0.1） | **5.331572538125329e-06** |
| **`raw_depth_before_floor`** | **10.873151263385152** |
| `poisson_depth`（截断后） | **10** |

> raw 10.873 > 上限 10 → 截断。**这就是「Frosting 原样 auto D 在 Truck 上零收益」的数字依据**：
> auto 估出来的就是原版硬编码的 10。

### 1.2 背景 D_bg（`depth_bg_centers` = **9**）← M1+a 的增量
| 量 | 值 |
|---|---|
| `n_gaussians_bg_bbox`（bg bbox 内，bbox_factor 4.0） | 209540 |
| `n_points_total` / **`n_points_used_for_knn`** | 136798 / **136798**（< `max_points_for_knn` 500000，未抽样） |
| `bbox_size` | **51.461238098144534**（≈ 前景 bbox 的 4 倍） |
| `quantile_dist_normalized_SQUARED`（q=0.1） | **9.828669135458767e-06**（≈ 前景的 1.84 倍 → 背景更稀疏） |
| **`raw_depth_before_floor`** | **9.99071629987679** |
| `poisson_depth` | **9** |

> **关键**：背景 raw = **9.9907**，**差 0.009 就会取整到 10**。
> 即 D_bg=9 这个增益本身贴着阈值，属"窄幅但可复现"的结论（同一 .pt 多次运行该值确定）。

## 2. 来源 B：`surface`（高斯表面采样点）— `depth_report_surface.json`（= M1+b）

| 量 | fg_surface | bg_surface |
|---|---|---|
| `n_points_total` | **7449788** | **1932190** |
| `n_points_used_for_knn` | 500000（命中 `max_points_for_knn` 上限，**已抽样**） | 500000（同，已抽样） |
| `bbox_size` | 12.863427543640137 | 51.46157379150391 |
| `quantile_dist_normalized_SQUARED` | **3.2150984452528064e-07** | **1.087120267584396e-06** |
| **`raw_depth_before_floor`** | **14.924777561361475** | **13.167200825707134** |
| `poisson_depth` | **10**（截断） | **10**（截断） |

> **M1+b 零收益的数字依据**：表面点比中心点密约 60–100 倍（fg 平方距离 5.33e-6 → 3.22e-7），
> raw 深度被抬到 14.92 / 13.17，**两者都远超上限 10，全部被截断成 10**，
> 前景与背景不再有任何区分度（对比 centers 口径的 10 / 9）。
> 因此 `depth_report_surface.json` 里 `poisson_depth_bg_used = 10`、`depth_fg_surface = depth_bg_surface = 10`。
> 只有在**比 Truck 稀疏得多**的场景（raw < 10）才可能起作用 —— 本次记为负结果，未用于任何提取 run。
> （该报告的 `extract_seed = 0`；centers 报告为 `-1`，只影响表面采样，不影响 centers 口径的估计值。）

## 3. 剪枝后 D_bg 回落：**bg raw 9.99 → 10.08** 的现象

- 在**未剪枝**的 `coarse_base_seed0/15000.pt` 上：bg raw = **9.99072** → D_bg = **9**。
- 在 **M2-B（p95）剪枝后**的 `pruned_p95/15000_pruned.pt` 上重新 auto 估计：
  bg raw ≈ **10.08** → floor 后 = **10**，即**退回原版深度**。
- 机制：DBSCAN 剪枝删掉的正是背景里孤立的小簇与噪声点（背景剪 16.68%），
  剩下的背景点**分布更紧凑**，`quantile_dist` 变小 → raw 深度上升，
  刚好跨过 10 的整数边界（原本只差 0.009）。
- 后果：`m2p95_m1a_q0`（M2-B + M1+a **auto**）实际生效 D_fg/D_bg = **10/10**，
  与 `m2p95_q0` 几乎逐位相同（连通分量 1435 vs 1434、碎片 1414 vs 1412、面数 325550 vs 325562）
  —— 即 **auto 路径下两项改进"看起来不可叠加"**。
- 解法与证据：另跑 `m2p95_bg9_q0`，**显式 `--poisson_depth_bg 9`** 绕开 auto 估计，
  得到全表最优：碎片 **1194**（比 q0 基线 1628 再 −26.6%）、连通分量 **1216**、
  边界边 **11225**（最低）、耗时 **240.2 s**（最快）。
  → 结论应表述为"**两项改进可叠加，但 auto 深度估计与剪枝相互干扰，需固定 D_bg**"。

## 4. 各提取 run 实际生效的 D_fg / D_bg（`summary.csv`）

| TAG | D_fg | D_bg | 来源 |
|---|---|---|---|
| `base_d10_q0` / `base_d10_q01` | 10 | 10 | 显式 `--poisson_depth 10`（原版默认） |
| `frosting_q0_s0` | 10 | 10 | `--poisson_depth auto`（raw 10.873 截断） |
| `m1a_q0` / `m1a_q01` | 10 | **9** | `--poisson_depth auto --poisson_depth_bg auto --depth_estimate_source centers` |
| `m2p95_q0` / `m2largest_q0` | 10 | 10 | 显式 10 |
| `m2p95_m1a_q0` | 10 | 10 | auto，但剪枝后 raw 回到 10.08 |
| **`m2p95_bg9_q0`** | 10 | **9** | 显式 `--poisson_depth_bg 9` |
