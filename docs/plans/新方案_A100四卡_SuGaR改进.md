# 新方案 v2.1：移植已开源的 SuGaR 后续改进 + 自有改动（供 4×A100-40GB 团队制定完整计划）

> 撰写：Claude Fable 5.1（hopper 侧主会话），2026-09-13 09:45（v2.1：在 v2 的移植基础上加入自有改动 M1+/M2+，对照设计为「原版 SuGaR vs 移植原样 vs 我们的改进版」）
> 用途：交给另一台机器（4×A100 40GB）上的 Claude Fable，据此制定完整计划并由 opus 执行。
> 调研依据：`notes/SURVEY_sugar_improvements.md`（一轮）、`notes/SURVEY2_sugar_direct_followups.md`（二轮，含逐字抄录的源码与 API 核对）。
> 代码与脚本：GitHub https://github.com/Find-Yit/sugar-dnc-zju（hopper 侧 11:00 前 push；含评测脚本、复现脚本、两份调研、本文件）。

---

## 0. TL;DR

**做两个有开源出处、可直接移植进 SuGaR 主流程的改动，各自做超参扫描，目标是在 SuGaR 基线上拿到可复现的几何提升：**

| # | 改动 | 来源（开源） | 落点 | 是否重训 | 直接命中的指标 |
|---|---|---|---|---|---|
| **M1** | Poisson 深度 D 自动选择 + 顶点密度清洗分位数联动 | Gaussian Frosting（SuGaR 同作者，ECCV 2024 Oral）`frosting_extractors/coarse_shell.py:17-49` `compute_optimal_poisson_depth` | **mesh 提取主流程**（Poisson 重建的核心参数） | 否 | 椭球疙瘩、孔洞、碎片数、点到网格距离 |
| **M2** | DBSCAN 聚类剪枝：剪掉空间孤立的高斯簇 | 2D-SuGaR（Eurographics 2026 Short）`prajwalcr/2d-sugar` `gaussian_model.py:429` + `train.py:140` | **coarse 训练**（剪枝/密度策略，iteration 7000 剪一次）；或提取前剪 | 是（M2-B 不用） | 连通分量数、碎片数 |

- 为什么是这两个：二轮调研拉了 Anttwo/SuGaR 全部 100 个 fork，**没有任何 fork 做过算法改进**；真正基于 SuGaR 代码的后续只有同作者的 Frosting/MILo。Frosting 的 coarse 训练器与 SuGaR **逐字节相同**（作者复查一年后一处未改），把改进全放在了提取端，M1 就是其中最核心、40 行即可移植的部分。M2 是所有候选里唯一显式针对"连通分量数"的方法。
- hopper 侧已验证的负结果（不要重复踩）：在 coarse 训练里加 2DGS 式深度-法向一致性损失（DNC），λ=0.05 就让 PSNR −2.2 dB、几何变差；λ=0.2 场景坍缩；官方仓库其实已内置同类 `dn_consistency`。coarse 训练超参空间很小，**提升空间在提取端与剪枝端**。
- hopper 侧正在本地先跑 M1（Frosting 自动 D）的一组快速验证，结果会随仓库 push；A100 团队以完整网格扫描为目标。
- 学术诚信：两个改动都要在报告里注明来源仓库与文件行号，用自己的话解释输入输出与假设，超参扫描与分析是自己的工作。这是题目允许的（可查资料、可生成代码，需能解释），不允许的是不注明来源。

---

## 1. 可直接复用的资产与环境（hopper 侧已完成）

| 资产 | 位置（仓库内） | 说明 |
|---|---|---|
| 环境配方 | `README_ZH.md` | torch 2.4.1+cu121 + pytorch3d 0.7.8 官方 wheel（py310_cu121_pyt241）+ 源码编译 diff-gaussian-rasterization / simple-knn。**A100 是 sm_80：`TORCH_CUDA_ARCH_LIST="8.0"`**。绝不要裸 `pip install torch`（会装 cu13）。numpy 锁 <2。scikit-learn 已在依赖里（M2 需要）。 |
| 数据 | 3DGS 官方 `tandt_db.zip`（683 MB） | Truck 251 张 979×546，`--eval` 划分 219/32。zip 内还有 train / drjohnson / playroom 做泛化。 |
| 3DGS 7k | 自训 2 分钟 | `gaussian_splatting/train.py -s <truck> -m <out> --iterations 7000 --save_iterations 7000 --eval` |
| coarse SuGaR | `train_coarse_density.py`（含 `--seed`，其余新参数默认=原版）；`train_coarse_official_dnc.py`（官方 dn_consistency 包装，带 seed 与 train_stats） | 8000 iter，H200 12–14 min，A100-40GB 预计 20–25 min，峰值显存 <10 GB → **一张 A100 可并行 3 组** |
| mesh 提取 | `extract_mesh.py -l 0.3 -d 200000 --eval True`（hopper 侧正在加 `--poisson_depth auto|int --vertices_density_quantile --cell_size_nn_distance_ratio`） | `-d` 是前景/背景**各自**的三角形目标数；`-c` 必须以 `/` 结尾 |
| 评测 | `scripts/eval/run_eval_for_run.sh <run> <pt> <ply> <gpu>`；`scripts/eval/eval_geometry.py`（仅几何，提取端扫参用这个）；`summarize.py` | PSNR/SSIM/LPIPS；G1 稀疏点→mesh 距离；G2 精度比例；G4 拓扑（连通分量、<100 面碎片、非流形边）；G5 二面角。约 1 分钟/组 |
| 图表/报告 | `scripts/deliverables/make_figures.py`、`make_report.py` | 读 summary.csv 自动出图与 DOCX（需 python-docx / matplotlib / PyMuPDF 的独立 venv） |

**踩坑清单（务必写进新计划）**
1. **SuGaR 不保存中间 checkpoint**，作业超时时 14900/15000 的两组全白跑。新计划第一件事：给 coarse 训练加 `--save_every 2000` 与 `--resume`，或把每组控制在剩余 walltime 一半以内。
2. `extract_mesh.py -c` 缺尾斜杠 → `FileNotFoundError`。
3. **不要编辑正在运行的 bash 脚本**（hopper 侧因此丢了一次提取），改动写新文件。
4. LibreOffice headless 转中文文件名 PDF 需 ASCII 文件名 + `LC_ALL=C.UTF-8`。
5. cameras.json 内参是 COLMAP 原分辨率，SuGaR 包装器换算后 PSNR 与 3DGS 日志差 0.008 dB，正常。
6. DBSCAN 在百万级点上很慢：先按前景 bbox 与不透明度过滤，或对高斯下采样估计 eps。

---

## 2. M1：Frosting 自动 Poisson 深度（提取端，主推）

### 2.1 原理
Poisson 重建把场景离散成 2^D 立方格。SuGaR 对所有场景硬编码 D=10；Frosting 补充材料 §7 指出 D 相对高斯密度过高时，高斯形状会以"椭球疙瘩"显现在表面、并出现孔洞。Frosting 用前景、不透明（>0.5）高斯的最近邻距离 10% 分位数 d（按包围盒尺寸归一化）反解 `D = min(floor(-log2(100·d)), 10)`：高斯越稀 D 越小。

### 2.2 移植（逐字，40 行；二轮调研已核对 SuGaR 中所有依赖现成）
```python
# 来源：https://github.com/Anttwo/Frosting/blob/main/frosting_extractors/coarse_shell.py  L17-49
def compute_optimal_poisson_depth(sugar, cell_size_nn_distance_ratio=100, max_poisson_depth=10,
                                  fg_bbox_factor=1., opacity_threshold=0.5, quantile_to_use=0.1):
    _ext, _avg = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_min = _avg - fg_bbox_factor * _ext * torch.ones(1, 3, device=sugar.device)
    fg_max = _avg + fg_bbox_factor * _ext * torch.ones(1, 3, device=sugar.device)
    fg_mask = (sugar.points > fg_min).all(dim=-1) * (sugar.points < fg_max).all(dim=-1)
    mask = (sugar.strengths[..., 0] > opacity_threshold) * fg_mask
    bbox_size = 1.1 * (sugar.points[mask].max(dim=0)[0] - sugar.points[mask].min(dim=0)[0]).max().item()
    nn_dists = knn_points(sugar.points[mask][None], sugar.points[mask][None], K=2).dists[0, ..., 1]  # 注意：平方距离，照抄勿开方
    quantile_dist = (nn_dists / bbox_size).quantile(quantile_to_use).item()
    D = int(np.floor(-np.log2(cell_size_nn_distance_ratio * quantile_dist)))
    return min(D, max_poisson_depth)
```
- 放进 `sugar_extractors/coarse_mesh.py`（`knn_points` 第 6 行已 import）；把第 42 行 `poisson_depth = 10`、第 43 行 `vertices_density_quantile = 0.1` 改为命令行参数（`--poisson_depth auto|int`、`--vertices_density_quantile`、`--cell_size_nn_distance_ratio`），默认值保持原版行为。
- 把算出的 D、quantile_dist、bbox_size、过滤后高斯数写进 `<mesh_out>/extract_stats.json`。

### 2.3 实验设计
- **第一步只打印**：对 baseline coarse 模型算 auto D。若 D==10（与硬编码相同），该场景 M1 零收益，主力转 M2 与 quantile 扫描；若 D<10，继续。
- 网格（同一 coarse 模型，不重训）：`cell_size_nn_distance_ratio ∈ {50, 100, 200}`（对应 D±1）× `vertices_density_quantile ∈ {0, 0.05, 0.1}` = 9 组，加原版 D=10/q=0.1 作基线，共 10 组提取，每组 5–6 min，4 卡并行 ≈ 15 min。可再加 `surface_level ∈ {0.1, 0.3, 0.5}`。
- 假设：H-M1a 自动 D 降低碎片分量数与二面角均值（疙瘩减少）且 G1 距离中位数不升；H-M1b quantile 由 0.1 降到 0 减少孔洞（边界边数下降）但可能增加碎片，与 D 联动扫描才能找到最优。
- 泛化：在 train / playroom 场景重复最优配置 vs 原版。

### 2.4 自有改动 M1+（区别于 Frosting 原版，报告里作为"我们的贡献"）
Frosting 只用**前景**高斯中心估计一个全局 D，并把同一个 D 用于前景与背景两次 Poisson。hopper 侧观测：Truck 基线网格 2237 个连通分量中 2147 个碎片，绝大多数在背景；背景高斯密度比前景低一个量级以上，用前景的 D 去重建背景正是疙瘩与碎片的来源。
- **M1+a 前景/背景分离的自适应深度**：对 `coarse_mesh.py` 里已分离的 fg / bg 两组分别调用 `compute_optimal_poisson_depth`（bg 用 bg 高斯或 bg 表面点估计），得到 D_fg、D_bg 两个值；预期 D_bg < D_fg。改动点：Poisson 的两次调用各用各的 D。
- **M1+b 用真实的 Poisson 输入点云估密度**：Frosting 用高斯中心的最近邻距离，但 Poisson 的输入是采样出的表面点（约 10M 点，密度与高斯中心不同）。改为在表面点云上估计最近邻距离分位数（对 10M 点做子采样 50 万估计即可），使格子边长直接匹配输入点密度，这是 Poisson 分辨率选择更本源的依据。
- **M1+c 清洗分位数按分量自适应（可选）**：`vertices_density_quantile` 现在是全局分位数，会优先删掉背景低密度区域并制造孔洞。改为在每个连通分量内部按分位数清洗，或只对前景清洗。
- 对照设计：原版(D=10) / Frosting 原样(auto D) / M1+a / M1+a+b / M1+a+b+c，每组同一 coarse 模型，指标 G1、G2、碎片数、边界边数（孔洞代理）、G5。
- 假设 H-M1+：M1+a 相对 Frosting 原样进一步降低背景碎片数 ≥20%，且前景 G1 不变；M1+b 使不同场景（Truck/train/playroom）不需手调 `cell_size_nn_distance_ratio`。

---

## 3. M2：2D-SuGaR 的 DBSCAN 聚类剪枝（训练端剪枝策略，次推）

### 3.1 原理与源码（逐字抄录见 `notes/SURVEY2_sugar_direct_followups.md` §2.3）
在 densify 结束的 iteration 7000 对全部高斯中心跑一次 DBSCAN，`eps` = 第 k 近邻距离的 90 分位数（自适应），原版只保留最大簇、其余簇与噪声点全剪掉；剪枝当次迭代跳过 densify。依赖 `sklearn.cluster.DBSCAN`、`sklearn.neighbors.NearestNeighbors`。

### 3.2 移植
- **M2-A（训练中，进入训练主流程，推荐）**：在 `sugar_trainers/coarse_density.py`（或官方 `coarse_density_and_dn_consistency.py`）iteration 7000 处（SuGaR 从 3DGS 7000 起训，可放在 coarse 起点或 9000 硬剪枝同一处）加 `cluster_prune`：新增参数 `--cluster_prune_iter`、`--dbscan_min_samples`（默认 6）、`--dbscan_knn_percentile`（默认 90）、`--keep_cluster_min_size`。
- **M2-B（提取前，不重训）**：在 `coarse_mesh.py` 表面采样之前对 `sugar.points` 做同样聚类并屏蔽被剪高斯（用 mask 索引子集）。作为快速消融。
- **必须改的一点**：Truck 这类大背景场景"只留最大簇"会整片剪掉背景。改为"保留所有 size ≥ 阈值（如总数 0.5%）的簇"，只剪小簇与噪声点。记录剪掉比例，目标 2–15%，**>30% 即 eps 太小把主体劈开，回退**。
- 性能：对 43 万高斯（剪枝后）DBSCAN 可接受；对 185 万（7000 时）先用前景 bbox + 不透明度>0.1 过滤，或对 20 万子样本估 eps。

### 3.3 实验设计
- 网格：`min_samples ∈ {6, 10, 20}` × `knn_percentile ∈ {80, 90, 95}` = 9 组训练（A100 每卡 3 组并行，约 25 min 一批）+ baseline；每组提取用 M1 最优配置与原版各一次。
- 假设：H-M2 碎片分量数下降 ≥30%，PSNR 变化 ≤0.3 dB（剪掉的是低贡献孤立簇），G1 距离不升。
- 失败案例预期：细长结构（电线、杆）被当噪声剪掉；背景被误剪。

### 3.4 自有改动 M2+（区别于 2D-SuGaR 原版）
2D-SuGaR 的规则是"全局单一 eps + 只保留最大簇"，它是为物体中心的 DTU 设计的；对带背景的真实场景有两个明显缺陷：背景与前景密度差别大，单一 eps 要么把前景劈碎要么对背景无效；"只留最大簇"会删掉整个背景。
- **M2+a 贡献度加权的簇评分**：训练循环里每次迭代都有 `visibility_filter`（本次视角可见的高斯），累计每个高斯的"被看见次数" v_i；簇得分 S_c = Σ_{i∈c} opacity_i · v_i。剪枝规则改为"删除 S_c 低于总得分 τ_S（如 0.1%）的簇"，而不是按簇大小或只留最大簇。动机：一个由 200 个低不透明度、只被两三个视角看到的高斯组成的浮子簇，比一个 50 个高不透明度、被全部视角看到的小结构更该删。这是 TrimGS"贡献度剪枝"思想与 DBSCAN 的结合，原版两者都没有。
- **M2+b 前景/背景分区估计 eps**：在前景 bbox 内外分别用第 k 近邻距离分位数估计 eps_fg、eps_bg，各自聚类。动机同 M1+a。
- **M2+c 剪枝时机对齐 SuGaR 的正则时间表**：放在 9000（SuGaR 的硬不透明度剪枝点，此时高斯已从 185 万降到 43 万，DBSCAN 成本低一个量级），而不是 2D-SuGaR 的 7000；并在提取前再做一次（M2-B）作为消融。
- 对照设计：baseline / 2D-SuGaR 原样（只留最大簇，会删背景，作为失败案例展示）/ 保留大簇（size 阈值）/ M2+a / M2+a+b，各配 M1 最优提取；指标：碎片数、连通分量、G1、PSNR（剪枝不应伤渲染）、剪掉比例。
- 假设 H-M2+：M2+a 在碎片数下降幅度相同的情况下 PSNR 损失更小（删的是低贡献簇）；M2+b 避免背景整片丢失（背景边界边数不暴涨）。

---

## 4. 建议时间分配（6 小时，4×A100）
| 时段 | 内容 | 卡 |
|---|---|---|
| 0:00–0:40 | 环境、数据、3DGS 7k（Truck + train）、加 checkpoint 保存 | 2 |
| 0:40–1:05 | coarse baseline（density 与官方 dn_consistency 各 1，Truck）；同时 train 场景 coarse | 3 |
| 0:40–1:20 | 与训练并行：移植 M1 原样 + 实现 M1+a/b（含"只打印 D"检查）；M2 原样 + M2+a/b（CPU 单元自检：合成点云验证聚类、评分与保留规则；训练循环里加 visibility 计数） | CPU |
| 1:05–1:40 | M1 网格 10 组提取 + 几何评测（4 卡并行） | 4 |
| 1:40–2:40 | M2 训练 9 组 + baseline（每卡 3 组并行）→ 提取 → 评测 | 4 |
| 2:40–3:40 | 泛化场景（train / playroom）最优配置 vs 原版；失败案例可视化 | 4 |
| 3:40–4:40 | 补跑 / 敏感性 / 消融汇总 | 4 |
| 4:40–5:40 | 图表、报告、README、push | CPU |
| 5:40–6:00 | 缓冲 | — |

---

## 5. 验收标准
- 默认参数下的提取/训练结果与原版逐位一致（证明默认路径未变）。
- 每组 run 有 `train_stats.json` / `extract_stats.json`、日志、命令；所有数字只来自 `summary.csv`。
- 独立复算：随机抽 3 组 mesh 重算碎片数与 G1 中位数；抽 3 个视角从 PNG 重算 PSNR。
- 报告披露：M1/M2 的来源仓库与行号；自有改动 M1+/M2+ 单独成节并有对照证据；负结果如实写。
- 报告结构建议：问题（碎片主要在背景、前后景密度差异）→ 借鉴（Frosting、2D-SuGaR，注明来源）→ 我们的改动（分区自适应、贡献度评分）→ 证据（原版 / 移植原样 / 改进版三列对照）→ 结论与代价。

## 6. 不建议做的
- 换 2DGS / GOF / RaDe-GS 光栅化器、真正的 depth distortion（需改 CUDA）。
- MILo 的剪枝部分（需光栅化器输出逐高斯贡献）。
- 在 coarse 训练里再调 DNC / dn_consistency 权重（hopper 侧已证明是负结果，Frosting 作者也一处未改）。
