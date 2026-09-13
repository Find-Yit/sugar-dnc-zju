# 第二轮调研：直接建立在 SuGaR 之上的改进工作（可移植清单）

> 调研时间：2026-09-13，限时 20 分钟。工具：GitHub REST API + WebSearch/WebFetch + 本地 `repo/SuGaR` 源码比对。
> 标注：**【确认】**＝有原文/源码/API 返回直接佐证；**【推测】**＝我的推断；**【未找到】**＝检索未拿到。
> 本文件不重复第一轮 `SURVEY_sugar_improvements.md` 已确认的内容（2DGS/GOF/PGSR/Gaussian Surfels 等的定位与数字请看第一轮）。
> 本文件只做调研，**不含任何代码改动**。

---

## 0. TL;DR（三条最重要的结论）

1. **【确认】GitHub 上不存在"对 SuGaR 做算法改进"的 fork。** 我把 Anttwo/SuGaR 的 100 个 fork 全拉了一遍，并对 star 最多的几个逐个做了 `compare` API diff：最多的一个（`TonyDua/SuGaR-Windows`）只 ahead 2 个 commit 且**只改了 README.md**；其余要么 ahead=0，要么是 Windows 移植/课程作业/应用封装。**这条路是死路，不要再花时间找 fork。**
2. **【确认】Frosting 的"自动选择那个关键超参"= Poisson 八叉树深度 `poisson_depth`，我拿到了它的完整实现源码（`compute_optimal_poisson_depth`，见 §2.1）。** 而且我做了一次关键比对：**Frosting 的 coarse 训练器与 SuGaR 的 coarse 训练器逐字节完全相同（`diff` 输出 0 行）**。也就是说 **Frosting 相对 SuGaR 的几何改进 100% 落在 mesh 提取端，训练端一个字都没改**。这使得该改动成为**风险最低、可信度最高的移植候选**：40 行函数、依赖全是 SuGaR 已有的 API、无需重训。
3. **【确认】2D-SuGaR 是开源的**（第一轮标为"未找到"，本轮找到了：`prajwalcr/2d-sugar`，11 star）。我读到了它的 **DBSCAN 聚类剪枝** 完整源码（§2.3），这是一条直接针对用户"mesh 连通分量/碎片数"指标的、约 30 行的可移植改动。

---

## 1. GitHub 侧：SuGaR 的 fork / 衍生仓库全面核查【确认】

### 1.1 Anttwo/SuGaR 的 fork（API：`/repos/Anttwo/SuGaR/forks?sort=stargazers`，返回 100 条）

有任何 star 或 fork 的全部列出如下，**无一例外都不是算法改进**：

| 仓库 | star | 性质 | `compare` API 结果（vs Anttwo:main）|
|---|---|---|---|
| `TonyDua/SuGaR-Windows` | 6 | Windows 移植 | **ahead_by 2，改动文件只有 `README.md`**【确认】 |
| `Egdon/SuGaR` | 2 | 2025 国科大计算机图形学课程作业 | 未查 |
| `latay/SuGaR-GS-to-Mesh` | 1 | 应用封装 | **ahead_by 0**（纯镜像，落后 13 个 commit）【确认】 |
| `heartxkl/SuGaR` | 1 | 镜像 | **ahead_by 0**【确认】 |
| `widyaanggara/SuGaR` | 1 | 镜像 | **ahead_by 0**【确认】 |
| `alestrami/SuGaR` | 0 | Objaverse 批量训练脚本 | ahead_by 1，改了 `train_all_objaverse_sugar.py` / `coarse_mesh.py` / `dataset_readers.py` 等 —— 属**数据集适配**，非算法改进【确认，未逐行读 diff 内容 →"非算法改进"属**【推测】**】 |
| `Monstermeister/SuGaR`, `BAoD1nH/SuGaR`, `numix74/SuGaR`, `urbste/SuGaR`, `varunagrawal/SuGaR`, `TowerCat/SuGaR`, `Taylor-Hunter/CS_535_...`, `MinHius/SuGaR_KLTN` | 各 1 | 课程/个人实验 | 未逐个查 |

### 1.2 关键词搜索（`SuGaR gaussian splatting mesh` / `built upon SuGaR` / `based on SuGaR` / `extends SuGaR` / `SuGaR improvement`）

| 仓库 | star | 是否算法改动 | 说明 |
|---|---|---|---|
| `prajwalcr/2d-sugar` | **11** | **✅ 是** | **[Eurographics 2026] 2D-SuGaR 官方实现。见 §2.3。**【确认】 |
| `Divyam10/2D-SuGaR` | 7 | 同上（同一项目的另一份/镜像）| 【确认存在，未细查关系】 |
| `antibloch/improved_SuGaR` | 5 | **❌ 否** | README 明确：只做**输入图像预处理**（DiffPIR/NAFNet 去模糊 + SPSR 超分）再喂给 COLMAP+GS，**SuGaR 本体代码未改**，无对比数字【确认】 |
| `zhumeng-bit/sugar-gaussian-to-mesh`、`4thDoor-JB/sugar-pipeline-wrapper`、`TEXTaiLES/SAMplify_SuGaR`、`ZachariVaia/NEPHELE` 等 | 0–3 | ❌ 否 | 全是流水线封装 / SAM 抠背景 / Windows 移植 |

> 注：`built upon SuGaR`、`extends SuGaR` 等英文短语搜 GitHub 全是噪声（"syntactic sugar" 语法糖）。**【确认】此路不通。**

**结论【确认】：真正"直接在 SuGaR 上改"的开源工作只有两类——(a) 同作者的 Frosting / MILo；(b) 第三方的 2D-SuGaR。下面逐个拆。**

---

## 2. 三个核心候选的源码级拆解

### 2.1 ⭐ Frosting 的 `compute_optimal_poisson_depth` —— 最高价值、最低风险

**来源【确认】**：`https://raw.githubusercontent.com/Anttwo/Frosting/main/frosting_extractors/coarse_shell.py`，第 17–49 行。
（第一轮没找到这个函数，是因为找错了文件名：Frosting 的提取器叫 `coarse_shell.py` 而不是 `coarse_mesh.py`。）

**论文侧佐证【确认】**（ECCV 2024 补充材料 §7 "Improving surface reconstruction"，`https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08270-supp.pdf`）：
> Poisson 重建把场景离散成 2^D × 2^D × 2^D 个格子；**SuGaR 默认对任何场景都用很大的 D=10**。但若分辨率相对于场景几何复杂度/细节尺度过高，**高斯的形状会以椭球疙瘩的形式显现在表面上，且 D 过大时几何上会出现孔洞**。Frosting 自动估计一个合适的 D。

**完整实现（逐字抄录）【确认】**：
```python
def compute_optimal_poisson_depth(
    sugar:SuGaR,
    cell_size_nn_distance_ratio:float=100,
    max_poisson_depth:int=10,
    fg_bbox_factor=1.,
    opacity_threshold=0.5,
    quantile_to_use=0.1,
    ):

    # Compute foreground bbox
    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_bbox_min_tensor = _camera_average_xyz - fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_bbox_max_tensor = _camera_average_xyz + fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_mask = (sugar.points > fg_bbox_min_tensor).all(dim=-1) * (sugar.points < fg_bbox_max_tensor).all(dim=-1)

    # Remove transparent gaussians
    opacity_mask = sugar.strengths[..., 0] > opacity_threshold
    mask = opacity_mask * fg_mask

    # Compute poisson bbox
    bbox_size = 1.1 * (sugar.points[mask].max(dim=0)[0] - sugar.points[mask].min(dim=0)[0]).max().item()

    # Compute nearest neighbor distances and normalize by bbox size
    nn_dists = knn_points(sugar.points[mask][None], sugar.points[mask][None], K=2).dists[0, ..., 1]
    norm_nn_dists = nn_dists / bbox_size
    quantile_dist = norm_nn_dists.quantile(quantile_to_use).item()

    # Compute optimal poisson depth
    poisson_depth = -np.log2(cell_size_nn_distance_ratio * quantile_dist)
    poisson_depth = np.floor(poisson_depth).astype(int)
    poisson_depth = min(poisson_depth, max_poisson_depth)

    return poisson_depth
```

**原理（一句话）**：取前景、不透明高斯的**最近邻距离的 10% 分位数**（用包围盒尺寸归一化），令 Poisson 的格子边长约等于 `1/100` 倍该距离，反解出 `D = floor(-log2(100 · d_10%))`，再截断到 ≤10。**即：高斯越密 → D 越大；高斯越稀 → D 自动调小，从而避免椭球疙瘩与孔洞。**

> ⚠️ 注意 `nn_dists` 是 pytorch3d `knn_points` 的返回值，**是平方距离**（pytorch3d 约定）。Frosting 原代码直接拿来用了，没开方。**【推测】** 这可能是作者有意为之（常数 100 是配套调出来的），移植时**照抄即可，不要"修正"成开方**，否则常数就不对了。

**调用处【确认】**（`coarse_shell.py` 第 234–240 行）：
```python
if poisson_depth == -1:
    CONSOLE.print("Computing optimal poisson depth...")
    poisson_depth = compute_optimal_poisson_depth(sugar)
    poisson_depth_str = 'auto'
    CONSOLE.print("Optimal poisson depth:", poisson_depth)
```
命令行【确认，`Frosting/train_full_pipeline.py` 第 31–32 行】：
`--poisson_depth`，`type=int, default=-1`，help 写 `"If -1, will compute automatically the depth based on the SuGaR model."`

#### 移植可行性：**极高**【确认，已做 API 核对】

我逐项核对了本地 `repo/SuGaR`，该函数依赖的东西**全部现成**：

| 依赖 | 本地 SuGaR 是否有 | 证据 |
|---|---|---|
| `from pytorch3d.ops import knn_points` | ✅ **已经 import 了** | `sugar_extractors/coarse_mesh.py` 第 6 行【确认】 |
| `import numpy as np` / `torch` / `SuGaR` | ✅ 已 import | 同文件第 2/4/8 行【确认】 |
| `sugar.get_cameras_spatial_extent(return_average_xyz=True)` | ✅ **签名完全一致** | `sugar_scene/sugar_model.py:853`：`def get_cameras_spatial_extent(self, nerf_cameras=None, return_average_xyz=False)`【确认】 |
| `sugar.points` | ✅ | `sugar_model.py:408` 是 `@property`【确认】 |
| `sugar.strengths` | ✅ | `sugar_model.py:425` 是 `@property`【确认】 |
| 被替换的目标变量 | ✅ | `sugar_extractors/coarse_mesh.py` 第 42 行 `poisson_depth = 10`、第 43 行 `vertices_density_quantile = 0.1`【确认，本轮再次亲眼读到】 |

→ **可以把这 40 行原样粘进 `sugar_extractors/coarse_mesh.py`，一行都不用改**（连 import 都不用加）。预计落地时间 **15 分钟**。

#### 一个额外的重要发现：Frosting 训练端对 SuGaR **零改动**【确认】

我把 `Frosting/frosting_trainers/coarse_density_and_dn_consistency.py` 下载下来，只做命名空间替换（`frosting_scene`→`sugar_scene` 等），再和本地 `repo/SuGaR/sugar_trainers/coarse_density_and_dn_consistency.py` 做 `diff`：

```
diff 输出行数 = 0
```

**两份文件逐字节相同**，包括 `num_iterations=15_000`、`start_dn_consistency_from=9000`、`dn_consistency_factor=0.05`、`start_entropy_regularization_from=7000` / `end=9000` / `factor=0.1`、`regularize_from=7000`、`prune_low_opacity_gaussians_at=[9000]`、`prune_hard_opacity_threshold=0.5`、densify 的全套阈值。

> **这条结论的分量**：Frosting 是 ECCV 2024 Oral、同一批作者、在 SuGaR 之后一年做的工作，他们**重新审视过 coarse 训练超参后决定一个都不改**。这说明：
> - **【推测，但证据很强】** 第一轮候选清单里的 ①（把 DNC 调回官方 λ=0.05 / start=9000）是正确方向——作者自己复用了同一套值；想靠再调 coarse 训练超参拿提升，空间很小。
> - **提升的空间被作者用行动指向了 mesh 提取端。** 这与用户的两个指标（COLMAP 点到 mesh 距离、连通分量/碎片数）恰好对齐。

**Frosting 论文里与 SuGaR 的具体对比数字（PSNR/SSIM/LPIPS 表）：【未找到】** —— arXiv HTML 版 404，ECVA 补充材料 PDF 抓回来是图像流无法解析，正文 PDF 超过 10MB 抓取上限。

---

### 2.2 MILo（Anttwo/MILo，SIGGRAPH Asia 2025 TOG，518 star）

**与 SuGaR 代码骨架的关系：【未找到】** —— GitHub API 在本轮末尾触发了限流，没能拉到文件树。**【推测】** 从同作者、同实验室（INRIA REVES）的惯例看，大概率仍是 `gaussian_splatting/` + 自有 scene/trainers/extractors 的同款骨架，但**未经验证，不要当事实用**。

**方法要点【确认，来自 themoonlight.io 的文献综述页】**：
- 每个训练 iteration 都**可微地**抽 mesh（顶点位置 + 连通性都可微），以高斯作为 Delaunay 三角化的 pivot，**每个被选中的高斯出 9 个点**（中心 + 8 个沿主轴的角点）。
- **挑选哪些高斯做 pivot，用的是 importance-weighted sampling，明确写着"adopted from Mini-Splatting2"**。
- 损失：体渲染侧 L1 + D-SSIM + 深度图法向一致性；mesh 侧 深度对齐 + 法向一致；另加 **Anti-Erosion Loss**（防细节侵蚀）与 **Interior Regularization**（消内部空腔）。
- 效率结论【确认】：T&T 上 **4.36M 顶点**，而同类方法 10M–30M+，少一个数量级。

**"能否只取其高斯剪枝/密度部分移植"**：
**【推测】** 理论上可以——"用 Mini-Splatting2 式重要性加权采样来挑高斯"是一个独立模块，可以放在 SuGaR 的 Poisson 采样之前当剪枝器用。但有两个实际障碍：(a) 需要额外引入 Mini-Splatting2 的重要性计算（依赖光栅化器输出每高斯的贡献/blending weight，SuGaR 用的原版 3DGS 光栅化器**默认不输出**，与第一轮 §3③ 遇到的是同一个障碍）；(b) MILo 的收益主要来自"mesh 在训练环里"，剥离后剩下的剪枝部分**是否还有增益无任何证据**。
→ **建议本次考核不碰。**

**MILo 论文里 DTU Chamfer / T&T F1 的具体数字：【未找到】** —— INRIA 的 PDF 直链 404，arXiv abs 页无表，alphaXiv 与 themoonlight 综述页都明说"未列出具体数值"。只拿到定性表述："achieves state-of-the-art performance among explicit methods on Tanks & Temples and DTU"。**不要在汇报里引用任何 MILo 的具体数字。**

---

### 2.3 ⭐ 2D-SuGaR 的 DBSCAN 聚类剪枝（`prajwalcr/2d-sugar`，11 star，Eurographics 2026 Short）

**代码基础【确认，README】**：它同时含 `gaussian_splatting/`（3DGS）、`gaussian_splatting_2d/`（2DGS）两套，**主体是在 2DGS 上改，但复用了 SuGaR 的 mesh refinement**（仓库里有 `extract_mesh.py`、`extract_refined_mesh_with_texture.py`、`blender/sugar_utils.py`，共 113 个 py 文件）。→ 与第一轮判断一致："名字叫 2D-SuGaR，但不是 SuGaR 的 fork"。

**暴露的超参【确认，README】**：`--initialization_prior monocular_depth_normal`、`--lambda_dist`（**default 1000**）、`--lambda_normal`、`--lambda_normal_prior`、`--depth_ratio`、`--cluster_prune_iterations`（例：7000）、`--dbscan_min_samples`、`--dbscan_knn_percentile`。

**聚类剪枝的完整实现【确认】**
`gaussian_splatting_2d/scene/gaussian_model.py:429` 起：
```python
def cluster_gaussians(self, save_path, opt_args):
    def estimate_eps(points, min_samples=6, percentile=90):
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(points)
        distances, _ = nbrs.kneighbors(points)
        k_distances = distances[:, -1]      # distance to the k-th neighbor
        eps = np.percentile(k_distances, percentile)
        return eps

    points = self._xyz.detach().cpu().numpy()
    min_samples = opt_args.dbscan_min_samples
    eps = estimate_eps(points, min_samples=min_samples, percentile=opt_args.dbscan_knn_percentile)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_ids = dbscan.fit_predict(points)
    ...
```
`gaussian_splatting_2d/train.py:140-151`：
```python
if iteration in opt.cluster_prune_iterations:
    cluster_sizes, cluster_ids = scene.save_clusters(iteration, opt)
    retained_cluster_id = max(cluster_sizes.items(), key=lambda item: item[1])[0]
    gaussians.prune_points(cluster_ids != retained_cluster_id)
# 注意：剪枝的那一个 iteration 跳过 densify
if iteration < opt.densify_until_iter and iteration not in opt.cluster_prune_iterations:
    ...
```

**一句话原理**：在 iteration 7000（densify 结束那一刻）对所有高斯中心跑一次 **DBSCAN**，`eps` 用"第 k 近邻距离的 90 分位数"自适应估计，然后**只保留最大的那个连通簇，其余簇 + 噪声点全部剪掉**。

**为什么它和用户的指标直接对口**：它剪掉的正是**空间上孤立的高斯团**——这些团在 Poisson 之后就变成用户要统计的"**碎片 / 额外连通分量**"。这是所有候选里**唯一一个直接、显式优化"连通分量数"这个指标**的方法。

**依赖**：`sklearn.cluster.DBSCAN` + `sklearn.neighbors.NearestNeighbors`。**【推测，需 1 分钟验证】** SuGaR 环境里 `scikit-learn` 大概率已随 open3d/其他包装上；若没有，`pip install scikit-learn` 对 torch 无风险（但按项目红线仍要挂 `PIP_CONSTRAINT`）。

**报告的数字**：README 只说"achieves state-of-the-art results in mesh reconstruction"（DTU），**具体数值 vs SuGaR/2DGS：【未找到】**。

---

## 3. 其他被点名工作的核查结果（快速）

| 工作 | 是否"直接建立在 SuGaR 上" | 开源 | 本次是否推荐 |
|---|---|---|---|
| **Gaussian Frosting** | **✅ 是**（同作者，代码骨架同源，coarse 训练器逐字节相同）| ✅ `Anttwo/Frosting` | **强烈推荐，见 §2.1** |
| **MILo** | ✅ 同作者，但重写了训练主循环 | ✅ `Anttwo/MILo`（518★）| ❌ 太重，见 §2.2 |
| **2D-SuGaR** | ⚠️ 在 2DGS 上改，借用 SuGaR refinement | ✅ `prajwalcr/2d-sugar`（11★）| **推荐取其 DBSCAN 剪枝，见 §2.3** |
| **ARGS** | ✅ 论文明确 "builds upon SuGaR" | **【未找到】开源仓库**（本轮与第一轮均未检索到）| ❌ 无代码，含 neural SDF，超时 |
| `antibloch/improved_SuGaR` | ❌ 只做输入图像预处理 | ✅（5★）| ❌ 无算法改动、无数字 |
| **GaMeS / MeshSplats / Mani-GS / SplatMesh / Textured Gaussians** | **【未找到】** —— 本轮 20 分钟预算内未能逐个核查，不排除其中有以 SuGaR 为 baseline 者 | — | 未评估 |

> 诚实说明：用户点名的 GaMeS、MeshSplats、Mani-GS、SplatMesh、Textured Gaussians、"Gaussian-Mesh hybrid" 这几项，**本轮没有时间逐个核查，一律标【未找到】，我不对它们做任何断言**。若需要，可以再开一轮专门查这 5 篇。

---

## 4. 可直接移植进 SuGaR 的候选清单（按 证据强度 × 移植难度 排序）

### 🥇 候选 1：Frosting 的 Poisson 深度自动选择（`compute_optimal_poisson_depth`）

| 项 | 内容 |
|---|---|
| **来源** | `Anttwo/Frosting` → `frosting_extractors/coarse_shell.py:17-49`【确认，源码已逐字抄录在 §2.1】；ECCV'24 补充材料 §7【确认】 |
| **改动落点** | **mesh 提取（不用重训！）** |
| **要改的 SuGaR 文件** | `sugar_extractors/coarse_mesh.py` —— **【确认】** 粘函数到文件头部（第 13 行 `def extract_mesh_from_coarse_sugar` 之前）；**【确认】** 改第 42 行 `poisson_depth = 10`；**【确认】** 依赖的 `knn_points` 在第 6 行已 import。调用点插在 SuGaR 对象构造完成之后（**【推测】** 对应 Frosting 的第 234 行位置，SuGaR 里需自行定位 sugar 实例建好的那一行）|
| **关键超参与推荐扫值** | `cell_size_nn_distance_ratio`（默认 **100**，主控旋钮；扫 **50 / 100 / 200** → D 分别 +1 / 基准 / −1）；`quantile_to_use`（默认 0.1）；`opacity_threshold`（默认 0.5）；`max_poisson_depth`（默认 10）。**建议：先跑一次只打印自动算出的 D**，与当前硬编码的 10 对比，若结果就是 10 则该场景无收益，立刻换候选 2/3 |
| **作者报告的收益** | 定性【确认】：避免"高斯形状显现为表面上的椭球疙瘩"与"D 过大导致的孔洞"。**定量数字【未找到】** |
| **风险** | 低。①若自动 D 恰好 = 10 则零收益（**先打印再决定**）；②`knn_points` 返回平方距离，**照抄勿改**；③在 10M 点上跑 KNN 有显存开销，但代码已先用 fg_bbox + opacity>0.5 过滤，**【推测】** H200 上无压力 |

### 🥈 候选 2：2D-SuGaR 的 DBSCAN 最大簇剪枝（直接打"碎片数"指标）

| 项 | 内容 |
|---|---|
| **来源** | `prajwalcr/2d-sugar` → `gaussian_splatting_2d/scene/gaussian_model.py:429` + `train.py:140`【确认，源码已抄录在 §2.3】 |
| **改动落点** | 二选一：**(A) 提取前**（推荐，不用重训）在 `sugar_extractors/coarse_mesh.py` 里对 `sugar.points` 跑一次 DBSCAN，只留最大簇再做表面采样；**(B) 训练中** 在 coarse 训练 iteration 7000（densify 结束点）剪 |
| **要改的 SuGaR 文件** | (A) `sugar_extractors/coarse_mesh.py`（**【推测】** 要插在表面点采样之前，具体行号需现场定位）；(B) `sugar_trainers/coarse_density_and_dn_consistency.py`（**【确认】** 其 densify 区间是 500→7000、`prune_low_opacity_gaussians_at=[9000]`，7000 这个点天然合适）|
| **关键超参与推荐扫值** | `dbscan_min_samples`（原默认 **6**，扫 6/10/20）；`dbscan_knn_percentile`（原默认 **90**，扫 80/90/95 —— 越大 eps 越大、越不容易把主体切碎）。**必看诊断量**：剪掉的高斯比例，**【推测】** 目标 2–15%，**若 >30% 说明 eps 太小把主体劈开了，必须回退** |
| **作者报告的收益** | "SOTA mesh reconstruction on DTU"（定性）【确认】；**相对 SuGaR 的具体数字【未找到】** |
| **风险** | 中。①DBSCAN 在百万级点上是 O(n log n)~O(n²)，**【推测】** 需先对高斯下采样或只在前景 bbox 内跑，否则可能很慢；②"只留最大簇"对**带大背景的场景（如 Truck）是危险的**——背景可能本身就是独立簇，会被整片剪掉。**【推测】 建议改成"保留所有 size ≥ 某阈值的簇"而非只留最大的一个**，更稳 |

### 🥉 候选 3：把 `vertices_density_quantile` 与自动 D 联动调（低垂果实，与候选 1 同一次提取里做完）

| 项 | 内容 |
|---|---|
| **来源** | SuGaR 官方 README Troubleshooting【第一轮已确认】+ Frosting 把它升格为命令行参数 `--cleaning_quantile`【确认，`train_full_pipeline.py:33`】 |
| **改动落点** | mesh 提取 |
| **要改的 SuGaR 文件** | `sugar_extractors/coarse_mesh.py` 第 **43** 行 `vertices_density_quantile = 0.1`【确认行号】 |
| **关键超参与推荐扫值** | 扫 **0.0 / 0.05 / 0.1**。注意 Frosting 源码注释【确认】：`"0.1 for most real scenes. 0. works well for most synthetic scenes"` |
| **收益** | 直接影响孔洞 vs 碎片的权衡；**与候选 1 是同一次提取，几乎零额外成本**——建议做 D(auto) × quantile{0, 0.05, 0.1} 的小网格 |
| **风险** | 极低。**注意**：它和候选 1 会互相影响（D 变小后原来的 0.1 清洗可能过激），**必须联合扫，不要各自单独调** |

### 候选 4（备选，证据弱）：把候选 1 的思路推广到 `surface_level`
第一轮已列（默认扫 `[0.1,0.3,0.5]`）。**【推测】** 与候选 1 正交，同一次提取里顺手扫，成本几乎为零，但**没有任何论文证据支持某个特定值更好**，纯经验调参。

### ❌ 明确不推荐（与第一轮结论一致，本轮新增证据）
- **MILo 的 mesh-in-the-loop / 重要性采样**：需要光栅化器输出逐高斯贡献，SuGaR 的原版 3DGS 光栅化器不输出 → 要改 CUDA。
- **再去调 coarse 训练超参**：**【本轮新证据】** Frosting 作者复查后逐字节沿用了 SuGaR 的全套 coarse 超参，暗示该处已近局部最优。
- **ARGS 的 neural SDF**：无开源代码【未找到】。

---

## 5. 事实 / 推测 分界速查

**【确认】**（有 API 返回、源码原文或论文原文）
- Anttwo/SuGaR 的 100 个 fork 中无算法改进；最活跃的 `TonyDua/SuGaR-Windows` ahead 2 commit 且只改 README.md。
- Frosting 的"关键超参"= `poisson_depth`；`--poisson_depth default=-1` 表示自动；实现为 `compute_optimal_poisson_depth`，公式 `D = min(floor(-log2(100 · quantile_0.1(nn_dist/bbox_size))), 10)`。
- **Frosting 的 `coarse_density_and_dn_consistency.py` 与 SuGaR 同名文件 diff = 0 行（逐字节相同）。**
- SuGaR 本地源码：`coarse_mesh.py:6` 已 import `knn_points`；`:42` `poisson_depth=10`；`:43` `vertices_density_quantile=0.1`；`sugar_model.py:853/408/425` 提供 `get_cameras_spatial_extent(return_average_xyz=)` / `points` / `strengths`。
- 2D-SuGaR 开源于 `prajwalcr/2d-sugar`；DBSCAN 剪枝在 iter 7000、eps=第 k 近邻距离的 90 分位、只保留最大簇；`--lambda_dist` default 1000。
- `antibloch/improved_SuGaR` 只做图像预处理，未改 SuGaR 算法。
- MILo 的 pivot 采样"adopted from Mini-Splatting2"，每高斯出 9 点；T&T 4.36M 顶点 vs 他法 10M–30M+。

**【推测】**（我的推断，未验证）
- `alestrami/SuGaR` 的改动属数据集适配而非算法改进（未逐行读 diff）。
- MILo 仓库与 SuGaR 骨架同源。
- 候选清单里所有"要改哪一行/插在哪"中**未标"确认行号"**者；所有推荐扫值区间；DBSCAN 的性能与"改成保留多簇更稳"的建议。
- `knn_points` 返回平方距离故不应"修正"为开方。

**【未找到】**
- Frosting 论文中与 SuGaR 的 PSNR/SSIM/LPIPS 对比表（arXiv HTML 404；ECVA 补充 PDF 为图像流；正文 PDF 超 10MB 抓取上限）。
- MILo 的 DTU Chamfer / T&T F1 具体数值（三个来源均未列出）。
- 2D-SuGaR 相对 SuGaR/2DGS 的具体数值。
- ARGS 的开源仓库。
- GaMeS / MeshSplats / Mani-GS / SplatMesh / Textured Gaussians 的核查（本轮时间不足，未做，不做任何断言）。
