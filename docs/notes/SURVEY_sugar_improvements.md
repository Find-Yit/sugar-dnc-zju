# SuGaR 之后的 3DGS 表面/Mesh 提取工作调研（含深度-法向一致性损失稳定化）

> 调研时间：2026-09-13，限时 15 分钟，检索工具：WebSearch / WebFetch + 本地 `repo/SuGaR` 源码核查。
> 标注约定：**【确认】**＝从论文页面 / 官方仓库 / 本地源码直接读到的原文事实；**【推测】**＝我的推断，未经验证；**【未找到】**＝检索里没拿到。
> 本文件只做调研，不含任何代码改动。

---

## 0. TL;DR（最重要的三条）

1. **【确认】官方 SuGaR 仓库里已经自带了 DN-Consistency（深度-法向一致性）正则**，不是我们首创。
   实现在 `sugar_trainers/coarse_density_and_dn_consistency.py`，README 明确写"**We recommend using `dn_consistency` for the best mesh quality**"。
   官方超参：`dn_consistency_factor = 0.05`、`start_dn_consistency_from = 9000`（coarse 总共 15000 iter，即**只在最后 6000 iter 才开启**）。
   → 我们用 λ=0.2 且（若）从头开启，正好踩在官方避开的两个点上；λ=0.05 是官方值。
2. **【确认】2DGS 原文/官方代码让梯度同时流经"渲染法向"和"由深度求得的法向"**，两侧都不 detach；唯一的 detach 是 `surf_normal * (render_alpha).detach()`。
   它防平凡解靠的是 **(a) 小权重 λ_normal=0.05、(b) 延迟开启（iter>7000/30000）、(c) 同时加 depth distortion 损失（iter>3000）**。
3. **【确认】GOF 论文明确指出**：把 2DGS 的 normal consistency 直接搬到 3D 高斯上是有问题的（"the gradient of 3D Gaussians always points outwards from the centers"、"the normal at the projected 2D center is not well-defined"），
   并且在 distortion 损失里**显式 detach 权重**（"we detach the gradient of weights and only minimize the distance between Gaussians"），否则会出现"exaggerated Gaussians, resulting in floaters"。

---

## 1. SuGaR 之后 / 同期的开源表面重建工作清单

| 工作 | 链接 | 改动落在哪一环 | 核心思想（1–2 句） | 报告收益 |
|---|---|---|---|---|
| **2DGS** (SIGGRAPH'24) | [arXiv 2403.17888](https://arxiv.org/abs/2403.17888) · [github hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | 表示（3D椭球→2D surfel）+ 训练 loss（depth distortion + normal consistency）+ mesh 提取（TSDF fusion + Marching Cubes） | 把高斯压成 2D 圆盘，法向唯一且视角无关，于是"渲染法向 vs 深度梯度法向"这个一致性损失才 well-posed；深度失真损失把沿光线的权重挤到同一深度。 | **【确认】** 官方 README 复现的 TnT F1：Barn 0.41 / Caterpillar 0.23 / Ignatius 0.51 / **Truck 0.45** / Meetingroom 0.17 / Courthouse 0.15，**mean 0.32**（对比 SuGaR mean 0.19，见 GOF 表）。DTU CD ≈ 0.80。 |
| **GOF (Gaussian Opacity Fields)** (SIGGRAPH Asia'24 TOG) | [arXiv 2404.10772](https://arxiv.org/abs/2404.10772) · [github autonomousvision/gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields) | mesh 提取（**Marching Tetrahedra + 二分搜索**，网格由高斯的 3D bbox 中心/角点诱导）+ 正则（detach 权重的 distortion） | 从 ray-tracing 形式的体渲染导出"不透明度场"，直接取等值面，**不再需要 Poisson / TSDF**；四面体网格自适应场景复杂度。 | **【确认】** DTU 平均 CD **0.74**（2DGS 0.80）；**TnT 平均 F1 0.46**，显著高于 2DGS 0.32 和 **SuGaR 0.19**。 |
| **PGSR** (TVCG 2024) | [arXiv 2406.06521](https://arxiv.org/abs/2406.06521) · [github zju3dv/PGSR](https://github.com/zju3dv/PGSR) | 渲染（无偏深度）+ 训练 loss（单视图几何正则 + **多视图光度/几何一致性**） | 把高斯压平成平面，直接渲染"相机原点到高斯平面的距离"和法向图，两者相除得到**无偏深度**；再用多视图一致性约束。 | **【确认】** 声称是首个 planar-based GS 表面重建，超过所有 3DGS-based 与 SDF-based 方法；**DTU CD 降到 0.47**（调参后，2024）。 |
| **RaDe-GS** (TOG) | [arXiv 2406.01467](https://arxiv.org/abs/2406.01467) · [项目页 baowenz.github.io/radegs](https://baowenz.github.io/radegs/) | 渲染（闭式解析地光栅化深度与法向）+ 保持 3DGS 效率 | 给一般 3D 高斯推导出**解析的深度/法向光栅化**，不需要退化成 2D surfel 就能拿到几何一致的深度。 | **【确认】** DTU 上 CD 误差可与 NeuraLangelo 相当，训练/渲染开销与原版 3DGS 接近。具体数字**【未找到】**（检索未拿到表格）。 |
| **Gaussian Surfels** (SIGGRAPH'24) | [arXiv 2404.17774](https://arxiv.org/abs/2404.17774) | 表示（z-scale=0 的 surfel）+ 训练 loss（**depth-normal consistency**）+ mesh 提取（**volumetric cutting → screened Poisson, depth=10**） | z 尺度归零后光度损失对法向**没有梯度**，用"深度图求法向 vs 渲染法向"的一致性损失把梯度引回来；再用体素裁剪去掉 alpha-blending 产生的错误深度点，最后 screened Poisson。 | **【确认】** DTU 平均 CD **1.19 mm**（6.67 min 训练）；BlendedMVS 3.62 mm（3.82 min）。 |
| **TrimGS / Trim3DGS** (2024) | [arXiv 2406.07499](https://arxiv.org/abs/2406.07499) · [trimgs.github.io](https://trimgs.github.io/) | **剪枝/密度策略** + mesh 提取（直接取 level set，**不用 Poisson / TSDF**） | 分析每个高斯对几何的"贡献度"，按贡献剪掉冗余/不准确的高斯，同时刻意维持较小的高斯尺度以保细节；可叠加在别人的几何正则上。 | **【确认】** 与已有几何正则（2DGS/3DGS）可组合；具体提升数字**【未找到】**。 |
| **GS2Mesh** (ECCV'24) | [arXiv 2404.01810](https://arxiv.org/abs/2404.01810) · [github yanivw12/gs2mesh](https://github.com/yanivw12/gs2mesh) | **纯 mesh 提取端**（不改训练） | 从训练好的 3DGS 渲染**立体像对**，用预训练立体匹配网络 DLNR 出深度，再 TSDF + Marching Cubes。训练侧零改动，只加一点后处理开销。 | **【确认】** 在 DTU 上优于 SuGaR / 2DGS / GOF（原文表述）；具体数值**【未找到】**。 |
| **DN-Splatter** (WACV'25) | [arXiv 2403.17822](https://arxiv.org/abs/2403.17822) · [github maturk/dn-splatter](https://github.com/maturk/dn-splatter) | 训练 loss（**单目深度/法向先验** + 基于彩色图梯度的自适应深度损失 + 近邻高斯平滑） | 用现成单目网络给深度和法向做监督，配上按图像梯度自适应加权的深度损失，让高斯贴合真实几何，之后可直接抽 mesh。 | **【确认】** 室内数据集上深度估计与 NVS 均优于多个 baseline；具体数值**【未找到】**。 |
| **GaussianPro** (ICML'24) | [PMLR v235](https://proceedings.mlr.press/v235/cheng24f.html) · [github kcheng1021/GaussianPro](https://github.com/kcheng1021/GaussianPro) | **密度/致密化策略** + planar loss | 借鉴 MVS 的 patch match，把邻域像素的深度/法向传播出去生成位置和朝向都靠谱的新高斯，解决 SfM 点在弱纹理区太稀的问题；再加平面损失。 | **【确认】** Waymo 上 PSNR +1.15 dB（vs 3DGS）。 |
| **Gaussian Frosting**（SuGaR 同作者，ECCV'24 Oral） | [arXiv 2403.14554](https://arxiv.org/abs/2403.14554) · [github Anttwo/Frosting](https://github.com/Anttwo/Frosting) | mesh 提取（**沿用并改进 SuGaR 的提取法**）+ 表示（mesh 外包一层可变厚度高斯"糖霜"） | 先用 SuGaR 方式抽基础 mesh，再在其周围建一层厚度自适应的高斯层，用来表达头发/草这类体积性细节；可编辑、可动画。 | **【确认】** 明确表述"relies on the mesh extraction method proposed in SuGaR, which it improves by **automatically selecting a critical hyperparameter**"（即自动选 surface level / 厚度那个关键超参）。具体指标**【未找到】**。 |
| **MILo**（SuGaR 一作 Guédon，SIGGRAPH Asia 2025 TOG） | [arXiv 2506.24096](https://arxiv.org/abs/2506.24096) · [github Anttwo/MILo](https://github.com/Anttwo/MILo) · [项目页](https://anttwo.github.io/milo/) | **训练中可微地抽 mesh**（顶点位置 + 连通性都可微）——彻底取代"训练完再 Poisson"这一环 | 每个训练 iteration 都以高斯为 pivot 做 Delaunay 三角化抽 mesh，并从高斯算 SDF；mesh 与高斯**双向一致性**互相监督，避免几何侵蚀。 | **【确认】** 声称 SOTA 质量、顶点数比以往方法少**一个数量级**、得到轻量中空 mesh。具体 CD/F1 数字**【未找到】**（摘要页无表）。 |
| **ARGS**（直接 build on SuGaR，2025） | [arXiv 2508.21344](https://arxiv.org/abs/2508.21344) | 训练 loss / 正则（两条） | 明确"builds upon SuGaR"：① **rank 正则**，抑制针状高斯、鼓励盘状；② 引入 **neural SDF + Eikonal loss** 作为连续的全局表面先验来引导高斯对齐。 | 指标、数据集、是否开源 **【未找到】**（arXiv 摘要页未给）。 |
| **2D-SuGaR**（Eurographics 2026 Short） | [arXiv 2605.00569](https://arxiv.org/abs/2605.00569) | 初始化 + 剪枝（+ 单目先验） | 在 2DGS 基础上引入单目深度/法向先验，做**深度引导的高斯初始化**，以及**基于聚类的退化高斯剪枝**。 | DTU 上"SOTA mesh reconstruction"，具体数字**【未找到】**；**开源情况【未找到】**（摘要页无 GitHub 链接）。注意：名字虽叫 2D-SuGaR，但它是在 2DGS 上改，不是 SuGaR 的 fork。 |

**其他检索到但未深挖（仅列出，供参考）**：
2DGS-Room（[arXiv 2412.03428](https://arxiv.org/abs/2412.03428)，室内 seed-guided）、GausSurf、GSurf（[arXiv 2411.15723](https://arxiv.org/abs/2411.15723)）、SolidGS（[arXiv 2412.15400](https://arxiv.org/abs/2412.15400)，稀疏视角）、CityGaussianV2、GS-2M。
**SuGaR 仓库的 fork**：检索到的 `TonyDua/SuGaR-Windows`、`latay/SuGaR-GS-to-Mesh`、`Digital-Humans-3DGS/SuGaR-DH` 均为**移植/应用向 fork，没有发现算法改进**。真正的"SuGaR 后续"是同作者的 **Frosting** 和 **MILo**，以及论文层面的 **ARGS**。

---

## 2. 深度-法向一致性损失（DNC）的稳定化技巧 —— 重点

### 2.1 2DGS 是否让梯度同时流经深度与法向？—— **是**【确认】

官方 `gaussian_renderer/__init__.py`：
```python
surf_depth  = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
surf_normal = surf_normal.permute(2,0,1)
# remember to multiply with accum_alpha since render_normal is unnormalized.
surf_normal = surf_normal * (render_alpha).detach()
```
官方 `train.py`：
```python
lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
lambda_dist   = opt.lambda_dist   if iteration > 3000 else 0.0
...
normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
normal_loss  = lambda_normal * (normal_error).mean()
dist_loss    = lambda_dist   * (rend_dist).mean()
```
**结论**：`surf_normal` 由**可微的** `surf_depth` 经 `depth_to_normal` 得到，**没有 detach**；`rend_normal` 也没有 detach。
**唯一的 detach 是 `render_alpha`**（注释说明它只是用来补偿"渲染法向未归一化"这个尺度）。所以梯度确实**同时**流向深度和法向两侧。
默认超参【确认，来自 `arguments/__init__.py`】：`lambda_normal = 0.05`、`lambda_dist = 0.0`、`depth_ratio = 0.0`、`lambda_dssim = 0.2`。
2DGS README【确认】："For unbounded/large scenes, we suggest using mean depth, i.e., `depth_ratio=0`, for less disk-aliasing artifacts"；DTU 示例用 `depth_ratio=1`（median depth）。
> 各数据集推荐的 `lambda_dist` 具体数值 **【未找到】**（README 抓取到的片段里没有逐数据集列表）。

### 2.2 有没有工作报告过"深度被抹平"的平凡解？

- **【确认】GOF 论文**明确讨论了把 normal consistency 用到 **3D**（非 2D surfel）高斯上的病态性：
  > "the gradient of 3D Gaussians always points outwards from the centers … the rendered normals at two different pixels will differ as long as the directions from the projected 2D Gaussian center to the pixel coordinates vary. Moreover, **the normal at the projected 2D center is not well-defined**."

  它的解法不是 detach，而是**换法向定义**：用**光线-高斯相交平面的法向**来近似高斯法向。
- **【确认】GOF 在 distortion 损失上用了 detach**：
  > "we **detach the gradient of weights** and only minimize the distance between Gaussians"

  动机是：不 detach 的话优化会去压低权重、抬高先混合高斯的 alpha，产生"exaggerated Gaussians, resulting in **floaters**"。这是"用 detach 掐断一条捷径"的直接先例。
- **【确认】Gaussian Surfels** 报告的是**互补方向**的问题：z-scale=0 时**光度损失对法向完全没有梯度**，所以必须靠 DNC 把梯度从深度侧引过来。它的做法是 **λ_c 从 0 线性升到 0.1**（λ_o=0.01、λ_m=1），即**权重调度**而非 detach。论文中**没有提到 stop-gradient**【确认：抓取的 v1 全文里没有】。
- **"深度被抹平 / 平凡解"这个说法本身**：我**没有**在任何一篇论文里找到明确写"depth collapses to a trivial/over-smoothed solution"的句子。**【未找到】**
  但**【推测】**其机理很清楚：`1 - cos(N_render, N_depth)` 对"深度局部为平面"是**零损失**的，所以只要 DNC 的权重相对光度损失过大、或开启过早（此时光度约束还很弱），最优解就会滑向"处处局部平面 = 深度被抹平"。业界通用的四个对策恰好都对应上：**小权重 / 延迟开启 / 加 depth distortion 让权重沿光线集中（防止深度变成多层混合的平均值）/ 掩码只在有效前景像素上算**。

### 2.3 官方 SuGaR 自己的 DN-Consistency（最关键的参照）【确认，本地源码 `repo/SuGaR` @ commit 7c10c4a】

文件：`sugar_trainers/coarse_density_and_dn_consistency.py`
```python
# Depth-Normal consistency
enforce_depth_normal_consistency = True
if enforce_depth_normal_consistency:
    start_dn_consistency_from = 9000   # 7000
    dn_consistency_factor     = 0.05   # 0.1
...
if enforce_depth_normal_consistency and iteration > start_dn_consistency_from:
    depth_img, normal_img = sugar.render_depth_and_normal(camera_indices=camera_indices.item())
    normal_error = depth_normal_consistency_loss(
        depth=depth_img[None], normal=normal_img.permute(2, 0, 1),
        camera=..., scale_rendered_normals=False, return_normal_maps=False)
    loss = loss + dn_consistency_factor * normal_error
```
损失体（同文件 line 59 起，注释直接写 "Comes from 2DGS"）：
```python
normal_from_depth = depth2normal_2dgs(camera, depth)                     # 与 2DGS 同款：unproject → cross(dx,dy)
normal_from_depth = (normal_from_depth @ camera.world_view_transform[:3,:3]).permute(2,0,1)
normal_error = (1 - (normal_view * normal_from_depth).sum(dim=0))        # 无 abs、无 detach
return normal_error.mean()
```
`SuGaR.render_depth_and_normal()`（`sugar_scene/sugar_model.py:2343`）的两个要点：
1. 它把 **深度 + 法向的 x,y 两个分量** 一起当作 3 通道"颜色"过光栅化器 alpha 合成；
2. **z 分量不是合成出来的，而是 `-sqrt(1 - x² - y²)` 解析补出来的** → 渲染法向天然是**单位向量**。
   → 这与 2DGS 不同（2DGS 的 rend_normal 未归一化，所以才要乘 detach 掉的 alpha）。**【推测】** 正因如此，SuGaR 版 DNC 不需要 alpha 补偿项，但也意味着它**没有"透明度守门"**：背景/低不透明度像素照样贡献损失。

**其他并行的稳定化设置【确认】**：同一 trainer 里 `entropy_regularization`（opacity 二值化熵）只在 **7000→9000** 区间开，SuGaR 自己的 density/SDF 正则从 7000 开始，DNC 从 **9000** 才开始。也就是说官方把三段正则**错峰**排布，DNC 是最后一棒。

### 2.4 与我们现状的对照【推测】

- λ=0.2 是官方 0.05 的 **4 倍**，且（若从 coarse 起点就开）比官方早 9000 iter → 出现"深度被抹平"完全符合 §2.2 的机理。
- 我们自研的 `sugar_utils/dnc_utils.py` 用的是 `1 - |cos|`（abs）并带 mask（`bg_depth_ratio`、`depth_grad_rel_thresh`、`min_normal_norm`），这**比官方版更保守**（已经有掩码守门）。**【推测】** 那么剩下的主因就是权重和起始 iteration。

---

## 3. 候选改动清单（1 小时内可落地，按"性价比"排序）

> 所有"要改的文件/函数"里，凡我没亲眼读到的都标了**【推测】**。

### ① 对齐官方 SuGaR 的 DN-Consistency 超参：λ=0.05 + 延迟到 coarse 的最后 40% iteration
- **来源**：官方 SuGaR `coarse_density_and_dn_consistency.py`【确认】；2DGS `train.py` 的 `iteration > 7000` 调度【确认】。
- **改哪**：我们改过的 `sugar_trainers/coarse_density.py`（`git status` 显示已 M）里 DNC 的 factor 与起始 iteration。
- **建议取值**：`dn_consistency_factor = 0.05`；coarse 7000→15000 的话，`start_dn_consistency_from = 12000`（对齐官方"总 iter 的后 40%"比例；官方是 9000/15000）。
  更保守可用 **线性 warm-up**：从 12000 起 0 → 0.05 线性升到 14000（来源：Gaussian Surfels 的 λ_c 线性升到 0.1【确认】）。
- **难度**：★（改两个常数）。**预期收益**：直接消除平凡解，是当前最该先做的一步。

### ② mesh 提取端调参：`poisson_depth` 与 `vertices_density_quantile`（治碎片/孔洞的"官方处方"）
- **来源**：SuGaR 官方 README 的 Troubleshooting【确认】+ 本地 `sugar_extractors/coarse_mesh.py` line 42/43【确认，已读到 `poisson_depth = 10`、`vertices_density_quantile = 0.1`】。
- **改哪**：`sugar_extractors/coarse_mesh.py:42-43`（**已确认行号**）。
- **建议取值**：碎片/椭球疙瘩多 → `poisson_depth` 10 → **8**（README 建议 6/7/8）；孔洞多或连通分量碎 → `vertices_density_quantile` 0.1 → **0.05 或 0**（=关掉清洗滤波）。
  另可扫 `surface_level`：脚本默认会扫 `[0.1, 0.3, 0.5]`，我们固定了 0.3，**【推测】** 对 Truck 这种带大背景的场景，0.1 会更"厚"更连通、0.5 更薄更碎。
- **难度**：★（改常数 + 重跑提取，**不用重训**）。**预期收益**：对"连通分量/碎片数"这个指标最直接、最快见效。

### ③ 加 depth distortion（深度失真）损失，和 DNC 配对使用
- **来源**：2DGS 原文两大正则之一【确认】；GOF 版本**detach 权重**【确认原文："we detach the gradient of weights and only minimize the distance between Gaussians"】。
- **改哪**：**【推测】** 需要光栅化器输出每条光线上的 `(weight_i, depth_i)` 才能算标准 distortion。SuGaR 用的是原版 3DGS 光栅化器，**默认不输出这些**，所以严格实现要改 CUDA → 一小时内做不完。
  **一小时内可行的替代**：在 `sugar_trainers/coarse_density.py` 里加一个**廓边保护的深度平滑/TV 项**（对 RGB 梯度小的像素才罚深度梯度），思想来自 **DN-Splatter 的"基于彩色图梯度的自适应深度损失"**【确认其存在，但具体公式我没读到 → 公式属**【推测】**】。
- **建议取值**：**【推测】** 权重 0.01–0.05，同样延迟到 12000 后开；RGB 梯度阈值用图像梯度的 80 分位。
- **难度**：★★★（真 distortion）/ ★★（替代版）。**预期收益**：中；主要是压浮点/层叠，间接减少碎片。

### ④ 只在"高不透明度前景 + 非深度不连续"像素上算 DNC（我们其实已经有，建议做消融）
- **来源**：Gaussian Surfels 的 volumetric cutting 思想（去掉 alpha-blending 产生的错误深度点）【确认】；GOF 的 "detach 掐捷径" 思想【确认】。
- **改哪**：`sugar_utils/dnc_utils.py::build_dnc_mask`（**已确认存在**，参数 `bg_depth_ratio=0.98`、`border=2`、`depth_grad_rel_thresh=0.05`、`min_normal_norm=0.1`）。
- **建议取值**：**【推测】** 把 `min_normal_norm` 从 0.1 提到 **0.3–0.5**（更严格地只在不透明处算），`depth_grad_rel_thresh` 从 0.05 收到 **0.02**（更严地跳过深度断崖）。同时记录 `valid_ratio`（函数已支持 `return_aux`）做为诊断。
- **难度**：★（改默认值）。**预期收益**：中；能把"抹平"局限在真正的表面区域，不波及背景。

### ⑤ Poisson 前处理：按"贡献度/尺度"剪掉不可靠高斯后再采样表面点
- **来源**：**TrimGS**（contribution-based trimming，且保持高斯尺度较小）【确认思想，数值未找到】；**2D-SuGaR** 的 clustering-based degenerate-Gaussian pruning【确认思想】。
- **改哪**：**【推测】** `sugar_extractors/coarse_mesh.py` 里采样表面点的那一段（`surface_level_*` 系列参数附近），在采样前按 opacity 阈值 + 最大尺度阈值过滤高斯；或在 coarse 训练末尾调一次 densifier 的 prune。
- **建议取值**：**【推测】** 剪掉 `opacity < 0.1` 或 `max_scale > 3×median_scale` 的高斯；先看剪掉的比例（目标 5–15%，别超 30%）。
- **难度**：★★（要读采样代码）。**预期收益**：中高，针对"碎片数 / COLMAP 点到 mesh 距离"两个指标都有帮助，但一小时内有踩坑风险。

> **不建议在本次考核里做**：换成 2DGS/GOF 的光栅化器、上 MILo 式训练中可微抽 mesh、上 neural SDF（ARGS）—— 都需要改 CUDA 或引入新依赖，远超 1 小时。

---

## 4. 事实 / 推测 分界速查

**【确认】**（有原文或源码出处）：
- SuGaR 官方带 `dn_consistency` 且 README 推荐它；factor 0.05、start 9000/15000；损失注释写 "Comes from 2DGS"；无 detach、无 abs。
- SuGaR `coarse_mesh.py` 第 42/43 行是 `poisson_depth=10` / `vertices_density_quantile=0.1`；默认扫 surface_level `[0.1,0.3,0.5]`。
- 2DGS：λ_normal=0.05、λ_dist=0.0、depth_ratio=0.0 默认；normal loss 从 iter 7000、dist loss 从 3000 开；只 detach `render_alpha`。
- 2DGS 官方复现 TnT F1：Truck 0.45，mean 0.32。GOF：DTU 0.74、TnT F1 0.46，SuGaR TnT F1 0.19。
- Gaussian Surfels：λ_c 由 0 线性升到 0.1；screened Poisson depth=10；DTU CD 1.19mm。
- GOF：3D 高斯法向 ill-defined 的论述；distortion 里 detach 权重防 floaters；Marching Tetrahedra 提取。
- Frosting "improves SuGaR's mesh extraction by automatically selecting a critical hyperparameter"。
- ARGS builds upon SuGaR，加 rank 正则 + neural SDF/Eikonal。
- 2D-SuGaR 是在 2DGS 上加单目先验 + 深度引导初始化 + 聚类剪枝，Eurographics 2026 Short。

**【推测】**（我的推断，未验证）：
- "λ=0.2 且过早开启 → 深度抹平" 的机理解释。
- SuGaR 渲染法向解析补 z 分量 ⇒ 无需 alpha 补偿但也无透明度守门。
- 候选改动 ③④⑤ 中所有具体数值与"要改哪个函数"（除已标注"已确认行号"者）。
- surface_level 0.1 更连通 / 0.5 更碎。

**【未找到】**：
- 任何论文里明确写"深度退化为平凡解 / depth collapse"的句子。
- 2DGS README 中逐数据集的 `lambda_dist` 推荐值表。
- RaDe-GS / TrimGS / GS2Mesh / DN-Splatter / MILo / ARGS / 2D-SuGaR 的精确指标表格。
- ARGS、2D-SuGaR 是否开源。
- 任何"对 SuGaR 官方仓库做算法改进"的 fork（检索到的 fork 都是移植/应用向）。
