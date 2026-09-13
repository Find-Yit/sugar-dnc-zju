# SELF_CHECK 阶段 3f — 移植 Frosting 的自动 Poisson 深度 + mesh 提取端扫参

执行者 9（claude-opus-5）。日期 2026-09-13。
依据：`notes/SURVEY2_sugar_direct_followups.md` §2.1 候选① / §4 候选③；`plans/SuGaR复现与改动_6小时考核.txt` §2b。
改动范围：**只在 mesh 提取端**，不重训 coarse，不改评测逻辑（`summarize.py` 有一处最小改动，diff 见 §5）。

---

## 0. 一句话结论（先说最重要的）

> **在 Truck 场景上，Frosting 的自动 Poisson 深度算出来就是 D = 10，与 SuGaR 原版硬编码的 10 完全相同 —— 该改动在本场景零收益。**
> 公式的原始值是 `raw = 10.873`，被 `max_poisson_depth=10` 截断到 10。
> 因此本阶段的实验重点转向：(a) 手动把 D 调小（pd9 / pd8）验证"更小的 D 是否真的更好"；
> (b) `vertices_density_quantile` 扫描（q0 / q005）。

> **第二条结论（本阶段真正有价值的发现）：把 `vertices_density_quantile` 从 0.1 改成 0，
> 在同一个 coarse 模型上就能把 mesh 的破碎问题基本解决** ——
> 最大连通分量的面数占比 **41.686% → 92.776%（+122.6%）**、连通分量数 **−23.9%**、
> 碎片(<100 面) **−22.0%**、边界边（洞的周长）**−70.0%**、
> 从测试视角光栅化的 mesh 命中率 **92.03% → 99.99%**、G1 mean **−5.3%**、G2<1% **+0.41%**；
> 代价只有 G1 median **+4.7%**、二面角 abs mean **+3.1%**。详见 §4.5。

> **第三条结论：手动把 D 调小（9/8）能减碎片、能变平滑、提取快一倍，但精度代价大**
> （D=8 时 G1 median 恶化 **+42.3%**），而且**治不好主体破碎**（最大分量占比只到 48.6%）。详见 §4.5(c)。

---

## 1. 移植内容（代码改动）

完整 diff：`notes/diffs/s3f_poisson_sugar.diff`（针对 `repo/SuGaR_dev`，已原样复制到 `repo/SuGaR` 同路径）。

### 1.1 `sugar_extractors/coarse_mesh.py`

新增函数 `compute_optimal_poisson_depth`，**函数体逐行照抄** Frosting：

- 来源：`Anttwo/Frosting` → `frosting_extractors/coarse_shell.py` 第 17–49 行
  <https://github.com/Anttwo/Frosting/blob/main/frosting_extractors/coarse_shell.py>
  （raw: <https://raw.githubusercontent.com/Anttwo/Frosting/main/frosting_extractors/coarse_shell.py>，
  本次执行时用 `wget` 实际下载核对过）
- 论文：Guédon & Lepetit, *Gaussian Frosting*, ECCV 2024 Oral（与 SuGaR 同作者）；
  补充材料 §7 "Improving surface reconstruction" 说明动机：SuGaR 对所有场景硬编码大 D=10，
  当八叉树分辨率相对场景细节过高时，高斯的椭球形状会以疙瘩形式显现在表面上，且会出现孔洞。
- **`knn_points(...).dists` 返回的是平方距离（pytorch3d 约定），Frosting 直接使用、未开方；
  常数 `cell_size_nn_distance_ratio=100` 是配套调出来的。本次移植照抄，没有"修正"成开方。**
- 相对原版**唯一**的改动：新增可选关键字 `return_details: bool = False`（默认 False ⇒ 签名与返回值
  与原版完全一致），为 True 时额外返回中间量（bbox_size / quantile_dist / raw_depth / 各掩码点数），
  用于落盘 `extract_stats.json`。
- 逐行核对证据：见 §2 的 C3（用 difflib 对比下载到的 Frosting 源码，除 `return_details` 相关行外完全一致）。

**调用位置（与调研笔记的推测不同，此处以实际源码为准）**：
Frosting 在 `coarse_shell.py:234-240` 调用它，位置是 **`for name, param in sugar.named_parameters()` 打印之后、
`# Pruning low opacity gaussians` 之前**，即"剪掉低不透明度高斯**之前**"。
本次移植放在 SuGaR `coarse_mesh.py` 的同一位置。这一点很重要：函数内部自带 `opacity_threshold=0.5` 掩码，
若放在 SuGaR 的 `drop_low_opacity_points(0.5)` 之后，该掩码会退化成空操作，口径就不再是 Frosting 的了。

原第 42/43 行的硬编码：
```python
poisson_depth = 10
vertices_density_quantile = 0.1
```
保留为默认值，随后由命令行参数覆盖。**不传新参数时行为与原版完全一致**（验证见 §2 的 C4）。

### 1.2 `extract_mesh.py` 新增 4 个参数

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--poisson_depth` | str | `"10"` | 整数或 `auto`（等价 `-1`，Frosting 的写法）|
| `--vertices_density_quantile` | float | `0.1` | Frosting 叫 `--cleaning_quantile`；0 表示不做低密度顶点清洗 |
| `--cell_size_nn_distance_ratio` | float | `100` | 自动深度公式里的常数，只在 `auto` 时生效 |
| `--only_report_depth` | str2bool | `False` | 只算深度、打印并写 `extract_stats.json`，不提取 mesh |

选定的 D 与所有中间量写入 `<mesh_out>/extract_stats.json`。

---

## 2. 验收条目逐条核对

| # | 验收条目 | 实际结果 | PASS/FAIL | 复算命令 |
|---|---|---|---|---|
| C1 | `py_compile` 通过（两个改动文件 × 两个仓库） | 4 个文件全部通过 | PASS | `python -m py_compile repo/SuGaR_dev/sugar_extractors/coarse_mesh.py repo/SuGaR_dev/extract_mesh.py repo/SuGaR/sugar_extractors/coarse_mesh.py repo/SuGaR/extract_mesh.py` |
| C2 | argparse 能解析新参数 | `--help` 列出 4 个新参数；实跑 `--poisson_depth auto/8/9/10`、`--vertices_density_quantile 0/0.05/0.1` 均正常 | PASS | `cd repo/SuGaR_dev && python extract_mesh.py --help \| tail -25` |
| C3 | 函数体与 Frosting 逐行一致（除 `return_details`） | difflib 对比：除新增的 `return_details` 相关行外**零差异** | PASS | 见下方 C3 复算脚本 |
| C4 | 默认值下行为与原版一致 | ① diff 显示默认路径上没有任何行为改动（只多了 4 行打印和一次 json 落盘，`poisson_depth`/`vertices_density_quantile` 的默认取值仍是 10 / 0.1）；② **不能用 md5 比对**：SuGaR 的提取本身不确定（`coarse_mesh.py:471` 无种子 `torch.randperm`），同参两次结果必然不同。改用统计等价性验证：`pdauto`（auto→10, q=0.1）vs 阶段 3.4 的 `base_seed0`（原版代码，D=10, q=0.1），13 项几何指标的相对差全部 ≤1.7%，落在重复运行噪声底内（见 §4.4） | PASS | `python scripts/make_s3f_table.py`（看 pdauto 那一行）|
| C5 | auto 深度实际算出的 D 与中间量落盘 | `extract_stats.json` 已写；D=10 | PASS | `cat outputs/runs/_depth_report_base_seed0/extract_stats.json` |
| C6 | 独立复算 auto D（不依赖 SuGaR/pytorch3d） | CPU scipy cKDTree：raw=10.8831 → **D=10**（GPU：raw=10.8732 → D=10） | PASS | `python scripts/check_auto_poisson_depth_cpu.py --coarse_pt ... --gs_checkpoint outputs/baseline/gs_truck` |
| C7 | 每组提取的几何指标写进 `outputs/metrics/summary.csv` | 6 组（pdauto / pdauto_q0 / q0 / q005 / pd9 / pd8）全部写入，run 名为 `base_seed0_<tag>`，`source_coarse` 列均为 `coarse_base_seed0`。**auto 深度那一组（`base_seed0_pdauto`）于 09:24 写入，早于 10:10 的硬性时限** | PASS | `python scripts/eval/summarize.py && grep base_seed0_ outputs/metrics/summary.csv` |
| C8 | 不删文件 / 不 commit / 不 pip install / 不重训 coarse | 全部遵守：只新增文件 + 改 2 个 SuGaR 文件 + `summarize.py` 一处最小改动 | PASS | `git -C repo/SuGaR_dev status --short` |

**C3 复算脚本**（可直接粘贴运行）：
```bash
cd /scratch/e1351071/zju_test
wget -qO /tmp/frosting_coarse_shell.py \
  https://raw.githubusercontent.com/Anttwo/Frosting/main/frosting_extractors/coarse_shell.py
python - <<'PY'
import io, difflib
fro = io.open("/tmp/frosting_coarse_shell.py", encoding="utf-8").read().split("\n")[16:49]   # 第 17-49 行
mine = io.open("repo/SuGaR_dev/sugar_extractors/coarse_mesh.py", encoding="utf-8").read().split("\n")
i = [k for k, l in enumerate(mine) if l.startswith("def compute_optimal_poisson_depth")][0]
j = [k for k, l in enumerate(mine) if l.strip().startswith("if return_details:")][0]
norm = lambda ls: [l.rstrip() for l in ls if l.strip() and "return_details" not in l]
d = list(difflib.unified_diff(norm(fro), norm(mine[i:j]), lineterm="", n=0))
print("VERBATIM MATCH" if not d else "\n".join(d))
PY
```
实际输出：只差最后一行 `return poisson_depth`（在移植版里被挪到 `if return_details:` 块之后，仍然存在），
其余 26 行逐字相同。

**C6 的小说明（两个数为什么差 0.01）**：GPU 版用 SuGaR 自己的 `CamerasWrapper` 取训练相机
（extent=5.847915，used=77554）；CPU 复算脚本用 `cameras.json` 按 `i % 8 != 0` 取训练相机
（extent=5.892506，used=77679），训练/测试划分的**排序口径**略有不同，导致相机包围球半径差 0.76%。
两者算出的 raw depth 分别是 10.8732 / 10.8831，**取 floor 后都是 10，结论不受影响**。

---

## 3. auto 深度：实际计算值与中间量

命令（12 秒，`logs/s3f_depth_report.log`）：
```bash
bash scripts/run_s3f_depth_report.sh
# 等价于
python extract_mesh.py -s data/tandt/truck -c outputs/baseline/gs_truck/ -i 7000 \
  -m outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 -o outputs/runs/_depth_report_base_seed0 \
  --poisson_depth auto --cell_size_nn_distance_ratio 100 --only_report_depth True
```

| 量 | 值 |
|---|---|
| `cameras_spatial_extent` | 5.847914695739747 |
| `camera_average_xyz` | [0.08619998, -0.00717573, 0.15169582] |
| 高斯数 total / 前景 / 不透明(>0.5) / **实际用于 KNN** | 429423 / 112209 / 282930 / **77554** |
| `bbox_size` = 1.1 × max 边长 | 12.854769802093507 |
| `quantile_dist`（归一化后的**平方**最近邻距离 10% 分位） | 5.331572538125329e-06 |
| `raw_depth = -log2(100 × quantile_dist)` | **10.873151263385152** |
| `floor` → `min(·, max_poisson_depth=10)` | **D = 10** |

**⇒ auto D = 10 = SuGaR 原版硬编码值。该改动在 Truck 上零收益。**

`cell_size_nn_distance_ratio` 敏感性（来自 `scripts/check_auto_poisson_depth_cpu.py` 的 CPU 复算）：

| ratio | raw depth | D |
|---|---|---|
| 25 | 12.883 | 10（截断）|
| 50 | 11.883 | 10（截断）|
| **100（Frosting 默认）** | **10.883** | **10（截断）** |
| 200 | 9.883 | 9 |
| 400 | 8.883 | 8 |
| 800 | 7.883 | 7 |

即：要让自动机制在本场景真的把 D 降下来，ratio 至少要 ≈189。Frosting 的默认常数在 Truck 上"够不着"。
---

## 4. 提取端扫参结果

### 4.1 实验设计

7 组**全部基于同一个 coarse 模型** `outputs/runs/coarse_base_seed0/.../15000.pt`（未重训），
提取参数除新参数外完全一致：`-l 0.3 -d 200000 --eval True --gpu 0`。
因此**渲染指标（PSNR/SSIM/LPIPS）在各组之间逐字相同**（同一批高斯），在 `summary.csv` 里由
`source_coarse = coarse_base_seed0` 标注并继承，不重复计算。

**⚠ 关键方法学前提：SuGaR 的 mesh 提取本身是不确定的。**
`sugar_extractors/coarse_mesh.py:471` 用无种子的 `torch.randperm` 对每帧表面交点做随机下采样
（`sugar_scene/sugar_model.py:1997` 同样），`extract_mesh.py` 全程没有 `torch.manual_seed`。
所以同参数跑两次结果不同。为此本轮特意安排了**两对同参重复**来量化噪声底：

| 重复对 | 参数 | 用途 |
|---|---|---|
| `base_seed0`（阶段 3.4） vs `pdauto` | D=10, q=0.1 | q=0.1 条件下的噪声底 |
| `q0` vs `pdauto_q0` | D=10, q=0 | q=0 条件下的噪声底 |

（`pdauto_q0` 之所以等价于 `q0`，正是因为 auto 算出来就是 10 —— 见 §3。）


### 4.2 各组命令 / 参数 / 耗时 / 顶点面数

| 组 | --poisson_depth | --vertices_density_quantile | 实际 D | 提取墙钟(s) | 顶点数 | 面数 | mesh 路径 |
|---|---|---|---|---|---|---|---|
| (基准, 阶段3.4) | 10 (原版硬编码) | 0.1 (原版硬编码) | 10 | 416 | 205548 | 370128 | `outputs/runs/mesh_base_seed0/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| pdauto | auto | 0.1 | 10 | 291 | 205400 | 369922 | `outputs/runs/mesh_base_seed0_pdauto/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| pdauto_q0 | auto | 0.0 | 10 | 304 | 169291 | 332145 | `outputs/runs/mesh_base_seed0_pdauto_q0/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| q0 | 10 | 0.0 | 10 | 415 | 169359 | 332214 | `outputs/runs/mesh_base_seed0_q0/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| q005 | 10 | 0.05 | 10 | 294 | 193738 | 360910 | `outputs/runs/mesh_base_seed0_q005/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| pd9 | 9 | 0.1 | 9 | 188 | 177161 | 337209 | `outputs/runs/mesh_base_seed0_pd9/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| pd8 | 8 | 0.1 | 8 | 158 | 151037 | 294882 | `outputs/runs/mesh_base_seed0_pd8/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |

> 所有组的固定参数：`-l 0.3 -d 200000 --eval True --gpu 0`，coarse 模型均为 `outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`。
> 「面数」是 FG/BG **各自** decimate 到 200000 三角形后再合并、清洗、投影的结果，所以总面数不等于 200000。
> **墙钟不可直接横比**：`基准(416 s)` 与 `q0(415 s)` 是在机器上还有别的训练/提取并发时跑的；
> `pdauto/pdauto_q0/q005` 是串行独占（D=10 时 291–304 s）。同为串行独占时的真实对比是
> **D=10 ≈ 291 s / D=9 = 188 s / D=8 = 158 s**，即把 D 降 1 可省约 35% 的提取时间。

### 4.3 几何指标并排表（数字全部来自 `outputs/metrics/geometry_*.json`）

| 组 | G1 median rel %↓ | G1 mean rel %↓ | G2 <1% ↑ | G2 <0.5% ↑ | 顶点数 | 面数 | 连通分量↓ | 最大分量面占比%↑ | 碎片(<100面)↓ | 碎片总面数↓ | 边界边↓ | mesh 命中率%↑ | G5 abs mean°↓ | G5 abs P90°↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 原版提取 D=10, q=0.1（基准） | 0.0687 | 0.1770 | 97.736 | 94.134 | 205548 | 370128 | 2233 | 41.686 | 2137 | 29408 | 38596 | 92.033 | 33.6019 | 75.1845 |
| pdauto: D=auto(=10), q=0.1 | 0.0689 | 0.1770 | 97.737 | 94.136 | 205400 | 369922 | 2267 | 42.051 | 2172 | 28924 | 38650 | 91.997 | 33.5632 | 75.0999 |
| pdauto_q0: D=auto(=10), q=0 | 0.0712 | 0.1667 | 98.145 | 94.519 | 169291 | 332145 | 1728 | 92.884 | 1702 | 18932 | 11443 | 99.995 | 34.6408 | 76.0438 |
| q0: D=10, q=0 | 0.0719 | 0.1676 | 98.138 | 94.510 | 169359 | 332214 | 1699 | 92.776 | 1668 | 18247 | 11566 | 99.989 | 34.6279 | 76.0753 |
| q005: D=10, q=0.05 | 0.0691 | 0.1724 | 97.922 | 94.277 | 193738 | 360910 | 1998 | 45.188 | 1924 | 25551 | 26522 | 92.890 | 34.0434 | 75.6431 |
| pd9: D=9, q=0.1 | 0.0748 | 0.1842 | 97.773 | 93.457 | 177161 | 337209 | 1385 | 44.781 | 1315 | 19748 | 21925 | 93.533 | 31.3625 | 72.7436 |
| pd8: D=8, q=0.1 | 0.0978 | 0.2195 | 97.031 | 91.543 | 151037 | 294882 | 748 | 48.648 | 704 | 9086 | 12900 | 95.459 | 26.7791 | 67.3277 |

相对基准（base_seed0）的变化百分比：
| 组 | G1 median rel %↓ | G1 mean rel %↓ | G2 <1% ↑ | G2 <0.5% ↑ | 顶点数 | 面数 | 连通分量↓ | 最大分量面占比%↑ | 碎片(<100面)↓ | 碎片总面数↓ | 边界边↓ | mesh 命中率%↑ | G5 abs mean°↓ | G5 abs P90°↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pdauto: D=auto(=10), q=0.1 | +0.19% | +0.02% | +0.00% | +0.00% | -0.07% | -0.06% | +1.52% | +0.88% | +1.64% | -1.65% | +0.14% | -0.04% | -0.12% | -0.11% |
| pdauto_q0: D=auto(=10), q=0 | +3.59% | -5.80% | +0.42% | +0.41% | -17.64% | -10.26% | -22.62% | +122.82% | -20.36% | -35.62% | -70.35% | +8.65% | +3.09% | +1.14% |
| q0: D=10, q=0 | +4.67% | -5.30% | +0.41% | +0.40% | -17.61% | -10.24% | -23.91% | +122.56% | -21.95% | -37.95% | -70.03% | +8.64% | +3.05% | +1.18% |
| q005: D=10, q=0.05 | +0.51% | -2.61% | +0.19% | +0.15% | -5.75% | -2.49% | -10.52% | +8.40% | -9.97% | -13.12% | -31.28% | +0.93% | +1.31% | +0.61% |
| pd9: D=9, q=0.1 | +8.88% | +4.09% | +0.04% | -0.72% | -13.81% | -8.89% | -37.98% | +7.42% | -38.47% | -32.85% | -43.19% | +1.63% | -6.66% | -3.25% |
| pd8: D=8, q=0.1 | +42.33% | +24.00% | -0.72% | -2.75% | -26.52% | -20.33% | -66.50% | +16.70% | -67.06% | -69.10% | -66.58% | +3.72% | -20.30% | -10.45% |

### 4.4 噪声底（两对同参重复，用来判断上表的差异是否真实）

| 指标 | base_seed0 vs pdauto（q=0.1） | q0 vs pdauto_q0（q=0） | 取保守上界 |
|---|---|---|---|
| G1 median rel | 0.19% | 1.00% | **±1%** |
| G1 mean rel | 0.02% | 0.54% | ±0.6% |
| G2 <1% / <0.5% | 0.00% | 0.01% | ±0.05% |
| 顶点数 / 面数 | 0.07% / 0.06% | 0.04% / 0.02% | ±0.1% |
| 连通分量数 | 1.52% | 1.71% | **±2%** |
| 最大分量面占比 | 0.88% | 0.12% | ±1% |
| 碎片数(<100 面) | 1.64% | 2.04% | **±2%** |
| 边界边数 | 0.14% | 1.07% | ±1.1% |
| G5 abs mean | 0.12% | 0.04% | ±0.2% |
| mesh 命中率 | 0.04% | 0.006% | ±0.1% |

> 噪声来源：`coarse_mesh.py:471` 的无种子 `torch.randperm` 表面点下采样 + open3d PoissonRecon 的多线程非确定性。

### 4.5 结果解读（每条结论都对照噪声底判定）

**(a) `pdauto` = 原版，无效**。全部 13 项指标相对基准的变化都 ≤ 1.7%，落在噪声底内
（最大的一项是连通分量 +1.52%，噪声底就是 ±2%）。这是**改动本身零收益**的直接证据，
同时也反过来验证了移植没有引入任何副作用。

**(b) `vertices_density_quantile` 才是本场景真正的旋钮，且 q=0 单调最好。**
q = 0.1 → 0.05 → 0 是单调趋势（以基准为 0%）：

| | q=0.1（基准） | q=0.05 | q=0 |
|---|---|---|---|
| 最大分量面占比 | 41.686% | 45.188%（+8.4%） | **92.776%（+122.6%）** |
| 连通分量 | 2233 | 1998（−10.5%） | **1699（−23.9%）** |
| 边界边（≈洞的周长） | 38596 | 26522（−31.3%） | **11566（−70.0%）** |
| mesh 光栅化命中率 | 92.03% | 92.89%（+0.9%） | **99.99%（+8.6%）** |
| G1 mean rel | 0.1770% | 0.1724%（−2.6%） | **0.1676%（−5.3%）** |
| G2 <1% | 97.736 | 97.922（+0.19%） | **98.138（+0.41%）** |
| G1 median rel | **0.0687%** | 0.0691%（+0.5%，噪声内） | 0.0719%（+4.7%，略差）|
| G5 abs mean ° | **33.60** | 34.04（+1.3%） | 34.63（+3.1%，略糙）|

机制解释：`vertices_density_quantile=0.1` 是在 Poisson 之后按顶点密度删掉最低 10% 的顶点。
它的本意是删掉"外推出来的假面"，但**它同时在本来闭合的曲面上打出大量洞**，
把主体切成上千块 —— 基准里最大连通分量只占 41.7% 的面，正是这么来的。
关掉它以后主体重新连成一片（92.8%），洞（边界边）少 70%，从测试视角看 mesh 几乎不漏（命中率 99.99%）。
代价很小：点到面距离的**中位数**变差 4.7%（超出 ±1% 噪声底，是真实的小退化），
二面角平均变糙 3.1%；但点到面距离的**均值**反而变好 5.3%，`G2 <1%` 也更好 —— 
说明被留下来的"低密度顶点"整体上并没有把面推离真实几何，反而补上了原来被挖掉的部分。

**(c) 手动把 D 调小（pd9/pd8）能减碎片，但代价大，而且治不好主体的破碎。**

| | D=10（基准） | D=9 | D=8 |
|---|---|---|---|
| G1 median rel（越小越好） | **0.0687%** | 0.0748%（+8.9%） | 0.0978%（**+42.3%**）|
| G2 <0.5% | **94.134** | 93.457（−0.72%） | 91.543（−2.75%）|
| 连通分量 | 2233 | 1385（−38.0%） | **748（−66.5%）** |
| 碎片(<100 面) | 2137 | 1315（−38.5%） | **704（−67.1%）** |
| **最大分量面占比** | 41.686% | 44.781%（+7.4%） | 48.648%（+16.7%）|
| G5 abs mean °（越小越平滑） | 33.60 | 31.36（−6.7%） | **26.78（−20.3%）** |
| 提取墙钟 | 291–416 s | **188 s** | **158 s** |

即：D 越小 → 碎片越少、表面越平滑、提取越快，但**点到面精度显著下降**（D=8 时中位距离恶化 42%），
而且**最大连通分量占比只从 41.7% 提到 48.6%，远不如 q=0 的 92.8%** ——
说明"主体被切碎"的主因是密度清洗打洞，不是八叉树太深。
这在方向上与 Frosting 补充材料的定性说法（D 过大 → 孔洞、疙瘩）一致（边界边 D=8 时 −66.6%，
二面角 −20.3% 即疙瘩变少），但在 Truck 上**代价大于收益**。

**(d) 一句话结论**：
> **本阶段最好的一组是 `q0`（≡`pdauto_q0`）：D 保持 10、`vertices_density_quantile` 从 0.1 改成 0。**
> 相对原版提取：最大连通分量面数占比 **41.686% → 92.776%（+122.6%）**、连通分量 **−23.9%**、
> 碎片数 **−22.0%**、碎片总面数 **−38.0%**、边界边 **−70.0%**、mesh 命中率 **92.03% → 99.99%**、
> G1 mean **−5.3%**、G2<1% **+0.41%**；代价是 G1 median **+4.7%**、二面角 abs mean **+3.1%**。
> 而本次移植的主角 —— Frosting 自动 Poisson 深度 —— 在 Truck 上算出 D=10，**与原版完全相同，零收益**。


---

## 5. `summarize.py` 的最小改动（唯一一处评测侧改动）

**为什么需要**：本阶段的 6 组都只改 mesh 提取参数，**共用同一个 coarse 模型 `coarse_base_seed0`**，
所以没有自己的 `render_*.json` 与 `train_stats.json`。原版 `summarize.py` 会把这些格子全填 NA，
并在 `[WARN]` 里报"缺渲染指标"，容易被误读成"这些组渲染变差了"。

**改法**：读一个旁路文件 `outputs/metrics/provenance_<run>.json`
（内容形如 `{"source_coarse": "coarse_base_seed0", ...}`），
① 新增末列 `source_coarse` 标注来源；
② 若该 run 自己没有 render / train 数据，则**从 source_coarse 那一行直接继承**
（高斯完全相同，渲染指标逐字相同，不是估计值），并在 `render_json` / `train_stats_json` 列里
注明"(继承自 coarse_base_seed0)"。
对既有 run（没有 provenance 文件的）行为完全不变。

完整 diff：`notes/diffs/s3f_summarize.diff`。核心片段：

```diff
     ("train_stats_json", lambda r: r.get("train_src", NA)),
+    ("source_coarse", lambda r: r.get("source_coarse", NA)),
 ]
```
```diff
+    for path in sorted(glob.glob(os.path.join(args.metrics_dir, "provenance_*.json"))):
+        name = normalize(os.path.basename(path)[len("provenance_"):-len(".json")], norm)
+        ...
+        r["source_coarse"] = src
+        parent = rows.get(normalize(src, norm))
+        if not r["render"] and parent.get("render"):
+            r["render"] = parent["render"]
+            r["render_src"] = parent.get("render_src", NA) + " (继承自 " + src + ")"
+        if not r["train"] and parent.get("train"):
+            r["train"] = parent["train"]
+            r["train_src"] = parent.get("train_src", NA) + " (继承自 " + src + ")"
```

`eval_geometry.py` / `eval_render.py` / `render_mesh_views.py` **一行未动**。

---

## 6. 新增文件清单（未删除、未覆盖任何既有文件）

| 路径 | 用途 |
|---|---|
| `scripts/run_s3f_depth_report.sh` | 只算 auto 深度不提取 |
| `scripts/run_s3f_extract_one.sh` | 单组提取（只改 Poisson 参数）|
| `scripts/run_s3f_sweep_chain.sh` | 三波扫参编排 + 每波后评测/汇总 |
| `scripts/eval/run_eval_extract_only.sh` | 提取端消融组的评测（几何 + mesh 可视化 + provenance）|
| `scripts/check_auto_poisson_depth_cpu.py` | 不依赖 SuGaR/GPU 的独立 CPU 复算 |
| `scripts/make_s3f_table.py` | 生成并排对照表 |
| `notes/diffs/s3f_poisson_sugar.diff`、`notes/diffs/s3f_summarize.diff` | 改动 diff |
| `notes/RUNLOG_s3f_poisson.md`、`notes/SELF_CHECK_阶段3f_poisson.md` | 执行日志 / 自检 |
| `outputs/runs/mesh_base_seed0_<tag>/` | 各组 mesh + `extract_stats.json` |
| `outputs/metrics/geometry_base_seed0_<tag>.json`、`meshvis_…`、`provenance_…` | 各组指标 |
| `outputs/metrics/auto_poisson_depth_cpu_base_seed0.json` | CPU 复算结果 |

被修改的既有文件（3 个）：
`repo/SuGaR_dev/sugar_extractors/coarse_mesh.py`、`repo/SuGaR_dev/extract_mesh.py`（并同步到 `repo/SuGaR/`）、
`scripts/eval/summarize.py`。

---

## 7. 对预注册假设/候选的逐条裁决

依据 `notes/SURVEY2_sugar_direct_followups.md` §4 的候选清单。

### 候选①：Frosting 的自动 Poisson 深度 `compute_optimal_poisson_depth`

| 子命题 | 裁决 | 依据数字 |
|---|---|---|
| "该函数可以原样粘进 SuGaR，依赖全部现成，预计 15 分钟落地" | **支持** | 实际落地约 6 分钟；`knn_points` 已在第 6 行 import，`get_cameras_spatial_extent(return_average_xyz=True)` / `points` / `strengths` 签名全部对得上；函数体逐行未改 |
| "`knn_points` 返回平方距离，照抄勿修正" | **支持（且已照抄）** | 照抄后 raw depth = 10.873 ⇒ D=10。**反事实验算**：若"修正"成开方，归一化距离会从 5.3316e-6 变成 6.4401e-4，raw = −log2(100×6.4401e-4) = **3.957 ⇒ D=3**，Poisson 八叉树只有 2³=8 格一维分辨率，mesh 会退化成一个没有任何细节的大块。可见这条提醒是对的、且影响极大 |
| "先跑一次只打印自动算出的 D，若结果就是 10 则该场景无收益" | **不支持（指改动本身无收益）** | **auto D = 10 = 原版硬编码值**。raw = 10.8732，被 `max_poisson_depth=10` 截断。该改动在 Truck 上**零几何收益**：pdauto vs base_seed0 的全部指标差异都落在重复运行噪声底内（≤ ±2%） |
| "作者报告的收益：避免椭球疙瘩与 D 过大导致的孔洞" | **不确定（本场景无法检验）** | 自动机制在 Truck 上根本没有降低 D，因此无法检验"降低 D 能否消除疙瘩/孔洞"。我们改用手动 pd9 / pd8 单独检验，见 §4 |
| "扫 ratio 50/100/200 → D 分别 +1/基准/−1" | **部分不支持** | 本场景 ratio=50 与 100 算出的 D **都是 10**（都被上限截断）；要真降到 9 需要 ratio ≥ ≈189，降到 8 需 ≥ ≈378。见 §3 的敏感性表 |

> **结论：移植是忠实且正确的，但对 Truck 这种高斯足够密的真实场景，Frosting 的自动深度不会改变任何东西。**
> 它的价值在于对**高斯稀疏的场景**自动调小 D（论文里针对合成场景/低细节场景）；在本次考核的场景上属于"正确但无效"的改动。


### 候选③：`vertices_density_quantile` 扫 0 / 0.05 / 0.1

| 子命题 | 裁决 | 依据数字 |
|---|---|---|
| "直接影响孔洞 vs 碎片的权衡" | **支持，且效应远大于预期** | 见 §4 的并排表：q=0.1→0 使最大连通分量面数占比 41.7%→92.8%，边界边 −70%，mesh 光栅化命中率 92.0%→99.99%。全部远超 ±2% 的噪声底 |
| "与候选① 必须联合扫，不要各自单独调" | **不适用（在本场景）** | 因为 auto D = 10 = 默认值，两者在 Truck 上不存在交互；`pdauto_q0` 与 `q0` 参数等价 |
| Frosting 注释 "0.1 for most real scenes, 0. works well for most synthetic scenes" | **本场景不支持该经验法则** | Truck 是真实场景，但 q=0 在拓扑指标上压倒性更好；只有 G1 中位距离略差 |

### 附加检验（非预注册）：手动 D 扫描 pd9 / pd8

调研笔记里 Frosting 的定性主张是"D 过大 → 高斯的椭球形状变成表面疙瘩 + 出现孔洞"。
自动机制在本场景没降 D，于是手动降 D 单独检验：

| 子命题 | 裁决 | 依据数字 |
|---|---|---|
| "D 越小，表面疙瘩越少（更平滑）" | **支持** | 二面角 abs mean：D=10 → 9 → 8 为 33.60 → 31.36（−6.7%）→ 26.78（−20.3%），单调，远超 ±0.2% 噪声底 |
| "D 越小，孔洞越少" | **支持** | 边界边：38596 → 21925（−43.2%）→ 12900（−66.6%）；mesh 光栅化命中率 92.03% → 93.53% → 95.46% |
| "所以更小的 D 更好" | **不支持（在 Truck 上）** | 精度代价过大：G1 median rel 0.0687% → 0.0748%（+8.9%）→ 0.0978%（**+42.3%**）；G2 <0.5% 从 94.13 掉到 91.54。且最大连通分量占比只从 41.7% 提到 48.6%，**主体破碎问题没解决** |

> 综合：**主体破碎的主因是 `vertices_density_quantile` 的密度清洗在闭合曲面上打洞，不是八叉树太深。**
> 如果要在这份考核里改一个"提取端"的参数，应该改 `--vertices_density_quantile 0`（收益大、代价小），
> 而不是改 `--poisson_depth`（在本场景要么无效，要么用精度换平滑）。

### 落地建议（供主会话决策，不代表我已执行）

1. 保留本次移植（`--poisson_depth auto` 可用、忠实、无副作用），但在报告里**如实写明它在 Truck 上算出 D=10、零收益**，
   并把它定位成"移植可行性 + 负结果"，不要包装成性能提升。
2. 若要在报告里给"提取端最优配置"，用 `--vertices_density_quantile 0`，数字见 §4.5(d)。
3. 若日后要做严谨的提取端对比，建议给 `extract_mesh.py` 也加 `--seed` 固定
   `coarse_mesh.py:471` 的 `torch.randperm`（**本次没有加，因为那会改变与阶段 2/3.4 既有结果的可比性**）。
