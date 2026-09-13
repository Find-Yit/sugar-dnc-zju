# 执行日志 — 阶段 3f：移植 Frosting 自动 Poisson 深度 + 提取端扫参

执行者 9（claude-opus-5）。所有时间为本机 `date`（Asia）。
对应计划：`plans/SuGaR复现与改动_6小时考核.txt` §2b（评测口径） + `notes/SURVEY2_sugar_direct_followups.md` 候选①/③。

---

## 09:05–09:11 — 阅读 + 移植实现

命令：
```
sed -n '1,60p' repo/SuGaR_dev/sugar_extractors/coarse_mesh.py
wget https://raw.githubusercontent.com/Anttwo/Frosting/main/frosting_extractors/coarse_shell.py
python <patch 脚本>            # 见 notes/diffs/s3f_poisson_sugar.diff
python -m py_compile repo/SuGaR_dev/{sugar_extractors/coarse_mesh.py,extract_mesh.py}
python extract_mesh.py --help
```

关键发现（与调研笔记不同、必须记录）：
- **Frosting 调用 `compute_optimal_poisson_depth` 的位置在「剪掉低不透明度高斯之前」**
  （`coarse_shell.py:234-240`，紧跟 `for name, param in sugar.named_parameters()` 打印之后，
  在 `# Pruning low opacity gaussians` 之前）。因此本次移植也放在同一位置，而不是剪枝之后
  ——函数内部自带 `opacity_threshold=0.5` 掩码，放在剪枝后会让该掩码变成空操作，口径就不是 Frosting 的了。
- 函数体逐行抄录，未做任何修改（含 `knn_points(...).dists` 返回**平方**距离、不开方）。
  唯一新增是可选关键字 `return_details:bool=False`（默认 False ⇒ 签名与行为与原版一致），
  用于把中间量写进 `extract_stats.json`。

耗时：约 6 min。

---

## 09:09:46–09:09:58 — 只算不提取（`--only_report_depth True`），12 s

命令：`bash scripts/run_s3f_depth_report.sh`（日志 `logs/s3f_depth_report.log`）

**结果：auto D = 10，与 SuGaR 原版硬编码的 10 完全相同 ⇒ 该改动在 Truck 上零收益。**
中间量（`outputs/runs/_depth_report_base_seed0/extract_stats.json`）：
- cameras_spatial_extent = 5.847914695739747
- n_gaussians: total 429423 / fg 112209 / opaque 282930 / **used 77554**
- bbox_size = 12.854769802093507
- quantile_dist（归一化后的**平方**最近邻距离 10% 分位）= 5.331572538125329e-06
- raw_depth_before_floor = **10.873151263385152** → floor = 10 → min(10, max_poisson_depth=10) = **10**

即：公式本身给出 10.87，被 `max_poisson_depth=10` 截断。Truck 的高斯足够密，自动深度顶到上限。

独立 CPU 复算（`scripts/check_auto_poisson_depth_cpu.py`，不依赖 SuGaR/pytorch3d，scipy cKDTree）：
raw = 10.8831 → **D = 10**，与 GPU 结果一致（差 0.01，来源是训练相机子集的取法略有不同，
见 SELF_CHECK 的说明）。

---
## 09:11–09:17 — 第一波提取（pdauto + q0 并发）：pdauto 段错误

命令：`TAG=pdauto PD=auto VDQ=0.1 bash scripts/run_s3f_extract_one.sh`（并发另一组 q0）。
当时机上同时有执行者 8 的 `coarse_official_dnc` mesh 提取 ⇒ **3 个 Poisson 重建并发**。

**[DEVIATION-1] `pdauto` 第一次尝试崩溃：`exit=139`（SIGSEGV），284 s。**
崩溃点在 open3d 的**背景 mesh** Poisson 重建（`create_from_point_cloud_poisson`，193 万点、depth=10）的 C++ 内部，
日志最后一行是 PoissonRecon 的 `Found bad data: 60` 警告。系统内存充足（可用 1630 GB），非 OOM；
当时 load average ≈ 27.6 / 24 核。判断为 **3 个 Poisson 并发时 PoissonRecon（多线程 C++）的偶发崩溃**，
不是本次代码改动引起的（改动只把 `depth=` 从常量换成变量，auto 模式算出来也是 10，与原版取值相同）。

处置（未改任何代码）：改为**严格串行**执行剩余各组，单组失败自动重试 1 次。
新编排脚本 `scripts/s3f_seq_runner.sh`，日志 `logs/s3f_seq_runner.log`。
崩溃日志保留为 `logs/s3f_mesh_pdauto_attempt0_exit139_SEGV.log`（未删除）。

## 09:17 — 发现：SuGaR 的 mesh 提取本身是**不确定的**（重要，影响所有对照的解读）

`sugar_extractors/coarse_mesh.py:471`：
```python
idx = torch.randperm(len(img_surface_points), device=sugar.device)[:n_pts_per_frame]
```
每帧对表面交点做**无种子的随机下采样**（`sugar_scene/sugar_model.py:1997` 同样用 `torch.randperm`）。
`extract_mesh.py` 全程没有任何 `torch.manual_seed`。因此**同样的参数跑两次，得到的 mesh 不一样**。

佐证：同一模型、同样 depth=10 的三次 FG Poisson，PoissonRecon 报告的 `Found bad data` 分别是
26（阶段 3.4 的 `mesh_base_seed0`）、27（pdauto）、32（q0，其 FG 输入与 pdauto 完全同参），
说明输入点云确实逐次不同。

**用途**：`pdauto`（D=auto=10, q=0.1）与阶段 3.4 的 `base_seed0`（D=10, q=0.1）参数完全等价，
二者的差就是**提取的重复运行噪声底**，正好用来判断 q0 / pd8 / pd9 的差异是否真实。

## 09:11–09:18 — q0（D=10, quantile=0）：420 s，成功

`outputs/runs/mesh_base_seed0_q0/`；几何评测 09:18 完成。首个结果就很显著：

| | base_seed0（q=0.1） | q0（q=0） |
|---|---|---|
| 连通分量数 ↓ | 2233 | **1699** |
| 最大分量面数占比 ↑ | 41.686% | **92.776%** |
| 碎片(<100 面) ↓ | 2137 | **1668** |
| G2 <1% ↑ | 97.736 | **98.138** |
| G1 median rel ↓ | **0.0687** | 0.0719 |
| G5 abs mean ° ↓ | **33.6019** | 34.6279 |

解释：`vertices_density_quantile=0.1` 会在 Poisson 之后删掉密度最低的 10% 顶点，
这一步在把"虚假外扩的面"删掉的同时，也把本来连通的闭合曲面**打出大量洞**，
于是主体被切成很多块（最大分量只剩 42% 的面）。关掉它，主体重新连成一片（92.8%）。
代价是 G1 中位距离略微变差（+4.7%）、二面角略糟（+3.1%）。

**噪声底的设计**：由于提取不确定（见上），本轮天然带两组重复：
`base_seed0` vs `pdauto`（都是 D=10,q=0.1）、`q0` vs `pdauto_q0`（都是 D=10,q=0）。
两两之差即重复运行噪声，用来判断上面的变化是否真实。

## 09:17:47 起 — 串行执行器（`scripts/s3f_seq_runner.sh`）

队列：`pdauto(重跑) -> pdauto_q0 -> pd8 -> pd9 -> q005`，每组结束立刻跑
`scripts/eval/run_eval_extract_only.sh`（几何 G1–G5 + mesh 可视化 + 写 provenance）再 `summarize.py`。
单组命令模板（只有最后三个参数在变）：

```bash
python extract_mesh.py -s data/tandt/truck -c outputs/baseline/gs_truck/ -i 7000 \
  -m outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 -o outputs/runs/mesh_base_seed0_<tag> \
  --poisson_depth <10|9|8|auto> --vertices_density_quantile <0.1|0.05|0> --cell_size_nn_distance_ratio 100
```

`summarize.py` 的最小改动（新增 `source_coarse` 列 + 从 `provenance_<run>.json` 继承渲染指标/训练代价）
实测生效：`base_seed0_q0` 行的 `render_json` 列显示
`outputs/metrics/render_coarse_base_seed0.json (继承自 coarse_base_seed0)`，
`source_coarse = coarse_base_seed0`；对既有 7 行没有任何影响（执行者 8 在 09:18 跑的 summarize 也正常）。

## 09:18:52–09:23:48 — pdauto 重跑成功（296 s，串行、机器负载降到 9）

`auto_poisson_depth = 10`（与 09:09 的 only_report 完全一致）。
**pdauto 与阶段 3.4 的 base_seed0 参数完全等价（D=10, q=0.1），二者之差 = 重复运行噪声底：**

| 指标 | base_seed0 | pdauto | 相对差 |
|---|---|---|---|
| G1 median rel % | 0.0687 | 0.0689 | **+0.19%** |
| G1 mean rel % | 0.1770 | 0.1770 | **+0.02%** |
| G2 <1% | 97.736 | 97.737 | +0.00% |
| 顶点 / 面 | 205548 / 370128 | 205400 / 369922 | −0.07% / −0.06% |
| 连通分量 | 2233 | 2267 | **+1.52%** |
| 最大分量面占比 % | 41.686 | 42.051 | +0.88% |
| 碎片(<100 面) | 2137 | 2172 | +1.64% |
| 边界边 | 38596 | 38650 | +0.14% |
| G5 abs mean ° | 33.6019 | 33.5632 | −0.12% |

⇒ **噪声底：连续量（G1/G2/G5/顶点面数）≈ ±0.2%；拓扑计数（分量数/碎片数）≈ ±2%。**
后面任何小于这个量级的差异都不能当成真实效应。

对照之下 q0 的变化（分量 −23.9%、最大分量占比 +122.6%、边界边 −70.0%、命中率 +8.6%、
G1 mean −5.3%、G1 median +4.7%）**全部远超噪声底，是真实效应**。

## [DEVIATION] 汇总（全部未自行改计划，均已在此说明）

- **[DEVIATION-1] pdauto 第一次提取 SIGSEGV（exit=139）**。原因：3 个 open3d Poisson 重建并发
  （我的 2 组 + 执行者 8 的 1 组），load ≈ 27/24 核；内存充足（可用 1630 GB），非 OOM。
  与本次代码改动无关（auto 模式算出的 D 就是 10，与原版取值相同）。
  处置：**不改代码**，改为严格串行 + 单组失败自动重试 1 次；重跑 296 s 成功。
  崩溃日志保留 `logs/s3f_mesh_pdauto_attempt0_exit139_SEGV.log`。
- **[DEVIATION-2] 调用位置与调研笔记的推测不同**。笔记说"插在 sugar 实例建好之后"（标【推测】）；
  实际下载 Frosting 源码核对后发现它在 `coarse_shell.py:234-240`，即 **`# Pruning low opacity gaussians`
  之前**。本次按实际源码放置。另外函数新增了可选关键字 `return_details`（默认 False ⇒ 行为不变），
  这是对"逐字抄录"的一处有意、已标注的偏离，目的是把中间量落盘。
- **[DEVIATION-3] 计划未预见：SuGaR 的 mesh 提取是不确定的**（`coarse_mesh.py:471` 无种子 randperm）。
  未改代码（改动提取的随机性会破坏与阶段 2/3.4 既有结果的可比性）。
  处置：安排两对同参重复（base_seed0/pdauto、q0/pdauto_q0）来量化噪声底，见 §4。
- **[DEVIATION-4] 候选① 在本场景零收益**：auto D = 10 = 原版硬编码值。按任务指示把重心转到
  `vertices_density_quantile` 扫描。同时 `pdauto_q0` 因 auto→10 与 `q0` 参数等价，
  故把它当作 q=0 条件的重复对照（噪声底），而不是一个独立条件。
- **[DEVIATION-5] 改了 `scripts/eval/summarize.py`**（任务允许的最小改动），diff 见
  `notes/diffs/s3f_summarize.diff` 与 SELF_CHECK §5。`eval_geometry.py` / `eval_render.py` /
  `render_mesh_views.py` 一行未动。
- **未产出独立 README.md**：本次任务的交付清单（第 5 条）只要求
  `notes/SELF_CHECK_阶段3f_poisson.md`，全部结果表与结论都在其中。

## 09:24:13–09:29:22 — pdauto_q0（D=auto=10, q=0），309 s，成功

第二对同参重复（`q0` vs `pdauto_q0`，都是 D=10,q=0）给出 q=0 条件下的噪声底：

| 指标 | q0 | pdauto_q0 | 相对差 |
|---|---|---|---|
| G1 median rel % | 0.0719 | 0.0712 | **1.0%** |
| G1 mean rel % | 0.1676 | 0.1667 | 0.5% |
| 连通分量 | 1699 | 1728 | 1.7% |
| 最大分量面占比 % | 92.776 | 92.884 | 0.1% |
| 碎片(<100 面) | 1668 | 1702 | **2.0%** |
| 边界边 | 11566 | 11443 | 1.1% |
| G5 abs mean ° | 34.6279 | 34.6408 | 0.04% |

⇒ 合并两对重复，取保守噪声底：**G1 median ±1%、拓扑计数 ±2%、其余 <1%**。

## 09:29:47–09:41:53 — pd8 / pd9 / q005（串行，全部 exit=0）

| tag | 参数 | 墙钟 | 结论一句话 |
|---|---|---|---|
| pd8 | D=8, q=0.1 | 158 s | 碎片 −67%、二面角 −20%（更平滑）、提取快 2.6 倍；但 G1 median **+42.3%**（精度大幅下降），最大分量占比只到 48.6% |
| pd9 | D=9, q=0.1 | 188 s | 碎片 −38%、二面角 −6.7%；G1 median +8.9%；最大分量占比 44.8% |
| q005 | D=10, q=0.05 | 294 s | 各项都恰好在 q=0.1 与 q=0 之间，单调；最大分量占比 45.2% |

全部 6 组 + 基准的并排表见 `notes/SELF_CHECK_阶段3f_poisson.md` §4.3 / §4.5。

## 09:42–09:45 — 收尾

- `python scripts/eval/summarize.py` 最终跑一次：11 行，其中 6 行是本阶段的提取端消融
  （`base_seed0_{pdauto,pdauto_q0,q0,q005,pd9,pd8}`），`source_coarse` 全部为 `coarse_base_seed0`，
  渲染指标继承（PSNR 24.6201 / SSIM 0.85552 / LPIPS 0.20316，与源 run 逐字相同）。
- 8 个 py 文件 `py_compile` 通过；5 个新 shell 脚本 `bash -n` 通过；
  `--poisson_depth` 对 `auto / -1 / 10 / 8` 四种取值解析正常。
- 重新生成 `notes/diffs/s3f_poisson_sugar.diff`、`notes/diffs/s3f_summarize.diff`。

**说明（非我操作）**：09:19:16 主会话（Find-Yit）在 `repo/SuGaR` 上做了一次
`中期快照：DNC 改动、官方 dn_consistency 包装、Frosting 自动 Poisson 深度移植、...` 的 commit
（HEAD = 39c0f56，分支 dnc），把我复制过去的两个文件一并提交了。**我本人没有执行任何 git commit。**
复核：`repo/SuGaR` 与 `repo/SuGaR_dev` 的 `coarse_mesh.py` / `extract_mesh.py` 仍逐字节相同。
