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

