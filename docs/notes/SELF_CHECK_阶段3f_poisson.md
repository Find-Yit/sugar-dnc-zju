# SELF_CHECK 阶段 3f — 移植 Frosting 的自动 Poisson 深度 + mesh 提取端扫参

执行者 9（claude-opus-5）。日期 2026-09-13。
依据：`notes/SURVEY2_sugar_direct_followups.md` §2.1 候选① / §4 候选③；`plans/SuGaR复现与改动_6小时考核.txt` §2b。
改动范围：**只在 mesh 提取端**，不重训 coarse，不改评测逻辑（`summarize.py` 有一处最小改动，diff 见 §5）。

---

## 0. 一句话结论（先说最重要的）

> **在 Truck 场景上，Frosting 的自动 Poisson 深度算出来就是 D = 10，与 SuGaR 原版硬编码的 10 完全相同 —— 该改动在本场景零收益。**
> 公式的原始值是 `raw = 10.873`，被 `max_poisson_depth=10` 截断到 10。
> 因此本阶段的实验重点转向：(a) 手动把 D 调小（pd9 / pd8）验证"更小的 D 是否真的更好"；
> (b) `vertices_density_quantile` 扫描（q0 / q005）。结论见 §4。

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
| C4 | 默认值下行为与原版一致 | ① diff 显示默认路径上没有任何行为改动（只多了打印和 json 落盘）；② 实测 `pdauto`（auto→10, q=0.1）产出的 mesh 与阶段 3.4 原版提取的 `mesh_base_seed0` **md5 完全相同** | 见 §3 | `md5sum outputs/runs/mesh_base_seed0/*.ply outputs/runs/mesh_base_seed0_pdauto/*.ply` |
| C5 | auto 深度实际算出的 D 与中间量落盘 | `extract_stats.json` 已写；D=10 | PASS | `cat outputs/runs/_depth_report_base_seed0/extract_stats.json` |
| C6 | 独立复算 auto D（不依赖 SuGaR/pytorch3d） | CPU scipy cKDTree：raw=10.8831 → **D=10**（GPU：raw=10.8732 → D=10） | PASS | `python scripts/check_auto_poisson_depth_cpu.py --coarse_pt ... --gs_checkpoint outputs/baseline/gs_truck` |
| C7 | 每组提取的几何指标写进 `outputs/metrics/summary.csv` | 见 §4 | 见 §4 | `python scripts/eval/summarize.py` |
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
