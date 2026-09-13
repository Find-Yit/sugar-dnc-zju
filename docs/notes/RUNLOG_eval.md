# RUNLOG_eval.md —— 阶段 4 评测脚本执行日志（执行者 3）

对应计划 `plans/SuGaR复现与改动_6小时考核.txt` 的 §2b 指标定义与阶段 4 的 S4.1–S4.4。
所有脚本只新增在 `scripts/eval/`，**不修改 `repo/SuGaR` 与 `repo/SuGaR_dev` 下任何文件**（只 import）。
所有命令都先 `source $PROJ_ROOT/env.sh`。

---

## 0. 环境就绪判据（05:53 检查通过）

```bash
source /scratch/e1351071/zju_test/env.sh
python -c "import torch, pytorch3d, open3d, diff_gaussian_rasterization"
```
输出：`torch 2.4.1+cu121 / cuda 12.1 / avail True`，`pytorch3d 0.7.8`，`open3d 0.19.0`，
`diff_gaussian_rasterization` 与 `simple_knn` 均可 import。

LPIPS 权重预下载（05:54，成功，落到 `$TORCH_HOME/hub/checkpoints/`）：
- `https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/.../v0.1/vgg.pth`
- `https://download.pytorch.org/models/vgg16-397923af.pth`（528 MB，约 2 s）

伪彩色 LUT（避免往 ML venv 装 matplotlib，遵守 CLAUDE.md §4.1 两套 venv 隔离）：
```bash
tools/doc_env/bin/python -c "import numpy as np; from matplotlib import colormaps; \
np.save('scripts/eval/turbo_lut.npy', (colormaps['turbo'](np.linspace(0,1,256))[:,:3]*255).round().astype('uint8'))"
```
→ `scripts/eval/turbo_lut.npy`（matplotlib 3.10.9 的 turbo，256x3 uint8）。

---

## 1. 脚本清单

| 脚本 | 对应 | 作用 |
|---|---|---|
| `scripts/eval/_common.py` | — | 路径注入、带 track 长度的 COLMAP 读取、相机空间尺度、turbo 伪彩色、固定视角常量 |
| `scripts/eval/eval_render.py` | S4.1 | 32 个测试视角的 PSNR / SSIM / LPIPS-VGG + 4 个固定视角的渲染图与误差热图 |
| `scripts/eval/eval_geometry.py` | S4.2 | G1–G5 几何指标 |
| `scripts/eval/render_mesh_views.py` | S4.3 | pytorch3d 渲染 mesh 法向图与深度图 |
| `scripts/eval/summarize.py` | S4.4 | 汇总 `outputs/metrics/summary.csv` |
| `scripts/eval/run_eval_for_run.sh` | — | 对一组 run 依次跑 S4.1–S4.4 的驱动脚本 |

固定可视化视角：**测试集内部顺序号 0 / 8 / 16 / 24**（共 32 张测试图），
对应全量 251 张里的第 0 / 64 / 128 / 192 张，即图片 `000001 / 000065 / 000129 / 000193`。
写死在 `_common.FIXED_TEST_VIEW_INDICES`。误差热图固定色标上限 `ERROR_HEATMAP_VMAX = 0.25`（三组共用，保证可比）。

---

## 2. 时间线与命令

### 05:55 S4.1 在 vanilla 3DGS 7k 上跑通（3DGS 官方渲染器）

```bash
python scripts/eval/eval_render.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --iteration 7000 --run_name vanilla3dgs7k --gpu 1
```
耗时 24.6 s（含 LPIPS）。结果：
- 训练视角 219 / 测试视角 32 ✅（llffhold=8 划分与计划一致）
- **PSNR 23.9003 dB / SSIM 0.84994 / LPIPS-VGG 0.20116**，高斯数 1,854,920
- 产物：`outputs/metrics/render_vanilla3dgs7k.{csv,json}`、`outputs/vis/vanilla3dgs7k/*.png`

**交叉验证**：3DGS 自己的训练日志 `logs/s21_3dgs.log` 里
`[ITER 7000] Evaluating test: L1 0.03830549860140309 PSNR 23.908734679222107`，
我们算出 23.9003，差 **0.0084 dB**。原因已定位（非 bug，见 `SELF_CHECK_阶段4_脚本.md` §5 的 N1）。

### 05:56 S4.1 的第二条渲染路径交叉验证（SuGaR 包装器）

```bash
python scripts/eval/eval_render.py ... --run_name vanilla3dgs7k_sugarpath \
  --gpu 1 --vanilla_render_mode sugar --no_vis
```
用 `SuGaR.render_image_gaussian_rasterizer` 渲染同一批高斯，得到
**PSNR 23.9003 / SSIM 0.84994 / LPIPS 0.20116**，与 3DGS 官方渲染器**逐视角完全一致**
（`render_vanilla3dgs7k.csv` 与 `crosscheck/render_vanilla3dgs7k_sugarpath.csv` 的
psnr/ssim 列逐行相同）。说明评 coarse SuGaR 时用的 SuGaR 渲染路径没有引入偏差。
该交叉验证产物已挪到 `outputs/metrics/crosscheck/`，避免污染 `summary.csv`。

### 05:58 几何工具函数单元测试（合成网格）

立方体：18 条唯一边全部被恰好 2 个面共享，二面角取值集合 `{0°, 90°}`（精确）；边界边 0、非流形边 0。
球面（resolution=30，3480 面）：二面角 mean 3.318° / P90 6.004° / max 6.008°，边界边 0、非流形边 0。
`o3d RaycastingScene.compute_distance` 对球心 / 球外 / 球内点返回 `[0.9973, 1.0, 0.5, 0.0]`（预期 1/1/0.5/0）。
→ `edge_face_table` / `dihedral_angles_deg` / 距离计算均正确。

### 05:59 S4.2 在合成网格上端到端跑通

```bash
python scripts/eval/eval_geometry.py --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --mesh_path <scratch>/test_mesh.ply --run_name _synthetic_selftest --out_dir <scratch>
```
耗时 6.7 s。关键数字：
- `cameras_extent = 5.847914695740`，`avg_camera_center = [0.08619998, -0.00717573, 0.15169582]`
- COLMAP 点 136,029 → track≥3: 123,843 → 且在前景 bbox 内: **88,066**（G1 的参考点集）
- 合成网格（大球 1520 面 + 小碎片 48 面）：连通分量 **2**、最大分量占比 **96.94%**（=1520/1568 ✅）、
  碎片(<100 面) **1** ✅、非流形边 0、边界边 0

### 06:01 S4.3 在合成网格上跑通 + 分辨率对齐修复

首次运行渲染出 **1920x1070**，与 `eval_render.py` 的 **979x546** 不一致。根因见 `SELF_CHECK_阶段4_脚本.md` §5 的 B1；
已把 `render_mesh_views.py` 改成 `load_gt_images=True`，重跑得到 979x546 ✅，
法向图为球面上的平滑渐变、背景纯白，深度区间每视角 [p1, p99] 正常。

### 06:02 相机尺度不变性验证

`load_gs_cameras(load_gt_images=False/True)` 两条路径算出的
`cameras_extent = 5.847914695740` 与 `avg_camera_center` **逐位相同**
（相机中心只依赖 cameras.json 的 rotation/position，与图像分辨率无关）。
→ `eval_geometry.py` 用 `load_gt_images=False`（省 25 s、省显存）不影响 G1/G2 的相对值。

### 06:03 S4.4 summarize 跑通

```bash
python scripts/eval/summarize.py
```
→ `outputs/metrics/summary.csv`，缺失项正确填 `NA` 并在 stderr 列出。

<!-- 后续阶段（baseline 真实产物、三组对照）继续往下追加 -->

---

## 3. baseline 真实产物上的完整评测（06:05–06:16）

### 06:05 S4.1 coarse SuGaR baseline

```bash
python scripts/eval/eval_render.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck --iteration 7000 \
  --coarse_pt $PROJ_ROOT/outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  --run_name coarse_baseline --gpu 1
```
耗时 24.1 s。**PSNR 24.6530 dB / SSIM 0.85644 / LPIPS-VGG 0.20280**，高斯数 429,075。
（对照 vanilla 3DGS 7k：23.9003 / 0.84994 / 0.20116，高斯数 1,854,920。）

> 注：coarse .pt 里是 429,075 个高斯；`extract_mesh.py` 提网格前又按 opacity<0.5 剪了一次，
> 日志显示剩 282,478 个。我们按训练结束的状态评渲染（不再剪枝），已写进 render JSON 的
> `low_opacity_pruning_applied: false`。

### 06:12 S4.2 geometry（baseline mesh）

```bash
python scripts/eval/eval_geometry.py --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --mesh_path $PROJ_ROOT/outputs/runs/mesh_baseline/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply \
  --run_name coarse_baseline
```
耗时 9.6 s。mesh 15,304,017 B，205,647 顶点 / 370,438 面 / 0 个 NaN 顶点。

| 指标 | 值 |
|---|---|
| cameras_extent | 5.847915 |
| G1 参考点数 | 88,066（136,029 → track≥3: 123,843 → 前景 bbox 内: 88,066） |
| G1 median | 0.004003（相对 **0.0684%**） |
| G1 mean | 0.010458（相对 0.1788%） |
| G1 P90 | 0.019381（相对 0.3314%） |
| G2 <0.5% / <1% | **94.169% / 97.688%** |
| G3 mean（全部顶点 / 前景顶点） | 5.772637 / 0.106971 |
| G4 连通分量 | **2237**；最大分量占比 **42.494%** |
| G4 碎片(<100 面) | **2147**（合计 28,657 面） |
| G4 非流形边 / 边界边 | **0** / 38,384 |
| G5 二面角 raw mean / P90 | 46.0908° / 113.2074° |
| G5 二面角 abs mean / P90 | 33.6579° / 75.1493°（raw>90° 占比 16.242%） |

**G5 的口径检查（重要）**：raw 的 18 个 10° 分箱计数从 `0-10°: 122,951 (22.92%)` 单调递减到
`170-180°: 9,557 (1.78%)`，**没有 180° 假峰**，说明网格绕序基本一致，raw 偏大是网格本身粗糙，
不是朝向 bug。两套口径都写进了 `geometry_*.json`（含直方图），报告可自行选用并注明。

**G4 连通分量的可信度检查**：顶点去重后仍是 205,647 个唯一位置（**0 个重复顶点**），
`merge_close_vertices(1e-9)` 后连通分量数仍为 2237、最大占比仍为 42.494%。
→ 2237 个分量是真实的漂浮碎片，不是顶点未焊接造成的假分裂。

### 06:14 S4.3 mesh 可视化（baseline mesh）

```bash
python scripts/eval/render_mesh_views.py --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck --mesh_path <上面那个 .ply> \
  --run_name coarse_baseline --gpu 1
```
耗时 16.4 s。979x546，4 个视角的 mesh 像素命中率 88.95 / 94.90 / 87.75 / 96.33 %，
深度区间（每视角 [p1,p99]）分别 [2.017,27.551] / [1.667,26.350] / [1.075,28.170] / [1.166,20.606]。
人眼看过 `mesh_normal_view00_000001.png`（卡车轮廓清晰、地面法向一致的品红、树叶噪声大）
与 `mesh_depth_view00_000001.png`（近蓝远红、天空为白色未命中），均正常。

### 06:15 驱动脚本端到端 + 可复现性检查

```bash
bash scripts/eval/run_eval_for_run.sh coarse_baseline <coarse.pt> <mesh.ply> 1
```
总耗时 **1 分 19 秒**（S4.1 24.9 s + S4.2 9.6 s + S4.3 39.5 s + S4.4）。
把重跑前后的 JSON 逐字段比对：render 的 17 个数值字段、geometry 的 52 个数值字段
**全部逐位相同**（0 处不同）→ 本评测流程是确定性的，前向光栅化没有引入随机抖动。

### 06:16 S4.4 summarize

```bash
python scripts/eval/summarize.py   # -> outputs/metrics/summary.csv
```
当前 2 行（vanilla3dgs7k / baseline），44 列。

### 编译与静态检查

```bash
python -m py_compile scripts/eval/*.py     # ALL OK
bash -n scripts/eval/run_eval_for_run.sh   # OK
```
ML venv 里 **没有** ruff / flake8 / pyflakes / pylint（按约束不安装），所以只做了
`py_compile` + 全部模块 import 冒烟（无 SyntaxWarning）。
