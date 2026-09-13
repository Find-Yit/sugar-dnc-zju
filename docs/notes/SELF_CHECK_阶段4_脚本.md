# SELF_CHECK —— 阶段 4 评测脚本（执行者 3）

计划：`plans/SuGaR复现与改动_6小时考核.txt` §2b（指标定义）+ 阶段 4 的 S4.1–S4.4。
约束遵守情况：**未修改 `repo/SuGaR` 与 `repo/SuGaR_dev` 下任何文件**、**未 pip install 任何包**、
**未删除任何文件**、**未 git commit**。所有新增文件都在 `scripts/eval/`、`outputs/metrics/`、
`outputs/vis/`、`notes/`。

> baseline（3DGS 7k + coarse SuGaR 15k + mesh）上的全部条目已实跑通过，数值见下。
> λ=0.05 / λ=0.2 两组待执行者 2 的产物就绪后，用 §4b 的同一条驱动脚本命令评测即可。

---

## 0. 就绪判据

| 条目 | 命令 | 实际 | 结果 |
|---|---|---|---|
| 环境 | `python -c "import torch, pytorch3d, open3d, diff_gaussian_rasterization"` | torch 2.4.1+cu121 / cuda 12.1 / avail True；pytorch3d 0.7.8；open3d 0.19.0；dgr、simple_knn OK | PASS |
| 3DGS 产物 | `ls outputs/baseline/gs_truck/point_cloud/iteration_7000/point_cloud.ply` | 460,021,692 B | PASS |
| coarse baseline | `ls outputs/runs/coarse_baseline/*/15000.pt` | 303,793,046 B（06:05:16 产出） | PASS |
| mesh baseline | `ls outputs/runs/mesh_baseline/*.ply` | `sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply`，15,304,017 B（06:11 产出） | PASS |
| LPIPS 权重 | 见 RUNLOG_eval.md §0 | vgg.pth + vgg16-397923af.pth 均下载成功 | PASS |

---

## 1. S4.1 `scripts/eval/eval_render.py`

**命令（vanilla 3DGS 7k 参考行）**
```bash
source $PROJ_ROOT/env.sh
python scripts/eval/eval_render.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --iteration 7000 --run_name vanilla3dgs7k --gpu 1
```
**命令（coarse SuGaR baseline）**
```bash
python scripts/eval/eval_render.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck --iteration 7000 \
  --coarse_pt $PROJ_ROOT/outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  --run_name coarse_baseline --gpu 1
```

| 验收条目 | 期望 | 实际 | 结果 |
|---|---|---|---|
| 是否跑通 | 无异常退出 | 两次均 exit 0，各 ~25 s | PASS |
| 测试视角数 | 32（llffhold=8） | 32（训练 219） | PASS |
| M1 PSNR baseline | 有数值 | **coarse_baseline 24.6530 dB**；vanilla3dgs7k 23.9003 dB | PASS |
| M2 SSIM baseline | 有数值 | **coarse_baseline 0.85644**；vanilla3dgs7k 0.84994 | PASS |
| M3 LPIPS-VGG baseline | 有数值或注明失败 | **coarse_baseline 0.20280**；vanilla3dgs7k 0.20116 | PASS |
| 逐视角 CSV | 32 行 | `outputs/metrics/render_*.csv`，32 行 | PASS |
| 汇总 JSON | 均值 + 配置 | `outputs/metrics/render_*.json` | PASS |
| 4 个固定视角 PNG | render + gt + 误差热图 | `outputs/vis/<run>/` 共 12 个 PNG（4×3） | PASS |
| 固定视角是否写死 | 是且注明 | `_common.FIXED_TEST_VIEW_INDICES = [0, 8, 16, 24]`，对应图片 `000001 / 000065 / 000129 / 000193` | PASS |
| 不用 matplotlib | 只用 PIL+numpy | 伪彩色用离线导出的 `turbo_lut.npy`（256×3 uint8），ML venv 未装 matplotlib | PASS |

**独立交叉验证（3 条，都通过）**

1. **对 3DGS 官方训练日志**：`logs/s21_3dgs.log` 的 `[ITER 7000] Evaluating test: PSNR 23.908734679222107`
   vs 本脚本 23.9003 dB，差 **0.0084 dB**。根因见 §5 的 N1（cameras.json 存的是 COLMAP 内参
   分辨率 1957×1091，与磁盘图 979×546 的宽高比差 0.04%，FoV 往返换算引入的微小差异），非 bug。
2. **两条渲染路径互验**：`--vanilla_render_mode gs`（3DGS 官方 `render`）与 `sugar`
   （`SuGaR.render_image_gaussian_rasterizer`）对同一批高斯，**32 个视角逐行完全相同**
   （PSNR 23.9003 / SSIM 0.84994 / LPIPS 0.20116）。说明评 coarse SuGaR 用的渲染路径无偏差。
   证据：`outputs/metrics/render_vanilla3dgs7k.csv` vs `outputs/metrics/crosscheck/render_vanilla3dgs7k_sugarpath.csv`。
3. **脱离 torch 的独立复算**：`scripts/eval/check_png_psnr.py` 只用 PIL+numpy 从保存的 8bit PNG
   重算 PSNR，与 CSV 比对。8 个视角最大差 **0.0024 dB**（纯量化误差），阈值 0.05 dB → PASS。
   ```bash
   python scripts/eval/check_png_psnr.py --runs vanilla3dgs7k,coarse_baseline
   ```

---

## 2. S4.2 `scripts/eval/eval_geometry.py`

**helper 单元测试（合成网格，全部通过）**
- 立方体：18 条唯一边全部 count==2；二面角取值集合精确等于 `{0°, 90°}`；边界边 0、非流形边 0。
- 球面（3480 面）：二面角 mean 3.318° / P90 6.004° / max 6.008°；边界边 0、非流形边 0。
- `RaycastingScene.compute_distance` 对 4 个已知点返回 `[0.9973, 1.0, 0.5, 0.0]`（期望 1/1/0.5/0）。
- 合成「大球 1520 面 + 小碎片 48 面」：连通分量 **2**、最大分量占比 **96.94%**（=1520/1568）、
  碎片(<100 面) **1**。→ G4 的分量/碎片统计正确。

**相机尺度不变性**：`load_gs_cameras(load_gt_images=False/True)` 两条路径算出的
`cameras_extent = 5.847914695740` 与 `avg_camera_center = [0.08619998, -0.00717573, 0.15169582]`
**逐位相同**（相机中心只依赖 rotation/position）。

**COLMAP 参考点过滤**：136,029 点 → track≥3: 123,843 → 且在前景 bbox 内: **88,066**。
前景 bbox 用 `center_bbox=True, fg_bbox_factor=1`，与 `sugar_extractors/coarse_mesh.py` 一致。

**命令（真实 baseline mesh）**
```bash
python scripts/eval/eval_geometry.py --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --mesh_path $PROJ_ROOT/outputs/runs/mesh_baseline/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply \
  --run_name coarse_baseline
```

| 验收条目 | 期望 | 实际（baseline） | 结果 |
|---|---|---|---|
| 是否跑通 | exit 0 | exit 0，9.6 s | PASS |
| mesh 可读、无 NaN | 顶点>0、面>0 | 205,647 顶点 / 370,438 面 / NaN 顶点 0 | PASS |
| G1 参考点过滤 | track≥3 且在前景 bbox | 136,029 → 123,843 → **88,066** | PASS |
| G1 mean/median/P90（绝对） | 有数值 | 0.010458 / **0.004003** / 0.019381 | PASS |
| G1 mean/median/P90（相对 extent） | 有数值 | 0.1788% / **0.0684%** / 0.3314% | PASS |
| G2 τ=0.5% / τ=1% | 有数值 | **94.169% / 97.688%** | PASS |
| G3 mean（全部顶点 / 前景顶点） | 有数值 | 5.772637 / 0.106971 | PASS |
| G4 连通分量 / 最大占比 | 有数值 | **2237** / **42.494%** | PASS |
| G4 碎片(<100 面) | 有数值 | **2147**（合计 28,657 面） | PASS |
| G4 非流形边 / 边界边 | 有数值 | **0** / 38,384 | PASS |
| G5 二面角 raw mean/P90 | 有数值 | 46.0908° / 113.2074° | PASS |
| G5 二面角 abs mean/P90 | 有数值 | 33.6579° / 75.1493°（raw>90° 占 16.242%） | PASS |
| 输出 JSON | `geometry_<run>.json` | `outputs/metrics/geometry_coarse_baseline.json` | PASS |

**两条可信度检查（都通过）**

1. **G4 分量数不是顶点未焊接造成的**：205,647 个顶点对应 205,647 个唯一位置（**0 个重复顶点**）；
   `merge_close_vertices(1e-9)` 后连通分量仍是 2237、最大占比仍是 42.494%。→ 2237 个是真实碎片。
2. **G5 raw 偏大不是绕序 bug**：raw 二面角的 18 个 10° 分箱从 `0-10°: 122,951 (22.92%)`
   **单调递减**到 `170-180°: 9,557 (1.78%)`，没有 180° 假峰。为防误读，JSON 里同时给出
   与绕序无关的 `abs = min(raw, 180-raw)` 口径和完整直方图。

---

## 3. S4.3 `scripts/eval/render_mesh_views.py`

```bash
python scripts/eval/render_mesh_views.py --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck --mesh_path <上面那个 .ply> \
  --run_name coarse_baseline --gpu 1
```

| 验收条目 | 期望 | 实际 | 结果 |
|---|---|---|---|
| 是否跑通 | exit 0 | exit 0，16.4 s | PASS |
| 分辨率与 eval_render 一致 | 979×546 | 979×546 | PASS |
| 视角与 eval_render 一致 | [0,8,16,24] | [0,8,16,24]，图片名 000001/000065/000129/000193 | PASS |
| mesh 像素命中率 | 合理（不是全空/全满） | 88.95 / 94.90 / 87.75 / 96.33 % | PASS |
| 法向图 PNG | 8 张（法向+深度各 4） | `outputs/vis/coarse_baseline/mesh_{normal,depth}_view*.png` | PASS |
| 人眼看过 | 是 | 法向图卡车轮廓清晰、地面法向一致（品红）、树叶噪声大；深度图近蓝远红、天空白色未命中 | PASS |
| 深度色标可复用 | 记录区间 | `meshvis_coarse_baseline.json` 存每视角 [p1,p99]，`--depth_range_from` 可让后续 run 复用 | PASS |

---

## 4. S4.4 `scripts/eval/summarize.py`

```bash
python scripts/eval/summarize.py      # -> outputs/metrics/summary.csv
```

| 验收条目 | 期望 | 实际 | 结果 |
|---|---|---|---|
| 是否跑通 | exit 0 | exit 0，2 行 × 44 列 | PASS |
| 表头带单位与方向 | `_up` / `_down` | 如 `PSNR_dB_up`、`G1_median_rel_pct_down`、`G4_n_fragments_lt100faces_down` | PASS |
| 缺失填 NA | 是 | 是，并在 stderr 列出缺哪一项 | PASS |
| 读 train_stats.json | 读到就填 | 两套键名都能读（见 §5 N2）；baseline 目前没有该文件（见 §5 N3） | 部分 |
| 溯源列 | 指回源 JSON | `render_json` / `geometry_json` / `train_stats_json` 三列 | PASS |

**当前 summary.csv 关键行**

| run | PSNR_dB_up | SSIM_up | LPIPS_VGG_down | G1_median_rel_pct_down | G4_n_components_down | G4_n_fragments_lt100faces_down |
|---|---|---|---|---|---|---|
| vanilla3dgs7k | 23.9003 | 0.84994 | 0.20116 | NA | NA | NA |
| baseline | 24.6530 | 0.85644 | 0.20280 | 0.0684 | 2237 | 2147 |

---

## 4b. 驱动脚本与可复现性

```bash
bash scripts/eval/run_eval_for_run.sh coarse_baseline <coarse.pt> <mesh.ply> 1
```
端到端 **1 分 19 秒**，exit 0。把重跑前后的 JSON 逐字段对比：
render 的 17 个数值字段、geometry 的 52 个数值字段 **全部逐位相同（0 处不同）**
→ 评测流程确定性，重复运行不会改变报告里的数字。

**编译 / 静态检查**：`python -m py_compile scripts/eval/*.py` 全过；
`bash -n scripts/eval/run_eval_for_run.sh` 通过；6 个模块 import 无 SyntaxWarning。
ML venv 里没有 ruff / flake8 / pyflakes / pylint（按约束**不安装**），故只做到这一层。

---

## 5. 已发现并处理的问题（非 bug 的写 N，是 bug 的写 B）

- **N1 PSNR 与 3DGS 日志差 0.0084 dB**：`cameras.json` 里 width/height 是 COLMAP 内参分辨率
  1957×1091，而 `images/` 里的图是 979×546；SuGaR 用 cameras.json 的 fx/height 做
  `focal2fov` 往返换算，宽高比差 0.04%，导致 FoV 有极小差异。不影响三组对照（三组用同一套相机）。
- **B1（已修）`render_mesh_views.py` 分辨率对不上**：初版用 `load_gt_images=False`，SuGaR 会按
  cameras.json 的 1957×1091 再按 `max_img_size=1920` 缩放成 **1920×1070**，与 `eval_render.py`
  的 **979×546** 不一致，法向图无法与渲染图逐像素并排。已改成 `load_gt_images=True`，重跑得到 979×546。
- **N2 `train_stats.json` 字段名不统一**：执行者 2 写的是 `train_wallclock_min` /
  `mean_time_per_iteration_ms` / `max_memory_allocated_MiB` / `n_gaussians_final`，
  与任务描述里的 `wall_clock_min` / `ms_per_iter` / `peak_mem_gb` / `n_gaussians` 不同。
  已在 `summarize.py` 里加 `TRAIN_ALIASES` 做别名归一 + 单位换算（MiB→GB），两套命名都能读。
- **N3 baseline 没有 `train_stats.json`**：`coarse_baseline` 是用未插桩的原版
  `train_coarse_density.py` 跑的，没有落训练代价。从日志可读出：START `05:52:34` → END `06:05:18`，
  总墙钟 **12 分 44 秒**；日志里 41 条 `computed in X minutes` 求和为 **12.114 min**（训练循环净时间），
  8000 次迭代 → **约 90.9 ms/iter**；峰值显存未插桩，**无法从日志得到**。
  → 这不是我能决定的事：要么由执行者 1 用插桩版重跑 baseline，要么 summary 里该行填 NA 并在报告注明。
- **B2（已修）`summarize.py` 把 0 显示成 NA**：列取值写成 `g(...) or NA`，而 `0` 是 falsy，
  导致「非流形边 0 条」这种**好结果**被显示成 NA。已改成显式 `None` 判断（新增 `gi()` 辅助函数），
  重跑后 `G4_n_non_manifold_edges_down` 正确显示 `0`。这是我自己写的 bug，不涉及他人代码。
- **N5 G5 二面角需要两套口径**：初版只报 raw（面法向夹角，依赖三角形绕序），baseline 实测
  mean 46.09° / P90 113.21°，容易被误读成「网格朝向坏了」。查直方图确认没有 180° 假峰
  （从 0-10° 的 22.92% 单调递减到 170-180° 的 1.78%）后，新增与绕序无关的
  `abs = min(raw, 180-raw)` 口径（33.66° / 75.15°）与完整直方图，两套都落盘，报告可自行选用。
- **N4 `decimation_target=200000` 是对前景和背景 mesh 各自生效**：日志显示
  Foreground mesh 3,271,742 面 / Background mesh 4,500,788 面分别 decimate 到 20 万，
  最终 .ply 是两者相加，所以总面数约 40 万而不是 20 万。报告里不要写成「20 万面」。

---

*最后更新：2026-09-13 06:18。baseline 全部条目已 PASS；λ=0.05 / λ=0.2 两组待其产物就绪后用同一条驱动脚本命令评测。*
